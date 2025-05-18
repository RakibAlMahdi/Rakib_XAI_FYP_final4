import argparse, os, random, math
import torch, torchaudio, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from utils import load_metadata, wav_paths_for_patient, load_physionet2016_metadata
from tqdm import tqdm

try:
    import shap  # noqa: E402
except ImportError:
    shap = None

# ------------------ reproducible seed -----------------
torch.manual_seed(42); random.seed(42); np.random.seed(42)

# ----------------------------- Dataset ----------------------------- #
class CirCorDataset(Dataset):
    """Dataset that exposes **fixed-length segments** (windows) of each WAV.

    For example, with resample_hz=1000 and segment_sec=10 each item is a
    1×10 000-sample tensor. If a recording is shorter than segment length we
    zero-pad; longer recordings are split into ⌊len/seg_len⌋ non-overlapping
    windows.
    """

    def __init__(self, csv_path: str, wav_dir: str, *, resample_hz: int = 1000,
                 segment_sec: int = 10, augment: bool = False):
        super().__init__()
        self.meta = load_metadata(csv_path)
        self.wav_dir = wav_dir
        self.resample_hz = resample_hz
        self.augment = augment
        self.seg_len = segment_sec * resample_hz

        # Build list of (wav_path, label, segment_idx)
        self.items: list[tuple[str,int,int]] = []
        for pid, y, _ in self.meta.values:
            label = int(y)
            for wav_path in wav_paths_for_patient(pid, wav_dir):
                info = torchaudio.info(wav_path)
                est_len = int(info.num_frames / info.sample_rate * self.resample_hz)
                n_segments = max(1, est_len // self.seg_len)
                for sidx in range(n_segments):
                    self.items.append((wav_path,label,sidx))

    # -----------------------------------------------------------------
    def _preprocess(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        # Mono channel: flatten
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        # Resample to target sr
        if orig_sr != self.resample_hz:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=self.resample_hz)
            waveform = resampler(waveform)
        # Band-pass 25-400 Hz (Butterworth biquad)
        waveform = torchaudio.functional.highpass_biquad(waveform, self.resample_hz, 25)
        waveform = torchaudio.functional.lowpass_biquad(waveform, self.resample_hz, 400)
        # Pad / crop
        target_len = self.seg_len
        if waveform.shape[1] < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))
        else:
            waveform = waveform[:, :target_len]
        return waveform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, seg_idx = self.items[idx]
        waveform, sr = torchaudio.load(path)
        x = self._preprocess(waveform, sr).float()  # (1, L)

        # slice to segment
        start = seg_idx * self.seg_len
        end = start + self.seg_len
        if x.shape[1] < self.seg_len:
            # zero-pad short recording
            pad = self.seg_len - x.shape[1]
            x = torch.nn.functional.pad(x,(0,pad))
        else:
            if end > x.shape[1]:
                # last segment pad end
                pad = end - x.shape[1]
                x = torch.nn.functional.pad(x,(0,pad))
            x = x[:, start:end]

        # ---------------- augmentation for training ----------------
        if self.augment:
            # (shift disabled for debugging stability)
            # gaussian noise low level, prob 0.2
            if random.random() < 0.2:
                noise = torch.randn_like(x) * 0.001
                x = x + noise
        return x, torch.tensor(label, dtype=torch.float32)

# ----------------------------- Model ----------------------------- #
class InceptionBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 32, bottleneck: int = 32,
                 kernel_size: int = 40):
        super().__init__()
        self.use_bottleneck = in_ch > 1
        self.bottleneck = (nn.Conv1d(in_ch, bottleneck, 1)
                           if self.use_bottleneck else nn.Identity())

        ks = [kernel_size // (2 ** i) for i in range(3)]
        self.conv_list = nn.ModuleList([
            nn.Conv1d(bottleneck if self.use_bottleneck else in_ch,
                      out_ch, k, padding="same") for k in ks])
        # dilated branch for larger receptive field
        self.dilated_conv = nn.Conv1d(bottleneck if self.use_bottleneck else in_ch,
                                      out_ch, kernel_size=7, dilation=4, padding="same")
        self.pool_conv = nn.Conv1d(in_ch, out_ch, 1, padding="same")
        self.bn = nn.BatchNorm1d(out_ch * 5)  # 5 branches now
        self.relu = nn.ReLU()

        # residual projection if channels differ
        out_channels = out_ch * 5
        if in_ch != out_channels:
            self.res_proj = nn.Conv1d(in_ch, out_channels, 1)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x) if self.use_bottleneck else x
        conv_outs = [conv(z) for conv in self.conv_list]
        conv_outs.append(self.dilated_conv(z))
        pooled = torch.nn.functional.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        conv_outs.append(self.pool_conv(pooled))
        out = torch.cat(conv_outs, dim=1)
        out = self.bn(out)
        # residual add
        res = self.res_proj(x)
        out = self.relu(out + res)
        return out


class AttentionPool1d(nn.Module):
    """Attention pooling with sigmoid weights to allow multi‐peak focus."""
    def __init__(self, in_ch):
        super().__init__()
        self.attn = nn.Conv1d(in_ch, 1, 1)

    def forward(self, x):  # (B,C,T)
        w = torch.sigmoid(self.attn(x))
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
        return torch.sum(x * w, dim=-1)


class SincConv1d(nn.Module):
    """Lightweight band-pass conv layer inspired by SincNet."""
    def __init__(self, out_channels: int = 32, kernel_size: int = 101, sr: int = 1000):
        super().__init__()
        self.out_channels, self.kernel_size, self.sr = out_channels, kernel_size, sr
        low = torch.linspace(0, sr//2, out_channels+1)[:-1] / sr
        high = torch.linspace(0, sr//2, out_channels+1)[1:] / sr
        self.low = nn.Parameter(low)
        self.band = nn.Parameter(high - low)

        n = torch.linspace(-(kernel_size//2), kernel_size//2, steps=kernel_size)
        self.register_buffer('n_', n)
        self.register_buffer('window', 0.54 - 0.46*torch.cos(2*math.pi*n/kernel_size))

    def forward(self, x):  # (B,1,T)
        low = 50 + torch.abs(self.low) * (self.sr/2 - 50)
        high = torch.clamp(low + torch.abs(self.band)*(self.sr/2 - 50), 80, self.sr/2-1)
        filters = []
        for l, h in zip(low, high):
            band = (torch.sinc(2*h*self.n_/self.sr) - torch.sinc(2*l*self.n_/self.sr))
            band *= self.window
            den = 2 * band.sum()
            band = band / (den + 1e-8)
            filters.append(band)
        weight = torch.stack(filters).unsqueeze(1)
        return torch.conv1d(x, weight, padding=self.kernel_size//2)


class InceptionNet1D(nn.Module):
    def __init__(self, n_blocks: int = 10, num_classes: int = 1):
        super().__init__()
        self.conv0 = SincConv1d(out_channels=32, kernel_size=101, sr=1000)
        self.blocks = nn.ModuleList()
        in_ch = 32
        branch_out = 32 * 5  # out_ch * num_branches (5)
        for i in range(n_blocks):
            self.blocks.append(InceptionBlock(in_ch))
            in_ch = branch_out  # 160 channels after each block

        self.apool = AttentionPool1d(in_ch)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, L)
        x = self.conv0(x)
        for block in self.blocks:
            x = block(x)
        x = self.apool(x)
        return self.fc(x)  # logits (B,1)

# ----------------------------- Metrics ----------------------------- #

def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor):
    y_pred_bin = (y_pred > 0.5).cpu().numpy().astype(int)
    y_true_np = y_true.cpu().numpy().astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred_bin, labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-7)
    spec = tn / (tn + fp + 1e-7)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    auc = roc_auc_score(y_true_np, y_pred.cpu().numpy()) if len(np.unique(y_true_np)) > 1 else 0.5
    return sens, spec, acc, auc

# ----------------------------- Training ----------------------------- #

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------------------------------------------
    # Build combined list of (wav_path, label)
    # ------------------------------------------------------------

    combined_items: list[tuple[str,int]] = []

    # -- CirCor
    circor_md = load_metadata(args.csv)
    for pid, lab in zip(circor_md["Patient ID"], circor_md["label"]):
        for wp in wav_paths_for_patient(pid, args.wav_dir):
            combined_items.append((wp, int(lab)))

    # -- PhysioNet 2016
    if args.physio16_csv and args.physio16_wav_dir:
        phys_md = load_physionet2016_metadata(args.physio16_csv)
        for wav, lab in phys_md.values:
            path = os.path.join(args.physio16_wav_dir, wav)
            if os.path.exists(path):
                combined_items.append((path, int(lab)))

    print(f"Total WAVs loaded: {len(combined_items)} (CirCor + Physio16)")

    # ------------------------------------------------------------
    # Dataset class operating on generic item list
    # ------------------------------------------------------------

    class GenericSegDataset(Dataset):
        def __init__(self, items, *, resample_hz: int = 1000, seg_sec: int = 10, augment: bool=False):
            self.items = items
            self.resample_hz = resample_hz
            self.seg_len = seg_sec * resample_hz
            self.augment = augment

        def __len__(self):
            return len(self.items)

        def _preprocess(self, waveform, orig_sr):
            if waveform.shape[0] > 1:
                waveform = waveform.mean(0, keepdim=True)
            if orig_sr != self.resample_hz:
                waveform = torchaudio.functional.resample(waveform, orig_sr, self.resample_hz)
            waveform = torchaudio.functional.highpass_biquad(waveform, self.resample_hz, 25)
            waveform = torchaudio.functional.lowpass_biquad(waveform, self.resample_hz, 400)
            if waveform.shape[1] < self.seg_len:
                waveform = torch.nn.functional.pad(waveform,(0, self.seg_len - waveform.shape[1]))
            return waveform[:, :self.seg_len]

        def __getitem__(self, idx):
            wav_path, lab = self.items[idx]
            wav, sr = torchaudio.load(wav_path)
            x = self._preprocess(wav, sr).float()
            if self.augment:
                if random.random() < 0.2:
                    x = x + torch.randn_like(x)*0.001
            return x, torch.tensor(lab, dtype=torch.float32)

    # ------------------------------------------------------------
    # Train/Val split by patient id
    # ------------------------------------------------------------

    def patient_id_from_path(path:str):
        base = os.path.basename(path)
        if '_' in base:
            return base.split('_')[0]
        return base.split('.')[0]

    patient_ids = [patient_id_from_path(p) for p,_ in combined_items]
    unique_ids = list(set(patient_ids)); random.shuffle(unique_ids)
    split = int(0.8*len(unique_ids))
    train_ids = set(unique_ids[:split])

    train_indices = [i for i,pid in enumerate(patient_ids) if pid in train_ids]
    val_indices   = [i for i,pid in enumerate(patient_ids) if pid not in train_ids]

    ds_train = GenericSegDataset([combined_items[i] for i in train_indices],
                                 resample_hz=args.sr, seg_sec=args.seg_sec, augment=True)
    ds_val   = GenericSegDataset([combined_items[i] for i in val_indices],
                                 resample_hz=args.sr, seg_sec=args.seg_sec, augment=False)

    # ------------------------------------------------------------

    # ---------------- model / loss / optimiser ----------------
    model = InceptionNet1D().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                   steps_per_epoch=len(ds_train),
                                                   epochs=args.epochs, pct_start=0.3,
                                                   div_factor=10)  # gentler ramp

    # --------------- optional focal loss after epoch 10 -------------
    class FocalLoss(torch.nn.Module):
        def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
            super().__init__()
            self.alpha = alpha; self.gamma = gamma
            self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

        def forward(self, logits, targets):
            bce_loss = self.bce(logits, targets)
            p_t = torch.exp(-bce_loss)
            focal_term = self.alpha * (1 - p_t) ** self.gamma
            return (focal_term * bce_loss).mean()

    history = {"sens": [], "spec": [], "acc": [], "auc": []}

    best_auc = -1.0
    best_epoch = -1
    best_thresh_final = 0.5
    patience = 6
    epochs_since_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in tqdm(ds_train, desc=f"Epoch {epoch:02d}/{args.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits.squeeze(1), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        # Evaluation ---------------------------------------------
        model.eval(); val_preds = []; val_true = []
        with torch.no_grad():
            for xb, yb in tqdm(ds_val, desc=f"Val {epoch:02d}/{args.epochs}", leave=False):
                xb, yb = xb.to(device), yb.to(device)
                val_logits = model(xb)
                val_preds.append(torch.sigmoid(val_logits))
                val_true.append(yb)
        val_preds = torch.cat(val_preds)
        val_true = torch.cat(val_true)

        # ---- threshold tuning with Youden's J ----
        probs_np = val_preds.cpu().numpy(); labels_np = val_true.cpu().numpy()
        fpr, tpr, thresholds = roc_curve(labels_np, probs_np)
        j_stat = tpr - fpr
        opt_idx = int(np.argmax(j_stat))
        opt_thresh = float(thresholds[opt_idx])

        def metrics_at_thresh(p,l,th):
            preds = (p >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(l, preds).ravel()
            sens_ = tp/(tp+fn+1e-7); spec_ = tn/(tn+fp+1e-7)
            acc_ = (tp+tn)/(tp+tn+fp+fn+1e-7)
            auc_ = roc_auc_score(l, p)
            return sens_, spec_, acc_, auc_

        sens, spec, acc, auc = metrics_at_thresh(probs_np, labels_np, opt_thresh)

        history["sens"].append(sens); history["spec"].append(spec)
        history["acc"].append(acc); history["auc"].append(auc)

        print(f"Epoch {epoch:03d}: Sens {sens:.3f} Spec {spec:.3f} Acc {acc:.3f} AUC {auc:.3f} (Thr={opt_thresh:.3f})")

        # save best model
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_thresh_final = opt_thresh
            torch.save({'state_dict': model.state_dict(), 'threshold': best_thresh_final},
                       'best_model.pth')
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        # early stopping
        if epochs_since_improve >= patience:
            print(f"Early stopping after {patience} epochs with no AUC improvement.")
            break

        # swap to Focal Loss after 10 epochs
        if epoch == 10:
            criterion = FocalLoss(alpha=0.75, gamma=2.0)

    # also save last epoch weights for reference
    torch.save(model.state_dict(), args.model_out)

    # ----------------- save training curves -----------------------
    epochs_range = range(1, len(history["sens"]) + 1)
    plt.figure(figsize=(8,5))
    for k in ["sens", "spec", "acc", "auc"]:
        plt.plot(epochs_range, history[k], label=k)
    plt.xlabel("Epoch"); plt.ylabel("Value")
    plt.title("Validation metrics vs Epoch")
    plt.legend(); plt.tight_layout()
    plt.savefig("metrics_curve.png")
    plt.close()

    print(f"\nBest AUC {best_auc:.3f} achieved at epoch {best_epoch} with threshold {best_thresh_final:.3f}")

    # SHAP explanation for first validation batch ----------------
    if shap is not None:
        batch_x, _ = next(iter(ds_val))
        batch_x = batch_x.to(device)[:32]
        explainer = shap.DeepExplainer(lambda t: torch.sigmoid(model(t)), batch_x)
        shap_values = explainer.shap_values(batch_x[:1])
        shap.image_plot(shap_values)
    else:
        print("Install 'shap' to generate explanations.")

    # ---------------- ROC curve on final model --------------------
    model.eval(); all_logits=[]; all_labels=[]
    with torch.no_grad():
        for xb, yb in ds_val:
            all_logits.append(torch.sigmoid(model(xb.to(device)).cpu()))
            all_labels.append(yb)
    probs = torch.cat(all_logits).numpy(); labels = torch.cat(all_labels).numpy()
    from sklearn.metrics import auc as sk_auc
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = sk_auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve – Outcome')
    plt.legend(); plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.close()

# ----------------------------- CLI ----------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 1D CNN on CirCor wav data")
    parser.add_argument("--csv", required=True, help="Path to training_data.csv")
    parser.add_argument("--wav_dir", required=True, help="Directory containing wav files")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sr", type=int, default=1000, help="Target sampling rate (Hz)")
    parser.add_argument("--seg_sec", type=int, default=10, help="Segment length (seconds)")
    parser.add_argument("--physio16_csv", help="Path to Online_Appendix_training_set.csv", default=None)
    parser.add_argument("--physio16_wav_dir", help="Directory containing PhysioNet16 wavs", default=None)
    parser.add_argument("--model_out", default="best_circor_net.pth")
    args = parser.parse_args()
    train(args) 