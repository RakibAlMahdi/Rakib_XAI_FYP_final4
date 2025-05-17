import argparse, random, os
import pandas as pd, torch, numpy as np
from torch.utils.data import DataLoader
from datasets import SegmentPCGDataset
from train_circor import InceptionNet1D
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def compute_metrics(pred, true, thr: float = 0.5):
    """Return sensitivity, specificity, accuracy, AUC at given threshold."""
    pred_bin = (pred > thr).cpu().numpy()
    y = true.cpu().numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred_bin, labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-7)
    spec = tn / (tn + fp + 1e-7)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    auc = roc_auc_score(y, pred.cpu())
    return sens, spec, acc, auc


def main():
    parser = argparse.ArgumentParser(description="Train PCG model on combined metadata")
    parser.add_argument('--meta_csv', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sr', type=int, default=1000)
    parser.add_argument('--seg_sec', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0, help='DataLoader workers (0 avoids Windows spawn issues)')
    parser.add_argument('--focal', action='store_true', help='Use focal loss instead of BCE')
    parser.add_argument('--freeze_blocks', type=int, default=0, help='Freeze first N inception blocks')
    parser.add_argument('--patience', type=int, default=8, help='Early stop patience epochs')
    parser.add_argument('--use_swa', action='store_true', help='Enable Stochastic Weight Averaging at end')
    parser.add_argument('--hard_neg_k', type=int, default=0, help='Max hard negative cache size')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--best_metric', choices=['auc','bal_acc'], default='auc', help='Metric used to track best checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    # ------------------------------------------------------------
    # Load metadata and split patients
    # ------------------------------------------------------------
    df = pd.read_csv(args.meta_csv)
    patient_ids = df['patient_id'].unique().tolist()
    random.shuffle(patient_ids)
    train_ids = set(patient_ids[: int(0.8 * len(patient_ids))])
    train_df = df[df['patient_id'].isin(train_ids)].reset_index(drop=True)
    val_df = df[~df['patient_id'].isin(train_ids)].reset_index(drop=True)

    # ------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------
    train_ds = SegmentPCGDataset(train_df, resample_hz=args.sr, segment_sec=args.seg_sec, augment=True)
    val_ds = SegmentPCGDataset(val_df, resample_hz=args.sr, segment_sec=args.seg_sec, augment=False)

    def collate_fn(batch):
        xs, ys, pids = zip(*batch)
        return torch.stack(xs), torch.tensor(ys), list(pids)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    # ------------------------------------------------------------
    # Model & optimiser
    # ------------------------------------------------------------
    model = InceptionNet1D().to(device)

    # ---------------- loss --------------------
    if args.focal:
        class FocalLoss(torch.nn.Module):
            def __init__(self, gamma=2.0, alpha=0.75):
                super().__init__()
                self.gamma = gamma; self.alpha = alpha
                self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
            def forward(self, logits, y):
                bce_loss = self.bce(logits, y)
                pt = torch.exp(-bce_loss)
                focal = self.alpha * (1-pt)**self.gamma * bce_loss
                return focal.mean()
        criterion = FocalLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_auc = -1.0
    best_epoch = -1
    best_thresh = 0.5
    best_acc_thr = 0.5

    # optional resume
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        best_thresh = ckpt.get('thr_youden', 0.5)
        best_acc_thr = ckpt.get('thr_acc', 0.5)
        print(f'Resumed from {args.resume}')

    # optional layer freezing
    if args.freeze_blocks > 0:
        for name, param in model.named_parameters():
            if name.startswith('blocks.'):
                block_idx = int(name.split('.')[1])
                if block_idx < args.freeze_blocks:
                    param.requires_grad = False

    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb, _pids in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            # ---------- mixup ----------
            if random.random() < 0.3:
                lam = np.random.beta(0.4,0.4)
                perm = torch.randperm(xb.size(0))
                xb = lam*xb + (1-lam)*xb[perm]
                yb = lam*yb + (1-lam)*yb[perm]

            logits = model(xb)
            # label smoothing 0.05
            yb_smooth = yb*0.95 + 0.025
            loss = criterion(logits.squeeze(1), yb_smooth)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch}: train_loss {epoch_loss / len(train_loader):.4f}')

        # ---------------- validation -----------------
        model.eval()
        probs, labels, patient_ids = [], [], []
        with torch.no_grad():
            for xb, yb, pids in val_loader:
                p = torch.sigmoid(model(xb.to(device))).cpu()
                probs.append(p); labels.append(yb); patient_ids.extend(pids)
        probs = torch.cat(probs); labels = torch.cat(labels)

        # ------- patient-level aggregation ---------
        patient_dict = {}
        for prob, lab, pid in zip(probs, labels, patient_ids):
            patient_dict.setdefault(pid, []).append(prob.item())
        pat_scores = torch.tensor([np.mean(v) for v in patient_dict.values()])
        pat_labels = torch.tensor([labels[patient_ids.index(pid)].item() for pid in patient_dict.keys()])

        # ----- thresholds -----
        fpr, tpr, thr = roc_curve(pat_labels.numpy(), pat_scores.numpy())
        youden = tpr - fpr
        ydx = youden.argmax(); thr_youden = thr[ydx]

        pos_frac = pat_labels.float().mean().item()
        accs = tpr * pos_frac + (1 - fpr) * (1 - pos_frac)
        adx = accs.argmax(); thr_acc = thr[adx]

        # metrics at both thresholds
        sens_y, spec_y, acc_y, _ = compute_metrics(pat_scores, pat_labels, thr_youden)
        sens_a, spec_a, acc_a, _ = compute_metrics(pat_scores, pat_labels, thr_acc)
        auc = roc_auc_score(pat_labels.numpy(), pat_scores.numpy())

        bal_acc = (sens_y + spec_y)/2

        print(
            f'Epoch {epoch}: AUC {auc:.3f} | '
            f'Thr_Youden {thr_youden:.3f}  Sens {sens_y:.3f} Spec {spec_y:.3f} Acc {acc_y:.3f} | '
            f'Thr_Acc {thr_acc:.3f}  Acc {acc_a:.3f}')

        # keep best by AUC but record both thresholds
        is_better = (auc > best_auc) if args.best_metric == 'auc' else (bal_acc > best_auc)
        if is_better:
            best_auc = auc if args.best_metric=='auc' else bal_acc
            best_epoch = epoch; best_thresh = thr_youden; best_acc_thr = thr_acc
            torch.save({
                'state_dict': model.state_dict(),
                'thr_youden': float(best_thresh),
                'thr_acc': float(best_acc_thr)
            }, 'best_combined.pth')
            print(f'Saved new best model (epoch {epoch})')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # early stopping
        if epochs_no_improve >= args.patience:
            print(f'Early stopping after {args.patience} epochs without AUC improvement.')
            break

    print(f'Best epoch {best_epoch}: AUC {best_auc:.3f} | thr_youden {best_thresh:.3f} | thr_acc {best_acc_thr:.3f}')

    # --------------------------- SWA -----------------------
    if args.use_swa:
        print('Running SWA fine-tune...')
        swa_model = torch.optim.swa_utils.AveragedModel(model).to(device)
        swa_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=args.lr*0.1)
        for _ in range(5):
            model.train()
            for xb,yb,_ in train_loader:
                xb,yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    logits=model(xb)
                    yb_s=yb*0.95+0.025
                    loss=criterion(logits.squeeze(1), yb_s)
                loss.backward(); optimizer.step()
            swa_model.update_parameters(model)
            swa_sched.step()

        swa_model_cpu = swa_model.cpu()
        torch.optim.swa_utils.update_bn(train_loader, swa_model_cpu)
        swa_model = swa_model_cpu.to(device)
        # evaluate swa model
        swa_model.eval(); probs_swa=[]; labels_swa=[]
        with torch.no_grad():
            for xb,yb,_ in val_loader:
                p=torch.sigmoid(swa_model(xb.to(device))).cpu(); probs_swa.append(p); labels_swa.append(yb)
        probs_swa=torch.cat(probs_swa); labels_swa=torch.cat(labels_swa)
        auc_swa=roc_auc_score(labels_swa.numpy(), probs_swa.numpy())
        print(f'SWA AUC {auc_swa:.3f}')
        if auc_swa>best_auc:
            torch.save({'state_dict': swa_model.state_dict(), 'thr_youden': best_thresh, 'thr_acc': best_acc_thr}, 'best_combined.pth')
            print('SWA model became new best checkpoint')


if __name__ == '__main__':
    # Needed for Windows to avoid recursive spawning
    import multiprocessing as mp

    mp.freeze_support()
    main() 