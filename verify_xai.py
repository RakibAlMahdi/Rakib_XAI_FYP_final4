# verify_xai.py
import argparse, torch, torchaudio, shap
from train_circor import InceptionNet1D           # or from train_pcg if you changed the name
from explain import integrated_gradients, attention_map

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="best_combined.pth",
                    help="checkpoint produced by train_pcg.py")
parser.add_argument("--wav", required=True,
                    help="any WAV file at 1 kHz you want to explain")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

# ------------------------------------------------------------------
# 1) Load model and weights
# ------------------------------------------------------------------
ckpt = torch.load(args.ckpt, map_location=device)
model = InceptionNet1D().to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print("Model weights loaded OK")

# ------------------------------------------------------------------
# 2) Load + preprocess the WAV into a 1 × L tensor (10 s → 10 000)
# ------------------------------------------------------------------
wav, sr = torchaudio.load(args.wav)
if sr != 1000:
    wav = torchaudio.functional.resample(wav, sr, 1000)
wav = torch.nn.functional.pad(wav, (0, max(0, 10_000 - wav.shape[1])))[:, :10_000]
segment = wav.to(device)

# ------------------------------------------------------------------
# 3) Attention map
# ------------------------------------------------------------------
_, attn = model(segment.unsqueeze(0), return_attn=True)
print("Attention shape :", attn.shape)

# ------------------------------------------------------------------
# 4) Integrated Gradients
# ------------------------------------------------------------------
ig = integrated_gradients(model, segment)
print("IG shape        :", ig.shape)

# ------------------------------------------------------------------
# 5) SHAP (use 32 random noise segments as background just for test)
# ------------------------------------------------------------------
background = torch.randn(32, 1, 10_000, device=device) * 0.01
# DeepExplainer supports PyTorch models directly; we pass the nn.Module so SHAP
# can compute gradients internally. Using raw logits is fine for binary class.
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(segment.unsqueeze(0))
# shap_values is a list with one array (one output class)
shap_vals = shap_values[0]  # shape (1, 1, L)
print("SHAP shape      :", torch.tensor(shap_vals).shape)

print("\nSUCCESS – all three explanation tensors produced.")