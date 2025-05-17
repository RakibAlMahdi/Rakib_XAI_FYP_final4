import torch
from train_circor import InceptionNet1D

__all__ = ["load_model", "predict_segment"]


def load_model(weight_path: str = "best_combined.pth", device: str = "cpu", *, use_clin: bool = False):
    """Load trained checkpoint and return (model, threshold).

    Parameters
    ----------
    weight_path : str
        Path to .pth file produced by train_pcg.py.
    device : str
        'cpu' or 'cuda'.
    use_clin : bool
        If True, uses thr_clin saved in the checkpoint; otherwise uses thr_acc.
    """
    ckpt = torch.load(weight_path, map_location=device)
    model = InceptionNet1D().to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    thr_key = "thr_clin" if use_clin and "thr_clin" in ckpt else "thr_acc"
    threshold = float(ckpt[thr_key])
    return model, threshold


def predict_segment(model: torch.nn.Module, segment: torch.Tensor, threshold: float) -> tuple[float, int]:
    """Predict probability and binary label for one waveform segment.

    segment : torch.Tensor  shape (1, L) at 1 kHz.
    Returns (probability, label).
    """
    with torch.no_grad():
        prob = torch.sigmoid(model(segment.unsqueeze(0).to(next(model.parameters()).device))).item()
    return prob, int(prob > threshold) 