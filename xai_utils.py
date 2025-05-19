"""xai_utils.py – Modular explainability helpers for the heart-sound CNN

This file exposes high-level functions that can be reused from scripts,
web back-ends or notebooks.

Exports
-------
load_model(ckpt_path, device="cuda") -> nn.Module
prepare_explainer(model, background_size=8, seg_len=10_000, device="cuda")
    -> shap.DeepExplainer (cached background inside)

generate_explanation(segment, model, explainer, *, threshold=0.5)
    -> dict with keys: prob (float), attention (np.ndarray),
       shap (np.ndarray), wave (np.ndarray)

visualise_explanation(data:dict, *, show_attention:bool=False)
    -> None (renders matplotlib figure)

The functions assume the training sampling rate (1 kHz) and a 10-second
window (10 000 samples).
"""
from __future__ import annotations

import os, torch, numpy as np, shap, matplotlib.pyplot as plt
from typing import Optional
from train_circor import InceptionNet1D
from explain import integrated_gradients, attention_map


# ---------------------------------------------------------------------
# 1) Model loader ------------------------------------------------------
# ---------------------------------------------------------------------

def load_model(ckpt_path: str, device: str = "cuda") -> torch.nn.Module:
    """Load weights into a fresh InceptionNet1D model and set *eval*."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = InceptionNet1D().to(dev)
    ckpt = torch.load(ckpt_path, map_location=dev)
    # checkpoint may store a full dict or just state_dict
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------
# 2) SHAP explainer cache ---------------------------------------------
# ---------------------------------------------------------------------

def prepare_explainer(
    model: torch.nn.Module,
    *,
    background_size: int = 8,
    seg_len: int = 10_000,
    device: str = "cuda",
) -> shap.DeepExplainer:
    """Return a cached *DeepExplainer* with random‐noise background."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    background = torch.randn(background_size, 1, seg_len, device=dev) * 0.01
    return shap.DeepExplainer(model, background)

# ---------------------------------------------------------------------
# 3) Single-segment explanation ---------------------------------------
# ---------------------------------------------------------------------

def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def generate_explanation(
    segment: np.ndarray | torch.Tensor,
    model: torch.nn.Module,
    explainer: shap.DeepExplainer,
    *,
    threshold: float = 0.5,
) -> dict:
    """Return dictionary with prediction, SHAP, attention, waveform."""
    if isinstance(segment, np.ndarray):
        segment = torch.from_numpy(segment)
    if segment.dim() == 1:
        segment = segment.unsqueeze(0)  # (1,L)

    device = next(model.parameters()).device
    seg = segment.to(device)

    # forward prob
    with torch.no_grad():
        prob = torch.sigmoid(model(seg.unsqueeze(0))).item()

    # SHAP (disable strict additivity)
    phi = explainer.shap_values(seg.unsqueeze(0), check_additivity=False)[0]
    phi = phi.squeeze()  # (L,)

    # attention & IG
    attn = attention_map(model, seg).numpy()
    ig   = integrated_gradients(model, seg).numpy()

    return {
        "prob": prob,
        "wave": seg.cpu().numpy().squeeze(),
        "shap": phi,
        "attention": attn,
        "ig": ig,
        "is_abnormal": prob >= threshold,
    }


# ---------------------------------------------------------------------
# 4) Visualisation -----------------------------------------------------
# ---------------------------------------------------------------------

def visualise_explanation(
    data: dict,
    *,
    show_attention: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 200,
):
    """Plot waveform with SHAP mask (+ optional attention curve).

    Parameters
    ----------
    data : dict
        Output of *generate_explanation*.
    show_attention : bool, default True
        Whether to draw the model's attention curve (green) on a
        secondary y-axis.
    save_path : str, optional
        If given, the figure is saved to this path (PNG/PDF/etc.).
    dpi : int, default 200
        Resolution when *save_path* is used.
    """
    wave = data["wave"]
    phi  = data["shap"]
    attn = data["attention"]
    t = np.arange(len(wave))

    # diverging colormap centred at zero (blue=negative, red=positive)
    vmax = np.abs(phi).max() + 1e-8
    norm_phi = (phi + vmax) / (2 * vmax)  # map [-vmax,vmax] → [0,1]
    cmap = plt.get_cmap("bwr")

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(t, wave, color="black", linewidth=0.8, label="waveform")

    # coloured alpha mask sample-wise (fast enough for 10k samples)
    for i in range(len(wave)):
        ax1.axvspan(i, i + 1, ymin=0, ymax=1, color=cmap(norm_phi[i]),
                    alpha=0.4, linewidth=0)

    if show_attention:
        ax2 = ax1.twinx()
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        ax2.plot(t, attn_norm, color="green", alpha=0.6, label="attention")
        ax2.set_yticks([])

    title = f"Prob= {data['prob']:.2f}  ➜  {'Abnormal' if data['is_abnormal'] else 'Normal'}"
    ax1.set_title(title)
    ax1.set_xlabel("sample"); ax1.set_ylabel("amplitude")
    lines, labels = ax1.get_legend_handles_labels()
    if show_attention:
        l2, lab2 = ax2.get_legend_handles_labels(); lines += l2; labels += lab2
    ax1.legend(lines, labels, loc="upper right")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi)

    plt.show() 