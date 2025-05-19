"""Explanation utilities for InceptionNet1D.

Provides:
    integrated_gradients(model, segment, baseline=None, steps=40)
        Returns IG attribution (Tensor length L).

    attention_map(model, segment)
        Returns attention weights (Tensor length L) produced by the
        model's AttentionPool1d layer.

Both functions expect *segment* to be a 1×L waveform tensor at the same
sampling rate used during training (1 kHz by default).
"""

from __future__ import annotations

import torch
from typing import Optional

# ---------------------------------------------------------------------
# Integrated Gradients -------------------------------------------------
# ---------------------------------------------------------------------

def integrated_gradients(
    model: torch.nn.Module,
    segment: torch.Tensor,
    *,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 40,
) -> torch.Tensor:
    """Compute Integrated Gradients for one 1-D waveform segment.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model; will be run in *eval* mode and gradients enabled.
    segment : torch.Tensor
        Shape (1, L) or (1, 1, L).  Values should be in the same scale
        the model was trained on (raw waveform in ±1 range).
    baseline : torch.Tensor, optional
        Reference input.  If None, an all-zero tensor of identical shape
        is used (interpreted as silence).
    steps : int, default 40
        Number of linear interpolation steps between baseline and input.
        Larger -> smoother attribution but slower (O(steps) forwards).

    Returns
    -------
    torch.Tensor
        Attribution vector of shape (L,) – one score per time step.
    """
    device = next(model.parameters()).device
    model.eval()

    if segment.dim() == 2:  # (1,L) -> (1,1,L)
        segment = segment.unsqueeze(1)
    segment = segment.to(device)

    if baseline is None:
        baseline = torch.zeros_like(segment)
    else:
        baseline = baseline.to(device)

    # generate interpolations: shape (steps, 1, L)
    # (no extra dimension so the batch axis is *steps*)
    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1, 1)
    interp = baseline + alphas * (segment - baseline)  # (steps, 1, L)
    interp.requires_grad_(True)

    # forward pass – sum outputs so we can call backward once
    logits = model(interp)
    if logits.dim() == 3:  # (steps,B,1) if model kept batch dim? ensure (steps,1)
        logits = logits.squeeze(2)
    preds = torch.sigmoid(logits).sum()

    preds.backward()
    grads = interp.grad                           # (steps,1,L)
    avg_grad = grads.mean(dim=0, keepdim=True)    # (1,1,L)
    attrib = (segment - baseline) * avg_grad      # (1,1,L)
    return attrib.squeeze().detach().cpu()        # (L,)

# ---------------------------------------------------------------------
# Attention-based explanation -----------------------------------------
# ---------------------------------------------------------------------

def attention_map(model: torch.nn.Module, segment: torch.Tensor) -> torch.Tensor:
    """Return the normalised attention weights for one segment.

    Requires that InceptionNet1D.forward was patched with *return_attn*.
    """
    model.eval()
    if segment.dim() == 2:
        segment = segment.unsqueeze(1)
    logits, attn = model(segment.to(next(model.parameters()).device), return_attn=True)
    return attn.squeeze().detach().cpu()  # (L,) 