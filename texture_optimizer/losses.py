"""
Loss functions for texture + per-camera PPISP optimization.

Losses:
  - PhotometricLoss   : L1 + perceptual-style SSIM in LDR space
  - TextureRegLoss    : TV (total variation) smoothness prior on texture
  - PPISPRegLoss      : soft regularisation to keep ISP params near identity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .ppisp import PPISPParams
    from .renderer import TextureMap

try:
    from pytorch_msssim import ssim as msssim_ssim
except Exception:
    msssim_ssim = None


# ---------------------------------------------------------------------------
# SSIM (structural similarity) — differentiable, no external deps
# ---------------------------------------------------------------------------

def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = torch.outer(g, g)
    return kernel / kernel.sum()


_SSIM_WINDOW: Optional[torch.Tensor] = None
_SSIM_WIN_SIZE = 11


def _ssim_loss_native(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    SSIM-based loss in [0, 1].  Lower is better.
    pred / target: (H, W, 3) float32 in [0, data_range]
    """
    global _SSIM_WINDOW
    device = pred.device

    if _SSIM_WINDOW is None or _SSIM_WINDOW.device != device:
        _SSIM_WINDOW = _gaussian_kernel(_SSIM_WIN_SIZE).to(device)

    kernel = _SSIM_WINDOW.to(dtype=pred.dtype).view(1, 1, _SSIM_WIN_SIZE, _SSIM_WIN_SIZE).expand(3, 1, -1, -1)
    p = pred.permute(2, 0, 1).unsqueeze(0)      # (1, 3, H, W)
    t = target.permute(2, 0, 1).unsqueeze(0)    # (1, 3, H, W)
    pad = _SSIM_WIN_SIZE // 2

    mu1 = F.conv2d(p, kernel, padding=pad, groups=3)
    mu2 = F.conv2d(t, kernel, padding=pad, groups=3)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12   = mu1 * mu2

    sigma1_sq = F.conv2d(p * p, kernel, padding=pad, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(t * t, kernel, padding=pad, groups=3) - mu2_sq
    sigma12   = F.conv2d(p * t, kernel, padding=pad, groups=3) - mu12

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / torch.clamp(denom, min=1e-8)

    return 1.0 - ssim_map.mean()


def _ssim_loss_msssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Fast SSIM path using pytorch_msssim if available.
    pred / target: (H, W, 3) float32 in [0, data_range]
    """
    if msssim_ssim is None:
        return _ssim_loss_native(pred, target, data_range=data_range)

    p = pred.permute(2, 0, 1).unsqueeze(0)
    t = target.permute(2, 0, 1).unsqueeze(0)
    try:
        ssim_val = msssim_ssim(
            p,
            t,
            data_range=data_range,
            size_average=True,
            nonnegative_ssim=True,
        )
    except TypeError:
        ssim_val = msssim_ssim(
            p,
            t,
            data_range=data_range,
            size_average=True,
        )
    return 1.0 - ssim_val


# ---------------------------------------------------------------------------
# Photometric loss
# ---------------------------------------------------------------------------

class PhotometricLoss(nn.Module):
    """
    Combined L1 + SSIM photometric loss.
    Both pred and target should be in [0, 1] (post-ISP LDR space).
    """

    def __init__(
        self,
        l1_weight: float = 0.8,
        ssim_weight: float = 0.2,
        ssim_backend: str = "auto",
    ):
        super().__init__()
        self.l1_weight   = l1_weight
        self.ssim_weight = ssim_weight
        self.ssim_backend = ssim_backend

    def _compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        backend = self.ssim_backend
        if backend == "msssim":
            return _ssim_loss_msssim(pred, target)
        if backend == "native":
            return _ssim_loss_native(pred, target)
        # auto: prefer accelerated backend when installed.
        if msssim_ssim is not None:
            return _ssim_loss_msssim(pred, target)
        return _ssim_loss_native(pred, target)

    def forward(
        self,
        pred:   torch.Tensor,   # (H, W, 3) predicted LDR image
        target: torch.Tensor,   # (H, W, 3) ground-truth photograph
        mask:   Optional[torch.Tensor] = None,  # (H, W) bool; None = all pixels
    ) -> torch.Tensor:
        if mask is not None:
            valid = mask
            if valid.any():
                pred_m   = pred[valid]
                target_m = target[valid]
            else:
                pred_m, target_m = pred, target
            m = mask.to(dtype=pred.dtype).unsqueeze(-1)
            pred_ssim = pred * m
            target_ssim = target * m
        else:
            pred_m, target_m = pred, target
            pred_ssim, target_ssim = pred, target

        l1   = F.l1_loss(pred_m, target_m)
        ssim = self._compute_ssim(pred_ssim, target_ssim)

        return self.l1_weight * l1 + self.ssim_weight * ssim


# ---------------------------------------------------------------------------
# Texture regularisation  (total variation)
# ---------------------------------------------------------------------------

class TextureRegLoss(nn.Module):
    """
    Total variation smoothness regulariser on the texture map.
    Penalises sharp high-frequency variations in the learned texture.
    """

    def forward(self, tex: torch.Tensor) -> torch.Tensor:
        """
        tex: (1, 3, H_tex, W_tex)
        """
        diff_h = tex[:, :, 1:, :] - tex[:, :, :-1, :]
        diff_w = tex[:, :, :, 1:] - tex[:, :, :, :-1]
        return diff_h.abs().mean() + diff_w.abs().mean()


# ---------------------------------------------------------------------------
# Per-camera ISP regularisation
# ---------------------------------------------------------------------------

class PPISPRegLoss(nn.Module):
    """
    Soft L2 regulariser to keep PPISP params near identity values.

    Prevents degenerate solutions (e.g. all exposure → 0, all contrast → ∞).
    Weights are typically much smaller than photometric loss.
    """

    def __init__(
        self,
        exposure_weight:   float = 0.01,
        gamma_weight:      float = 0.01,
        wb_weight:         float = 0.01,
        brightness_weight: float = 0.1,
        contrast_weight:   float = 0.01,
        vignette_weight:   float = 0.01,
    ):
        super().__init__()
        self.w_exp  = exposure_weight
        self.w_gam  = gamma_weight
        self.w_wb   = wb_weight
        self.w_bri  = brightness_weight
        self.w_con  = contrast_weight
        self.w_vig  = vignette_weight

    def forward(self, ppisp: "PPISPParams") -> torch.Tensor:
        loss = torch.tensor(0.0, device=ppisp.log_exposure.device)

        # exposure → 1  (log → 0)
        loss = loss + self.w_exp * ppisp.log_exposure.pow(2).mean()

        # gamma → 2.2  (softplus(raw) + 0.5 = 2.2  →  raw ≈ 1.67)
        loss = loss + self.w_gam * (ppisp.gamma() - 2.2).pow(2).mean()

        # wb → 1  (log → 0)
        loss = loss + self.w_wb  * ppisp.log_wb.pow(2).mean()

        # brightness → 0
        loss = loss + self.w_bri * ppisp.brightness.pow(2).mean()

        # contrast → 1  (log → 0)
        loss = loss + self.w_con * ppisp.log_contrast.pow(2).mean()

        # vignette → near-0  (log_k → -4 ≈ k≈0.018)
        if ppisp.learn_vignette:
            loss = loss + self.w_vig * (ppisp.log_vignette_k + 4.0).pow(2).mean()

        return loss


# ---------------------------------------------------------------------------
# Combined loss aggregator
# ---------------------------------------------------------------------------

class TotalLoss(nn.Module):
    """
    Combines all losses with configurable weights.
    """

    def __init__(
        self,
        photo_weight:      float = 1.0,
        tex_reg_weight:    float = 1e-4,
        ppisp_reg_weight:  float = 1e-2,
        l1_weight:         float = 0.8,
        ssim_weight:       float = 0.2,
        ssim_backend:      str = "auto",
    ):
        super().__init__()
        self.w_photo    = photo_weight
        self.w_tex_reg  = tex_reg_weight
        self.w_ppisp    = ppisp_reg_weight

        self.photo_loss   = PhotometricLoss(
            l1_weight=l1_weight,
            ssim_weight=ssim_weight,
            ssim_backend=ssim_backend,
        )
        self.tex_reg      = TextureRegLoss()
        self.ppisp_reg    = PPISPRegLoss()

    def forward(
        self,
        pred:    torch.Tensor,
        target:  torch.Tensor,
        texture: "TextureMap",
        ppisp:   "PPISPParams",
        mask:    Optional[torch.Tensor] = None,
    ) -> dict:
        photo = self.photo_loss(pred, target, mask)
        t_reg = self.tex_reg(texture.tex)
        p_reg = self.ppisp_reg(ppisp)

        total = (self.w_photo   * photo
               + self.w_tex_reg * t_reg
               + self.w_ppisp   * p_reg)

        return {
            "total":     total,
            "photo":     photo.detach(),
            "tex_reg":   t_reg.detach(),
            "ppisp_reg": p_reg.detach(),
        }
