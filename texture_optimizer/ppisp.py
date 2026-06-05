"""
Per-Camera PPISP (Per-Photo ISP) Parameter Model.

Models the camera-specific exposure/color pipeline as differentiable transforms:
  Linear scale → Gamma → White balance (RGB gains) → Brightness/Contrast → Optional vignette

All operations are differentiable; parameters are optimized via backprop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PPISPParams(nn.Module):
    """
    Per-camera image signal processing (ISP) model.

    Learnable parameters per camera:
      - log_exposure  : scalar, log-space exposure multiplier  (exp(x) ~ 0.2..5)
      - gamma         : scalar, display gamma                  (0.8..2.4)
      - wb_gains      : (3,) R/G/B white balance multipliers
      - brightness    : scalar additive shift post-gamma
      - contrast      : scalar multiplicative contrast         (0.5..2)
      - vignette_k    : (optional) vignette falloff strength

    Forward call takes a rendered HDR patch (H, W, 3) in [0,∞) and
    returns a tone-mapped LDR image in [0, 1] that should match the
    ground-truth photograph.
    """

    def __init__(
        self,
        num_cameras: int,
        image_width: int,
        image_height: int,
        init_gamma: float = 2.2,
        learn_vignette: bool = True,
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.W = image_width
        self.H = image_height
        self.learn_vignette = learn_vignette

        # --- exposure in log-space (numerically stable, symmetric around 0) ---
        self.log_exposure = nn.Parameter(torch.zeros(num_cameras))

        # gamma: learn residual around init_gamma  →  gamma = softplus(raw) + 0.5
        self.gamma_raw = nn.Parameter(
            torch.full((num_cameras,), self._gamma_to_raw(init_gamma))
        )

        # white balance gains (log-space per channel so they stay > 0)
        self.log_wb = nn.Parameter(torch.zeros(num_cameras, 3))

        # brightness/contrast (small residuals around identity)
        self.brightness = nn.Parameter(torch.zeros(num_cameras))
        self.log_contrast = nn.Parameter(torch.zeros(num_cameras))

        if learn_vignette:
            # vignette falloff: rendered_px *= exp(-k * r^2),  k > 0
            self.log_vignette_k = nn.Parameter(
                torch.full((num_cameras,), -4.0)  # starts near 0 (k ≈ 0.018)
            )
            # Precompute normalised radial distance map  (H, W)
            ys = torch.linspace(-1, 1, image_height)
            xs = torch.linspace(-1, 1, image_width)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            r2 = grid_x**2 + grid_y**2          # (H, W)  max ≈ 2 at corners
            self.register_buffer("r2", r2)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _gamma_to_raw(gamma: float) -> float:
        # inverse of softplus(x) + 0.5 = gamma  →  raw = softplus_inv(gamma - 0.5)
        g = gamma - 0.5
        if g <= 0:
            return -10.0
        import math
        return math.log(math.expm1(g))

    def gamma(self) -> torch.Tensor:
        """Return positive gamma values  (N,)"""
        return F.softplus(self.gamma_raw) + 0.5      # in (0.5, ∞)

    def wb_gains(self) -> torch.Tensor:
        """Return positive WB gains  (N, 3)"""
        return torch.exp(self.log_wb)                 # (N, 3) > 0

    def exposure(self) -> torch.Tensor:
        """Return positive exposure multipliers  (N,)"""
        return torch.exp(self.log_exposure)           # (N,) > 0

    def contrast(self) -> torch.Tensor:
        """Return positive contrast  (N,)"""
        return F.softplus(self.log_contrast) + 0.5   # (0.5, ∞)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        rendered: torch.Tensor,
        cam_idx: int,
    ) -> torch.Tensor:
        """
        Apply per-camera ISP transform.

        Args:
            rendered:  (H, W, 3) float32 tensor, linear HDR radiance in [0, ∞)
            cam_idx:   index into per-camera parameter arrays

        Returns:
            (H, W, 3) float32 tensor in [0, 1]
        """
        x = rendered.float()                              # (H, W, 3)

        # 1) Exposure
        exp = self.exposure()[cam_idx]                    # scalar
        x = x * exp

        # 2) White balance  (channel-wise)
        wb = self.wb_gains()[cam_idx]                     # (3,)
        x = x * wb.view(1, 1, 3)

        # 3) Vignette (optional)
        if self.learn_vignette:
            k = torch.exp(self.log_vignette_k[cam_idx])  # scalar > 0
            vign = torch.exp(-k * self.r2)               # (H, W)
            x = x * vign.unsqueeze(-1)

        # 4) Gamma compression  (safe power via clamp+eps)
        g = self.gamma()[cam_idx]
        x = torch.clamp(x, min=1e-8)
        x = x ** (1.0 / g)

        # 5) Brightness + Contrast (applied in [0,1] space)
        c = self.contrast()[cam_idx]
        b = self.brightness[cam_idx]
        x = c * (x - 0.5) + 0.5 + b

        # 6) Final clamp to [0, 1]
        x = torch.clamp(x, 0.0, 1.0)
        return x

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------
    def get_params_dict(self, cam_idx: int) -> dict:
        """Human-readable snapshot of one camera's ISP params."""
        with torch.no_grad():
            return {
                "cam_idx":    cam_idx,
                "exposure":   self.exposure()[cam_idx].item(),
                "gamma":      self.gamma()[cam_idx].item(),
                "wb_r":       self.wb_gains()[cam_idx, 0].item(),
                "wb_g":       self.wb_gains()[cam_idx, 1].item(),
                "wb_b":       self.wb_gains()[cam_idx, 2].item(),
                "brightness": self.brightness[cam_idx].item(),
                "contrast":   self.contrast()[cam_idx].item(),
            }

    def print_summary(self):
        print(f"\n{'─'*60}")
        print(f"  PPISP summary  ({self.num_cameras} cameras)")
        print(f"{'─'*60}")
        for i in range(self.num_cameras):
            p = self.get_params_dict(i)
            print(
                f"  cam {i:3d} | exp={p['exposure']:.3f}  γ={p['gamma']:.2f}  "
                f"wb=({p['wb_r']:.3f},{p['wb_g']:.3f},{p['wb_b']:.3f})  "
                f"B={p['brightness']:+.3f}  C={p['contrast']:.3f}"
            )
        print(f"{'─'*60}\n")
