"""
Differentiable mesh rasterizer.

Backend priority:
  1. nvdiffrast  (GPU, ~1ms per frame for 1M faces) — pip install nvdiffrast

Pipeline:
  world verts → clip space (R, t, K)
  → dr.rasterize  → dr.interpolate (UV + derivatives)
  → dr.texture    → dr.antialias
  → (H, W, 3) HDR render → PPISP → LDR prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Texture map
# ---------------------------------------------------------------------------

class TextureMap(nn.Module):
    """
    Learnable UV texture atlas, shape (1, 3, H_tex, W_tex).
    Initialised from an existing image if provided.
    """
    def __init__(self, H_tex: int, W_tex: int,
                 init_image: Optional[torch.Tensor] = None):
        super().__init__()
        if init_image is not None:
            tex = init_image.permute(2, 0, 1).unsqueeze(0)      # (1,3,H,W)
            tex = F.interpolate(tex, (H_tex, W_tex),
                                mode="bilinear", align_corners=False)
        else:
            tex = torch.full((1, 3, H_tex, W_tex), 0.5)
        self.tex = nn.Parameter(tex)

    def sample(self, uv: torch.Tensor) -> torch.Tensor:
        """Scattered UV lookup — (N,2) in [0,1] → (N,3). Software fallback only."""
        grid = uv * 2.0 - 1.0
        grid[..., 1] = -grid[..., 1]
        grid = grid.unsqueeze(0).unsqueeze(0)                    # (1,1,N,2)
        out = F.grid_sample(torch.clamp(self.tex, 0, 1), grid,
                            mode="bilinear", padding_mode="border",
                            align_corners=False)                 # (1,3,1,N)
        return out.squeeze(0).squeeze(1).T                       # (N,3)

    def as_image(self) -> torch.Tensor:
        """(H_tex, W_tex, 3) float32 in [0,1], detached."""
        with torch.no_grad():
            return torch.clamp(self.tex.squeeze(0).permute(1, 2, 0), 0, 1).float()


# ---------------------------------------------------------------------------
# Clip-space projection
# ---------------------------------------------------------------------------

def to_clip_space(vertices: torch.Tensor,
                  R: torch.Tensor, t: torch.Tensor, K: torch.Tensor,
                  W: int, H: int) -> torch.Tensor:
    """
    (V,3) world vertices → (1,V,4) homogeneous clip-space for nvdiffrast.

    COLMAP convention: x right, y down, z forward.
    nvdiffrast convention: OpenGL clip space, but its Python wrapper flips the
    output framebuffer back to image order (row 0 = top).  So we do NOT negate
    Y — just map pixel coords directly to NDC and multiply by w.

    x_ndc = 2*(fx*Xc/Zc + cx)/W - 1
    y_ndc = 2*(fy*Yc/Zc + cy)/H - 1   (no flip — wrapper handles it)
    z_ndc = linear mapping of depth to [-1, 1]
    w     = Zc  (perspective divide restores NDC)
    """
    pts_cam = (R @ vertices.T).T + t                             # (V,3)
    Xc, Yc, Zc = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    w = Zc.clamp(min=1e-4)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_ndc = (2.0 * (fx * (Xc / w) + cx) / W) - 1.0
    y_ndc = (2.0 * (fy * (Yc / w) + cy) / H) - 1.0

    near, far = 0.01, 10000.0
    z_ndc = (2.0 * w - (far + near)) / (far - near)

    clip = torch.stack([x_ndc * w, y_ndc * w, z_ndc * w, w], dim=1)  # (V,4)
    return clip.unsqueeze(0)                                           # (1,V,4)


# ---------------------------------------------------------------------------
# nvdiffrast rasterizer
# ---------------------------------------------------------------------------

class NvdiffrastRasterizer:
    """Fast GPU rasterizer — requires: pip install nvdiffrast"""

    def __init__(self, device: torch.device):
        self.device    = device
        self.available = False
        try:
            import nvdiffrast.torch as dr
            self.dr = dr
        except ImportError as e:
            print(f"[Renderer] nvdiffrast import failed: {e}")
            print("           pip install nvdiffrast")
            return

        use_cuda = device.type == "cuda"
        try:
            self.glctx = (dr.RasterizeCudaContext() if use_cuda
                          else dr.RasterizeGLContext())
            self.available = True
            print(f"[Renderer] nvdiffrast ready  ({'CUDA' if use_cuda else 'GL'})")
        except Exception as e:
            import traceback
            print(f"[Renderer] nvdiffrast context failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            if use_cuda:
                print("[Renderer] Trying GL fallback ...")
                try:
                    self.glctx = dr.RasterizeGLContext()
                    self.available = True
                    print("[Renderer] nvdiffrast ready  (GL fallback)")
                except Exception as e2:
                    print(f"[Renderer] GL fallback failed: {e2}")

    def render(self,
               vertices: torch.Tensor,  # (V,3) world, float32
               faces:    torch.Tensor,  # (F,3) int32
               uvs:      torch.Tensor,  # (V,2) [0,1]
               texture:  TextureMap,
               R: torch.Tensor, t: torch.Tensor, K: torch.Tensor,
             W: int, H: int,
             return_mask: bool = False):
        """Returns (H, W, 3) float32. Fully differentiable."""
        dr = self.dr

        faces_i32 = faces.to(torch.int32).contiguous()

        # 1) Project to clip space
        verts_clip = to_clip_space(vertices, R, t, K, W, H)      # (1,V,4)

        # 2) Rasterize
        rast, rast_db = dr.rasterize(self.glctx, verts_clip,
                         faces_i32, resolution=[H, W])   # (1,H,W,4)
        rast = rast.contiguous()
        rast_db = rast_db.contiguous()

        # 3) Interpolate UVs + screen-space derivatives for mip-mapping
        uv_attr = uvs.unsqueeze(0).contiguous()                   # (1,V,2)
        texc, texc_db = dr.interpolate(uv_attr, rast, faces_i32,
                                       rast_db=rast_db,
                                       diff_attrs="all")          # (1,H,W,2)

        # 4) Sample texture. Keep atlas dtype when possible to reduce VRAM.
        tex_hwc = texture.tex.permute(0, 2, 3, 1).contiguous()
        texc_c  = texc.to(dtype=tex_hwc.dtype).contiguous()
        texc_db_c = texc_db.to(dtype=tex_hwc.dtype).contiguous()
        try:
            color = dr.texture(tex_hwc, texc_c,
                               uv_da=texc_db_c,
                               filter_mode="linear-mipmap-linear")
        except Exception:
            # Conservative fallback for driver/backend combinations that only
            # accept float32 texture inputs.
            tex_hwc_f32 = tex_hwc.float().contiguous()
            texc_f32 = texc.float().contiguous()
            try:
                color = dr.texture(tex_hwc_f32, texc_f32,
                                   uv_da=texc_db.float().contiguous(),
                                   filter_mode="linear-mipmap-linear")
            except Exception:
                color = dr.texture(tex_hwc_f32, texc_f32, filter_mode="linear")

        # 5) Antialias edges
        color = dr.antialias(color, rast, verts_clip, faces_i32) # (1,H,W,3)
        color = color.squeeze(0)                                   # (H,W,3)
        if not return_mask:
            return color

        mask = (rast[..., 3:4] > 0).float().squeeze(0)            # (H,W,1)
        return color, mask


# ---------------------------------------------------------------------------
# Unified rasterizer — nvdiffrast only
# ---------------------------------------------------------------------------

class Rasterizer:
    def __init__(self, device: torch.device):
        self._nvdr = NvdiffrastRasterizer(device)
        if not self._nvdr.available:
            raise RuntimeError(
                "nvdiffrast is required but unavailable. Install with: pip install nvdiffrast"
            )
        self.uses_nvdiffrast = True

    def render(self, vertices, faces, uvs, texture, R, t, K, W, H) -> torch.Tensor:
        return self._nvdr.render(vertices, faces,
                                 uvs, texture, R, t, K, W, H)

    def render_with_mask(self, vertices, faces, uvs, texture, R, t, K, W, H):
        return self._nvdr.render(vertices, faces,
                                 uvs, texture, R, t, K, W, H,
                                 return_mask=True)