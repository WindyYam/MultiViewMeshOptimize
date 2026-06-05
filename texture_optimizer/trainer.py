"""
Texture + Per-Camera PPISP Joint Optimiser

Training loop:
  - Random camera sampling each iteration (like 3DGS)
  - Warmup phase: PPISP only, texture frozen
  - Joint phase: texture + PPISP together
  - Cosine LR decay
  - Checkpoint every N iters, export at end
"""

import os, math, time, random
from pathlib import Path
from typing import Optional, List, Callable
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np

from .ppisp    import PPISPParams
from .renderer import TextureMap, Rasterizer
from .losses   import TotalLoss
from .dataset  import ColmapScene, CameraView


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TrainConfig:
    num_iterations:   int   = 5_000
    warmup_iters:     int   = 500      # PPISP only, texture frozen

    lr_texture:       float = 1e-3
    lr_ppisp:         float = 5e-3
    lr_decay_start:   int   = 3_000
    lr_decay_factor:  float = 0.1
    lr_decay_iters:   int   = 2_000

    photo_weight:     float = 1.0
    tex_reg_weight:   float = 5e-5
    ppisp_reg_weight: float = 1e-2

    tex_H:            int   = 1024
    tex_W:            int   = 1024

    log_every:        int   = 50
    save_every:       int   = 500
    output_dir:       str   = "outputs"
    device:           str   = "cuda" if torch.cuda.is_available() else "cpu"
    seed:             int   = 42
    learn_vignette:   bool  = True
    learn_geometry:   bool  = False
    lr_geometry:      float = 1e-4
    geometry_warmup_iters: int = 1000
    geom_reg_l2_weight:    float = 1e-3
    geom_reg_edge_weight:  float = 1e-2
    max_vertex_offset:     Optional[float] = 0.03
    weld_geometry_vertices: bool = True
    weld_position_decimals: int = 6
    live_view:             bool  = False
    live_view_every:       int   = 50
    live_view_camera:      int   = 0
    live_view_max_size:    int   = 1200
    use_amp:               bool  = False
    amp_dtype:             str   = "fp16"   # fp16 or bf16
    use_tf32:              bool  = True


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TexturePPISPTrainer:

    def __init__(self, scene: ColmapScene, config: TrainConfig):
        self.scene  = scene
        self.cfg    = config
        self.device = torch.device(config.device)

        torch.manual_seed(config.seed)
        random.seed(config.seed)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        self._amp_enabled = False
        self._amp_dtype = None
        self._grad_scaler = None
        self._configure_acceleration()

        # ---- Mesh ----
        self.base_vertices = scene.mesh.vertices.to(self.device)
        self.faces    = scene.mesh.faces.to(self.device)
        self.uvs      = scene.mesh.uvs.to(self.device)

        # Optional geometry branch: optimise per-vertex offsets from COLMAP mesh.
        self.learn_geometry = bool(config.learn_geometry)
        self._weld_index = None
        self._num_weld_groups = int(self.base_vertices.shape[0])
        self._weld_enabled = False
        if self.learn_geometry and bool(config.weld_geometry_vertices):
            self._weld_index, self._num_weld_groups = self._build_weld_groups(
                self.base_vertices, config.weld_position_decimals
            )
            self._weld_enabled = self._num_weld_groups < int(self.base_vertices.shape[0])

        geom_shape = ((self._num_weld_groups, 3) if self._weld_enabled
                      else tuple(self.base_vertices.shape))
        self.geometry_offsets = nn.Parameter(
            torch.zeros(geom_shape, device=self.device, dtype=self.base_vertices.dtype),
            requires_grad=self.learn_geometry,
        )
        # Backward-compat alias for places that still refer to vertex_offsets.
        self.vertex_offsets = self.geometry_offsets
        self._max_vertex_offset = config.max_vertex_offset
        self._edges = self._build_unique_edges(self.faces)
        if self._edges is not None and self._edges.shape[0] > 0:
            v0 = self.base_vertices[self._edges[:, 0]]
            v1 = self.base_vertices[self._edges[:, 1]]
            self._base_edge_lengths = (v0 - v1).norm(dim=1).detach()
        else:
            self._base_edge_lengths = None

        # ---- Texture ----
        self.texture = TextureMap(
            config.tex_H, config.tex_W,
            init_image=scene.mesh.tex_image,
        ).to(self.device)

        # ---- PPISP ----
        v0 = scene.views[0]
        self.ppisp = PPISPParams(
            num_cameras=len(scene),
            image_width=v0.W, image_height=v0.H,
            learn_vignette=config.learn_vignette,
        ).to(self.device)

        # ---- Rasterizer ----
        self.rasterizer = Rasterizer(self.device)
        if not self.rasterizer.uses_nvdiffrast:
            print("[Trainer] WARNING: nvdiffrast not available — using slow software rasterizer")
            print("          Install with:  pip install nvdiffrast")

        # ---- Pre-cache camera matrices on GPU ----
        self._cam_R = [v.R.to(self.device) for v in scene.views]
        self._cam_t = [v.t.to(self.device) for v in scene.views]
        self._cam_K = [v.K.to(self.device) for v in scene.views]

        # ---- Pin GT images for fast async GPU transfer ----
        for v in scene.views:
            if v.gt_image is not None and self.device.type == "cuda":
                v.gt_image = v.gt_image.pin_memory()

        # ---- Loss ----
        self.loss_fn = TotalLoss(
            photo_weight=config.photo_weight,
            tex_reg_weight=config.tex_reg_weight,
            ppisp_reg_weight=config.ppisp_reg_weight,
        )

        # ---- Optimizers ----
        self.opt_tex   = optim.Adam(self.texture.parameters(),
                                    lr=config.lr_texture,
                                    betas=(0.9, 0.99), eps=1e-15)
        self.opt_ppisp = optim.Adam(self.ppisp.parameters(),
                                    lr=config.lr_ppisp)
        self.opt_geom  = (optim.Adam([self.geometry_offsets], lr=config.lr_geometry)
                  if self.learn_geometry else None)

        self.iter     = 0
        self.loss_log: List[dict] = []

        self._cv2 = None
        self._live_view_enabled = False
        self._live_view_window = "TextureOptimizer Live"
        self._live_view_cam = 0
        self._live_view_selector_root = None
        self._live_view_selector_var = None
        self._init_live_viewer()

        n_tex = sum(p.numel() for p in self.texture.parameters())
        n_ppisp = sum(p.numel() for p in self.ppisp.parameters())
        n_geom = int(self.geometry_offsets.numel()) if self.learn_geometry else 0
        print(f"\n[Trainer] Device    : {self.device}")
        print(f"[Trainer] Backend   : {'nvdiffrast' if self.rasterizer.uses_nvdiffrast else 'software (SLOW)'}")
        print(f"[Trainer] Cameras   : {len(scene)}")
        print(f"[Trainer] Mesh      : {self.base_vertices.shape[0]:,} verts, {self.faces.shape[0]:,} faces")
        print(f"[Trainer] Texture   : {config.tex_H}×{config.tex_W}  ({n_tex/1e6:.1f}M params)")
        print(f"[Trainer] PPISP     : {n_ppisp} params  ({len(scene)} cameras)")
        print(
            f"[Trainer] Geometry  : {'on' if self.learn_geometry else 'off'}"
            f"{f'  ({n_geom/1e6:.1f}M offset params)' if self.learn_geometry else ''}"
        )
        if self.learn_geometry:
            print(
                f"[Trainer] WeldGeom  : {'on' if self._weld_enabled else 'off'}"
                f"  ({self._num_weld_groups:,} groups from {self.base_vertices.shape[0]:,} verts)"
            )
        print(
            f"[Trainer] AMP       : {'on' if self._amp_enabled else 'off'}"
            f"{f' ({config.amp_dtype})' if self._amp_enabled else ''}"
        )
        print(f"[Trainer] TF32      : {'on' if (self.device.type == 'cuda' and config.use_tf32) else 'off'}")
        print(f"[Trainer] Iterations: {config.num_iterations}")

    def _configure_acceleration(self):
        cfg = self.cfg
        if self.device.type == "cuda" and bool(cfg.use_tf32):
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        if self.device.type != "cuda" or not bool(cfg.use_amp):
            return

        amp_mode = str(cfg.amp_dtype).lower()
        if amp_mode == "bf16":
            self._amp_dtype = torch.bfloat16
            self._amp_enabled = True
            # bf16 typically does not require gradient scaling.
            self._grad_scaler = None
        else:
            self._amp_dtype = torch.float16
            self._amp_enabled = True
            self._grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

    def _autocast_ctx(self):
        if not self._amp_enabled or self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._amp_dtype)

    def _init_live_viewer(self):
        cfg = self.cfg
        if not cfg.live_view:
            return
        try:
            import cv2
            self._cv2 = cv2
        except Exception as e:
            print(f"[LiveView] OpenCV unavailable ({e}). Disable live view or install opencv-python.")
            return

        if len(self.scene.views) == 0:
            print("[LiveView] No cameras available.")
            return

        self._live_view_cam = int(max(0, min(cfg.live_view_camera, len(self.scene.views) - 1)))
        self._live_view_enabled = True
        self._cv2.namedWindow(self._live_view_window, self._cv2.WINDOW_NORMAL)
        print("[LiveView] Enabled: dropdown select camera, keys n/] next cam, p/[ prev cam, q or ESC to close")

        # Optional dropdown camera selector window (Tkinter).
        self._init_live_view_dropdown()

    def _init_live_view_dropdown(self):
        try:
            import tkinter as tk
            from tkinter import ttk
        except Exception:
            print("[LiveView] Tkinter unavailable; dropdown selector disabled.")
            return

        try:
            root = tk.Tk()
            root.title("Camera Selector")
            root.geometry("300x90")
            root.resizable(False, False)

            ttk.Label(root, text="Live camera").pack(anchor="w", padx=10, pady=(8, 2))
            options = [f"cam_{i:04d}" for i in range(len(self.scene.views))]
            var = tk.StringVar(value=options[self._live_view_cam])

            combo = ttk.Combobox(root, textvariable=var, values=options, state="readonly")
            combo.pack(fill="x", padx=10)

            def _on_select(_event=None):
                value = var.get()
                if value.startswith("cam_"):
                    try:
                        self._live_view_cam = int(value.split("_", 1)[1])
                    except Exception:
                        pass

            combo.bind("<<ComboboxSelected>>", _on_select)

            ttk.Label(root, text="Close this window or press q in preview to disable live view.").pack(
                anchor="w", padx=10, pady=(6, 0)
            )

            self._live_view_selector_root = root
            self._live_view_selector_var = var
        except Exception as e:
            self._live_view_selector_root = None
            self._live_view_selector_var = None
            print(f"[LiveView] Could not create dropdown selector: {e}")

    def _poll_live_view_keys(self, key: int):
        if key < 0 or not self._live_view_enabled:
            return
        key_lo = key & 0xFF
        quit_codes = {27, ord("q")}
        next_codes = {ord("n"), ord("]"), 83, 2555904}
        prev_codes = {ord("p"), ord("["), 81, 2424832}

        if key in quit_codes or key_lo in quit_codes:
            self._live_view_enabled = False
            try:
                self._cv2.destroyWindow(self._live_view_window)
            except Exception:
                pass
            print("[LiveView] Closed.")
            return

        if key in next_codes or key_lo in next_codes:
            self._live_view_cam = (self._live_view_cam + 1) % len(self.scene.views)
            print(f"[LiveView] Camera -> {self._live_view_cam}")
        elif key in prev_codes or key_lo in prev_codes:
            self._live_view_cam = (self._live_view_cam - 1) % len(self.scene.views)
            print(f"[LiveView] Camera -> {self._live_view_cam}")

        if self._live_view_selector_var is not None:
            self._live_view_selector_var.set(f"cam_{self._live_view_cam:04d}")

    def _poll_live_view_events(self):
        if not self._live_view_enabled or self._cv2 is None:
            return

        # Process OpenCV events every iteration to keep the window responsive.
        key_fn = getattr(self._cv2, "waitKeyEx", self._cv2.waitKey)
        key = key_fn(1)
        self._poll_live_view_keys(key)

        # Pump Tkinter events so dropdown stays responsive.
        if self._live_view_selector_root is not None:
            try:
                self._live_view_selector_root.update_idletasks()
                self._live_view_selector_root.update()
            except Exception:
                self._live_view_selector_root = None
                self._live_view_selector_var = None

    def _update_live_view(self, losses: dict):
        if not self._live_view_enabled or self._cv2 is None:
            return
        if self.cfg.live_view_every <= 0:
            return
        if (self.iter + 1) % self.cfg.live_view_every != 0:
            return

        view = self.scene.views[self._live_view_cam]
        with torch.no_grad():
            pred = self.render_view(view).detach().clamp(0, 1).cpu().numpy()
        gt = view.gt_image
        if gt is not None:
            gt_np = gt.detach().cpu().numpy() if torch.is_tensor(gt) else np.asarray(gt)
            panel = np.concatenate([pred, gt_np], axis=1)
        else:
            panel = pred

        img = (np.clip(panel, 0, 1) * 255.0).astype(np.uint8)

        max_size = max(64, int(self.cfg.live_view_max_size))
        h, w = img.shape[:2]
        scale = min(1.0, max_size / float(max(h, w)))
        if scale < 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = self._cv2.resize(img, (new_w, new_h), interpolation=self._cv2.INTER_AREA)

        bgr = self._cv2.cvtColor(img, self._cv2.COLOR_RGB2BGR)
        label = f"iter={self.iter+1} cam={self._live_view_cam} loss={losses.get('total', 0.0):.4f}"
        self._cv2.putText(
            bgr, label, (12, 28), self._cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (50, 220, 50), 2, self._cv2.LINE_AA
        )
        self._cv2.putText(
            bgr, "left: pred, right: gt", (12, 56), self._cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (220, 220, 220), 1, self._cv2.LINE_AA
        )

        self._cv2.imshow(self._live_view_window, bgr)

    def _close_live_view(self):
        if self._live_view_selector_root is not None:
            try:
                self._live_view_selector_root.destroy()
            except Exception:
                pass
            self._live_view_selector_root = None
            self._live_view_selector_var = None

        if self._cv2 is None:
            return
        try:
            self._cv2.destroyWindow(self._live_view_window)
        except Exception:
            pass

    def _build_weld_groups(self, vertices: torch.Tensor, decimals: int):
        """
        Group vertices by rounded XYZ so duplicated seam vertices move together.
        Returns (vertex_to_group_index, num_groups).
        """
        verts_np = vertices.detach().cpu().numpy()
        key_np = np.round(verts_np, decimals=max(0, int(decimals)))
        _, inverse = np.unique(key_np, axis=0, return_inverse=True)
        v2g = torch.from_numpy(inverse.astype(np.int64)).to(self.device)
        return v2g, int(v2g.max().item() + 1) if v2g.numel() > 0 else 0

    def _expanded_geometry_offsets(self) -> torch.Tensor:
        if not self._weld_enabled or self._weld_index is None:
            return self.geometry_offsets
        return self.geometry_offsets[self._weld_index]

    def _build_unique_edges(self, faces: torch.Tensor) -> Optional[torch.Tensor]:
        if faces.numel() == 0:
            return None
        e01 = faces[:, [0, 1]]
        e12 = faces[:, [1, 2]]
        e20 = faces[:, [2, 0]]
        edges = torch.cat([e01, e12, e20], dim=0)
        edges = torch.sort(edges, dim=1).values
        return torch.unique(edges, dim=0)

    def current_vertices(self) -> torch.Tensor:
        if not self.learn_geometry:
            return self.base_vertices
        offsets_full = self._expanded_geometry_offsets()
        if self._max_vertex_offset is None:
            offsets = offsets_full
        else:
            offsets = torch.tanh(offsets_full) * float(self._max_vertex_offset)
        return self.base_vertices + offsets

    def _geometry_regularization(self, vertices: torch.Tensor) -> dict:
        if (not self.learn_geometry or self._edges is None or
                self._base_edge_lengths is None or self._edges.shape[0] == 0):
            zero = torch.tensor(0.0, device=self.device)
            return {"geom_l2": zero, "geom_edge": zero, "geom_reg": zero}

        disp = vertices - self.base_vertices
        geom_l2 = disp.pow(2).mean()

        v0 = vertices[self._edges[:, 0]]
        v1 = vertices[self._edges[:, 1]]
        edge_len = (v0 - v1).norm(dim=1)
        geom_edge = (edge_len - self._base_edge_lengths).abs().mean()

        geom_reg = (self.cfg.geom_reg_l2_weight * geom_l2
                    + self.cfg.geom_reg_edge_weight * geom_edge)
        return {
            "geom_l2": geom_l2,
            "geom_edge": geom_edge,
            "geom_reg": geom_reg,
        }

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render_view(self, view: CameraView) -> torch.Tensor:
        """Full differentiable render → LDR (H, W, 3) in [0,1]."""
        ci = view.cam_idx
        R  = self._cam_R[ci]
        t  = self._cam_t[ci]
        K  = self._cam_K[ci]
        W, H = view.W, view.H

        # Update vignette radial grid if camera resolution changed
        if self.ppisp.learn_vignette:
            if self.ppisp.r2.shape != torch.Size([H, W]):
                ys = torch.linspace(-1, 1, H, device=self.device)
                xs = torch.linspace(-1, 1, W, device=self.device)
                gy, gx = torch.meshgrid(ys, xs, indexing="ij")
                self.ppisp.r2 = gx**2 + gy**2

        hdr = self.rasterizer.render(
            self.current_vertices(), self.faces, self.uvs,
            self.texture, R, t, K, W, H,
        )
        return self.ppisp(hdr, ci)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, view: CameraView) -> dict:
        gt = view.gt_image
        if gt is None:
            raise ValueError(f"GT image not loaded for cam {view.cam_idx}")
        gt = gt.to(self.device, non_blocking=True)

        train_tex = self.iter >= self.cfg.warmup_iters
        for p in self.texture.parameters():
            p.requires_grad_(train_tex)

        train_geom = self.learn_geometry and (self.iter >= self.cfg.geometry_warmup_iters)
        self.geometry_offsets.requires_grad_(train_geom)

        self.opt_tex.zero_grad(set_to_none=True)
        self.opt_ppisp.zero_grad(set_to_none=True)
        if self.opt_geom is not None:
            self.opt_geom.zero_grad(set_to_none=True)

        with self._autocast_ctx():
            pred   = self.render_view(view)
            losses = self.loss_fn(pred, gt, self.texture, self.ppisp)
            geom_losses = self._geometry_regularization(self.current_vertices())
            total = losses["total"] + geom_losses["geom_reg"]

        if self._grad_scaler is not None:
            self._grad_scaler.scale(total).backward()

            self._grad_scaler.unscale_(self.opt_ppisp)
            torch.nn.utils.clip_grad_norm_(self.ppisp.parameters(), 1.0)
            if train_tex:
                self._grad_scaler.unscale_(self.opt_tex)
                torch.nn.utils.clip_grad_norm_(self.texture.parameters(), 1.0)
            if train_geom and self.opt_geom is not None:
                self._grad_scaler.unscale_(self.opt_geom)
                torch.nn.utils.clip_grad_norm_([self.geometry_offsets], 1.0)

            if train_tex:
                self._grad_scaler.step(self.opt_tex)
            if train_geom and self.opt_geom is not None:
                self._grad_scaler.step(self.opt_geom)
            self._grad_scaler.step(self.opt_ppisp)
            self._grad_scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.ppisp.parameters(), 1.0)
            if train_tex:
                torch.nn.utils.clip_grad_norm_(self.texture.parameters(), 1.0)
                self.opt_tex.step()
            if train_geom and self.opt_geom is not None:
                torch.nn.utils.clip_grad_norm_([self.geometry_offsets], 1.0)
                self.opt_geom.step()
            self.opt_ppisp.step()

        out = {
            "total": total,
            **losses,
            **geom_losses,
        }
        return {k: v.item() if hasattr(v, "item") else v
                for k, v in out.items()}

    # ------------------------------------------------------------------
    # LR schedule
    # ------------------------------------------------------------------

    def _update_lr(self):
        cfg = self.cfg
        if self.iter >= cfg.lr_decay_start:
            t = min((self.iter - cfg.lr_decay_start) / max(cfg.lr_decay_iters, 1), 1.0)
            f = cfg.lr_decay_factor + (1 - cfg.lr_decay_factor) * 0.5 * (
                1 + math.cos(math.pi * t))
            for g in self.opt_tex.param_groups:   g["lr"] = cfg.lr_texture * f
            for g in self.opt_ppisp.param_groups: g["lr"] = cfg.lr_ppisp   * f
            if self.opt_geom is not None:
                for g in self.opt_geom.param_groups:
                    g["lr"] = cfg.lr_geometry * f

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------

    def train(self, progress_callback: Optional[Callable] = None):
        cfg   = self.cfg
        views = [v for v in self.scene.views if v.gt_image is not None]
        if not views:
            raise RuntimeError("No views with GT images. Call scene.load_gt_images() first.")

        t0 = time.time()
        print(f"\n[Trainer] ▶  Starting {cfg.num_iterations} iterations  "
              f"({cfg.warmup_iters} warmup) ...\n")

        running = {k: 0.0 for k in (
            "total", "photo", "tex_reg", "ppisp_reg", "geom_reg", "geom_l2", "geom_edge"
        )}
        t_iter  = time.time()

        for self.iter in range(cfg.num_iterations):
            view   = random.choice(views)
            losses = self.step(view)
            self._update_lr()
            self._poll_live_view_events()
            self._update_live_view(losses)

            for k in running:
                running[k] += losses.get(k, 0.0)
            self.loss_log.append({"iter": self.iter, **losses})

            if (self.iter + 1) % cfg.log_every == 0:
                it_per_s = cfg.log_every / max(time.time() - t_iter, 1e-6)
                eta_s    = (cfg.num_iterations - self.iter - 1) / max(it_per_s, 1e-6)
                if self.iter < cfg.warmup_iters:
                    mode = "warmup"
                elif self.learn_geometry and self.iter < cfg.geometry_warmup_iters:
                    mode = "joint-no-geom"
                else:
                    mode = "joint"
                avg      = {k: v / cfg.log_every for k, v in running.items()}
                geom_txt = (f"  geom={avg['geom_reg']:.5f}"
                            if self.learn_geometry else "")
                print(
                    f"  {self.iter+1:5d}/{cfg.num_iterations}  [{mode}]  "
                    f"loss={avg['total']:.4f}  photo={avg['photo']:.4f}  "
                    f"tex_reg={avg['tex_reg']:.5f}  "
                    f"{geom_txt} "
                    f"{it_per_s:.1f} it/s  ETA {eta_s/60:.1f}m"
                )
                running = {k: 0.0 for k in running}
                t_iter  = time.time()
                if progress_callback:
                    progress_callback(self.iter + 1, cfg.num_iterations, losses)

            if (self.iter + 1) % cfg.save_every == 0:
                self._save_checkpoint(self.iter + 1)

        print(f"\n[Trainer] ✓  Done in {(time.time()-t0)/60:.1f} min")
        self._close_live_view()
        self.ppisp.print_summary()

    # ------------------------------------------------------------------
    # Checkpoint / export
    # ------------------------------------------------------------------

    def _save_checkpoint(self, iteration: int):
        path = os.path.join(self.cfg.output_dir, f"checkpoint_{iteration:06d}.pt")
        torch.save({
            "iteration":  iteration,
            "texture":    self.texture.state_dict(),
            "ppisp":      self.ppisp.state_dict(),
            "opt_tex":    self.opt_tex.state_dict(),
            "opt_ppisp":  self.opt_ppisp.state_dict(),
            "vertex_offsets": self.geometry_offsets.detach().cpu() if self.learn_geometry else None,
            "geometry_offsets": self.geometry_offsets.detach().cpu() if self.learn_geometry else None,
            "weld_enabled": self._weld_enabled,
            "weld_index": self._weld_index.detach().cpu() if self._weld_index is not None else None,
            "opt_geom":   self.opt_geom.state_dict() if self.opt_geom is not None else None,
            "amp_enabled": self._amp_enabled,
            "amp_dtype": self.cfg.amp_dtype,
            "grad_scaler": self._grad_scaler.state_dict() if self._grad_scaler is not None else None,
            "loss_log":   self.loss_log,
        }, path)
        # Keep only the most recent checkpoint file to limit disk usage.
        self._prune_old_checkpoints(path)
        print(f"  [ckpt] → {path}")

    def _prune_old_checkpoints(self, keep_path: str):
        keep_abs = os.path.abspath(keep_path)
        out_dir = Path(self.cfg.output_dir)
        for ckpt_path in out_dir.glob("checkpoint_*.pt"):
            ckpt_abs = os.path.abspath(str(ckpt_path))
            if ckpt_abs == keep_abs:
                continue
            try:
                ckpt_path.unlink()
            except OSError as e:
                print(f"  [ckpt] WARNING: could not remove old checkpoint {ckpt_path}: {e}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.texture.load_state_dict(ckpt["texture"])
        self.ppisp.load_state_dict(ckpt["ppisp"])
        self.opt_tex.load_state_dict(ckpt["opt_tex"])
        self.opt_ppisp.load_state_dict(ckpt["opt_ppisp"])
        if self.learn_geometry:
            geom_blob = ckpt.get("geometry_offsets", ckpt.get("vertex_offsets"))
            if geom_blob is not None:
                geom_blob = geom_blob.to(self.device)
                with torch.no_grad():
                    if geom_blob.shape == self.geometry_offsets.shape:
                        self.geometry_offsets.copy_(geom_blob)
                    elif (self._weld_enabled and geom_blob.shape == self.base_vertices.shape):
                        # Backward compatibility: old checkpoints may store per-vertex offsets.
                        # Collapse them to welded-group offsets by averaging per group.
                        sums = torch.zeros_like(self.geometry_offsets)
                        counts = torch.zeros(self.geometry_offsets.shape[0], device=self.device, dtype=self.geometry_offsets.dtype)
                        sums.index_add_(0, self._weld_index, geom_blob)
                        counts.index_add_(0, self._weld_index, torch.ones(self.base_vertices.shape[0], device=self.device, dtype=self.geometry_offsets.dtype))
                        self.geometry_offsets.copy_(sums / counts.clamp(min=1.0).unsqueeze(1))
                    else:
                        print("[Trainer] WARNING: geometry offset shape mismatch in checkpoint; keeping current geometry offsets")
            if self.opt_geom is not None and ckpt.get("opt_geom") is not None:
                self.opt_geom.load_state_dict(ckpt["opt_geom"])
        if self._grad_scaler is not None and ckpt.get("grad_scaler") is not None:
            self._grad_scaler.load_state_dict(ckpt["grad_scaler"])
        self.iter     = ckpt["iteration"]
        self.loss_log = ckpt.get("loss_log", [])
        print(f"[Trainer] Loaded checkpoint iter {self.iter}: {path}")

    def _export_mesh_ply(self, path: str, texture_file: str = "optimized_texture.png"):
        """
        Export textured mesh as binary little-endian PLY.
        Uses per-vertex UV properties and a TextureFile header comment.
        """
        verts = self.current_vertices().detach().cpu().numpy().astype(np.float32)
        uvs = self.uvs.detach().cpu().numpy().astype(np.float32)
        faces = self.faces.detach().cpu().numpy().astype(np.int32)

        if uvs.shape[0] != verts.shape[0]:
            raise RuntimeError(
                f"UV/vertex mismatch while exporting PLY: {uvs.shape[0]} uvs vs {verts.shape[0]} verts"
            )

        # PLY consumers typically interpret UV V with bottom-origin semantics.
        # Internal UVs are image-space top-origin, so flip V on export.
        uvs_out = uvs.copy()
        uvs_out[:, 0] = np.clip(uvs_out[:, 0], 0.0, 1.0)
        uvs_out[:, 1] = np.clip(1.0 - uvs_out[:, 1], 0.0, 1.0)

        vertex_block = np.concatenate([verts, uvs_out], axis=1).astype(np.float32, copy=False)

        face_dtype = np.dtype([("n", np.uint8), ("idx", np.int32, (3,))])
        face_block = np.empty(faces.shape[0], dtype=face_dtype)
        face_block["n"] = 3
        face_block["idx"] = faces

        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"comment TextureFile {texture_file}\n"
            f"element vertex {verts.shape[0]}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float texture_u\n"
            "property float texture_v\n"
            f"element face {faces.shape[0]}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )

        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            vertex_block.tofile(f)
            face_block.tofile(f)

    def export_results(self):
        out = self.cfg.output_dir

        # Optimised texture
        tex_np = self.texture.as_image().cpu().numpy()
        Image.fromarray((tex_np * 255).astype(np.uint8)).save(
            os.path.join(out, "optimized_texture.png"))
        print(f"[Export] Texture  → {out}/optimized_texture.png")

        # PPISP params
        import json
        with open(os.path.join(out, "ppisp_params.json"), "w") as f:
            json.dump([self.ppisp.get_params_dict(i)
                       for i in range(len(self.scene))], f, indent=2)
        print(f"[Export] PPISP    → {out}/ppisp_params.json")

        # Optimized geometry (binary textured PLY)
        mesh_ply = os.path.join(out, "optimized_mesh.ply")
        self._export_mesh_ply(mesh_ply, texture_file="optimized_texture.png")
        print(f"[Export] Mesh     → {mesh_ply}")

        # Loss curve
        np.save(os.path.join(out, "loss_log.npy"),
                np.array([l["total"] for l in self.loss_log]))

        # Re-render all views
        render_dir = os.path.join(out, "renders")
        os.makedirs(render_dir, exist_ok=True)
        print("[Export] Re-rendering all views ...")
        with torch.no_grad():
            for view in self.scene.views:
                if view.gt_image is None:
                    continue
                pred = self.render_view(view).cpu().numpy()
                Image.fromarray((pred * 255).astype(np.uint8)).save(
                    os.path.join(render_dir, f"cam_{view.cam_idx:04d}.png"))
        print(f"[Export] Renders  → {render_dir}/")


# ---------------------------------------------------------------------------
# Quick-start helper
# ---------------------------------------------------------------------------

def train_scene(scene_root: str, output_dir: str = "outputs",
                num_iterations: int = 5_000, image_scale: float = 0.5,
                tex_res: int = 1024, device: str = None, **kwargs):
    from .dataset import ColmapScene
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    scene  = ColmapScene(scene_root=scene_root, image_scale=image_scale,
                         device=device)
    scene.load_gt_images()
    cfg = TrainConfig()
    cfg.num_iterations = num_iterations
    cfg.output_dir     = output_dir
    cfg.tex_H = cfg.tex_W = tex_res
    cfg.device = device
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    trainer = TexturePPISPTrainer(scene, cfg)
    trainer.train()
    trainer.export_results()
    return trainer