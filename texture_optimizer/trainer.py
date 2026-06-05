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

import torch
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

        # ---- Mesh (fixed, no grad) ----
        self.vertices = scene.mesh.vertices.to(self.device)
        self.faces    = scene.mesh.faces.to(self.device)
        self.uvs      = scene.mesh.uvs.to(self.device)

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

        self.iter     = 0
        self.loss_log: List[dict] = []

        n_tex   = sum(p.numel() for p in self.texture.parameters())
        n_ppisp = sum(p.numel() for p in self.ppisp.parameters())
        print(f"\n[Trainer] Device    : {self.device}")
        print(f"[Trainer] Backend   : {'nvdiffrast' if self.rasterizer.uses_nvdiffrast else 'software (SLOW)'}")
        print(f"[Trainer] Cameras   : {len(scene)}")
        print(f"[Trainer] Mesh      : {self.vertices.shape[0]:,} verts, {self.faces.shape[0]:,} faces")
        print(f"[Trainer] Texture   : {config.tex_H}×{config.tex_W}  ({n_tex/1e6:.1f}M params)")
        print(f"[Trainer] PPISP     : {n_ppisp} params  ({len(scene)} cameras)")
        print(f"[Trainer] Iterations: {config.num_iterations}")

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
            self.vertices, self.faces, self.uvs,
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

        self.opt_tex.zero_grad()
        self.opt_ppisp.zero_grad()

        pred   = self.render_view(view)
        losses = self.loss_fn(pred, gt, self.texture, self.ppisp)
        losses["total"].backward()

        torch.nn.utils.clip_grad_norm_(self.ppisp.parameters(), 1.0)
        if train_tex:
            torch.nn.utils.clip_grad_norm_(self.texture.parameters(), 1.0)
            self.opt_tex.step()
        self.opt_ppisp.step()

        return {k: v.item() if hasattr(v, "item") else v
                for k, v in losses.items()}

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

        running = {k: 0.0 for k in ("total", "photo", "tex_reg", "ppisp_reg")}
        t_iter  = time.time()

        for self.iter in range(cfg.num_iterations):
            view   = random.choice(views)
            losses = self.step(view)
            self._update_lr()

            for k in running:
                running[k] += losses.get(k, 0.0)
            self.loss_log.append({"iter": self.iter, **losses})

            if (self.iter + 1) % cfg.log_every == 0:
                it_per_s = cfg.log_every / max(time.time() - t_iter, 1e-6)
                eta_s    = (cfg.num_iterations - self.iter - 1) / max(it_per_s, 1e-6)
                mode     = "warmup" if self.iter < cfg.warmup_iters else "joint"
                avg      = {k: v / cfg.log_every for k, v in running.items()}
                print(
                    f"  {self.iter+1:5d}/{cfg.num_iterations}  [{mode}]  "
                    f"loss={avg['total']:.4f}  photo={avg['photo']:.4f}  "
                    f"tex_reg={avg['tex_reg']:.5f}  "
                    f"{it_per_s:.1f} it/s  ETA {eta_s/60:.1f}m"
                )
                running = {k: 0.0 for k in running}
                t_iter  = time.time()
                if progress_callback:
                    progress_callback(self.iter + 1, cfg.num_iterations, losses)

            if (self.iter + 1) % cfg.save_every == 0:
                self._save_checkpoint(self.iter + 1)

        print(f"\n[Trainer] ✓  Done in {(time.time()-t0)/60:.1f} min")
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
            "loss_log":   self.loss_log,
        }, path)
        print(f"  [ckpt] → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.texture.load_state_dict(ckpt["texture"])
        self.ppisp.load_state_dict(ckpt["ppisp"])
        self.opt_tex.load_state_dict(ckpt["opt_tex"])
        self.opt_ppisp.load_state_dict(ckpt["opt_ppisp"])
        self.iter     = ckpt["iteration"]
        self.loss_log = ckpt.get("loss_log", [])
        print(f"[Trainer] Loaded checkpoint iter {self.iter}: {path}")

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