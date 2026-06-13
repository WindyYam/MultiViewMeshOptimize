"""
Texture + Per-Camera PPISP Joint Optimiser

Training loop:
  - Random camera sampling each iteration (like 3DGS)
  - Warmup phase: PPISP only, texture frozen
  - Joint phase: texture + PPISP together
  - Cosine LR decay
  - Checkpoint every N iters, export at end
"""

import os, math, time, random, hashlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Callable
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .ppisp    import PPISPParams
from .renderer import TextureMap, Rasterizer
from .losses   import TotalLoss, compute_face_normals_and_double_area
from .dataset  import ColmapScene, CameraView, load_image
from .exporter import TextureExportMixin


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TrainConfig:
    num_iterations:   int   = 5_000
    warmup_iters:     int   = 500      # PPISP only, texture frozen
    tex_update_every: int   = 1        # update texture every N iterations after warmup
    geom_update_every:int   = 1        # update geometry every N iterations after geometry warmup

    lr_texture:       float = 1e-3
    lr_ppisp:         float = 5e-3
    lr_decay_start:   Optional[int] = None
    lr_decay_factor:  float = 0.1
    lr_decay_iters:   Optional[int] = None

    photo_weight:     float = 1.0
    ppisp_reg_weight: float = 1e-2
    l1_weight:        float = 0.8
    ssim_weight:      float = 0.2
    ssim_backend:     str   = "auto"   # auto, native, msssim

    tex_H:            int   = 1024
    tex_W:            int   = 1024
    progressive_texture: bool = False
    texture_dtype:    str   = "auto"   # auto, fp32, fp16, bf16
    tex_optimizer:    str   = "adam"   # adam, sgd
    geometry_dtype:   str   = "auto"   # auto, fp32, fp16, bf16
    geom_optimizer:   str   = "adam"   # adam, sgd

    log_every:        int   = 50
    save_every:       int   = 2000
    output_dir:       str   = "outputs"
    device:           str   = "cuda" if torch.cuda.is_available() else "cpu"
    seed:             int   = 42
    learn_vignette:   bool  = True
    ppisp_gamma:      float = 2.2
    learn_gamma:      bool  = False
    learn_geometry:   bool  = False
    lr_geometry:      float = 1e-4
    geometry_warmup_iters: int = 1000
    geom_normal_tv_weight: float = 5e-3
    geom_normal_tv_delta:  float = 1e-3
    geom_edge_uniform_weight: float = 0.0
    geom_edge_uniform_eps: float = 1e-8
    geom_normal_tv_feature_sigma: float = 0.05
    geom_normal_tv_use_base_feature_weights: bool = True
    max_vertex_offset:     Optional[float] = 0.03
    weld_geometry_vertices: bool = True
    weld_position_decimals: int = 6
    live_view:             bool  = False
    live_view_every:       int   = 50
    live_view_camera:      int   = 0
    live_view_max_size:    int   = 1200
    use_amp:               bool  = False
    amp_dtype:             str   = "fp16"   # fp16 or bf16
    amp_loss_fp32:         bool  = True
    amp_init_scale:        float = 1024.0
    amp_growth_interval:   int   = 2000
    use_tf32:              bool  = True
    image_cpu_cache_size:  int   = 8
    image_gpu_cache_size:  int   = 3
    image_prefetch_ahead:  int   = 4
    image_loader_workers:  int   = 2
    image_fs_cache:        bool  = True
    image_cache_dir:       Optional[str] = None
    tex_seam_pad_px:       int   = 12
    # If >0, alternate between texture and geometry updates every N iterations
    # during the joint phase to avoid them fighting each other.
    alter_every:           int   = 0


class AsyncImageCache:
    """
    Bounded CPU/GPU cache for training images with optional filesystem cache.
    CPU decode/load runs in a background thread pool.
    CUDA transfers run on a dedicated prefetch stream.
    """

    def __init__(
        self,
        views: List[CameraView],
        device: torch.device,
        cpu_cache_size: int,
        gpu_cache_size: int,
        prefetch_ahead: int,
        loader_workers: int,
        use_fs_cache: bool,
        fs_cache_dir: Optional[str],
    ):
        self.views = views
        self.device = device
        self.pin_memory = (device.type == "cuda")
        self.cpu_cache_size = max(1, int(cpu_cache_size))
        self.gpu_cache_size = max(1, int(gpu_cache_size))
        self.prefetch_ahead = max(1, int(prefetch_ahead))
        self.use_fs_cache = bool(use_fs_cache)

        self.cpu_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.cpu_futures = {}
        self.gpu_cache: OrderedDict[int, tuple] = OrderedDict()

        self.executor = ThreadPoolExecutor(max_workers=max(1, int(loader_workers)))
        self.prefetch_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

        self.fs_cache_dir = None
        if self.use_fs_cache:
            cache_root = fs_cache_dir or ".image_cache"
            self.fs_cache_dir = Path(cache_root)
            self.fs_cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path_for_view(self, view: CameraView) -> Optional[Path]:
        if self.fs_cache_dir is None:
            return None
        try:
            st = os.stat(view.image_path)
            stamp = f"{os.path.abspath(view.image_path)}|{view.W}|{view.H}|{st.st_mtime_ns}|{st.st_size}"
        except OSError:
            stamp = f"{os.path.abspath(view.image_path)}|{view.W}|{view.H}"
        key = hashlib.sha1(stamp.encode("utf-8")).hexdigest()
        return self.fs_cache_dir / f"{key}.npy"

    def _load_cpu_image(self, idx: int) -> torch.Tensor:
        view = self.views[idx]
        if view.gt_image is not None:
            img = view.gt_image.detach().cpu().float().contiguous()
            return img.pin_memory() if self.pin_memory and not img.is_pinned() else img

        cache_path = self._cache_path_for_view(view)
        if cache_path is not None and cache_path.exists():
            try:
                arr = np.load(str(cache_path), allow_pickle=False)
                if arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] == 3:
                    t = torch.from_numpy(arr.astype(np.float32) / 255.0).contiguous()
                    return t.pin_memory() if self.pin_memory else t
            except Exception:
                pass

        img = load_image(view.image_path, view.W, view.H).float().contiguous()
        if cache_path is not None:
            try:
                arr_u8 = (img.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
                np.save(str(cache_path), arr_u8, allow_pickle=False)
            except Exception:
                pass
        return img.pin_memory() if self.pin_memory else img

    def _touch_cpu(self, idx: int, tensor: torch.Tensor):
        self.cpu_cache[idx] = tensor
        self.cpu_cache.move_to_end(idx)
        while len(self.cpu_cache) > self.cpu_cache_size:
            self.cpu_cache.popitem(last=False)

    def _get_cpu(self, idx: int) -> torch.Tensor:
        t = self.cpu_cache.get(idx)
        if t is not None:
            self.cpu_cache.move_to_end(idx)
            return t

        fut = self.cpu_futures.get(idx)
        if fut is None:
            fut = self.executor.submit(self._load_cpu_image, idx)
            self.cpu_futures[idx] = fut

        t = fut.result()
        self.cpu_futures.pop(idx, None)
        self._touch_cpu(idx, t)
        return t

    def _start_gpu_transfer(self, idx: int, cpu_tensor: torch.Tensor):
        if self.device.type != "cuda":
            return
        if idx in self.gpu_cache:
            self.gpu_cache.move_to_end(idx)
            return
        assert self.prefetch_stream is not None
        with torch.cuda.stream(self.prefetch_stream):
            gpu_tensor = cpu_tensor.to(self.device, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.prefetch_stream)
        self.gpu_cache[idx] = (gpu_tensor, done)
        self.gpu_cache.move_to_end(idx)
        while len(self.gpu_cache) > self.gpu_cache_size:
            _, (old_tensor, old_event) = self.gpu_cache.popitem(last=False)
            try:
                old_event.synchronize()
            except Exception:
                pass
            del old_tensor

    def _promote_ready(self, indices: List[int]):
        if self.device.type != "cuda":
            return
        for idx in indices:
            if idx in self.gpu_cache:
                continue
            t = self.cpu_cache.get(idx)
            if t is None:
                fut = self.cpu_futures.get(idx)
                if fut is None or not fut.done():
                    continue
                try:
                    t = fut.result()
                    self.cpu_futures.pop(idx, None)
                    self._touch_cpu(idx, t)
                except Exception:
                    self.cpu_futures.pop(idx, None)
                    continue
            self._start_gpu_transfer(idx, t)

    def prefetch(self, indices: List[int]):
        seen = set()
        ordered = []
        for idx in indices:
            if idx in seen:
                continue
            seen.add(idx)
            ordered.append(idx)

        for idx in ordered:
            if idx in self.cpu_cache or idx in self.cpu_futures:
                continue
            self.cpu_futures[idx] = self.executor.submit(self._load_cpu_image, idx)
        self._promote_ready(ordered)

    def get(self, idx: int) -> torch.Tensor:
        if self.device.type != "cuda":
            return self._get_cpu(idx)

        self._promote_ready([idx])
        entry = self.gpu_cache.get(idx)
        if entry is None:
            cpu = self._get_cpu(idx)
            self._start_gpu_transfer(idx, cpu)
            entry = self.gpu_cache[idx]
        gpu_tensor, done = entry
        torch.cuda.current_stream(self.device).wait_event(done)
        self.gpu_cache.move_to_end(idx)
        return gpu_tensor

    def get_cpu_cached(self, idx: int) -> Optional[torch.Tensor]:
        return self.cpu_cache.get(idx)

    def get_cpu(self, idx: int) -> torch.Tensor:
        return self._get_cpu(idx)

    def close(self):
        self.executor.shutdown(wait=True)
        self.cpu_futures.clear()
        self.cpu_cache.clear()
        self.gpu_cache.clear()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TexturePPISPTrainer(TextureExportMixin):

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
        self._texture_dtype = self._resolve_texture_dtype(config.texture_dtype)

        # ---- Mesh ----
        self.base_vertices = scene.mesh.vertices.to(self.device)
        self.faces    = scene.mesh.faces.to(self.device)
        self.uvs      = scene.mesh.uvs.to(self.device)
        self._geometry_dtype = self._resolve_geometry_dtype(config.geometry_dtype)

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
        self._cast_geometry_parameter()
        # Backward-compat alias for places that still refer to vertex_offsets.
        self.vertex_offsets = self.geometry_offsets
        self._max_vertex_offset = config.max_vertex_offset
        self._edges = self._build_unique_edges(self.faces)
        self._face_adj_pairs, self._face_adj_edges = self._build_face_adjacency(
            self.faces, vertex_groups=self._weld_index
        )
        self._base_face_pair_d = None
        self._base_face_pair_weight = None
        self._weld_group_inv_counts = None
        if self._face_adj_pairs is not None and self._face_adj_pairs.shape[0] > 0:
            base_n, _ = compute_face_normals_and_double_area(self.base_vertices, self.faces)
            p = self._face_adj_pairs
            cos_base = (base_n[p[:, 0]] * base_n[p[:, 1]]).sum(dim=1).clamp(-1.0, 1.0)
            self._base_face_pair_d = (1.0 - cos_base).detach()
            if bool(config.geom_normal_tv_use_base_feature_weights):
                sigma = float(max(1e-6, config.geom_normal_tv_feature_sigma))
                self._base_face_pair_weight = torch.exp(-self._base_face_pair_d / sigma).detach()
        if self._weld_enabled and self._weld_index is not None:
            counts = torch.zeros(
                self._num_weld_groups,
                device=self.device,
                dtype=self.base_vertices.dtype,
            )
            ones = torch.ones(
                self.base_vertices.shape[0],
                device=self.device,
                dtype=self.base_vertices.dtype,
            )
            counts.index_add_(0, self._weld_index, ones)
            self._weld_group_inv_counts = (1.0 / counts.clamp(min=1.0)).detach()
        if self._edges is not None and self._edges.shape[0] > 0:
            v0 = self.base_vertices[self._edges[:, 0]]
            v1 = self.base_vertices[self._edges[:, 1]]
            self._base_edge_lengths = (v0 - v1).norm(dim=1).detach()
            deg = torch.zeros(self.base_vertices.shape[0], device=self.device, dtype=self.base_vertices.dtype)
            ones = torch.ones(self._edges.shape[0], device=self.device, dtype=self.base_vertices.dtype)
            deg.index_add_(0, self._edges[:, 0], ones)
            deg.index_add_(0, self._edges[:, 1], ones)
            self._vertex_degree = deg.clamp(min=1.0)
        else:
            self._base_edge_lengths = None
            self._vertex_degree = None

        # ---- Texture ----
        self._full_tex_H = int(config.tex_H)
        self._full_tex_W = int(config.tex_W)
        self._tex_stage_scales = (0.5, 1.0)
        self._tex_stage_iters = self._build_tex_stage_iters(config.num_iterations)
        self._tex_stage = self._texture_stage_for_iter(0)
        init_tex_h, init_tex_w = self._stage_texture_resolution(self._tex_stage)

        self.texture = TextureMap(
            init_tex_h, init_tex_w,
            init_image=scene.mesh.tex_image,
        ).to(self.device)
        self._cast_texture_parameter()

        # ---- PPISP ----
        v0 = scene.views[0]
        self.ppisp = PPISPParams(
            num_cameras=len(scene),
            image_width=v0.W, image_height=v0.H,
            init_gamma=config.ppisp_gamma,
            learn_gamma=config.learn_gamma,
            learn_vignette=config.learn_vignette,
        ).to(self.device)

        # ---- Rasterizer ----
        self.rasterizer = Rasterizer(self.device)

        # ---- Pre-cache camera matrices on GPU ----
        self._cam_R = [v.R.to(self.device) for v in scene.views]
        self._cam_t = [v.t.to(self.device) for v in scene.views]
        self._cam_K = [v.K.to(self.device) for v in scene.views]

        # ---- Async GT image cache / prefetch ----
        img_cache_dir = config.image_cache_dir
        if not img_cache_dir:
            img_cache_dir = os.path.join(config.output_dir, "image_cache")
        self._image_cache = AsyncImageCache(
            views=scene.views,
            device=self.device,
            cpu_cache_size=config.image_cpu_cache_size,
            gpu_cache_size=config.image_gpu_cache_size,
            prefetch_ahead=config.image_prefetch_ahead,
            loader_workers=config.image_loader_workers,
            use_fs_cache=config.image_fs_cache,
            fs_cache_dir=img_cache_dir,
        )

        # ---- Loss ----
        self.loss_fn = TotalLoss(
            photo_weight=config.photo_weight,
            ppisp_reg_weight=config.ppisp_reg_weight,
            geom_normal_tv_weight=config.geom_normal_tv_weight,
            geom_normal_tv_delta=config.geom_normal_tv_delta,
            geom_edge_uniform_weight=config.geom_edge_uniform_weight,
            geom_edge_uniform_eps=config.geom_edge_uniform_eps,
            l1_weight=config.l1_weight,
            ssim_weight=config.ssim_weight,
            ssim_backend=config.ssim_backend,
        )

        # ---- Optimizers ----
        self.opt_tex   = self._create_tex_optimizer()
        self.opt_ppisp = self._create_adam(self.ppisp.parameters(),
                           lr=config.lr_ppisp)
        self.opt_geom  = (self._create_geom_optimizer()
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
        self._lr_decay_start_eff, self._lr_decay_iters_eff = self._resolve_lr_decay_schedule()

        n_tex = sum(p.numel() for p in self.texture.parameters())
        n_ppisp = sum(p.numel() for p in self.ppisp.parameters())
        n_geom = int(self.geometry_offsets.numel()) if self.learn_geometry else 0
        print(f"\n[Trainer] Device    : {self.device}")
        print("[Trainer] Backend   : nvdiffrast")
        print(f"[Trainer] Cameras   : {len(scene)}")
        print(f"[Trainer] Mesh      : {self.base_vertices.shape[0]:,} verts, {self.faces.shape[0]:,} faces")
        print(f"[Trainer] Texture   : {init_tex_h}×{init_tex_w} -> {self._full_tex_H}×{self._full_tex_W}  ({n_tex/1e6:.1f}M params)")
        tex_mem = self._estimate_texture_memory_bytes(n_tex)
        print(
            f"[Trainer] TexOpt    : {str(config.tex_optimizer).lower()}  dtype={str(config.texture_dtype).lower()}"
            f" (active={str(self._texture_dtype).split('.')[-1]})"
        )
        print(f"[Trainer] TexMem≈   : {tex_mem/1024**3:.2f} GiB (param+grad+opt state)")
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
            geom_mem = self._estimate_geometry_memory_bytes(n_geom)
            print(
                f"[Trainer] GeomOpt   : {str(config.geom_optimizer).lower()}  dtype={str(config.geometry_dtype).lower()}"
                f" (active={str(self._geometry_dtype).split('.')[-1]})"
            )
            print(f"[Trainer] GeomMem≈  : {geom_mem/1024**2:.2f} MiB (param+grad+opt state)")
        print(
            f"[Trainer] AMP       : {'on' if self._amp_enabled else 'off'}"
            f"{f' ({config.amp_dtype})' if self._amp_enabled else ''}"
        )
        if self._amp_enabled:
            print(
                f"[Trainer] AMP opts  : loss_fp32={'on' if config.amp_loss_fp32 else 'off'}, "
                f"init_scale={config.amp_init_scale:g}, growth_int={config.amp_growth_interval}"
            )
        print(f"[Trainer] TF32      : {'on' if (self.device.type == 'cuda' and config.use_tf32) else 'off'}")
        print(f"[Trainer] PhotoLoss : L1={config.l1_weight:.2f}, SSIM={config.ssim_weight:.2f} ({config.ssim_backend})")
        print(
            f"[Trainer] LRDecay   : start={self._lr_decay_start_eff}, "
            f"iters={self._lr_decay_iters_eff}, factor={float(config.lr_decay_factor):.4f}"
        )
        if config.progressive_texture:
            print(
                "[Trainer] TexProg   : on"
            )
        else:
            print("[Trainer] TexProg   : off")
        print(
            "[Trainer] UpdateFreq : "
            f"tex every {max(1, int(config.tex_update_every))} it, "
            f"geom every {max(1, int(config.geom_update_every))} it"
            + (f", alter every {int(config.alter_every)} it" if int(getattr(config, 'alter_every', 0)) > 0 else "")
        )
        print(
            f"[Trainer] ImgCache   : cpu={int(config.image_cpu_cache_size)} "
            f"gpu={int(config.image_gpu_cache_size)} prefetch={int(config.image_prefetch_ahead)} "
            f"workers={int(config.image_loader_workers)} fs={'on' if config.image_fs_cache else 'off'}"
        )
        if self.learn_geometry:
            print(
                f"[Trainer] GeomPrior  : normal_tv={float(config.geom_normal_tv_weight):.4g}, "
                f"edge_uniform={float(config.geom_edge_uniform_weight):.4g}"
            )
            n_adj = int(self._face_adj_pairs.shape[0]) if self._face_adj_pairs is not None else 0
            print(f"[Trainer] FaceAdj    : {n_adj:,} adjacent face pairs")
        print(f"[Trainer] Iterations: {config.num_iterations}")

    def _create_tex_optimizer(self):
        mode = str(self.cfg.tex_optimizer).lower()
        if mode == "sgd":
            return optim.SGD(self.texture.parameters(), lr=self.cfg.lr_texture, momentum=0.0)
        return self._create_adam(
            self.texture.parameters(),
            lr=self.cfg.lr_texture,
            betas=(0.9, 0.99),
            eps=1e-15,
        )

    def _create_geom_optimizer(self):
        mode = str(self.cfg.geom_optimizer).lower()
        if mode == "sgd":
            return optim.SGD([self.geometry_offsets], lr=self.cfg.lr_geometry, momentum=0.0)
        return self._create_adam([self.geometry_offsets], lr=self.cfg.lr_geometry)

    def _resolve_texture_dtype(self, mode: str):
        mode_l = str(mode).lower()
        if mode_l == "auto":
            if self.device.type == "cuda" and self._amp_enabled and self._amp_dtype is not None:
                return self._amp_dtype
            return torch.float32
        if mode_l == "fp16":
            return torch.float16
        if mode_l == "bf16":
            return torch.bfloat16
        return torch.float32

    def _resolve_geometry_dtype(self, mode: str):
        mode_l = str(mode).lower()
        if mode_l == "auto":
            if self.device.type == "cuda" and self._amp_enabled and self._amp_dtype is not None:
                return self._amp_dtype
            return self.base_vertices.dtype
        if mode_l == "fp16":
            return torch.float16
        if mode_l == "bf16":
            return torch.bfloat16
        return torch.float32

    def _cast_texture_parameter(self):
        target_dtype = self._texture_dtype
        if self.device.type != "cuda" and target_dtype != torch.float32:
            target_dtype = torch.float32
        if self.texture.tex.dtype == target_dtype:
            return
        with torch.no_grad():
            casted = self.texture.tex.detach().to(dtype=target_dtype)
        self.texture.tex = nn.Parameter(casted)

    def _cast_geometry_parameter(self):
        target_dtype = self._geometry_dtype
        if self.device.type != "cuda" and target_dtype != torch.float32:
            target_dtype = torch.float32
        if self.geometry_offsets.dtype == target_dtype:
            return
        with torch.no_grad():
            casted = self.geometry_offsets.detach().to(dtype=target_dtype)
        self.geometry_offsets = nn.Parameter(casted, requires_grad=self.learn_geometry)
        self.vertex_offsets = self.geometry_offsets

    def _estimate_texture_memory_bytes(self, n_tex: int) -> float:
        bpp = torch.finfo(self._texture_dtype).bits // 8
        mode = str(self.cfg.tex_optimizer).lower()
        # param + grad + optimizer state tensors
        state_mult = 2.0 if mode == "adam" else 0.0
        return float(n_tex) * float(bpp) * (2.0 + state_mult)

    def _estimate_geometry_memory_bytes(self, n_geom: int) -> float:
        bpp = torch.finfo(self.geometry_offsets.dtype).bits // 8
        mode = str(self.cfg.geom_optimizer).lower()
        state_mult = 2.0 if mode == "adam" else 0.0
        return float(n_geom) * float(bpp) * (2.0 + state_mult)

    def _create_adam(self, params, lr: float, betas=(0.9, 0.999), eps=1e-8):
        kwargs = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
        }
        if self.device.type == "cuda":
            try:
                return optim.Adam(params, fused=True, **kwargs)
            except Exception:
                try:
                    return optim.Adam(params, foreach=True, **kwargs)
                except Exception:
                    return optim.Adam(params, **kwargs)
        return optim.Adam(params, **kwargs)

    def _build_tex_stage_iters(self, total_iters: int):
        if total_iters <= 1:
            return 1
        return max(1, total_iters // 3)

    def _resolve_lr_decay_schedule(self):
        total = max(1, int(self.cfg.num_iterations))

        if self.cfg.lr_decay_start is None:
            start = int(round(0.7 * total))
        else:
            start = int(self.cfg.lr_decay_start)
        start = max(0, min(start, max(0, total - 1)))

        if self.cfg.lr_decay_iters is None:
            decay_iters = total - start
        else:
            decay_iters = int(self.cfg.lr_decay_iters)
        decay_iters = max(1, decay_iters)
        return start, decay_iters

    def _stage_texture_resolution(self, stage: int):
        scale = self._tex_stage_scales[max(0, min(stage, len(self._tex_stage_scales) - 1))]
        h = max(1, int(round(self._full_tex_H * scale)))
        w = max(1, int(round(self._full_tex_W * scale)))
        return h, w

    def _texture_stage_for_iter(self, iteration: int) -> int:
        if not self.cfg.progressive_texture:
            return len(self._tex_stage_scales) - 1
        if iteration < self._tex_stage_iters:
            return 0
        return 1

    def _build_training_schedule(self, num_views: int, num_iters: int) -> List[int]:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(self.cfg.seed))
        order = []
        while len(order) < num_iters:
            order.extend(torch.randperm(num_views, generator=gen).tolist())
        return order[:num_iters]

    def _set_texture_stage(self, stage: int):
        target_h, target_w = self._stage_texture_resolution(stage)
        cur_h = int(self.texture.tex.shape[2])
        cur_w = int(self.texture.tex.shape[3])
        if cur_h == target_h and cur_w == target_w:
            self._tex_stage = stage
            return

        with torch.no_grad():
            tex_now = self.texture.tex.detach()
            tex_up = F.interpolate(
                tex_now,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=self._texture_dtype)
        self.texture.tex = nn.Parameter(tex_up)
        self.opt_tex = self._create_tex_optimizer()
        self._tex_stage = stage
        print(f"[Trainer] Texture upscale -> {target_h}x{target_w} (stage {stage+1}/2)")

    def _maybe_update_texture_stage(self):
        target_stage = self._texture_stage_for_iter(self.iter)
        if target_stage != self._tex_stage:
            self._set_texture_stage(target_stage)

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
            self._grad_scaler = torch.cuda.amp.GradScaler(
                enabled=True,
                init_scale=float(max(1.0, cfg.amp_init_scale)),
                growth_interval=int(max(1, cfg.amp_growth_interval)),
                backoff_factor=0.5,
                growth_factor=2.0,
            )

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
            pred = self.render_view(view).detach().clamp(0, 1).float().cpu().numpy()
        gt = view.gt_image
        if gt is None and self._image_cache is not None:
            # Keep side-by-side preview behavior unchanged: fetch GT on demand.
            gt = self._image_cache.get_cpu(self._live_view_cam)
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
        p = self.ppisp.get_params_dict(self._live_view_cam)
        ppisp_line1 = (
            f"exp={p['exposure']:.3f} gamma={p['gamma']:.2f} "
            f"B={p['brightness']:+.3f} C={p['contrast']:.3f}"
        )
        ppisp_line2 = f"wb=({p['wb_r']:.3f},{p['wb_g']:.3f},{p['wb_b']:.3f})"
        self._cv2.putText(
            bgr, ppisp_line1, (12, 56), self._cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (240, 220, 60), 2, self._cv2.LINE_AA
        )
        self._cv2.putText(
            bgr, ppisp_line2, (12, 82), self._cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (240, 220, 60), 2, self._cv2.LINE_AA
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

    def _build_face_adjacency(self, faces: torch.Tensor, vertex_groups: Optional[torch.Tensor] = None):
        """
        Build face adjacency through shared edges.
        If vertex_groups is provided, adjacency is built on group ids rather than
        raw vertex ids, which makes it robust to UV seam vertex duplication.
        Returns:
          - face index pairs (M, 2)
          - shared edge endpoint ids (M, 2): vertex ids or group ids
        """
        if faces.numel() == 0:
            return None, None

        faces_np = faces.detach().cpu().numpy().astype(np.int64, copy=False)
        groups_np = None
        if vertex_groups is not None and vertex_groups.numel() >= faces.max().item() + 1:
            groups_np = vertex_groups.detach().cpu().numpy().astype(np.int64, copy=False)
        edge_to_face = {}
        pair_face = []
        pair_edge = []

        for fi, tri in enumerate(faces_np):
            i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
            if groups_np is None:
                gi0, gi1, gi2 = i0, i1, i2
            else:
                gi0, gi1, gi2 = int(groups_np[i0]), int(groups_np[i1]), int(groups_np[i2])

            for a, b in ((gi0, gi1), (gi1, gi2), (gi2, gi0)):
                if a == b:
                    continue
                e = (a, b) if a < b else (b, a)
                prev = edge_to_face.get(e)
                if prev is None:
                    edge_to_face[e] = fi
                else:
                    pair_face.append((prev, fi))
                    pair_edge.append(e)

        if not pair_face:
            return None, None

        pairs = torch.as_tensor(pair_face, dtype=torch.long, device=self.device)
        edges = torch.as_tensor(pair_edge, dtype=torch.long, device=self.device)
        return pairs, edges

    def current_vertices(self) -> torch.Tensor:
        if not self.learn_geometry:
            return self.base_vertices
        offsets_full = self._expanded_geometry_offsets().to(dtype=self.base_vertices.dtype)
        if self._max_vertex_offset is None:
            offsets = offsets_full
        else:
            offsets = torch.tanh(offsets_full) * float(self._max_vertex_offset)
        return self.base_vertices + offsets

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render_view(self, view: CameraView, return_mask: bool = False):
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

        hdr, mask = self.rasterizer.render_with_mask(
            self.current_vertices(), self.faces, self.uvs,
            self.texture, R, t, K, W, H,
        )
        ldr = self.ppisp(hdr, ci)
        pred = ldr * mask
        if return_mask:
            return pred, (mask[..., 0] > 0.5)
        return pred

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, view: CameraView, gt_image: Optional[torch.Tensor] = None) -> dict:
        gt = gt_image if gt_image is not None else view.gt_image
        if gt is None:
            raise ValueError(f"GT image not loaded for cam {view.cam_idx}")
        if gt.device != self.device:
            gt = gt.to(self.device, non_blocking=True)

        tex_every = max(1, int(self.cfg.tex_update_every))
        train_tex = (
            self.iter >= self.cfg.warmup_iters
            and ((self.iter - self.cfg.warmup_iters) % tex_every == 0)
        )
        for p in self.texture.parameters():
            p.requires_grad_(train_tex)

        geom_every = max(1, int(self.cfg.geom_update_every))
        train_geom = (
            self.learn_geometry
            and (self.iter >= self.cfg.geometry_warmup_iters)
            and ((self.iter - self.cfg.geometry_warmup_iters) % geom_every == 0)
        )
        # Optionally alternate between texture and geometry updates in blocks
        # of `alter_every` iterations once both warmups have completed. When
        # enabled, only one of texture/geometry is trained in each block.
        alter_every = int(getattr(self.cfg, "alter_every", 0) or 0)
        if alter_every > 0 and self.learn_geometry:
            start = max(int(self.cfg.warmup_iters), int(self.cfg.geometry_warmup_iters))
            if self.iter >= start:
                block = ((self.iter - start) // alter_every) & 1
                if block == 0:
                    # texture block: disable geometry training this iteration
                    train_geom = False
                else:
                    # geometry block: disable texture training this iteration
                    train_tex = False
        self.geometry_offsets.requires_grad_(train_geom)

        self.opt_tex.zero_grad(set_to_none=True)
        self.opt_ppisp.zero_grad(set_to_none=True)
        if self.opt_geom is not None:
            self.opt_geom.zero_grad(set_to_none=True)

        with self._autocast_ctx():
            pred, mesh_mask = self.render_view(view, return_mask=True)

        # Keep render in AMP, but evaluate objective in fp32 for stability.
        if self._amp_enabled and self.cfg.amp_loss_fp32:
            pred_loss = pred.float()
            gt_loss = gt.float()
        else:
            pred_loss = pred
            gt_loss = gt

        vertices_now = self.current_vertices() if train_geom else None
        losses = self.loss_fn(
            pred_loss,
            gt_loss,
            self.ppisp,
            mask=mesh_mask,
            learn_geometry=self.learn_geometry,
            train_geometry=train_geom,
            vertices=vertices_now,
            faces=self.faces if train_geom else None,
            face_adj_pairs=self._face_adj_pairs,
            face_adj_edges=self._face_adj_edges,
            base_face_pair_weight=self._base_face_pair_weight,
            weld_index=self._weld_index,
            num_weld_groups=self._num_weld_groups,
            weld_group_inv_counts=self._weld_group_inv_counts,
        )
        total = losses["total"]

        if not torch.isfinite(total):
            self.opt_tex.zero_grad(set_to_none=True)
            self.opt_ppisp.zero_grad(set_to_none=True)
            if self.opt_geom is not None:
                self.opt_geom.zero_grad(set_to_none=True)
            if self._grad_scaler is not None:
                cur_scale = float(self._grad_scaler.get_scale())
                self._grad_scaler.update(new_scale=max(cur_scale * 0.5, 1.0))
            return {
                "total": float("nan"),
                "photo": losses["photo"].detach().item(),
                "ppisp_reg": losses["ppisp_reg"].detach().item(),
                "geom_normal_tv": losses["geom_normal_tv"].detach().item(),
                "geom_edge_uniform": losses["geom_edge_uniform"].detach().item(),
            }

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
            **losses,
        }
        return {k: v.item() if hasattr(v, "item") else v
                for k, v in out.items()}

    # ------------------------------------------------------------------
    # LR schedule
    # ------------------------------------------------------------------

    def _update_lr(self):
        cfg = self.cfg
        if self.iter >= self._lr_decay_start_eff:
            t = min((self.iter - self._lr_decay_start_eff) / self._lr_decay_iters_eff, 1.0)
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
        views = list(self.scene.views)
        if not views:
            raise RuntimeError("No camera views found in scene.")

        schedule = self._build_training_schedule(len(views), cfg.num_iterations)
        warm_n = min(len(schedule), max(2, int(cfg.image_prefetch_ahead) + 1))
        self._image_cache.prefetch([schedule[i] for i in range(warm_n)])

        t0 = time.time()
        print(f"\n[Trainer] ▶  Starting {cfg.num_iterations} iterations  "
              f"({cfg.warmup_iters} warmup) ...\n")

        running = {k: 0.0 for k in (
            "total", "photo", "ppisp_reg", "geom_normal_tv", "geom_edge_uniform"
        )}
        t_iter  = time.time()
        try:
            for self.iter in range(cfg.num_iterations):
                self._maybe_update_texture_stage()
                vidx = schedule[self.iter]
                view = views[vidx]

                next_i0 = self.iter + 1
                next_i1 = min(cfg.num_iterations, next_i0 + int(cfg.image_prefetch_ahead))
                if next_i0 < next_i1:
                    self._image_cache.prefetch([schedule[j] for j in range(next_i0, next_i1)])

                gt = self._image_cache.get(vidx)
                losses = self.step(view, gt_image=gt)
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
                    w_photo = cfg.photo_weight * avg["photo"]
                    w_ppisp_reg = cfg.ppisp_reg_weight * avg["ppisp_reg"]
                    w_normal_tv = (float(cfg.geom_normal_tv_weight) * avg["geom_normal_tv"]
                                   if self.learn_geometry and self.iter >= cfg.geometry_warmup_iters
                                   else 0.0)
                    w_edge_uniform = (float(cfg.geom_edge_uniform_weight) * avg["geom_edge_uniform"]
                                      if self.learn_geometry and self.iter >= cfg.geometry_warmup_iters
                                      else 0.0)
                    w_total = w_photo + w_ppisp_reg + w_normal_tv + w_edge_uniform
                    geom_txt = (f"w_nTV={w_normal_tv:.3e} w_eUni={w_edge_uniform:.3e}"
                                if self.learn_geometry else "")
                    print(
                        f"  {self.iter+1:5d}/{cfg.num_iterations}  [{mode}]  "
                        f"w_loss={w_total:.4f}  w_photo={w_photo:.4f}  "
                        f"w_ppisp_reg={w_ppisp_reg:.5f}  "
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
            self.ppisp.print_summary()
        finally:
            self._image_cache.close()
            self._close_live_view()

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
            "tex_stage":  int(self._tex_stage),
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

        tex_blob = ckpt["texture"].get("tex", None)
        if tex_blob is not None:
            tex_h = int(tex_blob.shape[2])
            tex_w = int(tex_blob.shape[3])
            cur_h = int(self.texture.tex.shape[2])
            cur_w = int(self.texture.tex.shape[3])
            if (tex_h, tex_w) != (cur_h, cur_w):
                self._set_texture_stage(2)
                with torch.no_grad():
                    resized = F.interpolate(
                        self.texture.tex.detach(),
                        size=(tex_h, tex_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                self.texture.tex = nn.Parameter(resized)
                self.opt_tex = self._create_tex_optimizer()

        self.texture.load_state_dict(ckpt["texture"])
        self.ppisp.load_state_dict(ckpt["ppisp"])
        self.opt_tex.load_state_dict(ckpt["opt_tex"])
        self.opt_ppisp.load_state_dict(ckpt["opt_ppisp"])
        self._tex_stage = int(ckpt.get("tex_stage", self._texture_stage_for_iter(int(ckpt["iteration"]))))
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