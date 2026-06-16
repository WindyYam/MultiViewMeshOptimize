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
from typing import Optional, List, Callable, Tuple
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
    tex_tile_update:       bool  = False
    tex_tile_size:         int   = 1024
    tex_tile_stride:       int   = 0
    tex_tile_random:       bool  = True
    # If >0, alternate between texture and geometry updates every N iterations
    # during the joint phase to avoid them fighting each other.
    alter_every:           int   = 0
    # Adaptive topology update: split high-error faces, collapse low-error faces.
    topology_adapt_every:  int   = 0
    topology_error_beta:   float = 0.9
    topology_split_quantile: float = 0.9
    topology_merge_quantile: float = 0.2
    topology_max_splits:   int   = 128
    topology_max_merges:   int   = 128
    topology_start_iter:   Optional[int] = None
    topology_min_faces:    int   = 128
    topology_max_faces:    int   = 0


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
        self._face_error_ema = torch.zeros(
            int(self.faces.shape[0]),
            device=self.device,
            dtype=self.base_vertices.dtype,
        )
        self._topology_events = 0

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
        self._tile_tex_enabled = bool(getattr(config, "tex_tile_update", False))
        self._tile_tex_size = max(16, int(getattr(config, "tex_tile_size", 1024)))
        stride_cfg = int(getattr(config, "tex_tile_stride", 0))
        self._tile_tex_stride = self._tile_tex_size if stride_cfg <= 0 else stride_cfg
        self._tile_tex_random = bool(getattr(config, "tex_tile_random", True))
        self._texture_tiles: List[Tuple[int, int, int, int]] = []
        self._texture_tile_cursor = 0
        self._active_texture_tile_idx: Optional[int] = None
        self._active_texture_tile: Optional[Tuple[int, int, int, int]] = None
        self._texture_tile_param: Optional[nn.Parameter] = None
        self._opt_tex_tile = None
        if self._tile_tex_enabled:
            self._init_texture_tiling()

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
        self.loss_log_interval: List[dict] = []

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
        tex_mem = self._memory_breakdown(self.texture, self.opt_tex)
        print(
            f"[Trainer] TexOpt    : {str(config.tex_optimizer).lower()}  dtype={str(config.texture_dtype).lower()}"
            f" (active={str(self._texture_dtype).split('.')[-1]})"
        )
        print(
            "[Trainer] TexMemNow : "
            f"total={tex_mem['total_bytes']/1024**2:.2f} MiB "
            f"(param={tex_mem['param_bytes']/1024**2:.2f}, "
            f"grad={tex_mem['grad_bytes']/1024**2:.2f}, "
            f"opt={tex_mem['opt_state_bytes']/1024**2:.2f})"
        )
        if tex_mem["estimated_full_total_bytes"] > tex_mem["total_bytes"]:
            print(
                "[Trainer] TexMemEst : "
                f"{tex_mem['estimated_full_total_bytes']/1024**2:.2f} MiB "
                "(after first backward/optimizer step)"
            )
        print(f"[Trainer] PPISP     : {n_ppisp} params  ({len(scene)} cameras)")
        ppisp_mem = self._memory_breakdown(self.ppisp, self.opt_ppisp)
        print(
            "[Trainer] PPISPMem  : "
            f"total={ppisp_mem['total_bytes']/1024**2:.2f} MiB "
            f"(param={ppisp_mem['param_bytes']/1024**2:.2f}, "
            f"grad={ppisp_mem['grad_bytes']/1024**2:.2f}, "
            f"opt={ppisp_mem['opt_state_bytes']/1024**2:.2f})"
        )
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
                f"[Trainer] GeomOpt   : {str(config.geom_optimizer).lower()}  dtype={str(config.geometry_dtype).lower()}"
                f" (active={str(self._geometry_dtype).split('.')[-1]})"
            )
            geom_mem = self._memory_breakdown(self.texture, self.opt_geom, only_params=[self.geometry_offsets])
            print(
                "[Trainer] GeomMemNow: "
                f"total={geom_mem['total_bytes']/1024**2:.2f} MiB "
                f"(param={geom_mem['param_bytes']/1024**2:.2f}, "
                f"grad={geom_mem['grad_bytes']/1024**2:.2f}, "
                f"opt={geom_mem['opt_state_bytes']/1024**2:.2f})"
            )
            if geom_mem["estimated_full_total_bytes"] > geom_mem["total_bytes"]:
                print(
                    "[Trainer] GeomMemEst: "
                    f"{geom_mem['estimated_full_total_bytes']/1024**2:.2f} MiB "
                    "(after first backward/optimizer step)"
                )
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
        if self._tile_tex_enabled:
            print(
                "[Trainer] TexTile   : "
                f"on  size={self._tile_tex_size} stride={self._tile_tex_stride} "
                f"mode={'random' if self._tile_tex_random else 'sequential'} "
                f"tiles={len(self._texture_tiles)}"
            )
        else:
            print("[Trainer] TexTile   : off")
        print(
            "[Trainer] UpdateFreq : "
            f"tex every {max(1, int(config.tex_update_every))} it, "
            f"geom every {max(1, int(config.geom_update_every))} it"
            + (f", alter every {int(config.alter_every)} it" if int(getattr(config, 'alter_every', 0)) > 0 else "")
        )
        if int(getattr(config, "topology_adapt_every", 0)) > 0:
            print(
                "[Trainer] Topology   : "
                f"every {int(config.topology_adapt_every)} it, "
                f"split q>={float(config.topology_split_quantile):.2f}, "
                f"merge q<={float(config.topology_merge_quantile):.2f}, "
                f"max split={int(config.topology_max_splits)}, "
                f"max merge={int(config.topology_max_merges)}"
            )
        else:
            print("[Trainer] Topology   : off")
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

    def _create_tex_optimizer(self, params=None):
        if params is None:
            params = self.texture.parameters()
        mode = str(self.cfg.tex_optimizer).lower()
        if mode == "sgd":
            return optim.SGD(params, lr=self.cfg.lr_texture, momentum=0.9, nesterov=True)
        return self._create_adam(
            params,
            lr=self.cfg.lr_texture,
            betas=(0.9, 0.99),
            eps=1e-15,
        )

    def _create_geom_optimizer(self):
        mode = str(self.cfg.geom_optimizer).lower()
        if mode == "sgd":
            return optim.SGD([self.geometry_offsets], lr=self.cfg.lr_geometry, momentum=0.9, nesterov=True)
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

    @staticmethod
    def _tensor_nbytes(t: torch.Tensor) -> int:
        return int(t.numel()) * int(t.element_size())

    @classmethod
    def _sum_tensor_tree_bytes(cls, obj) -> int:
        if torch.is_tensor(obj):
            return cls._tensor_nbytes(obj)
        if isinstance(obj, dict):
            return sum(cls._sum_tensor_tree_bytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple, set)):
            return sum(cls._sum_tensor_tree_bytes(v) for v in obj)
        return 0

    def _optimizer_state_slots_hint(self, optimizer) -> int:
        if optimizer is None:
            return 0
        name = optimizer.__class__.__name__.lower()
        if "adam" in name:
            return 2  # exp_avg, exp_avg_sq
        if "sgd" in name:
            has_momentum = any(float(g.get("momentum", 0.0)) > 0.0 for g in optimizer.param_groups)
            return 1 if has_momentum else 0
        return 0

    def _memory_breakdown(self, module: nn.Module, optimizer, only_params: Optional[List[torch.Tensor]] = None) -> dict:
        if only_params is None:
            params = list(module.parameters())
        else:
            params = list(only_params)

        param_bytes = sum(self._tensor_nbytes(p.data) for p in params)
        grad_bytes = sum(self._tensor_nbytes(p.grad) for p in params if p.grad is not None)

        opt_state_bytes = 0
        if optimizer is not None:
            for state in optimizer.state.values():
                opt_state_bytes += self._sum_tensor_tree_bytes(state)

        missing_grad_bytes = sum(
            self._tensor_nbytes(p.data)
            for p in params
            if p.requires_grad and p.grad is None
        )
        slots_hint = self._optimizer_state_slots_hint(optimizer)
        missing_opt_state_bytes = 0
        if optimizer is not None and slots_hint > 0:
            for p in params:
                if p not in optimizer.state:
                    missing_opt_state_bytes += slots_hint * self._tensor_nbytes(p.data)

        total_bytes = int(param_bytes + grad_bytes + opt_state_bytes)
        estimated_full_total_bytes = int(total_bytes + missing_grad_bytes + missing_opt_state_bytes)
        return {
            "param_bytes": int(param_bytes),
            "grad_bytes": int(grad_bytes),
            "opt_state_bytes": int(opt_state_bytes),
            "total_bytes": total_bytes,
            "estimated_full_total_bytes": estimated_full_total_bytes,
        }

    def _create_adam(self, params, lr: float, betas=(0.9, 0.999), eps=1e-8):
        kwargs = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
        }
        if self.device.type == "cuda":
            try:
                return optim.AdamW(params, fused=True, **kwargs)
            except Exception:
                try:
                    return optim.AdamW(params, foreach=True, **kwargs)
                except Exception:
                    return optim.AdamW(params, **kwargs)
        return optim.AdamW(params, **kwargs)

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

    def _build_texture_tiles(self, tex_h: int, tex_w: int) -> List[Tuple[int, int, int, int]]:
        tile = max(16, int(self._tile_tex_size))
        stride = max(1, int(self._tile_tex_stride))

        ys = list(range(0, max(1, tex_h - tile + 1), stride))
        xs = list(range(0, max(1, tex_w - tile + 1), stride))
        if not ys:
            ys = [0]
        if not xs:
            xs = [0]
        if ys[-1] != max(0, tex_h - tile):
            ys.append(max(0, tex_h - tile))
        if xs[-1] != max(0, tex_w - tile):
            xs.append(max(0, tex_w - tile))

        tiles = []
        for y0 in ys:
            y1 = min(tex_h, y0 + tile)
            for x0 in xs:
                x1 = min(tex_w, x0 + tile)
                tiles.append((int(y0), int(y1), int(x0), int(x1)))
        return tiles

    def _init_texture_tiling(self):
        self.texture.tex.requires_grad_(False)
        self._texture_tiles = self._build_texture_tiles(
            int(self.texture.tex.shape[2]), int(self.texture.tex.shape[3])
        )
        self._texture_tile_cursor = 0
        self._active_texture_tile_idx = None
        self._active_texture_tile = None
        self._texture_tile_param = None
        self._opt_tex_tile = None

    def _commit_active_texture_tile(self, release: bool = True):
        if self._texture_tile_param is None or self._active_texture_tile is None:
            return
        y0, y1, x0, x1 = self._active_texture_tile
        with torch.no_grad():
            self.texture.tex[..., y0:y1, x0:x1].copy_(self._texture_tile_param.detach())
        if release:
            self._texture_tile_param = None
            self._opt_tex_tile = None
            self._active_texture_tile = None
            self._active_texture_tile_idx = None

    def _next_texture_tile_index(self) -> int:
        n_tiles = len(self._texture_tiles)
        if n_tiles <= 0:
            return 0
        if self._tile_tex_random:
            return random.randrange(n_tiles)
        idx = self._texture_tile_cursor % n_tiles
        self._texture_tile_cursor += 1
        return idx

    def _activate_texture_tile(self, tile_idx: int):
        if not self._tile_tex_enabled:
            return
        tile_idx = int(max(0, min(tile_idx, len(self._texture_tiles) - 1)))
        if self._active_texture_tile_idx == tile_idx and self._texture_tile_param is not None:
            return

        self._commit_active_texture_tile(release=True)

        y0, y1, x0, x1 = self._texture_tiles[tile_idx]
        patch = self.texture.tex[..., y0:y1, x0:x1].detach().clone()
        self._texture_tile_param = nn.Parameter(patch, requires_grad=True)
        self._opt_tex_tile = self._create_tex_optimizer(params=[self._texture_tile_param])
        self._active_texture_tile = (y0, y1, x0, x1)
        self._active_texture_tile_idx = tile_idx

    def _texture_tensor_for_render(self) -> Optional[torch.Tensor]:
        if not self._tile_tex_enabled:
            return None
        if self._texture_tile_param is None or self._active_texture_tile is None:
            return None

        y0, y1, x0, x1 = self._active_texture_tile
        base = self.texture.tex
        base_patch = base[..., y0:y1, x0:x1].detach()
        delta = self._texture_tile_param - base_patch
        pad = (x0, int(base.shape[3]) - x1, y0, int(base.shape[2]) - y1)
        return base + F.pad(delta, pad)

    def _set_texture_stage(self, stage: int):
        if self._tile_tex_enabled:
            self._commit_active_texture_tile(release=True)
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
        if self._tile_tex_enabled:
            self._init_texture_tiling()
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

    def _refresh_geometry_regularization_tensors(self):
        self._edges = self._build_unique_edges(self.faces)
        self._face_adj_pairs, self._face_adj_edges = self._build_face_adjacency(
            self.faces, vertex_groups=self._weld_index
        )
        self._base_face_pair_d = None
        self._base_face_pair_weight = None
        if self._face_adj_pairs is not None and self._face_adj_pairs.shape[0] > 0:
            base_n, _ = compute_face_normals_and_double_area(self.base_vertices, self.faces)
            p = self._face_adj_pairs
            cos_base = (base_n[p[:, 0]] * base_n[p[:, 1]]).sum(dim=1).clamp(-1.0, 1.0)
            self._base_face_pair_d = (1.0 - cos_base).detach()
            if bool(self.cfg.geom_normal_tv_use_base_feature_weights):
                sigma = float(max(1e-6, self.cfg.geom_normal_tv_feature_sigma))
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
        else:
            self._weld_group_inv_counts = None

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

    def _set_topology(self, vertices: torch.Tensor, faces: torch.Tensor, uvs: torch.Tensor):
        self.base_vertices = vertices.to(self.device)
        self.faces = faces.to(self.device)
        self.uvs = uvs.to(self.device)

        self._weld_index = None
        self._num_weld_groups = int(self.base_vertices.shape[0])
        self._weld_enabled = False
        if self.learn_geometry and bool(self.cfg.weld_geometry_vertices):
            self._weld_index, self._num_weld_groups = self._build_weld_groups(
                self.base_vertices, self.cfg.weld_position_decimals
            )
            self._weld_enabled = self._num_weld_groups < int(self.base_vertices.shape[0])

        geom_shape = ((self._num_weld_groups, 3) if self._weld_enabled
                      else tuple(self.base_vertices.shape))
        self.geometry_offsets = nn.Parameter(
            torch.zeros(geom_shape, device=self.device, dtype=self.base_vertices.dtype),
            requires_grad=self.learn_geometry,
        )
        self._cast_geometry_parameter()
        self.vertex_offsets = self.geometry_offsets
        self._refresh_geometry_regularization_tensors()

        self._face_error_ema = torch.zeros(
            int(self.faces.shape[0]),
            device=self.device,
            dtype=self.base_vertices.dtype,
        )
        if self.learn_geometry:
            self.opt_geom = self._create_geom_optimizer()

    def _accumulate_face_error(self, face_ids: torch.Tensor, per_pixel_error: torch.Tensor, mesh_mask: torch.Tensor):
        if self._face_error_ema is None or self._face_error_ema.numel() == 0:
            return
        if face_ids is None:
            return
        valid = mesh_mask.bool() & (face_ids >= 0)
        if not torch.any(valid):
            return

        ids = face_ids[valid].long()
        errs = per_pixel_error[valid].to(dtype=self._face_error_ema.dtype)
        num_faces = int(self.faces.shape[0])
        face_sum = torch.zeros(num_faces, device=self.device, dtype=self._face_error_ema.dtype)
        face_cnt = torch.zeros(num_faces, device=self.device, dtype=self._face_error_ema.dtype)
        face_sum.index_add_(0, ids, errs)
        face_cnt.index_add_(0, ids, torch.ones_like(errs))
        seen = face_cnt > 0
        if not torch.any(seen):
            return
        face_mean = face_sum[seen] / face_cnt[seen].clamp(min=1.0)
        beta = float(min(max(self.cfg.topology_error_beta, 0.0), 0.9999))
        self._face_error_ema[seen] = beta * self._face_error_ema[seen] + (1.0 - beta) * face_mean

    def _select_topology_face_sets(self) -> Tuple[torch.Tensor, torch.Tensor]:
        num_faces = int(self.faces.shape[0])
        if num_faces <= 0:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        e = self._face_error_ema.detach().float()
        if e.numel() != num_faces or torch.all(e <= 0):
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        split_q = float(min(max(self.cfg.topology_split_quantile, 0.0), 1.0))
        merge_q = float(min(max(self.cfg.topology_merge_quantile, 0.0), 1.0))
        split_thr = torch.quantile(e, split_q)
        merge_thr = torch.quantile(e, merge_q)

        split_idx = torch.nonzero(e >= split_thr, as_tuple=False).squeeze(1)
        merge_idx = torch.nonzero(e <= merge_thr, as_tuple=False).squeeze(1)

        max_split = max(0, int(self.cfg.topology_max_splits))
        max_merge = max(0, int(self.cfg.topology_max_merges))
        if split_idx.numel() > max_split > 0:
            vals, order = torch.sort(e[split_idx], descending=True)
            split_idx = split_idx[order[:max_split]]
        if merge_idx.numel() > max_merge > 0:
            vals, order = torch.sort(e[merge_idx], descending=False)
            merge_idx = merge_idx[order[:max_merge]]

        if split_idx.numel() > 0 and merge_idx.numel() > 0:
            split_set = set(split_idx.detach().cpu().tolist())
            merge_list = [i for i in merge_idx.detach().cpu().tolist() if i not in split_set]
            merge_idx = torch.as_tensor(merge_list, dtype=torch.long)
        return split_idx, merge_idx

    def _remesh_split_merge(self, split_idx: torch.Tensor, merge_idx: torch.Tensor):
        verts = self.current_vertices().detach().cpu().numpy().astype(np.float32)
        uvs = self.uvs.detach().cpu().numpy().astype(np.float32)
        faces = self.faces.detach().cpu().numpy().astype(np.int64)

        split_set = set(split_idx.detach().cpu().tolist()) if split_idx.numel() > 0 else set()
        merge_set = set(merge_idx.detach().cpu().tolist()) if merge_idx.numel() > 0 else set()

        verts_np = verts.copy()
        uvs_np = uvs.copy()

        # Build merge candidates in vectorized form from merge-marked faces.
        if merge_set:
            merge_ids = np.fromiter(merge_set, dtype=np.int64)
            merge_ids = merge_ids[(merge_ids >= 0) & (merge_ids < faces.shape[0])]
        else:
            merge_ids = np.empty((0,), dtype=np.int64)

        if merge_ids.size > 0:
            tri_m = faces[merge_ids]  # (M,3)
            ia, ib, ic = tri_m[:, 0], tri_m[:, 1], tri_m[:, 2]
            pa, pb, pc = verts[ia], verts[ib], verts[ic]
            l_ab = np.linalg.norm(pa - pb, axis=1)
            l_bc = np.linalg.norm(pb - pc, axis=1)
            l_ca = np.linalg.norm(pc - pa, axis=1)
            lens = np.stack([l_ab, l_bc, l_ca], axis=1)
            min_edge = np.argmin(lens, axis=1)

            u = np.where(min_edge == 0, ia, np.where(min_edge == 1, ib, ic)).astype(np.int64)
            v = np.where(min_edge == 0, ib, np.where(min_edge == 1, ic, ia)).astype(np.int64)
            w = np.minimum(u, v)
            z = np.maximum(u, v)
            min_len = lens[np.arange(lens.shape[0]), min_edge].astype(np.float32)
            merge_candidates = np.stack([min_len, w.astype(np.float32), z.astype(np.float32)], axis=1)
        else:
            merge_candidates = np.empty((0, 3), dtype=np.float32)

        # Split path needs topology updates, keep this loop but avoid set lookups.
        if split_set:
            split_mask = np.zeros((faces.shape[0],), dtype=bool)
            split_ids = np.fromiter(split_set, dtype=np.int64)
            split_ids = split_ids[(split_ids >= 0) & (split_ids < faces.shape[0])]
            split_mask[split_ids] = True

            verts_list = [v.copy() for v in verts_np]
            uvs_list = [u.copy() for u in uvs_np]
            out_faces = []
            out_parent = []
            for fi, tri in enumerate(faces):
                a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                if split_mask[fi]:
                    pm = (verts_np[a] + verts_np[b] + verts_np[c]) / 3.0
                    um = (uvs_np[a] + uvs_np[b] + uvs_np[c]) / 3.0
                    m = len(verts_list)
                    verts_list.append(pm.astype(np.float32, copy=False))
                    uvs_list.append(um.astype(np.float32, copy=False))
                    out_faces.append([a, b, m])
                    out_faces.append([b, c, m])
                    out_faces.append([c, a, m])
                    out_parent.extend([fi, fi, fi])
                else:
                    out_faces.append([a, b, c])
                    out_parent.append(fi)

            faces_np = np.asarray(out_faces, dtype=np.int64)
            parent_idx = np.asarray(out_parent, dtype=np.int64)
            verts_np = np.asarray(verts_list, dtype=np.float32)
            uvs_np = np.asarray(uvs_list, dtype=np.float32)
        else:
            faces_np = faces.copy()
            parent_idx = np.arange(faces.shape[0], dtype=np.int64)

        applied_collapses = 0
        if merge_candidates.shape[0] > 0:
            # Build welded geometric groups so UV-seam duplicates are treated as one
            # topological vertex for collapse validity checks.
            if self._weld_index is not None and int(self._weld_index.numel()) == int(verts_np.shape[0]):
                vert_to_group = self._weld_index.detach().cpu().numpy().astype(np.int64, copy=False)
            else:
                weld_decimals = max(0, int(getattr(self.cfg, "weld_position_decimals", 6)))
                weld_key = np.round(verts_np, decimals=weld_decimals)
                _, vert_to_group = np.unique(weld_key, axis=0, return_inverse=True)
            group_count = int(vert_to_group.max() + 1) if vert_to_group.size > 0 else 0
            if group_count <= 0:
                return None

            # Precompute vertex index ranges per welded group for fast remap updates.
            group_order = np.argsort(vert_to_group, kind="mergesort")
            groups_sorted = vert_to_group[group_order]
            group_starts = np.searchsorted(groups_sorted, np.arange(group_count), side="left")
            group_ends = np.searchsorted(groups_sorted, np.arange(group_count), side="right")

            # Group centroids are reused in collapse midpoint updates.
            group_sum = np.zeros((group_count, 3), dtype=np.float32)
            np.add.at(group_sum, vert_to_group, verts_np)
            group_cnt = np.bincount(vert_to_group, minlength=group_count).astype(np.float32)
            group_cnt = np.clip(group_cnt, 1.0, None)
            group_centroid = group_sum / group_cnt[:, None]

            # Build undirected edge incidence on welded groups.
            faces_g = vert_to_group[faces_np]
            ge01 = faces_g[:, [0, 1]]
            ge12 = faces_g[:, [1, 2]]
            ge20 = faces_g[:, [2, 0]]
            g_edges = np.concatenate([ge01, ge12, ge20], axis=0)
            g_edges = np.sort(g_edges, axis=1)
            uniq_g_edges, g_edge_counts = np.unique(g_edges, axis=0, return_counts=True)

            # Fast incidence lookup by integer edge key + searchsorted.
            g_count = int(max(1, group_count))
            g_edge_keys = uniq_g_edges[:, 0].astype(np.int64) * g_count + uniq_g_edges[:, 1].astype(np.int64)
            g_order = np.argsort(g_edge_keys)
            g_edge_keys_sorted = g_edge_keys[g_order]
            g_edge_counts_sorted = g_edge_counts[g_order]

            # Greedy shortest-edge collapse, restricted to group-disjoint interior edges.
            cand_order = np.argsort(merge_candidates[:, 0])
            cand_u = merge_candidates[cand_order, 1].astype(np.int64)
            cand_v = merge_candidates[cand_order, 2].astype(np.int64)

            cand_u_g = vert_to_group[cand_u]
            cand_v_g = vert_to_group[cand_v]
            cand_a_g = np.minimum(cand_u_g, cand_v_g)
            cand_b_g = np.maximum(cand_u_g, cand_v_g)
            valid_pair = cand_a_g != cand_b_g

            cand_keys = cand_a_g * g_count + cand_b_g
            pos = np.searchsorted(g_edge_keys_sorted, cand_keys, side="left")
            in_range = pos < g_edge_keys_sorted.shape[0]
            present = (
                valid_pair
                & in_range
                & (g_edge_keys_sorted[np.minimum(pos, g_edge_keys_sorted.shape[0] - 1)] == cand_keys)
            )
            edge_inc = np.zeros_like(cand_u, dtype=np.int64)
            edge_inc[present] = g_edge_counts_sorted[pos[present]]

            used_groups = np.zeros((group_count,), dtype=bool)
            selected_pairs = []
            for idx in range(cand_u.shape[0]):
                u = int(cand_u[idx])
                v = int(cand_v[idx])
                ug = int(cand_u_g[idx])
                vg = int(cand_v_g[idx])
                if u == v:
                    continue
                if ug == vg:
                    continue
                if used_groups[ug] or used_groups[vg]:
                    continue

                inc = int(edge_inc[idx])
                if inc <= 0:
                    continue

                # Interior-only collapse preserves watertightness on closed meshes.
                if inc != 2:
                    continue

                keep_g = ug
                rem_g = vg
                selected_pairs.append((u, v, keep_g, rem_g))
                used_groups[ug] = True
                used_groups[vg] = True

            if selected_pairs:
                remap = np.arange(verts_np.shape[0], dtype=np.int64)
                active = np.ones(verts_np.shape[0], dtype=bool)

                for keep, rem, keep_g, rem_g in selected_pairs:
                    if not active[keep] or not active[rem]:
                        continue

                    ks, ke = int(group_starts[keep_g]), int(group_ends[keep_g])
                    rs, re = int(group_starts[rem_g]), int(group_ends[rem_g])
                    keep_members = group_order[ks:ke]
                    rem_members = group_order[rs:re]
                    if keep_members.size == 0 or rem_members.size == 0:
                        continue

                    # Move all welded keep-group vertices to midpoint of collapsed pair.
                    p_mid = 0.5 * (group_centroid[keep_g] + group_centroid[rem_g])
                    verts_np[keep_members] = p_mid

                    # Merge all rem-group vertices into the keep representative.
                    remap[rem_members] = keep
                    active[rem_members] = False
                    applied_collapses += 1

                faces_np = remap[faces_np]

                old_to_new = -np.ones(verts_np.shape[0], dtype=np.int64)
                old_to_new[active] = np.arange(int(active.sum()), dtype=np.int64)
                faces_np = old_to_new[faces_np]
                verts_np = verts_np[active]
                uvs_np = uvs_np[active]

        # Re-orient faces to match their source (pre-remesh) face normal direction.
        if faces_np.shape[0] > 0 and parent_idx.shape[0] == faces_np.shape[0]:
            tri_new = verts_np[faces_np]  # (F,3,3)
            n_new = np.cross(tri_new[:, 1] - tri_new[:, 0], tri_new[:, 2] - tri_new[:, 0])

            tri_ref = verts[faces[parent_idx]]  # reference from current pre-remesh mesh
            n_ref = np.cross(tri_ref[:, 1] - tri_ref[:, 0], tri_ref[:, 2] - tri_ref[:, 0])

            dot = np.einsum("ij,ij->i", n_new, n_ref)
            flip = dot < 0.0
            if np.any(flip):
                tmp = faces_np[flip, 1].copy()
                faces_np[flip, 1] = faces_np[flip, 2]
                faces_np[flip, 2] = tmp

        nondeg = (
            (faces_np[:, 0] != faces_np[:, 1])
            & (faces_np[:, 1] != faces_np[:, 2])
            & (faces_np[:, 2] != faces_np[:, 0])
        )
        faces_np = faces_np[nondeg]
        parent_idx = parent_idx[nondeg]
        if faces_np.shape[0] == 0:
            return None

        # Remove duplicate triangles (same vertex set).
        key = np.sort(faces_np, axis=1)
        _, keep_idx = np.unique(key, axis=0, return_index=True)
        keep_idx = np.sort(keep_idx)
        faces_np = faces_np[keep_idx]
        parent_idx = parent_idx[keep_idx]

        return (
            torch.from_numpy(verts_np),
            torch.from_numpy(faces_np.astype(np.int64)),
            torch.from_numpy(uvs_np),
            int(len(split_set)),
            int(applied_collapses),
        )

    def _maybe_adapt_topology(self):
        every = int(getattr(self.cfg, "topology_adapt_every", 0) or 0)
        if every <= 0:
            return
        if (self.iter + 1) % every != 0:
            return

        def _skip(reason: str):
            print(f"[Topology] skip iter={self.iter+1}: {reason}")

        start_iter = self.cfg.topology_start_iter
        if start_iter is None:
            start_iter = max(int(self.cfg.warmup_iters), int(self.cfg.geometry_warmup_iters))
        if self.iter < int(start_iter):
            _skip(f"before start_iter ({self.iter} < {int(start_iter)})")
            return

        min_faces = max(4, int(getattr(self.cfg, "topology_min_faces", 4)))
        cur_faces = int(self.faces.shape[0])

        split_idx, merge_idx = self._select_topology_face_sets()
        raw_split = int(split_idx.numel())
        raw_merge = int(merge_idx.numel())

        top_max_faces = int(getattr(self.cfg, "topology_max_faces", 0) or 0)
        collapse_only_mode = False
        split_only_mode = False

        # If we are above max, only allow collapsing.
        if top_max_faces > 0 and cur_faces > top_max_faces:
            split_idx = torch.empty(0, dtype=torch.long)
            collapse_only_mode = True
            print(
                f"[Topology] iter={self.iter+1}: face_count {cur_faces:,} > max {top_max_faces:,}, "
                "collapse-only mode"
            )

        # If we are below min, only allow splitting.
        if cur_faces < min_faces:
            merge_idx = torch.empty(0, dtype=torch.long)
            split_only_mode = True
            print(
                f"[Topology] iter={self.iter+1}: face_count {cur_faces:,} < min {min_faces:,}, "
                "split-only mode"
            )

        if split_idx.numel() == 0 and merge_idx.numel() == 0:
            _skip(
                f"no candidates after constraints (raw split={raw_split}, raw merge={raw_merge})"
            )
            return

        remesh = self._remesh_split_merge(split_idx, merge_idx)
        if remesh is None:
            _skip("remesh returned None (all faces became degenerate/invalid)")
            return
        verts_new, faces_new, uvs_new, n_split, n_merge_edges = remesh
        new_faces = int(faces_new.shape[0])

        # When forced into one-sided adaptation, accept directional improvement
        # even if we are still outside the target range.
        if collapse_only_mode:
            if new_faces >= cur_faces:
                _skip(
                    f"collapse-only made no progress (faces {cur_faces:,} -> {new_faces:,})"
                )
                return
        elif split_only_mode:
            if new_faces <= cur_faces:
                _skip(
                    f"split-only made no progress (faces {cur_faces:,} -> {new_faces:,})"
                )
                return
        else:
            if new_faces < min_faces:
                _skip(
                    f"candidate topology dropped below min_faces "
                    f"({new_faces:,} < {min_faces:,})"
                )
                return
            if top_max_faces > 0 and new_faces > top_max_faces:
                _skip(
                    f"candidate topology exceeded max_faces "
                    f"({new_faces:,} > {top_max_faces:,})"
                )
                return

        old_v = int(self.base_vertices.shape[0])
        old_f = int(self.faces.shape[0])
        self._set_topology(verts_new, faces_new, uvs_new)
        self._topology_events += 1
        print(
            f"[Topology] event#{self._topology_events} iter={self.iter+1} "
            f"verts {old_v:,}->{int(self.base_vertices.shape[0]):,}, "
            f"faces {old_f:,}->{int(self.faces.shape[0]):,} "
            f"(split={n_split}, collapse_edges={n_merge_edges})"
        )

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

    def render_view(self, view: CameraView, return_mask: bool = False, return_face_ids: bool = False):
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

        tex_override = self._texture_tensor_for_render()
        if return_face_ids:
            hdr, mask, face_ids = self.rasterizer.render_with_mask_and_face_ids(
                self.current_vertices(), self.faces, self.uvs,
                self.texture, R, t, K, W, H,
                texture_tensor=tex_override,
            )
        else:
            hdr, mask = self.rasterizer.render_with_mask(
                self.current_vertices(), self.faces, self.uvs,
                self.texture, R, t, K, W, H,
                texture_tensor=tex_override,
            )
        ldr = self.ppisp(hdr, ci)
        pred = ldr * mask
        if return_mask:
            if return_face_ids:
                return pred, (mask[..., 0] > 0.5), face_ids
            return pred, (mask[..., 0] > 0.5)
        if return_face_ids:
            return pred, face_ids
        return pred

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, view: CameraView, gt_image: Optional[torch.Tensor] = None) -> dict:
        timing_sync_cuda = bool(getattr(self.cfg, "timing_sync_cuda", False))

        def _sync_timing():
            if timing_sync_cuda and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        _sync_timing()
        t_step0 = time.perf_counter()

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
        if self._tile_tex_enabled:
            for p in self.texture.parameters():
                p.requires_grad_(False)
            if train_tex and len(self._texture_tiles) > 0:
                self._activate_texture_tile(self._next_texture_tile_index())
            else:
                self._commit_active_texture_tile(release=True)
        else:
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

        tex_opt = self._opt_tex_tile if self._tile_tex_enabled else self.opt_tex
        tex_params = ([self._texture_tile_param]
                      if (self._tile_tex_enabled and self._texture_tile_param is not None)
                      else list(self.texture.parameters()))

        if tex_opt is not None:
            tex_opt.zero_grad(set_to_none=True)
        self.opt_ppisp.zero_grad(set_to_none=True)
        if self.opt_geom is not None:
            self.opt_geom.zero_grad(set_to_none=True)

        _sync_timing()
        t_render0 = time.perf_counter()
        with self._autocast_ctx():
            pred, mesh_mask, face_ids = self.render_view(
                view, return_mask=True, return_face_ids=True
            )
        _sync_timing()
        t_render_ms = (time.perf_counter() - t_render0) * 1000.0

        # Keep render in AMP, but evaluate objective in fp32 for stability.
        _sync_timing()
        t_loss0 = time.perf_counter()
        if self._amp_enabled and self.cfg.amp_loss_fp32:
            pred_loss = pred.float()
            gt_loss = gt.float()
        else:
            pred_loss = pred
            gt_loss = gt

        t_faceerr_ms = 0.0
        if int(getattr(self.cfg, "topology_adapt_every", 0) or 0) > 0:
            _sync_timing()
            t_faceerr0 = time.perf_counter()
            per_pixel_error = (pred_loss - gt_loss).abs().mean(dim=-1).detach()
            self._accumulate_face_error(face_ids, per_pixel_error, mesh_mask)
            _sync_timing()
            t_faceerr_ms = (time.perf_counter() - t_faceerr0) * 1000.0

        _sync_timing()
        t_lossfn0 = time.perf_counter()
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
        _sync_timing()
        t_lossfn_ms = (time.perf_counter() - t_lossfn0) * 1000.0
        _sync_timing()
        t_loss_ms = (time.perf_counter() - t_loss0) * 1000.0
        total = losses["total"]

        if not torch.isfinite(total):
            if tex_opt is not None:
                tex_opt.zero_grad(set_to_none=True)
            self.opt_ppisp.zero_grad(set_to_none=True)
            if self.opt_geom is not None:
                self.opt_geom.zero_grad(set_to_none=True)
            if self._grad_scaler is not None:
                cur_scale = float(self._grad_scaler.get_scale())
                self._grad_scaler.update(new_scale=max(cur_scale * 0.5, 1.0))
            _sync_timing()
            t_step_ms = (time.perf_counter() - t_step0) * 1000.0
            return {
                "total": float("nan"),
                "photo": losses["photo"].detach().item(),
                "ppisp_reg": losses["ppisp_reg"].detach().item(),
                "geom_normal_tv": losses["geom_normal_tv"].detach().item(),
                "geom_edge_uniform": losses["geom_edge_uniform"].detach().item(),
                "time_render_ms": float(t_render_ms),
                "time_loss_ms": float(t_loss_ms),
                "time_faceerr_ms": float(t_faceerr_ms),
                "time_lossfn_ms": float(t_lossfn_ms),
                "time_backward_ms": 0.0,
                "time_step_ms": float(t_step_ms),
            }

        _sync_timing()
        t_backward0 = time.perf_counter()
        if self._grad_scaler is not None:
            self._grad_scaler.scale(total).backward()

            self._grad_scaler.unscale_(self.opt_ppisp)
            torch.nn.utils.clip_grad_norm_(self.ppisp.parameters(), 1.0)
            if train_tex and tex_opt is not None:
                self._grad_scaler.unscale_(tex_opt)
                torch.nn.utils.clip_grad_norm_(tex_params, 1.0)
            if train_geom and self.opt_geom is not None:
                self._grad_scaler.unscale_(self.opt_geom)
                torch.nn.utils.clip_grad_norm_([self.geometry_offsets], 1.0)

            if train_tex and tex_opt is not None:
                self._grad_scaler.step(tex_opt)
            if train_geom and self.opt_geom is not None:
                self._grad_scaler.step(self.opt_geom)
            self._grad_scaler.step(self.opt_ppisp)
            self._grad_scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.ppisp.parameters(), 1.0)
            if train_tex and tex_opt is not None:
                torch.nn.utils.clip_grad_norm_(tex_params, 1.0)
                tex_opt.step()
            if train_geom and self.opt_geom is not None:
                torch.nn.utils.clip_grad_norm_([self.geometry_offsets], 1.0)
                self.opt_geom.step()
            self.opt_ppisp.step()

        if self._tile_tex_enabled and train_tex:
            self._commit_active_texture_tile(release=True)

        _sync_timing()
        t_backward_ms = (time.perf_counter() - t_backward0) * 1000.0
        t_step_ms = (time.perf_counter() - t_step0) * 1000.0

        out = {
            **losses,
            "time_render_ms": float(t_render_ms),
            "time_loss_ms": float(t_loss_ms),
            "time_faceerr_ms": float(t_faceerr_ms),
            "time_lossfn_ms": float(t_lossfn_ms),
            "time_backward_ms": float(t_backward_ms),
            "time_step_ms": float(t_step_ms),
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
            if self.opt_tex is not None:
                for g in self.opt_tex.param_groups:
                    g["lr"] = cfg.lr_texture * f
            if self._opt_tex_tile is not None:
                for g in self._opt_tex_tile.param_groups:
                    g["lr"] = cfg.lr_texture * f
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
        running_time = {
            "data_ms": 0.0,
            "render_ms": 0.0,
            "loss_ms": 0.0,
            "faceerr_ms": 0.0,
            "lossfn_ms": 0.0,
            "backward_ms": 0.0,
            "step_ms": 0.0,
        }
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

                t_data0 = time.perf_counter()
                gt = self._image_cache.get(vidx)
                data_ms = (time.perf_counter() - t_data0) * 1000.0
                losses = self.step(view, gt_image=gt)
                self._update_lr()
                self._maybe_adapt_topology()
                self._poll_live_view_events()
                self._update_live_view(losses)

                for k in running:
                    running[k] += losses.get(k, 0.0)
                running_time["data_ms"] += data_ms
                running_time["render_ms"] += float(losses.get("time_render_ms", 0.0))
                running_time["loss_ms"] += float(losses.get("time_loss_ms", 0.0))
                running_time["faceerr_ms"] += float(losses.get("time_faceerr_ms", 0.0))
                running_time["lossfn_ms"] += float(losses.get("time_lossfn_ms", 0.0))
                running_time["backward_ms"] += float(losses.get("time_backward_ms", 0.0))
                running_time["step_ms"] += float(losses.get("time_step_ms", 0.0))
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
                    self.loss_log_interval.append({
                        "iter": self.iter + 1,
                        "photo": float(avg["photo"]),
                        "total": float(avg["total"]),
                    })
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
                    t_data = running_time["data_ms"] / cfg.log_every
                    t_render = running_time["render_ms"] / cfg.log_every
                    t_loss = running_time["loss_ms"] / cfg.log_every
                    t_faceerr = running_time["faceerr_ms"] / cfg.log_every
                    t_lossfn = running_time["lossfn_ms"] / cfg.log_every
                    t_backward = running_time["backward_ms"] / cfg.log_every
                    t_step = running_time["step_ms"] / cfg.log_every
                    print(
                        f"  {self.iter+1:5d}/{cfg.num_iterations}  [{mode}]  "
                        f"w_loss={w_total:.4f}  w_photo={w_photo:.4f}  "
                        f"w_ppisp_reg={w_ppisp_reg:.5f}  "
                        f"{geom_txt} "
                        f"t_data={t_data:.1f}ms  t_render={t_render:.1f}ms  "
                        f"t_loss={t_loss:.1f}ms (face={t_faceerr:.1f}, lossfn={t_lossfn:.1f})  "
                        f"t_bw={t_backward:.1f}ms  "
                        f"t_step={t_step:.1f}ms  "
                        f"{it_per_s:.1f} it/s  ETA {eta_s/60:.1f}m"
                    )
                    running = {k: 0.0 for k in running}
                    running_time = {k: 0.0 for k in running_time}
                    t_iter  = time.time()
                    if progress_callback:
                        progress_callback(self.iter + 1, cfg.num_iterations, losses)

                if (self.iter + 1) % cfg.save_every == 0:
                    self._save_checkpoint(self.iter + 1)

            self._save_loss_plot()
            print(f"\n[Trainer] ✓  Done in {(time.time()-t0)/60:.1f} min")
            self.ppisp.print_summary()
        finally:
            self._image_cache.close()
            self._close_live_view()

    # ------------------------------------------------------------------
    # Checkpoint / export
    # ------------------------------------------------------------------

    def _save_loss_plot(self):
        points = list(self.loss_log_interval)
        if not points:
            step = max(1, int(self.cfg.log_every))
            points = [
                {
                    "iter": int(row.get("iter", i + 1)) + 1,
                    "photo": float(row.get("photo", 0.0)),
                    "total": float(row.get("total", 0.0)),
                }
                for i, row in enumerate(self.loss_log)
                if (i + 1) % step == 0
            ]
        if not points:
            print("[Trainer] LossPlot  : skipped (no interval loss samples)")
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[Trainer] LossPlot  : skipped (matplotlib unavailable: {e})")
            return

        out_path = os.path.join(self.cfg.output_dir, "loss_curve.png")
        iters = [int(p["iter"]) for p in points]
        photo = [float(p["photo"]) for p in points]
        total = [float(p["total"]) for p in points]

        plt.figure(figsize=(9.5, 5.0))
        plt.plot(iters, photo, linewidth=1.8, label="Photometric Loss")
        plt.plot(iters, total, linewidth=1.8, label="Total Loss")
        plt.xlabel(f"Iteration (every {max(1, int(self.cfg.log_every))} iters)")
        plt.ylabel("Loss")
        plt.title("Training Loss Curves")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"[Trainer] LossPlot  : {out_path}")

    def _save_checkpoint(self, iteration: int):
        if self._tile_tex_enabled:
            self._commit_active_texture_tile(release=True)
        path = os.path.join(self.cfg.output_dir, f"checkpoint_{iteration:06d}.pt")
        torch.save({
            "iteration":  iteration,
            "mesh_vertices": self.base_vertices.detach().cpu(),
            "mesh_faces": self.faces.detach().cpu(),
            "mesh_uvs": self.uvs.detach().cpu(),
            "texture":    self.texture.state_dict(),
            "ppisp":      self.ppisp.state_dict(),
            "opt_tex":    self.opt_tex.state_dict() if self.opt_tex is not None else None,
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
            "face_error_ema": self._face_error_ema.detach().cpu() if self._face_error_ema is not None else None,
            "topology_events": int(self._topology_events),
            "loss_log":   self.loss_log,
            "loss_log_interval": self.loss_log_interval,
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

        mesh_v = ckpt.get("mesh_vertices")
        mesh_f = ckpt.get("mesh_faces")
        mesh_uv = ckpt.get("mesh_uvs")
        if mesh_v is not None and mesh_f is not None and mesh_uv is not None:
            self._set_topology(mesh_v.to(self.device), mesh_f.to(self.device), mesh_uv.to(self.device))

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
        if self._tile_tex_enabled:
            self._init_texture_tiling()
        if self.opt_tex is not None and ckpt.get("opt_tex") is not None:
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
        fe = ckpt.get("face_error_ema")
        if fe is not None:
            fe = fe.to(self.device)
            if fe.shape[0] == self.faces.shape[0]:
                self._face_error_ema = fe.to(dtype=self.base_vertices.dtype)
        self._topology_events = int(ckpt.get("topology_events", 0))
        if self._grad_scaler is not None and ckpt.get("grad_scaler") is not None:
            self._grad_scaler.load_state_dict(ckpt["grad_scaler"])
        self.iter     = ckpt["iteration"]
        self.loss_log = ckpt.get("loss_log", [])
        self.loss_log_interval = ckpt.get("loss_log_interval", [])
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