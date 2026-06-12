#!/usr/bin/env python3
"""
Texture + Per-Camera PPISP Optimiser
=====================================
Jointly optimises mesh texture and per-camera ISP parameters to fix
exposure inconsistencies in COLMAP-reconstructed textured meshes.

Usage:
  python run_optimizer.py --scene /path/to/dense --mesh /path/to/mesh.ply
  python run_optimizer.py --synthetic --iters 300
  python run_optimizer.py --scene /path/to/dense --mesh mesh.obj --resume outputs/checkpoint_005000.pt
"""

import argparse
import sys
import os
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Texture + PPISP joint optimiser")
    p.add_argument("--scene",       type=str,   default=None,
                   help="Path to COLMAP scene root (contains sparse/ and images/)")
    p.add_argument("--mesh",        type=str,   default=None,
                   help="Path to .ply or .obj mesh (auto-detected if omitted)")
    p.add_argument("--synthetic",   action="store_true",
                   help="Run on a synthetic test scene (no real data needed)")
    p.add_argument("--output",      type=str,   default="outputs",
                   help="Output directory for texture, PPISP params, checkpoints")
    p.add_argument("--iters",       type=int,   default=5000,
                   help="Total training iterations")
    p.add_argument("--scale",       type=float, default=1,
                   help="GT image downscale factor (0.5 = half-res, faster)")
    p.add_argument("--tex_res",     type=int,   default=8192,
                   help="Target output texture long side (aspect ratio preserved from input texture when available)")
    p.add_argument("--uv_unwrap", action="store_true",
                   help="Run Open3D xatlas UV unwrap on input mesh before training")
    p.add_argument("--tex_dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"],
                   help="Texture parameter dtype (auto follows AMP on CUDA; lowers VRAM when fp16/bf16)")
    p.add_argument("--tex_optimizer", type=str, default="adam", choices=["adam", "sgd"],
                   help="Texture optimizer (sgd reduces VRAM vs adam)")
    p.add_argument("--progressive_tex", action="store_true",
                   help="Use progressive texture upscaling during training (1/2 -> full)")
    p.add_argument("--lr_tex",      type=float, default=1e-4,
                   help="Texture learning rate")
    p.add_argument("--lr_ppisp",    type=float, default=3e-3,
                   help="PPISP learning rate")
    p.add_argument("--lr_decay_start", type=int, default=None,
                   help="Iteration to start LR decay (default: auto at ~75%% of total iters)")
    p.add_argument("--lr_decay_iters", type=int, default=None,
                   help="Number of iterations over which LR decays (default: auto to end of run)")
    p.add_argument("--lr_decay_factor", type=float, default=0.1,
                   help="Final LR multiplier after cosine decay")
    p.add_argument("--l1_weight", type=float, default=0.8,
                   help="Photometric L1 weight")
    p.add_argument("--ssim_weight", type=float, default=0.2,
                   help="Photometric SSIM weight")
    p.add_argument("--ssim_backend", type=str, default="auto",
                   choices=["auto", "native", "msssim"],
                   help="SSIM backend: auto (prefer pytorch_msssim), native, or msssim")
    p.add_argument("--lr_geom",     type=float, default=1e-4,
                   help="Geometry (vertex offsets) learning rate")
    p.add_argument("--warmup",      type=int,   default=500,
                   help="Warmup iterations (PPISP only, texture frozen)")
    p.add_argument("--tex_update_every", type=int, default=1,
                   help="Update texture every N iterations after warmup")
    p.add_argument("--geom_warmup", type=int,   default=500,
                   help="Warmup iterations before geometry updates start")
    p.add_argument("--geom_update_every", type=int, default=1,
                   help="Update geometry every N iterations after geom warmup")
    p.add_argument("--geom_dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"],
                   help="Geometry offset dtype (auto follows AMP on CUDA; lowers VRAM when fp16/bf16)")
    p.add_argument("--geom_optimizer", type=str, default="adam", choices=["adam", "sgd"],
                   help="Geometry optimizer (sgd reduces VRAM vs adam)")
    p.add_argument("--geom_tv_weight", type=float, default=1,
                   help="Weight for geometry normal-TV regularization")
    p.add_argument("--geom_edge_uniform_weight", type=float, default=1e-1,
                   help="Weight for per-face edge-length uniformity regularization")
    p.add_argument("--geom_edge_uniform_eps", type=float, default=1e-8,
                   help="Numerical epsilon for edge uniformity regularization")
    p.add_argument("--learn_geometry", action="store_true",
                   help="Enable differentiable mesh geometry updates")
    p.add_argument("--max_vertex_offset", type=float, default=0,
                   help="Max absolute vertex displacement (scene units); set <=0 to disable clamp")
    p.add_argument("--no_weld_geometry", action="store_true",
                   help="Disable welding duplicate-position vertices for geometry optimization")
    p.add_argument("--no_vignette", action="store_true",
                   help="Disable per-camera vignette learning")
    p.add_argument("--ppisp_gamma", type=float, default=2.2,
                   help="PPISP gamma value (used as fixed gamma by default)")
    p.add_argument("--learn_gamma", action="store_true",
                   help="Enable per-camera PPISP gamma learning (default: fixed --ppisp_gamma)")
    p.add_argument("--resume",      type=str,   default=None,
                   help="Resume from checkpoint file")
    p.add_argument("--device",      type=str,   default=None,
                   help="cuda or cpu (auto-detected if omitted)")
    p.add_argument("--max_cameras", type=int,   default=None,
                   help="Cap number of cameras loaded (useful for quick tests)")
    p.add_argument("--live_view", action="store_true",
                   help="Show interactive live render window during training")
    p.add_argument("--live_view_every", type=int, default=50,
                   help="Update live render every N iterations")
    p.add_argument("--live_view_cam", type=int, default=0,
                   help="Initial camera index for live view")
    p.add_argument("--live_view_max_size", type=int, default=1200,
                   help="Max preview width/height in pixels for live window")
    p.add_argument("--amp", action="store_true", default=False,
                   help="Enable mixed precision acceleration on CUDA (default: disabled)")
    p.add_argument("--no_amp", action="store_false", dest="amp",
                   help="Disable mixed precision acceleration")
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["fp16", "bf16"],
                   help="AMP dtype to use when --amp is enabled")
    p.add_argument("--no_amp_loss_fp32", action="store_true",
                   help="Keep loss evaluation in AMP dtype (faster, less stable). Default computes losses in fp32")
    p.add_argument("--amp_init_scale", type=float, default=1024.0,
                   help="Initial GradScaler scale for fp16 AMP")
    p.add_argument("--amp_growth_interval", type=int, default=2000,
                   help="GradScaler growth interval for fp16 AMP")
    p.add_argument("--no_tf32", action="store_true",
                   help="Disable TF32 matmul/conv acceleration on CUDA")
    p.add_argument("--image_cpu_cache_size", type=int, default=32,
                   help="Number of GT images kept in CPU RAM cache")
    p.add_argument("--image_gpu_cache_size", type=int, default=4,
                   help="Number of GT images prefetched to GPU cache")
    p.add_argument("--image_prefetch_ahead", type=int, default=2,
                   help="How many future training views to prefetch")
    p.add_argument("--image_loader_workers", type=int, default=2,
                   help="Background worker threads for image decode/load")
    p.add_argument("--no_image_fs_cache", action="store_true",
                   help="Disable filesystem cache of resized GT images")
    p.add_argument("--image_cache_dir", type=str, default=None,
                   help="Filesystem cache directory for GT images (default: <output>/image_cache)")
    p.add_argument("--tex_seam_pad", type=int, default=12,
                   help="UV seam padding in pixels applied on export texture to reduce visible seams")
    return p.parse_args()


def main():
    args = parse_args()

    cache_dir = args.image_cache_dir or os.path.join(args.output, "image_cache")
    if os.path.isdir(cache_dir):
        print(f"[Startup] Removing existing image cache: {cache_dir}")
        shutil.rmtree(cache_dir)
    elif os.path.exists(cache_dir):
        print(f"[Startup] Removing existing cache file: {cache_dir}")
        os.remove(cache_dir)

    import torch
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    from texture_optimizer.trainer import TrainConfig, TexturePPISPTrainer

    # ------------------------------------------------------------------ synthetic
    if args.synthetic:
        print("\n[Demo] Running synthetic test scene ...\n")
        from texture_optimizer.dataset import make_synthetic_scene
        scene = make_synthetic_scene(num_cameras=12, W=128, H=96,
                                     device=str(device))
        cfg = TrainConfig()
        cfg.num_iterations  = args.iters
        cfg.output_dir      = args.output
        cfg.tex_H = cfg.tex_W = 256
        cfg.warmup_iters    = min(50, args.iters // 10)
        cfg.tex_update_every = max(1, args.tex_update_every)
        cfg.lr_texture      = args.lr_tex
        cfg.lr_ppisp        = args.lr_ppisp
        cfg.lr_decay_start  = args.lr_decay_start
        cfg.lr_decay_iters  = args.lr_decay_iters
        cfg.lr_decay_factor = args.lr_decay_factor
        cfg.progressive_texture = args.progressive_tex
        cfg.texture_dtype   = args.tex_dtype
        cfg.tex_optimizer   = args.tex_optimizer
        cfg.l1_weight       = args.l1_weight
        cfg.ssim_weight     = args.ssim_weight
        cfg.ssim_backend    = args.ssim_backend
        cfg.lr_geometry     = args.lr_geom
        cfg.learn_vignette  = not args.no_vignette
        cfg.ppisp_gamma     = args.ppisp_gamma
        cfg.learn_gamma     = args.learn_gamma
        cfg.learn_geometry  = args.learn_geometry
        cfg.geometry_warmup_iters = args.geom_warmup
        cfg.geom_update_every = max(1, args.geom_update_every)
        cfg.geometry_dtype  = args.geom_dtype
        cfg.geom_optimizer  = args.geom_optimizer
        cfg.geom_normal_tv_weight = args.geom_tv_weight
        cfg.geom_edge_uniform_weight = args.geom_edge_uniform_weight
        cfg.geom_edge_uniform_eps = args.geom_edge_uniform_eps
        cfg.max_vertex_offset     = (args.max_vertex_offset
                         if args.max_vertex_offset > 0 else None)
        cfg.weld_geometry_vertices = not args.no_weld_geometry
        cfg.live_view       = args.live_view
        cfg.live_view_every = args.live_view_every
        cfg.live_view_camera = args.live_view_cam
        cfg.live_view_max_size = args.live_view_max_size
        cfg.use_amp         = args.amp
        cfg.amp_dtype       = args.amp_dtype
        cfg.amp_loss_fp32   = not args.no_amp_loss_fp32
        cfg.amp_init_scale  = args.amp_init_scale
        cfg.amp_growth_interval = args.amp_growth_interval
        cfg.use_tf32        = not args.no_tf32
        cfg.image_cpu_cache_size = args.image_cpu_cache_size
        cfg.image_gpu_cache_size = args.image_gpu_cache_size
        cfg.image_prefetch_ahead = args.image_prefetch_ahead
        cfg.image_loader_workers = args.image_loader_workers
        cfg.image_fs_cache       = not args.no_image_fs_cache
        cfg.image_cache_dir      = args.image_cache_dir
        cfg.tex_seam_pad_px = max(0, int(args.tex_seam_pad))
        cfg.device          = str(device)
        cfg.log_every       = max(1, args.iters // 30)
        cfg.save_every      = max(1, args.iters // 5)
        trainer = TexturePPISPTrainer(scene, cfg)
        if args.resume:
            trainer.load_checkpoint(args.resume)
        trainer.train()
        trainer.export_results()
        return

    # ------------------------------------------------------------------ real scene
    if args.scene is None:
        print("ERROR: provide --scene or --synthetic")
        sys.exit(1)

    from texture_optimizer.dataset import ColmapScene
    scene = ColmapScene(
        scene_root=args.scene,
        image_scale=args.scale,
        max_cameras=args.max_cameras,
        mesh_path=args.mesh,
        uv_unwrap_mode=("open3d_xatlas" if args.uv_unwrap else "none"),
        uv_unwrap_size=args.tex_res,
        device=str(device),
    )
    cfg = TrainConfig()
    cfg.num_iterations  = args.iters
    cfg.output_dir      = args.output
    if scene.mesh.tex_image is not None:
        in_h = int(scene.mesh.tex_image.shape[0])
        in_w = int(scene.mesh.tex_image.shape[1])
        long_side = max(in_h, in_w)
        if long_side > 0:
            scale = float(args.tex_res) / float(long_side)
            cfg.tex_H = max(1, int(round(in_h * scale)))
            cfg.tex_W = max(1, int(round(in_w * scale)))
        else:
            cfg.tex_H = cfg.tex_W = args.tex_res
    else:
        cfg.tex_H = cfg.tex_W = args.tex_res
    cfg.warmup_iters    = args.warmup
    cfg.tex_update_every = max(1, args.tex_update_every)
    cfg.lr_texture      = args.lr_tex
    cfg.lr_ppisp        = args.lr_ppisp
    cfg.lr_decay_start  = args.lr_decay_start
    cfg.lr_decay_iters  = args.lr_decay_iters
    cfg.lr_decay_factor = args.lr_decay_factor
    cfg.progressive_texture = args.progressive_tex
    cfg.texture_dtype   = args.tex_dtype
    cfg.tex_optimizer   = args.tex_optimizer
    cfg.l1_weight       = args.l1_weight
    cfg.ssim_weight     = args.ssim_weight
    cfg.ssim_backend    = args.ssim_backend
    cfg.lr_geometry     = args.lr_geom
    cfg.learn_vignette  = not args.no_vignette
    cfg.ppisp_gamma     = args.ppisp_gamma
    cfg.learn_gamma     = args.learn_gamma
    cfg.learn_geometry  = args.learn_geometry
    cfg.geometry_warmup_iters = args.geom_warmup
    cfg.geom_update_every = max(1, args.geom_update_every)
    cfg.geometry_dtype  = args.geom_dtype
    cfg.geom_optimizer  = args.geom_optimizer
    cfg.geom_normal_tv_weight = args.geom_tv_weight
    cfg.geom_edge_uniform_weight = args.geom_edge_uniform_weight
    cfg.geom_edge_uniform_eps = args.geom_edge_uniform_eps
    cfg.max_vertex_offset     = (args.max_vertex_offset
                                 if args.max_vertex_offset > 0 else None)
    cfg.weld_geometry_vertices = not args.no_weld_geometry
    cfg.live_view       = args.live_view
    cfg.live_view_every = args.live_view_every
    cfg.live_view_camera = args.live_view_cam
    cfg.live_view_max_size = args.live_view_max_size
    cfg.use_amp         = args.amp
    cfg.amp_dtype       = args.amp_dtype
    cfg.amp_loss_fp32   = not args.no_amp_loss_fp32
    cfg.amp_init_scale  = args.amp_init_scale
    cfg.amp_growth_interval = args.amp_growth_interval
    cfg.use_tf32        = not args.no_tf32
    cfg.image_cpu_cache_size = args.image_cpu_cache_size
    cfg.image_gpu_cache_size = args.image_gpu_cache_size
    cfg.image_prefetch_ahead = args.image_prefetch_ahead
    cfg.image_loader_workers = args.image_loader_workers
    cfg.image_fs_cache       = not args.no_image_fs_cache
    cfg.image_cache_dir      = args.image_cache_dir
    cfg.tex_seam_pad_px = max(0, int(args.tex_seam_pad))
    cfg.device          = str(device)

    trainer = TexturePPISPTrainer(scene, cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()
    trainer.export_results()


if __name__ == "__main__":
    main()