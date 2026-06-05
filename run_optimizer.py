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
    p.add_argument("--tex_res",     type=int,   default=1024,
                   help="Target output texture long side (aspect ratio preserved from input texture when available)")
    p.add_argument("--lr_tex",      type=float, default=1e-3,
                   help="Texture learning rate")
    p.add_argument("--lr_ppisp",    type=float, default=5e-3,
                   help="PPISP learning rate")
    p.add_argument("--lr_geom",     type=float, default=1e-4,
                   help="Geometry (vertex offsets) learning rate")
    p.add_argument("--warmup",      type=int,   default=500,
                   help="Warmup iterations (PPISP only, texture frozen)")
    p.add_argument("--geom_warmup", type=int,   default=1000,
                   help="Warmup iterations before geometry updates start")
    p.add_argument("--learn_geometry", action="store_true",
                   help="Enable differentiable mesh geometry updates")
    p.add_argument("--geom_smooth", type=float, default=1e-2,
                   help="Geometry smoothness weight (higher = smoother, less rough deformations)")
    p.add_argument("--max_vertex_offset", type=float, default=0,
                   help="Max absolute vertex displacement (scene units); set <=0 to disable clamp")
    p.add_argument("--no_weld_geometry", action="store_true",
                   help="Disable welding duplicate-position vertices for geometry optimization")
    p.add_argument("--no_vignette", action="store_true",
                   help="Disable per-camera vignette learning")
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
    p.add_argument("--amp", action="store_true",
                   help="Enable mixed precision acceleration on CUDA")
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["fp16", "bf16"],
                   help="AMP dtype to use when --amp is enabled")
    p.add_argument("--no_tf32", action="store_true",
                   help="Disable TF32 matmul/conv acceleration on CUDA")
    return p.parse_args()


def main():
    args = parse_args()

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
        cfg.lr_texture      = args.lr_tex
        cfg.lr_ppisp        = args.lr_ppisp
        cfg.lr_geometry     = args.lr_geom
        cfg.learn_vignette  = not args.no_vignette
        cfg.learn_geometry  = args.learn_geometry
        cfg.geometry_warmup_iters = args.geom_warmup
        cfg.geom_smooth_weight = args.geom_smooth
        cfg.max_vertex_offset     = (args.max_vertex_offset
                         if args.max_vertex_offset > 0 else None)
        cfg.weld_geometry_vertices = not args.no_weld_geometry
        cfg.live_view       = args.live_view
        cfg.live_view_every = args.live_view_every
        cfg.live_view_camera = args.live_view_cam
        cfg.live_view_max_size = args.live_view_max_size
        cfg.use_amp         = args.amp
        cfg.amp_dtype       = args.amp_dtype
        cfg.use_tf32        = not args.no_tf32
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
        device=str(device),
    )
    scene.load_gt_images()

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
    cfg.lr_texture      = args.lr_tex
    cfg.lr_ppisp        = args.lr_ppisp
    cfg.lr_geometry     = args.lr_geom
    cfg.learn_vignette  = not args.no_vignette
    cfg.learn_geometry  = args.learn_geometry
    cfg.geometry_warmup_iters = args.geom_warmup
    cfg.geom_smooth_weight = args.geom_smooth
    cfg.max_vertex_offset     = (args.max_vertex_offset
                                 if args.max_vertex_offset > 0 else None)
    cfg.weld_geometry_vertices = not args.no_weld_geometry
    cfg.live_view       = args.live_view
    cfg.live_view_every = args.live_view_every
    cfg.live_view_camera = args.live_view_cam
    cfg.live_view_max_size = args.live_view_max_size
    cfg.use_amp         = args.amp
    cfg.amp_dtype       = args.amp_dtype
    cfg.use_tf32        = not args.no_tf32
    cfg.device          = str(device)

    trainer = TexturePPISPTrainer(scene, cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()
    trainer.export_results()


if __name__ == "__main__":
    main()