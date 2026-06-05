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
    p.add_argument("--scale",       type=float, default=0.5,
                   help="GT image downscale factor (0.5 = half-res, faster)")
    p.add_argument("--tex_res",     type=int,   default=1024,
                   help="Output texture resolution (square, e.g. 2048 or 8192)")
    p.add_argument("--lr_tex",      type=float, default=1e-3,
                   help="Texture learning rate")
    p.add_argument("--lr_ppisp",    type=float, default=5e-3,
                   help="PPISP learning rate")
    p.add_argument("--warmup",      type=int,   default=500,
                   help="Warmup iterations (PPISP only, texture frozen)")
    p.add_argument("--no_vignette", action="store_true",
                   help="Disable per-camera vignette learning")
    p.add_argument("--resume",      type=str,   default=None,
                   help="Resume from checkpoint file")
    p.add_argument("--device",      type=str,   default=None,
                   help="cuda or cpu (auto-detected if omitted)")
    p.add_argument("--max_cameras", type=int,   default=None,
                   help="Cap number of cameras loaded (useful for quick tests)")
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
        cfg.learn_vignette  = not args.no_vignette
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
    cfg.tex_H = cfg.tex_W = args.tex_res
    cfg.warmup_iters    = args.warmup
    cfg.lr_texture      = args.lr_tex
    cfg.lr_ppisp        = args.lr_ppisp
    cfg.learn_vignette  = not args.no_vignette
    cfg.device          = str(device)

    trainer = TexturePPISPTrainer(scene, cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()
    trainer.export_results()


if __name__ == "__main__":
    main()