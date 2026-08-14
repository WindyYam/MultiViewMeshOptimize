#!/usr/bin/env python3
"""
Interactive viewer for a texture_optimizer export directory.

Loads optimized_mesh.ply + optimized_texture.png + optimized_sh_coeff_*.npy
and renders the SH-shaded mesh with a free orbit/fly/pan camera.

Usage:
    python view_model.py --output outputs/
    python view_model.py --output outputs/ --max_tex_size 2048 --up y
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="SH texture/mesh viewer")
    p.add_argument("--output", type=str, required=True,
                   help="Export directory produced by run_optimizer.py (contains optimized_mesh.ply etc.)")
    p.add_argument("--max_tex_size", type=int, default=4096,
                   help="Downsample any texture layer whose long side exceeds this, to bound VRAM use")
    p.add_argument("--no_flip_v", action="store_true",
                   help=(
                       "Disable the default V-coordinate flip. exporter.py writes "
                       "texture_v = 1 - v into the PLY for generic viewer compatibility, "
                       "but trainer.py's render path samples the un-flipped UVs directly, "
                       "so the viewer flips by default to match the trained appearance."
                   ))
    p.add_argument("--up", type=str, default="z", choices=["y", "z"],
                   help="World up axis for the default camera orientation")
    p.add_argument("--window_w", type=int, default=1280)
    p.add_argument("--window_h", type=int, default=800)
    return p.parse_args()


def main():
    args = parse_args()
    from viewer.app import run
    run(
        output_dir=args.output,
        max_tex_size=args.max_tex_size,
        flip_v=not args.no_flip_v,
        up=args.up,
        window_size=(args.window_w, args.window_h),
    )


if __name__ == "__main__":
    main()
