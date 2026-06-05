#!/usr/bin/env python3
"""
Test forward rendering pass.

Loads the mesh + cameras, renders every view with the CURRENT texture
(either the initial loaded texture or a checkpoint), and saves to disk.
No training, no gradients — pure forward pass only.

Usage:
  # Render with initial texture (no checkpoint)
  python test_rendering.py --scene <scene> --mesh <mesh.ply> --output testRendering

  # Render from a trained checkpoint
  python test_rendering.py --scene <scene> --mesh <mesh.ply> \
      --checkpoint outputs/checkpoint_005000.pt --output testRendering

  # Also save side-by-side GT comparison
  python test_rendering.py --scene <scene> --mesh <mesh.ply> \
      --output testRendering --compare
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Forward rendering test")
    p.add_argument("--scene",      type=str, required=True)
    p.add_argument("--mesh",       type=str, default=None)
    p.add_argument("--output",     type=str, default="testRendering")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional trained checkpoint to load texture/PPISP from")
    p.add_argument("--scale",      type=float, default=0.5,
                   help="Image downscale factor (default 0.5)")
    p.add_argument("--tex_res",    type=int, default=2048,
                   help="Texture resolution (ignored if checkpoint provided)")
    p.add_argument("--compare",    action="store_true",
                   help="Save GT | render side-by-side comparison images")
    p.add_argument("--no_ppisp",   action="store_true",
                   help="Skip PPISP — render raw texture without ISP correction")
    p.add_argument("--device",     type=str, default=None)
    p.add_argument("--max_cameras",type=int, default=None)
    return p.parse_args()


def main():
    args  = parse_args()
    import torch
    import numpy as np
    from PIL import Image

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # ------------------------------------------------------------------ load scene
    from texture_optimizer.dataset  import ColmapScene
    from texture_optimizer.renderer import TextureMap, Rasterizer
    from texture_optimizer.ppisp    import PPISPParams

    scene = ColmapScene(
        scene_root=args.scene,
        image_scale=args.scale,
        max_cameras=args.max_cameras,
        mesh_path=args.mesh,
        device=device,
    )
    scene.load_gt_images()

    vertices = scene.mesh.vertices.to(device)
    faces    = scene.mesh.faces.to(device)
    uvs      = scene.mesh.uvs.to(device)

    # ------------------------------------------------------------------ texture
    if args.checkpoint:
        # Infer tex resolution from checkpoint
        ckpt    = torch.load(args.checkpoint, map_location=device)
        tex_state = ckpt["texture"]
        tex_shape = tex_state["tex"].shape          # (1, 3, H, W)
        tex_H, tex_W = tex_shape[2], tex_shape[3]
        print(f"Checkpoint texture: {tex_H}×{tex_W}")
        texture = TextureMap(tex_H, tex_W).to(device)
        texture.load_state_dict(tex_state)
    else:
        tex_H = tex_W = args.tex_res
        texture = TextureMap(tex_H, tex_W,
                             init_image=scene.mesh.tex_image).to(device)
        print(f"Using initial texture: {tex_H}×{tex_W}")

    # ------------------------------------------------------------------ PPISP
    v0    = scene.views[0]
    ppisp = PPISPParams(
        num_cameras=len(scene),
        image_width=v0.W, image_height=v0.H,
        learn_vignette=True,
    ).to(device)

    if args.checkpoint and not args.no_ppisp:
        ckpt = torch.load(args.checkpoint, map_location=device)
        ppisp.load_state_dict(ckpt["ppisp"])
        print("Loaded PPISP from checkpoint")
    else:
        print("Using identity PPISP (no correction)")

    # ------------------------------------------------------------------ rasterizer
    rasterizer = Rasterizer(device)
    faces_i32  = faces.to(torch.int32)

    # ------------------------------------------------------------------ output dirs
    os.makedirs(args.output, exist_ok=True)
    if args.compare:
        cmp_dir = os.path.join(args.output, "compare")
        os.makedirs(cmp_dir, exist_ok=True)

    # ------------------------------------------------------------------ render loop
    print(f"\nRendering {len(scene)} cameras → {args.output}/\n")
    t0 = time.time()

    with torch.no_grad():
        for view in scene.views:
            R = view.R.to(device)
            t = view.t.to(device)
            K = view.K.to(device)

            # Forward render
            hdr = rasterizer.render(
                vertices, faces_i32, uvs,
                texture, R, t, K, view.W, view.H,
            )                                           # (H, W, 3)

            if args.no_ppisp:
                ldr = torch.clamp(hdr, 0, 1)
            else:
                # Update vignette grid if needed
                if ppisp.learn_vignette and \
                        ppisp.r2.shape != torch.Size([view.H, view.W]):
                    ys = torch.linspace(-1, 1, view.H, device=device)
                    xs = torch.linspace(-1, 1, view.W, device=device)
                    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
                    ppisp.r2 = gx**2 + gy**2
                ldr = ppisp(hdr, view.cam_idx)          # (H, W, 3)

            # Save render
            render_np  = (ldr.cpu().numpy() * 255).clip(0, 255).astype("uint8")
            render_pil = Image.fromarray(render_np)
            out_path   = os.path.join(args.output, f"cam_{view.cam_idx:04d}.png")
            render_pil.save(out_path)

            # Save GT side-by-side if requested
            if args.compare and view.gt_image is not None:
                gt_np  = (view.gt_image.numpy() * 255).clip(0, 255).astype("uint8")
                gt_pil = Image.fromarray(gt_np)

                # Resize GT to match render size (in case of scale mismatch)
                if gt_pil.size != render_pil.size:
                    gt_pil = gt_pil.resize(render_pil.size, Image.BILINEAR)

                # Stack horizontally: GT | render
                cmp = Image.new("RGB", (view.W * 2, view.H))
                cmp.paste(gt_pil,    (0,       0))
                cmp.paste(render_pil,(view.W,  0))

                # Add a thin divider line
                for y in range(view.H):
                    cmp.putpixel((view.W - 1, y), (255, 0, 0))
                    cmp.putpixel((view.W,     y), (255, 0, 0))

                cmp.save(os.path.join(cmp_dir, f"cmp_{view.cam_idx:04d}.png"))

            elapsed = time.time() - t0
            fps     = (view.cam_idx + 1) / elapsed
            print(f"  cam {view.cam_idx:04d}  {view.W}×{view.H}  "
                  f"({fps:.1f} fps)  → {out_path}")

    elapsed = time.time() - t0
    print(f"\nDone. {len(scene)} renders in {elapsed:.1f}s  "
          f"({len(scene)/elapsed:.1f} fps avg)")
    print(f"Renders saved to: {os.path.abspath(args.output)}/")
    if args.compare:
        print(f"Comparisons:      {os.path.abspath(cmp_dir)}/")


if __name__ == "__main__":
    main()