#!/usr/bin/env python3
"""
Rendering pipeline diagnostics.

Checks each stage independently and saves debug images at every step:

  Stage 1 — Camera poses    : project mesh vertices, draw wireframe overlay on GT
  Stage 2 — Rasterization   : triangle coverage mask, depth buffer, face ID map
  Stage 3 — UV mapping      : UV coordinates visualised as colour (R=U, G=V)
  Stage 4 — Texture sample  : raw texture lookup before PPISP
  Stage 5 — PPISP           : after ISP correction
  Stage 6 — Diff vs GT      : absolute difference image (bright = large error)

Run on a handful of cameras to quickly spot which stage is broken.

Usage:
  python diagnose_rendering.py --scene <scene> --mesh <mesh> --output diagOut
  python diagnose_rendering.py --scene <scene> --mesh <mesh> --output diagOut --cameras 0 5 10
"""

import argparse, os, sys, math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ helpers

def save(img_tensor, path):
    """Save (H,W,3) or (H,W,1) float32 [0,1] tensor as PNG."""
    arr = img_tensor.detach().cpu().float().numpy()
    arr = (arr * 255).clip(0, 255).astype("uint8")
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    Image.fromarray(arr).save(path)


def label(img_pil, text):
    """Stamp a text label on a PIL image."""
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle([0, 0, len(text)*7+4, 14], fill=(0, 0, 0))
    draw.text((2, 1), text, fill=(255, 255, 0))
    return img_pil


def hstack(*pils):
    """Horizontally concatenate PIL images (resize to same height)."""
    h = max(p.height for p in pils)
    resized = [p.resize((int(p.width * h / p.height), h), Image.BILINEAR)
               if p.height != h else p for p in pils]
    w = sum(p.width for p in resized)
    out = Image.new("RGB", (w, h))
    x = 0
    for p in resized:
        out.paste(p, (x, 0))
        x += p.width
    return out


def colormap_depth(depth: torch.Tensor) -> torch.Tensor:
    """(H,W) depth → (H,W,3) false-colour in [0,1]."""
    d = depth.float()
    valid = d > 0
    if valid.any():
        dmin = d[valid].min()
        dmax = d[valid].max()
        d = torch.where(valid, (d - dmin) / (dmax - dmin + 1e-8),
                        torch.zeros_like(d))
    # Jet-like: blue=near, red=far
    r = torch.clamp(1.5 - abs(d * 4 - 3), 0, 1)
    g = torch.clamp(1.5 - abs(d * 4 - 2), 0, 1)
    b = torch.clamp(1.5 - abs(d * 4 - 1), 0, 1)
    return torch.stack([r, g, b], dim=-1)


# ------------------------------------------------------------------ stage functions

def stage1_pose(view, vertices, device):
    """Project vertices → pixel space, draw as dots on GT image."""
    R = view.R.to(device); t = view.t.to(device); K = view.K.to(device)
    pts_cam = (R @ vertices.T).T + t
    depth   = pts_cam[:, 2]
    pts_img = (K @ pts_cam.T).T
    uv      = pts_img[:, :2] / pts_img[:, 2:3].clamp(min=1e-6)

    # Only in-front, in-frame vertices
    mask = (depth > 0) & \
           (uv[:, 0] >= 0) & (uv[:, 0] < view.W) & \
           (uv[:, 1] >= 0) & (uv[:, 1] < view.H)

    img = view.gt_image.clone() if view.gt_image is not None \
          else torch.zeros(view.H, view.W, 3)
    img_np  = (img.numpy() * 255).clip(0, 255).astype("uint8")
    pil     = Image.fromarray(img_np)
    draw    = ImageDraw.Draw(pil)

    uv_vis  = uv[mask].cpu().numpy()
    # Subsample to max 5000 dots to keep it readable
    step = max(1, len(uv_vis) // 5000)
    for (x, y) in uv_vis[::step]:
        draw.ellipse([x-1, y-1, x+1, y+1], fill=(0, 255, 0))

    n_vis = mask.sum().item()
    pct   = 100 * n_vis / len(vertices)
    info  = (f"Vertices projected: {n_vis:,}/{len(vertices):,} ({pct:.1f}% visible)\n"
             f"Depth range: {depth[mask].min():.2f} – {depth[mask].max():.2f}")
    return pil, info


def stage2_raster(view, vertices, faces, uvs, rasterizer, device):
    """Rasterize → coverage mask, depth map, UV visualisation."""
    import nvdiffrast.torch as dr
    from texture_optimizer.renderer import to_clip_space

    R = view.R.to(device); t = view.t.to(device); K = view.K.to(device)
    verts_clip = to_clip_space(vertices, R, t, K, view.W, view.H)
    faces_i32  = faces.to(torch.int32)

    with torch.no_grad():
        rast, rast_db = dr.rasterize(rasterizer._nvdr.glctx, verts_clip,
                                     faces_i32, resolution=[view.H, view.W],
                                     grad_db=True)

    # Coverage mask
    coverage = (rast[0, :, :, 3] > 0).float().unsqueeze(-1).expand(-1, -1, 3)

    # Depth (w component of clip / face coverage)
    pts_cam = (R @ vertices.T).T + t
    depth_v = pts_cam[:, 2]
    uv_attr = uvs.unsqueeze(0)
    with torch.no_grad():
        depth_attr = depth_v.unsqueeze(0).unsqueeze(-1)           # (1,V,1)
        depth_i, _ = dr.interpolate(depth_attr, rast, faces_i32)  # (1,H,W,1)
    depth_map = colormap_depth(depth_i[0, :, :, 0] * (rast[0, :, :, 3] > 0).float())

    # UV as colour
    with torch.no_grad():
        texc, _ = dr.interpolate(uv_attr, rast, faces_i32)        # (1,H,W,2)
    uv_vis = torch.cat([texc[0], torch.zeros(view.H, view.W, 1, device=device)], dim=-1)
    uv_vis = uv_vis * (rast[0, :, :, 3:4] > 0).float()

    n_covered = (rast[0, :, :, 3] > 0).sum().item()
    pct = 100 * n_covered / (view.H * view.W)
    info = (f"Covered pixels: {n_covered:,}/{view.H*view.W:,} ({pct:.1f}%)\n"
            f"UV range: [{texc[0,...,0].min():.3f}, {texc[0,...,0].max():.3f}] × "
            f"[{texc[0,...,1].min():.3f}, {texc[0,...,1].max():.3f}]")
    return coverage, depth_map, uv_vis, rast, rast_db, info


def stage3_texture(view, vertices, faces, uvs, texture, rast_tuple, device):
    """Sample texture at interpolated UVs."""
    import nvdiffrast.torch as dr
    from texture_optimizer.renderer import to_clip_space

    R = view.R.to(device); t = view.t.to(device); K = view.K.to(device)
    faces_i32 = faces.to(torch.int32)
    uv_attr   = uvs.unsqueeze(0)

    rast, rast_db = rast_tuple
    with torch.no_grad():
        texc, texc_db = dr.interpolate(uv_attr, rast, faces_i32,
                                       rast_db=rast_db, diff_attrs="all")
        tex_hwc = torch.clamp(texture.tex, 0, 1).permute(0, 2, 3, 1).contiguous()
        try:
            color = dr.texture(tex_hwc, texc.contiguous(),
                               uv_da=texc_db.contiguous(),
                               filter_mode="linear-mipmap-linear")
        except Exception:
            color = dr.texture(tex_hwc, texc.contiguous(), filter_mode="linear")
        color = dr.antialias(color, rast, to_clip_space(vertices, R, t, K, view.W, view.H), faces_i32)

    tex_render = color[0].clamp(0, 1)                              # (H,W,3)
    info = (f"Texture size: {texture.tex.shape[2]}×{texture.tex.shape[3]}\n"
            f"Render value range: [{tex_render.min():.3f}, {tex_render.max():.3f}]")
    return tex_render, info


def stage4_ppisp(view, hdr, ppisp, device):
    """Apply PPISP and show before/after."""
    if ppisp.learn_vignette and ppisp.r2.shape != torch.Size([view.H, view.W]):
        ys = torch.linspace(-1, 1, view.H, device=device)
        xs = torch.linspace(-1, 1, view.W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        ppisp.r2 = gx**2 + gy**2

    with torch.no_grad():
        ldr = ppisp(hdr, view.cam_idx)

    p = ppisp.get_params_dict(view.cam_idx)
    info = (f"exposure={p['exposure']:.3f}  gamma={p['gamma']:.2f}  "
            f"wb=({p['wb_r']:.3f},{p['wb_g']:.3f},{p['wb_b']:.3f})\n"
            f"brightness={p['brightness']:+.3f}  contrast={p['contrast']:.3f}")
    return ldr, info


def stage5_diff(render, gt):
    """Absolute difference map."""
    if gt is None:
        return None, "No GT image"
    diff = (render - gt.float()).abs()
    # Scale diff for visibility (5× amplification)
    diff_vis = (diff * 5).clamp(0, 1)
    mse  = diff.pow(2).mean().item()
    mae  = diff.mean().item()
    info = f"MAE={mae:.4f}  MSE={mse:.6f}  PSNR={-10*math.log10(mse+1e-10):.1f}dB"
    return diff_vis, info


# ------------------------------------------------------------------ main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene",    type=str, required=True)
    p.add_argument("--mesh",     type=str, default=None)
    p.add_argument("--output",   type=str, default="diagOut")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--scale",    type=float, default=0.5)
    p.add_argument("--tex_res",  type=int, default=2048)
    p.add_argument("--cameras",  type=int, nargs="+", default=None,
                   help="Camera indices to diagnose (default: 0, N//4, N//2, 3N//4, N-1)")
    p.add_argument("--device",   type=str, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}\n")

    from texture_optimizer.dataset  import ColmapScene
    from texture_optimizer.renderer import TextureMap, Rasterizer
    from texture_optimizer.ppisp    import PPISPParams

    # Load scene
    scene = ColmapScene(scene_root=args.scene, image_scale=args.scale,
                        mesh_path=args.mesh, device=str(device))
    scene.load_gt_images()

    vertices = scene.mesh.vertices.to(device)
    faces    = scene.mesh.faces.to(device)
    uvs      = scene.mesh.uvs.to(device)

    # Texture
    if args.checkpoint:
        ckpt     = torch.load(args.checkpoint, map_location=device)
        ts       = ckpt["texture"]["tex"].shape
        texture  = TextureMap(ts[2], ts[3]).to(device)
        texture.load_state_dict(ckpt["texture"])
        print(f"Loaded texture from checkpoint: {ts[2]}×{ts[3]}")
    else:
        texture = TextureMap(args.tex_res, args.tex_res,
                             init_image=scene.mesh.tex_image).to(device)
        print(f"Using initial texture: {args.tex_res}×{args.tex_res}")

    # PPISP
    v0    = scene.views[0]
    ppisp = PPISPParams(len(scene), v0.W, v0.H).to(device)
    if args.checkpoint:
        ppisp.load_state_dict(torch.load(args.checkpoint,
                                         map_location=device)["ppisp"])
        print("Loaded PPISP from checkpoint")

    # Rasterizer — needs nvdiffrast for stages 2+
    rasterizer = Rasterizer(device)

    # Choose cameras
    n = len(scene)
    cam_indices = args.cameras or [0, n//4, n//2, 3*n//4, n-1]
    cam_indices = sorted(set(min(i, n-1) for i in cam_indices))
    print(f"Diagnosing cameras: {cam_indices}\n")

    os.makedirs(args.output, exist_ok=True)

    for ci in cam_indices:
        view  = scene.views[ci]
        cdir  = os.path.join(args.output, f"cam_{ci:04d}")
        os.makedirs(cdir, exist_ok=True)
        print(f"─── Camera {ci:04d}  ({view.W}×{view.H})  {view.image_path}")
        summary_panels = []

        # ---- Stage 1: Pose ----
        pil1, info1 = stage1_pose(view, vertices, device)
        label(pil1, "S1: pose projection")
        pil1.save(os.path.join(cdir, "s1_pose.png"))
        summary_panels.append(pil1)
        print(f"  S1 pose:     {info1}")

        import nvdiffrast.torch as dr

        # ---- Stage 2: Rasterization ----
        cov, dep, uv_vis, rast, rast_db, info2 = stage2_raster(
            view, vertices, faces, uvs, rasterizer, device)
        save(cov,    os.path.join(cdir, "s2_coverage.png"))
        save(dep,    os.path.join(cdir, "s2_depth.png"))
        save(uv_vis, os.path.join(cdir, "s2_uvmap.png"))
        rast_tuple = (rast, rast_db)
        summary_panels += [
            label(Image.fromarray((cov.cpu().numpy()*255).astype("uint8")),   "S2: coverage"),
            label(Image.fromarray((dep.cpu().numpy()*255).astype("uint8")),   "S2: depth"),
            label(Image.fromarray((uv_vis.cpu().numpy()*255).astype("uint8")),"S2: UV"),
        ]
        print(f"  S2 raster:   {info2}")

        # ---- Stage 3: Texture ----
        tex_render, info3 = stage3_texture(
            view, vertices, faces, uvs, texture, rast_tuple, device)
        save(tex_render, os.path.join(cdir, "s3_texture.png"))
        summary_panels.append(
            label(Image.fromarray((tex_render.cpu().numpy()*255).astype("uint8")),
                  "S3: texture"))
        print(f"  S3 texture:  {info3}")

        # ---- Stage 4: PPISP ----
        ldr, info4 = stage4_ppisp(view, tex_render, ppisp, device)
        save(ldr, os.path.join(cdir, "s4_ppisp.png"))
        summary_panels.append(
            label(Image.fromarray((ldr.cpu().numpy()*255).astype("uint8")),
                  "S4: PPISP"))
        print(f"  S4 PPISP:    {info4}")

        # ---- Stage 5: Diff vs GT ----
        gt = view.gt_image.to(device) if view.gt_image is not None else None
        diff, info5 = stage5_diff(ldr, gt)
        if diff is not None:
            save(diff, os.path.join(cdir, "s5_diff.png"))
            if gt is not None:
                save(gt,  os.path.join(cdir, "s5_gt.png"))
            summary_panels += [
                label(Image.fromarray((gt.cpu().numpy()*255).astype("uint8")),
                      "GT") if gt is not None else Image.new("RGB",(view.W,view.H)),
                label(Image.fromarray((diff.cpu().numpy()*255).astype("uint8")),
                      "S5: diff×5"),
            ]
        print(f"  S5 diff:     {info5}")

        # ---- Summary strip ----
        if summary_panels:
            strip = hstack(*summary_panels)
            strip.save(os.path.join(cdir, "summary.png"))
            print(f"  Saved summary strip → {cdir}/summary.png")

        print()

    print(f"Diagnostics saved to: {os.path.abspath(args.output)}/")
    print("\nWhat to look for:")
    print("  s1_pose.png    — green dots should cover the mesh surface in the GT image")
    print("                   If dots are off, camera poses (R/t/K) are wrong")
    print("  s2_coverage.png— white = rendered pixels; should match mesh silhouette")
    print("                   If mostly black, mesh may be behind the camera")
    print("  s2_uvmap.png   — smooth R=U / G=V gradient; seams OK, uniform black = missing UVs")
    print("  s3_texture.png — should look like a plausible (possibly dark/bright) scene")
    print("                   If grey/uniform, texture atlas isn't loading correctly")
    print("  s4_ppisp.png   — exposure-corrected version; should match GT brightness")
    print("  s5_diff.png    — bright areas = large error; should be dark after training")


if __name__ == "__main__":
    main()