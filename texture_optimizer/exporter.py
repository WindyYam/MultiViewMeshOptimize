import os
import json

import numpy as np
from PIL import Image, ImageDraw


class TextureExportMixin:

    def _export_mesh_ply(self, path: str, texture_file: str = "optimized_texture.png"):
        """
        Export textured mesh as binary little-endian PLY.
        Uses per-vertex UV properties and a TextureFile header comment.
        """
        texture_ref = texture_file or "optimized_texture.png"
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

        # Weld duplicated vertices by rounded XYZ to reduce seam duplication in output.
        # UVs remain face-corner attributes (texcoord list), so seam UVs are preserved.
        weld_decimals = max(0, int(getattr(self.cfg, "weld_position_decimals", 6)))
        weld_keys = np.round(verts, decimals=weld_decimals)
        _, unique_first_idx, inverse = np.unique(
            weld_keys, axis=0, return_index=True, return_inverse=True
        )
        verts_weld = verts[unique_first_idx].astype(np.float32, copy=False)
        faces_weld = inverse[faces].astype(np.int32, copy=False)

        # Keep only non-degenerate triangles after welding.
        valid_face_mask = (
            (faces_weld[:, 0] != faces_weld[:, 1])
            & (faces_weld[:, 1] != faces_weld[:, 2])
            & (faces_weld[:, 2] != faces_weld[:, 0])
        )
        faces_out = faces_weld[valid_face_mask]

        vertex_block = verts_weld.astype(np.float32, copy=False)

        # Face-level UV list layout: [u0, v0, u1, v1, u2, v2] per triangle.
        tri_uv = uvs_out[faces].reshape(-1, 6).astype(np.float32, copy=False)
        tri_uv_out = tri_uv[valid_face_mask]

        face_dtype = np.dtype([
            ("n", np.uint8),
            ("idx", "<i4", (3,)),
            ("tc_n", np.uint8),
            ("tc", "<f4", (6,)),
        ])
        face_block = np.empty(faces_out.shape[0], dtype=face_dtype)
        face_block["n"] = 3
        face_block["idx"] = faces_out
        face_block["tc_n"] = 6
        face_block["tc"] = tri_uv_out

        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"comment TextureFile {texture_ref}\n"
            f"element vertex {vertex_block.shape[0]}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            f"element face {face_block.shape[0]}\n"
            "property list uchar int vertex_indices\n"
            "property list uchar float texcoord\n"
            "end_header\n"
        )

        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            vertex_block.tofile(f)
            face_block.tofile(f)

    def _build_uv_coverage_mask(self, H: int, W: int) -> np.ndarray:
        """
        Rasterize UV triangles into a binary coverage mask in texture space.
        """
        mask_img = Image.new("L", (W, H), 0)
        drawer = ImageDraw.Draw(mask_img)

        uvs = self.uvs.detach().cpu().numpy().astype(np.float32)
        faces = self.faces.detach().cpu().numpy().astype(np.int64)

        xs = np.clip(uvs[:, 0], 0.0, 1.0) * float(W - 1)
        ys = np.clip(uvs[:, 1], 0.0, 1.0) * float(H - 1)

        for f0, f1, f2 in faces:
            poly = [
                (float(xs[f0]), float(ys[f0])),
                (float(xs[f1]), float(ys[f1])),
                (float(xs[f2]), float(ys[f2])),
            ]
            drawer.polygon(poly, fill=255)

        return np.asarray(mask_img, dtype=np.uint8) > 0

    def _dilate_texture_from_mask(self, tex_np: np.ndarray, valid_mask: np.ndarray, pad_px: int) -> np.ndarray:
        """
        Extend valid texels outward so bilinear/mipmap sampling across UV seams
        does not pull black/undefined values from outside UV islands.
        """
        if pad_px <= 0:
            return tex_np

        img = np.ascontiguousarray(tex_np.copy())
        valid = valid_mask.astype(bool).copy()
        H, W = valid.shape
        if H == 0 or W == 0:
            return img

        # 8-neighborhood offsets for dilation.
        neighbors = np.asarray([
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ], dtype=np.int32)

        for _ in range(int(pad_px)):
            valid_f = valid.astype(np.float32, copy=False)

            # Pad once, then gather all 8 shifted neighborhoods in batch.
            valid_pad = np.pad(valid_f, ((1, 1), (1, 1)), mode="constant")
            img_pad = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="constant")

            neigh_valid = np.stack([
                valid_pad[1 + int(dy):1 + int(dy) + H, 1 + int(dx):1 + int(dx) + W]
                for dy, dx in neighbors
            ], axis=0)
            neigh_img = np.stack([
                img_pad[1 + int(dy):1 + int(dy) + H, 1 + int(dx):1 + int(dx) + W, :]
                for dy, dx in neighbors
            ], axis=0)

            count = neigh_valid.sum(axis=0)
            accum = (neigh_img * neigh_valid[..., None]).sum(axis=0)

            fillable = (~valid) & (count > 0)
            if not np.any(fillable):
                break

            img[fillable] = (accum[fillable] / np.maximum(count[fillable, None], 1e-8)).astype(img.dtype)
            valid[fillable] = True

        return img

    def _seam_pad_texture(self, tex_np: np.ndarray) -> np.ndarray:
        pad_px = max(0, int(getattr(self.cfg, "tex_seam_pad_px", 0)))
        if pad_px <= 0:
            return tex_np

        H, W = int(tex_np.shape[0]), int(tex_np.shape[1])
        try:
            valid_mask = self._build_uv_coverage_mask(H, W)
            padded = self._dilate_texture_from_mask(tex_np, valid_mask, pad_px=pad_px)
            print(f"[Export] SeamPad  : {pad_px}px UV dilation")
            return padded
        except Exception as e:
            print(f"[Export] SeamPad  : skipped ({e})")
            return tex_np

    def export_results(self):
        out = self.cfg.output_dir

        # Optimised texture
        tex_np = self.texture.as_image().cpu().numpy()
        tex_np = self._seam_pad_texture(tex_np)
        # Bake average PPISP parameters (exposure, WB, gamma, brightness, contrast)
        try:
            with np.errstate(all='ignore'):
                import torch
                if hasattr(self, 'ppisp') and self.ppisp is not None:
                    with torch.no_grad():
                        exp = float(self.ppisp.exposure().mean().cpu().item())
                        wb = self.ppisp.wb_gains().mean(dim=0).cpu().numpy()  # (3,)
                        gamma = float(self.ppisp.gamma().mean().cpu().item())
                        contrast = float(self.ppisp.contrast().mean().cpu().item())
                        brightness = float(self.ppisp.brightness.mean().cpu().item())

                    # apply exposure and white balance (linear domain)
                    tex_np = tex_np * (exp * wb.reshape((1, 1, 3)))

                    # apply gamma compression (sign-preserving, texture is non-negative)
                    # avoid negative/zero issues
                    tex_np = np.sign(tex_np) * np.power(np.abs(tex_np) + 1e-8, 1.0 / max(1e-8, gamma))

                    # brightness & contrast applied in [0,1] domain
                    tex_np = contrast * (tex_np - 0.5) + 0.5 + brightness
                    # clamp to [0,1]
                    tex_np = np.clip(tex_np, 0.0, 1.0)
                    print(f"[Export] Baked PPISP avg: exp={exp:.3f} gamma={gamma:.3f} wb={wb.tolist()} C={contrast:.3f} B={brightness:+.3f}")
        except Exception as e:
            print(f"[Export] PPISP bake skipped: {e}")
        Image.fromarray((tex_np * 255).astype(np.uint8)).save(
            os.path.join(out, "optimized_texture.png"))
        print(f"[Export] Texture  -> {out}/optimized_texture.png (sRGB-encoded)")

        # PPISP params
        with open(os.path.join(out, "ppisp_params.json"), "w") as f:
            json.dump([self.ppisp.get_params_dict(i)
                       for i in range(len(self.scene))], f, indent=2)
        print(f"[Export] PPISP    -> {out}/ppisp_params.json")

        # Optimized geometry (binary textured PLY)
        mesh_ply = os.path.join(out, "optimized_mesh.ply")
        self._export_mesh_ply(mesh_ply, texture_file="optimized_texture.png")
        print(f"[Export] Mesh     -> {mesh_ply}")

        # Loss curve
        np.save(os.path.join(out, "loss_log.npy"),
                np.array([l["total"] for l in self.loss_log]))

        # Re-render all views
        # render_dir = os.path.join(out, "renders")
        # os.makedirs(render_dir, exist_ok=True)
        # print("[Export] Re-rendering all views ...")
        # with torch.no_grad():
        #     for view in self.scene.views:
        #         if view.gt_image is None:
        #             continue
        #         pred = self.render_view(view).float().cpu().numpy()
        #         Image.fromarray((pred * 255).astype(np.uint8)).save(
        #             os.path.join(render_dir, f"cam_{view.cam_idx:04d}.png"))
        # print(f"[Export] Renders  -> {render_dir}/")