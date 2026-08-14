"""
Interactive viewer for texture_optimizer exports: loads the optimized mesh,
base texture and SH coefficient textures, and renders the SH-shaded result
in real time (moderngl) with a free orbit/fly/pan camera and an imgui panel
to toggle SH bands / inspect individual coefficients.
"""

import time

import glfw
import imgui
import moderngl
import numpy as np
from imgui.integrations.glfw import GlfwRenderer

from .camera import OrbitFlyCamera, perspective
from .data import load_export_bundle
from .sh_math import compute_face_tangent_frames
from .shaders import FRAGMENT_SHADER, VERTEX_SHADER

MAX_COEFFS = 9  # coeff0 (DC) + up to 8 AC (order <= 2)


def _build_vertex_buffer(vertices, faces, uvs):
    T, B, N = compute_face_tangent_frames(vertices, faces, uvs)
    idx = faces.reshape(-1).astype(np.int64)
    pos_dup = vertices[idx]
    uv_dup = uvs[idx]
    T_dup = np.repeat(T, 3, axis=0)
    B_dup = np.repeat(B, 3, axis=0)
    N_dup = np.repeat(N, 3, axis=0)
    data = np.concatenate([pos_dup, uv_dup, T_dup, B_dup, N_dup], axis=1)
    return np.ascontiguousarray(data, dtype=np.float32)


class Viewer:
    def __init__(self, output_dir, max_tex_size=4096, flip_v=True, up="z", window_size=(1280, 800)):
        self.bundle = load_export_bundle(output_dir, max_tex_size=max_tex_size)
        self.flip_v = flip_v

        bbox_min = self.bundle.vertices.min(axis=0)
        bbox_max = self.bundle.vertices.max(axis=0)
        center = (bbox_min + bbox_max) / 2.0
        diag = float(np.linalg.norm(bbox_max - bbox_min))
        self.scene_diag = max(diag, 1e-3)

        up_vec = (0, 0, 1) if up == "z" else (0, 1, 0)
        self.camera = OrbitFlyCamera(target=center, distance=self.scene_diag * 1.2, up=up_vec)
        self._default_target = center.copy()
        self._default_distance = self.scene_diag * 1.2

        self.enable = [True] * MAX_COEFFS
        self.view_mode = 0  # 0=composite, 1=isolate, 2=raw
        self.selected_coeff = 1 if self.bundle.sh_order > 0 else 0
        self.cull_backfaces = False
        self.use_mipmaps = True

        self._last_cursor = None
        self._scroll_accum = 0.0
        self._last_frame_time = None

        self._init_window(window_size)
        self._init_gl()

    # ------------------------------------------------------------ setup

    def _init_window(self, window_size):
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

        self.window = glfw.create_window(window_size[0], window_size[1], "SH Texture Viewer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("glfw.create_window() failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)

        self.ctx = moderngl.create_context()

        imgui.create_context()
        self.impl = GlfwRenderer(self.window)

        # Chain the scroll callback: imgui needs it for GUI scrolling, we
        # need it for camera zoom (applied only when the GUI isn't hovered).
        imgui_scroll_cb = self.impl.scroll_callback

        def combined_scroll_callback(window, x_offset, y_offset):
            imgui_scroll_cb(window, x_offset, y_offset)
            self._scroll_accum += y_offset

        glfw.set_scroll_callback(self.window, combined_scroll_callback)

    def _init_gl(self):
        ctx = self.ctx
        ctx.enable(moderngl.DEPTH_TEST)

        self.program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

        vertex_data = _build_vertex_buffer(self.bundle.vertices, self.bundle.faces, self.bundle.uvs)
        self.n_draw_verts = vertex_data.shape[0]
        vbo = ctx.buffer(vertex_data.tobytes())
        self.vao = ctx.vertex_array(
            self.program,
            [(vbo, "3f 2f 3f 3f 3f", "in_pos", "in_uv", "in_T", "in_B", "in_N")],
        )

        self.dummy_tex = ctx.texture((1, 1), 3, data=np.zeros((1, 1, 3), dtype=np.float32).tobytes(), dtype="f4")

        self.textures = []
        for i in range(MAX_COEFFS):
            if i < len(self.bundle.textures):
                arr = self.bundle.textures[i]
                if self.flip_v:
                    arr = np.flipud(arr)
                arr = np.ascontiguousarray(arr, dtype=np.float32)
                tex = ctx.texture((arr.shape[1], arr.shape[0]), 3, data=arr.tobytes(), dtype="f4")
                tex.repeat_x = False
                tex.repeat_y = False
                tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                tex.anisotropy = 16.0
                tex.build_mipmaps()
            else:
                tex = None
            self.textures.append(tex)

        for i in range(MAX_COEFFS):
            self.program[f"texCoeff{i}"].value = i

        self.n_available = len(self.bundle.textures)
        for i in range(self.n_available, MAX_COEFFS):
            self.enable[i] = False

    def _apply_mipmap_setting(self):
        for tex in self.textures:
            if tex is None:
                continue
            if self.use_mipmaps:
                tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            else:
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    # ------------------------------------------------------------ callbacks

    def _on_resize(self, window, width, height):
        if width > 0 and height > 0:
            self.ctx.viewport = (0, 0, width, height)

    # ------------------------------------------------------------ per-frame input

    def _process_camera_input(self, dt):
        io = imgui.get_io()
        capture_mouse = io.want_capture_mouse
        capture_kbd = io.want_capture_keyboard

        cursor = glfw.get_cursor_pos(self.window)
        if self._last_cursor is None:
            self._last_cursor = cursor
        dx = cursor[0] - self._last_cursor[0]
        dy = cursor[1] - self._last_cursor[1]
        self._last_cursor = cursor

        if not capture_mouse:
            left = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
            right = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
            middle = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
            if left:
                self.camera.orbit(dx, dy)
            elif right or middle:
                self.camera.pan(dx, dy)

            if self._scroll_accum != 0.0:
                self.camera.zoom(self._scroll_accum)

        self._scroll_accum = 0.0

        if not capture_kbd:
            fwd = 0.0
            strafe = 0.0
            vert = 0.0
            if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
                fwd += 1.0
            if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
                fwd -= 1.0
            if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
                strafe += 1.0
            if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
                strafe -= 1.0
            if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
                vert += 1.0
            if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
                vert -= 1.0
            boost = (
                glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
                or glfw.get_key(self.window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
            )
            self.camera.fly(fwd, strafe, vert, dt, boost=boost)
            if glfw.get_key(self.window, glfw.KEY_R) == glfw.PRESS:
                self.camera.target = self._default_target.copy()
                self.camera.distance = self._default_distance

    # ------------------------------------------------------------ GUI

    def _build_gui(self):
        imgui.new_frame()
        imgui.begin("SH Viewer")
        imgui.text(
            f"{self.bundle.vertices.shape[0]:,} verts / {self.bundle.faces.shape[0]:,} faces\n"
            f"SH order {self.bundle.sh_order}  ({self.n_available} texture layer(s) loaded)"
        )
        imgui.separator()

        mode_labels = ["Composite (sum of enabled bands)", "Isolate single band", "Raw coefficient map"]
        changed, self.view_mode = imgui.combo("Render mode", self.view_mode, mode_labels)

        if self.view_mode == 0:
            imgui.text("Enabled coefficients:")
            if imgui.button("Order 0 only (DC)"):
                self.enable = [i == 0 for i in range(MAX_COEFFS)]
            imgui.same_line()
            if imgui.button("Order <= 1"):
                self.enable = [i < 4 for i in range(MAX_COEFFS)]
            imgui.same_line()
            if imgui.button("Order <= 2 (full)"):
                self.enable = [i < self.n_available for i in range(MAX_COEFFS)]

            for i in range(self.n_available):
                _, self.enable[i] = imgui.checkbox(f"c{i}" + (" (DC)" if i == 0 else ""), self.enable[i])
                if i != self.n_available - 1 and (i + 1) % 4 != 0:
                    imgui.same_line()
        else:
            lo = 1 if self.view_mode == 1 else 0
            _, self.selected_coeff = imgui.slider_int(
                "Coefficient index", max(self.selected_coeff, lo), lo, max(self.n_available - 1, lo)
            )

        imgui.separator()
        mip_changed, self.use_mipmaps = imgui.checkbox(
            "Use mipmaps (off = sharper but more shimmer at distance)", self.use_mipmaps
        )
        if mip_changed:
            self._apply_mipmap_setting()
        _, self.cull_backfaces = imgui.checkbox("Cull back faces", self.cull_backfaces)
        if imgui.button("Reset view"):
            self.camera.target = self._default_target.copy()
            self.camera.distance = self._default_distance
        imgui.text(f"Fly speed: {self.camera.fly_speed:.3g}  (scroll to zoom, Shift to boost)")
        imgui.text("Left-drag: orbit | Right/middle-drag: pan | WASDQE: fly | R: reset")

        imgui.end()
        imgui.render()

    # ------------------------------------------------------------ draw

    def _draw_scene(self):
        ctx = self.ctx
        ctx.clear(0.08, 0.08, 0.09)
        if self.cull_backfaces:
            ctx.enable(moderngl.CULL_FACE)
        else:
            ctx.disable(moderngl.CULL_FACE)

        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        aspect = max(fb_w, 1) / max(fb_h, 1)
        near = max(self.scene_diag * 0.0005, 1e-4)
        far = self.scene_diag * 20.0
        proj = perspective(60.0, aspect, near, far)
        view = self.camera.view_matrix()
        mvp = proj @ view

        prog = self.program
        prog["mvp"].write(np.ascontiguousarray(mvp.T, dtype=np.float32).tobytes())
        prog["camPos"].value = tuple(self.camera.eye.astype(np.float32))

        for i in range(MAX_COEFFS):
            tex = self.textures[i] if self.textures[i] is not None else self.dummy_tex
            tex.use(location=i)
            prog[f"enable{i}"].value = 1 if (self.textures[i] is not None and self.enable[i]) else 0

        prog["viewMode"].value = self.view_mode
        prog["selectedCoeff"].value = int(self.selected_coeff)

        self.vao.render(moderngl.TRIANGLES, vertices=self.n_draw_verts)

    # ------------------------------------------------------------ main loop

    def run(self):
        while not glfw.window_should_close(self.window):
            now = time.perf_counter()
            dt = 0.0 if self._last_frame_time is None else (now - self._last_frame_time)
            self._last_frame_time = now
            dt = min(dt, 0.1)

            glfw.poll_events()
            self.impl.process_inputs()

            self._process_camera_input(dt)
            self._build_gui()

            self._draw_scene()
            self.impl.render(imgui.get_draw_data())

            glfw.swap_buffers(self.window)

        self.impl.shutdown()
        glfw.terminate()


def run(output_dir, max_tex_size=4096, flip_v=True, up="z", window_size=(1280, 800)):
    viewer = Viewer(output_dir, max_tex_size=max_tex_size, flip_v=flip_v, up=up, window_size=window_size)
    viewer.run()
