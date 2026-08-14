"""Free orbit/fly/pan camera + minimal matrix math (no external math3d dep)."""

import numpy as np


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def look_at(eye, target, up):
    f = _normalize(target - eye)
    s = _normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    t = np.eye(4, dtype=np.float32)
    t[0:3, 3] = -eye
    return (m @ t).astype(np.float32)


def perspective(fov_y_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_y_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


class OrbitFlyCamera:
    """
    Left-drag  : orbit around target
    Right-drag : pan (translate target + eye)
    Scroll     : zoom (move eye toward/away from target)
    WASD/QE    : fly (translate eye + target along view axes)
    Shift      : speed boost while flying
    """

    def __init__(self, target, distance, up=(0, 0, 1)):
        self.target = np.array(target, dtype=np.float64)
        self.up_world = _normalize(np.array(up, dtype=np.float64))
        self.yaw = -90.0
        self.pitch = -20.0
        self.distance = float(distance)
        self.min_distance = 1e-3
        self.fly_speed = distance * 0.5

    def _basis(self):
        yaw = np.radians(self.yaw)
        pitch = np.radians(self.pitch)
        fwd = np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ])
        fwd = _normalize(fwd)
        right = _normalize(np.cross(fwd, self.up_world))
        up = np.cross(right, fwd)
        return fwd, right, up

    @property
    def eye(self):
        fwd, _, _ = self._basis()
        return self.target - fwd * self.distance

    def orbit(self, dx, dy):
        self.yaw += dx * 0.25
        self.pitch = float(np.clip(self.pitch - dy * 0.25, -89.0, 89.0))

    def pan(self, dx, dy):
        _, right, up = self._basis()
        scale = self.distance * 0.0015
        delta = (-right * dx + up * dy) * scale
        self.target += delta

    def zoom(self, dy_scroll):
        self.distance *= (0.9 ** dy_scroll)
        self.distance = max(self.min_distance, self.distance)
        self.fly_speed = self.distance * 0.5

    def fly(self, forward, strafe, vertical, dt, boost=False):
        if forward == 0 and strafe == 0 and vertical == 0:
            return
        fwd, right, up = self._basis()
        speed = self.fly_speed * (3.0 if boost else 1.0)
        delta = (fwd * forward + right * strafe + self.up_world * vertical) * speed * dt
        self.target += delta

    def view_matrix(self):
        return look_at(self.eye.astype(np.float32), self.target.astype(np.float32), self.up_world.astype(np.float32))
