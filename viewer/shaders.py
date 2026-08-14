"""
GLSL shaders reproducing texture_optimizer.trainer's SH shading equation:

    color = coeff0 + sum_{l=1..8} coeff_l * Y_l(local_view_dir)

local_view_dir is the camera->fragment direction expressed in the per-face
(T, B, N) frame built by sh_math.compute_face_tangent_frames (mirrors
Trainer._compute_face_tangent_frames / _view_dirs_from_face_ids exactly).

Caveat baked into this approximation: `coeff0` (texCoeff0) is the exported
optimized_texture.png, which already has *average* PPISP (exposure/WB/gamma/
contrast) baked in by exporter.py, whereas coeff1..8 are raw, un-tonemapped
linear residuals. There is no exported raw/un-tonemapped base texture to
combine them exactly as done at training time, so this viewer adds the raw
AC terms directly on top of the tonemapped DC texture and clips to [0,1].
This preserves the qualitative view-dependent behaviour (which is what the
viewer is for) but is not a pixel-exact reproduction of the training-time
render.
"""

VERTEX_SHADER = """
#version 330 core

uniform mat4 mvp;

in vec3 in_pos;
in vec2 in_uv;
in vec3 in_T;
in vec3 in_B;
in vec3 in_N;

out vec2 v_uv;
out vec3 v_T;
out vec3 v_B;
out vec3 v_N;
out vec3 v_worldPos;

void main() {
    v_uv = in_uv;
    v_T = in_T;
    v_B = in_B;
    v_N = in_N;
    v_worldPos = in_pos;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core

uniform vec3 camPos;

uniform sampler2D texCoeff0;
uniform sampler2D texCoeff1;
uniform sampler2D texCoeff2;
uniform sampler2D texCoeff3;
uniform sampler2D texCoeff4;
uniform sampler2D texCoeff5;
uniform sampler2D texCoeff6;
uniform sampler2D texCoeff7;
uniform sampler2D texCoeff8;

uniform int enable0;
uniform int enable1;
uniform int enable2;
uniform int enable3;
uniform int enable4;
uniform int enable5;
uniform int enable6;
uniform int enable7;
uniform int enable8;

// viewMode: 0 = composite (sum of enabled bands)
//           1 = isolate a single band's contribution, centered at gray
//           2 = raw stored coefficient texture (no view dependence)
uniform int viewMode;
uniform int selectedCoeff;

in vec2 v_uv;
in vec3 v_T;
in vec3 v_B;
in vec3 v_N;
in vec3 v_worldPos;

out vec4 fragColor;

vec3 sampleCoeff(int idx, vec2 uv) {
    if (idx == 0) return texture(texCoeff0, uv).rgb;
    if (idx == 1) return texture(texCoeff1, uv).rgb;
    if (idx == 2) return texture(texCoeff2, uv).rgb;
    if (idx == 3) return texture(texCoeff3, uv).rgb;
    if (idx == 4) return texture(texCoeff4, uv).rgb;
    if (idx == 5) return texture(texCoeff5, uv).rgb;
    if (idx == 6) return texture(texCoeff6, uv).rgb;
    if (idx == 7) return texture(texCoeff7, uv).rgb;
    return texture(texCoeff8, uv).rgb;
}

void main() {
    vec3 viewDirWorld = normalize(camPos - v_worldPos);
    vec3 lv = normalize(vec3(
        dot(viewDirWorld, v_T),
        dot(viewDirWorld, v_B),
        dot(viewDirWorld, v_N)
    ));
    float x = lv.x, y = lv.y, z = lv.z;

    float b1 = 0.4886025119029199 * y;
    float b2 = 0.4886025119029199 * z;
    float b3 = 0.4886025119029199 * x;
    float b4 = 1.0925484305920792 * x * y;
    float b5 = 1.0925484305920792 * y * z;
    float b6 = 0.31539156525252005 * (3.0 * z * z - 1.0);
    float b7 = 1.0925484305920792 * x * z;
    float b8 = 0.5462742152960396 * (x * x - y * y);

    vec3 color;
    if (viewMode == 1) {
        vec3 c = sampleCoeff(selectedCoeff, v_uv);
        float basis =
            selectedCoeff == 1 ? b1 :
            selectedCoeff == 2 ? b2 :
            selectedCoeff == 3 ? b3 :
            selectedCoeff == 4 ? b4 :
            selectedCoeff == 5 ? b5 :
            selectedCoeff == 6 ? b6 :
            selectedCoeff == 7 ? b7 :
            selectedCoeff == 8 ? b8 : 1.0;
        color = vec3(0.5) + c * basis;
    } else if (viewMode == 2) {
        vec3 c = sampleCoeff(selectedCoeff, v_uv);
        color = (selectedCoeff == 0) ? c : (vec3(0.5) + c);
    } else {
        color = vec3(0.0);
        if (enable0 != 0) color += sampleCoeff(0, v_uv);
        if (enable1 != 0) color += sampleCoeff(1, v_uv) * b1;
        if (enable2 != 0) color += sampleCoeff(2, v_uv) * b2;
        if (enable3 != 0) color += sampleCoeff(3, v_uv) * b3;
        if (enable4 != 0) color += sampleCoeff(4, v_uv) * b4;
        if (enable5 != 0) color += sampleCoeff(5, v_uv) * b5;
        if (enable6 != 0) color += sampleCoeff(6, v_uv) * b6;
        if (enable7 != 0) color += sampleCoeff(7, v_uv) * b7;
        if (enable8 != 0) color += sampleCoeff(8, v_uv) * b8;
    }

    fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
"""
