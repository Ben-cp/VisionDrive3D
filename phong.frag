#version 330 core

precision mediump float;

in vec3 v_normal;
in vec3 v_pos;
in vec3 v_color;

uniform mat3 K_materials;
uniform mat3 I_light;
uniform vec3 light_pos;
uniform float shininess;
uniform int mode;

out vec4 fragColor;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(light_pos - v_pos);
    vec3 R = reflect(-L, N);
    vec3 V = normalize(-v_pos);

    float specAngle = max(dot(R, V), 0.0);
    float specular = pow(specAngle, shininess);
    float diffuse = max(dot(L, N), 0.0);

    // g = (diffuse, specular, ambientFactor)
    vec3 g = vec3(diffuse, specular, 1.0);
    vec3 lit = matrixCompMult(K_materials, I_light) * g;

    // Mix with per-vertex color so OBJ-only meshes still look reasonable.
    vec3 rgb = 0.5 * lit + 0.5 * v_color;
    fragColor = vec4(rgb, 1.0);
}

