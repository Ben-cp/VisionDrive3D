#version 330 core

precision mediump float;

in vec3 v_normal;
in vec3 v_pos;
in vec3 v_color;
in vec2 v_uv;

uniform vec3 light_pos;
uniform vec3 light_color;
uniform float shininess;
uniform float ambient_strength;
uniform float specular_strength;
uniform sampler2D diffuse_map;
uniform int has_texture;
uniform int use_uniform_base_color;
uniform vec3 base_color;
uniform int use_emissive;
uniform vec3 emissive_color;
uniform float emissive_strength;
uniform float emissive_global_scale;

const int MAX_POINT_LIGHTS = 16;
uniform int num_point_lights;
uniform vec3 point_light_pos[MAX_POINT_LIGHTS];
uniform vec3 point_light_color[MAX_POINT_LIGHTS];
uniform float point_light_intensity[MAX_POINT_LIGHTS];
uniform float point_light_range[MAX_POINT_LIGHTS];

out vec4 fragColor;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(light_pos - v_pos);
    vec3 R = reflect(-L, N);
    vec3 V = normalize(-v_pos);

    vec3 surface_color = v_color;
    if (has_texture != 0) {
        surface_color = texture(diffuse_map, v_uv).rgb;
    } else if (use_uniform_base_color != 0) {
        surface_color = base_color;
    }

    float diffuse = max(dot(L, N), 0.0);
    float specAngle = max(dot(R, V), 0.0);
    float specular = pow(specAngle, shininess);

    vec3 ambient = ambient_strength * surface_color * light_color;
    vec3 diffuse_term = diffuse * surface_color * light_color;
    vec3 specular_term = specular_strength * specular * light_color;

    vec3 point_accum = vec3(0.0);
    for (int i = 0; i < MAX_POINT_LIGHTS; ++i) {
        if (i >= num_point_lights) {
            break;
        }

        vec3 Lp = point_light_pos[i] - v_pos;
        float d = length(Lp);
        if (d <= 1e-5) {
            continue;
        }
        vec3 Lpn = normalize(Lp);
        float range_f = max(point_light_range[i], 0.01);
        float attenuation = clamp(1.0 - d / range_f, 0.0, 1.0);
        attenuation *= attenuation;

        float diff_p = max(dot(N, Lpn), 0.0);
        vec3 Rp = reflect(-Lpn, N);
        float spec_p = pow(max(dot(Rp, V), 0.0), shininess);

        vec3 p_color = point_light_color[i] * point_light_intensity[i];
        point_accum += attenuation * (diff_p * surface_color * p_color + specular_strength * spec_p * p_color);
    }

    vec3 emissive = vec3(0.0);
    if (use_emissive != 0) {
        emissive = emissive_color * emissive_strength * max(emissive_global_scale, 0.0);
    }

    vec3 rgb = ambient + diffuse_term + specular_term + point_accum + emissive;
    fragColor = vec4(rgb, 1.0);
}

