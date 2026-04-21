#version 330 core

precision mediump float;

uniform vec3 instance_color;

in vec3 WorldNormal;
out vec4 fragColor;

void main(){
    if (instance_color.r < 0.0) {
        // Environment dynamic classification
        if (WorldNormal.y > 0.8) {
            // Ground -> Yellow
            fragColor = vec4(1.0, 1.0, 0.0, 1.0);
        } else {
            // House -> Orange
            fragColor = vec4(1.0, 0.5, 0.0, 1.0);
        }
    } else {
        // Direct Semantic+Instance color from CPU
        fragColor = vec4(instance_color, 1.0);
    }
}