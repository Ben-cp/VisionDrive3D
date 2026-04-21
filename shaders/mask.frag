#version 330 core

precision mediump float;

uniform vec3 instance_color;

in vec3 WorldNormal;
in vec3 WorldPos;
out vec4 fragColor;

void main(){
    if (instance_color.r < 0.0) {
        // Ground Condition: Pointing Upwards AND close to Floor Height (Y < 0.5)
        if (WorldNormal.y > 0.8 && WorldPos.y < 0.5) {
            // Ground -> Yellow
            fragColor = vec4(1.0, 1.0, 0.0, 1.0);
        } else {
            // House/Roof/Walls -> Orange
            fragColor = vec4(1.0, 0.5, 0.0, 1.0);
        }
    } else {
        // Direct Semantic+Instance color from CPU
        fragColor = vec4(instance_color, 1.0);
    }
}