#version 330 core

precision mediump float;

uniform vec3 instance_color;

out vec4 fragColor;

void main(){
    fragColor = vec4(instance_color, 1.0);
}