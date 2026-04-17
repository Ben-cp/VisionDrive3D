#version 330 core

// Attributes (required by the project spec)
layout(location = 0) in vec3 position;

uniform mat4 projection;
uniform mat4 modelview;

void main(){
    gl_Position = projection * modelview * vec4(position, 1.0);
}