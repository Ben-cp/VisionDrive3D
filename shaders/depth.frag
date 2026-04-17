#version 330 core

precision mediump float;

uniform float near;
uniform float far; 

out vec4 fragColor;

float linearize_depth(float depth){
    float z = depth * 2.0 - 1.0;
    return (2.0 * near * far) / (far + near - z *(far - near));
}

void main(){
    float depth = gl_FragCoord.z;
    float linear_depth = linearize_depth(depth);

    float normalized_depth = linear_depth / far;

    fragColor = vec4(vec3(normalized_depth), 1.0);
}