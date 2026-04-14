#version 450
layout(location=0) in vec3 fragColor;
layout(location=1) in vec3 fragNormal;
layout(location=0) out vec4 outColor;

void main() {
    // Human mesh bakes diffuse+fill into vertex color (makehuman.cpp). Cube / axes use flat colors.
    outColor = vec4(fragColor, 1.0);
}