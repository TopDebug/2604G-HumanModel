#version 450
layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inNormal;
layout(location=2) in vec3 inColor;

layout(binding=0) uniform UBO { mat4 model; mat4 view; mat4 proj; } ubo;

layout(location=0) out vec3 fragColor;
layout(location=1) out vec3 fragNormal;

void main() {
    fragColor = inColor;
    fragNormal = mat3(ubo.model) * inNormal;
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPos, 1.0);
}
