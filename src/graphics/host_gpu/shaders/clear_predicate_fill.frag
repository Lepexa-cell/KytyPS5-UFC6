#version 450

layout(push_constant) uniform Push {
    vec4 clear_color;
} pc;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = pc.clear_color;
}
