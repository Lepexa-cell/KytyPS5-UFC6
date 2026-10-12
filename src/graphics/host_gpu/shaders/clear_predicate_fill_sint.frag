#version 450

// Step 3b regression fix: predicated clear for SINT color targets. Same push-constant
// slot as the float variant; the fragment output type must match the attachment.
layout(push_constant) uniform Push {
    ivec4 clear_color;
} pc;

layout(location = 0) out ivec4 out_color;

void main() {
    out_color = pc.clear_color;
}
