#version 450

// Step 3b regression fix: predicated clear for UINT color targets. Same push-constant
// slot as the float variant; the fragment output type must match the attachment.
layout(push_constant) uniform Push {
    uvec4 clear_color;
} pc;

layout(location = 0) out uvec4 out_color;

void main() {
    out_color = pc.clear_color;
}
