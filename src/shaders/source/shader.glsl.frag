#version 460

//Passed this color from vertex
layout(location=0) in vec4 color;
layout(location=1) in vec2 uv;

layout(location=0) out vec4 frag_color;

layout(set=2, binding=0) uniform sampler2D tex;

void main() {
    frag_color = texture(tex, uv) * color;
}
