# Zig‑SDL3‑GPU‑Programming

A cross-platform Zig demo showcasing GPU rendering with SDL3, SIMD math, and SPIR‑V shaders.

## Features

- **Cross‑Platform Window & GPU Context**: Uses SDL3 (via `zig-sdl3`) to create windows and GPU devices.
- **SIMD‑Optimized Math**: Uses `zmath` for fast vector and matrix operations.
- **Texture Loading**: Uses `zstbi` to decode textures (PNG, JPEG, HDR).
- **Shader Pipeline**: Compiles GLSL to SPIR‑V at build time using `glslc`.
- **Frame Rate Control**: `FpsManager.zig` for accurate timing.
- **Single‑Command Build & Run**: `zig build run`.

## Requirements

- **Zig** ≥ 0.14.0
- **Vulkan SDK** (provides `glslc`)
- **C Compiler Toolchain**
- **Git**

## Installation & Usage

```bash
git clone https://github.com/stark26583/Zig-SDL3-GPU-Programming.git
cd Zig-SDL3-GPU-Programming
zig build <name>
```

---

## Contributing

Contributions welcome! Open issues or PRs. Please follow existing code style and include documentation.

## License

Released under the [MIT License](LICENSE).

---

## References

- [Zig Programming Language](https://ziglang.org/)
- [zig-sdl3 (Gota7/zig-sdl3)](https://github.com/Gota7/zig-sdl3)
- [zmath (zig-gamedev/zmath)](https://github.com/zig-gamedev/zmath)
- [zstbi (zig-gamedev/zstbi)](https://github.com/zig-gamedev/zstbi)
- [SDL3 (libsdl-org/SDL)](https://github.com/libsdl-org/SDL)
- [Vulkan SDK](https://vulkan.lunarg.com/)
- [Shaderc (google/shaderc)](https://github.com/google/shaderc)

