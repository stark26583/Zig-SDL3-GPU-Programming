## Zig‑SDL3‑GPU‑Programming

A cross-platform demo showcasing GPU rendering in [Zig](https://ziglang.org/) using [SDL3](https://github.com/libsdl-org/SDL) for windowing and context creation.

### Features

- **Cross-Platform Window & GPU Context**: Uses SDL3 to create a window and GPU device on Windows, macOS, Linux, and mobile platforms.
- **SIMD-Optimized Math**: Powered by [zmath](https://github.com/zelli/zmath), offering fast vector/matrix operations for 3D transforms.
- **Texture Loading**: Integrates [zstbi](https://github.com/MasterQ32/zstbi) (STB Image) for decoding textures (PNG, JPEG, HDR).
- **Shader Pipeline**: Compiles GLSL sources into SPIR‑V binaries via [`glslc`](https://github.com/google/shaderc) from the Vulkan SDK.
- **Frame Rate Control**: `FpsManager.zig` measures delta time and regulates rendering speed.
- **Single-Step Build & Run**: Orchestrated by Zig’s native build system—just `zig build run`.

### Requirements

- [Zig](https://ziglang.org/) ≥ 0.14.0
- [Vulkan SDK](https://vulkan.lunarg.com/) (for `glslc`)
- C compiler (for SDL3 bindings and stb libraries)
- Git (for cloning and updates)

### Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/stark26583/Zig-SDL3-GPU-Programming.git
   cd Zig-SDL3-GPU-Programming
   ```  
2. Build and run:  
   ```bash
   zig build run
   ```  
   This command fetches dependencies (`sdl3`, `zmath`, `zstbi`), compiles shaders to SPIR‑V, builds the executable, and launches the demo citeturn13search1.

### Usage  
- **Default demo**: Renders a rotating 3d model of sports car, rotating quad with perspective projection—run via `zig build run`.  
- **Colorful quad example**:  optional clean up to be done
  ```bash
  zig build run -- main_colorful_quad.zig
  ```  
  Displays a rotating quad with per‑vertex colors (without textures).  
- **Arguments**: Any additional runtime flags (e.g., window size or toggles) can be passed after `zig build run --`.

Additional runtime flags (e.g., window size) can be passed after `--`.

### Project Structure

```
.
├── build.zig            # Build script: defines executable and shader steps
├── build.zig.zon        # Dependency manifest (sdl3, zmath, zstbi)
├── LICENSE              # MIT License
├── src
│   ├── main.zig         # Textured quad demo
│   ├── main_colorful_quad.zig  # Color-only quad demo
│   ├── FpsManager.zig   # Frame timing utility
│   ├── data
│   │   └── cobblestone_1.png  # Sample texture
│   └── shaders
│       ├── source
│       │   ├── shader.glsl.vert  # Vertex shader source
│       │   └── shader.glsl.frag  # Fragment shader source
│       └── compiled
│           ├── shader.spv.vert   # SPIR‑V vertex shader
│           └── shader.spv.frag   # SPIR‑V fragment shader
└── README.md
```

### Contributing

Contributions are welcome! Please open issues for bugs or feature requests, and submit pull requests. Ensure code follows the existing style and includes documentation.

### License

This project is licensed under the [MIT License](LICENSE).

---

### References

- [Zig Programming Language](https://ziglang.org/)
- [SDL3 by libsdl-org](https://github.com/libsdl-org/SDL)
- [zmath SIMD Math Library](https://github.com/zelli/zmath)
- [zstbi - Zig STB Image](https://github.com/MasterQ32/zstbi)
- [Vulkan SDK](https://vulkan.lunarg.com/)
- [glslc Compiler](https://github.com/google/shaderc)
- [Shaderc GitHub Repository](https://github.com/google/shaderc)

