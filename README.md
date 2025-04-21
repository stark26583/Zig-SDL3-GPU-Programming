# Zig-SDL3-GPU-Programming

## Overview  
Zig‑SDL3‑GPU‑Programming is a cross‑platform demo showcasing GPU rendering in Zig, combining robust windowing with SDL3, efficient image loading, SIMD math, and SPIR‑V shaders for high‑performance graphics citeturn9search0turn10search9. It embeds GLSL shaders compiled to SPIR‑V at build time, enabling portable shader pipelines across Vulkan, Direct3D, and Metal citeturn11search5. The project leverages the `zmath` library for optimized linear algebra routines and `zstbi` for seamless texture loading from image files citeturn8search0turn12search0. Frame timing is handled by a custom `FpsManager.zig` module to ensure smooth animations, and the entire build process is orchestrated via Zig’s native build system with a single `zig build run` command citeturn13search0turn13search1.

## Features  
- **Cross‑platform windowing & GPU context**: Utilizes SDL3 for creating windows and GPU devices on Windows, macOS, Linux, and mobile platforms citeturn10search9.  
- **SIMD‑optimized math**: Powered by `zmath`, providing fast vector and matrix operations essential for 3D transformations citeturn8search0.  
- **Image loading**: Integrates `zstbi` for decoding textures (PNG, JPEG, HDR) directly into GPU memory citeturn12search0.  
- **Shader pipeline**: Compiles GLSL source files into SPIR‑V binaries using `glslc` from the Vulkan SDK citeturn16search1.  
- **Frame rate control**: Includes `FpsManager.zig` to measure delta times and regulate rendering speed.  
- **One‑step build & run**: Automates compilation, shader processing, and execution with `zig build run`, simplifying development and testing citeturn13search0.

## Requirements  
- **Zig ≥ 0.14.0**: A modern Zig compiler to support build scripts and dependencies citeturn9search0.  
- **Vulkan SDK (glslc)**: Provides `glslc` for offline GLSL→SPIR‑V compilation citeturn16search1.  
- **C compiler toolchain**: Enables building C dependencies such as `zstbi` and SDL3 language bindings.  
- **Git**: For cloning the repository and managing updates.

## Installation  
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

## Usage  
- **Default demo**: Renders a textured, rotating quad with perspective projection—run via `zig build run`.  
- **Colorful quad example**:  
  ```bash
  zig build run -- main_colorful_quad.zig
  ```  
  Displays a rotating quad with per‑vertex colors (without textures).  
- **Arguments**: Any additional runtime flags (e.g., window size or toggles) can be passed after `zig build run --`.

## Project Structure  
```
.
├── build.zig               # Zig build script defining executables and shader steps
├── build.zig.zon           # Dependency manifest (sdl3, zmath, zstbi)
├── LICENSE                 # MIT License
├── src
│   ├── main.zig            # Textured quad demo
│   ├── main_colorful_quad.zig  # Color‑only quad demo
│   ├── FpsManager.zig      # Frame rate manager
│   ├── data
│   │   └── cobblestone_1.png  # Sample texture
│   └── shaders
│       ├── source
│       │   ├── shader.glsl.vert  # GLSL vertex shader source
│       │   └── shader.glsl.frag  # GLSL fragment shader source
│       └── compiled
│           ├── shader.spv.vert   # SPIR‑V vertex shader
│           └── shader.spv.frag   # SPIR‑V fragment shader
└── README.md
```

## Contributing  
Contributions are welcome! Please open issues for bug reports or feature requests, and submit pull requests for fixes or enhancements. Ensure new code follows existing style, includes documentation, and passes any provided tests.

## License  
This project is released under the [MIT License](LICENSE), allowing unrestricted reuse in both open‑source and proprietary software.

## References  
- Zig official website and documentation citeturn9search5  
- SDL3 overview and latest release citeturn10search9  
- SPIR‑V specification by Khronos Group citeturn11search8  
- `zmath` SIMD math library for Zig citeturn8search0  
- `zstbi` Zig bindings for stb_image citeturn12search0  
- Zig build system guide citeturn13search0  
- Vulkan shader modules and `glslc` usage citeturn16search1  
- Shaderc GitHub repository (includes `glslc`) citeturn16search2  
- Zig build system examples citeturn13search1  
- Zig build system best practices citeturn13search2
