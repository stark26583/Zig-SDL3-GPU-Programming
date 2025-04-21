const std = @import("std");

const shaders_src_path = "src/shaders/source/";
const shaders_bin_path = "src/shaders/compiled/";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // b.verbose = true;

    const exe = b.addExecutable(.{
        .name = "main",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const zmath = b.dependency("zmath", .{});
    exe.root_module.addImport("zmath", zmath.module("root"));

    const zstbi = b.dependency("zstbi", .{});
    exe.root_module.addImport("zstbi", zstbi.module("root"));

    const sdl3 = b.dependency("sdl3", .{
        .target = target,
        .optimize = optimize,
        // .callbacks = true,
        .ext_image = true,
    });
    exe.root_module.addImport("sdl3", sdl3.module("sdl3"));
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const compile_frag_shader = b.addSystemCommand(&.{
        "glslc",
    });
    compile_frag_shader.addFileArg(b.path(shaders_src_path ++ "shader.glsl.frag"));
    compile_frag_shader.addArgs(&.{
        "-o",
    });
    compile_frag_shader.addFileArg(b.path(shaders_bin_path ++ "shader.spv.frag"));
    const compile_vert_shader = b.addSystemCommand(&.{
        "glslc",
    });
    compile_vert_shader.addFileArg(b.path(shaders_src_path ++ "shader.glsl.vert"));
    compile_vert_shader.addArgs(&.{
        "-o",
    });
    compile_vert_shader.addFileArg(b.path(shaders_bin_path ++ "shader.spv.vert"));

    b.getInstallStep().dependOn(&compile_vert_shader.step);
    b.getInstallStep().dependOn(&compile_frag_shader.step);

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
