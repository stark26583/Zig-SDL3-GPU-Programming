const std = @import("std");

const projects_path = "projects/";
const main_file_name = "/main.zig";

const Project = struct {
    name: []const u8,
    deps: []const []const u8,
};

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    var projects = std.ArrayList(Project).init(b.allocator);
    defer projects.deinit();

    var projects_dir = try std.fs.cwd().openDir(projects_path, .{ .iterate = true });
    defer projects_dir.close();

    var projects_dir_iterator = projects_dir.iterate();

    while (try projects_dir_iterator.next()) |entry| {
        if (entry.kind == .directory) {
            const proj_name = entry.name;

            const deps_file_path = try std.mem.concat(b.allocator, u8, &.{ proj_name, "/deps.txt" });
            defer b.allocator.free(deps_file_path);

            const deps_file = try projects_dir.openFile(deps_file_path, .{});
            defer deps_file.close();

            var deps_file_reader = deps_file.reader();
            const file_buf = try deps_file_reader.readAllAlloc(b.allocator, 1024);
            defer b.allocator.free(file_buf);

            var deps = std.ArrayList([]const u8).init(b.allocator);
            defer deps.deinit();

            // If it starts with the UTF-8 BOM, skip those three bytes:
            var raw_lines: std.mem.SplitIterator(u8, .scalar) = undefined;
            if (file_buf.len >= 3 and (file_buf[0] == 0xEF and file_buf[1] == 0xBB and file_buf[2] == 0xBF)) {
                raw_lines = std.mem.splitScalar(u8, file_buf[3..], '\n');
            } else {
                raw_lines = std.mem.splitScalar(u8, file_buf, '\n');
            }

            while (raw_lines.next()) |raw_line| {
                const line = std.mem.trim(u8, raw_line, std.ascii.whitespace[0..]);
                if (line.len == 0) continue; // 🛑 skip blank or pure-whitespace
                try deps.append(try b.allocator.dupe(u8, line));
            }

            try projects.append(.{
                .name = proj_name,
                .deps = try deps.toOwnedSlice(),
            });
        }
    }

    const projects_slice = try projects.toOwnedSlice();
    defer b.allocator.free(projects_slice);
    defer {
        for (projects_slice) |project| {
            for (project.deps) |dep| {
                b.allocator.free(dep);
            }
            b.allocator.free(project.deps);
        }
    }

    for (projects_slice) |project| {
        const exe = try build_proj(b, project, target, optimize);
        b.installArtifact(exe);
        const run_cmd = b.addRunArtifact(exe);

        const run_cwd_path = try std.mem.concat(b.allocator, u8, &.{ projects_path, project.name });
        defer b.allocator.free(run_cwd_path);

        run_cmd.setCwd(b.path(run_cwd_path));

        if (b.args) |args| {
            run_cmd.addArgs(args);
        }

        const deps_description = try std.mem.join(b.allocator, ", ", project.deps);
        defer b.allocator.free(deps_description);

        const run_description = try std.mem.concat(b.allocator, u8, &.{ "Run ", "(", project.name, ") with ", "deps: (", deps_description, ")" });
        defer b.allocator.free(run_description);

        const run_step = b.step(project.name, run_description);
        run_step.dependOn(&run_cmd.step);
    }
}

// fn create_mode()

fn build_proj(b: *std.Build, proj: Project, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) !*std.Build.Step.Compile {
    const root_source_file = try std.mem.concat(b.allocator, u8, &.{ projects_path, proj.name, main_file_name });
    defer b.allocator.free(root_source_file);


    const exe = b.addExecutable(.{
        .name = proj.name,
        .root_source_file = b.path(root_source_file),
        .target = target,
        .optimize = optimize,
    });

    for (proj.deps) |dep| {
        if (std.mem.eql(u8, dep, "zgui")) {
            const zgui = b.dependency("zgui", .{
                .backend = .sdl3_gpu,
                .shared = false,
                .with_implot = true,
            });
            exe.root_module.addImport("zgui", zgui.module("root"));
            exe.linkLibrary(zgui.artifact("imgui"));
        } else if (std.mem.eql(u8, dep, "zmesh")) {
            const zmesh = b.dependency("zmesh", .{
                .target = target,
            });
            exe.root_module.addImport("zmesh", zmesh.module("root"));
            exe.linkLibrary(zmesh.artifact("zmesh"));
        } else {
            const zdep = b.dependency(dep, .{});
            exe.root_module.addImport(dep, zdep.module("root"));
        }
    }

    const sdl3 = b.dependency("sdl3", .{
        .target = target,
        .optimize = optimize,
        // .callbacks = true,
        .ext_image = true,
    });

    exe.root_module.addImport("sdl3", sdl3.module("sdl3"));

    return exe;
}

// const compile_frag_shader = b.addSystemCommand(&.{
//     "glslc",
// });
// compile_frag_shader.addFileArg(b.path(shaders_src_path ++ "shader.glsl.frag"));
// compile_frag_shader.addArgs(&.{
//     "-o",
// });
// compile_frag_shader.addFileArg(b.path(shaders_bin_path ++ "shader.spv.frag"));
// const compile_vert_shader = b.addSystemCommand(&.{
//     "glslc",
// });
// compile_vert_shader.addFileArg(b.path(shaders_src_path ++ "shader.glsl.vert"));
// compile_vert_shader.addArgs(&.{
//     "-o",
// });
// compile_vert_shader.addFileArg(b.path(shaders_bin_path ++ "shader.spv.vert"));
//
// b.getInstallStep().dependOn(&compile_vert_shader.step);
// b.getInstallStep().dependOn(&compile_frag_shader.step);

// const run_step = b.step("run", "Run the app");
// run_step.dependOn(&run_cmd.step);
