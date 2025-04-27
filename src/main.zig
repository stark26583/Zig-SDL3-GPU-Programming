const std = @import("std");
const sdl3 = @import("sdl3");
const zmath = @import("zmath");
const zstbi = @import("zstbi");
const FpsManager = @import("FpsManager.zig");
const OBJ = @import("OBJ.zig");

pub export fn SDL_AppInit(
    app_state: *?*anyopaque,
    arg_count: c_int,
    arg_values: [*][*:0]u8,
) callconv(.C) sdl3.AppResult {
    return init(@ptrCast(app_state), arg_values[0..@intCast(arg_count)]) catch return .failure;
}

pub export fn SDL_AppIterate(
    app_state: ?*anyopaque,
) callconv(.C) sdl3.AppResult {
    return update(@alignCast(@ptrCast(app_state))) catch return .failure;
}

pub export fn SDL_AppEvent(
    app_state: ?*anyopaque,
    event: *sdl3.c.SDL_Event,
) callconv(.C) sdl3.AppResult {
    return eventHandler(@alignCast(@ptrCast(app_state)), sdl3.events.Event.fromSdl(event.*)) catch return .failure;
}

pub export fn SDL_AppQuit(
    app_state: ?*anyopaque,
    result: sdl3.AppResult,
) callconv(.C) void {
    quit(@alignCast(@ptrCast(app_state)), result);
}
//----------------------------------------------------------------------------
// Disable main hack.
pub const _start = void;
pub const WinMainCRTStartup = void;

var gpa: std.heap.GeneralPurposeAllocator(.{}) = undefined;
var allocator: std.mem.Allocator = undefined;

/// For logging system messages.
const log_app = sdl3.log.Category.application;

const CommonTypes = @import("CommonTypes.zig");
const Vec3 = CommonTypes.Vec3;
const Color = CommonTypes.Color;
const Vertex = CommonTypes.Vertex;
const UBO = CommonTypes.UBO;

const SCREEN_WIDTH = 1280;
const SCREEN_HEIGHT = 780;

const assert = std.debug.assert;

const GPU = sdl3.gpu;

const AppState = struct {
    init_flags: sdl3.init.Flags = .{ .video = true },

    window: sdl3.video.Window,
    gpu: GPU.Device,
    fps_manager: FpsManager,

    vertex_shader: GPU.Shader,
    fragment_shader: GPU.Shader,

    vertex_buffer: GPU.Buffer,
    index_buffer: GPU.Buffer,
    transfer_buffer: GPU.TransferBuffer,
    image_for_texture: zstbi.Image,
    texture: GPU.Texture,
    sampler: GPU.Sampler,
    depth_texture: GPU.Texture,

    obj_data: OBJ.OBJ_Data,
    vertices: []Vertex,
    indices: []u16,

    pipeline: GPU.GraphicsPipeline,

    paused: bool,

    const DEPTH_TEXTURE_FORMAT = GPU.TextureFormat.depth24_unorm;
    var indices_len: u32 = 0;
    var rotation: f32 = 0;
    const rotation_speed: f32 = std.math.degreesToRadians(90);

    const vertex_shader_code = @embedFile("shaders/compiled/shader.spv.vert");
    const fragment_shader_code = @embedFile("shaders/compiled/shader.spv.frag");

    var proj_mat: zmath.Mat = undefined;
};

fn sdlErr(
    err: ?[]const u8,
) void {
    if (err) |val| {
        std.debug.print("******* [Error! {s}] *******\n", .{val});
    } else {
        std.debug.print("******* [Unknown Error!] *******\n", .{});
    }
}

fn sdlLog(
    user_data: ?*anyopaque,
    category: c_int,
    priority: sdl3.c.SDL_LogPriority,
    message: [*c]const u8,
) callconv(.C) void {
    _ = user_data;
    const category_managed = sdl3.log.Category.fromSdl(category);
    const category_str: ?[]const u8 = if (category_managed) |val| switch (val.value) {
        sdl3.log.Category.application.value => "Application",
        sdl3.log.Category.errors.value => "Errors",
        sdl3.log.Category.assert.value => "Assert",
        sdl3.log.Category.system.value => "System",
        sdl3.log.Category.audio.value => "Audio",
        sdl3.log.Category.video.value => "Video",
        sdl3.log.Category.render.value => "Render",
        sdl3.log.Category.input.value => "Input",
        sdl3.log.Category.testing.value => "Testing",
        sdl3.log.Category.gpu.value => "Gpu",
        else => null,
    } else null;
    const priority_managed = sdl3.log.Priority.fromSdl(priority);
    const priority_str: [:0]const u8 = if (priority_managed) |val| switch (val) {
        .trace => "Trace",
        .verbose => "Verbose",
        .debug => "Debug",
        .info => "Info",
        .warn => "Warn",
        .err => "Error",
        .critical => "Critical",
    } else "Unknown";
    if (category_str) |val| {
        std.debug.print("[{s}:{s}] {s}\n", .{ val, priority_str, message });
    } else {
        std.debug.print("[Custom_{d}:{s}] {s}\n", .{ category, priority_str, message });
    }
}

fn init(
    app_state: *?*AppState,
    args: [][*:0]u8,
) !sdl3.AppResult {
    _ = args;
    // Setup logging.
    sdl3.errors.error_callback = &sdlErr;
    sdl3.log.setAllPriorities(.info);
    sdl3.log.setLogOutputFunction(&sdlLog, null);

    log_app.logInfo("Starting application...");

    gpa = std.heap.GeneralPurposeAllocator(.{}){};
    allocator = gpa.allocator();

    // Prepare app state.
    const state = try allocator.create(AppState);
    errdefer allocator.destroy(state);

    const num_available_gpus = sdl3.c.SDL_GetNumGPUDrivers();
    // const set_d3d12 = false;
    std.debug.print("{d} available GPU Drivers\n", .{num_available_gpus});
    for (0..@intCast(num_available_gpus)) |i| {
        const name = sdl3.c.SDL_GetGPUDriver(@intCast(i));
        // if (std.mem.startsWith(u8, name[0..2], "d")) {
        // set_d3d12 = true;
        // }
        std.debug.print("{d}) {s}\n", .{ i + 1, name });
    }

    const window = try sdl3.video.Window.init("GPU programming", SCREEN_WIDTH, SCREEN_HEIGHT, .{});
    const gpu = try GPU.Device.init(.{ .spirv = true }, true, null);
    try gpu.claimWindow(window);

    zstbi.init(allocator);

    const vertex_shader = try loadShader(
        gpu,
        AppState.vertex_shader_code,
        .vertex,
        1,
        0,
    );
    const fragment_shader = try loadShader(
        gpu,
        AppState.fragment_shader_code,
        .fragment,
        0,
        1,
    );
    //Create Obj
    const obj_data = try OBJ.parse(allocator, "./data/race.obj");
    //Create Texture
    const image = try zstbi.Image.loadFromFile("./data/colormap.png", 4);
    const pixels_byte_size = image.width * image.height * 4;

    const texture_c = sdl3.c.SDL_CreateGPUTexture(gpu.value, &sdl3.c.SDL_GPUTextureCreateInfo{
        .type = sdl3.c.SDL_GPU_TEXTURETYPE_2D,
        .width = image.width,
        .height = image.height,
        .format = sdl3.c.SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
        .usage = sdl3.c.SDL_GPU_TEXTUREUSAGE_SAMPLER,
        .layer_count_or_depth = 1,
        .num_levels = 1,
    });
    const texture = GPU.Texture{ .value = texture_c orelse unreachable };

    const depth_texture_c = sdl3.c.SDL_CreateGPUTexture(gpu.value, &sdl3.c.SDL_GPUTextureCreateInfo{
        .type = sdl3.c.SDL_GPU_TEXTURETYPE_2D,
        .width = SCREEN_WIDTH,
        .height = SCREEN_HEIGHT,
        .format = @intFromEnum(AppState.DEPTH_TEXTURE_FORMAT),
        .usage = sdl3.c.SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET,
        .layer_count_or_depth = 1,
        .num_levels = 1,
    });
    const depth_texture = GPU.Texture{ .value = depth_texture_c orelse unreachable };

    // create vertex data
    const White = Color{ 1.0, 1.0, 1.0, 1.0 };

    const vertices = try allocator.alloc(Vertex, obj_data.faces.len);
    const indices = try allocator.alloc(u16, obj_data.faces.len);

    for (obj_data.faces, 0..) |faces, i| {
        const uv = obj_data.uvs_tex_coords[faces.uv_index];
        vertices[i] = .{
            .pos = obj_data.positions[faces.position_index],
            .color = White,
            .uv = .{ uv[0], 1 - uv[1] },
        };

        indices[i] = @intCast(i);
    }

    AppState.indices_len = @intCast(indices.len);

    const vertices_byte_size = vertices.len * @sizeOf(@TypeOf(vertices[0]));
    const indices_byte_size = indices.len * @sizeOf(@TypeOf(indices[0]));
    // describe vertex attributes and vertex buffers in pipline
    const vertex_attributes = [_]GPU.VertexAttribute{
        GPU.VertexAttribute{
            .location = 0,
            .format = .f32x3,
            .offset = @offsetOf(Vertex, "pos"),
            .buffer_slot = 0,
        },
        GPU.VertexAttribute{
            .location = 1,
            .format = .f32x4,
            .offset = @offsetOf(Vertex, "color"),
            .buffer_slot = 0,
        },
        GPU.VertexAttribute{
            .location = 2,
            .format = .f32x2,
            .offset = @offsetOf(Vertex, "uv"),
            .buffer_slot = 0,
        },
    };

    const vertex_buffer_descriptions = [_]GPU.VertexBufferDescription{
        GPU.VertexBufferDescription{
            .slot = 0,
            .pitch = @sizeOf(Vertex),
            .input_rate = .vertex,
        },
    };
    // create vertex buffer
    const vertex_buffer = try gpu.createBuffer(.{
        .usage = .{ .vertex = true },
        .size = @intCast(vertices_byte_size),
    });
    // create index buffer
    const index_buffer = try gpu.createBuffer(.{
        .usage = .{ .index = true },
        .size = @intCast(indices_byte_size),
    });
    // Create Transfer Buffer
    const transfer_buffer_c = sdl3.c.SDL_CreateGPUTransferBuffer(gpu.value, &.{
        .usage = sdl3.c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = @intCast(vertices_byte_size + indices_byte_size),
    });
    const transfer_buffer = GPU.TransferBuffer{ .value = transfer_buffer_c orelse unreachable };
    const transfer_memory_ptr = sdl3.c.SDL_MapGPUTransferBuffer(gpu.value, transfer_buffer.value, false);
    memcpy_into_transfer_buff(transfer_memory_ptr.?, vertices, vertices_byte_size);
    memcpy_into_transfer_buff(@ptrFromInt(@as(usize, @intFromPtr(transfer_memory_ptr.?)) + vertices_byte_size), indices, indices_byte_size);
    gpu.unmapTransferBuffer(transfer_buffer);
    // create texture transfer buffer
    const texture_transfer_buffer_c = sdl3.c.SDL_CreateGPUTransferBuffer(gpu.value, &.{
        .usage = sdl3.c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = @intCast(pixels_byte_size),
    });
    const texture_transfer_buffer = GPU.TransferBuffer{ .value = texture_transfer_buffer_c orelse unreachable };
    defer sdl3.c.SDL_ReleaseGPUTransferBuffer(gpu.value, texture_transfer_buffer.value);
    const texture_transfer_memory_ptr = sdl3.c.SDL_MapGPUTransferBuffer(gpu.value, texture_transfer_buffer.value, false);
    memcpy_into_transfer_buff(texture_transfer_memory_ptr.?, image.data, pixels_byte_size);
    gpu.unmapTransferBuffer(texture_transfer_buffer);

    // - begin copy pass
    const copy_cmd_buffer = try gpu.aquireCommandBuffer();
    const copy_pass = copy_cmd_buffer.beginCopyPass();
    // - invoke upload commands
    copy_pass.uploadToBuffer(
        .{
            .transfer_buffer = transfer_buffer,
            .offset = 0,
        },
        .{
            .buffer = vertex_buffer,
            .offset = 0,
            .size = @intCast(vertices_byte_size),
        },
        false,
    );
    copy_pass.uploadToBuffer(
        .{
            .transfer_buffer = transfer_buffer,
            .offset = @intCast(vertices_byte_size),
        },
        .{
            .buffer = index_buffer,
            .offset = 0,
            .size = @intCast(indices_byte_size),
        },
        false,
    );
    copy_pass.uploadToTexture(.{
        .transfer_buffer = texture_transfer_buffer,
        .offset = 0,
        .pixels_per_row = 0,
        .rows_per_layer = 0,
    }, .{
        .texture = texture,
        .width = image.width,
        .height = image.height,
        .depth = 1,
        .layer = 0,
        .mip_level = 0,
        .x = 0,
        .y = 0,
        .z = 0,
    }, false);
    // - end copy pass
    sdl3.c.SDL_EndGPUCopyPass(copy_pass.value);
    try copy_cmd_buffer.submit();

    const sampler = try gpu.createSampler(.{});

    const pipeline = try gpu.createGraphicsPipeline(.{
        .vertex_shader = vertex_shader,
        .fragment_shader = fragment_shader,
        .primitive_type = .triangle_list,
        .vertex_input_state = .{
            .vertex_attributes = vertex_attributes[0..],
            .vertex_buffer_descriptions = vertex_buffer_descriptions[0..],
        },
        .depth_stencil_state = .{
            .enable_depth_test = true,
            .enable_depth_write = true,
            .compare = .less,
        },
        .target_info = .{
            .color_target_descriptions = &.{GPU.ColorTargetDescription{
                .format = @enumFromInt(sdl3.c.SDL_GetGPUSwapchainTextureFormat(gpu.value, window.value)),
            }},
            .depth_stencil_format = AppState.DEPTH_TEXTURE_FORMAT,
        },
    });

    var w_cint: c_int = undefined;
    var h_cint: c_int = undefined;
    assert(sdl3.c.SDL_GetWindowSize(window.value, &w_cint, &h_cint));

    const w = @as(f32, @floatFromInt(w_cint));
    const h = @as(f32, @floatFromInt(h_cint));

    AppState.proj_mat = zmath.perspectiveFovRh(std.math.degreesToRadians(70), w / h, 0.0001, 1000);
    try gpu.setSwapchainParameters(window, .sdr, .immediate);

    state.* = .{
        .init_flags = .{ .video = true },
        .window = window,
        .gpu = gpu,
        .fps_manager = FpsManager.init(.none),
        .paused = false,

        .vertex_shader = vertex_shader,
        .fragment_shader = fragment_shader,

        .vertex_buffer = vertex_buffer,
        .index_buffer = index_buffer,
        .transfer_buffer = transfer_buffer,
        .image_for_texture = image,
        .texture = texture,
        .sampler = sampler,
        .depth_texture = depth_texture,

        .obj_data = obj_data,
        .vertices = vertices,
        .indices = indices,

        .pipeline = pipeline,
    };

    app_state.* = state;

    log_app.logInfo("Finished initializing");
    return .run;
}

fn eventHandler(
    app_state: *AppState,
    event: sdl3.events.Event,
) !sdl3.AppResult {
    // _ = app_state;
    switch (event) {
        .terminating => return .success,
        .quit => return .success,
        .unknown => {
            if (event.unknown.event_type == sdl3.c.SDL_EVENT_KEY_DOWN) {
                switch (event.toSdl().key.scancode) {
                    sdl3.c.SDLK_ESCAPE => return .success,
                    sdl3.c.SDLK_KP_SPACE => {
                        app_state.paused = !app_state.paused;
                    },
                    sdl3.c.SDLK_KP_F => {
                        std.debug.print("FPS: {d}\n", .{app_state.fps_manager.getFps()});
                    },
                    else => {},
                }
            }
            return .run;
        },
        else => {
            return .run;
        },
    }
}

fn quit(
    app_state: ?*AppState,
    result: sdl3.AppResult,
) void {
    _ = result;
    if (app_state) |val| {
        val.gpu.releaseGraphicsPipeline(val.pipeline);
        val.gpu.releaseTransferBuffer(val.transfer_buffer);

        sdl3.c.SDL_ReleaseGPUBuffer(val.gpu.value, val.index_buffer.value);
        sdl3.c.SDL_ReleaseGPUBuffer(val.gpu.value, val.vertex_buffer.value);
        allocator.free(val.indices);
        allocator.free(val.vertices);
        val.gpu.releaseTexture(val.depth_texture);
        val.gpu.releaseTexture(val.texture);
        val.image_for_texture.deinit();
        val.obj_data.deinit(allocator);
        val.gpu.releaseSampler(val.sampler);
        val.gpu.releaseShader(val.vertex_shader);
        val.gpu.releaseShader(val.fragment_shader);
        val.gpu.releaseWindow(val.window);
        val.gpu.deinit();
        val.window.deinit();
        zstbi.deinit();
        sdl3.init.quit(val.init_flags);
        sdl3.init.shutdown();
        allocator.destroy(val);

        const leaked = gpa.deinit();
        assert(leaked == .ok);
    }
}

fn update(
    app_state: *AppState,
) !sdl3.AppResult {
    //Update Code
    app_state.fps_manager.tick();

    //Draw Code
    if (!app_state.paused) AppState.rotation += AppState.rotation_speed * app_state.fps_manager.getDelta();
    const rot = zmath.rotationY(AppState.rotation);
    const trans = zmath.translation(0.0, 0.0, 0.0);
    const view = zmath.lookAtRh(.{ 0, 0, -3, 1 }, .{ 0, 0, 0, 1 }, .{ 0, 1, 0, 0 });

    const model_mat = zmath.mul(rot, trans);
    const mv_mat = zmath.mul(model_mat, view);
    const ubo = UBO{
        .mvp = zmath.mul(mv_mat, AppState.proj_mat),
    };

    // render
    const cmd_buffer = try app_state.gpu.aquireCommandBuffer();
    const swapchain = try cmd_buffer.waitAndAquireSwapchainTexture(app_state.window);

    //Begin render pass
    if (swapchain.texture) |swapchain_tex| {
        const color_target = GPU.ColorTargetInfo{
            .texture = swapchain_tex,
            .load = .load,
            .clear_color = .{ .r = 0.1, .g = 0.1, .b = 0.1, .a = 1.0 },
            .store = .store,
        };

        const depth_stencil_target_info = GPU.DepthStencilTargetInfo{
            .texture = app_state.depth_texture,
            .load = .load,
            .clear_depth = 1.0,
            .store = .do_not_care,
            .cycle = false,
            .clear_stencil = 0,
            .stencil_load = .do_not_care,
            .stencil_store = .do_not_care,
        };
        const render_pass = cmd_buffer.beginRenderPass(&.{color_target}, depth_stencil_target_info);

        render_pass.bindGraphicsPipeline(app_state.pipeline);

        const bindings = [_]GPU.BufferBinding{.{ .buffer = app_state.vertex_buffer, .offset = 0 }};
        render_pass.bindVertexBuffers(0, bindings[0..]);
        render_pass.bindIndexBuffer(.{ .buffer = app_state.index_buffer, .offset = 0 }, .indices_16bit);

        sdl3.c.SDL_PushGPUVertexUniformData(cmd_buffer.value, 0, &ubo, @sizeOf(@TypeOf(ubo)));

        const fragment_samplers_bindings = [_]GPU.TextureSamplerBinding{.{ .texture = app_state.texture, .sampler = app_state.sampler }};
        render_pass.bindFragmentSamplers(0, fragment_samplers_bindings[0..]);

        sdl3.c.SDL_DrawGPUIndexedPrimitives(render_pass.value, AppState.indices_len, 1, 0, 0, 0);
        sdl3.c.SDL_EndGPURenderPass(render_pass.value);
    }
    try cmd_buffer.submit();
    return .run;
}

// fn loadShader(device: ?*sdl3.c.SDL_GPUDevice, code: []const u8, stage: sdl3.c.SDL_GPUShaderStage, num_uniform_buffers: u32, num_samplers: u32) ?*sdl3.c.SDL_GPUShader {
//     return sdl3.c.SDL_CreateGPUShader(
//         device,
//         &sdl3.c.SDL_GPUShaderCreateInfo{
//             .code_size = code.len,
//             .code = @ptrCast(code),
//             .entrypoint = "main",
//             .format = sdl3.c.SDL_GPU_SHADERFORMAT_SPIRV,
//             .stage = stage,
//             .num_uniform_buffers = num_uniform_buffers,
//             .num_samplers = num_samplers,
//         },
//     );
// }
fn loadShader(device: GPU.Device, code: []const u8, stage: GPU.ShaderStage, num_uniform_buffers: u32, num_samplers: u32) !GPU.Shader {
    return try device.createShader(.{
        .format = .{ .spirv = true },
        .code = code,
        .entry_point = "main",
        .stage = stage,
        .num_uniform_buffers = num_uniform_buffers,
        .num_samplers = num_samplers,
    });
}

fn memcpy_into_transfer_buff(dest: *anyopaque, src_data: anytype, size: usize) void {
    const dest_ptr: [*]u8 = @ptrCast(dest);
    const dest_slice = dest_ptr[0..size];
    const source_ptr: [*]const u8 = @ptrCast(src_data[0..].ptr);
    const source_slice = source_ptr[0..size];
    @memcpy(dest_slice, source_slice);
}
