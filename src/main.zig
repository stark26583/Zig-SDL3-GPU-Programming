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

const AppState = struct {
    init_flags: sdl3.init.Flags = .{ .video = true },

    window: sdl3.video.Window,
    gpu: ?*sdl3.c.SDL_GPUDevice,
    fps_manager: FpsManager,

    vertex_shader: ?*sdl3.c.SDL_GPUShader,
    fragment_shader: ?*sdl3.c.SDL_GPUShader,

    vertex_buffer: ?*sdl3.c.SDL_GPUBuffer,
    index_buffer: ?*sdl3.c.SDL_GPUBuffer,
    transfer_buffer: ?*sdl3.c.SDL_GPUTransferBuffer,
    image_for_texture: zstbi.Image,
    texture: ?*sdl3.c.SDL_GPUTexture,
    sampler: ?*sdl3.c.SDL_GPUSampler,
    depth_texture: ?*sdl3.c.SDL_GPUTexture,

    obj_data: OBJ.OBJ_Data,
    vertices: []Vertex,
    indices: []u16,

    pipeline: ?*sdl3.c.SDL_GPUGraphicsPipeline,

    paused: bool,

    const DEPTH_TEXTURE_FORMAT = sdl3.c.SDL_GPU_TEXTUREFORMAT_D24_UNORM;
    var indices_len: u32 = 0;
    var rotation: f32 = 0;
    const rotation_speed: f32 = std.math.degreesToRadians(50);

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
    const gpu = sdl3.c.SDL_CreateGPUDevice(sdl3.c.SDL_GPU_SHADERFORMAT_SPIRV, true, null);
    assert(sdl3.c.SDL_ClaimWindowForGPUDevice(gpu, window.value));

    zstbi.init(allocator);

    const vertex_shader = loadShader(
        gpu,
        AppState.vertex_shader_code,
        sdl3.c.SDL_GPU_SHADERSTAGE_VERTEX,
        1,
        0,
    );
    const fragment_shader = loadShader(
        gpu,
        AppState.fragment_shader_code,
        sdl3.c.SDL_GPU_SHADERSTAGE_FRAGMENT,
        0,
        1,
    );
    //Create Obj
    const obj_data = try OBJ.parse(allocator, "./data/race.obj");
    //Create Texture
    var image = try zstbi.Image.loadFromFile("./data/colormap.png", 4);
    const pixels_byte_size = image.width * image.height * 4;

    const texture = sdl3.c.SDL_CreateGPUTexture(gpu, &.{
        .type = sdl3.c.SDL_GPU_TEXTURETYPE_2D,
        .width = image.width,
        .height = image.height,
        .format = sdl3.c.SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
        .usage = sdl3.c.SDL_GPU_TEXTUREUSAGE_SAMPLER,
        .layer_count_or_depth = 1,
        .num_levels = 1,
    });
    const depth_texture = sdl3.c.SDL_CreateGPUTexture(gpu, &.{
        .type = sdl3.c.SDL_GPU_TEXTURETYPE_2D,
        .width = SCREEN_WIDTH,
        .height = SCREEN_HEIGHT,
        .format = AppState.DEPTH_TEXTURE_FORMAT,
        .usage = sdl3.c.SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET,
        .layer_count_or_depth = 1,
        .num_levels = 1,
    });
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
    const vertex_attributes = [_]sdl3.c.SDL_GPUVertexAttribute{
        .{
            .location = 0,
            .format = sdl3.c.SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
            .offset = @offsetOf(Vertex, "pos"),
        },
        .{
            .location = 1,
            .format = sdl3.c.SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4,
            .offset = @offsetOf(Vertex, "color"),
        },
        .{
            .location = 2,
            .format = sdl3.c.SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
            .offset = @offsetOf(Vertex, "uv"),
        },
    };

    const vertex_buffer_descriptions = [_]sdl3.c.SDL_GPUVertexBufferDescription{
        .{
            .slot = 0,
            .pitch = @sizeOf(Vertex),
        },
    };
    // create vertex buffer
    const vertex_buffer = sdl3.c.SDL_CreateGPUBuffer(gpu, &.{
        .usage = sdl3.c.SDL_GPU_BUFFERUSAGE_VERTEX,
        .size = @intCast(vertices_byte_size),
    });
    // create index buffer
    const index_buffer = sdl3.c.SDL_CreateGPUBuffer(gpu, &.{
        .usage = sdl3.c.SDL_GPU_BUFFERUSAGE_INDEX,
        .size = @intCast(indices_byte_size),
    });
    // Create Transfer Buffer
    const transfer_buffer = sdl3.c.SDL_CreateGPUTransferBuffer(gpu, &.{
        .usage = sdl3.c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = @intCast(vertices_byte_size + indices_byte_size),
    });
    const transfer_memory_ptr = sdl3.c.SDL_MapGPUTransferBuffer(gpu, transfer_buffer, false);
    const vertex_dest_ptr: [*]u8 = @ptrCast(transfer_memory_ptr.?);
    const vertex_dest_slice = vertex_dest_ptr[0..vertices_byte_size];
    const index_dest_ptr: [*]u8 = @ptrFromInt(@as(usize, @intFromPtr(vertex_dest_ptr)) + vertices_byte_size);
    const index_dest_slice = index_dest_ptr[0..indices_byte_size];
    const vertex_source_ptr: [*]const u8 = @ptrCast(vertices[0..].ptr);
    const vertex_source_slice = vertex_source_ptr[0..vertices_byte_size];
    const index_source_ptr: [*]const u8 = @ptrCast(indices[0..].ptr);
    const index_source_slice = index_source_ptr[0..indices_byte_size];
    @memcpy(vertex_dest_slice, vertex_source_slice);
    @memcpy(index_dest_slice, index_source_slice);
    sdl3.c.SDL_UnmapGPUTransferBuffer(gpu, transfer_buffer);
    // create texture transfer buffer
    const texture_transfer_buffer = sdl3.c.SDL_CreateGPUTransferBuffer(gpu, &.{
        .usage = sdl3.c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = @intCast(pixels_byte_size),
    });
    defer sdl3.c.SDL_ReleaseGPUTransferBuffer(gpu, texture_transfer_buffer);
    const texture_transfer_memory_ptr = sdl3.c.SDL_MapGPUTransferBuffer(gpu, texture_transfer_buffer, false);
    const texture_dest_ptr: [*]u8 = @ptrCast(texture_transfer_memory_ptr.?);
    const texture_dest_slice = texture_dest_ptr[0..pixels_byte_size];
    const texture_source_ptr: [*]const u8 = @ptrCast(image.data[0..].ptr);
    const texture_source_slice = texture_source_ptr[0..pixels_byte_size];
    @memcpy(texture_dest_slice, texture_source_slice);
    sdl3.c.SDL_UnmapGPUTransferBuffer(gpu, texture_transfer_buffer);

    // - begin copy pass
    const copy_cmd_buffer = sdl3.c.SDL_AcquireGPUCommandBuffer(gpu);
    const copy_pass = sdl3.c.SDL_BeginGPUCopyPass(copy_cmd_buffer);
    // - invoke upload commands
    sdl3.c.SDL_UploadToGPUBuffer(
        copy_pass,
        &.{
            .transfer_buffer = transfer_buffer,
            .offset = 0,
        },
        &.{
            .buffer = vertex_buffer,
            .offset = 0,
            .size = @intCast(vertices_byte_size),
        },
        false,
    );
    sdl3.c.SDL_UploadToGPUBuffer(
        copy_pass,
        &.{
            .transfer_buffer = transfer_buffer,
            .offset = @intCast(vertices_byte_size),
        },
        &.{
            .buffer = index_buffer,
            .offset = 0,
            .size = @intCast(indices_byte_size),
        },
        false,
    );

    sdl3.c.SDL_UploadToGPUTexture(
        copy_pass,
        &.{
            .transfer_buffer = texture_transfer_buffer,
        },
        &.{
            .texture = texture,
            .w = image.width,
            .h = image.height,
            .d = 1,
        },
        false,
    );
    sdl3.c.SDL_EndGPUCopyPass(copy_pass);
    assert(sdl3.c.SDL_SubmitGPUCommandBuffer(copy_cmd_buffer));

    const sampler = sdl3.c.SDL_CreateGPUSampler(gpu, &.{});

    const pipeline = sdl3.c.SDL_CreateGPUGraphicsPipeline(
        gpu,
        &sdl3.c.SDL_GPUGraphicsPipelineCreateInfo{
            .vertex_shader = vertex_shader,
            .fragment_shader = fragment_shader,
            .primitive_type = sdl3.c.SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
            .vertex_input_state = .{
                .num_vertex_buffers = vertex_buffer_descriptions.len,
                .vertex_buffer_descriptions = vertex_buffer_descriptions[0..].ptr,
                .num_vertex_attributes = vertex_attributes.len,
                .vertex_attributes = vertex_attributes[0..].ptr,
            },
            .depth_stencil_state = .{
                .enable_depth_test = true,
                .enable_depth_write = true,
                .compare_op = sdl3.c.SDL_GPU_COMPAREOP_LESS,
            },
            .target_info = .{
                .num_color_targets = 1,
                .color_target_descriptions = @ptrCast(&.{sdl3.c.SDL_GPUColorTargetDescription{
                    .format = sdl3.c.SDL_GetGPUSwapchainTextureFormat(gpu, window.value),
                }}),
                .has_depth_stencil_target = true,
                .depth_stencil_format = AppState.DEPTH_TEXTURE_FORMAT,
            },
        },
    );
    var w_cint: c_int = undefined;
    var h_cint: c_int = undefined;
    assert(sdl3.c.SDL_GetWindowSize(window.value, &w_cint, &h_cint));

    const w = @as(f32, @floatFromInt(w_cint));
    const h = @as(f32, @floatFromInt(h_cint));

    AppState.proj_mat = zmath.perspectiveFovRh(std.math.degreesToRadians(70), w / h, 0.0001, 1000);

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
        else => {
            if (event.unknown.event_type == sdl3.c.SDL_EVENT_KEY_DOWN) {
                app_state.paused = !app_state.paused;
            }
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
        sdl3.c.SDL_ReleaseGPUGraphicsPipeline(val.gpu, val.pipeline);
        sdl3.c.SDL_ReleaseGPUTransferBuffer(val.gpu, val.transfer_buffer);

        sdl3.c.SDL_ReleaseGPUBuffer(val.gpu, val.index_buffer);
        sdl3.c.SDL_ReleaseGPUBuffer(val.gpu, val.vertex_buffer);
        allocator.free(val.indices);
        allocator.free(val.vertices);
        sdl3.c.SDL_ReleaseGPUTexture(val.gpu, val.depth_texture);
        sdl3.c.SDL_ReleaseGPUTexture(val.gpu, val.texture);
        val.image_for_texture.deinit();
        val.obj_data.deinit(allocator);
        sdl3.c.SDL_ReleaseGPUShader(val.gpu, val.fragment_shader);
        sdl3.c.SDL_ReleaseGPUShader(val.gpu, val.vertex_shader);
        // defer sdl3.c.SDL_DestroyGPUDevice(gpu);
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
    const trans = zmath.translation(0.0, -1.0, -3.0);
    const model_mat = zmath.mul(rot, trans);
    const ubo = UBO{
        .mvp = zmath.mul(model_mat, AppState.proj_mat),
    };

    // render
    const cmd_buffer = sdl3.c.SDL_AcquireGPUCommandBuffer(app_state.gpu);
    var swapchain_texture: ?*sdl3.c.SDL_GPUTexture = undefined;
    assert(sdl3.c.SDL_WaitAndAcquireGPUSwapchainTexture(cmd_buffer, app_state.window.value, &swapchain_texture, null, null));

    //Begin render pass
    if (swapchain_texture) |swapchain_tex| {
        const color_target = sdl3.c.SDL_GPUColorTargetInfo{
            .texture = swapchain_tex,
            .load_op = sdl3.c.SDL_GPU_LOADOP_CLEAR,
            .clear_color = .{ .r = 0.1, .g = 0.1, .b = 0.1, .a = 1.0 },
            .store_op = sdl3.c.SDL_GPU_STOREOP_STORE,
        };

        const depth_stencil_target_info = sdl3.c.SDL_GPUDepthStencilTargetInfo{
            .texture = app_state.depth_texture,
            .load_op = sdl3.c.SDL_GPU_LOADOP_CLEAR,
            .clear_depth = 1.0,
            .store_op = sdl3.c.SDL_GPU_STOREOP_DONT_CARE,
        };
        const render_pass = sdl3.c.SDL_BeginGPURenderPass(cmd_buffer, &color_target, 1, &depth_stencil_target_info);

        sdl3.c.SDL_BindGPUGraphicsPipeline(render_pass, app_state.pipeline);
        const bindings = [_]sdl3.c.SDL_GPUBufferBinding{.{ .buffer = app_state.vertex_buffer }};
        sdl3.c.SDL_BindGPUVertexBuffers(render_pass, 0, bindings[0..].ptr, 1);
        const index_binding: sdl3.c.SDL_GPUBufferBinding = .{ .buffer = app_state.index_buffer };
        sdl3.c.SDL_BindGPUIndexBuffer(render_pass, &index_binding, sdl3.c.SDL_GPU_INDEXELEMENTSIZE_16BIT);
        sdl3.c.SDL_PushGPUVertexUniformData(cmd_buffer, 0, &ubo, @sizeOf(@TypeOf(ubo)));
        const fragment_samplers_bindings = [_]sdl3.c.SDL_GPUTextureSamplerBinding{.{ .texture = app_state.texture, .sampler = app_state.sampler }};
        sdl3.c.SDL_BindGPUFragmentSamplers(render_pass, 0, fragment_samplers_bindings[0..].ptr, 1);
        sdl3.c.SDL_DrawGPUIndexedPrimitives(render_pass, AppState.indices_len, 1, 0, 0, 0);
        sdl3.c.SDL_EndGPURenderPass(render_pass);
    }
    assert(sdl3.c.SDL_SubmitGPUCommandBuffer(cmd_buffer));
    return .run;
}

fn loadShader(device: ?*sdl3.c.SDL_GPUDevice, code: []const u8, stage: sdl3.c.SDL_GPUShaderStage, num_uniform_buffers: u32, num_samplers: u32) ?*sdl3.c.SDL_GPUShader {
    return sdl3.c.SDL_CreateGPUShader(
        device,
        &sdl3.c.SDL_GPUShaderCreateInfo{
            .code_size = code.len,
            .code = @ptrCast(code),
            .entrypoint = "main",
            .format = sdl3.c.SDL_GPU_SHADERFORMAT_SPIRV,
            .stage = stage,
            .num_uniform_buffers = num_uniform_buffers,
            .num_samplers = num_samplers,
        },
    );
}
