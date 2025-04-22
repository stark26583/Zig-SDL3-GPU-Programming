const std = @import("std");
const sdl3 = @import("sdl3");
const zmath = @import("zmath");
const zstbi = @import("zstbi");
const FpsManager = @import("FpsManager.zig");
const OBJ = @import("OBJ.zig");

const CommonTypes = @import("CommonTypes.zig");
const Vec3 = CommonTypes.Vec3;
const Color = CommonTypes.Color;
const Vertex = CommonTypes.Vertex;

const SCREEN_WIDTH = 1280;
const SCREEN_HEIGHT = 780;

const assert = std.debug.assert;

const vertex_shader_code = @embedFile("shaders/compiled/shader.spv.vert");
const fragment_shader_code = @embedFile("shaders/compiled/shader.spv.frag");

const UBO = struct {
    mvp: zmath.Mat,
};

pub fn main() !void {
    defer sdl3.init.shutdown();

    sdl3.log.setAllPriorities(.verbose);

    const init_flags = sdl3.init.Flags{ .video = true };
    try sdl3.init.init(init_flags);
    defer sdl3.init.quit(init_flags);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leaked = gpa.deinit();
        assert(leaked == .ok);
    }
    const allocator = gpa.allocator();

    zstbi.init(allocator);
    defer zstbi.deinit();

    const window = try sdl3.video.Window.init("GPU programming", SCREEN_WIDTH, SCREEN_HEIGHT, .{});
    defer window.deinit();

    var fps_manager = FpsManager.init(.none);

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

    const gpu = sdl3.c.SDL_CreateGPUDevice(sdl3.c.SDL_GPU_SHADERFORMAT_SPIRV, true, null);
    // defer sdl3.c.SDL_DestroyGPUDevice(gpu);

    assert(sdl3.c.SDL_ClaimWindowForGPUDevice(gpu, window.value));

    const vertex_shader = loadShader(
        gpu,
        vertex_shader_code,
        sdl3.c.SDL_GPU_SHADERSTAGE_VERTEX,
        1,
        0,
    );
    defer sdl3.c.SDL_ReleaseGPUShader(gpu, vertex_shader);
    const fragment_shader = loadShader(
        gpu,
        fragment_shader_code,
        sdl3.c.SDL_GPU_SHADERSTAGE_FRAGMENT,
        0,
        1,
    );
    defer sdl3.c.SDL_ReleaseGPUShader(gpu, fragment_shader);

    //Create Obj
    const obj_data = try OBJ.parse(allocator, "./data/race.obj");
    defer obj_data.deinit(allocator);

    //Create Texture
    var image = try zstbi.Image.loadFromFile("./data/colormap.png", 4);
    defer image.deinit();
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

    const DEPTH_TEXTURE_FORMAT = sdl3.c.SDL_GPU_TEXTUREFORMAT_D24_UNORM;
    const depth_texture = sdl3.c.SDL_CreateGPUTexture(gpu, &.{
        .type = sdl3.c.SDL_GPU_TEXTURETYPE_2D,
        .width = SCREEN_WIDTH,
        .height = SCREEN_HEIGHT,
        .format = DEPTH_TEXTURE_FORMAT,
        .usage = sdl3.c.SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET,
        .layer_count_or_depth = 1,
        .num_levels = 1,
    });
    defer sdl3.c.SDL_ReleaseGPUTexture(gpu, depth_texture);

    // create vertex data
    const White = Color{ 1.0, 1.0, 1.0, 1.0 };

    const vertices = try allocator.alloc(Vertex, obj_data.faces.len);
    defer allocator.free(vertices);
    const indices = try allocator.alloc(u16, obj_data.faces.len);
    defer allocator.free(indices);

    for (obj_data.faces, 0..) |faces, i| {
        const uv = obj_data.uvs_tex_coords[faces.uv_index];
        vertices[i] = .{
            .pos = obj_data.positions[faces.position_index],
            .color = White,
            .uv = .{ uv[0], 1 - uv[1] },
        };

        indices[i] = @intCast(i);
    }

    const indices_len: u32 = @intCast(indices.len);

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

    const transfer_buffer = sdl3.c.SDL_CreateGPUTransferBuffer(gpu, &.{
        .usage = sdl3.c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = @intCast(vertices_byte_size + indices_byte_size),
    });
    defer sdl3.c.SDL_ReleaseGPUTransferBuffer(gpu, transfer_buffer);
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

    //--------------------------
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

    const pipline = sdl3.c.SDL_CreateGPUGraphicsPipeline(
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
                .depth_stencil_format = DEPTH_TEXTURE_FORMAT,
            },
        },
    );

    var w_cint: c_int = undefined;
    var h_cint: c_int = undefined;
    assert(sdl3.c.SDL_GetWindowSize(window.value, &w_cint, &h_cint));

    const w = @as(f32, @floatFromInt(w_cint));
    const h = @as(f32, @floatFromInt(h_cint));

    var rotation: f32 = 0;
    const rotation_speed: f32 = std.math.degreesToRadians(50);

    const proj_mat = zmath.perspectiveFovRh(std.math.degreesToRadians(70), w / h, 0.0001, 1000);

    var paused = false;
    while (true) {
        const event = sdl3.events.poll();
        if (event) |e| {
            switch (e) {
                .quit => break,
                .terminating => break,
                else => {
                    if (e.unknown.event_type == sdl3.c.SDL_EVENT_KEY_DOWN) {
                        paused = !paused;
                    }
                },
            }
        }
        // Update game state
        fps_manager.tick();

        if (!paused) rotation += rotation_speed * fps_manager.getDelta();
        const rot = zmath.rotationY(rotation);
        const trans = zmath.translation(0.0, -1.0, -3.0);
        const model_mat = zmath.mul(rot, trans);
        const ubo = UBO{
            .mvp = zmath.mul(model_mat, proj_mat),
        };

        // render
        const cmd_buffer = sdl3.c.SDL_AcquireGPUCommandBuffer(gpu);
        var swapchain_texture: ?*sdl3.c.SDL_GPUTexture = undefined;
        assert(sdl3.c.SDL_WaitAndAcquireGPUSwapchainTexture(cmd_buffer, window.value, &swapchain_texture, null, null));

        //Begin render pass
        if (swapchain_texture) |swapchain_tex| {
            const color_target = sdl3.c.SDL_GPUColorTargetInfo{
                .texture = swapchain_tex,
                .load_op = sdl3.c.SDL_GPU_LOADOP_CLEAR,
                .clear_color = .{ .r = 0.1, .g = 0.1, .b = 0.1, .a = 1.0 },
                .store_op = sdl3.c.SDL_GPU_STOREOP_STORE,
            };

            const depth_stencil_target_info = sdl3.c.SDL_GPUDepthStencilTargetInfo{
                .texture = depth_texture,
                .load_op = sdl3.c.SDL_GPU_LOADOP_CLEAR,
                .clear_depth = 1.0,
                .store_op = sdl3.c.SDL_GPU_STOREOP_DONT_CARE,
            };
            const render_pass = sdl3.c.SDL_BeginGPURenderPass(cmd_buffer, &color_target, 1, &depth_stencil_target_info);

            sdl3.c.SDL_BindGPUGraphicsPipeline(render_pass, pipline);
            const bindings = [_]sdl3.c.SDL_GPUBufferBinding{.{ .buffer = vertex_buffer }};
            sdl3.c.SDL_BindGPUVertexBuffers(render_pass, 0, bindings[0..].ptr, 1);
            const index_binding: sdl3.c.SDL_GPUBufferBinding = .{ .buffer = index_buffer };
            sdl3.c.SDL_BindGPUIndexBuffer(render_pass, &index_binding, sdl3.c.SDL_GPU_INDEXELEMENTSIZE_16BIT);
            sdl3.c.SDL_PushGPUVertexUniformData(cmd_buffer, 0, &ubo, @sizeOf(@TypeOf(ubo)));
            const fragment_samplers_bindings = [_]sdl3.c.SDL_GPUTextureSamplerBinding{.{ .texture = texture, .sampler = sampler }};
            sdl3.c.SDL_BindGPUFragmentSamplers(render_pass, 0, fragment_samplers_bindings[0..].ptr, 1);
            sdl3.c.SDL_DrawGPUIndexedPrimitives(render_pass, indices_len, 1, 0, 0, 0);
            sdl3.c.SDL_EndGPURenderPass(render_pass);
        }
        assert(sdl3.c.SDL_SubmitGPUCommandBuffer(cmd_buffer));
    }
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
