const std = @import("std");
const sdl3 = @import("sdl3");
const zmath = @import("zmath");
const FpsManager = @import("FpsManager.zig");

const SCREEN_WIDTH = 1280;
const SCREEN_HEIGHT = 780;

const assert = std.debug.assert;

const vertex_shader_code = @embedFile("shaders/source/shader.spv.vert");
const fragment_shader_code = @embedFile("shaders/source/shader.spv.frag");

const UBO = struct {
    mvp: zmath.Mat,
};

pub fn main() !void {
    defer sdl3.init.shutdown();

    sdl3.log.setAllPriorities(.verbose);

    const init_flags = sdl3.init.Flags{ .video = true };
    try sdl3.init.init(init_flags);
    defer sdl3.init.quit(init_flags);

    const window = try sdl3.video.Window.init("GPU programming", SCREEN_WIDTH, SCREEN_HEIGHT, .{ .resizable = true });
    defer window.deinit();

    var fps_manager = FpsManager.init(.none);

    const gpu = sdl3.c.SDL_CreateGPUDevice(sdl3.c.SDL_GPU_SHADERFORMAT_SPIRV, true, null);
    // defer sdl3.c.SDL_DestroyGPUDevice(gpu);

    assert(sdl3.c.SDL_ClaimWindowForGPUDevice(gpu, window.value));

    const vertex_shader = loadShader(gpu, vertex_shader_code, sdl3.c.SDL_GPU_SHADERSTAGE_VERTEX, 1);
    defer sdl3.c.SDL_ReleaseGPUShader(gpu, vertex_shader);
    const fragment_shader = loadShader(gpu, fragment_shader_code, sdl3.c.SDL_GPU_SHADERSTAGE_FRAGMENT, 0);
    defer sdl3.c.SDL_ReleaseGPUShader(gpu, fragment_shader);

    // create vertex data

    const Vec3 = @Vector(3, f32);
    const Color = @Vector(4, f32);
    const Vertex = struct {
        pos: Vec3,
        color: Color,
    };

    const vertices = [_]Vertex{
        .{ .pos = .{ -1.0, 1.0, 0.0 }, .color = .{ 1.0, 0.0, 0.0, 1.0 } }, //top left
        .{ .pos = .{ 1.0, 1.0, 0.0 }, .color = .{ 0.0, 1.0, 1.0, 1.0 } }, //top right
        .{ .pos = .{ -1.0, -1.0, 0.0 }, .color = .{ 1.0, 0.0, 1.0, 1.0 } }, //bottom left
        .{ .pos = .{ 1.0, -1.0, 0.0 }, .color = .{ 1.0, 0.0, 1.0, 1.0 } }, //bottom right
    };
    const vertices_byte_size = vertices.len * @sizeOf(@TypeOf(vertices[0]));

    const indices = [_]u16{
        0, 1, 2,
        2, 1, 3,
    };
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

    // NOTE: upload vertex data to vertex buffer

    // - create transfer buffer for uploading both vertex and index buffers
    const transfer_buffer = sdl3.c.SDL_CreateGPUTransferBuffer(gpu, &.{
        .usage = sdl3.c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = @intCast(vertices_byte_size + indices_byte_size),
    });
    defer sdl3.c.SDL_ReleaseGPUTransferBuffer(gpu, transfer_buffer);
    // - map transfer buffer mem to copy from cpu
    const transfer_memory_ptr = sdl3.c.SDL_MapGPUTransferBuffer(gpu, transfer_buffer, false);
    //Copy the fucking data to the transfer_memory_ptr--------------------------------------
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
    //Copy the fucking data to the transfer_memory_ptr--------------------------------------

    // _ = sdl3.c.SDL_memcpy(transfer_memory_ptr, vertices[0..].ptr, vertices_byte_size);
    sdl3.c.SDL_UnmapGPUTransferBuffer(gpu, transfer_buffer);

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
            .offset = vertices_byte_size,
        },
        &.{
            .buffer = index_buffer,
            .offset = 0,
            .size = @intCast(indices_byte_size),
        },
        false,
    );
    // - end copy pass and submit
    sdl3.c.SDL_EndGPUCopyPass(copy_pass);
    assert(sdl3.c.SDL_SubmitGPUCommandBuffer(copy_cmd_buffer));

    // bind vertex buffer to draw call // NOTE: Done in draw call loop

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
            .target_info = .{
                .num_color_targets = 1,
                .color_target_descriptions = @ptrCast(&.{sdl3.c.SDL_GPUColorTargetDescription{
                    .format = sdl3.c.SDL_GetGPUSwapchainTextureFormat(gpu, window.value),
                }}),
            },
        },
    );

    //release shaders after pipline ceration
    // did in defer
    //

    var w_cint: c_int = undefined;
    var h_cint: c_int = undefined;
    assert(sdl3.c.SDL_GetWindowSize(window.value, &w_cint, &h_cint));

    const w = @as(f32, @floatFromInt(w_cint));
    const h = @as(f32, @floatFromInt(h_cint));

    var rotation: f32 = 0;
    const rotation_speed: f32 = std.math.degreesToRadians(90);

    const proj_mat = zmath.perspectiveFovRh(std.math.degreesToRadians(70), w / h, 0.0001, 1000);

    while (true) {
        const event = sdl3.events.poll();
        if (event) |e| {
            switch (e) {
                .quit => break,
                .terminating => break,
                else => {},
            }
        }
        // Update game state
        fps_manager.tick();

        rotation += rotation_speed * fps_manager.getDelta();
        const rot = zmath.rotationY(rotation);
        const trans = zmath.translation(0.0, 0.0, -5.0);
        const model_mat = zmath.mul(rot, trans);
        const ubo = UBO{
            .mvp = zmath.mul(model_mat, proj_mat),
        };

        // std.debug.print("Delta:{d} FPS:{d}\n", .{ fps_manager.getDelta(), fps_manager.getFps() });

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

            const render_pass = sdl3.c.SDL_BeginGPURenderPass(cmd_buffer, &color_target, 1, null);

            // Render_pass--------------------------------
            // Draw
            // - bind pipline
            sdl3.c.SDL_BindGPUGraphicsPipeline(render_pass, pipline);

            //Data to be passed to shader
            // Vertex attribute - per vertex
            // Uniform data - whole draw call

            // - bind vertex data

            // bind vertex buffer to draw call // NOTE: Done in draw call loop
            const bindings = [_]sdl3.c.SDL_GPUBufferBinding{.{ .buffer = vertex_buffer }};
            sdl3.c.SDL_BindGPUVertexBuffers(render_pass, 0, bindings[0..].ptr, 1);
            const index_binding: sdl3.c.SDL_GPUBufferBinding = .{ .buffer = index_buffer };
            sdl3.c.SDL_BindGPUIndexBuffer(render_pass, &index_binding, sdl3.c.SDL_GPU_INDEXELEMENTSIZE_16BIT);

            // - bind uniform data
            sdl3.c.SDL_PushGPUVertexUniformData(cmd_buffer, 0, &ubo, @sizeOf(@TypeOf(ubo)));
            // - draw calls
            // sdl3.c.SDL_DrawGPUPrimitives(render_pass, 3, 1, 0, 0);
            sdl3.c.SDL_DrawGPUIndexedPrimitives(render_pass, indices.len, 1, 0, 0, 0);
            //end render pass
            sdl3.c.SDL_EndGPURenderPass(render_pass);
        }
        //submit
        assert(sdl3.c.SDL_SubmitGPUCommandBuffer(cmd_buffer));
    }
}

fn loadShader(device: ?*sdl3.c.SDL_GPUDevice, code: []const u8, stage: sdl3.c.SDL_GPUShaderStage, num_uniform_buffers: u32) ?*sdl3.c.SDL_GPUShader {
    return sdl3.c.SDL_CreateGPUShader(
        device,
        &sdl3.c.SDL_GPUShaderCreateInfo{
            .code_size = code.len,
            .code = @ptrCast(code),
            .entrypoint = "main",
            .format = sdl3.c.SDL_GPU_SHADERFORMAT_SPIRV,
            .stage = stage,
            .num_uniform_buffers = num_uniform_buffers,
        },
    );
}
