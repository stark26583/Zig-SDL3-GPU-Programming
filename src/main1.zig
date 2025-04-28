const std = @import("std");
const sdl3 = @import("sdl3");
const zmath = @import("zmath");
const zstbi = @import("zstbi");
const zgui = @import("zgui");
const zgui_sdl = zgui.backend;
const FpsManager = @import("FpsManager.zig");
const Scancode = sdl3.Scancode;
const OBJ = @import("OBJ.zig");
const gpu = sdl3.gpu;

const CommonTypes = @import("CommonTypes.zig");
const Vec3 = CommonTypes.Vec3;
const Color = CommonTypes.Color;
const Vertex = CommonTypes.Vertex;
const UBO = CommonTypes.UBO;

const assert = std.debug.assert;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leaked = gpa.deinit();
        assert(leaked == .ok);
    }

    var app = try App.init(gpa.allocator());
    defer app.deinit();

    try app.run();
}

//------------------------------------------------------------
// Application Context
const App = struct {
    const SCREEN_WIDTH = 1280;
    const SCREEN_HEIGHT = 780;
    const vertex_shader_code = @embedFile("shaders/compiled/shader.spv.vert");
    const fragment_shader_code = @embedFile("shaders/compiled/shader.spv.frag");
    const DEPTH_TEXTURE_FORMAT = gpu.TextureFormat.depth24_unorm;
    // var font: zgui.Font = undefined;

    var proj_mat: zmath.Mat = undefined;

    allocator: std.mem.Allocator,
    window: sdl3.video.Window,
    device: gpu.Device,
    fps_manager: FpsManager,
    paused: bool,

    // Shaders & pipeline
    vertex_shader: gpu.Shader,
    fragment_shader: gpu.Shader,
    pipeline: gpu.GraphicsPipeline,
    depth_texture: gpu.Texture,

    // Camera / UBO
    ubo: UBO,
    camera: Camera,

    // Model
    model: Model,

    fn run(self: *App) !void {
        while (true) {
            if (!self.events()) break;

            self.update(self.fps_manager.getDelta());
            try self.render(.{ .r = 0.0, .g = 0.0, .b = 0.0, .a = 1.0 });
        }
    }

    pub fn init(allocator: std.mem.Allocator) !App {
        try sdl3.init.init(.{ .video = true, .events = true });
        sdl3.log.setAllPriorities(.info);
        zstbi.init(allocator);
        zgui.init(allocator);

        const window = try sdl3.video.Window.init("GPU programming", SCREEN_WIDTH, SCREEN_HEIGHT, .{});
        const device = try gpu.Device.init(.{ .spirv = true }, true, null);
        try device.claimWindow(window);

        const vertex_shader = try loadShader(
            device,
            vertex_shader_code,
            .vertex,
            1,
            0,
        );
        const fragment_shader = try loadShader(
            device,
            fragment_shader_code,
            .fragment,
            0,
            1,
        );
        const depth_texture = try device.createTexture(.{
            .texture_type = .two_dimensional, //default
            .width = SCREEN_WIDTH,
            .height = SCREEN_HEIGHT,
            .format = App.DEPTH_TEXTURE_FORMAT,
            .usage = .{ .depth_stencil_target = true },
            .layer_count_or_depth = 1,
            .num_levels = 1,
        });

        const pipeline = try setup_pipline(device, window, vertex_shader, fragment_shader, App.DEPTH_TEXTURE_FORMAT);

        zgui_sdl.init(window.value, .{
            .device = device.value,
            .color_target_format = @intFromEnum(device.getSwapchainTextureFormat(window)),
            .msaa_samples = 0,
        });

        // font = zgui.io.addFontFromFile(
        //     "./fonts/Candara.ttf",
        //     40,
        // );

        const w: f32 = @floatFromInt(SCREEN_WIDTH);
        const h: f32 = @floatFromInt(SCREEN_HEIGHT);
        App.proj_mat = zmath.perspectiveFovRh(std.math.degreesToRadians(70), w / h, 0.0001, 1000);

        return App{
            .allocator = allocator,
            .window = window,
            .device = device,
            .fps_manager = FpsManager.init(.none),
            .paused = false,
            .vertex_shader = vertex_shader,
            .fragment_shader = fragment_shader,
            .pipeline = pipeline,
            .depth_texture = depth_texture,
            .ubo = UBO{ .mvp = zmath.identity() },
            .model = try Model.load(device, allocator, "./data/ambulance.obj", "data/colormap.png"),
            .camera = Camera{
                .position = .{ 0, 1, 3, 1 },
                .target = .{ 0, 1, 0, 1 },
                .view = zmath.identity(),
            },
        };
    }

    fn render(self: *App, clear_color: sdl3.pixels.FColor) !void {
        //---------------------------------------------------------------------------------------
        zgui_sdl.newFrame(SCREEN_WIDTH, SCREEN_HEIGHT, 1);
        // zgui.text("DepthTexture.format = {any}\n", .{self.depth_texture});
        // zgui.text("Pipeline target_info.depth_stencil_format = {any}\n", .{App.DEPTH_TEXTURE_FORMAT});
        // zgui.text("window swapchain texture format {any}", .{self.device.getSwapchainTextureFormat(self.window)});
        zgui.showDemoWindow(null);
        //---------------------------------------------------------------------------------------

        const cmd_buffer = try self.device.aquireCommandBuffer();
        const swapchain = try cmd_buffer.waitAndAquireSwapchainTexture(self.window);

        zgui.render(); //--------------------Render Zgui----------------------

        if (swapchain.texture) |swapchain_tex| {
            //Begin render pass
            const color_target = gpu.ColorTargetInfo{
                .texture = swapchain_tex,
                .load = .clear,
                .clear_color = clear_color,
            };

            const depth_target = gpu.DepthStencilTargetInfo{
                .texture = self.depth_texture,
                .load = .clear,
                .store = .do_not_care,
                .stencil_load = @enumFromInt(0),
                .stencil_store = @enumFromInt(0),
                .clear_stencil = 0,
                .clear_depth = 1,
                .cycle = false,
            };

            self.push_uniform_data_to_gpu(cmd_buffer);

            const render_pass = cmd_buffer.beginRenderPass(&.{color_target}, depth_target);
            render_pass.bindGraphicsPipeline(self.pipeline);
            self.model.render(render_pass);

            render_pass.end();
            //----------------------Zgui RenderPass----------------------
            zgui_sdl.prepareDrawData(cmd_buffer.value);
            const zgui_color_target = gpu.ColorTargetInfo{
                .texture = swapchain_tex,
                .load = .load,
            };
            const zgui_render_pass = cmd_buffer.beginRenderPass(&.{zgui_color_target}, null);
            zgui_sdl.renderDrawData(cmd_buffer.value, zgui_render_pass.value, null);
            // zgui_sdl.render();
            zgui_render_pass.end();
            //------------------------------------------------------------
        }
        try cmd_buffer.submit();
    }

    fn events(self: *App) bool {
        _ = self;
        var event: sdl3.c.SDL_Event = undefined;
        while (sdl3.c.SDL_PollEvent(&event)) {
            const zgui_event = zgui_sdl.processEvent(&event); //----------------------------------
            _ = zgui_event;
            switch (event.type) {
                sdl3.c.SDL_EVENT_QUIT => {
                    return false;
                },
                sdl3.c.SDL_EVENT_TERMINATING => {
                    return false;
                },
                sdl3.c.SDL_EVENT_KEY_DOWN => {
                    const scancode = Scancode{ .value = @intCast(event.key.scancode) };
                    if (Scancode.space.matches(scancode)) {
                        Model.paused = !Model.paused;
                        std.debug.print("Mass space\n", .{});
                    }
                    if (Scancode.w.matches(scancode)) {
                        Camera.w_pressed = true;
                    }
                    if (Scancode.a.matches(scancode)) {
                        Camera.a_pressed = true;
                    }
                    if (Scancode.s.matches(scancode)) {
                        Camera.s_pressed = true;
                    }
                    if (Scancode.d.matches(scancode)) {
                        Camera.d_pressed = true;
                    }
                },
                sdl3.c.SDL_EVENT_KEY_UP => {
                    const scancode = Scancode{ .value = @intCast(event.key.scancode) };
                    if (Scancode.w.matches(scancode)) {
                        Camera.w_pressed = false;
                    }
                    if (Scancode.a.matches(scancode)) {
                        Camera.a_pressed = false;
                    }
                    if (Scancode.s.matches(scancode)) {
                        Camera.s_pressed = false;
                    }
                    if (Scancode.d.matches(scancode)) {
                        Camera.d_pressed = false;
                    }
                },
                else => {},
            }
        }
        return true;
    }

    fn update_mvp(self: *App) void {
        const mv = zmath.mul(self.model.mat, self.camera.view);
        const mvp = zmath.mul(mv, App.proj_mat);
        self.ubo.mvp = mvp;
    }

    fn update(self: *App, delta: f32) void {
        self.fps_manager.tick();
        self.model.update(delta);
        self.camera.update(.boring, delta);
        self.update_mvp();
    }

    fn deinit(self: *App) void {
        self.device.releaseGraphicsPipeline(self.pipeline);
        self.model.unload(self.device);
        self.device.releaseTexture(self.depth_texture);
        self.device.releaseShader(self.fragment_shader);
        self.device.releaseShader(self.vertex_shader);
        self.device.releaseWindow(self.window);
        zgui_sdl.deinit();
        self.device.deinit();
        self.window.deinit();
        zstbi.deinit();

        zgui.deinit();
        sdl3.init.quit(.{ .video = true, .events = true });
    }

    fn push_uniform_data_to_gpu(self: *App, cmd_buffer: gpu.CommandBuffer) void {
        cmd_buffer.pushVertexUniformData(0, std.mem.asBytes(&self.ubo));
    }
};

const CameraUpdateMode = enum {
    none,
    drone,
    boring,
};

var angle: f32 = 0.0;
const Camera = struct {
    var w_pressed: bool = false;
    var a_pressed: bool = false;
    var s_pressed: bool = false;
    var d_pressed: bool = false;

    // var mouse

    var vel: f32 = 5;
    const eye_height: f32 = 1;
    position: zmath.Vec,
    target: zmath.Vec,
    up: zmath.Vec = .{ 0, eye_height, 0, 1 },
    view: zmath.Mat,

    fn update(self: *Camera, mode: CameraUpdateMode, delta: f32) void {
        switch (mode) {
            .none => {},
            .drone => {},
            .boring => {
                if (w_pressed) {
                    self.position[2] -= vel * delta;
                    self.target[2] -= vel * delta;
                }
                if (a_pressed) {
                    self.position[0] -= vel * delta;
                    self.target[0] -= vel * delta;
                }
                if (s_pressed) {
                    self.position[2] += vel * delta;
                    self.target[2] += vel * delta;
                }
                if (d_pressed) {
                    self.position[0] += vel * delta;
                    self.target[0] += vel * delta;
                }
            },
        }
        self.view = zmath.lookAtRh(self.position, self.target, self.up);
    }
};

const Model = struct {
    var paused = false;
    vertex_buffer: gpu.Buffer,
    index_buffer: gpu.Buffer,
    texture: gpu.Texture,
    sampler: gpu.Sampler,
    index_count: u32,

    mat: zmath.Mat = undefined,

    rotation: f32 = 0,
    rotation_speed: f32 = std.math.degreesToRadians(90),

    pub fn load(device: gpu.Device, allocator: std.mem.Allocator, obj_path: []const u8, texture_path: [:0]const u8) !Model {
        //Create Obj
        const obj_data = try OBJ.parse(allocator, obj_path);
        defer obj_data.deinit(allocator);

        //Create Texture
        var image = try zstbi.Image.loadFromFile(texture_path, 4);
        defer image.deinit();
        const pixels_byte_size = image.width * image.height * 4;

        const texture = try device.createTexture(.{
            // .texture_type = .two_dimensional //default
            .width = @intCast(image.width),
            .height = @intCast(image.height),
            .format = .r8g8b8a8_unorm,
            .usage = .{ .sampler = true },
            .layer_count_or_depth = 1,
            .num_levels = 1,
        });
        // defer device.releaseTexture(texture);

        const sampler = try device.createSampler(.{});

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

        // create vertex buffer
        const vertex_buffer = try device.createBuffer(.{
            .usage = .{ .vertex = true },
            .size = @intCast(vertices_byte_size),
        });
        // defer device.releaseBuffer(vertex_buffer);
        // create index buffer
        const index_buffer = try device.createBuffer(.{
            .usage = .{ .index = true },
            .size = @intCast(indices_byte_size),
        });
        // defer device.releaseBuffer(index_buffer);

        const transfer_buffer = try device.createTransferBuffer(.{
            .usage = .upload,
            .size = @intCast(vertices_byte_size + indices_byte_size),
        });
        defer device.releaseTransferBuffer(transfer_buffer);
        const map_tb = (try device.mapTransferBuffer(transfer_buffer, false));
        memcpy_into_transfer_buff(map_tb, vertices, vertices_byte_size);
        memcpy_into_transfer_buff(@ptrFromInt(@as(usize, @intFromPtr(map_tb)) + vertices_byte_size), indices, indices_byte_size);
        device.unmapTransferBuffer(transfer_buffer);

        const texture_transfer_buffer = try device.createTransferBuffer(.{
            .usage = .upload,
            .size = @intCast(pixels_byte_size),
        });
        defer device.releaseTransferBuffer(texture_transfer_buffer);
        const map_ttb = (try device.mapTransferBuffer(texture_transfer_buffer, false));
        memcpy_into_transfer_buff(map_ttb, image.data, pixels_byte_size);
        device.unmapTransferBuffer(texture_transfer_buffer);

        //--------------------------
        // - begin copy pass
        const copy_cmd_buffer = try device.aquireCommandBuffer();
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

        copy_pass.uploadToTexture(
            .{
                .transfer_buffer = texture_transfer_buffer,
                .offset = 0,
                .pixels_per_row = 0,
                .rows_per_layer = 0,
            },
            .{
                .texture = texture,
                .width = image.width,
                .height = image.height,
                .depth = 1,
                .mip_level = 0,
                .x = 0,
                .y = 0,
                .z = 0,
                .layer = 0,
            },
            false,
        );
        copy_pass.end();
        try copy_cmd_buffer.submit();

        return Model{
            .vertex_buffer = vertex_buffer,
            .index_buffer = index_buffer,
            .texture = texture,
            .sampler = sampler,
            .index_count = indices_len,
        };
    }

    fn unload(self: Model, device: gpu.Device) void {
        device.releaseTexture(self.texture);
        device.releaseSampler(self.sampler);
        device.releaseBuffer(self.index_buffer);
        device.releaseBuffer(self.vertex_buffer);
    }

    fn render(self: Model, render_pass: gpu.RenderPass) void {
        const vertex_bindings = [_]gpu.BufferBinding{.{ .buffer = self.vertex_buffer, .offset = 0 }};
        render_pass.bindVertexBuffers(0, vertex_bindings[0..]);
        render_pass.bindIndexBuffer(.{ .buffer = self.index_buffer, .offset = 0 }, .indices_16bit);
        const fragment_samplers_bindings = [_]gpu.TextureSamplerBinding{.{ .texture = self.texture, .sampler = self.sampler }};
        render_pass.bindFragmentSamplers(0, fragment_samplers_bindings[0..]);
        render_pass.drawIndexedPrimitives(self.index_count, 1, 0, 0, 0);
    }

    fn update(self: *Model, delta: f32) void {
        if (!paused) self.rotation += self.rotation_speed * delta;
        const rot = zmath.rotationY(self.rotation);
        const trans = zmath.translation(0, 0, 0);
        const model_mat = zmath.mul(rot, trans);
        self.mat = model_mat;
    }
};

fn setup_pipline(device: gpu.Device, window: sdl3.video.Window, vertex_shader: gpu.Shader, fragment_shader: gpu.Shader, depth_texture_format: gpu.TextureFormat) !gpu.GraphicsPipeline {
    const vertex_attributes = [_]gpu.VertexAttribute{
        gpu.VertexAttribute{
            .location = 0,
            .format = .f32x3,
            .offset = @offsetOf(Vertex, "pos"),
            .buffer_slot = 0,
        },
        gpu.VertexAttribute{
            .location = 1,
            .format = .f32x3,
            .offset = @offsetOf(Vertex, "color"),
            .buffer_slot = 0,
        },
        gpu.VertexAttribute{
            .location = 2,
            .format = .f32x3,
            .offset = @offsetOf(Vertex, "uv"),
            .buffer_slot = 0,
        },
    };

    const vertex_buffer_descriptions = [_]gpu.VertexBufferDescription{
        gpu.VertexBufferDescription{
            .slot = 0,
            .pitch = @sizeOf(Vertex),
            .input_rate = .vertex,
            .instance_step_rate = 0,
        },
    };

    return try device.createGraphicsPipeline(
        .{
            .vertex_shader = vertex_shader,
            .fragment_shader = fragment_shader,
            .primitive_type = .triangle_list,
            .vertex_input_state = .{
                .vertex_buffer_descriptions = vertex_buffer_descriptions[0..],
                .vertex_attributes = vertex_attributes[0..],
            },
            .depth_stencil_state = .{
                .enable_depth_test = true,
                .enable_depth_write = true,
                .compare = .less,
            },
            .rasterizer_state = .{ .cull_mode = .back },
            .target_info = .{
                .color_target_descriptions = &[_]gpu.ColorTargetDescription{
                    .{
                        .format = device.getSwapchainTextureFormat(window),
                    },
                },
                .depth_stencil_format = depth_texture_format,
            },
        },
    );
}

fn loadShader(device: gpu.Device, code: []const u8, stage: gpu.ShaderStage, num_uniform_buffers: u32, num_samplers: u32) !gpu.Shader {
    return try device.createShader(.{
        .code = code,
        .entry_point = "main",
        .format = .{ .spirv = true },
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
