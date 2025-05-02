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

const assert = std.debug.assert;

// pub fn main() !void {
//     defer {
//         const leaked = gpa.deinit();
//         assert(leaked == .ok);
//     }
//
//     defer app.deinit();
//
//     try app.run();
// }

//------------------------------------------------------------
// Application Context
const App = struct {
    const SCREEN_WIDTH = 1680;
    const SCREEN_HEIGHT = 880;
    const vertex_shader_code = @embedFile("shaders/compiled/shader.spv.vert");
    const fragment_shader_code = @embedFile("shaders/compiled/shader.spv.frag");
    const DEPTH_TEXTURE_FORMAT = gpu.TextureFormat.depth24_unorm;
    var driver_name: [:0]const u8 = undefined;

    var clear_color: Vec3 = .{ 0, 0, 0 };

    var proj_mat: zmath.Mat = undefined;
    var present_mode: gpu.PresentMode = .vsync;
    var scancode: c_uint = 0;
    var event: sdl3.c.SDL_Event = undefined;

    init_flags: sdl3.init.Flags,

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

    // fn run(self: *App) !void {
    //     while (true) {
    //         if (!self.events()) break;
    //
    //         self.update(self.fps_manager.getDelta());
    //         try self.render();
    //     }
    // }

    pub fn init() !App {
        const init_flags = sdl3.init.Flags{ .video = true, .events = true };
        try sdl3.init.init(init_flags);
        zgui.init(allocator);
        zstbi.init(allocator);

        const window = try sdl3.video.Window.init("GPU programming", SCREEN_WIDTH, SCREEN_HEIGHT, .{});

        const device = try gpu.Device.init(.{ .spirv = true }, true, null);
        try device.claimWindow(window);

        driver_name = try device.getDriver();

        _ = zgui.io.addFontFromFile("./fonts/Candara.ttf", std.math.floor(21 * try window.getDisplayScale()));
        zgui_sdl.init(window.value, .{
            .device = device.value,
            .color_target_format = @intFromEnum(device.getSwapchainTextureFormat(window)),
            .msaa_samples = 0,
        });

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

        const w: f32 = @floatFromInt(SCREEN_WIDTH);
        const h: f32 = @floatFromInt(SCREEN_HEIGHT);
        App.proj_mat = zmath.perspectiveFovRh(std.math.degreesToRadians(70), w / h, 0.0001, 1000);

        return App{
            .init_flags = init_flags,
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
            .model = try Model.load(device, "./data/ambulance.obj", "data/colormap.png"),
            .camera = Camera{
                .position = .{ 0, 1, 3, 1 },
                .target = .{ 0, 1, 0, 1 },
                .view = zmath.identity(),
            },
        };
    }

    fn render(self: *App) !void {
        //---------------------------------------------------------------------------------------
        zgui_sdl.newFrame(SCREEN_WIDTH, SCREEN_HEIGHT, 1);
        zgui.showDemoWindow(null);
        if (zgui.begin("Inspector", .{})) {
            zgui.text("GPU: {s}", .{driver_name});
            zgui.text("FPS: {d}", .{self.fps_manager.getFps()});
            zgui.text("Delta: {d}", .{self.fps_manager.getDelta()});
            zgui.text("event: {any}", .{event});
            zgui.text("scancode: {d}", .{scancode});
            if (zgui.comboFromEnum("Render Mode", &present_mode)) {
                try self.device.setSwapchainParameters(self.window, .sdr, present_mode);
            }

            // zgui.endCombo();
            if (zgui.button("Pause", .{})) {
                self.model.rotation_speed = 0;
            }
            _ = zgui.sliderAngle("Model Rotation Speed", .{ .vrad = &self.model.rotation_speed, .deg_max = 180, .deg_min = -180 });
            _ = zgui.colorEdit3("Clear Color", .{ .col = &clear_color });
        }
        zgui.end();
        //---------------------------------------------------------------------------------------

        const cmd_buffer = try self.device.aquireCommandBuffer();
        const swapchain = try cmd_buffer.waitAndAquireSwapchainTexture(self.window);

        zgui.render(); //--------------------Render Zgui----------------------

        if (swapchain.texture) |swapchain_tex| {
            //Begin render pass
            const color_target = gpu.ColorTargetInfo{
                .texture = swapchain_tex,
                .load = .clear,
                .clear_color = .{ .r = clear_color[0], .g = clear_color[1], .b = clear_color[2], .a = 1 },
            };

            const depth_target = gpu.DepthStencilTargetInfo{
                .texture = self.depth_texture,
                .load = .clear,
                .store = .do_not_care,
                .stencil_load = .do_not_care,
                .stencil_store = .do_not_care,
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

    // fn events(self: *App) bool {
    //     _ = self;
    //     var event: sdl3.c.SDL_Event = undefined;
    //     while (sdl3.c.SDL_PollEvent(&event)) {
    //         const guiConsumed = zgui_sdl.processEvent(&event);
    //         _ = guiConsumed;
    //
    //         const io = zgui.io;
    //         if (io.getWantCaptureMouse() or io.getWantCaptureKeyboard() or io.getWantTextInput()) {
    //             continue;
    //         }
    //         switch (event.type) {
    //             sdl3.c.SDL_EVENT_QUIT => {
    //                 return false;
    //             },
    //             sdl3.c.SDL_EVENT_TERMINATING => {
    //                 return false;
    //             },
    //             sdl3.c.SDL_EVENT_KEY_DOWN => {
    //                 const scancode = Scancode{ .value = @intCast(event.key.scancode) };
    //                 if (Scancode.space.matches(scancode)) {
    //                     Model.paused = !Model.paused;
    //                     std.debug.print("Mass space\n", .{});
    //                 }
    //                 if (Scancode.w.matches(scancode)) {
    //                     Camera.w_pressed = true;
    //                 }
    //                 if (Scancode.a.matches(scancode)) {
    //                     Camera.a_pressed = true;
    //                 }
    //                 if (Scancode.s.matches(scancode)) {
    //                     Camera.s_pressed = true;
    //                 }
    //                 if (Scancode.d.matches(scancode)) {
    //                     Camera.d_pressed = true;
    //                 }
    //             },
    //             sdl3.c.SDL_EVENT_KEY_UP => {
    //                 const scancode = Scancode{ .value = @intCast(event.key.scancode) };
    //                 if (Scancode.w.matches(scancode)) {
    //                     Camera.w_pressed = false;
    //                 }
    //                 if (Scancode.a.matches(scancode)) {
    //                     Camera.a_pressed = false;
    //                 }
    //                 if (Scancode.s.matches(scancode)) {
    //                     Camera.s_pressed = false;
    //                 }
    //                 if (Scancode.d.matches(scancode)) {
    //                     Camera.d_pressed = false;
    //                 }
    //             },
    //             else => {},
    //         }
    //     }
    //     return true;
    // }
    // fn events(self: *App, event_: sdl3.events.Event) sdl3.AppResult {
    //     _ = self;
    //     const event = &event_.toSdl();
    //     const guiConsumed = zgui_sdl.processEvent(event);
    //     _ = guiConsumed;
    //
    //     const io = zgui.io;
    //     if (io.getWantCaptureMouse() or io.getWantCaptureKeyboard() or io.getWantTextInput()) {
    //         // continue;
    //         return sdl3.AppResult.run;
    //     }
    //     switch (event.type) {
    //         sdl3.c.SDL_EVENT_QUIT => {
    //             std.debug.print("quit", .{});
    //             return sdl3.AppResult.success;
    //         },
    //         sdl3.c.SDL_EVENT_TERMINATING => {
    //             return sdl3.AppResult.success;
    //         },
    //         sdl3.c.SDL_EVENT_KEY_DOWN => {
    //             const scancode = Scancode{ .value = @intCast(event.key.scancode) };
    //             if (Scancode.space.matches(scancode)) {
    //                 Model.paused = !Model.paused;
    //                 std.debug.print("Mass space\n", .{});
    //             }
    //             if (Scancode.w.matches(scancode)) {
    //                 Camera.w_pressed = true;
    //             }
    //             if (Scancode.a.matches(scancode)) {
    //                 Camera.a_pressed = true;
    //             }
    //             if (Scancode.s.matches(scancode)) {
    //                 Camera.s_pressed = true;
    //             }
    //             if (Scancode.d.matches(scancode)) {
    //                 Camera.d_pressed = true;
    //             }
    //         },
    //         sdl3.c.SDL_EVENT_KEY_UP => {
    //             const scancode = Scancode{ .value = @intCast(event.key.scancode) };
    //             if (Scancode.w.matches(scancode)) {
    //                 Camera.w_pressed = false;
    //             }
    //             if (Scancode.a.matches(scancode)) {
    //                 Camera.a_pressed = false;
    //             }
    //             if (Scancode.s.matches(scancode)) {
    //                 Camera.s_pressed = false;
    //             }
    //             if (Scancode.d.matches(scancode)) {
    //                 Camera.d_pressed = false;
    //             }
    //         },
    //         else => {},
    //     }
    //     // }
    //     return sdl3.AppResult.run;
    // }

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
        sdl3.init.quit(self.init_flags);
        sdl3.init.shutdown();
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

    pub fn load(device: gpu.Device, obj_path: []const u8, texture_path: [:0]const u8) !Model {
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
    app_state: *?*App,
    args: [][*:0]u8,
) !sdl3.AppResult {
    _ = args;
    gpa = std.heap.GeneralPurposeAllocator(.{}){};

    sdl3.errors.error_callback = &sdlErr;
    sdl3.log.setAllPriorities(.info);
    sdl3.log.setLogOutputFunction(&sdlLog, null);

    log_app.logInfo("Starting application...");

    gpa = std.heap.GeneralPurposeAllocator(.{}){};
    allocator = gpa.allocator();

    // Prepare app state.
    const state = try allocator.create(App);
    errdefer allocator.destroy(state);

    state.* = try App.init();

    app_state.* = state;
    sdl3.log.log("Finished initializing ...\n");
    return .run;
}

fn eventHandler(
    app_state: *App,
    event: sdl3.events.Event,
) !sdl3.AppResult {
    const e = event.toSdl();
    App.event = e;
    _ = zgui_sdl.processEvent(&e);
    switch (event) {
        .terminating => return .success,
        .quit => return .success,
        // .unknown => {
        //     const e = event.toSdl();
        //     App.event = e;
        //     if (e.type == sdl3.c.SDL_EVENT_KEY_DOWN) {
        //         App.scancode = e.key.scancode;
        //         switch (App.scancode) {
        //             sdl3.c.SDL_SCANCODE_ESCAPE => return .success,
        //             sdl3.c.SDL_SCANCODE_SPACE => {
        //                 app_state.paused = !app_state.paused;
        //             },
        //             else => {},
        //         }
        //     }
        //     return .run;
        // },
        else => {
            if (e.type == sdl3.c.SDL_EVENT_KEY_DOWN) {
                App.scancode = e.key.scancode;
                switch (App.scancode) {
                    sdl3.c.SDL_SCANCODE_ESCAPE => return .success,
                    sdl3.c.SDL_SCANCODE_SPACE => {
                        app_state.paused = !app_state.paused;
                    },
                    else => {},
                }
            }
            return .run;
        },
    }
}

fn update(
    app_state: *App,
) !sdl3.AppResult {
    app_state.update(app_state.fps_manager.getDelta());
    try app_state.render();
    return .run;
}

fn quit(
    app_state: ?*App,
    result: sdl3.AppResult,
) void {
    _ = result;

    if (app_state) |app| {
        app.deinit();
        allocator.destroy(app);

        const leaked = gpa.deinit();
        assert(leaked == .ok);
    }
}
