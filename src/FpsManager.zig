const std = @import("std");
const sdl3 = @import("sdl3");
const timer = sdl3.timer;
const Self = @This();

/// Modes for FPS management:
pub const FPSMode = union(enum) {
    /// Auto (default) caps at 60 FPS
    auto,
    /// No frame limiting; just measure FPS
    none,
    /// Manual target FPS
    manual: u32,
};

/// Unified FPS limiter + monitor for SDL3 game loops.
/// Use `tick()` each frame, then `getFps()` or `getDelta()` to retrieve metrics.
// Frame cap (ticks per frame)
ticks_per_frame: u64 = 0,
// Timestamps
last_tick_count: u64 = 0,
last_fps_count: u64 = 0,
// Metrics
last_delta_ticks: u64 = 0,
frame_counter: u64 = 0,
current_fps: f64 = 0.0,

/// Initialize with desired FPS mode. Call once after SDL_Init.
pub fn init(mode: FPSMode) Self {
    var self: Self = undefined;
    const freq = timer.getPerformanceFrequency();
    const fps = switch (mode) {
        .auto => 60,
        .none => 0,
        .manual => |v| v,
    };

    if (fps > 0) {
        // Compute ticks per frame (integer division)
        self.ticks_per_frame = @intFromFloat(@as(f64, @floatFromInt(freq)) / @as(f64, @floatFromInt(fps)));
    } else {
        // Unlimited mode
        self.ticks_per_frame = 0;
    }

    // Initialize timing state
    self.last_tick_count = timer.getPerformanceCounter();
    self.last_fps_count = self.last_tick_count;
    self.last_delta_ticks = 0;
    self.frame_counter = 0;
    self.current_fps = 0.0;
    return self;
}

/// Call at the end of each frame: applies cap, updates FPS, and records delta.
pub fn tick(self: *Self) void {
    const now = timer.getPerformanceCounter();
    const elapsed = now - self.last_tick_count;

    // Frame limiting
    if (self.ticks_per_frame > 0 and elapsed < self.ticks_per_frame) {
        const rem_ticks = self.ticks_per_frame - elapsed;
        // Convert ticks to nanoseconds: (rem_ticks / freq) * 1e9
        const wait_ns = @as(f64, @floatFromInt(rem_ticks)) * 1_000_000_000.0 / @as(f64, @floatFromInt(timer.getPerformanceFrequency()));
        timer.delayNanosecondsPrecise(@as(u64, @intFromFloat(wait_ns)));
    }

    // Record delta for this frame
    const then = timer.getPerformanceCounter();
    self.last_delta_ticks = then - self.last_tick_count;
    self.last_tick_count = then;

    // FPS measurement
    self.frame_counter += 1;
    const fps_dt = then - self.last_fps_count;
    if (fps_dt >= timer.getPerformanceFrequency()) {
        const fc_f = @as(f64, @floatFromInt(self.frame_counter));
        const freq_f = @as(f64, @floatFromInt(timer.getPerformanceFrequency()));
        const dt_f = @as(f64, @floatFromInt(fps_dt));
        self.current_fps = fc_f * freq_f / dt_f;
        self.last_fps_count = then;
        self.frame_counter = 0;
    }
}

/// Returns the most recently computed FPS (updates ~once/sec).
pub fn getFps(self: Self) f64 {
    return self.current_fps;
}

/// Returns the delta time (in seconds) of the last frame as f32.
pub fn getDelta(self: Self) f32 {
    // delta_ticks / frequency gives seconds
    return @as(f32, @floatFromInt(self.last_delta_ticks)) / @as(f32, @floatFromInt(timer.getPerformanceFrequency()));
}

/// Change FPS mode on the fly (re-initializes cap).
pub fn setMode(self: *Self, mode: FPSMode) void {
    const new = Self.init(mode);
    @memcpy(self, new);
}

/// Adjust the target FPS on the fly.
pub fn setFps(self: *Self, fps: u32) void {
    const new = Self.init(.{ .manual = fps });
    @memcpy(self, new);
}
