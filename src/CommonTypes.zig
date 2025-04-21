pub const Vec2 = @Vector(2, f32);
pub const Vec3 = @Vector(3, f32);
pub const Color = @Vector(4, f32);
pub const Vertex = struct {
    pos: Vec3,
    color: Color,
    uv: Vec2,
};
