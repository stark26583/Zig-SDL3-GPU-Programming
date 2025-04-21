const std = @import("std");

const CommonTypes = @import("CommonTypes.zig");
const Vec3 = CommonTypes.Vec3;
const Color = CommonTypes.Color;
const Vertex = CommonTypes.Vertex;
const Vec2 = CommonTypes.Vec2;

const OBJ_Face_Index = struct {
    position_index: u32,
    uv_index: u32,
};

pub const OBJ_Data = struct {
    positions: []Vec3,
    uvs_tex_coords: []Vec2,
    faces: []OBJ_Face_Index,

    ///Remember to deinit the OBJ_Data
    pub fn deinit(self: OBJ_Data, allocator: std.mem.Allocator) void {
        allocator.free(self.positions);
        allocator.free(self.uvs_tex_coords);
        allocator.free(self.faces);
    }
};

///Remember to free the obj data after done do it in defer
pub fn parse(allocator: std.mem.Allocator, path: []const u8) !OBJ_Data {
    const obj_file = try std.fs.cwd().openFile(path, .{});
    defer obj_file.close();

    var obj_file_stream = obj_file.reader();

    var positions = std.ArrayList(Vec3).init(allocator);
    defer positions.deinit();
    var uvs_tex_coords = std.ArrayList(Vec2).init(allocator);
    defer uvs_tex_coords.deinit();
    var faces = std.ArrayList(OBJ_Face_Index).init(allocator);
    defer faces.deinit();

    while (try obj_file_stream.readUntilDelimiterOrEofAlloc(allocator, '\n', 1024)) |line| : (allocator.free(line)) {
        if (std.mem.startsWith(u8, line, "v ")) {
            var vertices_iter = std.mem.splitScalar(u8, line[2..], ' ');

            const x = try std.fmt.parseFloat(f32, vertices_iter.next().?);
            const y = try std.fmt.parseFloat(f32, vertices_iter.next().?);
            const z = try std.fmt.parseFloat(f32, vertices_iter.next().?);

            try positions.append(.{ x, y, z });
        } else if (std.mem.startsWith(u8, line, "vt ")) {
            var uvs_tex_coords_iter = std.mem.splitScalar(u8, line[3 .. line.len - 1], ' ');

            const u = try std.fmt.parseFloat(f32, uvs_tex_coords_iter.next().?);
            const v = try std.fmt.parseFloat(f32, uvs_tex_coords_iter.next().?);

            try uvs_tex_coords.append(.{ u, v });
        } else if (std.mem.startsWith(u8, line, "f ")) {
            var obj_face_index_triplet_iter = std.mem.splitScalar(u8, line[2 .. line.len - 1], ' ');
            while (obj_face_index_triplet_iter.next()) |obj_face_index_triplet| {
                var obj_face_index_triplet_value_iter = std.mem.splitScalar(u8, obj_face_index_triplet, '/');
                const position_index = try std.fmt.parseInt(u32, obj_face_index_triplet_value_iter.next().?, 10);
                const uv_index = try std.fmt.parseInt(u32, obj_face_index_triplet_value_iter.next().?, 10);

                try faces.append(.{ .position_index = position_index - 1, .uv_index = uv_index - 1 });
            }
        } else {
            continue;
        }
    }

    return OBJ_Data{
        .positions = try positions.toOwnedSlice(),
        .uvs_tex_coords = try uvs_tex_coords.toOwnedSlice(),
        .faces = try faces.toOwnedSlice(),
    };
}

// pub fn parse(allocator: std.mem.Allocator, path: []const u8) !void {
//     const obj_file = try std.fs.cwd().openFile(path, .{});
//     defer obj_file.close();
//
//     var line_number: u32 = 0;
//
//     var obj_file_stream = obj_file.reader();
//     while (try obj_file_stream.readUntilDelimiterOrEofAlloc(allocator, '\n', 1024)) |line| : ({
//         allocator.free(line);
//         line_number += 1;
//     }) {
//         std.debug.print("{d})", .{line_number + 1});
//         if (std.mem.startsWith(u8, line, "v ")) {
//             var vertices_iter = std.mem.splitScalar(u8, line[2..], ' ');
//
//             const x = try std.fmt.parseFloat(f32, vertices_iter.next().?);
//             const y = try std.fmt.parseFloat(f32, vertices_iter.next().?);
//             const z = try std.fmt.parseFloat(f32, vertices_iter.next().?);
//             std.debug.print("Vertex: {d} {d} {d}\n", .{ x, y, z });
//         } else if (std.mem.startsWith(u8, line, "vt ")) {
//             var uvs_tex_coords_iter = std.mem.splitScalar(u8, line[3 .. line.len - 1], ' ');
//             var u: f32 = 0;
//             var v: f32 = 99.1;
//             if (uvs_tex_coords_iter.next()) |u_unparsed| {
//                 u = try std.fmt.parseFloat(f32, u_unparsed);
//             }
//             if (uvs_tex_coords_iter.next()) |v_unparsed| {
//                 v = std.fmt.parseFloat(f32, v_unparsed) catch blk: {
//                     std.debug.print("Failed to parse v: {s}\n", .{v_unparsed});
//                     break :blk 99.11;
//                 };
//             }
//             std.debug.print("Texture Cordinates: {d} {d}\n", .{ u, v });
//         } else if (std.mem.startsWith(u8, line, "f ")) {
//             std.debug.print("{s}\n", .{line});
//         } else {
//             std.debug.print("-----------------------------------\n", .{});
//         }
//     }
// }
