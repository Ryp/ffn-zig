const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();
    // const allocator = std.heap.page_allocator;

    const m1 = f32_4x4{ .data = try allocator.alloc(f32, 4 * 4) };
    defer allocator.free(m1.data);

    const m2 = f32_4x4{ .data = try allocator.alloc(f32, 4 * 4) };
    defer allocator.free(m2.data);

    // Identity matrix
    for (m1.data, m2.data) |*elt, *elt2| {
        elt.* = 0.0;
        elt2.* = 0.0;
    }

    m1.data[0] = 1.0;
    m1.data[5] = 1.0;
    m1.data[10] = 1.0;
    m1.data[15] = 1.0;

    m2.data[0] = 1.0;
    m2.data[5] = 2.0;
    m2.data[10] = 3.0;
    m2.data[15] = 4.0;

    for (m1.data, 0..) |elt, i| {
        std.debug.print("m[{}] = {}\n", .{ i, elt });
    }

    const v: [4]f32 = .{ 1, 2, 3, 4 };

    const rslt = mul(f32, m1, v);

    for (rslt, 0..) |elt, i| {
        std.debug.print("v[{}] = {}\n", .{ i, elt });
    }

    const r = Matrix(m1.subtype, m2.columns, m1.rows){ .data = try allocator.alloc(f32, 4 * 4) };
    defer allocator.free(r.data);

    for (r.data) |*elt| {
        elt.* = 0.0;
    }

    mul_mat(m1, m2, r);

    for (r.data, 0..) |elt, i| {
        std.debug.print("m[{}] = {}\n", .{ i, elt });
    }
}

const f32_4x4 = Matrix(f32, 4, 4);

fn Matrix(comptime T: type, columns: usize, rows: usize) type {
    return struct {
        data: []T,
        comptime columns: usize = columns,
        comptime rows: usize = rows,
        comptime subtype: type = T,
    };
}

fn dot(comptime T: type, v1: @Vector(4, T), v2: @Vector(4, T)) T {
    return @reduce(.Add, v1 * v2);
}

fn mul(comptime T: type, m: f32_4x4, v: [4]f32) [4]f32 {
    return .{
        dot(T, m.data[0..4].*, v),
        dot(T, m.data[4..8].*, v),
        dot(T, m.data[8..12].*, v),
        dot(T, m.data[12..16].*, v),
    };
}

fn mul_mat(m1: f32_4x4, m2t: f32_4x4, mat_d: f32_4x4) void {
    comptime {
        std.debug.assert(m1.subtype == m2t.subtype);
        std.debug.assert(m1.columns == m2t.rows);
        std.debug.assert(mat_d.rows == m1.rows);
        std.debug.assert(mat_d.columns == m2t.columns);
    }

    for (0..mat_d.rows) |index_row| {
        const offset_a = index_row * m1.columns;
        const row_a = m1.data[offset_a..][0..m1.columns];

        for (0..mat_d.columns) |index_col| {
            const offset_b = index_col * m2t.columns; // FIXME
            const col_b = m2t.data[offset_b..][0..m2t.columns]; // FIXME treated as col/major here

            mat_d.data[index_row + index_col * m2t.columns] = dot(mat_d.subtype, row_a.*, col_b.*);
        }
    }
}
