const std = @import("std");
const assert = std.debug.assert;

const matrix = @import("matrix.zig");
const Matrix = matrix.Matrix;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();
    // const allocator = std.heap.page_allocator;

    const m1 = Matrix(f32, 4, 4){ .data = try allocator.alloc(f32, 4 * 4) };
    defer allocator.free(m1.data);

    var m2 = Matrix(f32, 8, 4){ .data = try allocator.alloc(f32, 4 * 8) };
    defer allocator.free(m2.data);

    // Identity matrix
    for (m1.data) |*elt| {
        elt.* = 0.0;
    }
    for (m2.data) |*elt| {
        elt.* = 1.0;
    }

    m1.data[0] = 1.0;
    m1.data[5] = 1.0;
    m1.data[10] = 1.0;
    m1.data[15] = 1.0;

    m1.data[2] = 2.0;
    m1.data[6] = 2.0;
    m1.data[11] = 2.0;
    m1.data[1] = 2.0;

    m2.data[0] = 1.0;
    m2.data[5] = 1.0;
    m2.data[10] = 1.0;
    m2.data[15] = 1.0;

    m1.print();

    m2.print();

    const v: [4]f32 = .{ 1, 2, 3, 4 };

    const rslt = m1.mul_vec(v);

    for (0..4) |i| {
        std.debug.print("v[{}] = {}\n", .{ i, rslt[i] });
    }

    const r = Matrix(m1.scalar_type, m2.columns, m1.rows){ .data = try allocator.alloc(f32, 4 * 8) };
    defer allocator.free(r.data);

    for (r.data) |*elt| {
        elt.* = 0.0;
    }

    m2.transpose_storage_copy();

    m1.mul_mat(m2.columns, m2, r);

    r.print();
}
