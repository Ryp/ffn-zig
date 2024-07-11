const std = @import("std");

const matrix = @import("matrix.zig");
const Matrix = matrix.Matrix;

const testing = std.testing;
const expectEqual = testing.expectEqual;

test "basic" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var m = Matrix(u8, 2, 2){ .data = try allocator.alloc(u8, 4) };
    defer allocator.free(m.data);

    const idx1_0 = m.index_flat(1, 0);

    m.data[0] = 0;
    m.data[1] = 1;
    m.data[2] = 2;
    m.data[3] = 3;

    try testing.expectEqual(1, m.data[idx1_0]);
    try testing.expectEqual(.{ 2, 3 }, m.get_row(1));

    m.transpose_storage_copy();

    const idx1_0_t = m.index_flat(1, 0);

    try testing.expectEqual(.{ 1, 3 }, m.get_column(1));
    try testing.expectEqual(1, m.data[idx1_0_t]);

    try testing.expect(idx1_0 != idx1_0_t);
}

test "matmul" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();
    // const allocator = std.heap.page_allocator;

    const M = 8192 / 2;
    const N = 128;
    const k = 8192 / 2;

    const flops = 2 * M * N * k;
    const bytes = 2 * (M * k + N * k + N * M);
    const ai = @as(f32, @floatFromInt(flops)) / @as(f32, @floatFromInt(bytes));
    _ = ai;

    // std.debug.print("Arithmetic intensity = {}\n", .{ai});

    const m1 = Matrix(f32, k, M){ .data = try allocator.alloc(f32, k * M) };
    defer allocator.free(m1.data);

    var m2 = Matrix(f32, N, k){ .data = try allocator.alloc(f32, N * k) };
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

    // m1.print();

    // m2.print();

    const r = Matrix(f32, N, M){ .data = try allocator.alloc(f32, N * M) };
    defer allocator.free(r.data);

    for (r.data) |*elt| {
        elt.* = 0.0;
    }

    m2.transpose_storage_copy();

    m1.mul_mat(m2.columns, m2, r);

    // r.print();
}
