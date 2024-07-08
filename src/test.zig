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
