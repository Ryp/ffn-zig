const std = @import("std");
const assert = std.debug.assert;

const matrix = @import("matrix.zig");
const Matrix = matrix.Matrix;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();
    // const allocator = std.heap.page_allocator;

    const input_layer_dim = 28 * 28;
    const hidden_layer_dim = 16;
    const output_layer_dim = 10;

    var w_h1 = Matrix(input_layer_dim, hidden_layer_dim, f32){};
    var w_h2 = Matrix(hidden_layer_dim, hidden_layer_dim, f32){};
    var w_o = Matrix(hidden_layer_dim, output_layer_dim, f32){};

    try w_h1.allocate(allocator);
    defer allocator.free(w_h1.data);

    try w_h2.allocate(allocator);
    defer allocator.free(w_h2.data);

    try w_o.allocate(allocator);
    defer allocator.free(w_o.data);

    // FIXME Set random weights
    const bias_h1: @Vector(hidden_layer_dim, f32) = undefined;
    const bias_h2: @Vector(hidden_layer_dim, f32) = undefined;
    const bias_o: @Vector(output_layer_dim, f32) = undefined;

    // Forward pass
    const input: @Vector(input_layer_dim, f32) = undefined;

    // FIXME Sigmoid/ReLU
    const a_h1 = w_h1.mul_vec(input) + bias_h1;
    const a_h2 = w_h2.mul_vec(a_h1) + bias_h2;
    const a_o = w_o.mul_vec(a_h2) + bias_o;

    _ = a_o;
}
