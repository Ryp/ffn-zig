const std = @import("std");
const assert = std.debug.assert;

const activation = @import("activation.zig");
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

    var prng = std.Random.DefaultPrng.init(42); // FIXME
    var rng = prng.random();

    for (w_h1.data) |*element| {
        element.* = rng.float(f32);
    }
    for (w_h2.data) |*element| {
        element.* = rng.float(f32);
    }
    for (w_o.data) |*element| {
        element.* = rng.float(f32);
    }

    const bias_h1 = random_vec(f32, hidden_layer_dim, &rng);
    const bias_h2 = random_vec(f32, hidden_layer_dim, &rng);
    const bias_o = random_vec(f32, output_layer_dim, &rng);

    // Forward pass
    const input: @Vector(input_layer_dim, f32) = undefined;

    const z_h1 = w_h1.mul_vec(input) + bias_h1;
    const a_h1 = activation.sigmoid_vec(w_h1.scalar_type, w_h1.rows, z_h1);

    const z_h2 = w_h2.mul_vec(a_h1) + bias_h2;
    const a_h2 = activation.sigmoid_vec(w_h2.scalar_type, w_h2.rows, z_h2);

    const z_o = w_o.mul_vec(a_h2) + bias_o;
    const a_o = activation.sigmoid_vec(w_o.scalar_type, w_o.rows, z_o);

    _ = a_o;
}

pub fn random_vec(comptime T: type, comptime size: usize, rng: *std.Random) @Vector(size, T) {
    var result: @Vector(size, T) = undefined;

    for (0..size) |index| {
        result[index] = rng.float(T);
    }

    return result;
}
