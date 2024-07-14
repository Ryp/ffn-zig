const std = @import("std");
const assert = std.debug.assert;

const activation = @import("activation.zig");
const matrix = @import("matrix.zig");
const Matrix = matrix.Matrix;

const digit_pixel_dim = 28;
const digit_count = 10;
const digit_pixel_count = digit_pixel_dim * digit_pixel_dim;
const dataset_items = 60_000; // 50k train + 10k test
const dataset_item_size_bytes = digit_pixel_count + 1;
const dataset_size_bytes = dataset_items * dataset_item_size_bytes;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();
    // const allocator = std.heap.page_allocator;

    // NOTE: I could use @embedFile here
    const dataset_filename = "mnist-dataset.bin";

    var file = if (std.fs.cwd().openFile(dataset_filename, .{})) |f| f else |err| {
        std.debug.print("error: couldn't open file: '{s}'\n", .{dataset_filename});
        return err;
    };
    defer file.close();

    const dataset_buffer: []u8 = try allocator.alloc(u8, dataset_size_bytes);
    defer allocator.free(dataset_buffer);

    const bytes_read = try file.read(dataset_buffer);

    std.debug.assert(bytes_read == dataset_size_bytes);

    const input_layer_dim = digit_pixel_count;
    const hidden_layer_dim = 16;
    const output_layer_dim = digit_count;

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

    const digit0 = get_dataset_digit_item(dataset_buffer, 0);

    const input: @Vector(input_layer_dim, f32) = convert_pixel_data_to_input_vector(digit0.pixel_col_major);
    const expected_activation: @Vector(output_layer_dim, f32) = convert_labeled_digit_to_output_vector(digit0.labeled_digit);

    // Forward pass
    const z_h1 = w_h1.mul_vec(input) + bias_h1;
    const a_h1 = activation.sigmoid_vec(w_h1.scalar_type, w_h1.rows, z_h1);

    const z_h2 = w_h2.mul_vec(a_h1) + bias_h2;
    const a_h2 = activation.sigmoid_vec(w_h2.scalar_type, w_h2.rows, z_h2);

    const z_o = w_o.mul_vec(a_h2) + bias_o;
    const a_o = activation.sigmoid_vec(w_o.scalar_type, w_o.rows, z_o);

    const a_diff = a_o - expected_activation;

    _ = a_diff;
}

const DigitItem = struct {
    labeled_digit: u8,
    pixel_col_major: [digit_pixel_count]u8,
};

fn convert_pixel_data_to_input_vector(pixels: [digit_pixel_count]u8) @Vector(digit_pixel_count, f32) {
    var result: @Vector(digit_pixel_count, f32) = undefined;

    for (0..digit_pixel_count) |index|
        result[index] = @floatFromInt(pixels[index]);

    return result;
}

fn convert_labeled_digit_to_output_vector(labeled_digit: u8) @Vector(digit_count, f32) {
    var result = std.mem.zeroes(@Vector(digit_count, f32));

    result[labeled_digit] = 1.0;

    return result;
}

fn get_dataset_digit_item(dataset: []u8, index: usize) DigitItem {
    const offset = index * dataset_item_size_bytes;
    const dataset_slice = dataset[offset..];

    return .{
        .labeled_digit = dataset_slice[0],
        .pixel_col_major = dataset_slice[1 .. 1 + digit_pixel_count].*,
    };
}

pub fn random_vec(comptime T: type, comptime size: usize, rng: *std.Random) @Vector(size, T) {
    var result: @Vector(size, T) = undefined;

    for (0..size) |index| {
        result[index] = rng.float(T);
    }

    return result;
}
