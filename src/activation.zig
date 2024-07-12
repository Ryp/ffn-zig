const std = @import("std");

fn sigmoid(comptime T: type, value: T) T {
    return 1 / (1 + std.math.exp(-value));
}

fn sigmoid_derivative_from_sigmoid(comptime T: type, sigmoid_value: T) T {
    return sigmoid_value * (1 - sigmoid_value);
}

pub fn sigmoid_vec(comptime T: type, comptime size: usize, vector: @Vector(size, T)) @Vector(size, T) {
    var result: @Vector(size, T) = undefined;

    for (0..size) |index| {
        result[index] = sigmoid(T, vector[index]);
    }

    return result;
}
