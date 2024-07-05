const std = @import("std");
const assert = std.debug.assert;

const tracy = @import("tracy.zig");

const Storage = enum(u1) {
    ColumnMajor,
    RowMajor,
};

pub fn Matrix(comptime T: type, columns: usize, rows: usize) type {
    return struct {
        comptime scalar_type: type = T,
        comptime columns: usize = columns,
        comptime rows: usize = rows,
        storage: Storage = .RowMajor,
        data: []T,

        fn index_flat_storage(storage: Storage, col_index: usize, row_index: usize) usize {
            return switch (storage) {
                .RowMajor => row_index * columns + col_index,
                .ColumnMajor => col_index * rows + row_index,
            };
        }

        fn index_flat(self: @This(), col_index: usize, row_index: usize) usize {
            return index_flat_storage(self.storage, col_index, row_index);
        }

        fn get_row(self: @This(), row_index: usize) @Vector(columns, T) {
            assert(self.storage == .RowMajor);
            const offset = row_index * columns;
            return self.data[offset..][0..columns].*;
        }

        fn get_column(self: @This(), col_index: usize) @Vector(rows, T) {
            assert(self.storage == .ColumnMajor);
            const offset = col_index * rows;
            return self.data[offset..][0..rows].*;
        }

        fn at(self: @This(), col_index: usize, row_index: usize) T {
            const index = index_flat(self, col_index, row_index);
            return self.data[index];
        }

        pub fn mul_vec(self: @This(), vector: @Vector(rows, T)) @Vector(rows, T) {
            var result: @Vector(rows, T) = undefined;

            for (0..rows) |row_index| {
                result[row_index] = dot(T, rows, self.get_row(row_index), vector);
            }

            return vector;
        }

        pub fn mul_mat(self: @This(), comptime n: usize, m2: Matrix(T, n, columns), mat_d: Matrix(T, n, rows)) void {
            const scope = tracy.trace(@src());
            defer scope.end();

            assert(self.storage == .RowMajor);
            assert(m2.storage == .ColumnMajor);

            for (0..mat_d.rows) |row_index| {
                const row_a = self.get_row(row_index);

                for (0..mat_d.columns) |col_index| {
                    const col_b = m2.get_column(col_index);

                    const index = mat_d.index_flat(col_index, row_index);

                    mat_d.data[index] = dot(T, columns, row_a, col_b);
                }
            }
        }

        pub fn transpose_storage_copy(self: *@This()) void {
            var temp_array: [columns * rows]T = undefined;
            const new_storage: Storage = if (self.storage == .ColumnMajor) .RowMajor else .ColumnMajor;

            for (0..rows) |row| {
                for (0..columns) |column| {
                    const flat_index_transposed = index_flat_storage(new_storage, column, row);

                    // Read with inverse mapping
                    temp_array[flat_index_transposed] = self.at(column, row);
                }
            }

            std.mem.copyForwards(T, self.data, &temp_array);

            self.storage = new_storage;
        }

        pub fn print(self: @This()) void {
            std.debug.print("mat {}x{} = [\n", .{ columns, rows });

            for (0..rows) |row_index| {
                std.debug.print("  [ ", .{});
                for (0..columns) |col_index| {
                    std.debug.print("{:<4} ", .{self.at(col_index, row_index)});
                }
                std.debug.print("]\n", .{});
            }

            std.debug.print("]\n", .{});
        }
    };
}

pub fn dot(comptime T: type, comptime size: usize, v1: @Vector(size, T), v2: @Vector(size, T)) T {
    return @reduce(.Add, v1 * v2);
}
