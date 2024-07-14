const std = @import("std");
const assert = std.debug.assert;

const tracy = @import("tracy.zig");

const StorageOrder = enum(u1) {
    ColumnMajor,
    RowMajor,
};

pub fn Matrix(columns: usize, rows: usize, T: type) type {
    return struct {
        comptime scalar_type: type = T,
        comptime columns: usize = columns,
        comptime rows: usize = rows,
        storage_order: StorageOrder = .RowMajor,
        data: []T = undefined,

        pub fn allocate(self: *@This(), allocator: std.mem.Allocator) !void {
            self.data = try allocator.alloc(T, columns * rows);
        }

        fn index_flat_storage_order(storage_order: StorageOrder, col_index: usize, row_index: usize) usize {
            return switch (storage_order) {
                .RowMajor => row_index * columns + col_index,
                .ColumnMajor => col_index * rows + row_index,
            };
        }

        pub fn index_flat(self: @This(), col_index: usize, row_index: usize) usize {
            return index_flat_storage_order(self.storage_order, col_index, row_index);
        }

        pub fn get_row(self: @This(), row_index: usize) @Vector(columns, T) {
            assert(self.storage_order == .RowMajor);
            const row_offset = row_index * columns;
            return self.data[row_offset..][0..columns].*;
        }

        pub fn get_column(self: @This(), col_index: usize) @Vector(rows, T) {
            assert(self.storage_order == .ColumnMajor);
            const col_offset = col_index * rows;
            return self.data[col_offset..][0..rows].*;
        }

        pub fn at(self: @This(), col_index: usize, row_index: usize) T {
            const index = index_flat(self, col_index, row_index);
            return self.data[index];
        }

        pub fn mul_vec(self: @This(), vector: @Vector(columns, T)) @Vector(rows, T) {
            var result: @Vector(rows, T) = undefined;

            for (0..rows) |row_index| {
                result[row_index] = dot(T, columns, self.get_row(row_index), vector);
            }

            return result;
        }

        pub fn mul_mat(self: @This(), comptime n: usize, m2: Matrix(n, columns, T), mat_d: Matrix(n, rows, T)) void {
            const scope = tracy.trace(@src());
            defer scope.end();

            assert(self.storage_order == .RowMajor);
            assert(m2.storage_order == .ColumnMajor);

            for (0..mat_d.rows) |row_index| {
                const row_a = self.get_row(row_index);

                for (0..mat_d.columns) |col_index| {
                    const col_b = m2.get_column(col_index);

                    const index = mat_d.index_flat(col_index, row_index);

                    mat_d.data[index] = dot(T, columns, row_a, col_b);
                }
            }
        }

        // NOTE: returns a Matrix sharing the same storage, be careful!
        // It also switches the storage order in the process, to keep the operation very cheap.
        pub fn transpose_share(self: @This()) Matrix(rows, columns, T) {
            return Matrix(rows, columns, T){
                .storage_order = if (self.storage_order == .ColumnMajor) .RowMajor else .ColumnMajor,
                .data = self.data,
            };
        }

        // Transpose the internal storage order, without changing the Matrix order.
        pub fn transpose_storage_copy(self: *@This()) void {
            var temp_array: [columns * rows]T = undefined;
            const new_storage_order: StorageOrder = if (self.storage_order == .ColumnMajor) .RowMajor else .ColumnMajor;

            for (0..rows) |row| {
                for (0..columns) |column| {
                    const flat_index_transposed = index_flat_storage_order(new_storage_order, column, row);

                    // Read with inverse mapping
                    temp_array[flat_index_transposed] = self.at(column, row);
                }
            }

            std.mem.copyForwards(T, self.data, &temp_array);

            self.storage_order = new_storage_order;
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
    const scope = tracy.trace(@src());
    defer scope.end();

    return @reduce(.Add, v1 * v2);
}
