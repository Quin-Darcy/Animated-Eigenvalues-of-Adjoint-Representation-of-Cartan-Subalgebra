import numpy as np
import random
import math


class Util:
    @staticmethod
    def get_dimension(n) -> int:
        return (n-1) * (n+1)

    @staticmethod
    def lie_bracket(X, Y) -> np.array:
        return np.matmul(X, Y) - np.matmul(Y, X)

    @staticmethod
    def decompose_lc(lin_com, dim, matrix_dim) -> np.array:
        coeff_index = 0
        coeffs = np.zeros((1, dim), dtype=np.complex_)
        for i in range(matrix_dim):
            for j in range(matrix_dim):
                if i != j:
                    coeffs[0][coeff_index] = lin_com[i][j]
                    coeff_index += 1
        
        last_entry = complex(0)
        for i in range(matrix_dim-1):
            coeffs[0][coeff_index] = lin_com[i][i] - last_entry
            last_entry = lin_com[i+1][i+1]
            coeff_index += 1

        return coeffs

    def get_adjoint_rep(self, mat, basis, dim, matrix_dim) -> np.array:
        adjoint_rep = np.zeros((dim, dim), dtype=np.complex_)
        for i in range(dim):
            bracket = self.lie_bracket(mat, basis[i])
            adjoint_rep[i] = self.decompose_lc(bracket, dim, matrix_dim)

        return adjoint_rep

    @staticmethod
    def get_index(sequence_len, num_of_chars) -> list:
        """
            This function returns all length [sequence_len] sequences whose terms can be
            one of [num_of_chars] many values.

            Ex. If sequence_len = 3 and num_of_chars = 2, then this function would return
            all length 8 binary sequences (000, 001, 010, 011, 100, 101, 110, 111)
        """
        num_of_seqs = num_of_chars ** sequence_len
        zeros = [0 for i in range(sequence_len)]
        indices = [zeros.copy() for i in range(num_of_seqs)]
        temp = zeros.copy()
        powers_of_two = [num_of_chars ** i for i in range(sequence_len)]
        for i in range(1, num_of_seqs):
            for j in range(sequence_len):
                if i % (powers_of_two[j]) == 0:
                    temp[j] = (temp[j] + 1) % num_of_chars
                indices[i][sequence_len - j - 1] = temp[j]

        return indices

    def get_indices(self, sequence_len, num_of_chars):
        import itertools
        if sequence_len % 2 == 0:
            half_len = sequence_len // 2
            indices = self.get_index(half_len, num_of_chars)
            indices = itertools.product(indices, indices)
            indices = [a[0] + a[1] for a in indices]
        else:
            half_len1 = sequence_len - sequence_len // 2
            half_len2 = sequence_len // 2
            index1 = self.get_index(half_len1, num_of_chars)
            index2 = self.get_index(half_len2, num_of_chars)
            indices = itertools.product(index1, index2)
            indices = [a[0] + a[1] for a in indices]
        return indices

    @staticmethod
    def get_unit_circle_coeffs(num_of_coeffs) -> list:
        theta = 2 * math.pi / num_of_coeffs
        return [complex(math.cos(i*theta), math.sin(i*theta)) for i in range(num_of_coeffs)]

    @staticmethod
    def get_random_coeffs(num_coeffs) -> list:
        return [complex(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)) for i in range(num_coeffs)]

    @staticmethod
    def linear_transformation(x, y, z):
        linear_t = [[math.sqrt(3) / 2, 0, 1 / 2], [-math.sqrt(2) / 4, math.sqrt(2) / 2, math.sqrt(6) / 4],
               [-math.sqrt(2) / 4, -math.sqrt(2) / 2, math.sqrt(6) / 4]]

        vec = [x, y, z]
        new_vec = [0, 0, 0]

        for i in range(3):
            for j in range(3):
                new_vec[i] += int(vec[j] * linear_t[i][j])
        return new_vec

    @staticmethod
    def dotproduct(v1, v2) -> float:
        return sum((a * b) for a, b in zip(v1, v2))

    def length(self, v) -> float:
        return math.sqrt(self.dotproduct(v, v))

    def angle(self, v1, v2) -> float:
        return math.acos(self.dotproduct(v1, v2).real / (self.length(v1) * self.length(v2) + 0.0000000000001))

    def get_color(self, t) -> tuple:
        arr = [0, 0, 0]
        k = math.pi / 3
        a = math.sqrt(3)
        theta = (math.pi / 2) - np.arccos(a / 3)
        r = 127.5 * math.sqrt(3)
        c = r * a
        if 0 <= t < k:
            x = c / (math.tan(t) + a)
            y = math.tan(t) * x
            z = (-(1 / r) * 2 * x + 2) * 255 * (math.sqrt(2) * math.sin(theta) - 1 / 2) + 127.5
            arr = self.linear_transformation(x, y, z)
        if k <= t < 2 * k:
            x = c / (2 * math.tan(t))
            y = math.tan(t) * x
            z = -(1 / r) * (x - r / 2) * 255 * (1 / 2 - math.sqrt(2) * math.sin(theta)) + 255 * math.sqrt(2) * math.sin(
                theta)
            arr = self.linear_transformation(x, y, z)
        if 2 * k <= t < 3 * k:
            x = c / (math.tan(t) - a)
            y = math.tan(t) * x
            z = -(1 / r) * (2 * x + r) * 255 * (math.sqrt(2) * math.sin(theta) - 1 / 2) + 127.5
            arr = self.linear_transformation(x, y, z)
        if 3 * k <= t < 4 * k:
            x = -c / (math.tan(t) + a)
            y = math.tan(t) * x
            z = (1 / r) * (2 * x + 2 * r) * 255 * (1 / 2 - math.sqrt(2) * math.sin(theta)) + 255 * math.sqrt(
                2) * math.sin(theta)
            arr = self.linear_transformation(x, y, z)
        if 4 * k <= t < 5 * k:
            x = -c / (2 * math.tan(t))
            y = math.tan(t) * x
            z = (1 / r) * (x + r / 2) * 255 * (math.sqrt(2) * math.sin(theta) - 1 / 2) + 127.5
            arr = self.linear_transformation(x, y, z)
        if 5 * k <= t < 6 * k:
            x = -c / (math.tan(t) - a)
            y = math.tan(t) * x
            z = (1 / r) * (2 * x - r) * 255 * (1 / 2 - math.sqrt(2) * math.sin(theta)) + 255 * math.sqrt(2) * math.sin(
                theta)
            arr = self.linear_transformation(x, y, z)

        return tuple(arr)

