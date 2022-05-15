from utils import utils
import numpy as np

util = utils.Util()


class LinearCombinations:
    def __init__(self, num_of_coeffs, matrices, num_of_matrices):
        self.coeffs = None
        self.indices = None
        self.linear_coms = None
        self.num_of_linear_coms = None

        self.matrices = matrices
        self.num_of_coeffs = num_of_coeffs
        self.num_of_matrices = num_of_matrices
        self.matrix_dim = self.matrices.shape[1]

        self.get_coeffs()
        self.get_indices()
        self.set_linear_coms()

    def get_coeffs(self) -> None:
        """"
            Uncomment the coefficient generator needed.
        """
        self.coeffs = util.get_unit_circle_coeffs(self.num_of_coeffs)
        #self.coeffs = util.get_random_coeffs(self.num_of_coeffs)

    def get_indices(self) -> None:
        self.indices = util.get_indices(self.num_of_matrices, self.num_of_coeffs)

    def set_linear_coms(self) -> None:
        self.num_of_linear_coms = len(self.indices)
        zeros = [0 for i in range(self.matrix_dim)]
        zeros = [zeros.copy() for i in range(self.matrix_dim)]
        self.linear_coms = [zeros.copy() for i in range(self.num_of_linear_coms)]

        for i in range(self.num_of_linear_coms):
            for j in range(self.num_of_matrices):
                self.linear_coms[i] += self.matrices[j] * self.coeffs[self.indices[i][j]]
