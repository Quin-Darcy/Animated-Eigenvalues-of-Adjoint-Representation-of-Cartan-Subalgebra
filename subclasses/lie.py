import numpy as np
from utils import utils as util

util = util.Util()


class LieAlgebra:
    """
    This is a class which holds information about a user specified Lie algebra. 
    """
    def __init__(self, n=2) -> None:
        self.matrix_dim = n
        self.basis = None
        self.cartan_basis = None
        self.adjoint_reps = None
        self.dimension = util.get_dimension(n)
        self.cartan_dimension = n-1
        
        self.set_algebra()

    def set_algebra(self) -> None:
        self.set_basis()
        self.set_adjoint_reps()

    def set_basis(self) -> None:
        self.basis = np.zeros((self.dimension, self.matrix_dim, self.matrix_dim), dtype=np.complex_)
        self.cartan_basis = np.zeros((self.matrix_dim-1, self.matrix_dim, self.matrix_dim), dtype=np.complex_)

        basis_index = 0
        cartan_basis_index = 0
        for i in range(self.matrix_dim):
            for j in range(self.matrix_dim):
                if i != j:
                    self.basis[basis_index][i][j] = complex(1, 0)
                    basis_index += 1

        for i in range(self.matrix_dim-1):
            self.basis[basis_index][i][i] = complex(1, 0)
            self.basis[basis_index][i+1][i+1] = complex(-1, 0)
            self.cartan_basis[cartan_basis_index][i][i] = complex(1, 0)
            self.cartan_basis[cartan_basis_index][i+1][i+1] = complex(-1, 0)
            basis_index += 1
            cartan_basis_index += 1

    def set_adjoint_reps(self) -> None:
        self.adjoint_reps = np.zeros((self.cartan_dimension, self.dimension, self.dimension), dtype=np.complex_)

        for i in range(self.cartan_dimension):
            self.adjoint_reps[i] = util.get_adjoint_rep(self.cartan_basis[i], self.basis, self.dimension, self.matrix_dim)

    def show(self) -> None:
        util.show_matrices(self.adjoint_reps, self.cartan_dimension, self.dimension, self.dimension)

