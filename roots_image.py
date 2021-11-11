from subclasses import lie, linear_combinations, eigs, plot
from utils import utils

util = utils.Util()


class RootsImage:
    def __init__(self, matrix_dim=2, num_of_coeffs=50) -> None:
        self.lie_algebra = lie.LieAlgebra(matrix_dim)
        self.linear_combos = linear_combinations.LinearCombinations(num_of_coeffs, self.lie_algebra.adjoint_reps,
                                                                    self.lie_algebra.cartan_dimension)

        if 2*self.lie_algebra.dimension <= 20:
            self.indices = util.get_indices(2*self.lie_algebra.dimension, 2)
        else:
            self.indices = self.linear_combos.indices
            pad = 2*self.lie_algebra.dimension - len(self.indices[0])
            for i in range(len(self.indices)):
                for i in range(pad):
                    self.indices[i].insert(0, 0)

        for i in range(len(self.indices)):
            print('EIGEN', end='\r')
            self.eigens = eigs.Eigs(self.linear_combos.linear_coms, self.linear_combos.num_of_linear_coms,
                                    self.indices[i+15], len(self.indices), i)
            print('PLOT', end='\r')
            self.image = plot.Plot(self.eigens.eig_xy_color, i)

        