from subclasses import lie, linear_combinations, eigs, plot
from utils import utils, to_gif
import os
import glob

util = utils.Util()


class RootsImage:
    def __init__(self, matrix_dim=2, num_of_coeffs=50,
            num_of_frames=1000) -> None:
        self.lie_algebra = lie.LieAlgebra(matrix_dim)
        self.linear_combos = linear_combinations.LinearCombinations(num_of_coeffs, 
                self.lie_algebra.adjoint_reps, self.lie_algebra.cartan_dimension)
        self.plots_directory = str(os.getcwd())+'/Plots'
        self.save_directory = str(os.getcwd())+'/gifs'

        if 2*self.lie_algebra.dimension <= 20:
            self.indices = util.get_indices(2*self.lie_algebra.dimension, 2)
        else:
            self.indices = self.linear_combos.indices
            pad = 2*self.lie_algebra.dimension - len(self.indices[0])
            for i in range(len(self.indices)):
                for i in range(pad):
                    self.indices[i].insert(0, 0)

        # Clear contents of Plots folder
        old_plots = glob.glob(os.getcwd()+'/Plots/*.png')
        for f in old_plots:
            os.remove(f)

        for i in range(min(num_of_frames, len(self.indices))):
            self.eigens = eigs.Eigs(self.linear_combos.linear_coms, 
                    self.linear_combos.num_of_linear_coms,self.indices[i+27], 
                    len(self.indices), i)
            print("{:<20} {:<0} {:<2.0%}".format("Making Images", ":",
                i/num_of_frames), end="\r")
            self.image = plot.Plot(self.eigens.eig_xy_color, i)


        if 'gifs' not in os.listdir(os.getcwd()):
            os.mkdir('gifs')

        print("Making gif file ...\n")
        to_gif.Gif(self.plots_directory, self.save_directory,
                matrix_dim)

        
