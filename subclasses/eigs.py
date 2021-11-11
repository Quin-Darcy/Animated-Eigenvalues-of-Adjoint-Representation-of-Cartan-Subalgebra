from utils import utils
from numpy import linalg as la
import math

util = utils.Util()


class Eigs:
    def __init__(self, matrices, num_of_matrices, fixed_vector, num_of_frames, frame) -> None:
        self.eigs = []
        self.eig_xy_color = []
        self.matrices = matrices
        self.num_of_matrices = num_of_matrices
        self.fixed_vector = fixed_vector
        self.num_of_frames = num_of_frames
        self.frame = frame

        self.get_eigs()

    def get_eigs(self) -> None:
        """
            This function gets the eigenvalue/eigenvector pairs from each matrix. The eigenvalue is converted
            into an ordered pair to be plotted. The complex eigenvector with, say n components, is converted
            into a real valued vector with 2n components, utilizing the vector space isomorphism between C^n
            and R^2n.

            With the real eigenvector in hand, we take its angle from a fixed vector, and to this angle we can
            associate an RGB color. Thus, eig_xy_color contains a list whose elements are themselves lists
            containing an xy-pair and an RGB triple.
        """
        for matrix in self.matrices:
            w, v = la.eig(matrix)
            temp_vector = [f(v) for v in w for f in (self.get_real, self.get_imag)]
            theta = util.angle(temp_vector, self.fixed_vector)
            theta = theta ** (2*math.sqrt(len(w))) % (2*math.pi)
            color = util.get_color(theta)
            eigval_set = [[[temp_vector[i], temp_vector[i+1]], color] for i in range(0, len(temp_vector)-1, 2)]
            self.eig_xy_color += eigval_set


    @staticmethod
    def get_real(v):
        return v.real

    @staticmethod
    def get_imag(v):
        return v.imag
