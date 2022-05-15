import roots_image
import math
from utils import utils

util = utils.Util()


def main():
    input_type = int(input("Input type: (dimension, num_of_coeffs) or (dimension, frame_size, num_of_frames)? (1/2)"))

    if input_type == 1:
        n = int(input('Enter n: '))
        coeffs = int(input('Enter number of coefficients: '))
        util.display_stats(n, coeffs)
        proc = input('\nDo you wish to proceed? (y/n)')
        if proc == 'y':
            print('\n')
            roots_image.RootsImage(n, coeffs)
        else:
            print('Seeya!')
    else:
        n = int(input("Enter n: "))
        frame_size = int(input("Enter side length: "))
        num_of_frames = int(input("Enter number of frames: "))
        coeffs = util.calculate_coeffs(n, frame_size)
        util.display_stats(n, coeffs, frame_size, num_of_frames)
        proc = input('\nDo you wish to proceed? (y/n)')
        if proc == 'y':
            print('\n')
            roots_image.RootsImage(n, coeffs, num_of_frames)
        else:
            print('Seeya!')


if __name__ == '__main__':
    main()
