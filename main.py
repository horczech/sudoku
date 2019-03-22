import matplotlib
import cv2
import sys
import os
import argparse


from CellClassifier.BasicDigitRecogniser import BasicDigitRecogniser
from GridFinder.ContourBasedSudokuFinder import ContourBasedSudokuFinder
from ImagePreprocessor.BasicPreprocessor import BasicImgPreprocessor
from utilities.utils import load_image, show_sudoku_grid, timeit


def basic_pipeline(img_path, config_path):
    sys.path.append(os.path.dirname(os.path.expanduser(config_path)))

    from config import GridFinder as grid_finder_config
    from config import preprocessing as preprocessing_config
    from config import DigitRecogniser as digit_recogniser_config


    sudoku_image = load_image(img_path, cv2.IMREAD_GRAYSCALE)

    preprocessor = BasicImgPreprocessor(sudoku_image, preprocessing_config)
    binary_image = preprocessor.do_preprocessing()


    grid_finder = ContourBasedSudokuFinder(sudoku_image, binary_image, grid_finder_config)
    transformed_grid = grid_finder.find_grid()

    digit_recogniser = BasicDigitRecogniser(transformed_grid, digit_recogniser_config)
    sudoku_array = digit_recogniser.get_digits()

    return sudoku_array

def run(img_path, config_path):
    print(f'OpenCV version\n{cv2.__version__}')
    print(f'Python version\n{sys.version}')
    print('\n----------------------------------\n')

    sudoku_array = basic_pipeline(img_path, config_path)
    show_sudoku_grid(sudoku_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Digitize SUDOKU image')

    parser.add_argument('img', type=str, help='Path to image')
    parser.add_argument('-c','--config', type=str, help='Path to config file')

    args = parser.parse_args()

    if args.config is None:
        args.config = r'configs/config_1/config.py'

    print(args.config)
    run(args.img, args.config)
