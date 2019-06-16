import cv2
import yaml

from GridFinder.ContourGridFinder import ContourGridFinder
from Classifier.HoughLineClassifier import HoughLineClassifier
from utilities.utils import timeit

@timeit
def sudoku_img_to_string(image, grid_finder, digit_classifier):

    cropped_sudoku_img, transforamtion_matrix = grid_finder.cut_sudoku_grid(image, is_debug_mode=False)

    if cropped_sudoku_img is None and transforamtion_matrix is None:
        return None

    digital_sudoku = digit_classifier.classify_cells(cropped_sudoku_img, is_debugging_mode=False)
    digital_sudoku.set_transformation_matrix(transforamtion_matrix)
    digital_sudoku.set_cropped_sudoku_grid_img(cropped_sudoku_img)

    return digital_sudoku


if __name__ == '__main__':
    image_path = r'sudoku_imgs/standard_imgs/4.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    config_path = r'configs/config_07'
    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    grid_finder = ContourGridFinder(config['grid_finder'])
    digit_classificator = HoughLineClassifier(config['digit_classifier'])
    digital_sudoku = sudoku_img_to_string(image, grid_finder, digit_classificator)

    print(digital_sudoku)

