'''
Approach based on digit detection using the CNN
The digit recognition is based on assimtion that the sudoku grid was detected peffectly and one cell width is 1/9 of
the sudoku width. --- not stable for more diffiult images where sudoku is not perfectly planar


'''

# todo: RENAME!!!!!!

import cv2
import yaml
from GridFinder.ContourGridFinder import ContourGridFinder
from Classifier.UpgradedKerasClassifier import UpgradedKerasClassifier

def convert(image_path, config_path):

    sudoku_img = cv2.imread(image_path)
    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    grid_finder = ContourGridFinder(config['grid_finder'])
    digit_classificator = UpgradedKerasClassifier(config['digit_classifier'])

    cropped_sudoku_img = grid_finder.cut_sudoku_grid(sudoku_img, is_debug_mode=False)
    digital_sudoku = digit_classificator.classify_cells(cropped_sudoku_img, is_debugging_mode=False)

    print(digital_sudoku)

    return digital_sudoku


if __name__ == '__main__':
    image_path = r'sudoku_imgs/annotated_test_imgs/image1072.jpg'
    config_path = r'configs/config_05'

    convert(image_path, config_path)

