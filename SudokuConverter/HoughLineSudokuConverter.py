'''
This approach uses contours to find sudoku on image followed by hough transform to extract individual cells and digits
'''


import cv2
import yaml
from GridFinder.ContourGridFinder import ContourGridFinder
from Classifier.HoughLineClassifier import HoughLineClassifier

def convert(image_path, config_path):

    sudoku_img = cv2.imread(image_path)
    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    grid_finder = ContourGridFinder(config['grid_finder'])
    digit_classificator = HoughLineClassifier(config['digit_classifier'])

    cropped_sudoku_img, transforamtion_matrix = grid_finder.cut_sudoku_grid(sudoku_img, is_debug_mode=False)
    digital_sudoku = digit_classificator.classify_cells(cropped_sudoku_img, is_debugging_mode=False)

    return digital_sudoku


if __name__ == '__main__':
    image_path = r'sudoku_imgs/standard_imgs/image109.jpg'
    config_path = r'configs/config_06'

    sudoku = convert(image_path, config_path)

    print(sudoku)

    # cv2.imshow("Cropped Sudoku Result", digital_sudoku.draw_result(cropped_sudoku_img))
    # cv2.imshow("Cropped Sudoku Result", digital_sudoku.draw_full_result(sudoku_img, transforamtion_matrix))
    # cv2.waitKey()

