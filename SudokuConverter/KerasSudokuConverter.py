import cv2
import yaml
from GridFinder.ContourGridFinder import ContourGridFinder
from Classifier.KerasClassifier import KerasClassifier

def convert(image_path, config_path):

    sudoku_img = cv2.imread(image_path)
    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    grid_finder = ContourGridFinder(config['grid_finder'])
    digit_classificator = KerasClassifier(config['digit_classifier'])

    cropped_sudoku_img = grid_finder.cut_sudoku_grid(sudoku_img, is_debug_mode=False)
    digital_sudoku = digit_classificator.classify_cells(cropped_sudoku_img)

    print(digital_sudoku)

    return digital_sudoku


if __name__ == '__main__':
    image_path = r'sudoku_imgs/annotated_test_imgs/image1072.jpg'
    config_path = r'configs/config_03'

    convert(image_path, config_path)

