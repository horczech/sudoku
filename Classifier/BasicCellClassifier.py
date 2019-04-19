from Classifier.CellClassifier import CellClassifier
import cv2
import numpy as np
from constants import ERROR_VALUE, SUDOKU_VALUE_RANGE
from utilities.utils import timeit
from utilities.Sudoku import Sudoku
import pytesseract


class BasicDigitRecogniser(CellClassifier):

    def __init__(self, config):
        super().__init__(config)

        self.pytesseract_config = config['pytesseract_config']

    def classify_cells(self, cropped_sudoku_img, is_debugging_mode=False):
        binary_img = self.preprocess_image(cropped_sudoku_img, is_debugging_mode=is_debugging_mode)

        digit_bboxes = self.get_digit_bboxes(binary_img)
        grid_indexes = self.get_digit_grid_indexes(binary_img, digit_bboxes)
        sudoku = self.clasiffy_digits(binary_img, digit_bboxes, grid_indexes)

        if is_debugging_mode:
            from matplotlib import pyplot as plt
            from utilities.utils import draw_bboxes

            plt.figure('Digit classification')

            plt.subplot(2,3,1)
            plt.imshow(cropped_sudoku_img, cmap='gray')
            plt.title('Input image')

            plt.subplot(2,3,2)
            plt.imshow(binary_img, cmap='gray')
            plt.title('Binarized img')

            plt.subplot(2,3,3)
            bbox_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
            bbox_img = draw_bboxes(bbox_img, digit_bboxes,thickness=3)
            plt.imshow(bbox_img)
            plt.title('Found bboxes')

            cv2.imshow('Found bboxes', bbox_img)

            plt.show()

        return sudoku

    @timeit
    def clasiffy_digits(self, binary_img, digit_bboxes, grid_indexes):
        classified_digits = []
        binary_img = cv2.bitwise_not(binary_img)
        for digit_bbox in digit_bboxes:
            # crop
            x, y, w, h = digit_bbox
            top_left_pt = np.array([x, y]) - self.digit_padding
            bottom_right_pt = np.array([x + w, y + h]) + self.digit_padding
            digit_img = binary_img[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]]

            # classify
            classified_digit = self.classify_digit(digit_img)

            # handle wrong results
            try:
                classified_digit = int(classified_digit)
                if not SUDOKU_VALUE_RANGE[0] <= classified_digit <= SUDOKU_VALUE_RANGE[1]:
                    classified_digit = ERROR_VALUE

            except ValueError:
                classified_digit = ERROR_VALUE

            classified_digits.append(classified_digit)

        return Sudoku.from_digit_and_idx(classified_digits, grid_indexes)

    def classify_digit(self, digit_img):
        try:
            digit_img = cv2.resize(digit_img, (25,25))
            classified_digit = pytesseract.image_to_string(digit_img, lang='eng', config=self.pytesseract_config)
        except:
            classified_digit = -1
        return classified_digit


if __name__ == '__main__':
    from glob import glob
    from os import path
    import yaml
    from GridFinder.ContourGridFinder import ContourGridFinder

    img_format = r'.jpg'
    folder_path = r'sudoku_imgs/easy_dataset'
    # folder_path = r'sudoku_imgs/annotated_test_imgs'

    config_path = r'configs/config_03'

    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    img_path_list = glob(path.join(folder_path, '*' + img_format))


    grid_finder = ContourGridFinder(config['grid_finder'])
    cell_classifier = BasicDigitRecogniser(config['digit_classifier'])



    for img_path in img_path_list:
        print(img_path)

        cropped_sudoku_img = grid_finder.cut_sudoku_grid(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
        digital_sudoku = cell_classifier.classify_cells(cropped_sudoku_img, is_debugging_mode=True)

        print(digital_sudoku)




