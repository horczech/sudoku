from Classifier.CellClassifier import CellClassifier
from utilities.utils import timeit
from constants import ERROR_VALUE, SUDOKU_VALUE_RANGE
from utilities.Sudoku import Sudoku


import cv2
from keras import models
import numpy as np


class KerasClassifier(CellClassifier):

    def __init__(self, config):
        super().__init__(config)

        # ToDo: Put somewhere else
        self.model = models.load_model(r'KerasDigitRecognition/models/model_with_ocr_data.h5')

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

        digit_list = np.empty((len(digit_bboxes), 28, 28, 1))
        for idx, digit_bbox in enumerate(digit_bboxes):
            # crop
            x, y, w, h = digit_bbox
            top_left_pt = np.array([x, y]) - self.digit_padding
            bottom_right_pt = np.array([x + w, y + h]) + self.digit_padding

            digit_img = binary_img[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]]
            digit_img = cv2.resize(digit_img, (28, 28))
            digit_img = digit_img.astype('float32') / 255

            digit_list[idx, :, :, 0] = digit_img

        # classify
        classified_digits = self.classify_digit(digit_list)

        return Sudoku.from_digit_and_idx(classified_digits, grid_indexes)

    def classify_digit(self, digit_list):

        try:
            classified_digits = self.model.predict(digit_list)
            # we have to add 1 because we have numbers 1-9 so the class on zero position is 1
            classified_digits = np.argmax(classified_digits, axis=1) + 1

            # ToDo: The network should be learned without zeros
            classified_digits[classified_digits == 0] = ERROR_VALUE
        except:
            classified_digits = np.ones(len(digit_list), dtype=int)*ERROR_VALUE

        return classified_digits
