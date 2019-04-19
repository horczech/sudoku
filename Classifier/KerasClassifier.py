from Classifier.BasicCellClassifier import BasicDigitRecogniser
from utilities.utils import timeit
from constants import ERROR_VALUE, SUDOKU_VALUE_RANGE
from utilities.Sudoku import Sudoku


import cv2
from keras import models
import numpy as np



class KerasClassifier(BasicDigitRecogniser):

    def __init__(self, config):
        super().__init__(config)

        self.model = models.load_model(r'KerasDigitRecognition/models/model_01.h5')

    @timeit
    def clasiffy_digits(self, binary_img, digit_bboxes, grid_indexes):

        digit_list = np.empty((len(digit_bboxes), 28, 28, 1))
        for idx, digit_bbox in enumerate(digit_bboxes):
            # crop
            x, y, w, h = digit_bbox
            top_left_pt = np.array([x, y]) - self.digit_padding
            bottom_right_pt = np.array([x + w, y + h]) + self.digit_padding

            digit_img = binary_img[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]]
            cv2.imshow('digit_img', digit_img)
            cv2.waitKey()


            digit_img = cv2.resize(digit_img, (28, 28))
            digit_img = digit_img.astype('float32') / 255

            digit_list[idx, :, :, 0] = digit_img

        # classify
        classified_digits = self.classify_digit(digit_list)

        return Sudoku.from_digit_and_idx(classified_digits, grid_indexes)

    def classify_digit(self, digit_list):

        try:
            classified_digits = self.model.predict(digit_list)
            classified_digits = np.argmax(classified_digits, axis=1)

            # ToDo: The network should be learned without zeros
            classified_digits[classified_digits == 0] = ERROR_VALUE
        except:
            classified_digits = np.ones(len(digit_list), dtype=int)*ERROR_VALUE

        return classified_digits
