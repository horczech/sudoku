from CellClassifier.CellClassifier import CellClassifier
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pytesseract
from constants import SUDOKU_GRID_SIZE, ERROR_VALUE, EMPTY_CELL_VALUE, SUDOKU_VALUE_RANGE
from utilities.utils import timeit
from Sudoku import Sudoku


class BasicDigitRecogniser(CellClassifier):
    def __init__(self, config):
        self.blur_kernel = tuple(config['blur_kernel'])
        self.blur_sigma = config['blur_sigma']

        self.thresh_adaptiveMethod = getattr(cv2, config['thresh_adaptiveMethod'])
        self.thresh_blockSize = config['thresh_blockSize']
        self.thresh_C = config['thresh_C']

        # filtering of horizontal and vertical lines
        self.shorter_side_px = config['shorter_side_px']
        self.longer_side_factor = config['longer_side_factor']

        # filtering contours that does not contain digit
        self.aspect_ratio_range = config['aspect_ratio_range']
        self.min_digit_area = config['min_digit_area']

        # the space around the digit
        self.digit_padding = config['digit_padding']
        self.pytesseract_config = config['pytesseract_config']

    def preprocess_image(self, gray_img):
        self._blur_img = cv2.GaussianBlur(gray_img,
                                          ksize=self.blur_kernel,
                                          sigmaX=self.blur_sigma)
        self._thresholded_img = cv2.adaptiveThreshold(self._blur_img,
                                                      maxValue=255,
                                                      adaptiveMethod=self.thresh_adaptiveMethod,
                                                      thresholdType=cv2.THRESH_BINARY_INV,
                                                      blockSize=self.thresh_blockSize,
                                                      C=self.thresh_C)

        # filter out horizontal and vertical lines
        horizontal_lines_img = self.filter_lines(self._thresholded_img, is_horizontal=True)
        vertical_lines_img = self.filter_lines(self._thresholded_img, is_horizontal=False)
        self._grid_lines_img = cv2.bitwise_or(horizontal_lines_img, vertical_lines_img)
        self._gridless_img = cv2.bitwise_and(self._thresholded_img, self._thresholded_img, mask=cv2.bitwise_not(self._grid_lines_img))

        return self._gridless_img


    def get_digit_bboxes(self, cropped_binary_img):
        # find digits and its possitions
        _, contours, _ = cv2.findContours(cropped_binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self._filtered_digit_bboxes = []
        self._unfiltered_digit_bboxes = []
        for contour in contours:
            boundary_box = cv2.boundingRect(contour)
            self._unfiltered_digit_bboxes.append(boundary_box)
            x, y, w, h = boundary_box

            aspect_ratio = w/h
            area = w*h
            if self.aspect_ratio_range[0] < aspect_ratio < self.aspect_ratio_range[1] and area > self.min_digit_area:
                self._filtered_digit_bboxes.append(boundary_box)

        return np.asarray(self._filtered_digit_bboxes)

    def get_digit_grid_indexes(self, binary_img, digit_bboxes):
        cell_size = binary_img.shape[0]/9

        grid_positions = []
        for digit_bbox in digit_bboxes:
            x, y, w, h = digit_bbox

            # format [row, column]
            center = np.array([y + h/2, x + w/2])
            grid_positions.append(np.floor(center/cell_size).astype(int))

        return np.asarray(grid_positions)

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

    @timeit
    def classify_digit(self, digit_img):
        classified_digit = pytesseract.image_to_string(digit_img, lang='eng', config=self.pytesseract_config)
        return classified_digit

    def filter_lines(self, binary_image, is_horizontal):
        # prepare the kernels for the filter
        ksize = int(binary_image.shape[0]/self.longer_side_factor)
        horizontal_kernel = tuple([ksize, self.shorter_side_px])
        vertical_kernel = tuple([self.shorter_side_px, ksize])

        filtered_img = np.copy(binary_image)
        if is_horizontal:
            structure = cv2.getStructuringElement(cv2.MORPH_RECT, horizontal_kernel)
        else:
            structure = cv2.getStructuringElement(cv2.MORPH_RECT, vertical_kernel)

        filtered_img = cv2.erode(filtered_img, structure)
        filtered_img = cv2.dilate(filtered_img, structure)

        return filtered_img

