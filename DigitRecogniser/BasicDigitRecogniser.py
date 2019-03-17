from DigitRecogniser.DigitRecogniser import DigitRecogniser
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pytesseract
from constants import SUDOKU_GRID_SIZE, ERROR_VALUE, EMPTY_CELL_VALUE, SUDOKU_VALUE_RANGE


class BasicDigitRecogniser(DigitRecogniser):
    def __init__(self, grid_img, config):
        super().__init__(grid_img, config)

    def get_digits(self):
        ## binarize image
        self._blur_img = cv2.GaussianBlur(self.image,
                                          ksize=self.config.blur_kernel,
                                          sigmaX=self.config.blur_sigma)
        self._thresholded_img = cv2.adaptiveThreshold(self._blur_img,
                                                      maxValue=255,
                                                      adaptiveMethod=self.config.thresh_adaptiveMethod,
                                                      thresholdType=cv2.THRESH_BINARY_INV,
                                                      blockSize=self.config.thresh_blockSize,
                                                      C=self.config.thresh_C)

        # filter out horizontal and vertical lines
        horizontal_lines_img = self.filter_lines(self._thresholded_img, is_horizontal=True)
        vertical_lines_img = self.filter_lines(self._thresholded_img, is_horizontal=False)
        self._grid_lines_img = cv2.bitwise_or(horizontal_lines_img, vertical_lines_img)
        self._filtered_grid_img = cv2.bitwise_and(self._thresholded_img, self._thresholded_img, mask=cv2.bitwise_not(self._grid_lines_img))

        # find digits and its possitions
        digit_bboxes = self.get_digit_bboxes(self._filtered_grid_img)
        digit_grid_possitions = self.get_digit_grid_possition(digit_bboxes)

        # classify digits
        classified_digits = self.clasiffy_digits(self._thresholded_img, digit_bboxes)

        # convert to 1x81 array
        sudoku_array = self.convert_to_array(classified_digits, digit_grid_possitions)

        return sudoku_array.astype(int)

    def convert_to_array(self, digits, digit_grid_possitions):
        if len(digits) != len(digit_grid_possitions):
            raise ValueError(f'Number of elements  of digits and possitions should match.')

        sudoku_grid = np.empty((SUDOKU_GRID_SIZE, SUDOKU_GRID_SIZE))
        sudoku_grid[:] = EMPTY_CELL_VALUE

        for digit, possition in zip(digits, digit_grid_possitions):
            sudoku_grid[possition[0], possition[1]] = digit

        return sudoku_grid.flatten()


    def clasiffy_digits(self, image, bboxes):

        classified_digits = []
        for bbox in bboxes:
            x, y, w, h = bbox

            top_left_pt = np.array([x, y]) - self.config.digit_padding
            bottom_right_pt = np.array([x + w, y + h]) + self.config.digit_padding

            self._digit_img = cv2.bitwise_not(image[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]])

            classified_digit = pytesseract.image_to_string(self._digit_img, lang='eng', config=self.config.pytesseract_config)

            try:
                classified_digit = int(classified_digit)
                if not SUDOKU_VALUE_RANGE[0] <= classified_digit <= SUDOKU_VALUE_RANGE[1]:
                    classified_digit = ERROR_VALUE

            except ValueError:
                classified_digit = ERROR_VALUE


            # cv2.imshow('digit', self._digit_img)
            # print(classified_digit)
            # cv2.waitKey()

            classified_digits.append(classified_digit)

        return classified_digits

    def get_digit_grid_possition(self, digit_bboxes):

        cell_size = self.image.shape[0]/9

        grid_positions = []
        for digit_bbox in digit_bboxes:
            x, y, w, h = digit_bbox

            # format [row, column]
            center = np.array([y + h/2, x + w/2])
            grid_positions.append(np.floor(center/cell_size).astype(int))

        return np.asarray(grid_positions)







    def get_digit_bboxes(self, binary_image):
        _, contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self._filtered_digit_bboxes = []
        self._unfiltered_digit_bboxes = []
        for contour in contours:
            boundary_box = cv2.boundingRect(contour)
            self._unfiltered_digit_bboxes.append(boundary_box)
            x, y, w, h = boundary_box

            aspect_ratio = w/h
            area = w*h
            if self.config.aspect_ratio_range[0] < aspect_ratio < self.config.aspect_ratio_range[1] and \
                    area > self.config.min_digit_area:
                self._filtered_digit_bboxes.append(boundary_box)

        return np.asarray(self._filtered_digit_bboxes)

    def filter_lines(self, binary_image, is_horizontal):
        filtered_img = np.copy(binary_image)

        if is_horizontal:
            structure = cv2.getStructuringElement(cv2.MORPH_RECT, self.config.horizontal_ksize)
        else:
            structure = cv2.getStructuringElement(cv2.MORPH_RECT, self.config.vertical_ksize)

        filtered_img = cv2.erode(filtered_img, structure)
        filtered_img = cv2.dilate(filtered_img, structure)

        return filtered_img

