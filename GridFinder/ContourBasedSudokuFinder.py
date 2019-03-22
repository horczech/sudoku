from GridFinder.SudokuFinder import SudokuFinder
import cv2
import numpy as np
from utilities.utils import timeit
from GridFinder.SudokuCorners import SudokuCorners


class ContourBasedSudokuFinder(SudokuFinder):
    def __init__(self, config):

        # config params
        self.blur_kernel = tuple(config['blur_kernel'])
        self.blur_sigma = config['blur_sigma']

        self.thresh_adaptiveMethod = getattr(cv2, config['thresh_adaptiveMethod'])
        self.thresh_blockSize = config['thresh_blockSize']
        self.thresh_C = config['thresh_C']

        self.opening_kernel = tuple(config['opening_kernel'])
        self.closing_kernel = tuple(config['closing_kernel'])

        self.epsilon = config['epsilon']
        self.output_grid_size = config['output_grid_size']

    def preprocess_image(self, gray_image):
        self._blur_img = cv2.GaussianBlur(gray_image,
                                          ksize=self.blur_kernel,
                                          sigmaX=self.blur_sigma)
        self._thresholded_img = cv2.adaptiveThreshold(self._blur_img,
                                                      maxValue=255,
                                                      adaptiveMethod=self.thresh_adaptiveMethod,
                                                      thresholdType=cv2.THRESH_BINARY_INV,
                                                      blockSize=self.thresh_blockSize,
                                                      C=self.thresh_C)

        opening_kernel = np.ones(self.opening_kernel, np.uint8)
        closing_kernel = np.ones(self.closing_kernel, np.uint8)
        self._opening = cv2.morphologyEx(self._thresholded_img, cv2.MORPH_OPEN, opening_kernel)
        self._opening_closing = cv2.morphologyEx(self._opening, cv2.MORPH_CLOSE, closing_kernel)

        return self._opening_closing

    def find_sudoku_corners(self, binary_img):
        _, self._contours, _ = cv2.findContours(binary_img,
                                                mode=cv2.RETR_EXTERNAL,
                                                method=cv2.CHAIN_APPROX_SIMPLE)

        contour_area = np.asarray([cv2.contourArea(contour) for contour in self._contours])
        self._largest_contour = self._contours[np.argmax(contour_area)]

        sudoku_corners = self.get_corners(self._largest_contour)

        return sudoku_corners

    def crop_sudoku(self, gray_img, sudoku_corners: SudokuCorners):

        # do the perspective transformation
        destination_pts = np.asarray([[0, 0],
                                      [self.output_grid_size, 0],
                                      [self.output_grid_size, self.output_grid_size],
                                      [0, self.output_grid_size]],
                                     dtype=np.float32)

        self.transformation_matrix = cv2.getPerspectiveTransform(sudoku_corners.get_array(dtype=np.float32), destination_pts)
        self._cropped_sudoku_img = cv2.warpPerspective(gray_img, self.transformation_matrix, (self.output_grid_size, self.output_grid_size))

        return self._cropped_sudoku_img

    def get_corners(self, contour) -> SudokuCorners:

        #  epsilon is maximum distance from contour to approximated contour
        epsilon = self.epsilon * cv2.arcLength(contour, True)
        self._box = np.squeeze(cv2.approxPolyDP(contour, epsilon, True)).astype(int)

        if self._box.shape != (4, 2):
            raise ValueError(f'Approximation of biggest contour failed. Expected shape is (4,2), but {self._box.shape} returned.')

        return SudokuCorners.from_unordered_points(self._box)
















