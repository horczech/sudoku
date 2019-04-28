from abc import ABCMeta, abstractmethod
import cv2
import numpy as np


class CellClassifier(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config

        self.blur_kernel = tuple(config['blur_kernel'])

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

    @abstractmethod
    def classify_cells(self, cropped_sudoku_img):
        pass

    def preprocess_image(self, gray_img, is_debugging_mode=False):
        blur_img = cv2.GaussianBlur(gray_img, ksize=self.blur_kernel, sigmaX=0)
        thresholded_img = cv2.adaptiveThreshold(blur_img,
                                                  maxValue=255,
                                                  adaptiveMethod=self.thresh_adaptiveMethod,
                                                  thresholdType=cv2.THRESH_BINARY_INV,
                                                  blockSize=self.thresh_blockSize,
                                                  C=self.thresh_C)

        opening_kernel = np.ones((3, 3), np.uint8)
        morph_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_OPEN, opening_kernel)


        # filter out horizontal and vertical lines
        horizontal_lines_img = self.filter_lines(morph_img, is_horizontal=True)
        vertical_lines_img = self.filter_lines(morph_img, is_horizontal=False)
        grid_lines_img = cv2.bitwise_or(horizontal_lines_img, vertical_lines_img)
        gridless_img = cv2.bitwise_and(morph_img, morph_img, mask=cv2.bitwise_not(grid_lines_img))


        if is_debugging_mode:
            import matplotlib.pyplot as plt

            plt.figure('Image preprocessing')

            plt.subplot(2,3,1)
            plt.imshow(gray_img, cmap='gray')
            plt.title('Input image')

            plt.subplot(2,3,2)
            plt.imshow(blur_img, cmap='gray')
            plt.title('Blurred img')

            plt.subplot(2,3,3)
            plt.imshow(thresholded_img, cmap='gray')
            plt.title('Thresholded img')

            plt.subplot(2,3,4)
            plt.imshow(morph_img, cmap='gray')
            plt.title('Morphological transf. img')

            plt.subplot(2,3,5)
            plt.imshow(grid_lines_img, cmap='gray')
            plt.title('Grid lines')

            plt.subplot(2,3,6)
            plt.imshow(gridless_img, cmap='gray')
            plt.title('Img without grid lines')

            plt.show()

        return gridless_img

    def get_digit_bboxes(self, cropped_binary_img):
        # find digits and its possitions
        contours, _ = cv2.findContours(cropped_binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_digit_bboxes = []
        unfiltered_digit_bboxes = []
        for contour in contours:
            boundary_box = cv2.boundingRect(contour)
            unfiltered_digit_bboxes.append(boundary_box)
            x, y, w, h = boundary_box

            aspect_ratio = w/h
            area = w*h
            if self.aspect_ratio_range[0] < aspect_ratio < self.aspect_ratio_range[1] and area > self.min_digit_area:
                filtered_digit_bboxes.append(boundary_box)

        return np.asarray(filtered_digit_bboxes)

    def get_digit_grid_indexes(self, binary_img, digit_bboxes):
        cell_size = binary_img.shape[0]/9

        grid_positions = []
        for digit_bbox in digit_bboxes:
            x, y, w, h = digit_bbox

            # format [row, column]
            center = np.array([y + h/2, x + w/2])
            grid_positions.append(np.floor(center/cell_size).astype(int))

        return np.asarray(grid_positions)


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



