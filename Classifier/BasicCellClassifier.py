from Classifier.CellClassifier import CellClassifier
import cv2
import numpy as np
from constants import ERROR_VALUE, SUDOKU_VALUE_RANGE
from utilities.utils import timeit
from utilities.Sudoku import Sudoku
import pytesseract

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

    def preprocess_image(self, gray_img, is_debugging_mode=False):
        blur_img = cv2.GaussianBlur(gray_img,
                                          ksize=self.blur_kernel,
                                          sigmaX=self.blur_sigma)
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
        try:
            digit_img = cv2.resize(digit_img, (25,25))
            classified_digit = pytesseract.image_to_string(digit_img, lang='eng', config=self.pytesseract_config)
        except:
            classified_digit = -1
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




