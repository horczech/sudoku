# ToDo: come up with better class name

from Classifier.KerasClassifier import KerasClassifier
import cv2
import numpy as np
from utilities.Sudoku_Cell import SudokuCell
from utilities.utils import timeit
from utilities.Sudoku import Sudoku


class UpgradedKerasClassifier(KerasClassifier):

    def __init__(self, config):
        super().__init__(config)

        self.cell_pixel_ration = config['cell_pixel_ration']
        self.cell_dead_zone = config['cell_dead_zone']

    def classify_cells(self, cropped_sudoku_img, is_debugging_mode=False):
        binary_img = self.preprocess_image(cropped_sudoku_img, is_debugging_mode=is_debugging_mode)
        cells = self.get_sudoku_cells(binary_img)
        cells = self.filter_out_empty_cells(cells)

        sudoku = self.clasiffy_digits(cells)

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
            digit_bboxes = [cell.get_bbox(bbox_img.shape) for cell in cells]
            bbox_img = draw_bboxes(bbox_img, digit_bboxes, thickness=3)
            plt.imshow(bbox_img)
            plt.title('Found bboxes')

            cv2.imshow('Found bboxes', bbox_img)

            plt.show()

        return sudoku

    @timeit
    def clasiffy_digits(self, cells):

        digit_list = np.empty((len(cells), 28, 28, 1))
        # DEBUG_LIST = np.empty((1, 28, 28, 1))
        grid_indexes = []
        for idx, cell in enumerate(cells):
            cell_img = cell.image

            digit_img = cv2.resize(cell_img, (28, 28))
            cv2.imshow('digit_img', digit_img)

            digit_img = digit_img.astype('float32') / 255

            # DEBUG_LIST[0, :, :, 0] = digit_img
            # print(self.classify_digit(DEBUG_LIST))
            # cv2.waitKey()


            digit_list[idx, :, :, 0] = digit_img
            grid_indexes.append(cell.get_row_col_id())

        # classify
        classified_digits = self.classify_digit(digit_list)

        return Sudoku.from_digit_and_idx(classified_digits, grid_indexes)


    def filter_out_empty_cells(self, cells):

        filtered_cells = []
        for cell in cells:
            if not self.is_empty(cell.image):
                filtered_cells.append(cell)

        return filtered_cells

    def is_empty(self, cell_img):

        # if not enough white pixels than cell is empty
        pixels = cell_img.flatten()
        white_pixels = pixels[pixels > 0]
        ratio = len(white_pixels)/len(pixels)

        if ratio <= self.cell_pixel_ration:

            return True

        contours, _ = cv2.findContours(cell_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # check if there is at least one contour with center out of the dead zone aka that is
        # at least self.cell_dead_zone percent away from the cell border
        width = cell_img.shape[1]
        height = cell_img.shape[0]

        x_min = width * self.cell_dead_zone
        x_max = width - x_min
        y_min = height * self.cell_dead_zone
        y_max = height - y_min

        for contour in contours:
            # compute the center of the contour
            M = cv2.moments(contour)

            if M["m00"] == 0:
                cX = 0
                cY = 0
                continue

            else:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                if x_min < cX < x_max and y_min < cY < y_max:
                    return False


            # ToDo:DELETE
            # DEBUG CODE
            # print(ratio * 100)
            #
            # cell_img_2 = cv2.cvtColor(cell_img, cv2.COLOR_GRAY2BGR)
            #
            # cv2.drawContours(cell_img_2,contours, -1, (255, 0,0), 1)
            # cv2.rectangle(cell_img_2, (int(x_min), int(y_min)), (int(x_max), int(y_max)),(0,255,0), 1)
            # cv2.circle(cell_img_2, (cX, cY), 2, (0,0,255), -1)
            #
            # cv2.imshow('cell_img', cell_img)
            # cv2.imshow('cell_img_2', cell_img_2)
            # cv2.waitKey()


        return True

    def get_sudoku_cells(self, binary_img):
        cell_size = binary_img.shape[0]/9

        cell_array = []
        for row_id in range(0, 9):
            for col_id in range(0, 9):
                x_min = int(col_id*cell_size) + self.digit_padding
                x_max = int((col_id+1)*cell_size) - self.digit_padding

                y_min = int(row_id*cell_size) + self.digit_padding
                y_max = int((row_id+1)*cell_size) - self.digit_padding

                cell_img = binary_img[y_min:y_max, x_min:x_max]
                cell = SudokuCell(cell_img, row_id, col_id)

                cell_array.append(cell)

                # cv2.imshow('cell', cell_img)
                # cv2.waitKey()

        return cell_array