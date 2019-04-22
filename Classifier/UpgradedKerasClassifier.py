# ToDo: come up with better class name

from Classifier.KerasClassifier import KerasClassifier
import cv2
import numpy as np

class UpgradedKerasClassifier(KerasClassifier):

    def __init__(self, config):
        super().__init__(config)


    def classify_cells(self, cropped_sudoku_img, is_debugging_mode=False):
        binary_img = self.preprocess_image(cropped_sudoku_img, is_debugging_mode=is_debugging_mode)
        cells = self.get_sudoku_cells(binary_img)
        cells = self.filter_out_empy_cells(cells)

        sudoku_cells = self.get_sudoku_cells(binary_img)

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

    def filter_out_empy_cells(self, cells):

        filtered_cells = []
        for cell in cells:
            if not self.is_empty(cell):
                filtered_cells.append(cell)

    def is_empty(self, cell_img):

        contours, _ = cv2.findContours(cell_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


        pixels = cell_img.flatten()
        white_pixels = pixels[pixels>0]
        ratio = len(white_pixels)/len(pixels)
        print(ratio*100)


        cell_img_2 = cv2.cvtColor(cell_img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(cell_img_2,contours, -1, (255, 0,0), 1)



        cv2.imshow('cell_img', cell_img)
        cv2.imshow('cell_img_2', cell_img_2)

        cv2.waitKey()


    def get_sudoku_cells(self, binary_img):
        cell_size = binary_img.shape[0]/9

        cell_img_array = []
        for row_id in range(0,9):
            for col_id in range(0,9):
                x_min = int(col_id*cell_size)
                x_max = int((col_id+1)*cell_size)

                y_min = int(row_id*cell_size)
                y_max = int((row_id+1)*cell_size)

                cell_img = binary_img[y_min:y_max, x_min:x_max]
                cell_img_array.append(cell_img)

                # cv2.imshow('cell', cell_img)
                # cv2.waitKey()

        return cell_img_array