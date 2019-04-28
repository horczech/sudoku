import numpy as np
from constants import SUDOKU_GRID_SIZE, EMPTY_CELL_VALUE, ERROR_VALUE
import cv2

class Sudoku:
    def __init__(self, cell_values):
        self.cell_values = np.asarray(cell_values, dtype=int)

        self.cropped_cell_coordinates = None

    def set_cropped_cell_bboxes(self, cell_bboxes):
        if len(cell_bboxes) != 81:
            raise ValueError('The size of cell bboxes array should be 81')

        self.cropped_cell_coordinates = np.asarray(cell_bboxes, dtype=int)

    @classmethod
    def from_digit_and_cell_idx(cls, digits, cell_idxs):
        if len(digits) != len(cell_idxs):
            raise ValueError(f'Number of elements  of digits and possitions should match.')

        sudoku_array = np.ones(81, dtype=int) * EMPTY_CELL_VALUE

        for digit, cell_idx in zip(digits, cell_idxs):
            sudoku_array[cell_idx] = digit

        return cls(sudoku_array)

    @classmethod
    def from_digit_and_row_col_idx(cls, digits, row_col_array):
        if len(digits) != len(row_col_array):
            raise ValueError(f'Number of elements  of digits and possitions should match.')

        sudoku_grid = np.empty((SUDOKU_GRID_SIZE, SUDOKU_GRID_SIZE))
        sudoku_grid[:] = EMPTY_CELL_VALUE

        for digit, row_col_idx in zip(digits, row_col_array):
            row, col = row_col_idx
            sudoku_grid[row, col] = digit

        return cls(sudoku_grid.flatten())


    def __str__(self):
        # replace empty cells with dots and wrongly classified digits to 'x'
        sudoku_array = self.cell_values.astype(str)

        sudoku_array[sudoku_array == str(EMPTY_CELL_VALUE)] = '.'
        sudoku_array[sudoku_array == str(ERROR_VALUE)] = 'x'

        # First part. Adds pipes and hyphens to inputted string to seperate boxes in the grid.
        old = list(sudoku_array)
        new = []
        count = 1
        for one in old:
            new.append(one)
            if count % 3 == 0 and count % 9 != 0:
                new.append('|')
            if count % 27 == 0 and count < 81:
                [new.append('-') for i in range(1, 12)]
            count += 1

        sudoku_array = ''.join(new)

        # Second part. Prints out a nice grid from the result above.
        row = []
        col = 0
        output_string = ''
        for one in sudoku_array:
            row.append(one + ' ')
            col += 1
            if col == 11:
                output_string += ''.join(row) + '\n'
                col = 0
                row = []

        return (output_string)


    def draw_result(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for cell_value, cell_coordinate in zip(self.cell_values, self.cropped_cell_coordinates):
            if cell_value != EMPTY_CELL_VALUE:
                pt1 = (cell_coordinate[0], cell_coordinate[2])
                pt2 = (cell_coordinate[1], cell_coordinate[3])

                text_center = (pt1[0] + int((cell_coordinate[1]-cell_coordinate[0])/2), pt1[1] + int(cell_coordinate[3]-cell_coordinate[2]))

                cv2.rectangle(image, pt1, pt2, (255, 0,0), 3)
                cv2.putText(image, str(cell_value), text_center, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 3)


        return image

    def draw_full_result(self, image, transformation_matrix):
        height, width, _ = image.shape

        empty_img = np.zeros((500, 500), dtype=np.uint8)

        cropped_img_result = self.draw_result(empty_img)

        cropped_img_result = cv2.warpPerspective(cropped_img_result, transformation_matrix, (width, height), flags=cv2.WARP_INVERSE_MAP)


        mask = np.max(cropped_img_result, axis=2).astype(dtype=np.uint8)
        foreground = cv2.bitwise_or(cropped_img_result, cropped_img_result, mask=mask)
        mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_or(image, image, mask=mask)

        result = cv2.bitwise_or(foreground, background)

        return result

