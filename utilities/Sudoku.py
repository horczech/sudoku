import numpy as np
from constants import SUDOKU_GRID_SIZE, EMPTY_CELL_VALUE, ERROR_VALUE
import cv2

class Sudoku:
    def __init__(self, cell_values):
        self.cell_values = np.asarray(cell_values, dtype=int)
        self.is_solution_value = None
        self.cropped_cell_coordinates = None
        self.transformation_matrix = None
        self.cropped_sudoku_grid_img = None

    def set_is_solution_value(self, input_sudoku):
        self.is_solution_value = input_sudoku.cell_values == EMPTY_CELL_VALUE


    def set_transformation_matrix(self, transformation_matrix):
        self.transformation_matrix = transformation_matrix

    def set_cropped_cell_bboxes(self, cell_bboxes):
        if len(cell_bboxes) != 81:
            raise ValueError('The size of cell bboxes array should be 81')

        self.cropped_cell_coordinates = np.asarray(cell_bboxes, dtype=int)

    def set_cropped_sudoku_grid_img(self, cropped_sudoku_grid_img):
        self.cropped_sudoku_grid_img = cropped_sudoku_grid_img


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

        sudoku_array[sudoku_array == str(EMPTY_CELL_VALUE)] = '_'
        sudoku_array[sudoku_array == str(ERROR_VALUE)] = '_'

        sudoku_string = ''

        sudoku_string += '+-------+-------+-------+\n|'
        for cell_id, sudoku_cell in enumerate(sudoku_array):
            cell_id = cell_id + 1
            divider_line_idxs = [27, 54, 81]

            sudoku_string += ' ' + str(sudoku_cell)

            if cell_id % 3 == 0:
                sudoku_string += ' |'


            if cell_id in divider_line_idxs:
                sudoku_string += '\n+-------+-------+-------+'
                if cell_id != 81:
                    sudoku_string += '\n|'


            if cell_id % 3 == 0 and cell_id % 9 == 0 and cell_id not in divider_line_idxs:
                sudoku_string += '\n|'

        return (sudoku_string)


    def draw_result(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # draw rectangles around cells
        for cell_value, cell_coordinate in zip(self.cell_values, self.cropped_cell_coordinates):
            if cell_value != EMPTY_CELL_VALUE:
                pt1 = (cell_coordinate[0], cell_coordinate[2])
                pt2 = (cell_coordinate[1], cell_coordinate[3])

                cv2.rectangle(image, pt1, pt2, (255, 0,0), 3)
        # draw numbers
        for cell_value, cell_coordinate, is_solution_value in zip(self.cell_values, self.cropped_cell_coordinates, self.is_solution_value):
            if cell_value != EMPTY_CELL_VALUE:
                pt1 = (cell_coordinate[0], cell_coordinate[2])
                pt2 = (cell_coordinate[1], cell_coordinate[3])

                text_center = (pt1[0] + int((cell_coordinate[1]-cell_coordinate[0])/2), pt1[1] + int(cell_coordinate[3]-cell_coordinate[2]))

                if is_solution_value:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)

                cv2.putText(image, str(cell_value), text_center, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)


        return image



    def draw_full_result(self, image):
        if self.transformation_matrix is None:
            raise ValueError("Transformation matrix is not set.")

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        height, width, _ = image.shape

        empty_img = np.zeros((500, 500), dtype=np.uint8)

        cropped_img_result = self.draw_result(empty_img)

        cropped_img_result = cv2.warpPerspective(cropped_img_result, self.transformation_matrix, (width, height), flags=cv2.WARP_INVERSE_MAP)


        mask = np.max(cropped_img_result, axis=2).astype(dtype=np.uint8)
        foreground = cv2.bitwise_or(cropped_img_result, cropped_img_result, mask=mask)
        mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_or(image, image, mask=mask)

        result = cv2.bitwise_or(foreground, background)

        return result

    def draw_cropped_result(self):
        if self.transformation_matrix is None:
            raise ValueError("Transformation matrix is not set.")
        cropped_img = self.cropped_sudoku_grid_img
        if len(cropped_img.shape) == 2:
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)

        empty_img = np.zeros(self.cropped_sudoku_grid_img.shape, dtype=np.uint8)

        cropped_img_result = self.draw_result(empty_img)

        mask = np.max(cropped_img_result, axis=2).astype(dtype=np.uint8)
        foreground = cv2.bitwise_or(cropped_img_result, cropped_img_result, mask=mask)
        mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_or(cropped_img, cropped_img, mask=mask)

        result = cv2.bitwise_or(foreground, background)

        return result

    def convert_nativ_to_dense(self, N, i, j, n):
        n = N ** 2 * (i - 1) + N * (j - 1) + (n - 1) + 1
        return str(n) + " "

    def parse_sudoku(self):
        output_string = ''

        sudoku = np.reshape(self.cell_values, (9, 9))

        for row_id, row in enumerate(sudoku):
            for col_id, digit in enumerate(row):
                if digit != EMPTY_CELL_VALUE and digit != ERROR_VALUE:
                    output_string += str(self.convert_nativ_to_dense(SUDOKU_GRID_SIZE, row_id+1, col_id+1, digit)) + "0\n"

        return output_string


