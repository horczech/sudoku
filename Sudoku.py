import numpy as np
from constants import SUDOKU_GRID_SIZE, EMPTY_CELL_VALUE, ERROR_VALUE

class Sudoku:
    def __init__(self, cell_values):
        # ToDo: find better name for this field -> it should represent value in every cell of sudoku after flattening of the array
        self.values = np.asarray(cell_values, dtype=int)

    @classmethod
    def from_digit_and_idx(cls, digits, grid_indexes):
        if len(digits) != len(grid_indexes):
            raise ValueError(f'Number of elements  of digits and possitions should match.')

        sudoku_grid = np.empty((SUDOKU_GRID_SIZE, SUDOKU_GRID_SIZE))
        sudoku_grid[:] = EMPTY_CELL_VALUE

        for digit, possition in zip(digits, grid_indexes):
            sudoku_grid[possition[0], possition[1]] = digit

        return cls(sudoku_grid.flatten())


    def __str__(self):
        # replace empty cells with dots and wrongly classified digits to 'x'
        sudoku_array = self.values.astype(str)

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

