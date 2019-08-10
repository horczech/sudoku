import numpy as np
from constants import SUDOKU_GRID_SIZE, ERROR_VALUE, EMPTY_CELL_VALUE
import subprocess
from utilities.parse_output import parse_output


class CppSatSolver:

    def solve(self, sudoku_array):
        if len(sudoku_array) != 81:
            raise ValueError('Incorrect input format. The expected format is 1D array of length 81')


        sudoku = self._parse(sudoku_array)

        p = subprocess.Popen([r'./Solvers/sat_cpp/solver.sh', str(3), sudoku], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        solver_result, err = p.communicate()

        solved_sudoku = parse_output(solver_result)

        if solved_sudoku is None:
            raise ValueError('The sollution of given sudoku NOT found :(')


        return solved_sudoku

    def _parse(self, sudoku_data):
        output_string = ''

        sudoku = np.reshape(sudoku_data, (9, 9))

        for row_id, row in enumerate(sudoku):
            for col_id, digit in enumerate(row):
                if digit != EMPTY_CELL_VALUE and digit != ERROR_VALUE:
                    output_string += str(self._convert_nativ_to_dense(SUDOKU_GRID_SIZE, row_id+1, col_id+1, digit)) + "0\n"

        return output_string

    def _convert_nativ_to_dense(self, N, i, j, n):
        n = N ** 2 * (i - 1) + N * (j - 1) + (n - 1) + 1
        return str(n) + " "




if __name__ == '__main__':
    sudoku = [1, -2, -2, 9, -2, 7, -2, 8, 3, -2, -2, -2, -2, -2, 3, -2, -2, 6, -2, -2, 3, -2, -2, 5, -2, 2, -2,
              3, -2, 5, -2, -2, 1, -2, -2, -2, 9, 7, 2, 5, -2, 8, 4, 6, 1, -2, -2, -2, 7, -2, -2, 3, -2,
              9, -2, 4, -2, 8, -2, -2, 1, -2, -2, 8, -2, -2, 6, -2, -2, -2, -2, -2, 5, 1, -2, 3, -2, 2, -2,
              -2, 8]

    solver = CppSatSolver()
    result = solver.solve(sudoku)
    print(result)

    from utilities.Sudoku import Sudoku
    print(Sudoku(result))

