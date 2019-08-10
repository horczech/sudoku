import pycosat
import numpy as np
from constants import EMPTY_CELL_VALUE


class PythonSatSolver:

    N = 9 #sudoku size (9x9)
    N_sub = 3 #sub-grid size of sudoku aka 3x3 regions

    def __init__(self):
        self.cnf = []
        self.cnf.extend(self._generate_cell_constraints())
        self.cnf.extend(self._generate_column_constraints())
        self.cnf.extend(self._generate_row_constraints())
        self.cnf.extend(self._generate_sub_grid_constraints())

    def solve(self, sudoku):
        if len(sudoku) != 81:
            raise ValueError('Incorrect input format. The expected format is 1D array of length 81')

        sudoku = self._parse_input_data(sudoku)

        cnf = self.cnf + [[self._transform_to_variable_integer(value[0], value[1], value[2])] for value in sudoku]

        solver_result = pycosat.solve(cnf)

        if type(solver_result) is not list:
            raise ValueError('The sollution of given sudoku NOT found :(')

        result = self._parse_solver_result(solver_result)

        return result

    def _parse_solver_result(self, solver_result):
        solver_result = np.asarray(solver_result)
        solver_result = solver_result[solver_result > 0]

        result_indexes = [self._transform_to_variable_triplet(result) for result in solver_result]

        parsed_data = np.full(81, EMPTY_CELL_VALUE, dtype=int)
        for row, col, val in result_indexes:
            parsed_data[row*9+col] = val

        return parsed_data

    def _parse_input_data(self, sudoku_data):
        '''
        Filter just values of full cells

        :param sudoku_data: 1D array of length 81
        :return: array of triplets [row, col, value]
        '''
        parsed_data = []

        sudoku_data = np.asarray(sudoku_data, dtype=np.int32)
        indexes = np.squeeze(np.where(sudoku_data > 0)) #pick just regular sudoku values

        for index in indexes:
            row_id, col_id = divmod(index, 9)
            value = sudoku_data[index]

            parsed_data.append((int(row_id), int(col_id), int(value)))

        return parsed_data

    def _transform_to_variable_integer(self, row, col, val):
        if row not in range(0, 9) or \
           col not in range(0, 9) or \
           val not in range(1, 10):
            raise ValueError('One of the values specifying SUDOKU variable triplet is out of range.')

        return row * self.N * self.N + col * self.N + val

    def _transform_to_variable_triplet(self, variable_integer):
        if variable_integer not in range(1, 730):
            raise ValueError('variable_integer is out of range')

        variable_integer, val = divmod(variable_integer - 1, self.N)
        variable_integer, col = divmod(variable_integer, self.N)
        variable_integer, row = divmod(variable_integer, self.N)

        return row, col, val+1

    def _generate_validation_rules(self, variables):
        '''
        Create constraints ensuring that 9 inserted variables contain all numbers EXACTLY once.
        '''

        cnf = []
        var_count = len(variables)

        if var_count != self.N:
            raise ValueError(f'Number of inserted variables should be exactly 9.')

        # this constrain ensures that there will be at least one number in each in all input variables
        cnf.append(variables)

        # this constrain ensures that there will be at most one number in all input variables
        for i in range(var_count):
            for j in range(i+1, var_count):
                var_1 = variables[i]
                var_2 = variables[j]

                cnf.append([-var_1, -var_2])

        return cnf

    def _generate_cell_constraints(self):
        cnf = []

        for row_id in range(9):
            for col_id in range(9):
                    cnf.extend(self._generate_validation_rules([self._transform_to_variable_integer(row_id, col_id, value) for value in range(1, 10)]))

        return cnf

    def _generate_row_constraints(self):
        cnf = []

        for row_id in range(9):
            for value in range(1, 10):
                    cnf.extend(self._generate_validation_rules([self._transform_to_variable_integer(row_id, col_id, value) for col_id in range(9)]))

        return cnf

    def _generate_column_constraints(self):
        cnf = []

        for col_id in range(9):
            for value in range(1, 10):
                    cnf.extend(self._generate_validation_rules([self._transform_to_variable_integer(row_id, col_id, value) for row_id in range(9)]))

        return cnf

    def _generate_sub_grid_constraints(self):
        cnf = []

        for sub_grid_row_id in range(3):
            for sub_grid_col_id in range(3):
                for value in range(1, 10):
                    var_ids = []

                    for row_id in range(0, 3):
                        for col_id in range(0, 3):
                            var_row_id = 3 * sub_grid_row_id + row_id
                            var_col_id = 3 * sub_grid_col_id + col_id

                            var_ids.append(self._transform_to_variable_integer(var_row_id, var_col_id, value))

                    cnf.extend(self._generate_validation_rules(var_ids))

        return cnf


if __name__ == '__main__':
    sudoku = [1, -2, -2, 9, -2, 7, -2, 8, 3, -2, -2, -2, -2, -2, 3, -2, -2, 6, -2, -2, 3, -2, -2, 5, -2, 2, -2,
              3, -2, 5, -2, -2, 1, -2, -2, -2, 9, 7, 2, 5, -2, 8, 4, 6, 1, -2, -2, -2, 7, -2, -2, 3, -2,
              9, -2, 4, -2, 8, -2, -2, 1, -2, -2, 8, -2, -2, 6, -2, -2, -2, -2, -2, 5, 1, -2, 3, -2, 2, -2,
              -2, 8]


    solver = PythonSatSolver()

    result = solver.solve(sudoku)
    print(result)

    from utilities.Sudoku import Sudoku
    print(Sudoku(result))

