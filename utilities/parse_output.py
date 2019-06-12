import numpy as np
from constants import SUDOKU_GRID_SIZE, EMPTY_CELL_VALUE
from utilities.Sudoku import Sudoku
from utilities.utils import timeit


N = SUDOKU_GRID_SIZE
nn = int(np.sqrt(N))


def convert_dense_to_native(clause, N=4):
    clause -= 1
    first = clause % N ** 2
    i = (clause - (first)) / N ** 2 + 1
    second = first % N
    j = (first - second) / N + 1
    n = second + 1
    return i, j, n


def make_numb_str(n, N):
    numb_str = str(int(n))
    while len(numb_str) < N:
        numb_str = " " + numb_str
    return numb_str

@timeit
def parse_output(solver_result):
    solver_list = solver_result.decode("utf-8").split('\n')
    if str(solver_list).find('c Answer: 1') == -1:
        return None

    sodoku = np.zeros((N, N))

    for line in solver_list:
        if line and line[0] == "v":
            clauses = line[2:].split(" ")
            for n in clauses:
                n = int(n)
                if n > 0:
                    i, j, n = convert_dense_to_native(n, N=N)
                    i = int(i)
                    j = int(j)
                    sodoku[i - 1][j - 1] = n

    flatten_sudoku = sodoku.flatten().astype(dtype=int)
    flatten_sudoku[flatten_sudoku > 9] = EMPTY_CELL_VALUE
    flatten_sudoku[flatten_sudoku < 1] = EMPTY_CELL_VALUE

    digital_sudoku = Sudoku(flatten_sudoku)

    return digital_sudoku


if __name__ == '__main__':
    parse_output()
