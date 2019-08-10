import cv2
import yaml
from SudokuConverter.SudokuConverter import sudoku_img_to_string
from GridFinder.ContourGridFinder import ContourGridFinder
from Classifier.HoughLineClassifier import HoughLineClassifier
import argparse
from Solvers.cpp_sat_solver import CppSatSolver as Solver
from Solvers.python_sat_solver import PythonSatSolver as Solver
from utilities.Sudoku import Sudoku


def solve_sudoku(sudoku_img, grid_finder, digit_classificator):

    input_sudoku = sudoku_img_to_string(sudoku_img, grid_finder, digit_classificator)

    # print(digital_sudoku)

    solver = Solver()
    solved_sudoku = Sudoku(solver.solve(input_sudoku.cell_values))

    if solved_sudoku is None:
        print(">>> Sollution not found")
        solved_sudoku = input_sudoku

    solved_sudoku.set_cropped_cell_bboxes(input_sudoku.cropped_cell_coordinates)
    solved_sudoku.set_transformation_matrix(input_sudoku.transformation_matrix)
    solved_sudoku.set_cropped_sudoku_grid_img(input_sudoku.cropped_sudoku_grid_img)
    solved_sudoku.set_is_solution_value(input_sudoku)

    return solved_sudoku, input_sudoku



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sudoku Solvers')

    parser.add_argument('img', type=str, help='Path to image')
    parser.add_argument('-c','--config', type=str, help='Path to config file')

    args = parser.parse_args()

    if args.config is None:
        args.config = r'configs/config_07'


    image_path, config_path = args.img, args.config


    # image_path = r'sudoku_imgs/web_cam/webcam_clean_1.jpg'
    # config_path = r'configs/config_07'


    image = cv2.imread(image_path)

    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    try:
        grid_finder = ContourGridFinder(config['grid_finder'])
        digit_classificator = HoughLineClassifier(config['digit_classifier'])

        solved_sudoku, input_sudoku = solve_sudoku(image, grid_finder, digit_classificator)

        # Draw result
        solved_sudoku.draw_cropped_result()

        result_img = solved_sudoku.draw_full_result(image)
        cropped_result = solved_sudoku.draw_cropped_result()


        cv2.imshow('frame', result_img)
        cv2.imshow('result', cropped_result)

        print(solved_sudoku)

        cv2.waitKey()
        cv2.destroyAllWindows()


    except Exception as e:
        print(e)
