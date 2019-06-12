import cv2
import yaml
from SudokuConverter.SudokuConverter import sudoku_img_to_string
from GridFinder.ContourGridFinder import ContourGridFinder
from Classifier.HoughLineClassifier import HoughLineClassifier
import subprocess
from utilities.parse_output import parse_output
from utilities.utils import timeit


def solve_sudoku(sudoku_img, grid_finder, digit_classificator):

    input_sudoku = sudoku_img_to_string(sudoku_img, grid_finder, digit_classificator)

    # print(digital_sudoku)

    parsed_sudoku = input_sudoku.parse_sudoku()

    solver_result = solver(parsed_sudoku)

    solved_sudoku = parse_output(solver_result)

    if solved_sudoku is None:
        print(">>> Sollution not found")
        solved_sudoku = input_sudoku

    solved_sudoku.set_cropped_cell_bboxes(input_sudoku.cropped_cell_coordinates)
    solved_sudoku.set_transformation_matrix(input_sudoku.transformation_matrix)
    solved_sudoku.set_cropped_sudoku_grid_img(input_sudoku.cropped_sudoku_grid_img)

    return solved_sudoku, input_sudoku

@timeit
def solver(parsed_sudoku):
    p = subprocess.Popen([r'./Solver/solver.sh', str(3), parsed_sudoku], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    solver_result, err = p.communicate()

    return solver_result

if __name__ == '__main__':
    # image_path = r'sudoku_imgs/standard_imgs/4.jpg'
    image_path = r'sudoku_imgs/phone/web_cam_font2_2.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    config_path = r'configs/config_07'
    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    try:
        grid_finder = ContourGridFinder(config['grid_finder'])
        digit_classificator = HoughLineClassifier(config['digit_classifier'])

        solved_sudoku, input_sudoku = solve_sudoku(image, grid_finder, digit_classificator)

        # Draw result
        solved_sudoku.draw_cropped_result()

        result_img = solved_sudoku.draw_full_result(image)
        cv2.imshow('frame', result_img)

        print(solved_sudoku)

        cv2.waitKey()


    except Exception as e:
        print(e)
