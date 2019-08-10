import numpy as np
import cv2
import yaml
from sudoku_solver import solve_sudoku
from utilities.parse_output import parse_output
from SudokuConverter.SudokuConverter import sudoku_img_to_string

from GridFinder.ContourGridFinder import ContourGridFinder
from Classifier.HoughLineClassifier import HoughLineClassifier

def run():

    config_path = r'configs/config_07'

    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    grid_finder = ContourGridFinder(config['grid_finder'])
    digit_classificator = HoughLineClassifier(config['digit_classifier'])

    cap = cv2.VideoCapture(0)

    output_img = None
    is_paused = False
    is_waiting_for_solution = False
    is_sollution_found = False

    while(True):
        if not is_paused:
            # Capture frame-by-frame
            _, frame = cap.read()

            try:
                solved_sudoku, input_sudoku = solve_sudoku(frame, grid_finder, digit_classificator)

                if solved_sudoku is None:
                    print(">>> Sollution not found")
                    solved_sudoku = input_sudoku
                else:
                    if is_waiting_for_solution:
                        is_sollution_found = True

                solved_sudoku.set_cropped_cell_bboxes(input_sudoku.cropped_cell_coordinates)
                solved_sudoku.set_transformation_matrix(input_sudoku.transformation_matrix)

                output_img = solved_sudoku.draw_full_result(frame)

            except:
                output_img = frame



        if is_waiting_for_solution and is_sollution_found:
            print(f'INPUT SUDOKU:\n{input_sudoku}\n\n')
            print(f'SOLVED SUDOKU:\n{solved_sudoku}\n')

            cv2.imshow('camera stream', solved_sudoku.draw_cropped_result())
            cv2.waitKey(0)
            break
        else:
            cv2.imshow('camera stream', output_img)

        pressed_key = cv2.waitKey(1)

        if pressed_key & 0xFF == ord('q'):
            # Q - QUIT
            break
        elif pressed_key & 0xFF == ord('f'):
            # P - PAUSE
            is_paused = True
        elif pressed_key & 0xFF == ord('r'):
            # R - RETURN
            is_paused = False
        elif pressed_key & 0xFF == ord('p'):
            # P - PRINT
            is_waiting_for_solution = True
        else:
            continue



    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
