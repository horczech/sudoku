import numpy as np
import cv2
import yaml

from GridFinder.ContourGridFinder import ContourGridFinder
from Classifier.HoughLineClassifier import HoughLineClassifier

def run():

    config_path = r'configs/config_06'

    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    grid_finder = ContourGridFinder(config['grid_finder'])
    digit_classificator = HoughLineClassifier(config['digit_classifier'])



    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here

        try:
            cropped_sudoku_img, transforamtion_matrix = grid_finder.cut_sudoku_grid(frame, is_debug_mode=False)
            digital_sudoku = digit_classificator.classify_cells(cropped_sudoku_img, is_debugging_mode=False)

            result_img = digital_sudoku.draw_full_result(frame, transforamtion_matrix)

            # Display the resulting frame
            cv2.imshow('frame', result_img)

        except:
            cv2.imshow('frame', frame)




        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
