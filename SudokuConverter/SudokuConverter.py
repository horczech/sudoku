'''
Digit recognition based on tesseract - very slow (cca 200ms per digit)
Digit detection using contours - some digit contours are broken and are not detected
'''


import cv2
import yaml

if __name__ == '__main__':
    from GridFinder.ContourGridFinder import ContourGridFinder
    from Classifier.BasicCellClassifier import BasicDigitRecogniser

    image_path = r'sudoku_imgs/standard_imgs/4.jpg'
    config_path = r'configs/config_03'

    sudoku_img = cv2.imread(image_path)
    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    grid_finder = ContourGridFinder(config['grid_finder'])
    digit_classificator = BasicDigitRecogniser(config['digit_classifier'])

    cropped_sudoku_img, transforamtion_matrix = grid_finder.cut_sudoku_grid(sudoku_img, is_debug_mode=False)
    digital_sudoku = digit_classificator.classify_cells(cropped_sudoku_img, is_debugging_mode=False)

    # # cv2.imshow("Cropped Sudoku Result", digital_sudoku.draw_result(cropped_sudoku_img))
    cv2.imshow("Cropped Sudoku Result", digital_sudoku.draw_full_result(sudoku_img, transforamtion_matrix))
    cv2.waitKey()
