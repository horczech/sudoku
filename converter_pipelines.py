import cv2
import yaml
from SudokuConverter.SudokuConverter import SudokuConverter


def converter_v1(image_path, config_path):
    from GridFinder.ContourBasedSudokuFinder import ContourBasedSudokuFinder
    from CellClassifier.BasicDigitRecogniser import BasicDigitRecogniser

    sudoku_img = cv2.imread(image_path)
    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    grid_finder = ContourBasedSudokuFinder(config['grid_finder'])
    digit_classificator = BasicDigitRecogniser(config['digit_classifier'])

    converter = SudokuConverter(grid_finder, digit_classificator)
    digital_sudoku = converter.convert_image(sudoku_img)

    return digital_sudoku

