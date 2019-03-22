from GridFinder.SudokuFinder import SudokuFinder
from CellClassifier.CellClassifier import CellClassifier
import cv2
import yaml


class SudokuConverter:
    def __init__(self, grid_finder: SudokuFinder, cell_classifier: CellClassifier):
        self.grid_finder = grid_finder
        self.cell_classifier = cell_classifier

    def convert_image(self, image):

        # Extract the SUDOKU GRID from the image
        self._gray_img = self.convert_to_grayscale(image)
        self._binary_img = self.grid_finder.preprocess_image(self._gray_img)
        self._sudoku_corners = self.grid_finder.find_sudoku_corners(self._binary_img)
        self._cropped_sudoku = self.grid_finder.crop_sudoku(self._gray_img, self._sudoku_corners)

        # Classify each cell
        self._cropped_binary_img = self.cell_classifier.preprocess_image(self._gray_img)
        self._digit_bboxes = self.cell_classifier.get_digit_bboxes(self._cropped_binary_img)
        self._grid_indexes = self.cell_classifier.get_digit_grid_indexes(self._cropped_binary_img, self._digit_bboxes)
        self.sudoku = self.cell_classifier.clasiffy_digits(self._cropped_binary_img, self._digit_bboxes, self._grid_indexes)

        return self.sudoku

    def convert_to_grayscale(self, image):
        # if RGB convert to Gray
        if len(image.shape) > 2:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image


if __name__ == '__main__':
    from GridFinder.ContourBasedSudokuFinder import ContourBasedSudokuFinder
    from CellClassifier.BasicDigitRecogniser import BasicDigitRecogniser

    a = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    b = cv2.ADAPTIVE_THRESH_MEAN_C



    image_path = r'sudoku_imgs/easy_dataset/2.jpg'
    config_path = r'configs/config_01'

    sudoku_img = cv2.imread(image_path)
    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    grid_finder = ContourBasedSudokuFinder(config['grid_finder'])
    digit_classificator = BasicDigitRecogniser(config['digit_classifier'])

    converter = SudokuConverter(grid_finder, digit_classificator)
    digital_sudoku = converter.convert_image(sudoku_img)

    print(digital_sudoku)

