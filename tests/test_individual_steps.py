import pytest
import glob
from GridFinder.ContourBasedSudokuFinder import ContourBasedSudokuFinder
from CellClassifier.BasicDigitRecogniser import BasicDigitRecogniser
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utilities.utils import load_image, draw_bboxes
import yaml

from converter_pipelines import converter_v1 as convertor
from GridFinder.ContourBasedSudokuFinder import ContourBasedSudokuFinder as GridFinder
from CellClassifier.BasicDigitRecogniser import BasicDigitRecogniser as CellClassifier


TEST_IMG_PATH = r'sudoku_imgs/annotated_test_imgs/'
CONFIG_FILE_PATH = r'configs/config_01'


@pytest.fixture
def config_file():
    with open(CONFIG_FILE_PATH, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    return config


@pytest.fixture
def grid_finder(config_file):
    return GridFinder(config_file['grid_finder'])


@pytest.fixture
def cell_classifier(config_file):
    return CellClassifier(config_file['digit_classifier'])


@pytest.mark.parametrize("sudoku_img_path", glob.glob(TEST_IMG_PATH + '/*.jpg'))
def test_grid_finder(sudoku_img_path, grid_finder):

    try:
        gray_img = load_image(sudoku_img_path, cv2.IMREAD_GRAYSCALE)
        binary_img = grid_finder.preprocess_image(gray_img)
        sudoku_corners = grid_finder.find_sudoku_corners(binary_img)
        cropped_sudoku = grid_finder.crop_sudoku(gray_img, sudoku_corners)
    except ValueError:
        sudoku_corners = np.array([[0, 0], [0, 0],[0, 0],[0, 0]])

        cropped_sudoku = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        center = np.asarray(cropped_sudoku.shape)/2
        cv2.putText(cropped_sudoku, 'FAIL', (int(center[1]), int(center[0])),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), thickness=5)


    # draw contours
    contour_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(contour_img, grid_finder._contours, -1, (255, 0, 0), 2)
    cv2.drawContours(contour_img, [grid_finder._largest_contour], 0, (0, 0, 255), 2)


    # draw corner points
    corners_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    for point in sudoku_corners:
        cv2.circle(corners_img, tuple(point), 30, (255, 0, 0), -1)



    plt.subplot(2, 4, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Original img')

    plt.subplot(2, 4, 2)
    plt.imshow(grid_finder._blur_img, cmap='gray')
    plt.title('Blurred img')

    plt.subplot(2, 4, 3)
    plt.imshow(grid_finder._thresholded_img, cmap='gray')
    plt.title('Thresholded img')

    plt.subplot(2, 4, 4)
    plt.imshow(grid_finder._opening_closing, cmap='gray')
    plt.title('After opening and closing image')

    plt.subplot(2, 4, 5)
    plt.imshow(contour_img)
    plt.title('Contours')

    plt.subplot(2, 4, 6)
    plt.imshow(corners_img)
    plt.title('Corners image')

    plt.subplot(2, 4, 7)
    plt.imshow(cropped_sudoku, cmap='gray')
    plt.title('Cropped image')

    cv2.imshow('Binarized', grid_finder._opening_closing)
    cv2.imshow('Contours', contour_img)

    plt.show()



@pytest.mark.parametrize("sudoku_img_path", glob.glob(TEST_IMG_PATH + '/*.jpg'))
def test_digit_classifier(sudoku_img_path, grid_finder, cell_classifier):


    # Extract the SUDOKU GRID from the image
    gray_img = load_image(sudoku_img_path, cv2.IMREAD_GRAYSCALE)
    binary_img = grid_finder.preprocess_image(gray_img)
    sudoku_corners = grid_finder.find_sudoku_corners(binary_img)
    cropped_sudoku = grid_finder.crop_sudoku(gray_img, sudoku_corners)

    # Classify each cell
    cropped_binary_img = cell_classifier.preprocess_image(cropped_sudoku)
    digit_bboxes = cell_classifier.get_digit_bboxes(cropped_binary_img)
    grid_indexes = cell_classifier.get_digit_grid_indexes(cropped_binary_img, digit_bboxes)
    sudoku = cell_classifier.clasiffy_digits(cropped_binary_img, digit_bboxes, grid_indexes)
    print(sudoku)

    unfiltered_bboxes_img = draw_bboxes(cell_classifier._gridless_img, cell_classifier._unfiltered_digit_bboxes, (255, 0, 0))
    digit_bboxes_img = draw_bboxes(unfiltered_bboxes_img, cell_classifier._filtered_digit_bboxes, (0, 0, 255))

    plt.subplot(2,3,1)
    plt.imshow(cropped_sudoku, cmap='gray')
    plt.title("Input img")

    plt.subplot(2,3,2)
    plt.imshow(cell_classifier._blur_img, cmap='gray')
    plt.title("Blured img")

    plt.subplot(2,3,3)
    plt.imshow(cell_classifier._thresholded_img, cmap='gray')
    plt.title("Thresholded img")

    plt.subplot(2,3,4)
    plt.imshow(cell_classifier._grid_lines_img, cmap='gray')
    plt.title("grid image")

    plt.subplot(2,3,5)
    plt.imshow(cell_classifier._gridless_img, cmap='gray')
    plt.title("Filtered grid")

    plt.subplot(2,3,6)
    plt.imshow(digit_bboxes_img)
    plt.title("non filtered bbox")

    plt.show()
