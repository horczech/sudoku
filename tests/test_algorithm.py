import pytest
import glob
from ImagePreprocessor.BasicPreprocessor import BasicImgPreprocessor
from GridFinder.BasicGridFinder import BasicGridFinder
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utilities.image import load_image


TEST_IMG_PATH = r'tests/test_images'


@pytest.mark.parametrize("sudoku_img_path", glob.glob(TEST_IMG_PATH + '/*.jpg'))
def test_preprocessing(sudoku_img_path):
    from configs.default_config import preprocessing as config

    sudoku_image = load_image(sudoku_img_path, cv2.IMREAD_GRAYSCALE)


    preprocessor = BasicImgPreprocessor(sudoku_image, config)
    preprocessor.do_preprocessing()

    plt.subplot(2, 3, 1)
    plt.imshow(preprocessor.image, cmap='gray')
    plt.title('Original')

    plt.subplot(2, 3, 2)
    plt.imshow(preprocessor._blur_img, cmap='gray')
    plt.title('Blurred')

    plt.subplot(2, 3, 3)
    plt.imshow(preprocessor._thresholded_img, cmap='gray')
    plt.title('thresholded image')

    plt.subplot(2, 3, 4)
    plt.imshow(preprocessor._opening_closing, cmap='gray')
    plt.title('opening and closing image')

    plt.show()


@pytest.mark.parametrize("sudoku_img_path", glob.glob(TEST_IMG_PATH + '/*.jpg'))
def test_grid_finder(sudoku_img_path):
    from configs.default_config import GridFinder as grid_finder_config
    from configs.default_config import preprocessing as preprocessing_config

    sudoku_image = load_image(sudoku_img_path, cv2.IMREAD_GRAYSCALE)

    preprocessor = BasicImgPreprocessor(sudoku_image, preprocessing_config)
    binary_image = preprocessor.do_preprocessing()


    grid_finder = BasicGridFinder(sudoku_image, binary_image, grid_finder_config)
    transformed_grid = grid_finder.find_grid()







    contour_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    biggest_contour_black_img = np.zeros_like(contour_img)
    corners_img = cv2.cvtColor(sudoku_image, cv2.COLOR_GRAY2BGR)


    cv2.drawContours(contour_img, grid_finder._contours, -1, (0, 255, 0), 10)
    cv2.drawContours(contour_img, [grid_finder._largest_contour], 0, (255, 0, 0), 10)
    cv2.drawContours(biggest_contour_black_img, [grid_finder._largest_contour], 0, (255, 0, 0), 10)


    # draw corner points
    for point in grid_finder._grid_corners:
        cv2.circle(corners_img, tuple(point), 50, (255, 0, 0), -1)




    plt.subplot(2, 3, 1)
    plt.imshow(sudoku_image, cmap='gray')
    plt.title('Original')

    plt.subplot(2, 3, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.title('binary_image')

    plt.subplot(2, 3, 3)
    plt.imshow(contour_img)
    plt.title('contour_img')

    plt.subplot(2, 3, 4)
    plt.imshow(biggest_contour_black_img)
    plt.title('biggest_contour_black_img')

    plt.subplot(2, 3, 5)
    plt.imshow(corners_img)
    plt.title('corners')

    plt.subplot(2, 3, 6)
    plt.imshow(transformed_grid, cmap='gray')
    plt.title('Transformed image')


    plt.show()
