import pytest
import glob
from ImagePreprocessor.BasicPreprocessor import BasicImgPreprocessor
import cv2
import matplotlib.pyplot as plt

TEST_IMG_PATH = r'tests/test_images'


@pytest.mark.parametrize("sudoku_img", glob.glob(TEST_IMG_PATH + '/*.jpg'))
def test_algorithm(sudoku_img):
    from configs import default_config as config

    preprocessor = BasicImgPreprocessor(sudoku_img, config)
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

    cv2.imshow("theresh", preprocessor._thresholded_img)
    cv2.imshow("result", preprocessor._opening_closing)

    plt.show()
