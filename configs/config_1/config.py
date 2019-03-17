import cv2


class preprocessing:

    blur_kernel = (7, 7)  # must be odd number
    blur_sigma = 0

    thresh_adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_blockSize = 15  # must be odd number
    thresh_C = 3

    # Morphological Transformations
    opening_kernel = (3, 3)  # must be odd number
    closing_kernel = (3, 3)  # must be odd number


class GridFinder:

    # length of square edge
    output_grid_size = 500


class DigitRecogniser:
    blur_kernel = (7, 7)  # must be odd number
    blur_sigma = 0

    thresh_adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_blockSize = 15  # must be odd number
    thresh_C = 3

    # filtering of horizontal and vertical lines
    _ksize = int(GridFinder.output_grid_size/9)

    horizontal_ksize = (_ksize, 1)
    vertical_ksize = (1, _ksize)

    # filtering contours that does not contain digit
    aspect_ratio_range = (0.2, 0.9)
    min_digit_area = 100  # ToDo: Absolute value is not optimal

    # the space around the digit
    digit_padding = 10
    pytesseract_config = '--psm 10 -c tessedit_char_whitelist=123456789'



