import cv2


class preprocessing:

    blur_kernel = (7, 7)  # must be odd number
    blur_sigma = 0

    thresh_adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_blockSize = 9  # must be odd number
    thresh_C = 2

    # Morphological Transformations
    opening_kernel = (3, 3)  # must be odd number
    closing_kernel = (3, 3)  # must be odd number

