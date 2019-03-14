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
    output_grid_size = 300

