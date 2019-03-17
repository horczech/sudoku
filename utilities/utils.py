import cv2
import numpy as np
from constants import EMPTY_CELL_VALUE, ERROR_VALUE
import time


def load_image(img_path, flag=cv2.IMREAD_UNCHANGED):
    img = cv2.imread(img_path, flags=flag)

    if img is None:
        raise AttributeError(f"Input image path is not valid. Path: {img_path}")

    return img


def draw_bboxes(img, bboxes, color=(0,0,255), thickness=2):
    output_img = np.copy(img)

    # check if input image is RGB or Gray
    if len(output_img.shape) != 3:
        output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)

    for bbox in bboxes:
        x, y, w, h = bbox

        top_left_pt = (x, y)
        bottom_right_pt = (x+w, y+h)
        cv2.rectangle(output_img, top_left_pt, bottom_right_pt, color, thickness)

    return output_img


def show_sudoku_grid(sudoku_array):
    # replace empty cells with dots and wrongly classified digits to 'x'
    sudoku_array = sudoku_array.astype(str)

    sudoku_array[sudoku_array == str(EMPTY_CELL_VALUE)] = '.'
    sudoku_array[sudoku_array == str(ERROR_VALUE)] = 'x'


    # First part. Adds pipes and hyphens to inputted string to seperate boxes in the grid.
    old = list(sudoku_array)
    new = []
    count = 1
    for one in old:
        new.append(one)
        if count % 3 == 0 and count % 9 != 0:
            new.append('|')
        if count % 27 == 0 and count < 81:
            [new.append('-') for i in range(1, 12)]
        count += 1

    sudoku_array = ''.join(new)

    # Second part. Prints out a nice grid from the result above.
    row = []
    col = 0
    print('\n\n')
    for one in sudoku_array:
        row.append(one + ' ')
        col += 1
        if col == 11:
            print(''.join(row))
            col = 0
            row = []


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed
