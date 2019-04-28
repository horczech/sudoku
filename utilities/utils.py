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

def draw_line_objects(line_objects, image, color=(255, 0, 0), line_thickness=3):
    img = np.copy(image)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for line_obj in line_objects:
        line = line_obj.get_line()
        draw_single_line(color, img, line, line_thickness)

    return img



def draw_lines(lines, image, color=(255, 0, 0), line_thickness=3):
    img = np.copy(image)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for line in lines:
        draw_single_line(color, img, line, line_thickness)

    return img


def draw_single_line(color, img, line, line_thickness):
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img, (x1, y1), (x2, y2), color, line_thickness)


def draw_lines2(lines, image, color=(255, 0, 0), line_thickness=3):
    img = image
    for line in lines:
        draw_single_line(color, img, line, line_thickness)

    return img



def draw_probabilistic_lines(lines, image, color=(255, 0, 0), line_thickness=3):
    img = np.copy(image)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(img, (x1, y1), (x2, y2), color, line_thickness)

    return img

def convert_idx_to_row_col_pos(idx):
    '''
    Sudoku grid can be indexed as row and col both in range 0-9 OR as single index from 0-80
    This function does conversion from single index to row and col format
    :param idx: number between 0 and 80
    :return: (row, col)
    '''

    row = int(idx / 9)
    col = int(idx % 9)

    return (row, col)


def convert_row_col_pos_to_idx(row, col):


    idx = row * 9 + col

    return idx


