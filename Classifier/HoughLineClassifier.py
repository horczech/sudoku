import cv2
import numpy as np
from keras import models
from utilities.utils import timeit
from utilities.Sudoku import Sudoku
from sklearn.cluster import DBSCAN
from collections import namedtuple
from utilities.utils import convert_idx_to_row_col_pos, convert_row_col_pos_to_idx
from constants import ERROR_VALUE

cell = namedtuple('cell', ['x_min', 'x_max', 'y_min', 'y_max'])


class HoughLineClassifier:

    def __init__(self, config):

        self.model = models.load_model(config['model_path'])
        self.blur_kernel = tuple(config['blur_kernel'])
        self.thresh_adaptiveMethod = getattr(cv2, config['thresh_adaptiveMethod'])
        self.thresh_blockSize = config['thresh_blockSize']
        self.thresh_C = config['thresh_C']
        self.aspect_ratio_range = config['aspect_ratio_range']
        self.morph_transf_ksize = tuple(config['morph_transf_ksize'])
        self.hough_line_threshold = config['hough_line_threshold']
        self.angle_deviation_rad = np.deg2rad(config['angle_deviation_deg'])
        self.cell_dead_zone = config['cell_dead_zone']
        self.blob_center_distance_threshold = config['blob_center_distance_threshold']
        self.min_white_px_cell_ratio = config['min_white_px_cell_ratio']
        self.min_contour_area = config['min_contour_area']

    @timeit
    def classify_cells(self, cropped_sudoku_img, is_debugging_mode=False):

        binary_img = self.preprocess_image(cropped_sudoku_img, is_debugging_mode=is_debugging_mode)

        cells_coordinates = self.get_cells_coordinates(binary_img, is_debugging_mode=is_debugging_mode)
        digit_imgs, cell_idxs = self.get_digit_imgs(cropped_sudoku_img, cells_coordinates, is_debug_mode=False)

        sudoku = self.clasiffy_digits(digit_imgs, cell_idxs)
        sudoku.set_cropped_cell_bboxes(cells_coordinates)



        if is_debugging_mode:
            from matplotlib import pyplot as plt

            # draw result
            # result_img = sudoku.draw_result(cropped_sudoku_img)

            plt.figure('Digit classification')

            plt.subplot(2, 3, 1)
            plt.imshow(cropped_sudoku_img, cmap='gray')
            plt.title('Input image')

            plt.subplot(2, 3, 2)
            plt.imshow(binary_img, cmap='gray')
            plt.title('Binarized img')

            # cv2.imshow('RESULT', result_img)

            plt.show()

        return sudoku

    # @timeit
    def clasiffy_digits(self, digit_imgs, cell_idxs):

        digit_list = np.empty((len(digit_imgs), 28, 28, 1))

        for idx, digit_img in enumerate(digit_imgs):
            digit_img = cv2.resize(digit_img, (28, 28))
            digit_img = digit_img.astype('float32') / 255
            digit_list[idx, :, :, 0] = digit_img

        # classify
        classified_digits = self.classify_digit(digit_list)

        return Sudoku.from_digit_and_cell_idx(classified_digits, cell_idxs)

    def preprocess_image(self, gray_img, is_debugging_mode=False):
        blur_1_img = cv2.GaussianBlur(gray_img, self.blur_kernel, 0)

        thresholded_img = cv2.adaptiveThreshold(blur_1_img,
                                                maxValue=255,
                                                adaptiveMethod=self.thresh_adaptiveMethod,
                                                thresholdType=cv2.THRESH_BINARY_INV,
                                                blockSize=self.thresh_blockSize,
                                                C=self.thresh_C)

        cleaned_img = self.remove_small_blobs(thresholded_img)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, self.morph_transf_ksize)
        dilated_img = cv2.morphologyEx(cleaned_img, cv2.MORPH_DILATE, kernel)

        if is_debugging_mode:
            from matplotlib import pyplot as plt

            plt.figure('Preprocessing')

            plt.subplot(2, 3, 1)
            plt.imshow(gray_img, cmap='gray')
            plt.title('input img')

            plt.subplot(2, 3, 2)
            plt.imshow(blur_1_img, cmap='gray')
            plt.title('blur_1_img')

            plt.subplot(2, 3, 3)
            plt.imshow(thresholded_img, cmap='gray')
            plt.title('thresholded_img')

            plt.subplot(2, 3, 4)
            plt.imshow(cleaned_img, cmap='gray')
            plt.title('cleaned_img')

            plt.subplot(2, 3, 5)
            plt.imshow(dilated_img, cmap='gray')
            plt.title('dilated_img')

            plt.show()

        return dilated_img

    def remove_small_blobs(self, thresholded_img):
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(thresholded_img, connectivity=8)
        cleaned_img = np.copy(thresholded_img)
        for label in range(1, nb_components):
            if stats[label, cv2.CC_STAT_AREA] < self.min_contour_area:
                cleaned_img[output == label] = 0
        return cleaned_img

    def get_digit_imgs(self, img, cell_coordinates_array, is_debug_mode=False):

        digit_imgs = []
        cell_idxs = []


        # ToDo: move to config
        cell_padding_ratio = 0.2
        variance_threshold = 70


        for cell_idx, cell_coordinate in enumerate(cell_coordinates_array):
            cell_size = max(cell_coordinate.y_max - cell_coordinate.y_min, cell_coordinate.x_max - cell_coordinate.x_min)
            cell_padding = int(cell_size*cell_padding_ratio)

            y_min = max(0, cell_coordinate.y_min)
            y_max = min(img.shape[0]-1, cell_coordinate.y_max)
            x_min = max(0, cell_coordinate.x_min)
            x_max = min(img.shape[1]-1, cell_coordinate.x_max)

            cropped_cell_center = img[y_min+cell_padding:y_max-cell_padding, x_min+cell_padding:x_max-cell_padding]
            blured_cell_center = cv2.GaussianBlur(cropped_cell_center, (3, 3), 0)

            cell_center_variance = np.var(blured_cell_center)
            if cell_center_variance < variance_threshold:
                if is_debug_mode:
                    print(f'variance of cell: {cell_center_variance}')

                filtered_cell_img = None
            else:
                cropped_cell = img[y_min:y_max, x_min:x_max]

                blur = cv2.GaussianBlur(cropped_cell, (3, 3), 0)
                threshold_val, _ = cv2.threshold(blured_cell_center, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                _, binary_cell_img = cv2.threshold(blur, threshold_val, 255, cv2.THRESH_BINARY_INV)

                filtered_cell_img = self.remove_grid_lines(binary_cell_img)

                if is_debug_mode:
                    cv2.imshow('cropped_cell', cropped_cell)
                    cv2.imshow('th3', binary_cell_img)
                    cv2.imshow('filtered_img', filtered_cell_img)



            if self.is_full_cell(filtered_cell_img, is_debug_mode):
                digit_imgs.append(filtered_cell_img)
                cell_idxs.append(cell_idx)

            if is_debug_mode:
                cv2.imshow('Cropped center', blured_cell_center)
                cv2.waitKey()

        return digit_imgs, cell_idxs

    def is_full_cell(self, binary_cell_img, is_debug_mode):
        if binary_cell_img is None:
            if is_debug_mode:
                print('>>> DEFINITELY EMPTY - The variance is lower than threshold')
            return False


        nb_components, _, stats, centroids = cv2.connectedComponentsWithStats(binary_cell_img, connectivity=8)

        if is_debug_mode:
            print(f'Number of blobs: {nb_components}')

        if nb_components == 1:
            if is_debug_mode:
                print('>>> DEFINITELY NO DIGIT')
            return False

        elif nb_components == 2 and self.has_digit_aspect_ratio(stats[1], is_debug_mode) and \
                self.is_near_cell_center(centroids[1], binary_cell_img.shape, is_debug_mode):
            if is_debug_mode:
                print('>>> DEFINITELY DIGIT')

            return True

        else:
            for label in range(1, nb_components):
                if self.is_near_cell_center(centroids[label], binary_cell_img.shape, is_debug_mode) and self.is_big_enough(
                        binary_cell_img, is_debug_mode):

                    if is_debug_mode:
                        print('>>> MOST LIKELY DIGIT')
                    return True

            if is_debug_mode:
                print(">>> MOST LIKELY NO DIGIT")
            return False

    def is_big_enough(self, binary_cell_img, is_debug_mode=False):
        pixels = binary_cell_img.flatten()

        white_px_count = len(pixels[pixels > 0])
        cell_px_count = len(pixels)

        ratio = white_px_count / cell_px_count

        if is_debug_mode:
            print(f'Ratio of white pixels: {ratio}')

        if ratio > self.min_white_px_cell_ratio:
            return True
        else:
            return False

    def is_near_cell_center(self, blob_center, cell_shape, is_debug_mode=False):
        cell_center = np.array([cell_shape[1] / 2, cell_shape[0] / 2])

        dist = np.linalg.norm(blob_center - cell_center)

        dist_cell_ratio = dist / np.max(cell_shape)

        if is_debug_mode:
            print(f'dist_cell_ratio = {dist_cell_ratio}')

        if dist_cell_ratio < self.blob_center_distance_threshold:
            return True
        else:
            return False

    def has_digit_aspect_ratio(self, blob_stats, is_debug_mode=False):
        width = blob_stats[cv2.CC_STAT_WIDTH]
        height = blob_stats[cv2.CC_STAT_HEIGHT]

        ratio = width / height

        if is_debug_mode:
            print(f'Blob aspect ratio = {ratio}')

        if self.aspect_ratio_range[0] < ratio < self.aspect_ratio_range[1]:
            return True
        else:
            return False

    def remove_grid_lines(self, thresholded_img):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresholded_img, connectivity=8)
        filtered_img = np.copy(thresholded_img)
        for label in range(1, nb_components):
            blob_stats = stats[label]
            if self.is_near_bottom_edge(blob_stats, thresholded_img.shape) or \
                    self.is_near_top_edge(blob_stats, thresholded_img.shape) or \
                    self.is_near_left_edge(blob_stats, thresholded_img.shape) or \
                    self.is_near_right_edge(blob_stats, thresholded_img.shape):
                filtered_img[output == label] = 0
        return filtered_img

    def is_near_top_edge(self, blob_stats, cell_shape):
        y_min = blob_stats[cv2.CC_STAT_TOP]

        cell_height = cell_shape[0]
        threshold = cell_height * self.cell_dead_zone

        return y_min <= threshold

    def is_near_bottom_edge(self, blob_stats, cell_shape):
        y_max = blob_stats[cv2.CC_STAT_TOP] + blob_stats[cv2.CC_STAT_HEIGHT]

        cell_height = cell_shape[0]
        threshold = cell_height - cell_height * self.cell_dead_zone

        return y_max >= threshold

    def is_near_left_edge(self, blob_stats, cell_shape):
        x_min = blob_stats[cv2.CC_STAT_LEFT]

        cell_width = cell_shape[1]
        threshold = cell_width * self.cell_dead_zone

        return x_min <= threshold

    def is_near_right_edge(self, blob_stats, cell_shape):
        x_max = blob_stats[cv2.CC_STAT_LEFT] + blob_stats[cv2.CC_STAT_WIDTH]

        cell_width = cell_shape[1]
        threshold = cell_width - cell_width * self.cell_dead_zone

        return x_max >= threshold

    def get_cells_coordinates(self, binary_img, is_debugging_mode=False):

        lines = np.squeeze(cv2.HoughLines(binary_img, 1, np.pi / 180, self.hough_line_threshold))
        horizontal_lines, vertical_lines = self.get_horizontal_and_vertical_lines(lines)

        horizontal_lines_merged = self.merge_close_lines(horizontal_lines, threshold=binary_img.shape[1] / 27)
        vertical_lines_merged = self.merge_close_lines(vertical_lines, threshold=binary_img.shape[0] / 27)

        if not is_debugging_mode and (len(horizontal_lines_merged) < 10 or len(vertical_lines_merged) < 10):
            raise ValueError(f'SUDOKU grid was not detected. Number of horizontal lines: {len(horizontal_lines_merged)}. Number of vertical lines: {len(vertical_lines_merged)}')

        # get intersections
        intersection_matrix = self.get_intersections(horizontal_lines_merged, vertical_lines_merged)

        # get cell coordinates
        cell_coordinates = self.calculate_cell_coordinates(intersection_matrix)

        if is_debugging_mode:
            from matplotlib import pyplot as plt

            from utilities.utils import draw_lines
            all_lines = draw_lines(lines, binary_img, (0, 0, 255), 2)
            horizontal_and_vertical_lines = draw_lines(horizontal_lines + vertical_lines, binary_img, (0, 0, 255), 2)
            merged_horizontal_and_vertical_lines = draw_lines(horizontal_lines_merged + vertical_lines_merged,
                                                              binary_img, (0, 0, 255), 2)

            # draw intersections
            for i in range(0, 10):
                for j in range(0, 10):
                    cv2.circle(merged_horizontal_and_vertical_lines, tuple(intersection_matrix[i, j]), 5, (255, 0, 0),
                               cv2.FILLED)

            cv2.imshow('merged_horizontal_and_vertical_lines', merged_horizontal_and_vertical_lines)

            plt.figure('Cell extraction')

            plt.subplot(1, 3, 1)
            plt.imshow(horizontal_and_vertical_lines)
            plt.title('horizontal_and_vertical_lines')

            plt.subplot(1, 3, 2)
            plt.imshow(all_lines)
            plt.title('all_lines')

            plt.subplot(1, 3, 3)
            plt.imshow(merged_horizontal_and_vertical_lines)
            plt.title('merged_horizontal_and_vertical_lines')

            plt.show()

        if len(horizontal_lines_merged) < 10 or len(vertical_lines_merged) < 10:
            raise ValueError('SUDOKU grid was not detected. Number of horizontal lines: {len(horizontal_lines_merged)}. Number of vertical lines: {len(vertical_lines_merged)}')

        return cell_coordinates

    def calculate_cell_coordinates(self, intersection_matrix):

        cell_coordinates = []
        for row_id in range(0, 9):
            for col_id in range(0, 9):
                top_left_pt = intersection_matrix[row_id, col_id]
                bottom_right_pt = intersection_matrix[row_id + 1, col_id + 1]

                cell_coordinates.append(cell(x_min=top_left_pt[0],
                                             x_max=bottom_right_pt[0],
                                             y_min=top_left_pt[1],
                                             y_max=bottom_right_pt[1]))
        return cell_coordinates

    def get_intersections(self, horizontal_lines, vertical_lines):

        horizontal_lines = np.asarray(horizontal_lines)
        vertical_lines = np.asarray(vertical_lines)

        # sort according to rho
        horizontal_lines = horizontal_lines[np.abs(horizontal_lines[:, 0]).argsort()]
        vertical_lines = vertical_lines[np.abs(vertical_lines[:, 0]).argsort()]

        intersections = np.zeros((10, 10, 2), dtype=int)
        for idx_vertical, vertical_line in enumerate(vertical_lines):
            for idx_horizontal, horizontal_line in enumerate(horizontal_lines):
                intersections[idx_horizontal, idx_vertical] = self.intersection(horizontal_line, vertical_line)

        return intersections

    def intersection(self, line1, line2):
        rho1, theta1 = line1
        rho2, theta2 = line2
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
            ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return x0, y0

    def get_horizontal_and_vertical_lines(self, lines):

        horizontal_lines = []
        vertical_lines = []
        for line in lines:
            if self.is_horizontal(line):
                horizontal_lines.append(line)
            elif self.is_vertical(line):
                vertical_lines.append(line)
            else:
                continue

        return horizontal_lines, vertical_lines

    def is_horizontal(self, line):
        _, theta = line

        angle_min = np.pi / 2 - self.angle_deviation_rad
        angle_max = np.pi / 2 + self.angle_deviation_rad

        is_in_range = angle_min <= theta <= angle_max

        if is_in_range:
            return True
        else:
            return False

    def is_vertical(self, line):
        _, theta = line

        is_in_range_pos = 0 <= theta <= self.angle_deviation_rad
        is_in_range_neg = np.pi - self.angle_deviation_rad <= theta <= np.pi

        if is_in_range_neg or is_in_range_pos:
            return True
        else:
            return False

    def merge_close_lines(self, lines, threshold):
        lines = np.asarray(lines)

        distances = np.reshape(np.abs(lines[:, 0]), (len(lines), 1))

        clusters = DBSCAN(eps=threshold, min_samples=1).fit_predict(distances)

        merged_lines = []
        for cluster_id in range(0, clusters.max() + 1):
            lines_to_merge = np.asarray(lines[clusters == cluster_id])
            idxs = np.array(range(0, lines_to_merge.shape[0]))

            res = np.argsort(np.abs(lines_to_merge[:, 0]))
            sorted_lines = lines_to_merge[res]
            sorted_idxs = idxs[res]

            merged_lines.append(lines_to_merge[sorted_idxs[int(lines_to_merge.shape[0]/2)], :])

        return merged_lines


    def classify_digit(self, digit_list):

        try:
            classified_digits = self.model.predict(digit_list)
            # we have to add 1 because we have numbers 1-9 so the class on zero position is 1
            classified_digits = np.argmax(classified_digits, axis=1) + 1

            classified_digits[classified_digits == 0] = ERROR_VALUE
        except:
            classified_digits = np.ones(len(digit_list), dtype=int)*ERROR_VALUE

        return classified_digits
