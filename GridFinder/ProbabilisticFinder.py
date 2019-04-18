from GridFinder.SudokuFinder import GridFinder
import cv2
import numpy as np


class LineBasedGridFinder(GridFinder):

    def __init__(self, config):
        super().__init__(config)

        self.canny_threshold_1 = config['canny_threshold_1']
        self.canny_threshold_2 = config['canny_threshold_2']
        self.canny_aperture_size = config['canny_aperture_size']

        self.hough_lines_rho = config['hough_lines_rho']
        self.hough_lines_theta_rad = np.deg2rad(config['hough_lines_theta_deg'])
        self.hough_lines_threshold = config['hough_lines_threshold']
        self.hough_lines_min_length = config['hough_lines_min_length']
        self.hough_lines_max_gap = config['hough_lines_max_gap']

        self.line_enlargement_factor = config['line_enlargement_factor']

        self.histogram_bins = config['histogram_bins']
        self.histogram_range = tuple(config['histogram_range'])
        self.histogram_min_distance_between_peaks_deg = config['histogram_min_distance_between_peaks_deg']

    def find_sudoku_corners(self, binary_img):

        self._edges_img = cv2.Canny(binary_img,
                                    threshold1=self.canny_threshold_1,
                                    threshold2=self.canny_threshold_2,
                                    apertureSize=self.canny_aperture_size)

        self._thinning_img = cv2.ximgproc.thinning(binary_img)
        self._lines11 = cv2.HoughLines(self._thinning_img,
                                        rho=self.hough_lines_rho,
                                        theta=self.hough_lines_theta_rad,
                                        threshold=150)
        self._lines11 = np.squeeze(self._lines11)

        self._lines = cv2.HoughLinesP(self._edges_img,
                                      rho=self.hough_lines_rho,
                                      theta=self.hough_lines_theta_rad,
                                      threshold=self.hough_lines_threshold,
                                      minLineLength=self.hough_lines_min_length,
                                      maxLineGap=self.hough_lines_max_gap)
        self._lines = np.squeeze(self._lines)

        self._enlarged_lines = self.enlarge_lines(self._lines, factor=self.line_enlargement_factor)

        # ToDo: somehow cluster lines, pick biggest cluster of lines connect broken lines



        # self._filtered_lines = self.filter_horizontal_and_vertical_lines(self._lines, 1)



    def enlarge_lines(self, lines, factor):
        output_lines = []
        for line in lines:
            x1, y1, x2, y2 = line

            vector = np.array([x2 - x1, y2 - y1])
            vector = vector * factor

            x1 -= vector[0]
            y1 -= vector[1]

            x2 += vector[0]
            y2 += vector[1]

            output_lines.append((x1, y1, x2, y2))

        return np.asarray(output_lines, dtype=int)


    def filter_horizontal_and_vertical_lines(self, lines, histeresis_deg):
        self._theta_deg = np.rad2deg(lines[:, 1])
        hist, bin_edges = np.histogram(self._theta_deg, bins=self.histogram_bins, range=self.histogram_range)

        hist, bin_centers = self.find_peaks(hist, bin_edges, min_peak_distance_deg=self.histogram_min_distance_between_peaks_deg)

        filtered_lines = []
        for angle_deg in bin_centers:
            angle_min_rad = np.deg2rad(np.max([0, angle_deg - histeresis_deg]))
            angle_max_rad = np.deg2rad(np.min([180, angle_deg + histeresis_deg]))

            filtered_lines.append(lines[(angle_min_rad <= lines[:, 1]) & (lines[:, 1] <= angle_max_rad)])

        # ToDo: should be better way how to do it. This is disgusting...
        filtered_lines = np.vstack([filtered_lines[0], filtered_lines[1]])

        return filtered_lines



    def find_peaks(self, hist, bins, min_peak_distance_deg):
        values = np.stack((hist, bins[:-1]), axis=1)

        sorted = values[values[:, 0].argsort()[::-1]]


        peaks = [sorted[0]]
        idx = 1
        while len(peaks) != 2:
            if idx >= sorted.shape[0]:
                raise ValueError('Second peak not found')

            if sorted[idx, 1] >= min_peak_distance_deg:
                peaks.append(sorted[idx])
            idx += 1

        # get center of the bin
        peaks = np.asarray(peaks)
        bin_size = bins[1]-bins[0]
        peaks[:, 1] = peaks[:, 1] + bin_size/2

        return peaks[:, 0], peaks[:, 1]


    def visualize_steps(self, gray_img):
        from matplotlib import pyplot as plt
        from utilities.utils import draw_lines, draw_probabilistic_lines

        binary_img = self.preprocess_image(gray_img)
        self.find_sudoku_corners(binary_img)

    # ------------------ IMG PREPROCESSING ------------------
        plt.figure("Image Preprocessing")

        plt.subplot(2, 4, 1)
        plt.imshow(gray_img, cmap='gray')
        plt.title('Original img')

        plt.subplot(2, 4, 2)
        plt.imshow(self._resized_img, cmap='gray')
        plt.title('Resize')

        plt.subplot(2, 4, 3)
        plt.imshow(self._median_blur_1_img, cmap='gray')
        plt.title('Median blur')

        plt.subplot(2, 4, 4)
        plt.imshow(self._thresholded_img, cmap='gray')
        plt.title('Adaptive thresholding')

        plt.subplot(2, 4, 5)
        plt.imshow(self._median_blur_2_img, cmap='gray')
        plt.title('Median blur 2')

        plt.subplot(2, 4, 6)
        plt.imshow(self._dilated_img, cmap='gray')
        plt.title('Cross Dilatation')

    # ------------------ GRID DETECTION ------------------
        unfiltered_lines_img = draw_probabilistic_lines(self._lines, self._resized_img, line_thickness=2)
        enlarged_lines_img = draw_probabilistic_lines(self._enlarged_lines, self._resized_img, line_thickness=2)
        normal_lines = draw_lines(self._lines11, self._resized_img, line_thickness=2)

        plt.figure("Grid detection")

        plt.subplot(2, 4, 1)
        plt.imshow(binary_img, cmap='gray')
        plt.title('Binary img')

        plt.subplot(2, 4, 2)
        plt.imshow(self._edges_img, cmap='gray')
        plt.title('Canny edge')

        plt.subplot(2, 4, 3)
        plt.imshow(unfiltered_lines_img)
        plt.title('Unfiltered lines')

        # plt.subplot(2, 4, 3)
        # plt.hist(self._theta_deg, bins=self.histogram_bins, range=self.histogram_range)
        # plt.title('Histogram')


        cv2.imshow('binary', binary_img)
        cv2.imshow('hough', unfiltered_lines_img)
        cv2.imshow('canny', self._edges_img)
        cv2.imshow('enlarged', enlarged_lines_img)
        cv2.imshow('NORMAL', normal_lines)
        cv2.imshow('thinning', self._thinning_img)
        cv2.imshow('ORIGINAL', self._resized_img)

        plt.show()





if __name__ == '__main__':
    from glob import glob
    from os import path
    import yaml

    img_format = r'.jpg'
    folder_path = r'sudoku_imgs/easy_dataset'
    folder_path = r'sudoku_imgs/annotated_test_imgs'

    config_path = r'configs/probabilistic_lines_config'

    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    img_path_list = glob(path.join(folder_path, '*' + img_format))

    grid_finder = LineBasedGridFinder(config['grid_finder'])
    for img_path in img_path_list:
        grid_finder.visualize_steps(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))






