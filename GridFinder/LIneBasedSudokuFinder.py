from GridFinder.SudokuFinder import GridFinder
import cv2
import numpy as np
from scipy.signal import find_peaks
from GridFinder.Line import Line

class LineBasedGridFinder(GridFinder):

    def __init__(self, config):
        super().__init__(config)

        self.hough_lines_rho = config['hough_lines_rho']
        self.hough_lines_theta_rad = np.deg2rad(config['hough_lines_theta_deg'])
        self.hough_lines_threshold = config['hough_lines_threshold']

        self.histogram_bins = config['histogram_bins']
        self.histogram_range = tuple(config['histogram_range'])
        self.histogram_min_distance_between_peaks_deg = config['histogram_min_distance_between_peaks_deg']

        self.filter_lines_angle_hysteresis_deg = config['filter_lines_angle_hysteresis_deg']





    def preprocess_image(self, gray_img):

        self._binary_img = super().preprocess_image(gray_img)
        self._thinning_img = cv2.ximgproc.thinning(self._binary_img)

        return self._thinning_img


    def find_sudoku_corners(self, binary_img):

        self._lines = cv2.HoughLines(binary_img,
                                     rho=self.hough_lines_rho,
                                     theta=self.hough_lines_theta_rad,
                                     threshold=self.hough_lines_threshold)
        self._lines = np.squeeze(self._lines)

        vertical_lines, horizontal_lines = self.filter_horizontal_and_vertical_lines(self._lines, self.filter_lines_angle_hysteresis_deg)


        # sort according to rho
        vertical_lines = vertical_lines[np.abs(vertical_lines[:, 0]) .argsort()]
        horizontal_lines = horizontal_lines[np.abs(horizontal_lines[:, 0]).argsort()]


        # create array of Line objects for both horizontal and vertical array
        # Todo: Refactor aka its disgusting
        self.verticals = []
        for idx, line in enumerate(vertical_lines):
            line_obj = Line(line)
            line_obj.set_id(idx)

            self.verticals.append(line_obj)

        self.horizontals = []
        for idx, line in enumerate(horizontal_lines):
            line_obj = Line(line)
            line_obj.set_id(idx)

            self.horizontals.append(line_obj)



        # compute intersections between horizontal and vertical lines

        for vertical_line in self.verticals:
            for horizontal_line in self.horizontals:

                intersection_pt = self.intersection(vertical_line.get_line(), horizontal_line.get_line())

                if intersection_pt[0] >= 0 and intersection_pt[0] < self.resized_img_shape[1] and \
                     intersection_pt[1] >= 0 and intersection_pt[1] < self.resized_img_shape[0]:

                    vertical_line.add_intersection(intersection_pt, horizontal_line.id)
                    horizontal_line.add_intersection(intersection_pt, vertical_line.id)


        # scan one line in the center and classify orthogonal lines
        self._filtered_horizontal_lines_2 = self.classify_lines(self.verticals, self.horizontals)









        # self._filtered_vertical_lines_2 = self.merge_close_lines(vertical_lines, 5)



        self._filtered_vertical_lines = np.vstack([vertical_lines])
        self._filtered_horizontal_lines = np.vstack([horizontal_lines])



        distance_filtered_lines = self.relative_distance_filter(vertical_lines, 2)

        # from matplotlib import pyplot as plt
        #
        #
        # plt.subplot(1, 2, 1)
        # plt.hist(self._theta_deg, bins=self.histogram_bins, range=self.histogram_range)
        # plt.title('Histogram 1')
        #
        # plt.subplot(1, 2, 2)
        # plt.hist(self._theta_deg, bins=self.histogram_bins, range=self.histogram_range)
        # plt.title('Histogram 2')
        #
        # plt.show()
        #

        # self._filtered_lines = np.vstack([filtered_lines[0], filtered_lines[1]])

    def classify_lines(self, lines1, lines2):
        DISTANCE_THRESHOLD_PX = 20
        DISTANCE_RATIO_THRESH = 1.3

        if not lines1 or not lines2:
            raise ValueError('List with lines is empty.')
            return None

        # get the line in the middle
        middle_line_id = round(len(lines1)/2)
        middle_line = lines1[middle_line_id]

        # check if enough intersections found
        if len(middle_line.intersections) <= 9:
            raise ValueError("Less than 10 intersections found in the middle lane. There is most likely no sudoku on img.")

        intersection_distances = []
        for idx in range(2, len(middle_line.intersections)):
            intersection_1 = np.asarray(middle_line.intersections[idx])
            intersection_2 = np.asarray(middle_line.intersections[idx - 1])
            intersection_3 = np.asarray(middle_line.intersections[idx - 2])

            dist_12 = np.linalg.norm(intersection_1 - intersection_2)
            dist_13 = np.linalg.norm(intersection_1 - intersection_3)


            intersection_distances.append([dist_12, middle_line.intersection_line_ids[idx-1], middle_line.intersection_line_ids[idx]])
            intersection_distances.append([dist_13, middle_line.intersection_line_ids[idx-2], middle_line.intersection_line_ids[idx]])



        # sort parirs of intersection with respect to the distance
        intersection_distances = np.asarray(intersection_distances)
        intersection_distances = intersection_distances[np.argsort(intersection_distances[:, 0])]

        # filter out distances lower than threshold
        intersection_distances = intersection_distances[intersection_distances[:, 0] > DISTANCE_THRESHOLD_PX]

        # find 9 of the most similar distances
        min_dist_diff = np.inf
        min_dist_idx = None
        for idx in range(8, len(intersection_distances)):
            dist_1 = intersection_distances[idx-8, 0]
            dist_2 = intersection_distances[idx, 0]
            dist_diff = np.abs(dist_1-dist_2)

            if dist_diff < min_dist_diff:
                min_dist_diff = dist_diff
                min_dist_idx = idx-8



        # did we found it?
        distance_ratio = max(intersection_distances[min_dist_idx,0], intersection_distances[min_dist_idx+8, 0])/min(intersection_distances[min_dist_idx,0], intersection_distances[min_dist_idx+8, 0])

        if min_dist_idx is None:
            raise ValueError("Minimal distance not found")
        elif distance_ratio > DISTANCE_RATIO_THRESH:
            raise ValueError("Differences between the found intersections is too big")


        lines_ids = np.asarray(np.unique(intersection_distances[min_dist_idx:min_dist_idx+8,1:].flatten()), dtype=int)


        output_lines = []
        for line in lines2:
            if line.id in lines_ids:
                output_lines.append(line)



        return output_lines



















    def intersection(self, line1, line2):
        """Finds the intersection of two lines given in Hesse normal form.

        Returns closest integer pixel locations.
        See https://stackoverflow.com/a/383527/5087436
        """
        rho1, theta1 = line1
        rho2, theta2 = line2
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [x0, y0]

    def merge_close_lines(self, lines, distance):

        # from scipy.spatial import cKDTree

        # # distances = np.abs(lines[:,0])
        #
        # lines[:, 0] = np.abs(lines[:, 0])
        #
        # tree = cKDTree(lines)
        # rows_to_fuse = tree.query_pairs(r=30)

        from scipy.spatial.distance import pdist, squareform
        pdist_res = pdist(lines, 'euclidean')
        d = squareform(pdist_res)
        d = np.ma.array(d, mask=np.isclose(d, 0))

        merged_lines = []
        for idx in range(len(d)):
            merge_idx = d[:, idx] < distance
            merged_line = np.mean(lines[merge_idx], axis=0)
            d[:, merge_idx] = float('inf')

            if not np.any(np.isnan(merged_line)):
                merged_lines.append(merged_line)

        return np.asarray(merged_lines)



    def relative_distance_filter(self, lines, hysteresis_px):


        distances = []
        rhos = np.abs(lines[:, 0])
        for idx, line in enumerate(lines):
            rho, _ = line

            distance_diff = np.abs(np.delete(rhos, idx) - rho)
            distances.append(distance_diff)


        distances = np.asarray(distances).flatten()


        from matplotlib import pyplot as plt
        plt.figure('HISTTTT')
        plt.hist(distances, bins=50)







    def filter_horizontal_and_vertical_lines(self, lines, hysteresis_deg):
        self._theta_deg = np.rad2deg(lines[:, 1])
        hist, bin_edges = np.histogram(self._theta_deg, bins=self.histogram_bins, range=self.histogram_range)

        hist, bin_centers = self.find_peaks(hist, bin_edges, min_peak_distance_deg=self.histogram_min_distance_between_peaks_deg)

        bin_centers = np.sort(bin_centers)

        filtered_lines = []
        for angle_deg in bin_centers:
            angle_min_rad = np.deg2rad(angle_deg - hysteresis_deg)
            angle_max_rad = np.deg2rad(angle_deg + hysteresis_deg)

            angles_pos_rad = lines[:, 1]
            # Todo: double check if abs wont make a trouble in some cases
            angles_neg_rad = lines[:, 1] - np.pi

            lines_pos_bool = (angle_min_rad <= angles_pos_rad) & (angles_pos_rad <= angle_max_rad)
            lines_neg_bool = (angle_min_rad <= angles_neg_rad) & (angles_neg_rad <= angle_max_rad)
            lines_bool = np.logical_or(lines_pos_bool, lines_neg_bool)

            filtered_lines.append(lines[lines_bool, :])

        vertical_lines, horizontal_lines = filtered_lines

        return vertical_lines, horizontal_lines



    def find_peaks(self, hist, bins, min_peak_distance_deg):
        values = np.stack((hist, bins[:-1]), axis=1)

        sorted = values[values[:, 0].argsort()[::-1]]


        peaks = [sorted[0]]
        idx = 1
        while len(peaks) != 2:
            if idx >= sorted.shape[0]:
                raise ValueError('Second peak not found')

            angle_deg = sorted[idx, 1]
            peak_angle_deg = peaks[0][1]

            if np.abs(angle_deg-peak_angle_deg) >= min_peak_distance_deg and np.abs(angle_deg-180-peak_angle_deg) >= min_peak_distance_deg:
                peaks.append(sorted[idx])
            idx += 1

        # get center of the bin
        peaks = np.asarray(peaks)
        bin_size = bins[1]-bins[0]
        peaks[:, 1] = peaks[:, 1] + bin_size/2

        return peaks[:, 0], peaks[:, 1]


    def visualize_steps(self, gray_img):
        from matplotlib import pyplot as plt
        from utilities.utils import draw_lines, draw_probabilistic_lines, draw_line_objects

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

        plt.subplot(2, 4, 7)
        plt.imshow(self._thinning_img, cmap='gray')
        plt.title('Thinning')


    # ------------------ GRID DETECTION ------------------
        line_img_1 = draw_lines(self._lines, self._resized_img, line_thickness=2)

        for horizontal in self.horizontals:
            intersections = horizontal.get_intersections()

            for intersection in intersections:
                cv2.drawMarker(line_img_1, tuple(intersection), (0, 0, 255), cv2.MARKER_CROSS, 10, 3)



        filtered_lines_2_img = draw_line_objects(self._filtered_horizontal_lines_2, self._resized_img, line_thickness=2)



        filtered_vertical_line_img = draw_lines(self._filtered_vertical_lines, self._resized_img, line_thickness=2)
        filtered_horizontal_line_img = draw_lines(self._filtered_horizontal_lines, self._resized_img, line_thickness=2)

        plt.figure("Grid detection")

        plt.subplot(2, 4, 1)
        plt.imshow(binary_img, cmap='gray')
        plt.title('Binary img')

        plt.subplot(2, 4, 2)
        plt.imshow(line_img_1)
        plt.title('Hough lines')

        plt.subplot(2, 4, 3)
        plt.hist(self._theta_deg, bins=self.histogram_bins, range=self.histogram_range)
        plt.title('Histogram')

        plt.subplot(2, 4, 4)
        plt.imshow(filtered_horizontal_line_img)
        plt.title('Horizontal lines')

        plt.subplot(2, 4, 5)
        plt.imshow(filtered_vertical_line_img)
        plt.title('Vertical lines')


        cv2.imshow('binary', self._thinning_img)
        cv2.imshow('hough', line_img_1)
        cv2.imshow('filtered vertical', filtered_vertical_line_img)
        cv2.imshow('filtered horizontal', filtered_horizontal_line_img)
        cv2.imshow('FFFFILTERED', filtered_lines_2_img)
        plt.show()








if __name__ == '__main__':
    from glob import glob
    from os import path
    import yaml

    img_format = r'.jpg'
    folder_path = r'sudoku_imgs/easy_dataset'
    # folder_path = r'sudoku_imgs/annotated_test_imgs'

    config_path = r'configs/hough_lines_config'

    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    img_path_list = glob(path.join(folder_path, '*' + img_format))

    grid_finder = LineBasedGridFinder(config['grid_finder'])
    for img_path in img_path_list:
        print(img_path)
        try:
            grid_finder.visualize_steps(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
        except Exception as e:
            print(e)






