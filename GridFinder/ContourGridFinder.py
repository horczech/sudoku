import cv2
import numpy as np
from utilities.SudokuCorners import SudokuCorners
from utilities.utils import timeit
from sklearn.cluster import DBSCAN



from GridFinder.SudokuFinder import GridFinder


class ContourGridFinder(GridFinder):
    def __init__(self, config):
        super().__init__(config)


        self.epsilon = config['epsilon']
        self.output_grid_size = config['output_grid_size']
        self.contour_area_threshold = config['contour_area_threshold']
        self.angle_deviation_rad = np.deg2rad(config['angle_deviation_deg'])
        self.line_theshold_count = config['line_theshold_count']


    @timeit
    def cut_sudoku_grid(self, sudoku_img, is_debug_mode=False):

        try:
            gray_img = self.convert_to_grayscale(sudoku_img)
            resized_gray_img, fx, fy = self.resize_img(gray_img)

            binary_img = self.preprocess_image(resized_gray_img, is_debug_mode=is_debug_mode)

            grid_corners, grid_contour = self.find_grid_corners(binary_img)
            cropped_sudoku_img, transformation_matrix = self.crop_sudoku(gray_img, grid_corners.get_scaled_corners(fx=fx, fy=fy))
        except:
            print('>>> DETECTION OF SUDOKU GRID FAILED')
            return None



        if is_debug_mode:
            self.visualize_steps(input_img=gray_img,
                                 resized_img=resized_gray_img,
                                 binarized_img=binary_img,
                                 biggest_contour=grid_contour,
                                 grid_corners=grid_corners,
                                 cropped_sudoku_img=cropped_sudoku_img)

        return cropped_sudoku_img

    def filter_out_small_contours(self, contours, threshold):
        return [contour for contour in contours if cv2.contourArea(contour) > threshold]


    def find_grid_corners(self, binary_img):
        # ToDo: If his method will be the one that will be used as pripary method than I should add more conditions than
        # ToDo: the biggest contour e.g. shape
        contours, _ = cv2.findContours(binary_img,
                                                mode=cv2.RETR_EXTERNAL,
                                                method=cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = binary_img.shape[0] * binary_img.shape[1] * self.contour_area_threshold
        filtered_contours = self.filter_out_small_contours(contours, threshold=min_contour_area)
        filtered_contours = self.filter_out_non_quadrilaterals(filtered_contours, epsilon=self.epsilon)
        filtered_contours = self.filter_out_non_sudoku_contours(binary_img, filtered_contours)

        if len(filtered_contours) == 0:
            raise ValueError(f'No sudoku grid found on the image')

        elif len(filtered_contours) == 1:
            largest_contour = filtered_contours

        else:
            contour_area = np.asarray([cv2.contourArea(contour) for contour in filtered_contours])
            largest_contour = contours[np.argmax(contour_area)]

        sudoku_corners = SudokuCorners.from_unordered_points(np.squeeze(largest_contour))

        return sudoku_corners, largest_contour

    def filter_out_non_sudoku_contours(self, binary_img, contours):

        filtered_contours = []
        for contour in contours:
            if self.is_sudoku(binary_img, contour):
                filtered_contours.append(contour)

        return filtered_contours



    def is_sudoku(self, binary_img, contour):

        corners = SudokuCorners.from_unordered_points(np.squeeze(contour))
        cropped_sudoku, _ = self.crop_sudoku(binary_img, corners.get_array())

        lines = np.squeeze(cv2.HoughLines(cropped_sudoku, 1, np.pi/180, 400))
        horizontal_lines, vertical_lines = self.get_horizontal_andvertical_lines(lines)

        horizontal_lines_merged = self.merge_close_lines(horizontal_lines, threshold=binary_img.shape[1]/27, default_angle_rad=np.pi/2)
        vertical_lines_merged = self.merge_close_lines(vertical_lines, threshold=binary_img.shape[0]/27, default_angle_rad=0)

        # from utilities.utils import draw_lines
        # lines_img = draw_lines(lines, cropped_sudoku, (255,0,0), 2)
        #
        #
        # horizontal_line_img = draw_lines(horizontal_lines, cropped_sudoku, (255,0,0), 2)
        # vertical_line_img = draw_lines(vertical_lines, cropped_sudoku, (255,0,0), 2)
        #
        # horizontal_line_merge_img = draw_lines(horizontal_lines_merged, cropped_sudoku, (255,0,0), 2)
        # vertical_line_merge_img = draw_lines(vertical_lines_merged, cropped_sudoku, (255,0,0), 2)
        #
        # cv2.imshow('binary_img', binary_img)
        # cv2.imshow('all lines ', lines_img)
        #
        # cv2.imshow('horizontal_line_img ', horizontal_line_img)
        # cv2.imshow('vertical_line_img ', vertical_line_img)
        #
        # cv2.imshow('horizontal_line_merge_img ', horizontal_line_merge_img)
        # cv2.imshow('vertical_line_merge_img ', vertical_line_merge_img)
        #
        # cv2.waitKey()

        if len(horizontal_lines_merged) >= self.line_theshold_count and len(vertical_lines_merged) >= self.line_theshold_count:
            return True
        else:
            return False

    def merge_close_lines(self, lines, threshold, default_angle_rad):
        lines = np.asarray(lines)

        distances = np.reshape(np.abs(lines[:, 0]), (len(lines), 1))

        clusters = DBSCAN(eps=threshold, min_samples=1).fit_predict(distances)

        merged_lines = []
        for cluster_id in range(0,clusters.max()+1):
            lines_to_merge = np.abs(lines[clusters == cluster_id])
            mean_rho = np.mean(lines_to_merge[:, 0])


            merged_lines.append([mean_rho, default_angle_rad])

        return merged_lines


    def get_horizontal_andvertical_lines(self, lines):

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

        angle_min = np.pi/2 - self.angle_deviation_rad
        angle_max = np.pi/2 + self.angle_deviation_rad

        is_in_range = angle_min <= theta <= angle_max

        if is_in_range:
            return True
        else:
            return False

    def is_vertical(self, line):
        _, theta = line

        is_in_range_pos = 0 <= theta <= self.angle_deviation_rad
        is_in_range_neg = np.pi-self.angle_deviation_rad <= theta <= np.pi

        if is_in_range_neg or is_in_range_pos:
            return True
        else:
            return False

    def filter_out_non_quadrilaterals(self, contours, epsilon):

        filtered_contours = []
        for contour in contours:
            is_quadrilateral, contour = self.is_quadrilateral(contour, epsilon)

            if is_quadrilateral:
                filtered_contours.append(contour)

        return filtered_contours

    def is_quadrilateral(self, contour, epsilon):
        if len(contour) == 4:
            return True, contour

        contour = self.simplify_contour(contour, epsilon)
        if len(contour) == 4:
            return True, contour

        return False, contour

    def simplify_contour(self, contour, epsilon):
        #  epsilon is maximum distance from contour to approximated contour
        epsilon = epsilon * cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, epsilon, True)


    # ToDo: DELETE
    # def get_corners(self, contour) -> SudokuCorners:
    #
    #     #  epsilon is maximum distance from contour to approximated contour
    #     epsilon = self.epsilon * cv2.arcLength(contour, True)
    #     self._box = np.squeeze(cv2.approxPolyDP(contour, epsilon, True)).astype(int)
    #
    #     if self._box.shape != (4, 2):
    #         raise ValueError(f'Approximation of biggest contour failed. Expected shape is (4,2), but {self._box.shape} returned.')
    #
    #     return SudokuCorners.from_unordered_points(self._box)

    def crop_sudoku(self, gray_img, sudoku_corners):

        # do the perspective transformation
        destination_pts = np.asarray([[0, 0],
                                      [self.output_grid_size, 0],
                                      [self.output_grid_size, self.output_grid_size],
                                      [0, self.output_grid_size]],
                                     dtype=np.float32)

        transformation_matrix = cv2.getPerspectiveTransform(sudoku_corners, destination_pts)
        cropped_sudoku_img = cv2.warpPerspective(gray_img, transformation_matrix, (self.output_grid_size, self.output_grid_size))

        return cropped_sudoku_img, transformation_matrix


    def visualize_steps(self, input_img, resized_img, binarized_img, biggest_contour, grid_corners, cropped_sudoku_img):
        from matplotlib import pyplot as plt

        plt.figure("Grid Finder")

        plt.subplot(2, 3, 1)
        plt.imshow(input_img, cmap='gray')
        plt.title('Input img')

        plt.subplot(2, 3, 2)
        plt.imshow(resized_img, cmap='gray')
        plt.title('Resized img')

        plt.subplot(2, 3, 3)
        plt.imshow(binarized_img, cmap='gray')
        plt.title('Binarized img')

        plt.subplot(2, 3, 4)
        # draw contour
        biggest_contour_img = cv2.cvtColor(binarized_img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(biggest_contour_img, [biggest_contour], -1, (255, 0, 0), 5)
        plt.imshow(biggest_contour_img, cmap='brg')
        plt.title('Biggest contour img')

        plt.subplot(2, 3, 5)
        # draw corner points
        corners_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
        for point in grid_corners:
            cv2.circle(corners_img, tuple(point), 15, (255, 0, 0), -1)
        plt.imshow(corners_img)
        plt.title('Grid corners')

        plt.subplot(2, 3, 6)
        plt.imshow(cropped_sudoku_img, cmap='gray')
        plt.title('Cropped sudoku')

        plt.show()



if __name__ == '__main__':
    from glob import glob
    from os import path
    import yaml

    img_format = r'.jpg'
    folder_path = r'sudoku_imgs/easy_dataset'
    folder_path = r'sudoku_imgs/annotated_test_imgs'

    config_path = r'configs/config_03'

    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    img_path_list = glob(path.join(folder_path, '*' + img_format))


    grid_finder = ContourGridFinder(config['grid_finder'])
    for img_path in img_path_list:
        print(img_path)
        grid_finder.cut_sudoku_grid(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), is_debug_mode=True)












