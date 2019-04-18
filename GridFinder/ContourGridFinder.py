import cv2
import numpy as np
from utilities.SudokuCorners import SudokuCorners

from GridFinder.SudokuFinder import GridFinder


class ContourGridFinder(GridFinder):
    def __init__(self, config):
        super().__init__(config)


        self.epsilon = config['epsilon']
        self.output_grid_size = config['output_grid_size']

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






    def find_grid_corners(self, binary_img):
        # ToDo: If his method will be the one that will be used as pripary method than I should add more conditions than
        # ToDo: the biggest contour e.g. shape
        _, contours, _ = cv2.findContours(binary_img,
                                                mode=cv2.RETR_EXTERNAL,
                                                method=cv2.CHAIN_APPROX_SIMPLE)

        contour_area = np.asarray([cv2.contourArea(contour) for contour in contours])
        largest_contour = contours[np.argmax(contour_area)]

        sudoku_corners = self.get_corners(largest_contour)

        return sudoku_corners, largest_contour

    def get_corners(self, contour) -> SudokuCorners:

        #  epsilon is maximum distance from contour to approximated contour
        epsilon = self.epsilon * cv2.arcLength(contour, True)
        self._box = np.squeeze(cv2.approxPolyDP(contour, epsilon, True)).astype(int)

        if self._box.shape != (4, 2):
            raise ValueError(f'Approximation of biggest contour failed. Expected shape is (4,2), but {self._box.shape} returned.')

        return SudokuCorners.from_unordered_points(self._box)

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












