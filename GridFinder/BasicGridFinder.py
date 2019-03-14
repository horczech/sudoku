from GridFinder.GridFinder import GridFinder
import cv2
import numpy as np


class BasicGridFinder(GridFinder):
    def __init__(self, original_img, binary_image, config):
        super().__init__(original_img, binary_image, config)

    def find_grid(self):
        _, self._contours, _ = cv2.findContours(self.binary_img,
                                                mode=cv2.RETR_EXTERNAL,
                                                method=cv2.CHAIN_APPROX_SIMPLE)

        contour_area = np.asarray([cv2.contourArea(contour) for contour in self._contours])

        self._largest_contour = self._contours[np.argmax(contour_area)]

        top_left_pt, top_right_pt, bottom_right_pt, bottom_left_pt = self.get_corners(self._largest_contour)
        self._grid_corners = [top_left_pt, top_right_pt, bottom_right_pt, bottom_left_pt]

        # do the perspective transformation
        source_pts = np.asarray([top_left_pt, top_right_pt, bottom_right_pt, bottom_left_pt], dtype=np.float32)
        destination_pts = np.asarray([[0, 0],
                                      [self.config.GridFinder.output_grid_size, 0],
                                      [self.config.GridFinder.output_grid_size, self.config.GridFinder.output_grid_size],
                                      [0, self.config.GridFinder.output_grid_size]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(source_pts, destination_pts)
        self._straightened_grid = cv2.warpPerspective(self.original_image, M, (self.config.GridFinder.output_grid_size, self.config.GridFinder.output_grid_size))

        return self._straightened_grid

    def get_corners(self, contour):
        epsilon = 0.01 * cv2.arcLength(contour, True)
        self._box = np.squeeze(cv2.approxPolyDP(contour, epsilon, True)).astype(int)

        if self._box.shape != (4, 2):
            raise ValueError(f'Approximation of biggest contour failed. Expected shape is (4,2), but {self._box.shape} returned.')

        # decide which point is top_left, top_right, bottom_left and bottom_right
        pt_sum = np.sum(self._box, axis=1)
        min_idx = np.argmin(pt_sum)
        max_idx = np.argmax(pt_sum)

        top_left_pt = self._box[min_idx]
        bottom_right_pt = self._box[max_idx]

        # filter out top_left_pt and bottom_right_pt
        remaining_pts = []
        for idx, point in enumerate(self._box):
            if idx != min_idx and idx != max_idx:
                remaining_pts.append(point)
            else:
                continue

        remaining_pts = np.asarray(remaining_pts)

        bottom_left_pt = remaining_pts[np.argmin(remaining_pts[:,0])]
        top_right_pt = remaining_pts[np.argmax(remaining_pts[:,0])]

        return top_left_pt, top_right_pt, bottom_right_pt, bottom_left_pt

















