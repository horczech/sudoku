import numpy as np


class SudokuCorners:
    def __init__(self, top_left_pt, top_right_pt, bottom_right_pt, bottom_left_pt):
        self.bottom_left_pt = bottom_left_pt
        self.bottom_right_pt = bottom_right_pt
        self.top_right_pt = top_right_pt
        self.top_left_pt = top_left_pt

    def get_array(self, dtype=np.float32):
        return np.array([self.top_left_pt, self.top_right_pt, self.bottom_right_pt, self.bottom_left_pt], dtype=dtype)

    @classmethod
    def from_unordered_points(cls, point_array):
        pt_sum = np.sum(point_array, axis=1)
        min_idx = np.argmin(pt_sum)
        max_idx = np.argmax(pt_sum)

        top_left_pt = point_array[min_idx]
        bottom_right_pt = point_array[max_idx]

        # filter out top_left_pt and bottom_right_pt
        remaining_pts = []
        for idx, point in enumerate(point_array):
            if idx != min_idx and idx != max_idx:
                remaining_pts.append(point)
            else:
                continue

        remaining_pts = np.asarray(remaining_pts)

        bottom_left_pt = remaining_pts[np.argmin(remaining_pts[:, 0])]
        top_right_pt = remaining_pts[np.argmax(remaining_pts[:, 0])]

        return cls(top_left_pt, top_right_pt, bottom_right_pt, bottom_left_pt)





