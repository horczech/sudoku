


class SudokuCell:
    def __init__(self, cell_img, row_id, col_id):
        self.image = cell_img
        self.row_id = row_id
        self.col_id = col_id

    def get_position(self):
        return 9 * self.row_id + self.col_id

    def get_row_col_id(self):
        return (self.row_id, self.col_id)

    def get_bbox(self, img_shape, padding=5):
        cell_width = img_shape[1]/9
        cell_height = img_shape[0]/9

        x = int(cell_width * self.col_id) + padding
        y = int(cell_height * self.row_id) + padding
        w = int(cell_width) - padding
        h = int(cell_height) - padding

        return x, y, w, h
