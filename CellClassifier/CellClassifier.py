from abc import ABCMeta, abstractmethod


class CellClassifier(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def preprocess_image(self, gray_img):
        pass

    @abstractmethod
    def get_digit_bboxes(self, cropped_binary_img):
        pass

    @abstractmethod
    def get_digit_grid_indexes(self, binary_img, digit_bboxes):
        pass

    @abstractmethod
    def clasiffy_digits(self, binary_img, digit_bboxes, grid_indexes):
        pass




