import cv2
from abc import ABCMeta, abstractmethod


class GridFinder(metaclass=ABCMeta):
    def __init__(self, original_image, binary_image, config):
        self.original_image = original_image
        self.binary_img = binary_image
        self.config = config

    @abstractmethod
    def find_grid(self):
        pass
