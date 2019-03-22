import cv2
from abc import ABCMeta, abstractmethod


class SudokuFinder(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def preprocess_image(self, gray_img):
        pass

    @abstractmethod
    def find_sudoku_corners(self, binary_img):
        pass

    def crop_sudoku(self, gray_img, sudoku_corners):
        pass