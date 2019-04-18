from abc import ABCMeta, abstractmethod


class CellClassifier(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def classify_cells(self, cropped_sudoku_img):
        pass




