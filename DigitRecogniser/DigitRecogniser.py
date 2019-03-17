from abc import ABCMeta, abstractmethod


class DigitRecogniser(metaclass=ABCMeta):
    def __init__(self, grid_img, config):
        self.image = grid_img
        self.config = config

    @abstractmethod
    def get_digits(self):
        pass
