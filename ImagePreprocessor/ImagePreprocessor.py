import cv2
from abc import ABCMeta, abstractmethod
from utilities.image import load_image


class ImagePreprocessor(metaclass=ABCMeta):
    def __init__(self, img_path, config):
        self.image = load_image(img_path, cv2.IMREAD_GRAYSCALE)

        self.config = config

    @abstractmethod
    def do_preprocessing(self):
        pass
