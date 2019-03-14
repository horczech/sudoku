import cv2
from abc import ABCMeta, abstractmethod


class ImagePreprocessor(metaclass=ABCMeta):
    def __init__(self, img_gray, config):
        self.image = img_gray
        self.config = config

    @abstractmethod
    def do_preprocessing(self):
        pass
