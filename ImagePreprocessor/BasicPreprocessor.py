from ImagePreprocessor.ImagePreprocessor import ImagePreprocessor
import cv2


class BasicImgPreprocessor(ImagePreprocessor):

    def __init__(self, input_image):
        super().__init__(input_image)

    def do_preprocessing(self):
        cv2.imshow('aaa', self.image)
        cv2.waitKey()
