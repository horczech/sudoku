from ImagePreprocessor.ImagePreprocessor import ImagePreprocessor
import cv2
import numpy as np


class BasicImgPreprocessor(ImagePreprocessor):

    def __init__(self, img_path, config_path):
        super().__init__(img_path, config_path)

    def do_preprocessing(self):
        self._blur_img = cv2.GaussianBlur(self.image,
                                          ksize=self.config.blur_kernel,
                                          sigmaX=self.config.blur_sigma)
        self._thresholded_img = cv2.adaptiveThreshold(self._blur_img,
                                                      maxValue=255,
                                                      adaptiveMethod=self.config.thresh_adaptiveMethod,
                                                      thresholdType=cv2.THRESH_BINARY_INV,
                                                      blockSize=self.config.thresh_blockSize,
                                                      C=self.config.thresh_C)

        opening_kernel = np.ones(self.config.opening_kernel, np.uint8)
        closing_kernel = np.ones(self.config.closing_kernel, np.uint8)
        self._opening = cv2.morphologyEx(self._thresholded_img, cv2.MORPH_OPEN, opening_kernel)
        self._opening_closing = cv2.morphologyEx(self._opening, cv2.MORPH_CLOSE, closing_kernel)

        return self._opening_closing
