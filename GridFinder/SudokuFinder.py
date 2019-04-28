import cv2
from abc import ABCMeta, abstractmethod


class GridFinder(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config

        self.resized_img_shape = tuple(config['resized_img_shape'])

        self.median_ksize = config['blur_kernel']

        self.thresh_adaptiveMethod = getattr(cv2, config['thresh_adaptiveMethod'])
        self.thresh_blockSize = config['thresh_blockSize']
        self.thresh_C = config['thresh_C']

        self.morph_transf_ksize = tuple(config['morph_transf_ksize'])

    def preprocess_image(self, gray_img, is_debug_mode=False):

        median_blur_1_img = cv2.medianBlur(gray_img, self.median_ksize)

        thresholded_img = cv2.adaptiveThreshold(median_blur_1_img,
                                                  maxValue=255,
                                                  adaptiveMethod=self.thresh_adaptiveMethod,
                                                  thresholdType=cv2.THRESH_BINARY_INV,
                                                  blockSize=self.thresh_blockSize,
                                                  C=self.thresh_C)

        median_blur_2_img = cv2.medianBlur(thresholded_img, self.median_ksize)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, self.morph_transf_ksize)
        dilated_img = cv2.morphologyEx(median_blur_2_img, cv2.MORPH_DILATE, kernel)

        if is_debug_mode:
            from matplotlib import pyplot as plt

            plt.figure("Image Preprocessing")

            plt.subplot(2, 3, 1)
            plt.imshow(gray_img, cmap='gray')
            plt.title('Input img')

            plt.subplot(2, 3, 2)
            plt.imshow(median_blur_1_img, cmap='gray')
            plt.title('Median blur')

            plt.subplot(2, 3, 3)
            plt.imshow(thresholded_img, cmap='gray')
            plt.title('Adaptive thresholding')

            plt.subplot(2, 3, 4)
            plt.imshow(median_blur_2_img, cmap='gray')
            plt.title('Median blur 2')

            plt.subplot(2, 3, 5)
            plt.imshow(dilated_img, cmap='gray')
            plt.title('Cross Dilatation')

            plt.show()

        return dilated_img

    def resize_img(self, img):
        fx = self.resized_img_shape[1]/img.shape[1]
        fy = self.resized_img_shape[0]/img.shape[0]

        resized_img = cv2.resize(img, self.resized_img_shape)

        return resized_img, fx, fy

    def convert_to_grayscale(self, image):
        # if RGB convert to Gray
        if len(image.shape) > 2:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image

    @abstractmethod
    def cut_sudoku_grid(self, sudoku_img):
        pass

