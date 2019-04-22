from glob import glob
import cv2
import os
import numpy as np

def convert(folder_path):


    paths = glob(folder_path + '/*.jpg')


    labels = []
    images = []
    for path in paths:
        class_id = int(os.path.basename(path).split('_')[0]) - 1

        if class_id == -1:
            continue

        # convert to binary image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28,28))

        # blur = cv2.GaussianBlur(img, (3, 3), 0)
        _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


        labels.append(class_id)
        images.append(binary_img)

        # cv2.imshow('original', img)
        # cv2.imshow('binary_img', binary_img)
        # cv2.waitKey()

    return np.asarray(labels), np.asarray(images)








if __name__ == '__main__':
    train_path = r'sudoku_imgs/ocr_data/train'
    test_path = r'sudoku_imgs/ocr_data/test'

    y_train, x_train = convert(train_path)
    y_test, x_test = convert(test_path)

    np.savez(r'KerasDigitRecognition/data/ocr_dataset.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


