import cv2


def run():
    from ImagePreprocessor.BasicPreprocessor import BasicImgPreprocessor

    img = BasicImgPreprocessor('/Users/horczech/PycharmProjects/sudoku/tests/test_images/test_img_4.jpg')
    img.do_preprocessing()

    a=5

if __name__ == '__main__':
    run()
