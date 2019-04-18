import matplotlib
import cv2
import sys
import os
import argparse
import converter_pipelines


def run(img_path, config_path):
    print(f'OpenCV version\n{cv2.__version__}')
    print(f'Python version\n{sys.version}')
    print('\n----------------------------------\n')

    sudoku_array = converter_pipelines.converter_v1(img_path, config_path)
    print(sudoku_array)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Digitize SUDOKU image')

    parser.add_argument('img', type=str, help='Path to image')
    parser.add_argument('-c','--config', type=str, help='Path to config file')

    args = parser.parse_args()

    if args.config is None:
        args.config = r'configs/config_01'

    print(args.config)
    run(args.img, args.config)
