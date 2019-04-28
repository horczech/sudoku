from glob import glob
import os
import cv2
import shutil
from matplotlib import pyplot as plt


def sort(source_dir, target_blur_dir, target_wavy_dir, target_standard_dir, target_rotated_dir):

    img_paths = glob(os.path.join(source_dir, '*.jpg'))

    for img_count, img_path in enumerate(img_paths):

        cv2.imshow('Image', cv2.imread(img_path))
        cv2.waitKey(500)


        is_correct_input = False
        while not is_correct_input:
            print(f'IMG No.{img_count}/{len(img_paths)}')



            print('\n[1] Normal\n[2] Blurred\n[3] Wavy\n[4] Rotated')
            val = int(input('Select: '))

            if val == 1:
                is_correct_input = True
                move_file(img_path, source_dir, target_standard_dir)
                cv2.destroyWindow('Image')

            elif val == 2:
                is_correct_input = True
                move_file(img_path, source_dir, target_blur_dir)
                cv2.destroyWindow('Image')


            elif val == 3:
                is_correct_input = True
                move_file(img_path, source_dir, target_wavy_dir)
                cv2.destroyWindow('Image')


            elif val == 4:
                is_correct_input = True
                move_file(img_path, source_dir, target_rotated_dir)
                cv2.destroyWindow('Image')


            else:
                print(f'Inserted input "{val}" is not acceptable.')
                is_correct_input = False

        print('..................................\n\n')


    print('>>>>>>>>>>>>>>>>DONE<<<<<<<<<<<<<<<<<')

def move_file(img_path, source_dir, target_standard_dir):
    img_name, _ = os.path.basename(img_path).split('.')
    annotation_name = img_name + '.dat'
    target_img_path = os.path.join(target_standard_dir, img_name + '.jpg')
    target_data_path = os.path.join(target_standard_dir, img_name + '.dat')
    source_data_path = os.path.join(source_dir, img_name + '.dat')
    shutil.move(img_path, target_img_path)
    shutil.move(source_data_path, target_data_path)


if __name__ == '__main__':
    source_dir = r'/Users/horczech/Desktop/SUDOKU IMGS/old_structure/all_imgs'

    target_blur_dir = r'/Users/horczech/Desktop/SUDOKU IMGS/new_structure/blured_imgs'
    target_wavy_dir = r'/Users/horczech/Desktop/SUDOKU IMGS/new_structure/wavy_imgs'
    target_standard_dir = r'/Users/horczech/Desktop/SUDOKU IMGS/new_structure/standard_imgs'
    target_rotated_dir = r'/Users/horczech/Desktop/SUDOKU IMGS/new_structure/rotated_imgs'

    sort(source_dir, target_blur_dir, target_wavy_dir, target_standard_dir, target_rotated_dir)



