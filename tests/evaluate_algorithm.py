import glob
import os
from constants import EMPTY_CELL_VALUE
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd



# ToDo: WROOOOOOOOOONG!!!!
from main import basic_pipeline


IMG_FORMAT = '.jpg'
ANNOTATION_FORMAT = '.dat'


class EvaluationData:
    def __init__(self, img_path, expected_result):
        self.img_path = img_path
        self.true = expected_result

        self.image_name = os.path.basename(img_path)


def parse_annotation_file(file_path, skip_lines_count=2):
    with open(file_path, 'r') as file:
        lines = file.readlines()[skip_lines_count:]

    string_array = ''.join(lines).replace('\n', ' ')
    int_array = np.array([int(number) for number in string_array.split(' ') if number!=''], dtype=int)

    int_array[int_array==0] = EMPTY_CELL_VALUE

    return int_array




def load_data(data_path):
    image_paths = glob.glob(os.path.join(data_path, '*' + IMG_FORMAT))
    annotation_paths = glob.glob(os.path.join(data_path, '*' + ANNOTATION_FORMAT))

    data = []
    for image_path in image_paths:
        file_name, _ = os.path.splitext(os.path.basename(image_path))
        annotation_file_name = file_name + ANNOTATION_FORMAT
        annotation_paths = glob.glob(os.path.join(data_path, annotation_file_name))

        if len(annotation_paths) == 0:
            raise ValueError(f"Annotation file '{annotation_file_name}' not found for image path '{image_path}'")
        elif len(annotation_paths) > 1:
            raise ValueError(f"More than one annotation files '{annotation_file_name}' found for image path '{image_path}'")
        else:
            annotation_path = annotation_paths[0]


        expected_result = parse_annotation_file(annotation_path)
        data.append(EvaluationData(image_path, expected_result))

    return data


def evaluate(evaluated_method, evaluation_data_path, config_path):
    data_array = load_data(evaluation_data_path)

    all_predicted = []
    all_true = []
    failed_evaluation_count = 0
    for data in data_array:

        try:
            predicted = evaluated_method(data.img_path, config_path)

            print('\n_______________OK___________________')
            print(f'Image name: {data.image_name}')
            print_accuracy(data.true, predicted)

            all_predicted.append(predicted)
            all_true.append(data.true)
        except Exception as e:
            print('\n_______________FAIL___________________')
            print(f'Failed Image name: {data.image_name}')
            print('Exception message:')
            print(e)
            failed_evaluation_count = failed_evaluation_count + 1





    all_predicted = np.array(all_predicted).flatten()
    all_true = np.array(all_true).flatten()
    print('\n__________________________________')
    print('OVERALL STATS')
    print_accuracy(all_true, all_predicted)
    print(f'Number of failed tests: {failed_evaluation_count}')
    print('\n\nConfusion Matrix')
    print(pd.crosstab(all_true, all_predicted, rownames=['True'], colnames=['Predicted'], margins=True))


def print_accuracy(true, pred):
    full_cell_idx = np.argwhere(true != EMPTY_CELL_VALUE)
    empty_cell_idx = np.argwhere(true == EMPTY_CELL_VALUE)
    print(f'    Accuracy - all cells: {accuracy_score(true, pred)}')
    print(f'    Accuracy - full cells: {accuracy_score(true[full_cell_idx], pred[full_cell_idx])}')
    print(f'    Accuracy - empty cells: {accuracy_score(true[empty_cell_idx], pred[empty_cell_idx])}')


if __name__ == '__main__':
    evaluation_data_path = r'sudoku_imgs/easy_dataset'
    config_path = r'configs/config_1/config.py'

    evaluate(basic_pipeline, evaluation_data_path, config_path)
