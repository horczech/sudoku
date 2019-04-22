# Code stolen from https://github.com/raahatg21/Digit-Recognition-MNIST-Dataset-with-Keras/blob/master/MNIST_9914.ipynb


import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras import layers
from keras.utils import to_categorical

# VALIDATION_DATA_COUNT = 300
VALIDATION_DATA_COUNT = 10000

EPOCHS = 20
BATCH_SIZE = 128


def learn_model(dataset_path, load_model_path, train_from_scratch=False, save_model_path=None):
    x_train, y_train, x_test, y_test, val_x_train, val_y_train = load_dataset(dataset_path)


    if train_from_scratch:
        print('Build new model')
        model = build_model()
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    else:
        print('Loading saved model...')
        model = models.load_model(load_model_path)

    print(f'Model sumarry:\n {model.summary()}')
    print('...........................\n')

    print('Training the model...')
    history = model.fit(x_train,
                        y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(val_x_train, val_y_train),
                        verbose=2)

    print('Saving the model')

    if save_model_path is None:
        save_model_path = load_model_path

    model.save(save_model_path)

    evaluate_model(history, model, x_test, y_test)


def evaluate_model(history, model, x_test, y_test):
    print('Evaluate model on test dataset')
    test_loss, test_acc = model.evaluate(x_test, y_test)

    print(f'test_loss: {test_loss}\ntest_acc: {test_acc}')

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    plt.figure('Training and Validation Loss')
    epochs = range(1, EPOCHS+1)
    plt.plot(epochs, loss, 'ko', label='Training Loss')
    plt.plot(epochs, val_loss, 'k', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure('Training and Validation Accuracy')
    plt.plot(epochs, acc, 'yo', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'y', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)


def reformat_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        x_train = x_train[y_train != 0]
        y_train = y_train[y_train != 0]
        y_train = y_train - 1

        x_test = x_test[y_test != 0]
        y_test = y_test[y_test != 0]
        y_test = y_test-1

        np.savez(r'KerasDigitRecognition/data/mnist_dataset_without_zero.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)



def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(9, activation='softmax'))

    return model

def load_dataset(dataset_path):
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = load_data(dataset_path)

    print(f'x_train shape: {x_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'x_test shape: {x_test.shape}')
    print(f'y_test shape: {y_test.shape}')
    print('...........................\n')

    # Preprocessing the Data
    print('Preprocessing the data...')
    x_train = x_train.reshape((len(y_train), 28, 28, 1))
    x_train = x_train.astype('float32') / 255

    x_test = x_test.reshape((len(y_test), 28, 28, 1))
    x_test = x_test.astype('float32') / 255

    # Preprocessing the Labels
    y_train = to_categorical(y_train, num_classes=9)
    y_test = to_categorical(y_test, num_classes=9)

    # Validation Split
    val_x_train = x_train[:VALIDATION_DATA_COUNT]
    val_y_train = y_train[:VALIDATION_DATA_COUNT]

    x_train = x_train[VALIDATION_DATA_COUNT:]
    y_train = y_train[VALIDATION_DATA_COUNT:]

    print('Split training dataset into validation and training')
    print(f'val_x_train shape: {val_x_train.shape}')
    print(f'val_y_train shape: {val_y_train.shape}')

    print(f'x_train shape: {x_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print('...........................\n')

    return x_train, y_train, x_test, y_test, val_x_train, val_y_train


if __name__ == '__main__':

    # dataset_path = r'KerasDigitRecognition/data/ocr_dataset.npz'
    dataset_path = r'KerasDigitRecognition/data/mnist_dataset_without_zero.npz'

    load_model_path = r'KerasDigitRecognition/models/model_with_ocr_data.h5'
    save_model_path = r'KerasDigitRecognition/models/model_pokus.h5'

    learn_model(dataset_path, load_model_path, train_from_scratch=False, save_model_path=save_model_path)
