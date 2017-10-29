from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from preprocess_data import *
import numpy as np
import matplotlib.pyplot as plt

# shows Keras outputs a history object that contains
# the training and validation loss for each epoch
def show_learn_history(data, title=None):
    plt.plot(data.history['loss'])
    plt.plot(data.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    if title != None:
        plt.title(title)
    plt.show()

# CNN NVIDIA modyfied to handle images having 64x64x3 size
class Nvidia_Model:

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
        self.__create_model__()
        return

    def __create_model__(self):
        self.model = Sequential()
        # the first layer in a model, provide the keyword argument input_shape (64x64x3 in our case)
        self.model.add(Conv2D(24, (5, 5), input_shape=self.input_shape, activation='elu'))
        self.model.add(MaxPooling2D(padding='same'))
        self.model.add(Conv2D(36, (5, 5), activation='elu'))
        self.model.add(MaxPooling2D(padding='same'))
        self.model.add(Conv2D(48, (5, 5), activation='elu'))
        self.model.add(MaxPooling2D(padding='same'))
        self.model.add(Conv2D(64, (3, 3), activation='elu'))
        self.model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))

        self.model.add(Flatten())

        self.model.add(Dense(576, activation='elu'))
        self.model.add(Dense(100, activation='elu'))
        self.model.add(Dense(50, activation='elu'))
        self.model.add(Dense(10, activation='elu'))
        self.model.add(Dense(1))
        return

    # train model
    def train(self, generator):
        self.model.compile(loss='mse', optimizer='adam')

        history = self.model.fit_generator(
            generator=generator.create_train_data(),
            steps_per_epoch=generator.get_steps_per_epoch(),
            epochs=5,
            verbose=1,
            callbacks=None,
            validation_data=generator.create_validation_data(),
            validation_steps=generator.get_validation_steps())

        print("Saving model..")
        self.model.save('model.h5')

        print("Model summary..")
        print(self.model.summary())
        print(history.history.keys())
        print("Show learn history..")
        show_learn_history(history)
        return

if __name__ == '__main__':

    print("Loading driving log..")
    log_loader = DrivingLogLoader('../3laps_data/driving_log.csv')
    logged_data = log_loader.load_log()
    logged_data = log_loader.filter_log_data(logged_data)
    data_generator = DataGenerator(logged_data)

    print("Train model..")
    model = Nvidia_Model(data_generator.get_output_shape())
    model.train(data_generator)
    print("Training of  model finished")
