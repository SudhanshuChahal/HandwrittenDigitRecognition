from tensorflow import keras
import numpy as np

def load_data():

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()   # Load MNIST data from keras

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255   # Normalize pixel values to be between 0 and 1

    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]   # Add a channel dimension

    return (x_train, y_train), (x_test, y_test)