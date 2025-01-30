from tensorflow import keras

def load_data():

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() # Load MNIST data from keras
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    return (x_train, y_train), (x_test, y_test)