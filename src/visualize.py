import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_data
from tensorflow import keras


def visualize_data(num_samples=10):
    (x_test, y_test) = load_data()[1]
    model = keras.models.load_model('../models/model.keras')

    indices = np.random.choice(range(len(x_test)), num_samples)
    plt.figure(figsize=(15, 3))

    for i, index in enumerate(indices):
        img = x_test[index]
        img = 1.0 - x_test[index]
        prediction = np.argmax(model.predict(img[np.newaxis, ...]))

        plt.subplot(1, num_samples, i+1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f'Predicted: {prediction},\nTrue: {y_test[index]}')
        plt.axis('off')

    plt.show()



if __name__ == '__main__':
    visualize_data()
