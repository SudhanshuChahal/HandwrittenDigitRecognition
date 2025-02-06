from tensorflow import keras
import numpy as np
import os


def predict_digit(image_path):
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'model.keras'))

    #Load trained model
    model = keras.models.load_model(model_path)

    # preprocess input image
    img = keras.preprocessing.image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')

    # convert PIL image to array and preprocess
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = 255.0 - img_array # invert colors
    img_array = img_array.astype('float32')/255.0 # normalize image
    img_array = img_array.reshape(1, 28, 28, 1) # add batch dimension

    # predcit , return the digit 
    prediction = model.predict(img_array)
    return np.argmax(prediction)
