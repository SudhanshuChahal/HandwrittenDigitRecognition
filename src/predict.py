from tensorflow import keras
import numpy as np


def predict_digit(image_path='../image/digit3.webp'):
    #Load trained model
    model = keras.models.load_model('../models/model.keras')

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


if __name__ == '__main__':
    pr =predict_digit()
    print("predicted digit is: ",pr)