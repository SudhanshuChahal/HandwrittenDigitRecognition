from data_loader import load_data 
from model import create_model # import cnn 
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

def train():
    
     # Load data
    (x_train, y_train), (x_test, y_test) = load_data()
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    #  create model
    model = create_model() 

    # train model with augmented data
    datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1) 
    model.fit(datagen.flow(x_train, y_train, batch_size=32),
              epochs=10, 
              validation_data=(x_test, y_test))

    # Save model
    model.save("../models/model.keras") 

    # Evaluate model performance
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test accuracy: {test_acc}') 


if __name__ == '__main__':
    train()