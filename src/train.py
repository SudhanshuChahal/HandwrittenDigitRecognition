from data_loader import load_data 
from model import create_model # import cnn 
import os  

def train():
    
    (x_train, y_train), (x_test, y_test) = load_data()
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape) # Load data

    model = create_model() 
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test)) #  create and train model

    model.save("../models/model.keras") # Save model



if __name__ == '__main__':
    train()