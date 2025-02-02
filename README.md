# HandwrittenDigitRecognition



## Setup instructions
1. create your python environment.
2. command : py -3.10-64 -m venv env.
          env/Scripts/activate.

3. install libraries ref. requirement.txt .


## Training commands
1. first of all load the mnist data in data_loader.py file from keras.
2. model training using CNN.


## Prediction examples
1. about loading data.
1.1  load mnist data form keras
1.2  normalize pixel values
1.3  return test and train sets

2. about model.
2.1  Conv2D → Finds edges/curves.
2.2  MaxPooling2D → Keeps only the strongest features.
2.3  Flatten → Organizes features into a list.
2.4  Dense → Makes a decision (e.g., "It’s an 8!").
2.5  Dropout → Ensures the model doesn’t rely too much on one feature.

3. about prediction.
3.1  Load trained model.
3.2  preprocess input image.
3.3  convert PIL image to array and preprocess.
3.4  predcit , return the digit 

4. about training.
4.1  Load data.
4.2  create and train data.
4.3  save model.