from data_loader import load_data   

def train():

    (x_train, y_train), (x_test, y_test) = load_data()
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)


if __name__ == '__main__':
    train()