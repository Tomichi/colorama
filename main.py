from tensorflow import keras
from tensorflow.keras import layers
import random
from pandas import read_csv
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def build_callbacks():
    checkpointer = ModelCheckpoint(filepath='color.h5', verbose=1, save_best_only=True, save_weights_only=True)
    callbacks = [checkpointer]
    return callbacks


def create_model():
    model = keras.Sequential()
    model.add(layers.Dense(20, input_dim=3, activation="relu"))  # input 3 neurons | hidden layers 20
    model.add(layers.Dense(18, activation="softmax"))  # output 18 layers
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.summary() # sumary describing NN
    return model


if __name__ == '__main__':
    colors_rgb = read_csv('rgb_color.csv', header=None, sep=',')
    dataset = colors_rgb.values
    X = dataset[:, 0:3].astype(float)  # indexing in 2D numpy array select matrix Nx3
    Y = dataset[:, 3]  # indexing in 2D numpy array Select vector Nx1

    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    Y_one_hot_encoding = to_categorical(encoded_Y)

    random.seed(42)
    model = create_model()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_one_hot_encoding,
        test_size=0.2, random_state=random.randint(0, 1000), shuffle=True
    )

    # it neccessary to create filepath wieghts
    model.fit(X_train, Y_train, epochs=200, batch_size=1, verbose=1, validation_data=(X_test, Y_test),
              callbacks=[
                  ModelCheckpoint(filepath='weights/model.{val_loss:.3f}-{val_accuracy:.3f}.h5', save_best_only=True, save_weights_only=False)
              ],

    )

    # model.load('weights/my-weights.h5')

    predictions = model.predict(X_test)
    encoded_output = np.argmax(predictions, axis=1)
    encoded_output_name = encoder.inverse_transform(encoded_output)
    y_test_max = np.argmax(Y_test, axis=1)
    encoded_output_reference = encoder.inverse_transform(y_test_max)
    # calculate F1 score...
