from keras.models import Sequential
from keras.layers import (Conv2D, Dense, Flatten, MaxPooling2D)

def CNN(input_shape):
    model = Sequential()
    model.add(Conv2D(input_shape=input_shape[1:],filters=5,kernel_size=(3,1),padding="same", activation="relu"))
    model.add(Conv2D(filters=5,kernel_size=(3,1),padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,1),strides=(2,1)))
    model.add(Conv2D(filters=5, kernel_size=(3,1), padding="same", activation="relu"))
    model.add(Conv2D(filters=5, kernel_size=(3,1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,1),strides=(2,1)))
    model.add(Conv2D(filters=10, kernel_size=(3,1), padding="same", activation="relu"))
    model.add(Conv2D(filters=10, kernel_size=(3,1), padding="same", activation="relu"))
    model.add(Conv2D(filters=10, kernel_size=(3,1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,1),strides=(2,1)))
    model.add(Flatten(input_shape=input_shape[1:]))
    model.add(Dense(1024,activation='relu', name='fc1'))
    model.add(Dense(512,activation='relu', name='fc2'))
    model.add(Dense(1024,activation='relu', name='fc3'))
    model.add(Dense(2,activation='softmax', name='fc4'))
    return model