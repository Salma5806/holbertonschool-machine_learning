#!/usr/bin/env python3
"""
Builds a modified version of the LeNet-5 architecture using tensorflow
"""
import tensorflow.keras as K


def lenet5(X):
    """
    a function that builds a modified LeNet-5 using tensorflow"""
    model = K.Sequential()
    model.add(K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding="same", activation="relu", kernel_initializer='he_normal', input_shape=(28, 28, 1)))
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding="valid", activation="relu", kernel_initializer='he_normal'))
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(units=120, activation="relu", kernel_initializer='he_normal'))
    model.add(K.layers.Dense(units=84, activation="relu", kernel_initializer='he_normal'))
    model.add(K.layers.Dense(units=10, activation="softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model