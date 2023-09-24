#!/usr/bin/env python3
"""
Builds a modified version of the LeNet-5 architecture using tensorflow
"""
import tensorflow.keras as K


def lenet5(X):
    """
    a function that builds a modified LeNet-5 using tensorflow"""
    init = K.initializers.he_normal(seed=None)
    conv1 = K.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                            activation="relu", kernel_initializer=init)(X)
    pool1 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)
    conv2 = K.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                            activation="relu", kernel_initializer=init)(pool1)
    pool2 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)
    flat = K.layers.Flatten()(pool2)
    fc1 = K.layers.Dense(units=120, activation="relu",
                         kernel_initializer=init)(flat)
    fc2 = K.layers.Dense(units=84, activation="relu",
                         kernel_initializer=init)(fc1)
    y_pred = K.layers.Dense(units=10, activation="softmax",
                            kernel_initializer=init)(fc2)
    model = K.models.Model(inputs=X, outputs=y_pred)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
