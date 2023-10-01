#!/usr/bin/env python3
""" Inception Block """
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """inception network"""
    X = K.Input(shape=(224, 224, 3))
    x1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                         strides=(2, 2),
                         activation='relu',
                         padding='same')(X)
    x2 = K.layers.MaxPool2D(pool_size=(
        3, 3), strides=(2, 2), padding='same')(x1)
    x3 = K.layers.Conv2D(filters=64, kernel_size=(1, 1),
                         activation='relu')(x2)
    x4 = K.layers.Conv2D(filters=192, kernel_size=(
        3, 3), activation='relu', padding='same')(x3)
    x5 = K.layers.MaxPool2D(pool_size=(
        3, 3), strides=(2, 2), padding='same')(x4)
    x6 = inception_block(x5, [64, 96, 128, 16, 32, 32])
    x7 = inception_block(x6, [128, 128, 192, 32, 96, 64])
    x8 = K.layers.MaxPool2D((3, 3), 2, padding='same')(x7)
    x9 = inception_block(x8, [192, 96, 208, 16, 48, 64])
    x10 = inception_block(x9, [160, 112, 224, 24, 64, 64])
    x11 = inception_block(x10, [128, 128, 256, 24, 64, 64])
    x12 = inception_block(x11, [112, 144, 288, 32, 64, 64])
    x13 = inception_block(x12, [256, 160, 320, 32, 128, 128])
    union = K.layers.MaxPooling2D((3, 3), 2, padding='same')(x13)
    x14 = inception_block(union, [256, 160, 320, 32, 128, 128])
    x15 = inception_block(x14, [384, 192, 384, 48, 128, 128])
    final_union = K.layers.AveragePooling2D(
        pool_size=(7, 7), strides=(1, 1))(x15)
    y = K.layers.Dropout(0.4)(final_union)
    Y = K.layers.Dense(units=1000, activation="softmax")(y)
    model_1 = K.Model(inputs=X, outputs=Y)
    return model_1
