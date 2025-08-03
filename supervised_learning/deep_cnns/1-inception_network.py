#!/usr/bin/env python3
""" Inception Block """

from tensorflow import keras as K

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Builds the full Inception network"""
    X = K.Input(shape=(224, 224, 3))

    x1 = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                         padding='same', activation='relu')(X)
    x2 = K.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x1)

    x3 = K.layers.Conv2D(64, (1, 1), activation='relu')(x2)
    x4 = K.layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x3)
    x5 = K.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x4)

    x6 = inception_block(x5, [64, 96, 128, 16, 32, 32])
    x7 = inception_block(x6, [128, 128, 192, 32, 96, 64])
    x8 = K.layers.MaxPool2D((3, 3), strides=2, padding='same')(x7)

    x9 = inception_block(x8, [192, 96, 208, 16, 48, 64])
    x10 = inception_block(x9, [160, 112, 224, 24, 64, 64])
    x11 = inception_block(x10, [128, 128, 256, 24, 64, 64])
    x12 = inception_block(x11, [112, 144, 288, 32, 64, 64])
    x13 = inception_block(x12, [256, 160, 320, 32, 128, 128])
    x14 = K.layers.MaxPool2D((3, 3), strides=2, padding='same')(x13)

    x15 = inception_block(x14, [256, 160, 320, 32, 128, 128])
    x16 = inception_block(x15, [384, 192, 384, 48, 128, 128])

    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         strides=(1, 1))(x16)
    dropout = K.layers.Dropout(0.4)(avg_pool)
    Y = K.layers.Dense(1000, activation='softmax')(dropout)

    model = K.Model(inputs=X, outputs=Y)
    return model
