#!/usr/bin/env python3
""" Inception Block """
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """ denseNet-121 architecture from Densely Connected Convolutional"""
    x = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()
    # conv1
    bn1 = K.layers.BatchNormalization(axis=3)(x)
    activation = K.layers.Activation('relu')(bn1)
    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                            padding='same',
                            kernel_initializer=init)(activation)
    max_pool = K.layers.MaxPooling2D(pool_size=(
        3, 3), strides=(2, 2), padding='same')(conv1)
    # dense block 1
    dense1, nb_filters = dense_block(max_pool, 64, growth_rate, 6)
    # transition layer 1
    trans1, nb_filters = transition_layer(dense1, nb_filters, compression)
    # dense block 2
    dense2, nb_filters = dense_block(trans1, nb_filters, growth_rate, 12)
    # transition layer 2
    trans2, nb_filters = transition_layer(dense2, nb_filters, compression)
    # dense block 3
    dense3, nb_filters = dense_block(trans2, nb_filters, growth_rate, 24)
    # transition layer 3
    trans3, nb_filters = transition_layer(dense3, nb_filters, compression)
    # dense block 4
    dense4, nb_filters = dense_block(trans3, nb_filters, growth_rate, 16)
    # avg pooling
    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         strides=(1, 1))(dense4)
    # dense layer softmax
    dense = K.layers.Dense(units=1000, activation='softmax',
                           kernel_initializer=init)(avg_pool)
    model = K.Model(inputs=x, outputs=dense)
    return model
