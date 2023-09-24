#!/usr/bin/env python3
""" LeNet-5 in Tensorflow """
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Builds a modified version of 
    the LeNet-5 architecture using tensorflow"""
    init = tf.contrib.layers.variance_scaling_initializer()
    activation = tf.nn.relu
    conv1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5), padding="same",
                             activation=activation, kernel_initializer=init)(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)


    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding="valid",
                             activation=activation, kernel_initializer=init)(
        pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    flatten = tf.layers.Flatten()(pool2)
    fc1 = tf.layers.Dense(units=120, activation=activation,
                          kernel_initializer=init)(flatten)
    fc2 = tf.layers.Dense(units=84, activation=activation,
                          kernel_initializer=init)(fc1)
    fc3 = tf.layers.Dense(units=10, activation=None,
                          kernel_initializer=init)(fc2)

    y_pred = tf.nn.softmax(fc3)
    loss = tf.losses.softmax_cross_entropy(y, fc3)
    accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(fc3, 1))
    mean = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    train = tf.train.AdamOptimizer().minimize(loss)

    return y_pred, train, loss, mean
