#!/usr/bin/env python3
""" LeNet-5 in Tensorflow """
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy"""
    x = tf.cast(tf.shape(y)[0], tf.float32)
    y_pred = tf.argmax(y_pred, axis=1)
    y = tf.argmax(y, axis=1)
    true_false_array = tf.equal(y_pred, y)
    num_of_true = tf.cast(tf.reduce_sum(
        tf.cast(true_false_array, tf.int32)), tf.float32)
    accuracy = num_of_true / x
    mean_accuracy = tf.reduce_mean(accuracy, name="Mean")
    return mean_accuracy


def lenet5(x, y):
    """Builds a modified version of
    the LeNet-5 architecture using tensorflow"""
    init = tf.keras.initializers.VarianceScaling(scale=2.0)

    conv1 = tf.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                             kernel_initializer=init, activation='relu')(x)
    pool1 = tf.layers.MaxPooling2D(2, 2)(conv1)

    conv2 = tf.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                             kernel_initializer=init, activation='relu')(pool1)
    pool2 = tf.layers.MaxPooling2D(2, 2)(conv2)

    fc = tf.layers.Flatten()(pool2)
    fc1 = tf.layers.Dense(120, activation='relu',
                          kernel_initializer=init)(fc)
    fc2 = tf.layers.Dense(84, activation='relu',
                          kernel_initializer=init)(fc1)
    fc3 = tf.layers.Dense(10, kernel_initializer=init)(fc2)
    y_pred = tf.nn.softmax(fc3)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=fc3)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    accuracy = calculate_accuracy(y, y_pred)
    return y_pred, train_op, loss, accuracy
