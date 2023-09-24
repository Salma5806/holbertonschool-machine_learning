#!/usr/bin/env python3
""" LeNet-5 in Tensorflow """
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Builds a modified version of 
    the LeNet-5 architecture using tensorflow"""
    k_init = tf.contrib.layers.variance_scaling_initializer()

    l1 = tf.layers.Conv2D(filters=6,
                          kernel_size=(5, 5),
                          padding='same',
                          kernel_initializer=k_init,
                          activation=tf.nn.relu)
    l2 = tf.layers.MaxPooling2D(pool_size=2,
                                strides=2)
    l3 = tf.layers.Conv2D(filters=16,
                          kernel_size=(5, 5),
                          padding='valid',
                          kernel_initializer=k_init,
                          activation=tf.nn.relu)
    l4 = tf.layers.MaxPooling2D(pool_size=2,
                                strides=2)
    l5 = tf.layers.Dense(units=120,
                         activation=tf.nn.relu,
                         kernel_initializer=k_init)
    l6 = tf.layers.Dense(units=84,
                         activation=tf.nn.relu,
                         kernel_initializer=k_init)
    l7 = tf.layers.Dense(units=10,
                         kernel_initializer=k_init)

    l1_out = l1(x)
    l2_out = l2(l1_out)
    l3_out = l3(l2_out)
    l4_out = l4(l3_out)
    l5_out = l5(tf.layers.Flatten()(l4_out))
    l6_out = l6(l5_out)
    l7_out = l7(l6_out)
    y_pred = tf.nn.softmax(l7_out)

    prediction = tf.math.argmax(l7_out, axis=1)
    correct = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(prediction, correct)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

    loss = tf.losses.softmax_cross_entropy(y, l7_out)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.apply_gradients(optimizer.compute_gradients(loss))

    return y_pred, training_op, loss, accuracy
