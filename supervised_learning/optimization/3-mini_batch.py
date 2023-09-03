#!/usr/bin/env python3
""" trains a loaded neural network model using mini-batch gradient descent """

import numpy as np
import tensorflow.compat.v1 as tf


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent.

    Args:
        X_train: numpy.ndarray of shape (m, 784) containing the training data
        Y_train: one-hot numpy.ndarray of shape (m, 10) containing the training labels
        X_valid: numpy.ndarray of shape (m, 784) containing the validation data
        Y_valid: one-hot numpy.ndarray of shape (m, 10) containing the validation labels
        batch_size: number of data points in a batch
        epochs: number of times the training should pass through the whole dataset
        load_path: path from which to load the model
        save_path: path to where the model should be saved after training

    Returns:
        The path where the model was saved.
    """
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        m = X_train.shape[0]
        num_batches = m // batch_size
        if m % batch_size != 0:
            num_batches += 1

        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            X_train = X_train[permutation]
            Y_train = Y_train[permutation]
            train_cost = 0.0
            train_accuracy = 0.0

            for i in range(0, m, batch_size):
                X_batch = X_train[i:i + batch_size]
                Y_batch = Y_train[i:i + batch_size]
                feed_dict = {x: X_batch, y: Y_batch}
                _, batch_cost, batch_accuracy = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)

                train_cost += batch_cost
                train_accuracy += batch_accuracy

                if (i / batch_size) % 100 == 0:
                    step_number = i / batch_size
                    print("\tStep {}: ".format(step_number))
                    print("\t\tCost: {:.6f}".format(batch_cost))
                    print("\t\tAccuracy: {:.4f}".format(batch_accuracy))

            train_cost /= num_batches
            train_accuracy /= num_batches

            feed_dict = {x: X_valid, y: Y_valid}
            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict=feed_dict)

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {:.6f}".format(train_cost))
            print("\tTraining Accuracy: {:.4f}".format(train_accuracy))
            print("\tValidation Cost: {:.6f}".format(valid_cost))
            print("\tValidation Accuracy: {:.4f}".format(valid_accuracy))

        saver = tf.train.Saver()
        saver.save(sess, save_path)

    return save_path
