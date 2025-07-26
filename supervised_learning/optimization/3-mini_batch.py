#!/usr/bin/env python3
"""
Using mini-batch gradient descent algorithm
"""

import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data
tf.disable_eager_execution()


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Trains a loaded neural network using mini-batch gradient descent"""
    saver = tf.train.import_meta_graph(load_path + '.meta')

    with tf.Session() as sess:
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        # Print initial metrics (After 0 epochs)
        train_cost, train_accuracy = sess.run([loss, accuracy],
                                              feed_dict={x: X_train, y: Y_train})
        valid_cost, valid_accuracy = sess.run([loss, accuracy],
                                              feed_dict={x: X_valid, y: Y_valid})
        print("After 0 epochs:")
        print(f"\tTraining Cost: {train_cost}")
        print(f"\tTraining Accuracy: {train_accuracy}")
        print(f"\tValidation Cost: {valid_cost}")
        print(f"\tValidation Accuracy: {valid_accuracy}")

        for epoch in range(epochs):
            X_train, Y_train = shuffle_data(X_train, Y_train)
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                Y_batch = Y_train[i:i + batch_size]
                _, step_cost, step_accuracy = sess.run(
                    [train_op, loss, accuracy],
                    feed_dict={x: X_batch, y: Y_batch}
                )
                if i % 100 == 0 and i != 0:
                    print(f"\tStep {i}:")
                    print(f"\t\tCost: {step_cost}")
                    print(f"\t\tAccuracy: {step_accuracy}")

            # After each epoch, print metrics
            train_cost, train_accuracy = sess.run([loss, accuracy],
                                                  feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run([loss, accuracy],
                                                  feed_dict={x: X_valid, y: Y_valid})

            print(f"After {epoch + 1} epochs:")
            print(f"\tFinal Training Cost: {train_cost}, "
                  f"Accuracy: {train_accuracy}")
            print(f"\tFinal Validation Cost: {valid_cost}, "
                  f"Accuracy: {valid_accuracy}")

        return saver.save(sess, save_path)
