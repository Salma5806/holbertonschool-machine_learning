#!/usr/bin/env python3
""" Train a loaded neural network model using mini-batch gradient descent """


import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid,
                     Y_valid, batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ train a loaded neural network model
    using mini-batch gradient descent """
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(load_path + ".meta")
        loader.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]
        for i in range(epochs + 1):
            cost_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            accuracy_train = sess.run(accuracy,
                                      feed_dict={x: X_train, y: Y_train})
            cost_val = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            accuracy_val = sess.run(accuracy,
                                    feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tTraining Accuracy: {}".format(accuracy_train))
            print("\tValidation Cost: {}".format(cost_val))
            print("\tValidation Accuracy: {}".format(accuracy_val))

            if i == epochs:
                return loader.save(sess, save_path)

            X_shuff, Y_shuff = shuffle_data(X_train, Y_train)

            length_of_dataset = X_shuff.shape[0]
            steps = []
            for j in range(0, length_of_dataset, batch_size):
                steps.append((j, j + batch_size))
                """
                for k in range(1, len(steps)):
                start, end = steps[k]
                x_batch = X_shuff[start:end]
                y_batch = Y_shuff[start:end]
                feed_dict = {x: x_batch, y: y_batch}
                sess.run(train_op, feed_dict=feed_dict)
                if k % 100 == 0:
                    loss_mini_batch = sess.run(loss, feed_dict=feed_dict)
                    accuracy_mini_batch = sess.run(accuracy,
                                                   feed_dict=feed_dict)
                    print("\tStep {}:".format(k))
                    print("\t\tCost: {}".format(loss_mini_batch))
                    print("\t\tAccuracy: {}".format(accuracy_mini_batch))
                    """
            for k, (start, end) in enumerate(steps, start=1):
                x_batch = X_shuff[start:end]
                y_batch = Y_shuff[start:end]
                feed_dict = {x: x_batch, y: y_batch}
                sess.run(train_op, feed_dict=feed_dict)
                if k % 100 == 0:
                    loss_mini_batch = sess.run(loss, feed_dict=feed_dict)
                    accuracy_mini_batch = sess.run(accuracy,
                                                   feed_dict=feed_dict)
                    print("\tStep {}:".format(k))
                    print("\t\tCost: {}".format(loss_mini_batch))
                    print("\t\tAccuracy: {}".format(accuracy_mini_batch))
