#!/usr/bin/env python3
"""
Defines a function that evaluates output of
neural network classifier
"""


import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

def evaluate(X, Y, save_path):
    """Evaluates the output of a neural network"""
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(save_path + '.meta')
        loader.restore(sess, save_path)

        x = loaded_graph.get_tensor_by_name('x:0')
        y = loaded_graph.get_tensor_by_name('y:0')
        prediction = loaded_graph.get_tensor_by_name('layer/Tanh:0')
        accuracy = loaded_graph.get_tensor_by_name('Mean_1:0')
        loss = loaded_graph.get_tensor_by_name('Mean:0')

        feed_dict = {x: X, y: Y}
        pred, acc, cost = sess.run([prediction, accuracy, loss], feed_dict=feed_dict)

    return pred, acc, cost
