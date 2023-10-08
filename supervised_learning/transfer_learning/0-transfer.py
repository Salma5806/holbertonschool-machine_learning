#!/usr/bin/env python3

import tensorflow.keras as K
import tensorflow as tf
import numpy as np

# Define a callback for TensorBoard
tensorboard_callback = K.callbacks.TensorBoard(log_dir="./logs")


def preprocess_data(X, Y):
    """pre-processes the data from cifar10"""
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


def scheduler(epoch, lr):
    """piecewise learning rate scheduler"""
    lr = 0.0001
    for i in range(10, 100, 10):
        if epoch == i:
            lr = lr / 10
    return lr


if __name__ == "__main__":
    data = np.load('cifar10_data.npz')
    X = data['train_images']
    Y = data['train_labels']
    X_t = data['test_images']
    Y_t = data['test_labels']

    # Define the input shape to match ResNet-50
    old_input = K.Input((32, 32, 3))  # Use the original input size

    # Resize images to (224, 224, 3) using a Lambda layer
    resized_input = K.layers.Lambda(
        lambda x: tf.image.resize(x, (224, 224)))(old_input)
    data_aug = K.Sequential([
        K.layers.experimental.preprocessing.RandomFlip(
            "horizontal_and_vertical"),
        K.layers.experimental.preprocessing.RandomRotation(0.1),
        K.layers.experimental.preprocessing.RandomZoom(0.1)])
    data_aug = data_aug(resized_input)

    base_model = K.applications.ResNet50(
        include_top=False, weights='imagenet', input_tensor=data_aug)  # Use input_tensor

    # Freeze the hidden layers of the base_model
    for layer in base_model.layers:
        layer.trainable = False

    callbacks = []

    callbacks.append(K.callbacks.LearningRateScheduler(scheduler, verbose=1))
    callbacks.append(K.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, min_delta=0.0005))

    X_p, Y_p = preprocess_data(X, Y)
    X_tp, Y_tp = preprocess_data(X_t, Y_t)  # Preprocess test data

    x2 = base_model.output
    x3 = K.layers.Flatten()(x2)
    x4 = K.layers.BatchNormalization()(x3)
    l2 = K.regularizers.l2(0.001)  # Define l2 regularization
    x5 = K.layers.Dense(128, activation='relu', kernel_regularizer=l2)(x4)
    x6 = K.layers.Dropout(0.2)(x5)
    x7 = K.layers.BatchNormalization()(x6)
    x8 = K.layers.Dense(10, activation='softmax', kernel_regularizer=l2)(x7)
    model = K.Model(inputs=old_input, outputs=x8)  # Specify inputs and outputs

    # Define the optimizer with the custom learning rate
    op = K.optimizers.Adam(learning_rate=0.0001)  # Lower initial learning rate

    model.compile(optimizer=op, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(
        X_p, Y_p, validation_data=(X_tp, Y_tp), batch_size=64, epochs=100,
        verbose=1, shuffle=True, callbacks=[callbacks, tensorboard_callback]
    )

    # Save the model only if it meets the validation accuracy criteria
    if max(model.history.history['val_accuracy']) >= 0.87:
        model.save('cifar10.h5')
        print("Model saved.")
    else:
        print("Validation accuracy did not meet the criteria (87% or higher). Model not saved.")
        # save the weights to train more if the validation accuracy is not enough
        model.save_weights('cifar10_needmore_training_last.h5')