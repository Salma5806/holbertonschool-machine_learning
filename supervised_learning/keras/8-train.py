#!/usr/bin/env python3
""" Trains a model using mini-batch gradient descent and optionally
    validates it"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent and optionally
    validates it with learning rate decay."""
    callbacks = []

    if validation_data:
        val_data, val_labels = validation_data
        validation_data = (val_data, val_labels)

        if early_stopping:
            early_stopping_callback = K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                verbose=verbose,
                restore_best_weights=True
            )
            callbacks.append(early_stopping_callback)
        if learning_rate_decay:
            def lr_schedule(epoch):
                """
                Calculates the learning rate for each
                epoch using inverse time decay.
                """
                return alpha / (1 + decay_rate * epoch)

            learning_rate_decay_callback = K.callbacks.LearningRateScheduler(
                schedule=lr_schedule,
                verbose=1
            )
            callbacks.append(learning_rate_decay_callback)

        if save_best and filepath:
            model_checkpoint_callback = K.callbacks.ModelCheckpoint(
                filepath,
                monitor='val_loss',
                save_best_only=True,
                verbose=verbose
            )
            callbacks.append(model_checkpoint_callback)
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )

    return history
