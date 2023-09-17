import tensorflow.keras as K

def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent and optionally validates it with learning rate decay.

    Args:
        network (Keras model): The model to train.
        data (numpy.ndarray): Input data of shape (m, nx).
        labels (numpy.ndarray): One-hot encoded labels of shape (m, classes).
        batch_size (int): Size of the mini-batches.
        epochs (int): Number of passes through the entire dataset.
        validation_data (tuple): Validation data as a tuple (val_data, val_labels).
        early_stopping (bool): Whether to use early stopping (default is False).
        patience (int): The patience used for early stopping (default is 0).
        learning_rate_decay (bool): Whether to use learning rate decay (default is False).
        alpha (float): The initial learning rate (default is 0.1).
        decay_rate (float): The decay rate (default is 1).
        save_best (bool): Whether to save the best model based on validation loss (default is False).
        filepath (str): The file path where the best model should be saved (required if save_best is True).
        verbose (bool): Whether to print training progress (default is True).
        shuffle (bool): Whether to shuffle the data between epochs (default is False).

    Returns:
        Keras History object: Contains training history information.
    """
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
                Calculates the learning rate for each epoch using inverse time decay.
                """
                return alpha / (1 + decay_rate * epoch)

            learning_rate_decay_callback = K.callbacks.LearningRateScheduler(
                schedule=lr_schedule,
                verbose=1
            )
            callbacks.append(learning_rate_decay_callback)

        if save_best and filepath:
            # Create a ModelCheckpoint callback to save the best model
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
