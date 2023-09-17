import tensorflow.keras as K

def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent and optionally validates it.

    Args:
        network (Keras model): The model to train.
        data (numpy.ndarray): Input data of shape (m, nx).
        labels (numpy.ndarray): One-hot encoded labels of shape (m, classes).
        batch_size (int): Size of the mini-batches.
        epochs (int): Number of passes through the entire dataset.
        validation_data (tuple): Validation data as a tuple (val_data, val_labels).
        verbose (bool): Whether to print training progress (default is True).
        shuffle (bool): Whether to shuffle the data between epochs (default is False).

    Returns:
        Keras History object: Contains training history information.
    """
    if validation_data:
        val_data, val_labels = validation_data
        validation_data = (val_data, val_labels)
    
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data  # Add validation data here
    )
    
    return history
