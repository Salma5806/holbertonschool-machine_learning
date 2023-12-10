#!/usr/bin/env python3
"""Task 1"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Create a sparse autoencoder model for dimensionality reduction
    and feature learning.

    Parameters:
    - input_dims (int): The number of features or dimensions
      in the model input.
    - hidden_layers (list): A list containing the number of
      nodes for each hidden layer
      in the encoder, respectively. The hidden layers should be
      reversed for the decoder.
    - latent_dims (int): The dimension of the latent space representation.
    - lambtha (float): The regularization parameter used for L1
      regularization on the encoded output.

    Returns:
    - encoder (keras.Model): The encoder model that maps input data to
      the latent space.
    - decoder (keras.Model): The decoder model that maps data from the
      latent space to
      the output space, attempting to reconstruct the input data.
    - auto (keras.Model): The full sparse autoencoder model that combines
      the encoder and decoder.

    This function defines & returns a sparse autoencoder model, which consists
    of an encoder and a decoder. The encoder takes the input data and maps
    it to a lower-dimensional latent space, while the decoder maps data from
    the latent space back to the original input space, aiming to reconstruct
    the input data. The encoder and decoder models can be used independently,
    and the full autoencoder model can be used for training and inference.

    The sparse autoencoder model is compiled using Adam optimization and binary
    cross-entropy loss. All layers in the encoder and decoder use the ReLU
    activation function, except for the last layer in the decoder, which uses
    the sigmoid activation function for output reconstruction. The last layer
    of the encoder also has an L1 activity regularizer associated with it,
    encouraging sparsity in the encoded representations. Sparse representations
    are efficient, interpretable, and offer better generalization and feature
    selection compared to dense representations.
    """
    # Input layer that takes data with 'input_dims' features
    input_layer = keras.layers.Input(shape=(input_dims,))

    # Encoder network
    x = input_layer
    for n in hidden_layers:
        # Create a Dense layer with 'n' neurons and ReLU activation
        x = keras.layers.Dense(n, activation='relu')(x)
    # Latent layer with L1 regularization
    encodeD = keras.layers.Dense(latent_dims, activation='relu',
                                 activity_regularizer=keras.regularizers.l1(
                                   lambtha))(x)

    # Create an encoder model that maps the input to the latent space
    encoder = keras.Model(input_layer, encodeD)

    # Decoder network
    decoder_input = keras.layers.Input(shape=(latent_dims,))
    x = decoder_input
    for n in reversed(hidden_layers):
        # Create a Dense layer with 'n' neurons and ReLU activation
        x = keras.layers.Dense(n, activation='relu')(x)

    # Output layer that reconstructs the input data
    decoded_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    # Create a decoder that maps the latent space to the output
    decoder = keras.Model(decoder_input, decoded_output)

    # Combine the encoder and decoder to create the full autoencoder model
    autoencoder_output = decoder(encoder(input_layer))
    auto = keras.Model(input_layer, autoencoder_output)

    # Compile autoencoder model with Adam optimization & Binary Crossentropy
    auto.compile(optimizer='Adam', loss='binary_crossentropy')

    # Return the encoder, decoder, and the sparse autoencoder models
    return encoder, decoder, auto
