#!/usr/bin/env python3
"""Task 3"""

import tensorflow as tf
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Create a variational autoencoder model for dimensionality reduction
    and feature learning.
    """
    # Input layer that takes data with 'input_dims' features
    inputs = keras.layers.Input(shape=(input_dims))

    # Encoder network
    x = inputs
    for n in hidden_layers:
        x = keras.layers.Dense(n, activation='relu')(x)

    # Latent layer, the bottleneck of the autoencoder
    mean_log = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    # Reparameterization trick
    def sampling(args):
        mean_log, log_var = args
        batch = tf.shape(mean_log)[0]
        epsilon = tf.keras.backend.random_normal(
          shape=(batch, latent_dims), mean=0.0, stddev=1.0)
        return mean_log + tf.exp(0.5 * log_var) * epsilon
    bridge = tf.keras.layers.Lambda(
        sampling)([mean_log, log_var])

    # Create an encoder model that maps the input to the latent space
    encoder = keras.Model(inputs, [bridge, mean_log, log_var])

    # Decoder network
    latent_inputs = keras.layers.Input(shape=(latent_dims,))
    x = latent_inputs
    for n in reversed(hidden_layers):
        x = keras.layers.Dense(n, activation='relu')(x)

    # Output layer that reconstructs the input data
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    # Create a decoder that maps the latent space to the output
    decoder = keras.Model(latent_inputs, outputs)

    # Combine the encoder and decoder to create the full autoencoder model
    outputs = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs, outputs)

    # Define VAE loss
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= input_dims
    kl_loss = 1 + log_var - tf.square(mean_log) - tf.exp(log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

    # Compile autoencoder model with Adam optimization & VAE loss
    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    # Return the encoder, decoder, and full autoencoder models
    return encoder, decoder, auto
