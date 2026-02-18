#!/usr/bin/env python3
"""
Variational Autoencoder
"""
import tensorflow.keras as keras
import tensorflow.keras.backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Returns: encoder, decoder, auto
    """

    # ========== Encoder ==========
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    # Hidden layers
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Mean and Log Variance layers (no activation)
    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    # Reparameterization trick
    def sample(args):
        mu, log_var = args
        epsilon = K.random_normal(shape=K.shape(mu))
        return mu + K.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(sample)([mu, log_var])

    encoder = keras.Model(inputs, [z, mu, log_var])

    # ========== Decoder ==========
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(latent_inputs, outputs)

    # ========== VAE Model ==========
    reconstructed = decoder(z)
    auto = keras.Model(inputs, reconstructed)

    # ---------- Loss Function ----------
    reconstruction_loss = keras.losses.binary_crossentropy(
        inputs, reconstructed
    )
    reconstruction_loss *= input_dims

    kl_loss = -0.5 * K.sum(
        1 + log_var - K.square(mu) - K.exp(log_var),
        axis=-1
    )

    vae_loss = K.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)

    auto.compile(optimizer='adam')

    return encoder, decoder, auto
