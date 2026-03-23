#!/usr/bin/env python3
"""
Variational Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder"""

    # ================= Encoder =================
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Mean and log variance (MUST be linear)
    mu = keras.layers.Dense(latent_dims, activation='linear')(x)
    log_var = keras.layers.Dense(latent_dims, activation='linear')(x)

    # Reparameterization trick
    def sampling(args):
        mu, log_var = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(mu)
        )
        return mu + keras.backend.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(sampling)([mu, log_var])

    encoder = keras.Model(inputs, [z, mu, log_var])

    # ================= Decoder =================
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    outputs = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(latent_inputs, outputs)

    # ================= Auto =================
    reconstructed = decoder(z)
    auto = keras.Model(inputs, reconstructed)

    # Reconstruction loss
    reconstruction_loss = keras.losses.binary_crossentropy(
        inputs,
        reconstructed
    )
    reconstruction_loss = keras.backend.sum(
        reconstruction_loss,
        axis=-1
    )

    # KL divergence
    kl_loss = -0.5 * keras.backend.sum(
        1 + log_var - keras.backend.square(mu)
        - keras.backend.exp(log_var),
        axis=-1
    )

    auto.add_loss(
        keras.backend.mean(reconstruction_loss + kl_loss)
    )

    auto.compile(optimizer='adam')

    return encoder, decoder, auto
