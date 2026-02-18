#!/usr/bin/env python3
"""
Sparse Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder

    input_dims: integer, dimensions of the input
    hidden_layers: list of integers, number of nodes in each hidden layer
                   of the encoder
    latent_dims: integer, dimensions of the latent space
    lambtha: L1 regularization parameter for the latent layer

    Returns: encoder, decoder, auto
    """

    # ========== Encoder ==========
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    # Hidden layers (encoder)
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Latent layer with L1 activity regularization
    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)

    encoder = keras.Model(inputs=inputs, outputs=latent)

    # ========== Decoder ==========
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    # Reverse hidden layers for decoder
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Output layer
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    # ========== Autoencoder ==========
    encoded = encoder(inputs)
    reconstructed = decoder(encoded)

    auto = keras.Model(inputs=inputs, outputs=reconstructed)

    # Compile
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
