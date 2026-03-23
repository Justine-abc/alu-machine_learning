#!/usr/bin/env python3
"""
Vanilla Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder

    input_dims: integer, dimensions of the input
    hidden_layers: list of integers, number of nodes in each hidden layer
                   of the encoder
    latent_dims: integer, dimensions of the latent space

    Returns: encoder, decoder, auto
    """

    # ========== Encoder ==========
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    # Hidden layers (encoder)
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Latent space
    latent = keras.layers.Dense(latent_dims, activation='relu')(x)

    encoder = keras.Model(inputs=inputs, outputs=latent)

    # ========== Decoder ==========
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    # Reverse hidden layers for decoder
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Output layer (sigmoid)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    # ========== Autoencoder ==========
    auto_input = inputs
    encoded = encoder(auto_input)
    reconstructed = decoder(encoded)

    auto = keras.Model(inputs=auto_input, outputs=reconstructed)

    # Compile
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
