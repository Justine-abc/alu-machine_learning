#!/usr/bin/env python3
"""
Convolutional Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder

    input_dims: tuple (h, w, c)
    filters: list of integers (encoder filters)
    latent_dims: tuple (h_latent, w_latent, c_latent)

    Returns: encoder, decoder, auto
    """

    # ========== Encoder ==========
    inputs = keras.Input(shape=input_dims)
    x = inputs

    # Convolution + MaxPooling layers
    for f in filters:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )(x)

    encoder = keras.Model(inputs, x)

    # ========== Decoder ==========
    latent_inputs = keras.Input(shape=latent_dims)
    x = latent_inputs

    reversed_filters = filters[::-1]

    # All decoder layers except last two
    for f in reversed_filters[:-1]:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # Second to last convolution (valid padding)
    x = keras.layers.Conv2D(
        filters=reversed_filters[-1],
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    )(x)
    x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # Final convolution (restore channels, sigmoid, no upsampling)
    outputs = keras.layers.Conv2D(
        filters=input_dims[2],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid'
    )(x)

    decoder = keras.Model(latent_inputs, outputs)

    # ========== Autoencoder ==========
    encoded = encoder(inputs)
    reconstructed = decoder(encoded)

    auto = keras.Model(inputs, reconstructed)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
