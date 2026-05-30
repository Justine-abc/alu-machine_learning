#!/usr/bin/env python3
"""
Module containing the NST class for Neural Style Transfer tasks.
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Class that performs tasks for Neural Style Transfer.
    """
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for Neural Style Transfer.

        Args:
            style_image (np.ndarray): Image used as style reference.
            content_image (np.ndarray): Image used as content reference.
            alpha (float): Weight for content cost.
            beta (float): Weight for style cost.
        """
        if (not isinstance(style_image, np.ndarray) or
                len(style_image.shape) != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels.

        Args:
            image (np.ndarray): Image to be scaled.

        Returns:
            tf.Tensor: Scaled image tensor with shape (1, h_new, w_new, 3).
        """
        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape

        if h > w:
            h_new = 512
            w_new = int(round(w * (512 / h)))
        else:
            w_new = 512
            h_new = int(round(h * (512 / w)))

        image_tensor = tf.expand_dims(image, axis=0)

        resized_image = tf.image.resize(
            image_tensor,
            size=[h_new, w_new],
            method=tf.image.ResizeMethod.BICUBIC
        )

        scaled_image = resized_image / 255.0
        scaled_image = tf.clip_by_value(scaled_image, 0.0, 1.0)

        return scaled_image

    def load_model(self):
        """
        Creates the model used for Neural Style Transfer.
        """
        vgg = tf.keras.applications.vgg19.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        outputs = []
        for name in self.style_layers:
            outputs.append(vgg.get_layer(name).output)
        outputs.append(vgg.get_layer(self.content_layer).output)

        self.model = tf.keras.models.Model(vgg.input, outputs)

    def generate_features(self):
        """
        Extracts the features used for neural style transfer.
        """
        pre_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255.0
        )
        pre_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255.0
        )

        style_outputs = self.model(pre_style)[:-1]
        self.content_feature = self.model(pre_content)[-1]

        self.style_image_features = [
            self.gram_matrix(layer) for layer in style_outputs
        ]

    @staticmethod
    def gram_matrix(input_tensor):
        """
        Calculates an unnormalized gram matrix.

        Args:
            input_tensor: A tf.Tensor of shape (1, h, w, c)

        Returns:
            A tf.Tensor of shape (1, c, c) containing the gram matrix
        """
        if not isinstance(input_tensor, (tf.Tensor, tf.Variable)) or \
           len(input_tensor.shape) != 4:
            raise TypeError("input_tensor must be a tensor of rank 4")

        channels = input_tensor.shape[-1]
        features = tf.reshape(input_tensor, [-1, channels])
        gram = tf.matmul(features, features, transpose_a=True)

        return tf.expand_dims(gram, axis=0)

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for the generated image.

        Args:
            style_outputs: a list of tf.Tensor style outputs for the
                           generated image

        Returns:
            The style cost
        """
        if not isinstance(style_outputs, list) or \
           len(style_outputs) != len(self.style_layers):
            raise TypeError(
                "style_outputs must be a list with a length of {}"
                .format(len(self.style_layers))
            )

        weight_per_layer = 1.0 / float(len(self.style_layers))
        total_style_cost = 0.0

        for i, generated_output in enumerate(style_outputs):
            G = self.gram_matrix(generated_output)
            A = self.style_image_features[i]

            _, h, w, c = generated_output.shape
            h = float(h)
            w = float(w)
            c = float(c)

            layer_square_diff = tf.reduce_sum(tf.square(G - A))
            normalization_factor = 4.0 * (h ** 2) * (w ** 2) * (c ** 2)
            layer_cost = layer_square_diff / normalization_factor

            total_style_cost += layer_cost * weight_per_layer

        return total_style_cost
