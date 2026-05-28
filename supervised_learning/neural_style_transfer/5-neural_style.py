@staticmethod
    def gram_matrix(input_tensor):
        """
        Calculates a gram matrix for an unnormalized tensor.
        
        Args:
            input_tensor: A tf.Tensor of shape (1, h, w, c)
            
        Returns:
            A tf.Tensor of shape (1, c, c) containing the unnormalized gram matrix
        """
        if not isinstance(input_tensor, (tf.Tensor, tf.Variable)) or \
           len(input_tensor.shape) != 4:
            raise TypeError("input_tensor must be a tensor of rank 4")

        # Reshape to (H*W, C)
        channels = input_tensor.shape[-1]
        features = tf.reshape(input_tensor, [-1, channels])
        
        # Compute unnormalized Gram Matrix: G = F^T * F
        gram = tf.matmul(features, features, transpose_a=True)
        
        # Expand dims to match (1, c, c) if required by your earlier tasks
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

        # Weight each layer evenly
        weight_per_layer = 1.0 / float(len(self.style_layers))
        total_style_cost = 0.0

        for i, generated_output in enumerate(style_outputs):
            # 1. Get unnormalized Gram Matrices
            # (Ensure your self.style_image_features contains UNNORMALIZED gram matrices too)
            G = self.gram_matrix(generated_output)
            A = self.gram_matrix(self.style_image_features[i])
            
            # 2. Extract dimensions from the layer output (H, W, C)
            _, h, w, c = generated_output.shape
            h = float(h)
            w = float(w)
            c = float(c)
            
            # 3. Calculate layer squared differences sum
            # Note: We sum the squared errors, we do NOT use reduce_mean here
            layer_square_diff = tf.reduce_sum(tf.square(G - A))
            
            # 4. Apply standard style normalization factor: 4 * (H^2 * W^2 * C^2)
            normalization_factor = 4.0 * (h ** 2) * (w ** 2) * (c ** 2)
            layer_cost = layer_square_diff / normalization_factor
            
            # 5. Accumulate weighted layer cost
            total_style_cost += layer_cost * weight_per_layer

        return total_style_cost
