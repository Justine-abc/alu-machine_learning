def optimize(self, iterations=100):
    """
    Optimizes the black-box function.

    iterations: max number of iterations

    Returns:
    X_opt: numpy.ndarray of shape (1,)
    Y_opt: numpy.ndarray of shape (1,)
    """
    for _ in range(iterations):
        X_next, _ = self.acquisition()
        # Stop if X_next has already been sampled
        if np.any(np.isclose(X_next, self.gp.X, atol=1e-8)):
            break

        Y_next = self.f(X_next)
        self.gp.update(X_next, Y_next)

    if self.minimize:
        idx = np.argmin(self.gp.Y)
    else:
        idx = np.argmax(self.gp.Y)

    X_opt = self.gp.X[idx].reshape(1,)
    Y_opt = self.gp.Y[idx].reshape(1,)
    return X_opt, Y_opt
