import numpy as np

def pegasos_kernel_svm(data, labels, kernel='linear',
                       lambda_val=0.01, iterations=100, sigma=1.0):

    data = np.array(data, dtype=float)
    labels = np.array(labels, dtype=float)

    n = len(data)

    alphas = np.zeros(n)
    bias = 0.0

    # Kernel calculation
    def calculate_kernel(x, z):

        if kernel == 'linear':
            return np.dot(x, z)

        elif kernel == 'rbf':
            distance = np.sum((x - z) ** 2)
            return np.exp(-distance / (2 * sigma ** 2))

        else:
            raise ValueError("Unsupported kernel")

    # Precompute kernel matrix
    K = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            K[i, j] = calculate_kernel(data[j], data[i])

    # Pegasos training
    for t in range(1, iterations + 1):

        eta = 1 / (lambda_val * t)

        for i in range(n):

            # Calculate prediction using current alphas and bias
            g = np.sum(alphas * labels * K[:, i]) + bias

            if labels[i] * g < 1:

                alphas[i] += eta * (1 - lambda_val * alphas[i])

                bias += eta * labels[i]

            else:

                alphas[i] -= eta * lambda_val * alphas[i]

    return alphas.tolist(), float(bias)