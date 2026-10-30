import numpy as np
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    a = np.array(a)
    b = np.array(b)

    if a.ndim != 2 or b.ndim != 1 or a.shape[1] != b.shape[0]:
        return -1

    return np.dot(a, b).tolist()