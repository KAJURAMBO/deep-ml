import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    A = np.array(A, dtype=float)

    if A.shape != (2, 2):
        raise ValueError("Input must be a 2x2 matrix")

    # Step 1: Calculate A transpose multiplied by A
    ATA = A.T @ A

    # Step 2: Extract elements
    a = ATA[0, 0]
    b = ATA[0, 1]
    d = ATA[1, 1]

    # Step 3: Calculate Jacobi rotation angle
    if abs(b) < 1e-12:
        V = np.eye(2)
    else:
        theta = 0.5 * np.arctan2(2 * b, a - d)

        c = np.cos(theta)
        s = np.sin(theta)

        V = np.array([
            [c, -s],
            [s,  c]
        ])

    # Step 4: Diagonalize ATA
    diagonal = V.T @ ATA @ V

    # Step 5: Calculate singular values
    S = np.sqrt(np.maximum(np.diag(diagonal), 0))

    # Sort singular values in descending order
    idx = np.argsort(S)[::-1]
    S = S[idx]
    V = V[:, idx]

    # Step 6: Calculate U
    U = np.zeros((2, 2))

    for i in range(2):
        if S[i] > 1e-12:
            U[:, i] = (A @ V[:, i]) / S[i]

    # Handle zero singular values
    if S[0] > 1e-12 and S[1] <= 1e-12:
        U[:, 1] = [-U[1, 0], U[0, 0]]

    elif S[0] <= 1e-12:
        U = np.eye(2)

    return U, S, V.T