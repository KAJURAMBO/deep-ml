
import numpy as np
import math

def svd_2x2(A: np.ndarray) -> tuple:
    A = np.array(A, dtype=float)

    # Step 1: Calculate A transpose multiplied by A
    B = A.T @ A

    a = B[0, 0]
    b = B[0, 1]
    d = B[1, 1]

    # Step 2: Calculate eigenvalues of A.T @ A
    delta = math.sqrt((a - d)**2 + 4 * b**2)

    eigenvalue1 = (a + d + delta) / 2
    eigenvalue2 = (a + d - delta) / 2

    # Singular values are square roots of eigenvalues
    s = np.array([
        math.sqrt(max(0, eigenvalue1)),
        math.sqrt(max(0, eigenvalue2))
    ])

    # Step 3: Calculate eigenvectors using rotation angle
    theta = 0.5 * math.atan2(2 * b, a - d)

    c = math.cos(theta)
    sn = math.sin(theta)

    # Eigenvectors are columns of Q
    Q = np.array([
        [c, -sn],
        [sn,  c]
    ])

    # Step 4: Calculate U = A @ Q @ inverse(S)
    tolerance = 1e-12

    if s[0] > tolerance:
        u1 = A @ Q[:, 0] / s[0]
    else:
        u1 = np.array([1.0, 0.0])

    if s[1] > tolerance:
        u2 = A @ Q[:, 1] / s[1]
    else:
        # Complete an orthonormal basis
        u2 = np.array([-u1[1], u1[0]])

    U = np.column_stack((u1, u2))

    # Step 5: Return V transpose to match A = U @ diag(s) @ V
    V = Q.T

    return U, s, V
