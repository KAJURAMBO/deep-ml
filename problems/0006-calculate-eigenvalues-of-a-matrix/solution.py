import math

def calculate_eigenvalues(matrix):
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]

    # Calculate trace
    trace = a + d

    # Calculate determinant
    det = (a * d) - (b * c)

    # Calculate discriminant
    discriminant = trace**2 - (4 * det)

    if discriminant < 0:
        return -1

    # Calculate eigenvalues
    lambda1 = (trace + math.sqrt(discriminant)) / 2
    lambda2 = (trace - math.sqrt(discriminant)) / 2

    return sorted([lambda1, lambda2], reverse=True)


