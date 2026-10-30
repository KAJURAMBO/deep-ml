def determinant_4x4(a):
    n = len(a)

    # Base case: 1x1 matrix
    if n == 1:
        return a[0][0]

    # Base case: 2x2 matrix
    if n == 2:
        return a[0][0] * a[1][1] - a[0][1] * a[1][0]

    det = 0

    # Expand along the first row
    for j in range(n):

        # Create minor matrix by removing first row and j-th column
        minor = []
        for i in range(1, n):
            row = []
            for k in range(n):
                if k != j:
                    row.append(a[i][k])
            minor.append(row)

        # Cofactor expansion
        det += ((-1) ** j) * a[0][j] * determinant_4x4(minor)

    return det
