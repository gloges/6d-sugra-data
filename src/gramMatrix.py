import numpy as np


def signature(matrix):
    """Returns the signature, (n(+), n(-), n(0)), of `matrix`."""

    shape = np.shape(matrix)
    if len(shape) != 2 or shape[0] != shape[1]:
        err = 'Unexpected shape: gram must be a square matrix.'
        raise Exception(err)

    if len(matrix) == 0:
        # Empty matrix (0x0)
        return 0, 0, 0
    
    # Get eigenvalues, rank and nullity
    eigs = np.linalg.eigvalsh(np.array(matrix, dtype=float))
    rank = np.linalg.matrix_rank(np.array(matrix, dtype=float))
    nullity = int(len(matrix) - rank)

    # Correct fp errors by setting (near-)zero eigenvalues
    # to exactly zero based on the known nullity
    order = np.argsort(abs(eigs))
    eigs = eigs[order]
    eigs[:nullity] = 0

    # Count number of positive and negative eigenvalues
    n_pos = np.count_nonzero(eigs > 0)
    n_neg = np.count_nonzero(eigs < 0)

    return n_pos, n_neg, nullity

def integer_kernel_basis(matrix):
    """Returns an integer basis for the kernel of `matrix` using Gaussian elimination with integer arithmetic."""

    shape = np.shape(matrix)
    if len(shape) != 2:
        err = 'Unexpected shape: must be a 2D matrix.'
        raise Exception(err)

    if min(shape) == 0:
        return []

    # Number of rows and columns
    num_rows, num_cols = shape

    # Augment by an identity matrix (use dtype=object for python ints and arbitrary precision)
    matrix_augmented = np.block([matrix, np.identity(num_rows, dtype=object)])

    # Perform Gaussian elimination
    pivot_row, pivot_col = 0, 0
    while pivot_row < num_rows and pivot_col < num_cols:

        if matrix_augmented[pivot_row, pivot_col] == 0:
            # If pivot is zero, look for a lower row to swap with
            swap_row = pivot_row
            while swap_row < num_rows and matrix_augmented[swap_row, pivot_col] == 0:
                swap_row += 1

            if swap_row < num_rows:
                # Perform the swap of rows
                matrix_augmented[[pivot_row, swap_row]] = matrix_augmented[[swap_row, pivot_row]]

        if matrix_augmented[pivot_row, pivot_col] != 0:
            pivot_sign = np.sign(matrix_augmented[pivot_row, pivot_col])

            # Use non-zero pivot for elimination for each lower row
            for jj in range(pivot_row+1, num_rows):
                gcd = GCD(matrix_augmented[pivot_row, pivot_col], matrix_augmented[jj, pivot_col])

                # Replace with linear combination with a zero below the pivot
                matrix_augmented[jj] = matrix_augmented[pivot_row, pivot_col] * matrix_augmented[jj] \
                                        - matrix_augmented[jj, pivot_col] * matrix_augmented[pivot_row]
                matrix_augmented[jj] = pivot_sign * matrix_augmented[jj] // gcd

                # Reduce row by gcd to keep matrix elements small
                for value in matrix_augmented[jj]:
                    gcd = GCD(gcd, value)
                matrix_augmented[jj] = matrix_augmented[jj] // gcd

            pivot_row += 1
        pivot_col += 1

    # Nullity is the number of rows of matrix (ignoring augmentation) which are all zeros
    nullity = np.count_nonzero(np.max(np.abs(matrix_augmented[:, :num_cols]), axis=1) == 0)
    
    if nullity == 0:
        # gram has trivial kernel
        kernel_basis = []
    else:
        # An integer basis for the kernel is given by the rows in the augmentation
        # corresponding to the zero rows
        kernel_basis = matrix_augmented[(-nullity):, num_cols:]

    return kernel_basis

def GCD(x: int, y: int):
    """Computes the GCD of x and y using the Euclidean algorithm."""
    if y == 0: return abs(x)
    else:      return GCD(y, x % y)
