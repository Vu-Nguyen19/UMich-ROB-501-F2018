#%%
import torch
# %%
def get_A(limits: tuple, shape: tuple) -> torch.Tensor:
    """
    Create a matrix A of the given shape with the following properties: All numbers are random integers
    Parameters
    ----------
    shape : tuple
        Shape of the matrix to be created.

    Returns
    -------
    np.ndarray
        Matrix A of the given shape with the specified properties.
    """
    return torch.randint(limits[0], limits[1], shape, dtype=torch.float32)


def get_eigen(X: torch.Tensor) -> tuple:
    """
    Calculate the eigenvalues and eigenvectors of a matrix A.

    Parameters
    ----------
    A : np.ndarray
        The input matrix.

    Returns
    -------
    tuple
        A tuple containing the eigenvalues and eigenvectors of the matrix A.
    """
    return torch.linalg.eigh(X)




# %%
# i. form 10 matrix A shape n x m, and a square matrix M = A^T * A
n = 5
m = 4
A = get_A((0, 10), (n, m))
M = torch.matmul(A.T, A)
# ii. calculate the eigenvalues and eigenvectors of M
eigenvalues, eigenvectors = get_eigen(M)
# iii. form the inner product of the eigenvectors
inner_product = torch.sum(eigenvectors * eigenvectors.flip(dims=[0]), dim=1)
print("Inner product of eigenvectors:\n", inner_product)
# Note that the eigenvectors are orthogonal, so the inner product should be close to 0 for different eigenvectors
# iv. sum the eigenvalues and compare it to the trace of M
trace_M = torch.trace(M)
sum_eigenvalues = torch.sum(eigenvalues)
print("Trace of M:\n", trace_M)
print("Sum of eigenvalues:\n", sum_eigenvalues)
# v. summarize the results
"""
The eigenvalues and eigenvectors of the matrix M were calculated. 
The inner product of the eigenvectors was computed, 
which should be close to 0 for different eigenvectors due to their orthogonality. 
The trace of M was compared to the sum of the eigenvalues, which should be equal.
"""




# %%
