"""
This file contains the definitions of covariance functions.
Author: Alberto Dalla Libera
"""

import numpy as np
import torch


def diagonal_covariance_ARD(pos_par=None, free_par=None):
    """
    Returns a diagonal covariance matrix.
    The diagonal elements corresponds to the square of the pos_par
    """
    return torch.diag(pos_par**2)


def diagonal_covariance(pos_par=None, free_par=None, num_par=None):
    """
    Returns a diagonal covariance matrix.
    The diagonal elements are assumed to be all equal and
    correspondent to the square of pos_par
    """
    return pos_par**2 * torch.eye(num_par, dtype=pos_par.dtype, device=pos_par.device)


def diagonal_covariance_semidef(pos_par=None, free_par=None):
    """
    Returns a diagonal covariance matrix.
    The diagonal elements are distinct and non-negative
    """
    return torch.diag(free_par**2)


def full_covariance(pos_par, free_par, num_row):
    """
    Returns a full covariance parametrized through the elements of the cholesky decomposition

    inputs:
    - free_par = elements in the lower triangular elements of the cholesky decomposition
    - pos_par = diagonal elements of the cholesky decomposition
    - num_row = number of rows
    """
    # map the parameters in a vector opportunely ordered
    parameters_vector = par2vect_chol(pos_par, free_par, num_row)
    # map the vector in the uppper triangular matrix
    U = torch.zeros(num_row, num_row, dtype=pos_par.dtype, device=pos_par.device)
    U[torch.triu(torch.ones(num_row, num_row, dtype=pos_par.dtype, device=pos_par.device)) == 1] = parameters_vector
    # get the sigma
    return torch.matmul(U.transpose(1, 0), U)


def par2vect_chol(pos_par_vect, free_par_vect, num_row):
    """
    Maps pos_par and free_par in the chol vector
    """
    vect = torch.tensor([], dtype=pos_par_vect.dtype, device=pos_par_vect.device)
    free_par_index_to = 0
    for row in range(0, num_row):
        free_par_index_from = free_par_index_to
        free_par_index_to = free_par_index_to + num_row - row - 1
        vect = torch.cat([vect, pos_par_vect[row].reshape(1), free_par_vect[free_par_index_from:free_par_index_to]])
    return vect


def get_initial_par_chol(num_row, mode="Identity"):
    """
    Returns numpy initialization of pos_par and free par for the upper triangular cholesky decomposition
    """
    num_free_par = int(num_row * (num_row - 1) / 2)
    if mode == "Identity":
        pos_par = np.ones(num_row)
        free_par = np.zeros(num_free_par)
    elif mode == "Random":
        pos_par = np.ones(num_row)
        free_par = 0.01 * np.random.randn(num_free_par)
    else:
        print("Specify an initialization mode!")
        raise RuntimeError
    return pos_par, free_par
