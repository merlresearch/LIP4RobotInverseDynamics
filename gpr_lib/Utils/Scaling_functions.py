"""
Author: Alberto Dalla Libera
This file contains the definitions of scaling functions
"""

import torch


def f_get_sign(X_active, pos_par=None, free_par=None, flg_sign_pos=True):
    """
    Returns a vecor containing ones and zeros, depending on the fact that X_active greater or lower than free_par.
    """
    if flg_sign_pos:
        return torch.prod(X_active > free_par, 1, keepdim=True, dtype=free_par.dtype)
    else:
        return torch.prod(X_active < free_par, 1, keepdim=True, dtype=free_par.dtype)


def f_get_sign_abs(X_active, pos_par=None, free_par=None, flg_sign_pos=True):
    """
    Returns a vecor containing zeros and ones, depending on the fact that X is positive or negative.
    """
    if flg_sign_pos:
        return torch.prod(torch.abs(X_active) > pos_par, 1, keepdim=True, dtype=pos_par.dtype)
    else:
        return torch.prod(torch.abs(X_active) < pos_par, 1, keepdim=True, dtype=pos_par.dtype)
