"""
Author: Alberto Dalla Libera
This file contains the definitions of Gaussian likelihoods.
"""

import torch


class Marginal_log_likelihood(torch.nn.modules.loss._Loss):
    """
    Computes the negative marginal log likelihood under gaussian assumption
    """

    def forward(self, output_GP_prior, Y):
        """
        returns the negative marginal log likelihood:
        0.5*( (Y-m_X)^T*K_X_inv*(Y-m_x) + log_det(K_X) + N*log(2*pi) )
        """
        m_X, _, K_X_inv, log_det = output_GP_prior
        Y = Y - m_X
        # N = Y.size()[0]
        MLL = torch.matmul(Y.transpose(0, 1), torch.matmul(K_X_inv, Y))
        # MLL += log_det + N*np.log( 2*np.pi)
        MLL += log_det  # + N*np.log( 2*np.pi)
        return 0.5 * MLL
