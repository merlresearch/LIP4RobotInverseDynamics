# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
GP model with priors on kinetic and potential energy
The inverse dynamics kernel is derived by applying the Lagrangian equations to the priors

Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
"""
import time

import numpy as np
import torch


class m_GP_Lagrangian_kernel(torch.nn.Module):
    """
    Superclass of the models based on GP with Lagrangian kernel
    """

    def __init__(
        self,
        num_dof,
        pos_indices,
        vel_indices,
        acc_indices,
        init_param_dict,
        f_K_blocks,
        f_K_blocks_ltr,
        f_K_blocks_diag,
        f_K_T_Y_blocks=None,
        f_K_U_Y_blocks=None,
        f_K_L_Y_blocks=None,
        friction_model=None,
        f_phi_friction=None,
        flg_np=False,
        sigma_n_init=None,
        flg_train_sigma_n=False,
        name="",
        dtype=torch.float64,
        device=None,
        sigma_n_num=None,
        norm_coef=None,
        flg_norm_noise=False,
    ):
        super().__init__()
        # save model properties
        self.num_dof = num_dof
        self.pos_indices = pos_indices
        self.vel_indices = vel_indices
        self.acc_indices = acc_indices
        self.init_param_dict = init_param_dict
        self.f_K_blocks = f_K_blocks
        self.f_K_blocks_ltr = f_K_blocks_ltr
        self.f_K_blocks_diag = f_K_blocks_diag
        self.f_K_T_Y_blocks = f_K_T_Y_blocks
        self.f_K_U_Y_blocks = f_K_U_Y_blocks
        self.f_K_L_Y_blocks = f_K_L_Y_blocks
        self.flg_train_sigma_n = flg_train_sigma_n
        self.name = name
        self.dtype = dtype
        self.device = device
        # set noise parameters
        if sigma_n_num is not None:
            self.sigma_n_num = torch.tensor(sigma_n_num, dtype=self.dtype, device=self.device)
        else:
            self.sigma_n_num = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        self.sigma_n_log = torch.nn.Parameter(
            torch.tensor(np.log(sigma_n_init), dtype=self.dtype, device=self.device), requires_grad=flg_train_sigma_n
        )
        if norm_coef is None:
            norm_coef = np.ones(num_dof)
        self.norm_coef = torch.tensor(norm_coef, dtype=self.dtype, device=self.device)
        if flg_norm_noise:
            self.get_sigma_n2 = self.get_sigma_n2_norm
        else:
            self.get_sigma_n2 = self.get_sigma_n2_no_norm
        self.init_param()
        # check friction
        if friction_model is None:
            self.flg_frict = False
        else:
            self.flg_frict = True
            if friction_model == "linear":
                self.get_K_friction = self.get_K_friction_linear
                self.get_K_diag_friction = self.get_K_diag_friction_linear
                self.f_phi_friction = f_phi_friction
                self.log_K_friction_par = torch.nn.Parameter(
                    torch.tensor(np.log(init_param_dict["K_friction_par_init"]), dtype=self.dtype, device=self.device),
                    requires_grad=init_param_dict["flg_train_friction_par"],
                )
            elif friction_model == "RBF":
                self.get_K_friction = self.get_K_friction_RBF
                self.get_K_diag_friction = self.get_K_diag_friction_RBF
                self.f_phi_friction = f_phi_friction
                self.log_scale_friciton = torch.nn.Parameter(
                    torch.tensor(
                        np.log(init_param_dict["scale_friction_par_init"]), dtype=self.dtype, device=self.device
                    ),
                    requires_grad=init_param_dict["flg_train_friction_par"],
                )
                self.log_lengthscale_friciton = torch.nn.Parameter(
                    torch.tensor(
                        np.log(init_param_dict["lengthscale_friction_par_init"]), dtype=self.dtype, device=self.device
                    ),
                    requires_grad=init_param_dict["flg_train_friction_par"],
                )
        # check NP component
        if flg_np:
            # init an independent RBF kernel for each joint
            # (can compensate friction and other behaviors not included in rigif body dynamics)
            self.flg_np = True
            self.get_K_NP = self.get_K_RBF
            self.get_K_diag_NP = self.get_K_diag_RBF
            self.active_dims_NP = init_param_dict["active_dims_NP"]
            self.log_scale_NP = torch.nn.Parameter(
                torch.tensor(np.log(init_param_dict["scale_NP_par_init"]), dtype=self.dtype, device=self.device),
                requires_grad=init_param_dict["flg_train_NP_par"],
            )
            self.log_lengthscale_NP = torch.nn.Parameter(
                torch.tensor(np.log(init_param_dict["lengthscale_NP_par_init"]), dtype=self.dtype, device=self.device),
                requires_grad=init_param_dict["flg_train_NP_par"],
            )
        else:
            self.flg_np = False
            self.get_K_NP = lambda X1, X2, joint_index: 0.0
            self.get_K_diag_NP = lambda X, joint_index: 0.0

    def init_param(self):
        raise NotImplementedError

    def get_param_dict(self):
        raise NotImplementedError

    def f_K_blocks_wrapper(self, X1, X2):
        param_dict = self.get_param_dict()
        return self.f_K_blocks(
            X1=X1,
            X2=X2,
            pos_indices=self.pos_indices,
            vel_indices=self.vel_indices,
            acc_indices=self.acc_indices,
            **param_dict
        )

    def f_K_blocks_ltr_wrapper(self, X):
        param_dict = self.get_param_dict()
        return self.f_K_blocks_ltr(
            X1=X,
            X2=X,
            pos_indices=self.pos_indices,
            vel_indices=self.vel_indices,
            acc_indices=self.acc_indices,
            **param_dict
        )

    def f_K_blocks_diag_wrapper(self, X):
        param_dict = self.get_param_dict()
        return self.f_K_blocks_diag(
            X1=X,
            X2=X,
            pos_indices=self.pos_indices,
            vel_indices=self.vel_indices,
            acc_indices=self.acc_indices,
            **param_dict
        )

    def f_K_L_Y_blocks_wrapper(self, X1, X2):
        param_dict = self.get_param_dict()
        return self.f_K_L_Y_blocks(
            X1=X1,
            X2=X2,
            pos_indices=self.pos_indices,
            vel_indices=self.vel_indices,
            acc_indices=self.acc_indices,
            **param_dict
        )

    def f_K_T_Y_blocks_wrapper(self, X1, X2):
        param_dict = self.get_param_dict()
        return self.f_K_T_Y_blocks(
            X1=X1,
            X2=X2,
            pos_indices=self.pos_indices,
            vel_indices=self.vel_indices,
            acc_indices=self.acc_indices,
            **param_dict
        )

    def f_K_U_Y_blocks_wrapper(self, X1, X2):
        param_dict = self.get_param_dict()
        return self.f_K_U_Y_blocks(
            X1=X1,
            X2=X2,
            pos_indices=self.pos_indices,
            vel_indices=self.vel_indices,
            acc_indices=self.acc_indices,
            **param_dict
        )

    def print_model(self):
        """
        Print the model
        """
        print("\n" + self.name + " parameters:")
        for par_name, par in self.named_parameters():
            print("-", par_name, ":", par.data)

    def to(self, dev):
        """
        Set the device and move the parameters
        """
        # set the new device
        super().to(dev)
        self.device = dev
        self.sigma_n_num = self.sigma_n_num.to(dev)
        self.norm_coef = self.norm_coef.to(dev)

    def get_sigma_n2_no_norm(self):
        return torch.exp(self.sigma_n_log) ** 2 + self.sigma_n_num**2 * torch.ones(
            self.num_dof, dtype=self.dtype, device=self.device
        )

    def get_sigma_n2_norm(self):
        return (torch.exp(self.sigma_n_log) ** 2 + self.sigma_n_num**2) / (self.norm_coef**2)

    def forward(self, X):
        """
        Returns the prior distribution and the inverse anf the log_det of the prior covariace

        input:
        - X = training inputs (X has dimension [num_samples, num_features])

        output:
        - K_X = prior covariance of X
        - K_X_inv = inverse of the prior covariance
        - log_det = log det of the prior covariance
        """
        # get the covariance
        N = X.size()[0]
        noise_var = torch.cat(
            [
                sigma_n2_joint * torch.ones(N, dtype=self.dtype, device=self.device)
                for sigma_n2_joint in self.get_sigma_n2()
            ]
        )
        K_X = self.get_K_signle_input(X, flg_frict=self.flg_frict, flg_np=self.flg_np)
        K_X[range(N * self.num_dof), range(N * self.num_dof)] += noise_var
        L = torch.linalg.cholesky(K_X)
        log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        K_X_inv = torch.cholesky_solve(torch.eye(N * self.num_dof, dtype=self.dtype, device=self.device), L)

        return K_X, K_X_inv, log_det

    def get_K_signle_input(self, X, flg_frict, flg_np):
        N = X.shape[0]
        # get the kernel blocks
        K_blocks_list = self.f_K_blocks_ltr_wrapper(X)
        # init the kernel
        K = torch.zeros([self.num_dof * N, self.num_dof * N], dtype=self.dtype, device=self.device)
        K_blocks_index = 0
        index_1_to = 0
        # index_2_to_ = 0
        for joint_index_1 in range(self.num_dof):
            # upper diag indices
            index_1_from = index_1_to
            index_1_to = index_1_to + N
            index_2_to = joint_index_1 * N
            index_2_to = 0
            for joint_index_2 in range(joint_index_1 + 1):
                index_2_from = index_2_to
                index_2_to = index_2_to + N
                K_block_norm = K_blocks_list[K_blocks_index] / (
                    self.norm_coef[joint_index_1] * self.norm_coef[joint_index_2]
                )
                K[index_1_from:index_1_to, index_2_from:index_2_to] = K_block_norm
                if not (joint_index_1 == joint_index_2):
                    K[index_2_from:index_2_to, index_1_from:index_1_to] = K_block_norm.T
                if joint_index_1 == joint_index_2 and flg_frict:
                    K[index_1_from:index_1_to, index_2_from:index_2_to] += self.get_K_friction(
                        dq1=X[:, self.vel_indices[joint_index_1] : self.vel_indices[joint_index_1] + 1],
                        dq2=X[:, self.vel_indices[joint_index_1] : self.vel_indices[joint_index_1] + 1],
                        joint_index=joint_index_1,
                    )
                if joint_index_1 == joint_index_2 and flg_np:
                    K[index_1_from:index_1_to, index_2_from:index_2_to] += self.get_K_NP(
                        X1=X, X2=X, joint_index=joint_index_1
                    )
                K_blocks_index = K_blocks_index + 1
        return K

    def get_K(self, X1, X2, flg_frict, flg_np):
        N1 = X1.shape[0]
        N2 = X2.shape[0]
        # get the kernel blocks
        K_blocks_list = self.f_K_blocks_wrapper(X1, X2)
        # init the kernel
        K = torch.zeros([self.num_dof * N1, self.num_dof * N2], dtype=self.dtype, device=self.device)
        K_blocks_index = 0
        index_1_to = 0
        for joint_index_1 in range(self.num_dof):
            # upper diag indices
            index_1_from = index_1_to
            index_1_to = index_1_to + N1
            index_2_to = 0
            for joint_index_2 in range(self.num_dof):
                index_2_from = index_2_to
                index_2_to = index_2_to + N2
                K[index_1_from:index_1_to, index_2_from:index_2_to] = K_blocks_list[K_blocks_index] / (
                    self.norm_coef[joint_index_1] * self.norm_coef[joint_index_2]
                )
                if joint_index_1 == joint_index_2 and flg_frict:
                    K[index_1_from:index_1_to, index_2_from:index_2_to] += self.get_K_friction(
                        dq1=X1[:, self.vel_indices[joint_index_1] : self.vel_indices[joint_index_1] + 1],
                        dq2=X2[:, self.vel_indices[joint_index_1] : self.vel_indices[joint_index_1] + 1],
                        joint_index=joint_index_1,
                    )
                if joint_index_1 == joint_index_2 and flg_np:
                    K[index_1_from:index_1_to, index_2_from:index_2_to] += self.get_K_NP(
                        X1=X1, X2=X2, joint_index=joint_index_1
                    )
                K_blocks_index = K_blocks_index + 1
        return K

    def get_K_friction_linear(self, dq1, dq2, joint_index):
        friction_par = torch.exp(self.log_K_friction_par[joint_index, :])
        phi1 = self.f_phi_friction(dq1)
        phi2 = self.f_phi_friction(dq2)
        return (phi1 * friction_par) @ (phi2.transpose(1, 0))

    def get_K_diag_friction_linear(self, dq, joint_index):
        friction_par = torch.exp(self.log_K_friction_par[joint_index, :])
        phi = self.f_phi_friction(dq)
        return torch.sum((phi**2 * friction_par), 1)

    def get_K_friction_RBF(self, dq1, dq2, joint_index):
        scale = torch.exp(self.log_scale_friciton[joint_index])
        lengthscales = torch.exp(self.log_lengthscale_friciton[joint_index, :])
        return scale * torch.exp(
            -get_weigted_distances(X1=self.f_phi_friction(dq1), X2=self.f_phi_friction(dq2), l=lengthscales)
        )

    def get_K_diag_friction_RBF(self, dq, joint_index):
        scale = torch.exp(self.log_scale_friciton[joint_index])
        return scale * torch.ones_like(dq)

    def get_K_RBF(self, X1, X2, joint_index):
        scale = torch.exp(self.log_scale_NP[joint_index])
        lengthscale = torch.exp(self.log_lengthscale_NP[joint_index, :])
        return scale * torch.exp(
            -get_weigted_distances(
                X1=X1[:, self.active_dims_NP], X2=X2[:, self.active_dims_NP], lengthscale=lengthscale
            )
        )

    def get_K_diag_RBF(self, X, joint_index):
        scale = torch.exp(self.log_scale_NP[joint_index])
        return scale * torch.ones([X.shape[0], 1], dtype=X.dtype, device=X.device)

    def get_K_T_Y(self, X1, X2):
        # K_T_Y = torch.cat(self.f_K_T_Y_blocks_wrapper(X1, X2),1)
        K_T_Y = torch.cat([k / c for c, k in zip(self.norm_coef, self.f_K_T_Y_blocks_wrapper(X1, X2))], 1)
        return K_T_Y

    def get_K_U_Y(self, X1, X2):
        # K_U_Y = torch.cat(self.f_K_U_Y_blocks_wrapper(X1, X2),1)
        K_U_Y = torch.cat([k / c for c, k in zip(self.norm_coef, self.f_K_U_Y_blocks_wrapper(X1, X2))], 1)
        return K_U_Y

    def get_K_L_Y(self, X1, X2):
        # K_L_Y = torch.cat(self.f_K_L_Y_blocks_wrapper(X1, X2),1)
        K_L_Y = torch.cat([k / c for c, k in zip(self.norm_coef, self.f_K_L_Y_blocks_wrapper(X1, X2))], 1)
        return K_L_Y

    def get_diag_covariance(self, X):
        """
        Computes the diagonal of the prior coviariance matrix

        inputs:
        - X = input locations (shape [N,D])

        outputs:
        - diagonal of K(X,X)n (shape [N,D])
        """
        # Compute the diagonal elements of the diagonal kernel blocks
        # diagonals_K_blocks = [K_block for K_block in self.f_K_blocks_diag_wrapper(X)]
        # return torch.cat(diagonals_K_blocks,1)
        diag = torch.cat([k / (c**2) for c, k in zip(self.norm_coef, self.f_K_blocks_diag_wrapper(X))], 1)
        if self.flg_frict:
            friction_diag = torch.cat(
                [
                    self.get_K_diag_friction(
                        dq=X[:, self.vel_indices[joint_index] : self.vel_indices[joint_index] + 1],
                        joint_index=joint_index,
                    ).reshape([-1, 1])
                    for joint_index in range(self.num_dof)
                ],
                1,
            )
            diag += friction_diag
        if self.flg_np:
            NP_diag = torch.cat(
                [
                    self.get_K_diag_NP(X=X, joint_index=joint_index).reshape([-1, 1])
                    for joint_index in range(self.num_dof)
                ],
                1,
            )
            diag += NP_diag
        return diag

    def get_alpha(self, X, Y):
        """
        Returns alpha, the vector of coefficients defining the posterior distribution

        inputs:
        - X = training input [N,D]
        - Y = training output [N,num_dof]

        outputs:
        - alpha = vector defining the posterior distribution
        - K_X_inv = inverse of the prior covariance of X
        """
        _, K_X_inv, _ = self.forward(X)
        alpha = torch.matmul(K_X_inv, Y.transpose(0, 1).reshape([-1, 1]))
        return alpha, K_X_inv

    def get_estimate_from_alpha(self, X, X_test, alpha, K_X_inv=None, flg_frict=None, flg_np=None):
        """
        Compute the posterior distribution in X_test, given the alpha vector.

        input:
        - X = training input locations (used to compute alpha)
        - X_test = test input locations
        - alpha = vector of coefficients defining the posterior
        - K_X_inv = inverse of the prior covariance of X

        output:
        - Y_hat = posterior mean
        - var = diagonal elements of the posterior variance

        If K_X_inv is given the method returns also the confidence intervals (variance of the gaussian)
        If Y_test is given the method prints the MSE
        """
        # get covariance and prior mean
        if flg_frict is None:
            flg_frict = self.flg_frict
        if flg_np is None:
            flg_np = self.flg_np
        K_X_test_X = self.get_K(X_test, X, flg_frict, flg_np)
        # get the estimate
        Y_hat = torch.matmul(K_X_test_X, alpha).reshape([self.num_dof, X_test.shape[0]]).transpose(0, 1)
        # if K_X_inv is given compute the confidence intervals
        if K_X_inv is not None:
            # num_test = X_test.size()[0]
            var = self.get_diag_covariance(X_test) - torch.sum(
                torch.matmul(K_X_test_X, K_X_inv) * (K_X_test_X), dim=1
            ).reshape([self.num_dof, X_test.shape[0]]).transpose(0, 1)
        else:
            var = None
        return Y_hat, var

    def get_T_estimate_from_alpha(self, X, X_test, alpha, K_X_inv=None):
        """
        Compute the posterior distribution of the kinetic energy in X_test, given the alpha vector.

        input:
        - X = training input locations (used to compute alpha)
        - X_test = test input locations
        - alpha = vector of coefficients defining the posterior
        - K_X_inv = inverse of the prior covariance of X

        output:
        - Y_hat = posterior mean
        - var = diagonal elements of the posterior variance

        If K_X_inv is given the method returns also the confidence intervals (variance of the gaussian)
        """
        # get covariance and prior mean
        K_T_Y = self.get_K_T_Y(X_test, X)
        # get the estimate
        Y_T_hat = torch.matmul(K_T_Y, alpha)
        var = None
        return Y_T_hat, var

    def get_U_estimate_from_alpha(self, X, X_test, alpha, K_X_inv=None):
        """
        Compute the posterior distribution of the potential energy in X_test, given the alpha vector.

        input:
        - X = training input locations (used to compute alpha)
        - X_test = test input locations
        - alpha = vector of coefficients defining the posterior
        - K_X_inv = inverse of the prior covariance of X

        output:
        - Y_hat = posterior mean
        - var = diagonal elements of the posterior variance

        If K_X_inv is given the method returns also the confidence intervals (variance of the gaussian)
        """
        # get covariance and prior mean
        K_U_Y = self.get_K_U_Y(X_test, X)
        # get the estimate
        Y_U_hat = torch.matmul(K_U_Y, alpha)
        var = None
        return Y_U_hat, var

    def get_L_estimate_from_alpha(self, X, X_test, alpha, K_X_inv=None):
        """
        Compute the posterior distribution of the larangian energy in X_test, given the alpha vector.

        input:
        - X = training input locations (used to compute alpha)
        - X_test = test input locations
        - alpha = vector of coefficients defining the posterior
        - K_X_inv = inverse of the prior covariance of X

        output:
        - Y_hat = posterior mean
        - var = diagonal elements of the posterior variance

        If K_X_inv is given the method returns also the confidence intervals (variance of the gaussian)
        """
        # get covariance and prior mean
        K_L_Y = self.get_K_L_Y(X_test, X)
        # get the estimate
        Y_L_hat = torch.matmul(K_L_Y, alpha)
        var = None
        return Y_L_hat, var

    def get_energy_estimate_from_alpha(self, X, X_test, alpha, K_X_inv=None):
        """
        Compute the posterior distribution of the lagrangian, kinetic and potential energy in X_test,
        given the alpha vector.

        input:
        - X = training input locations (used to compute alpha)
        - X_test = test input locations
        - alpha = vector of coefficients defining the posterior
        - K_X_inv = inverse of the prior covariance of X

        output:
        - L_hat = posterior mean of the lagrangian
        - T_hat = posterior mean of the kinetic energy
        - U_hat = posterior mean of the potential energy
        - L_var = diagonal elements of the lagrangian posterior variance
        - T_var = diagonal elements of the kinetic energy posterior variance
        - U_var = diagonal elements of the potential energy posterior variance

        If K_X_inv is given the method returns also the confidence intervals (variance of the gaussian)
        """
        # lagrangian
        L_hat, L_var = self.get_L_estimate_from_alpha(
            X=torch.tensor(X, dtype=self.dtype, device=self.device),
            X_test=torch.tensor(X_test, dtype=self.dtype, device=self.device),
            alpha=alpha,
            K_X_inv=None,
        )
        L_hat = L_hat.detach().cpu().numpy()
        # kinetic
        T_hat, T_var = self.get_T_estimate_from_alpha(
            X=torch.tensor(X, dtype=self.dtype, device=self.device),
            X_test=torch.tensor(X_test, dtype=self.dtype, device=self.device),
            alpha=alpha,
            K_X_inv=None,
        )
        T_hat = T_hat.detach().cpu().numpy()
        # potential
        U_hat, U_var = self.get_U_estimate_from_alpha(
            X=torch.tensor(X, dtype=self.dtype, device=self.device),
            X_test=torch.tensor(X_test, dtype=self.dtype, device=self.device),
            alpha=alpha,
            K_X_inv=None,
        )
        U_hat = U_hat.detach().cpu().numpy()
        return L_hat, T_hat, U_hat, L_var, T_var, U_var

    def get_torques_estimate(self, X, Y, X_test, flg_return_K_X_inv=False):
        """
        Returns the posterior distribution in X_test, given the training samples X Y.

        input:
        - X = training input [N, D]
        - Y = training output [N, num_dof]
        - X_test = test input

        output:
        - Y_hat = mean of the test posterior
        - var = diagonal elements of the variance posterior
        - alpha = coefficients defining the posterior
        - K_X_inv = inverse of the training covariance

        The function returns:
           -a vector containing the sigma squared confidence intervals
           -the vector of the coefficient
           -the K_X inverse in case required through flg_return_K_X_inv"""
        # get the coefficent and the mean
        alpha, K_X_inv = self.get_alpha(X, Y)
        # get the estimate and the confidence intervals
        Y_hat, var = self.get_estimate_from_alpha(X, X_test, alpha, K_X_inv=K_X_inv)
        # return the opportune values
        if flg_return_K_X_inv:
            return Y_hat, var, alpha, K_X_inv
        else:
            return Y_hat, var, alpha

    def train_model(self, X, Y, f_optimizer, criterion, batch_size, shuffle, N_epoch, N_epoch_print, drop_last=False):
        """
        Train the GPs hyperparameters by marginal likelihood maximization
        Shapes:
        X: [N, D]
        Y: [N, num_dof]
        """
        # create the dataset and the data loader
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, requires_grad=False, dtype=self.dtype, device=self.device),
            torch.tensor(Y, requires_grad=False, dtype=self.dtype, device=self.device),
        )
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        # initialize the optimizer
        optimizer = f_optimizer(self.parameters())
        t_start = time.time()
        # train the models
        for epoch in range(N_epoch):
            running_loss = 0.0
            N_btc = 0
            optimizer.zero_grad()
            # print('\nEPOCH:', epoch)
            for i, data in enumerate(trainloader, 0):
                # get the data
                inputs, labels = data
                # compute the loss
                _, K_X_inv, log_det = self(inputs)
                loss = criterion((0.0, 0.0, K_X_inv, log_det), labels.transpose(0, 1).reshape([-1, 1]))
                # update the loss
                running_loss = running_loss + loss.item()
                N_btc = N_btc + 1
                # propagate the gradient and update the parameters
                loss.backward(retain_graph=False)
                optimizer.step()
                # zero the parameter gradients
                optimizer.zero_grad()
            if epoch % N_epoch_print == 0:
                with torch.no_grad():
                    print("\nEPOCH:", epoch)
                    self.print_model()
                    print("Running loss:", running_loss / N_btc)
                    t_stop = time.time()
                    print("Time elapsed:", t_stop - t_start)
                    t_start = time.time()

    def get_g_estimates(self, X_tr, X_test, alpha):
        """
        Returns an estimate of the gravity torque related to the configurations
        defined by the input locations in X_test

        input:
        - X_tr = training input [N, D]
        - X_test = test input [N_t, D]
        - alpha = vector of coefficients defining the posterior

        output:
        - g = gravity torque [N_t, n]
        """

        # get the input locations with acc and vel null
        X_grav = torch.zeros_like(X_test)
        X_grav[:, self.pos_indices] = X_test[:, self.pos_indices]
        g, _ = self.get_estimate_from_alpha(X_tr, X_grav, alpha, K_X_inv=None, flg_frict=False, flg_np=False)
        return g * self.norm_coef

    def get_m_estimates(self, X_tr, X_test, alpha):
        """
        Returns an estimate of the inertial torque (M*ddq) related to the configurations
        defined by the input locations in X_test

        input:
        - X_tr = training input [N, D]
        - X_test = test input [N_t, D]
        - alpha = vector of coefficients defining the posterior

        output:
        - m = inertial torque [N_t, n]
        """

        # get the input locations with acc and vel null
        X_grav = torch.zeros_like(X_test)
        X_grav[:, self.pos_indices] = X_test[:, self.pos_indices]
        g, _ = self.get_estimate_from_alpha(X_tr, X_grav, alpha, K_X_inv=None, flg_frict=False, flg_np=False)

        # get the input locations with vel null
        X_acc = torch.zeros_like(X_test)
        X_acc[:, self.pos_indices] = X_test[:, self.pos_indices]
        X_acc[:, self.acc_indices] = X_test[:, self.acc_indices]
        mg, _ = self.get_estimate_from_alpha(X_tr, X_acc, alpha, K_X_inv=None, flg_frict=False, flg_np=False)
        m = mg - g

        return m * self.norm_coef

    def get_c_estimates(self, X_tr, X_test, alpha):
        """
        Returns an estimate of the coriolis torque (C*dq) related to the configurations
        defined by the input locations in X_test

        input:
        - X_tr = training input [N, D]
        - X_test = test input [N_t, D]
        - alpha = vector of coefficients defining the posterior

        output:
        - c = coriolis torque [N_t, n]
        """

        # get the input locations with acc and vel null
        X_grav = torch.zeros_like(X_test)
        X_grav[:, self.pos_indices] = X_test[:, self.pos_indices]
        g, _ = self.get_estimate_from_alpha(X_tr, X_grav, alpha, K_X_inv=None, flg_frict=False, flg_np=False)

        # get the input locations with vel null
        X_vel = torch.zeros_like(X_test)
        X_vel[:, self.pos_indices] = X_test[:, self.pos_indices]
        X_vel[:, self.vel_indices] = X_test[:, self.vel_indices]
        cg, _ = self.get_estimate_from_alpha(X_tr, X_vel, alpha, K_X_inv=None, flg_frict=False, flg_np=False)
        c = cg - g

        return c * self.norm_coef

    def get_M_estimates(self, X_tr, X_test, alpha):
        """
        Returns an estimate of the model inertia matrix related to the configurations
        defined by the input locations in X_test

        input:
        - X_tr = training input [N, D]
        - X_test = test input [N_t, D]
        - alpha = vector of coefficients defining the posterior

        output:
        - M = inertia matrix [N_t, n, n]
        """

        M_list = []
        # get the input locations with acc and vel null
        g = self.get_g_estimates(X_tr=X_tr, X_test=X_test, alpha=alpha)

        for joint_index in range(0, self.num_dof):
            # X_acc = torch.zeros_like(X_grav)
            X_acc = torch.zeros_like(X_test)
            X_acc[:, self.pos_indices] = X_test[:, self.pos_indices]
            X_acc[:, self.acc_indices[joint_index]] = 1.0
            Mg, _ = self.get_estimate_from_alpha(X_tr, X_acc, alpha, K_X_inv=None, flg_frict=False, flg_np=False)
            M_list.append((Mg * self.norm_coef - g).reshape([-1, 1, self.num_dof]))

        M = torch.cat(M_list, 1)
        return M

    def get_M_estimates_T(self, X_tr, X_test, alpha):
        """
        Returns an estimate of the model inertia matrix related to the configurations
        defined by the input locations in X_test, computed from the kinetic energy

        input:
        - X_tr = training input [N, D]
        - X_test = test input [N_t, D]
        - alpha = vector of coefficients defining the posterior

        output:
        - M = inertia matrix [N_t, n, n]
        """
        N_t, D = X_test.shape
        M = torch.zeros([N_t, self.num_dof, self.num_dof], device=self.device, dtype=self.dtype)
        X_pos = torch.zeros_like(X_test)
        X_pos[:, self.pos_indices] = X_test[:, self.pos_indices]
        # compute the diagonal elements
        for joint_index in range(0, self.num_dof):
            X_tmp = X_pos.clone()
            X_tmp[:, self.vel_indices[joint_index]] = 1.0
            tmp, _ = self.get_T_estimate_from_alpha(X=X_tr, X_test=X_tmp, alpha=alpha, K_X_inv=None)
            tmp = tmp.squeeze()
            M[:, joint_index, joint_index] = tmp
        # compute the off-diagonal elements
        for joint_index_1 in range(0, self.num_dof):
            for joint_index_2 in range(joint_index_1 + 1, self.num_dof):
                X_tmp = X_pos.clone()
                X_tmp[:, [self.vel_indices[joint_index_1], self.vel_indices[joint_index_2]]] = 1.0
                tmp, _ = self.get_T_estimate_from_alpha(X=X_tr, X_test=X_tmp, alpha=alpha, K_X_inv=None)
                tmp = (tmp.squeeze() - M[:, joint_index_1, joint_index_1] - M[:, joint_index_2, joint_index_2]) / 2
                M[:, joint_index_1, joint_index_2] += tmp
                M[:, joint_index_2, joint_index_1] += tmp
        return 2 * M

    def get_friction_torques(self, X_tr, X_test, alpha):
        """
        Returns an estimate of the friction torques related to the configurations
        defined by the input locations in X_test, computed from the kinetic energy

        input:
        - X_tr = training input [N, D]
        - X_test = test input [N_t, D]
        - alpha = vector of coefficients defining the posterior

        output:
        - tau_frict = [N_t, n]
        """
        N = X_tr.shape[0]
        tau_frict_list = [
            self.get_K_friction(
                dq1=X_test[:, self.vel_indices[joint_index] : self.vel_indices[joint_index] + 1],
                dq2=X_tr[:, self.vel_indices[joint_index] : self.vel_indices[joint_index] + 1],
                friction_par=torch.exp(self.log_K_friction_par[joint_index, :]),
            )
            @ alpha[N * joint_index : N * (joint_index + 1), :]
            for joint_index in range(self.num_dof)
        ]
        return torch.cat(tau_frict_list, 1)

    def get_friction_parameters(self, X_tr, alpha):
        """
        Returns the posterior estimate of the friction parameters
        works only with linear friction kernel
        """
        N = X_tr.shape[0]
        w_friction_list = []
        for joint_index in range(self.num_dof):
            Sigma = torch.diag(torch.exp(self.log_K_friction_par[joint_index, :]))
            phi_T = self.f_phi_friction(
                X_tr[:, self.vel_indices[joint_index] : self.vel_indices[joint_index] + 1]
            ).transpose(1, 0)
            w_friction_list.append(Sigma @ phi_T @ alpha[N * joint_index : N * (joint_index + 1), :])
        return w_friction_list


class m_GP_LK_RBF(m_GP_Lagrangian_kernel):
    """
    Implementation of the m_GP_Lagrangian_kernel
    with RBF prior on kinetic and potential energy
    """

    def __init__(
        self,
        num_dof,
        pos_indices,
        vel_indices,
        acc_indices,
        init_param_dict,
        f_K_blocks,
        f_K_blocks_ltr,
        f_K_blocks_diag,
        f_K_T_Y_blocks=None,
        f_K_U_Y_blocks=None,
        f_K_L_Y_blocks=None,
        friction_model=None,
        f_phi_friction=None,
        flg_np=False,
        sigma_n_init=None,
        flg_train_sigma_n=True,
        name="",
        dtype=torch.float64,
        device=None,
        sigma_n_num=None,
        norm_coef=None,
        flg_norm_noise=False,
    ):
        super().__init__(
            num_dof=num_dof,
            pos_indices=pos_indices,
            vel_indices=vel_indices,
            acc_indices=acc_indices,
            init_param_dict=init_param_dict,
            f_K_blocks=f_K_blocks,
            f_K_blocks_ltr=f_K_blocks_ltr,
            f_K_blocks_diag=f_K_blocks_diag,
            f_K_T_Y_blocks=f_K_T_Y_blocks,
            f_K_U_Y_blocks=f_K_U_Y_blocks,
            f_K_L_Y_blocks=f_K_L_Y_blocks,
            friction_model=friction_model,
            f_phi_friction=f_phi_friction,
            flg_np=flg_np,
            sigma_n_init=sigma_n_init,
            flg_train_sigma_n=flg_train_sigma_n,
            name=name,
            dtype=dtype,
            device=device,
            sigma_n_num=sigma_n_num,
            norm_coef=norm_coef,
            flg_norm_noise=flg_norm_noise,
        )

    def init_param(self):
        """
        Initialize the RBF parameters
        """
        # -------------- KINETIC ENERGY HYPERPARAMETERS --------------------
        self.log_lengthscales_par_T = torch.nn.Parameter(
            torch.tensor(np.log(self.init_param_dict["lengthscales_T_init"]), dtype=self.dtype, device=self.device),
            requires_grad=self.init_param_dict["flg_train_lengthscales_T"],
        )
        self.scale_log_T = torch.nn.Parameter(
            torch.tensor(np.log(self.init_param_dict["scale_T_init"]), dtype=self.dtype, device=self.device),
            requires_grad=self.init_param_dict["flg_train_scale_T"],
        )
        # -------------- POTENTIAL ENERGY HYPERPARAMETERS --------------------
        self.log_lengthscales_par_U = torch.nn.Parameter(
            torch.tensor(np.log(self.init_param_dict["lengthscales_U_init"]), dtype=self.dtype, device=self.device),
            requires_grad=self.init_param_dict["flg_train_lengthscales_U"],
        )
        self.scale_log_U = torch.nn.Parameter(
            torch.tensor(np.log(self.init_param_dict["scale_U_init"]), dtype=self.dtype, device=self.device),
            requires_grad=self.init_param_dict["flg_train_scale_U"],
        )

    def get_param_dict(self):
        """
        Return a dictionary with the kernel parameters (transfromed if necessary)
        """
        d = {}
        d["lT"] = torch.exp(self.log_lengthscales_par_T)
        d["lU"] = torch.exp(self.log_lengthscales_par_U)
        d["sT"] = torch.exp(self.scale_log_T)
        d["sU"] = torch.exp(self.scale_log_U)
        return d


class m_GP_LK_RBF_1(m_GP_Lagrangian_kernel):
    """
    Implementation of the m_GP_Lagrangian_kernel
    with RBF prior on Lagrangian function
    """

    def __init__(
        self,
        num_dof,
        pos_indices,
        vel_indices,
        acc_indices,
        init_param_dict,
        f_K_blocks,
        f_K_blocks_ltr,
        f_K_blocks_diag,
        f_K_T_Y_blocks=None,
        f_K_U_Y_blocks=None,
        f_K_L_Y_blocks=None,
        friction_model=None,
        f_phi_friction=None,
        flg_np=False,
        sigma_n_init=None,
        flg_train_sigma_n=True,
        name="",
        dtype=torch.float64,
        device=None,
        sigma_n_num=None,
        norm_coef=None,
        flg_norm_noise=False,
    ):
        super().__init__(
            num_dof=num_dof,
            pos_indices=pos_indices,
            vel_indices=vel_indices,
            acc_indices=acc_indices,
            init_param_dict=init_param_dict,
            f_K_blocks=f_K_blocks,
            f_K_blocks_ltr=f_K_blocks_ltr,
            f_K_blocks_diag=f_K_blocks_diag,
            f_K_T_Y_blocks=f_K_T_Y_blocks,
            f_K_U_Y_blocks=f_K_U_Y_blocks,
            f_K_L_Y_blocks=f_K_L_Y_blocks,
            friction_model=friction_model,
            f_phi_friction=f_phi_friction,
            flg_np=flg_np,
            sigma_n_init=sigma_n_init,
            flg_train_sigma_n=flg_train_sigma_n,
            name=name,
            dtype=dtype,
            device=device,
            sigma_n_num=sigma_n_num,
            norm_coef=norm_coef,
            flg_norm_noise=flg_norm_noise,
        )

    def init_param(self):
        """
        Initialize the RBF parameters
        """
        # -------------- KINETIC ENERGY HYPERPARAMETERS --------------------
        self.log_lengthscales_par_L = torch.nn.Parameter(
            torch.tensor(np.log(self.init_param_dict["lengthscales_L_init"]), dtype=self.dtype, device=self.device),
            requires_grad=self.init_param_dict["flg_train_lengthscales_L"],
        )
        self.scale_log_L = torch.nn.Parameter(
            torch.tensor(np.log(self.init_param_dict["scale_L_init"]), dtype=self.dtype, device=self.device),
            requires_grad=self.init_param_dict["flg_train_scale_L"],
        )

    def get_param_dict(self):
        """
        Return a dictionary with the kernel parameters (transfromed if necessary)
        """
        d = {}
        d["lL"] = torch.exp(self.log_lengthscales_par_L)
        d["sL"] = torch.exp(self.scale_log_L)
        return d

    def get_energy_estimate_from_alpha(self, X, X_test, alpha, K_X_inv=None):
        """
        Compute the posterior distribution of the lagrangian, kinetic and potential energy in X_test,
        given the alpha vector.

        input:
        - X = training input locations (used to compute alpha)
        - X_test = test input locations
        - alpha = vector of coefficients defining the posterior
        - K_X_inv = inverse of the prior covariance of X

        output:
        - L_hat = posterior mean of the lagrangian
        - T_hat = posterior mean of the kinetic energy
        - U_hat = posterior mean of the potential energy
        - L_var = diagonal elements of the lagrangian posterior variance
        - T_var = diagonal elements of the kinetic energy posterior variance
        - U_var = diagonal elements of the potential energy posterior variance

        If K_X_inv is given the method returns also the confidence intervals (variance of the gaussian)
        """
        # get the lagrangian
        K_L_Y = self.get_K_L_Y(
            torch.tensor(X_test, dtype=self.dtype, device=self.device),
            torch.tensor(X, dtype=self.dtype, device=self.device),
        )
        L_hat = torch.matmul(K_L_Y, alpha).detach().cpu().numpy()
        L_var = None
        # get_potential
        X_test[:, self.vel_indices] = 0.0
        X_test[:, self.acc_indices] = 0.0
        K_U_Y = self.get_K_L_Y(
            torch.tensor(X_test, dtype=self.dtype, device=self.device),
            torch.tensor(X, dtype=self.dtype, device=self.device),
        )
        U_hat = torch.matmul(K_U_Y, alpha).detach().cpu().numpy()
        U_var = None
        # get kinetic
        T_hat = L_hat + U_hat
        T_var = None
        return L_hat, T_hat, U_hat, L_var, T_var, U_var

    def get_U_estimate_from_alpha(self, X, X_test, alpha, K_X_inv=None):
        """
        Compute the posterior distribution of the potential energy in X_test, given the alpha vector.

        input:
        - X = training input locations (used to compute alpha)
        - X_test = test input locations
        - alpha = vector of coefficients defining the posterior
        - K_X_inv = inverse of the prior covariance of X

        output:
        - Y_hat = posterior mean
        - var = diagonal elements of the posterior variance

        If K_X_inv is given the method returns also the confidence intervals (variance of the gaussian)
        """
        # get covariance and prior mean
        X_test[:, self.vel_indices] = 0.0
        X_test[:, self.acc_indices] = 0.0
        K_U_Y = self.get_K_L_Y(X_test, X)
        # get the estimate
        Y_U_hat = torch.matmul(K_U_Y, alpha)
        var = None
        return Y_U_hat, var


class m_GP_LK_GIP(m_GP_Lagrangian_kernel):
    """
    Implementation of the m_GP_Lagrangian_kernel
    with GIP prior on kinetic and potential energy
    """

    def __init__(
        self,
        num_dof,
        pos_indices,
        vel_indices,
        acc_indices,
        init_param_dict,
        f_K_blocks,
        f_K_blocks_ltr,
        f_K_blocks_diag,
        f_K_T_Y_blocks=None,
        f_K_U_Y_blocks=None,
        f_K_L_Y_blocks=None,
        friction_model=None,
        f_phi_friction=None,
        flg_np=False,
        sigma_n_init=None,
        flg_train_sigma_n=True,
        name="",
        dtype=torch.float64,
        device=None,
        sigma_n_num=None,
        norm_coef=None,
        flg_norm_noise=False,
    ):
        super().__init__(
            num_dof=num_dof,
            pos_indices=pos_indices,
            vel_indices=vel_indices,
            acc_indices=acc_indices,
            init_param_dict=init_param_dict,
            f_K_blocks=f_K_blocks,
            f_K_blocks_ltr=f_K_blocks_ltr,
            f_K_blocks_diag=f_K_blocks_diag,
            f_K_T_Y_blocks=f_K_T_Y_blocks,
            f_K_U_Y_blocks=f_K_U_Y_blocks,
            f_K_L_Y_blocks=f_K_L_Y_blocks,
            friction_model=friction_model,
            f_phi_friction=f_phi_friction,
            flg_np=flg_np,
            sigma_n_init=sigma_n_init,
            flg_train_sigma_n=flg_train_sigma_n,
            name=name,
            dtype=dtype,
            device=device,
            sigma_n_num=sigma_n_num,
            norm_coef=norm_coef,
            flg_norm_noise=flg_norm_noise,
        )

    def init_param(self):
        """
        Initialize the RBF parameters
        """
        # -------------- KINETIC ENERGY HYPERPARAMETERS --------------------
        self.log_sigma_kin_vel = torch.nn.Parameter(
            torch.tensor(np.log(self.init_param_dict["sigma_kin_vel_init"]), dtype=self.dtype, device=self.device),
            requires_grad=self.init_param_dict["flg_train_sigma_kin_vel"],
        )

        if "sigma_kin_pos_prism_init" in self.init_param_dict:
            self.log_sigma_kin_pos_prism = torch.nn.Parameter(
                torch.tensor(
                    np.log(self.init_param_dict["sigma_kin_pos_prism_init"]), dtype=self.dtype, device=self.device
                ),
                requires_grad=self.init_param_dict["flg_train_sigma_kin_pos_prism"],
            )
        else:
            self.log_sigma_kin_pos_prism = torch.tensor([])

        if "sigma_kin_pos_rev_init" in self.init_param_dict:
            self.log_sigma_kin_pos_rev = torch.nn.Parameter(
                torch.tensor(
                    np.log(self.init_param_dict["sigma_kin_pos_rev_init"]), dtype=self.dtype, device=self.device
                ),
                requires_grad=self.init_param_dict["flg_train_sigma_kin_pos_rev"],
            )
        else:
            self.log_sigma_kin_pos_rev = torch.tensor([])

        # -------------- POTENTIAL ENERGY HYPERPARAMETERS --------------------
        if "sigma_pot_prism_init" in self.init_param_dict:
            self.log_sigma_pot_prism = torch.nn.Parameter(
                torch.tensor(
                    np.log(self.init_param_dict["sigma_pot_prism_init"]), dtype=self.dtype, device=self.device
                ),
                requires_grad=self.init_param_dict["flg_train_sigma_pot_prism"],
            )
        else:
            self.log_sigma_pot_prism = torch.tensor([])

        if "sigma_pot_rev_init" in self.init_param_dict:
            self.log_sigma_pot_rev = torch.nn.Parameter(
                torch.tensor(np.log(self.init_param_dict["sigma_pot_rev_init"]), dtype=self.dtype, device=self.device),
                requires_grad=self.init_param_dict["flg_train_sigma_pot_rev"],
            )
        else:
            self.log_sigma_pot_rev = torch.tensor([])

    def get_param_dict(self):
        """
        Return a dictionary with the kernel parameters (transfromed if necessary)
        """
        d = {}
        d["sigma_kin_vel"] = torch.exp(self.log_sigma_kin_vel)
        d["sigma_kin_pos_prism"] = torch.exp(self.log_sigma_kin_pos_prism)
        d["sigma_pot_prism"] = torch.exp(self.log_sigma_pot_prism)
        d["sigma_kin_pos_rev"] = torch.exp(self.log_sigma_kin_pos_rev)
        d["sigma_pot_rev"] = torch.exp(self.log_sigma_pot_rev)
        return d


class m_GP_LK_POLY_RBF(m_GP_Lagrangian_kernel):
    """
    Implementation of the m_GP_Lagrangian_kernel
    with RBF*POLY prior on kinetic energy
    and RBF prior on potential energy
    """

    def __init__(
        self,
        num_dof,
        pos_indices,
        vel_indices,
        acc_indices,
        init_param_dict,
        f_K_blocks,
        f_K_blocks_ltr,
        f_K_blocks_diag,
        f_K_T_Y_blocks=None,
        f_K_U_Y_blocks=None,
        f_K_L_Y_blocks=None,
        friction_model=None,
        f_phi_friction=None,
        flg_np=False,
        sigma_n_init=None,
        flg_train_sigma_n=True,
        name="",
        dtype=torch.float64,
        device=None,
        sigma_n_num=None,
        norm_coef=None,
        flg_norm_noise=False,
    ):
        super().__init__(
            num_dof=num_dof,
            pos_indices=pos_indices,
            vel_indices=vel_indices,
            acc_indices=acc_indices,
            init_param_dict=init_param_dict,
            f_K_blocks=f_K_blocks,
            f_K_blocks_ltr=f_K_blocks_ltr,
            f_K_blocks_diag=f_K_blocks_diag,
            f_K_T_Y_blocks=f_K_T_Y_blocks,
            f_K_U_Y_blocks=f_K_U_Y_blocks,
            f_K_L_Y_blocks=f_K_L_Y_blocks,
            friction_model=friction_model,
            f_phi_friction=f_phi_friction,
            flg_np=flg_np,
            sigma_n_init=sigma_n_init,
            flg_train_sigma_n=flg_train_sigma_n,
            name=name,
            dtype=dtype,
            device=device,
            sigma_n_num=sigma_n_num,
            norm_coef=norm_coef,
            flg_norm_noise=flg_norm_noise,
        )

    def init_param(self):
        """
        Initialize the POLY-RBF parameters
        """
        # -------------- KINETIC ENERGY HYPERPARAMETERS --------------------
        self.log_lengthscales_par_T = torch.nn.Parameter(
            torch.tensor(np.log(self.init_param_dict["lengthscales_T_init"]), dtype=self.dtype, device=self.device),
            requires_grad=self.init_param_dict["flg_train_lengthscales_T"],
        )
        self.scale_log_T = torch.nn.Parameter(
            torch.tensor(np.log(self.init_param_dict["scale_T_init"]), dtype=self.dtype, device=self.device),
            requires_grad=self.init_param_dict["flg_train_scale_T"],
        )
        self.log_sigma_POLY = torch.nn.Parameter(
            torch.tensor(np.log(self.init_param_dict["sigma_POLY_init"]), dtype=self.dtype, device=self.device),
            requires_grad=self.init_param_dict["flg_train_sigma_POLY"],
        )
        # -------------- POTENTIAL ENERGY HYPERPARAMETERS --------------------
        self.log_lengthscales_par_U = torch.nn.Parameter(
            torch.tensor(np.log(self.init_param_dict["lengthscales_U_init"]), dtype=self.dtype, device=self.device),
            requires_grad=self.init_param_dict["flg_train_lengthscales_U"],
        )
        self.scale_log_U = torch.nn.Parameter(
            torch.tensor(np.log(self.init_param_dict["scale_U_init"]), dtype=self.dtype, device=self.device),
            requires_grad=self.init_param_dict["flg_train_scale_U"],
        )

    def get_param_dict(self):
        """
        Return a dictionary with the kernel parameters (transfromed if necessary)
        """
        d = {}
        d["lT"] = torch.exp(self.log_lengthscales_par_T)
        d["lU"] = torch.exp(self.log_lengthscales_par_U)
        d["sT"] = torch.exp(self.scale_log_T)
        d["sU"] = torch.exp(self.scale_log_U)
        d["sigmaT"] = torch.exp(self.log_sigma_POLY)
        return d


class m_GP_LK_GIP_sum(m_GP_Lagrangian_kernel):
    """
    Implementation of the m_GP_Lagrangian_kernel
    with GIP sum prior on kinetic and potential energy
    """

    def __init__(
        self,
        num_dof,
        pos_indices,
        vel_indices,
        acc_indices,
        init_param_dict,
        f_K_blocks,
        f_K_blocks_ltr,
        f_K_blocks_diag,
        f_K_T_Y_blocks=None,
        f_K_U_Y_blocks=None,
        f_K_L_Y_blocks=None,
        friction_model=None,
        f_phi_friction=None,
        flg_np=False,
        sigma_n_init=None,
        flg_train_sigma_n=True,
        name="",
        dtype=torch.float64,
        device=None,
        sigma_n_num=None,
        norm_coef=None,
        flg_norm_noise=False,
    ):
        super().__init__(
            num_dof=num_dof,
            pos_indices=pos_indices,
            vel_indices=vel_indices,
            acc_indices=acc_indices,
            init_param_dict=init_param_dict,
            f_K_blocks=f_K_blocks,
            f_K_blocks_ltr=f_K_blocks_ltr,
            f_K_blocks_diag=f_K_blocks_diag,
            f_K_T_Y_blocks=f_K_T_Y_blocks,
            f_K_U_Y_blocks=f_K_U_Y_blocks,
            f_K_L_Y_blocks=f_K_L_Y_blocks,
            friction_model=friction_model,
            f_phi_friction=f_phi_friction,
            flg_np=flg_np,
            sigma_n_init=sigma_n_init,
            flg_train_sigma_n=flg_train_sigma_n,
            name=name,
            dtype=dtype,
            device=device,
            sigma_n_num=sigma_n_num,
            norm_coef=norm_coef,
            flg_norm_noise=flg_norm_noise,
        )

    def init_param(self):
        """
        Initialize the RBF parameters
        """
        self.log_sigma_kin_vel_list = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.tensor(
                        np.log(self.init_param_dict["sigma_kin_vel_list_init"][joint_index]),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    requires_grad=self.init_param_dict["flg_train_sigma_kin_vel"],
                )
                for joint_index in range(self.num_dof)
            ]
        )
        self.log_sigma_kin_pos_list = torch.nn.ParameterList([])
        self.log_sigma_pot_list = torch.nn.ParameterList([])
        pos_index = 0
        for joint_index_1 in range(self.num_dof):
            for joint_index_2 in range(joint_index_1 + 1):
                self.log_sigma_kin_pos_list.append(
                    torch.nn.Parameter(
                        torch.tensor(
                            np.log(self.init_param_dict["sigma_kin_pos_list_init"][pos_index]),
                            dtype=self.dtype,
                            device=self.device,
                        ),
                        requires_grad=self.init_param_dict["flg_train_sigma_kin_pos"],
                    )
                )
                pos_index += 1
            self.log_sigma_pot_list.append(
                torch.nn.Parameter(
                    torch.tensor(
                        np.log(self.init_param_dict["sigma_pot_list_init"][joint_index_1]),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    requires_grad=self.init_param_dict["flg_train_sigma_pot"],
                )
            )

    def get_param_dict(self):
        """
        Return a dictionary with the kernel parameters (transfromed if necessary)
        """
        d = {}
        d["sigma_kin_vel_list"] = [torch.exp(p) for p in self.log_sigma_kin_vel_list]
        d["sigma_kin_pos_list"] = [torch.exp(p) for p in self.log_sigma_kin_pos_list]
        d["sigma_pot_list"] = [torch.exp(p) for p in self.log_sigma_pot_list]
        return d


class m_GP_LK_RBF_M(m_GP_Lagrangian_kernel):
    """
    Implementation of the m_GP_Lagrangian_kernel
    with RBF prior on kinetic and potential energy
    """

    def __init__(
        self,
        num_dof,
        pos_indices,
        vel_indices,
        acc_indices,
        init_param_dict,
        f_K_blocks,
        f_K_blocks_ltr,
        f_K_blocks_diag,
        f_K_T_Y_blocks=None,
        f_K_U_Y_blocks=None,
        f_K_L_Y_blocks=None,
        friction_model=None,
        f_phi_friction=None,
        flg_np=False,
        sigma_n_init=None,
        flg_train_sigma_n=True,
        name="",
        dtype=torch.float64,
        device=None,
        sigma_n_num=None,
        norm_coef=None,
        flg_norm_noise=False,
    ):
        super().__init__(
            num_dof=num_dof,
            pos_indices=pos_indices,
            vel_indices=vel_indices,
            acc_indices=acc_indices,
            init_param_dict=init_param_dict,
            f_K_blocks=f_K_blocks,
            f_K_blocks_ltr=f_K_blocks_ltr,
            f_K_blocks_diag=f_K_blocks_diag,
            f_K_T_Y_blocks=f_K_T_Y_blocks,
            f_K_U_Y_blocks=f_K_U_Y_blocks,
            f_K_L_Y_blocks=f_K_L_Y_blocks,
            friction_model=friction_model,
            f_phi_friction=f_phi_friction,
            flg_np=flg_np,
            sigma_n_init=sigma_n_init,
            flg_train_sigma_n=flg_train_sigma_n,
            name=name,
            dtype=dtype,
            device=device,
            sigma_n_num=sigma_n_num,
            norm_coef=norm_coef,
            flg_norm_noise=flg_norm_noise,
        )

    def init_param(self):
        """
        Initialize the RBF parameters
        """
        # -------------- KINETIC ENERGY HYPERPARAMETERS --------------------
        par_lT_list = [
            torch.nn.Parameter(
                torch.tensor(np.log(lenghtscale), dtype=self.dtype, device=self.device),
                requires_grad=self.init_param_dict["flg_train_lengthscales_T"],
            )
            for lenghtscale in self.init_param_dict["lengthscales_T_init_list"]
        ]
        self.log_lengthscales_par_T_list = torch.nn.ParameterList(par_lT_list)
        par_sT_list = [
            torch.nn.Parameter(
                torch.tensor(np.log(s), dtype=self.dtype, device=self.device),
                requires_grad=self.init_param_dict["flg_train_scale_T"],
            )
            for s in self.init_param_dict["scale_T_init_list"]
        ]
        self.scale_log_T_list = torch.nn.ParameterList(par_sT_list)
        # -------------- POTENTIAL ENERGY HYPERPARAMETERS --------------------
        self.log_lengthscales_par_U = torch.nn.Parameter(
            torch.tensor(np.log(self.init_param_dict["lengthscales_U_init"]), dtype=self.dtype, device=self.device),
            requires_grad=self.init_param_dict["flg_train_lengthscales_U"],
        )
        self.scale_log_U = torch.nn.Parameter(
            torch.tensor(np.log(self.init_param_dict["scale_U_init"]), dtype=self.dtype, device=self.device),
            requires_grad=self.init_param_dict["flg_train_scale_U"],
        )

    def get_param_dict(self):
        """
        Return a dictionary with the kernel parameters (transfromed if necessary)
        """
        d = {}
        d["lU"] = torch.exp(self.log_lengthscales_par_U)
        d["sU"] = torch.exp(self.scale_log_U)
        d["lT_list"] = [torch.exp(lT) for lT in self.log_lengthscales_par_T_list]
        d["sT_list"] = [torch.exp(lT) for lT in self.scale_log_T_list]
        return d


class m_GP_LK_POLY_RBF_sum(m_GP_Lagrangian_kernel):
    """
    Implementation of the m_GP_Lagrangian_kernel
    with sum of RBF*POLY prior on kinetic energy
    and RBF prior on potential energy
    """

    def __init__(
        self,
        num_dof,
        pos_indices,
        vel_indices,
        acc_indices,
        init_param_dict,
        f_K_blocks,
        f_K_blocks_ltr,
        f_K_blocks_diag,
        f_K_T_Y_blocks=None,
        f_K_U_Y_blocks=None,
        f_K_L_Y_blocks=None,
        friction_model=None,
        f_phi_friction=None,
        flg_np=False,
        sigma_n_init=None,
        flg_train_sigma_n=True,
        name="",
        dtype=torch.float64,
        device=None,
        sigma_n_num=None,
        norm_coef=None,
        flg_norm_noise=False,
    ):
        super().__init__(
            num_dof=num_dof,
            pos_indices=pos_indices,
            vel_indices=vel_indices,
            acc_indices=acc_indices,
            init_param_dict=init_param_dict,
            f_K_blocks=f_K_blocks,
            f_K_blocks_ltr=f_K_blocks_ltr,
            f_K_blocks_diag=f_K_blocks_diag,
            f_K_T_Y_blocks=f_K_T_Y_blocks,
            f_K_U_Y_blocks=f_K_U_Y_blocks,
            f_K_L_Y_blocks=f_K_L_Y_blocks,
            friction_model=friction_model,
            f_phi_friction=f_phi_friction,
            flg_np=flg_np,
            sigma_n_init=sigma_n_init,
            flg_train_sigma_n=flg_train_sigma_n,
            name=name,
            dtype=dtype,
            device=device,
            sigma_n_num=sigma_n_num,
            norm_coef=norm_coef,
            flg_norm_noise=flg_norm_noise,
        )

    def init_param(self):
        """
        Initialize the POLY-RBF parameters
        """
        # -------------- KINETIC ENERGY HYPERPARAMETERS --------------------
        self.log_sigma_kin_vel_list = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.tensor(
                        np.log(self.init_param_dict["sigma_kin_vel_list_init"][joint_index]),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    requires_grad=self.init_param_dict["flg_train_sigma_kin_vel"],
                )
                for joint_index in range(self.num_dof)
            ]
        )
        self.log_lengthscales_par_T_list = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.tensor(
                        np.log(self.init_param_dict["lengthscales_T_list_init"][joint_index]),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    requires_grad=self.init_param_dict["flg_train_lengthscales_T"],
                )
                for joint_index in range(self.num_dof)
            ]
        )
        self.scale_log_T = torch.nn.Parameter(
            torch.tensor(np.log(self.init_param_dict["scale_T_init"]), dtype=self.dtype, device=self.device),
            requires_grad=self.init_param_dict["flg_train_scale_T"],
        )
        # -------------- POTENTIAL ENERGY HYPERPARAMETERS --------------------
        self.log_lengthscales_par_U = torch.nn.Parameter(
            torch.tensor(np.log(self.init_param_dict["lengthscales_U_init"]), dtype=self.dtype, device=self.device),
            requires_grad=self.init_param_dict["flg_train_lengthscales_U"],
        )
        self.scale_log_U = torch.nn.Parameter(
            torch.tensor(np.log(self.init_param_dict["scale_U_init"]), dtype=self.dtype, device=self.device),
            requires_grad=self.init_param_dict["flg_train_scale_U"],
        )

    def get_param_dict(self):
        """
        Return a dictionary with the kernel parameters (transfromed if necessary)
        """
        d = {}
        d["lT_list"] = [torch.exp(p) for p in self.log_lengthscales_par_T_list]
        d["lU"] = torch.exp(self.log_lengthscales_par_U)
        d["sT"] = torch.exp(self.scale_log_T)
        d["sU"] = torch.exp(self.scale_log_U)
        d["sigmaT_list"] = [torch.exp(p) for p in self.log_sigma_kin_vel_list]
        return d


def f_phi_friction_basic(dq1):
    """
    computes the kernel of the basic friction model
    tau_frict = -b*dq -c*sign(dq)
    """
    return torch.cat([dq1, torch.sign(dq1)], 1)


def f_phi_friction_basic_with_offset(dq1):
    """
    computes the kernel of the basic friction model
    tau_frict = -b*dq -c*sign(dq) -const
    """
    return torch.cat([dq1, torch.sign(dq1), torch.ones_like(dq1)], 1)


def get_weigted_distances(X1, X2, lengthscale):
    """
    Computes (X1-X2)^T*sigma^-2*(X1-X2),
    where Sigma = diag(l)
    """
    # N1 = X1.size()
    # slice the inputs and get the weighted distances
    X1_norm = X1 / lengthscale
    X1_squared = torch.sum(X1_norm.mul(X1_norm), dim=1, keepdim=True)
    X2_norm = X2 / lengthscale
    X2_squared = torch.sum(X2_norm.mul(X2_norm), dim=1, keepdim=True)
    dist = (
        X1_squared + X2_squared.transpose(dim0=0, dim1=1) - 2 * torch.matmul(X1_norm, X2_norm.transpose(dim0=0, dim1=1))
    )
    return dist
