"""
Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
This file contains the definition of GIP kernel.
"""


import numpy as np
import torch

from . import GP_prior


class GIP_GP(GP_prior.GP_prior):
    """
    Implementation of the GIP kernel, a polynomial kernel for inverse dynamics identification
    """

    def __init__(
        self,
        active_dims,
        num_dof,
        q_dims,
        dq_dims,
        ddq_dims,
        rev_indices=None,
        prism_indices=None,
        sigma_n_init=None,
        flg_train_sigma_n=False,
        f_mean=None,
        f_mean_add_par_dict={},
        pos_par_mean_init=None,
        flg_train_pos_par_mean=False,
        free_par_mean_init=None,
        flg_train_free_par_mean=False,
        Sigma_par_acc_init=None,
        flg_train_Sigma_par_acc=True,
        Sigma_par_vel1_init=None,
        Sigma_par_vel2_init=None,
        Sigma_par_vel3_init=None,
        flg_train_Sigma_par_vel=True,
        Sigma_par_pos_rev1_init=None,
        Sigma_par_pos_rev2_init=None,
        Sigma_par_pos_rev3_init=None,
        flg_train_Sigma_par_pos_rev=True,
        Sigma_par_pos_prism1_init=None,
        Sigma_par_pos_prism2_init=None,
        Sigma_par_pos_prism3_init=None,
        flg_train_Sigma_par_pos_prism=True,
        scale_init=np.ones(1),
        flg_train_scale=False,
        name="",
        dtype=torch.float64,
        sigma_n_num=None,
        device=None,
    ):
        """
        Initialization of the object:
        - mean_init and Sigma_function define the prior on the model parameters w
        - Sigma_pos_par Sigma_free par are, respectively, the positive and free parameters of the w prior
        - if flg_offset is true a constant feature equal to 1 is added to the input X
        - active_dims define the input dimension interested i.e. the phi function
        """
        # initilize the GP object
        super().__init__(
            active_dims,
            sigma_n_init=sigma_n_init,
            flg_train_sigma_n=flg_train_sigma_n,
            scale_init=scale_init,
            flg_train_scale=flg_train_scale,
            f_mean=f_mean,
            f_mean_add_par_dict=f_mean_add_par_dict,
            pos_par_mean_init=pos_par_mean_init,
            flg_train_pos_par_mean=flg_train_pos_par_mean,
            free_par_mean_init=free_par_mean_init,
            flg_train_free_par_mean=flg_train_free_par_mean,
            name=name,
            dtype=dtype,
            sigma_n_num=sigma_n_num,
            device=device,
        )
        # save robot parameters
        self.num_dof = num_dof
        self.rev_indices = rev_indices
        self.prism_indices = prism_indices
        # if rev_indices is None:
        #     self.rev_indices = []
        # else:
        #     self.rev_indices = rev_indices
        # if prism_indices is None:
        #     self.prism_indices = []
        # else:
        #     self.prism_indices = prism_indices
        self.q_dims = q_dims
        self.dq_dims = dq_dims
        self.ddq_dims = ddq_dims
        # init Sigma parameters
        self.Sigma_par_acc_log = torch.nn.Parameter(
            torch.tensor(np.log(Sigma_par_acc_init), dtype=self.dtype, device=self.device),
            requires_grad=flg_train_Sigma_par_acc,
        )
        self.Sigma_par_vel1_log = torch.nn.Parameter(
            torch.tensor(np.log(Sigma_par_vel1_init), dtype=self.dtype, device=self.device),
            requires_grad=flg_train_Sigma_par_vel,
        )
        self.Sigma_par_vel2_log = torch.nn.Parameter(
            torch.tensor(np.log(Sigma_par_vel2_init), dtype=self.dtype, device=self.device),
            requires_grad=flg_train_Sigma_par_vel,
        )
        self.Sigma_par_vel3_log = torch.nn.Parameter(
            torch.tensor(np.log(Sigma_par_vel3_init), dtype=self.dtype, device=self.device),
            requires_grad=flg_train_Sigma_par_vel,
        )
        if rev_indices is not None:
            self.Sigma_par_pos_rev1_log = torch.nn.Parameter(
                torch.tensor(np.log(Sigma_par_pos_rev1_init), dtype=self.dtype, device=self.device),
                requires_grad=flg_train_Sigma_par_pos_rev,
            )
            self.Sigma_par_pos_rev2_log = torch.nn.Parameter(
                torch.tensor(np.log(Sigma_par_pos_rev2_init), dtype=self.dtype, device=self.device),
                requires_grad=flg_train_Sigma_par_pos_rev,
            )
            self.Sigma_par_pos_rev3_log = torch.nn.Parameter(
                torch.tensor(np.log(Sigma_par_pos_rev3_init), dtype=self.dtype, device=self.device),
                requires_grad=flg_train_Sigma_par_pos_rev,
            )
        if prism_indices is not None:
            self.Sigma_par_pos_prism1_log = torch.nn.Parameter(
                torch.tensor(np.log(Sigma_par_pos_prism1_init), dtype=self.dtype, device=self.device),
                requires_grad=flg_train_Sigma_par_pos_prism,
            )
            self.Sigma_par_pos_prism2_log = torch.nn.Parameter(
                torch.tensor(np.log(Sigma_par_pos_prism2_init), dtype=self.dtype, device=self.device),
                requires_grad=flg_train_Sigma_par_pos_prism,
            )
            self.Sigma_par_pos_prism3_log = torch.nn.Parameter(
                torch.tensor(np.log(Sigma_par_pos_prism3_init), dtype=self.dtype, device=self.device),
                requires_grad=flg_train_Sigma_par_pos_prism,
            )

    def get_covariance(self, X1, X2=None, flg_noise=False, p_drop=0.0):
        """
        Returns the GIP covariance
        (K_acc+K_vel)*K_pos
        """
        # get phi matrices
        N1 = X1.shape[0]
        phi_acc_X1 = torch.cat([X1[:, self.ddq_dims], torch.ones([N1, 1], dtype=self.dtype, device=self.device)], 1)
        Sigma_acc = torch.diag(torch.exp(self.Sigma_par_acc_log))
        phi_vel_X1 = X1[:, self.dq_dims]
        Sigma_vel1 = torch.diag(torch.exp(self.Sigma_par_vel1_log))
        Sigma_vel2 = torch.diag(torch.exp(self.Sigma_par_vel2_log))
        Sigma_vel3 = torch.diag(torch.exp(self.Sigma_par_vel3_log))
        if self.rev_indices is not None:
            phi_pos_rev_X1 = torch.cat(
                [
                    torch.cat(
                        [
                            torch.sin(X1[:, self.q_dims[q_dim_i] : self.q_dims[q_dim_i] + 1]),
                            torch.cos(X1[:, self.q_dims[q_dim_i] : self.q_dims[q_dim_i] + 1]),
                            torch.ones([N1, 1], dtype=self.dtype, device=self.device),
                        ],
                        1,
                    ).unsqueeze(0)
                    for q_dim_i in self.rev_indices
                ],
                0,
            )
            Sigma_pos_rev1 = torch.cat(
                [
                    torch.diag(torch.exp(self.Sigma_par_pos_rev1_log[q_dim_i])).unsqueeze(0)
                    for q_dim_i in self.rev_indices
                ],
                0,
            )
            Sigma_pos_rev2 = torch.cat(
                [
                    torch.diag(torch.exp(self.Sigma_par_pos_rev2_log[q_dim_i])).unsqueeze(0)
                    for q_dim_i in self.rev_indices
                ],
                0,
            )
            Sigma_pos_rev3 = torch.cat(
                [
                    torch.diag(torch.exp(self.Sigma_par_pos_rev3_log[q_dim_i])).unsqueeze(0)
                    for q_dim_i in self.rev_indices
                ],
                0,
            )
        if self.prism_indices is not None:
            phi_pos_prism_X1 = torch.cat(
                [
                    torch.cat(
                        [
                            X1[:, self.q_dims[q_dim_i] : self.q_dims[q_dim_i] + 1],
                            torch.ones([N1, 1], dtype=self.dtype, device=self.device),
                        ],
                        1,
                    ).unsqueeze(0)
                    for q_dim_i in self.prism_indices
                ],
                0,
            )
            Sigma_pos_prism1 = torch.cat(
                [
                    torch.diag(torch.exp(self.Sigma_par_pos_prism1_log[q_dim_i])).unsqueeze(0)
                    for q_dim_i in self.prism_indices
                ],
                0,
            )
            Sigma_pos_prism2 = torch.cat(
                [
                    torch.diag(torch.exp(self.Sigma_par_pos_prism2_log[q_dim_i])).unsqueeze(0)
                    for q_dim_i in self.prism_indices
                ],
                0,
            )
            Sigma_pos_prism3 = torch.cat(
                [
                    torch.diag(torch.exp(self.Sigma_par_pos_prism3_log[q_dim_i])).unsqueeze(0)
                    for q_dim_i in self.prism_indices
                ],
                0,
            )
        if X2 is None:
            K_acc = torch.matmul(phi_acc_X1, torch.matmul(Sigma_acc, phi_acc_X1.transpose(1, 0)))
            K_vel = torch.matmul(phi_vel_X1, torch.matmul(Sigma_vel1, phi_vel_X1.transpose(1, 0))) * torch.matmul(
                phi_vel_X1, torch.matmul(Sigma_vel2, phi_vel_X1.transpose(1, 0))
            )
            K_pos = 1.0
            if self.rev_indices is not None:
                K_pos = K_pos * torch.prod(
                    torch.matmul(phi_pos_rev_X1, torch.matmul(Sigma_pos_rev1, phi_pos_rev_X1.transpose(2, 1)))
                    + torch.matmul(
                        phi_pos_rev_X1[:, :, 0:2],
                        torch.matmul(Sigma_pos_rev2, phi_pos_rev_X1[:, :, 0:2].transpose(2, 1)),
                    )
                    * torch.matmul(
                        phi_pos_rev_X1[:, :, 0:2],
                        torch.matmul(Sigma_pos_rev3, phi_pos_rev_X1[:, :, 0:2].transpose(2, 1)),
                    ),
                    0,
                )
            if self.prism_indices is not None:
                K_pos = K_pos * torch.prod(
                    torch.matmul(phi_pos_prism_X1, torch.matmul(Sigma_pos_prism1, phi_pos_prism_X1.transpose(2, 1)))
                    + torch.matmul(
                        phi_pos_prism_X1[:, :, 0:1],
                        torch.matmul(Sigma_pos_prism2, phi_pos_prism_X1[:, :, 0:1].transpose(2, 1)),
                    )
                    * torch.matmul(
                        phi_pos_prism_X1[:, :, 0:1],
                        torch.matmul(Sigma_pos_prism3, phi_pos_prism_X1[:, :, 0:1].transpose(2, 1)),
                    ),
                    0,
                )
            if flg_noise & self.GP_with_noise:
                return (K_acc + K_vel) * K_pos + self.get_sigma_n_2() * torch.eye(
                    N1, dtype=self.dtype, device=self.device
                )
            else:
                return (K_acc + K_vel) * K_pos
        else:
            N2 = X2.shape[0]
            phi_acc_X2 = torch.cat([X2[:, self.ddq_dims], torch.ones([N2, 1], dtype=self.dtype, device=self.device)], 1)
            phi_vel_X2 = X2[:, self.dq_dims]
            if self.rev_indices is not None:
                phi_pos_rev_X2 = torch.cat(
                    [
                        torch.cat(
                            [
                                torch.sin(X2[:, self.q_dims[q_dim_i] : self.q_dims[q_dim_i] + 1]),
                                torch.cos(X2[:, self.q_dims[q_dim_i] : self.q_dims[q_dim_i] + 1]),
                                torch.ones([N2, 1], dtype=self.dtype, device=self.device),
                            ],
                            1,
                        ).unsqueeze(0)
                        for q_dim_i in self.rev_indices
                    ],
                    0,
                )
            if self.prism_indices is not None:
                phi_pos_prism_X2 = torch.cat(
                    [
                        torch.cat(
                            [
                                X2[:, self.q_dims[q_dim_i] : self.q_dims[q_dim_i] + 1],
                                torch.ones([N2, 1], dtype=self.dtype, device=self.device),
                            ],
                            1,
                        ).unsqueeze(0)
                        for q_dim_i in self.prism_indices
                    ],
                    0,
                )
            K_acc = torch.matmul(phi_acc_X1, torch.matmul(Sigma_acc, phi_acc_X2.transpose(1, 0)))
            K_vel = torch.matmul(phi_vel_X1, torch.matmul(Sigma_vel1, phi_vel_X2.transpose(1, 0))) * torch.matmul(
                phi_vel_X1, torch.matmul(Sigma_vel2, phi_vel_X2.transpose(1, 0))
            )
            K_pos = 1.0
            if self.rev_indices is not None:
                K_pos = K_pos * torch.prod(
                    torch.matmul(phi_pos_rev_X1, torch.matmul(Sigma_pos_rev1, phi_pos_rev_X2.transpose(2, 1)))
                    + torch.matmul(
                        phi_pos_rev_X1[:, :, 0:2],
                        torch.matmul(Sigma_pos_rev2, phi_pos_rev_X2[:, :, 0:2].transpose(2, 1)),
                    )
                    * torch.matmul(
                        phi_pos_rev_X1[:, :, 0:2],
                        torch.matmul(Sigma_pos_rev3, phi_pos_rev_X2[:, :, 0:2].transpose(2, 1)),
                    ),
                    0,
                )
            if self.prism_indices is not None:
                K_pos = K_pos * torch.prod(
                    torch.matmul(phi_pos_prism_X1, torch.matmul(Sigma_pos_prism1, phi_pos_prism_X2.transpose(2, 1)))
                    + torch.matmul(
                        phi_pos_prism_X1[:, :, 0:1],
                        torch.matmul(Sigma_pos_prism2, phi_pos_prism_X2[:, :, 0:1].transpose(2, 1)),
                    )
                    * torch.matmul(
                        phi_pos_prism_X1[:, :, 0:1],
                        torch.matmul(Sigma_pos_prism3, phi_pos_prism_X2[:, :, 0:1].transpose(2, 1)),
                    ),
                    0,
                )
            return (K_acc + K_vel) * K_pos

    def get_diag_covariance(self, X, flg_noise=False):
        """
        Returns the diag of the cov matrix
        """
        return torch.diag(self.get_covariance(X, flg_noise=flg_noise))
