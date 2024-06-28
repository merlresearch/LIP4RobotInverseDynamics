# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Superclass of robot inverse dynamics estimator

Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
"""
import numpy as np
import torch

import gpr_lib.GP_prior.GIP as GIP
import gpr_lib.GP_prior.GP_prior as GP_prior
import gpr_lib.GP_prior.Sparse_GP as SGP
import gpr_lib.GP_prior.Stationary_GP as Stat_GP
import gpr_lib.Utils.Parameters_covariance_functions as cov


class m_independent_joint(torch.nn.Module):
    """
    Model that considers each joint standalone
    """

    def __init__(self, num_dof, name="", dtype=torch.float64, device=None):
        # init torch module
        super().__init__()
        # save perameters
        self.num_dof = num_dof
        self.name = name
        self.dtype = dtype
        self.device = device
        self.models_list = []

    def to(self, dev):
        """
        Move model to device
        """
        super().to(dev)
        self.device = dev
        for model in self.models_list:
            model.to(dev)

    def print_model(self):
        """
        Print the model
        """
        for joint_index, model in enumerate(self.models_list):
            print("Models joint torque " + str(joint_index + 1) + ":")
            model.print_model()

    def init_joint_torque_model(self, joint_index):
        """
        Init the model of the joint_index torque
        """
        raise NotImplementedError

    def train_model(
        self,
        joint_index_list,
        X,
        Y,
        criterion,
        f_optimizer,
        batch_size,
        shuffle,
        N_epoch,
        N_epoch_print,
        p_drop=0.0,
        additional_par_dict={},
        indices_list=None,
        drop_last=False,
    ):
        """
        Train the torque models
        """
        if indices_list is None:
            indices_list = [range(0, X.shape[0]) for joint_index in range(len(joint_index_list))]
        for joint_index in joint_index_list:
            print("\nJoint " + str(joint_index + 1) + " training:")
            print("Num training data: " + str(len(indices_list[joint_index])))
            self.train_joint_model(
                joint_index=joint_index,
                X=X[indices_list[joint_index], :],
                Y=Y[indices_list[joint_index], joint_index].reshape([-1, 1]),
                criterion=criterion,
                f_optimizer=f_optimizer,
                batch_size=batch_size,
                shuffle=shuffle,
                N_epoch=N_epoch,
                N_epoch_print=N_epoch_print,
                p_drop=p_drop,
                drop_last=drop_last,
                **additional_par_dict
            )

    def train_joint_model(
        self,
        joint_index,
        X,
        Y,
        criterion,
        f_optimizer,
        batch_size,
        shuffle,
        N_epoch,
        N_epoch_print,
        p_drop=0.0,
        drop_last=False,
    ):
        """
        Train the joint_index model
        """
        # get the dataloader
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, requires_grad=False, dtype=self.dtype, device=self.device),
            torch.tensor(Y, requires_grad=False, dtype=self.dtype, device=self.device),
        )
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        # get the optimizer
        optimizer = f_optimizer(self.models_list[joint_index].parameters())
        # trian the model
        for epoch in range(N_epoch):
            running_loss = 0.0
            N_btc = 0
            # print('\nEPOCH:', epoch)
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                out = self.models_list[joint_index](inputs)
                loss = criterion(out, labels)
                loss.backward(retain_graph=True)
                optimizer.step()
                running_loss = running_loss + loss.item()
                N_btc = N_btc + 1
            if epoch % N_epoch_print == 0:
                print("\nEPOCH:", epoch)
                print("Runnung loss:", running_loss / N_btc)

    def get_torque_estimates(self, X_test, joint_indices_list=None):
        """
        Returns the joint torques estimate
        """
        # check the joint to test
        if joint_indices_list is None:
            joint_indices_list = range(0, self.num_dof)
        # initialize the outputs
        estimate_list = []
        # get the estimate for each joint torques
        for i, joint_index in enumerate(joint_indices_list):
            print("\nJoint " + str(joint_index + 1) + " estimate:")
            estimate = self.get_joint_torque_estimate(
                joint_index=joint_index,
                X_test=torch.tensor(X_test, dtype=self.dtype, device=self.device, requires_grad=False),
            )
            estimate_list.append(estimate.detach().cpu().numpy())
        return estimate_list

    def get_joint_torque_estimate(self, joint_index, X_test):
        """
        Return the estimate of the joint_index torque
        """
        return self.models_list[joint_index](X_test)


##############################
# ----------GP MODELS---------#
##############################


class m_indep_GP(m_independent_joint):
    """
    Superclass of the models based on GP
    """

    def __init__(
        self,
        num_dof,
        active_dims_list,
        sigma_n_init_list=None,
        flg_train_sigma_n=True,
        pos_indices=None,
        acc_indices=None,
        name="",
        dtype=torch.float64,
        device=None,
        max_input_loc=100000,
        downsampling_mode="Downsampling",
        sigma_n_num=None,
    ):
        super().__init__(num_dof=num_dof, name=name, dtype=dtype, device=device)
        self.active_dims_list = active_dims_list
        self.sigma_n_init_list = sigma_n_init_list
        self.flg_train_sigma_n = flg_train_sigma_n
        self.max_input_loc = max_input_loc
        self.downsampling_mode = downsampling_mode
        self.sigma_n_num = sigma_n_num
        self.pos_indices = pos_indices
        self.acc_indices = acc_indices

    def train_joint_model(
        self,
        joint_index,
        X,
        Y,
        criterion,
        f_optimizer,
        batch_size,
        shuffle,
        N_epoch,
        N_epoch_print,
        p_drop=0.0,
        drop_last=False,
    ):
        # get the dataloader
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, requires_grad=False, dtype=self.dtype, device=self.device),
            torch.tensor(Y, requires_grad=False, dtype=self.dtype, device=self.device),
        )
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        # fit the model
        self.models_list[joint_index].fit_model(
            trainloader=trainloader,
            optimizer=f_optimizer(self.models_list[joint_index].parameters()),
            criterion=criterion,
            N_epoch=N_epoch,
            N_epoch_print=N_epoch_print,
            f_saving_model=None,
            f_print=None,
            p_drop=p_drop,
        )
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def get_torque_estimates(
        self,
        X,
        Y,
        X_test,
        alpha_list_par=None,
        m_X_list_par=None,
        K_X_inv_list_par=None,
        flg_return_K_X_inv=False,
        indices_list=None,
        joint_indices_list=None,
    ):
        """
        Performs the estimate for each joint torques
        """
        if joint_indices_list is None:
            joint_indices_list = range(0, self.num_dof)
        # check alpha_list and K_X_inv_inv
        if alpha_list_par is None:
            alpha_list_par = [None for i in range(0, self.num_dof)]
        if K_X_inv_list_par is None:
            K_X_inv_list_par = [None for i in range(0, self.num_dof)]
        if m_X_list_par is None:
            m_X_list_par = [None for i in range(0, self.num_dof)]
        # check the number of data and if required downsample
        if indices_list is None:
            if X.shape[0] > self.max_input_loc:
                indices_list = self.downsample_data(X, Y)
            else:
                indices_list = [range(0, X.shape[0]) for joint_index in range(0, self.num_dof)]
        # initialize the outputs
        estimate_list = []
        var_list = []
        alpha_list = []
        m_X_list = []
        K_X_inv_list = []
        # get the estimate for each link
        for i, joint_index in enumerate(joint_indices_list):
            # print('\nLink '+str(joint_index+1)+' estimate:')
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            X_tc = torch.tensor(
                X[indices_list[joint_index], :], dtype=self.dtype, device=self.device, requires_grad=False
            )
            Y_tc = torch.tensor(
                Y[indices_list[joint_index], joint_index].reshape([-1, 1]),
                dtype=self.dtype,
                device=self.device,
                requires_grad=False,
            )
            X_test_tc = torch.tensor(X_test, dtype=self.dtype, device=self.device, requires_grad=False)
            estimate, var, alpha, m_X, K_X_inv = self.get_joint_torque_estimate(
                joint_index=joint_index,
                X=X_tc,
                Y=Y_tc,
                X_test=X_test_tc,
                flg_return_K_X_inv=flg_return_K_X_inv,
                alpha_par=alpha_list_par[joint_index],
                m_X_par=m_X_list_par[joint_index],
                K_X_inv_par=K_X_inv_list_par[joint_index],
            )
            estimate_list.append(estimate.detach().cpu().numpy())
            if var is not None:
                var_list.append(var.detach().cpu().numpy().reshape([-1, 1]))
            else:
                var_list.append(var)
            alpha_list.append(alpha)
            m_X_list.append(m_X)
            K_X_inv_list.append(K_X_inv)
        return estimate_list, var_list, alpha_list, m_X_list, K_X_inv_list, indices_list

    def get_joint_torque_estimate(
        self, joint_index, X, Y, X_test, flg_return_K_X_inv=False, alpha_par=None, m_X_par=None, K_X_inv_par=None
    ):
        """
        Return the estimate of the joint_index torque
        """
        if alpha_par is None:
            if flg_return_K_X_inv:
                return self.models_list[joint_index].get_estimate(
                    X=X, Y=Y, X_test=X_test, Y_test=None, flg_return_K_X_inv=flg_return_K_X_inv
                )
            else:
                Y_hat, var, alpha, m_X = self.models_list[joint_index].get_estimate(
                    X=X, Y=Y, X_test=X_test, Y_test=None, flg_return_K_X_inv=flg_return_K_X_inv
                )
        else:
            if K_X_inv_par is None:
                Y_hat, _, m_X = self.models_list[joint_index].get_estimate_from_alpha(
                    X=X, X_test=X_test, alpha=alpha_par, m_X=m_X_par, K_X_inv=K_X_inv_par, Y_test=None
                )
                var = None
                alpha = None
            else:
                Y_hat, var, m_X = self.models_list[joint_index].get_estimate_from_alpha(
                    X=X, X_test=X_test, alpha=alpha_par, m_X=m_X_par, K_X_inv=K_X_inv_par, Y_test=None
                )
                alpha = None
        K_X_inv = None
        # m_X = None
        return Y_hat, var, alpha, m_X, K_X_inv

    def downsample_data(self, X, Y):
        """
        Downsample data
        """
        if self.downsampling_mode == "Downsampling":
            print("Downsampling....")
            num_sample = X.shape[0]
            downsampling_step = int(num_sample / self.max_input_loc)
            indices = range(0, num_sample, downsampling_step)
            indices_list = [indices for joint_index in range(0, self.num_dof)]
        elif self.downsampling_mode == "Random":
            print("Random downsampling")
            num_sample = X.shape[0]
            indices = np.random.sample(range(0, num_sample), self.max_input_loc)
            indices_list = [indices for joint_index in range(0, self.num_dof)]
        return indices_list

    def get_M_estimates(self, X_tr, X_test, alpha_par_list, norm_coef):
        """
        Returns and estimate of the model inertia matrix related to the configurations
        defined by the input locations in X
        """
        M_list = []
        # get the input locations with acc and vel null
        X_grav = torch.zeros_like(X_test)
        X_grav[:, self.pos_indices] = X_test[:, self.pos_indices]
        for joint_index in range(0, self.num_dof):
            M_joint = []
            # get the contribution due to gravity
            g, _, _, _, _ = self.get_joint_torque_estimate(
                joint_index=joint_index,
                X=X_tr,
                Y=None,
                X_test=X_grav,
                flg_return_K_X_inv=False,
                alpha_par=alpha_par_list[joint_index],
            )
            for M_index in range(0, self.num_dof):
                # get gravity contribution + M[joint_index,M_index]
                X_acc = X_grav.clone()
                X_acc[:, self.acc_indices[M_index]] = 1.0
                Mg, _, _, _, _ = self.get_joint_torque_estimate(
                    joint_index=joint_index,
                    X=X_tr,
                    Y=None,
                    X_test=X_acc,
                    flg_return_K_X_inv=False,
                    alpha_par=alpha_par_list[joint_index],
                )
                M_joint.append(norm_coef[joint_index] * (Mg - g).reshape([-1, 1, 1]))
            M_list.append(torch.cat(M_joint, 2))
        M = torch.cat(M_list, 1)
        return M

    def get_m_estimates(self, X_tr, X_test, alpha_par_list):
        """
        Returns and estimate of the the sum of inertial contribution defined by the input locations in X
        """
        m_list = []
        X_tr = torch.tensor(X_tr, dtype=self.dtype, device=self.device, requires_grad=False)
        X_test = torch.tensor(X_test, dtype=self.dtype, device=self.device, requires_grad=False)
        X_m = torch.zeros_like(X_test)
        X_m[:, self.acc_indices] = X_test[:, self.acc_indices]
        X_m[:, self.pos_indices] = X_test[:, self.pos_indices]
        X_grav = torch.zeros_like(X_test)
        X_grav[:, self.pos_indices] = X_test[:, self.pos_indices]
        for joint_index in range(0, self.num_dof):
            mg, _, _, _, _ = self.get_joint_torque_estimate(
                joint_index=joint_index,
                X=X_tr,
                Y=None,
                X_test=X_m,
                flg_return_K_X_inv=False,
                alpha_par=alpha_par_list[joint_index],
            )
            g, _, _, _, _ = self.get_joint_torque_estimate(
                joint_index=joint_index,
                X=X_tr,
                Y=None,
                X_test=X_grav,
                flg_return_K_X_inv=False,
                alpha_par=alpha_par_list[joint_index],
            )
            m_list.append((mg - g).detach().cpu().numpy())
        return m_list

    def get_c_estimates(self, X_tr, X_test, alpha_par_list):
        """
        Returns and estimate of the the sum of coriolis contribution defined by the input locations in X
        """
        c_list = []
        X_tr = torch.tensor(X_tr, dtype=self.dtype, device=self.device, requires_grad=False)
        X_test = torch.tensor(X_test, dtype=self.dtype, device=self.device, requires_grad=False)
        X_grav = torch.zeros_like(X_test)
        X_grav[:, self.pos_indices] = X_test[:, self.pos_indices]
        X_test[:, self.acc_indices] = 0
        for joint_index in range(0, self.num_dof):
            g, _, _, _, _ = self.get_joint_torque_estimate(
                joint_index=joint_index,
                X=X_tr,
                Y=None,
                X_test=X_grav,
                flg_return_K_X_inv=False,
                alpha_par=alpha_par_list[joint_index],
            )
            cg, _, _, _, _ = self.get_joint_torque_estimate(
                joint_index=joint_index,
                X=X_tr,
                Y=None,
                X_test=X_test,
                flg_return_K_X_inv=False,
                alpha_par=alpha_par_list[joint_index],
            )
            c_list.append((cg - g).detach().cpu().numpy())
        return c_list

    def get_g_estimates(self, X_tr, X_test, alpha_par_list):
        """
        Returns and estimate of the the grav contribution defined by the input locations in X
        """
        g_list = []
        X_tr = torch.tensor(X_tr, dtype=self.dtype, device=self.device, requires_grad=False)
        X_test = torch.tensor(X_test, dtype=self.dtype, device=self.device, requires_grad=False)
        X_grav = torch.zeros_like(X_test)
        X_grav[:, self.pos_indices] = X_test[:, self.pos_indices]
        for joint_index in range(0, self.num_dof):
            g, _, _, _, _ = self.get_joint_torque_estimate(
                joint_index=joint_index,
                X=X_tr,
                Y=None,
                X_test=X_grav,
                flg_return_K_X_inv=False,
                alpha_par=alpha_par_list[joint_index],
            )
            g_list.append(g.detach().cpu().numpy())
        return g_list

    def get_cg_estimates(self, X_tr, X_test, alpha_par_list, norm_coef, mean_coef=None):
        """
        Returns and estimate of the the sum of coriolis contribution + grav contribution defined by the input locations
        in X
        """
        if mean_coef is None:
            mean_coef = np.zeros(len(norm_coef))
        cg_list = []
        X_test[:, self.acc_indices] = 0
        for joint_index in range(0, self.num_dof):
            cg, _, _, _, _ = self.get_joint_torque_estimate(
                joint_index=joint_index,
                X=X_tr,
                Y=None,
                X_test=X_test,
                flg_return_K_X_inv=False,
                alpha_par=alpha_par_list[joint_index],
            )
            cg_list.append(norm_coef[joint_index] * cg + mean_coef[joint_index])
        cg_estimates = torch.cat(cg_list, 1)
        return cg_estimates

    def get_acc_estimates(self, X_tr, X_test, alpha_par_list, tau_test, norm_coef, mean_coef=None):
        """
        Returns the acc estimates computed as inv(M)*(tau-cg)
        tau_test.shape = [N,self.num_dof]
        """
        if mean_coef is None:
            mean_coef = torch.zeros(len(norm_coef), device=self.device, dtype=self.dtype)
        N = X_test.shape[0]

        # Compute M
        M_hat = self.get_M_estimates(
            X_tr=X_tr, X_test=X_test, alpha_par_list=alpha_par_list, norm_coef=norm_coef
        )  # shape = [N, self.num_dof, self.num_dof]
        # Compute cg
        cg_hat = self.get_cg_estimates(
            X_tr=X_tr, X_test=X_test, alpha_par_list=alpha_par_list, norm_coef=norm_coef, mean_coef=mean_coef
        ).reshape(
            N, self.num_dof, 1
        )  # shape = [N, self.num_dof,1]
        # Compute acc
        # norm_coef = np.array(norm_coef).reshape(self.num_dof)
        q_ddot_hat = torch.linalg.solve(
            M_hat, -cg_hat + (norm_coef * tau_test + mean_coef).reshape([N, self.num_dof, 1])
        ).reshape(N, self.num_dof)

        return M_hat, cg_hat, q_ddot_hat

    def get_covariance_matrix(self, input_data, X_tr):
        """
        Function that computes covanriance between input_data and X_tr:
        -input_data: numpy array with [q, q_dot, q_ddot]
        -X_tr: training inputs
        output:
        -covariance_matrix: numpy matrix with [cov_joint_1;...cov_joint_2]
        """
        # map input_data in GP input_data
        GP_input = self.data2GP_input(torch.tensor(input_data, dtype=self.dtype, device=self.device))
        # get covarince_matrix
        return (
            torch.cat([self.models_list[i].get_covariance(GP_input, X_tr) for i in range(0, self.num_dof)], 0)
            .detach()
            .cpu()
            .numpy()
        )

    def data2GP_input(self, input_data):
        """
        Maps input data in GP_input
        input_data: [q,q_dot,q_ddot]
        """
        return input_data


class m_ind_RBF(m_indep_GP):
    """
    Model that considers each link standalone
    and models each gp with a RBF kernel
    """

    def __init__(
        self,
        num_dof,
        active_dims_list,
        sigma_n_init_list=None,
        flg_train_sigma_n=True,
        f_mean_list=None,
        f_mean_add_par_dict_list=None,
        pos_par_mean_init_list=None,
        flg_train_pos_par_mean=False,
        free_par_mean_init_list=None,
        flg_train_free_par_mean=False,
        lengthscales_init_list=None,
        flg_train_lengthscales_list=True,
        lambda_init_list=None,
        flg_train_lambda_list=True,
        mean_init_list=None,
        flg_train_mean_list=False,
        pos_indices=None,
        acc_indices=None,
        norm_coef_input=None,
        name="RBF",
        dtype=torch.float64,
        device=None,
        max_input_loc=100000,
        downsampling_mode="Downsampling",
        sigma_n_num=None,
    ):
        # initialize the superclass
        super().__init__(
            num_dof=num_dof,
            active_dims_list=active_dims_list,
            sigma_n_init_list=sigma_n_init_list,
            flg_train_sigma_n=flg_train_sigma_n,
            pos_indices=pos_indices,
            acc_indices=acc_indices,
            name=name,
            dtype=dtype,
            device=device,
            max_input_loc=max_input_loc,
            downsampling_mode=downsampling_mode,
            sigma_n_num=sigma_n_num,
        )
        # save model parameters
        self.lengthscales_init_list = lengthscales_init_list
        self.lambda_init_list = lambda_init_list
        self.mean_init_list = mean_init_list
        self.flg_train_lengthscales_list = flg_train_lengthscales_list
        self.flg_train_lambda_list = flg_train_lambda_list
        self.flg_train_mean_list = flg_train_mean_list
        self.norm_coef_input = norm_coef_input
        if f_mean_list is None:
            f_mean_list = [None] * num_dof
            f_mean_add_par_dict_list = [None] * num_dof
            pos_par_mean_init_list = [None] * num_dof
            free_par_mean_init_list = [None] * num_dof
            flg_train_pos_par_mean = [False] * num_dof
            flg_train_free_par_mean = [False] * num_dof
        self.f_mean_list = f_mean_list
        self.f_mean_add_par_dict_list = f_mean_add_par_dict_list
        self.pos_par_mean_init_list = pos_par_mean_init_list
        self.flg_train_pos_par_mean = flg_train_pos_par_mean
        self.free_par_mean_init_list = free_par_mean_init_list
        self.flg_train_free_par_mean = flg_train_free_par_mean
        # initialize initial par if not initialized
        if lengthscales_init_list is None:
            self.lengthscales_init_list = [None for active_dims_link in active_dims_list]
        if lambda_init_list is None:
            self.lambda_init_list = [None for active_dims_link in active_dims_list]
        if mean_init_list is None:
            self.mean_init_list = [None for active_dims_link in active_dims_list]
        # get the GP of each link
        self.models_list = torch.nn.ModuleList(
            [self.init_joint_torque_model(joint_index) for joint_index in range(0, num_dof)]
        )

    def init_joint_torque_model(self, joint_index):
        """Return a RBF gp"""
        return Stat_GP.RBF(
            self.active_dims_list[joint_index],
            sigma_n_init=self.sigma_n_init_list[joint_index],
            flg_train_sigma_n=self.flg_train_sigma_n,
            f_mean=self.f_mean_list[joint_index],
            f_mean_add_par_dict=self.f_mean_add_par_dict_list[joint_index],
            pos_par_mean_init=self.pos_par_mean_init_list[joint_index],
            flg_train_pos_par_mean=self.flg_train_pos_par_mean[joint_index],
            free_par_mean_init=self.free_par_mean_init_list[joint_index],
            flg_train_free_par_mean=self.flg_train_free_par_mean[joint_index],
            lengthscales_init=self.lengthscales_init_list[joint_index],
            flg_train_lengthscales=self.flg_train_lengthscales_list[joint_index],
            scale_init=self.lambda_init_list[joint_index],
            flg_train_scale=self.flg_train_lambda_list[joint_index],
            norm_coef_input=self.norm_coef_input,
            name=self.name + "RBF_" + str(joint_index),
            dtype=self.dtype,
            sigma_n_num=self.sigma_n_num,
            device=self.device,
        )


class m_ind_LIN(m_indep_GP):
    """
    Model that considers each link standalone
    and models each gp with a linear kernel
    """

    def __init__(
        self,
        num_dof,
        active_dims_list,
        f_transform=None,
        f_add_par_list=None,
        sigma_n_init_list=None,
        flg_train_sigma_n=True,
        Sigma_function_list=None,
        Sigma_f_additional_par_list=None,
        Sigma_pos_par_init_list=None,
        flg_train_Sigma_pos_par=True,
        Sigma_free_par_init_list=None,
        flg_train_Sigma_free_par=True,
        pos_indices=None,
        acc_indices=None,
        name="",
        dtype=torch.float64,
        max_input_loc=100000,
        downsampling_mode="Downsampling",
        sigma_n_num=None,
        device=None,
    ):
        super().__init__(
            num_dof=num_dof,
            active_dims_list=active_dims_list,
            sigma_n_init_list=sigma_n_init_list,
            flg_train_sigma_n=flg_train_sigma_n,
            pos_indices=pos_indices,
            acc_indices=acc_indices,
            name=name,
            dtype=dtype,
            device=device,
            max_input_loc=max_input_loc,
            downsampling_mode=downsampling_mode,
            sigma_n_num=sigma_n_num,
        )
        # save parameters of the linear kernel
        self.Sigma_function_list = Sigma_function_list
        self.Sigma_f_additional_par_list = Sigma_f_additional_par_list
        self.flg_train_Sigma_pos_par = flg_train_Sigma_pos_par
        self.flg_train_Sigma_free_par = flg_train_Sigma_free_par
        self.Sigma_pos_par_init_list = Sigma_pos_par_init_list
        self.Sigma_free_par_init_list = Sigma_free_par_init_list
        self.f_transform = f_transform
        self.f_add_par_list = f_add_par_list
        # check parameters initialization
        if Sigma_pos_par_init_list is None:
            self.Sigma_pos_par_init_list = [None for active_dims_joint in active_dims_list]
        if Sigma_free_par_init_list is None:
            self.Sigma_free_par_init_list = [None for active_dims_joint in active_dims_list]

        self.models_list = torch.nn.ModuleList(
            [self.init_joint_torque_model(joint_index) for joint_index in range(0, num_dof)]
        )

    def init_joint_torque_model(self, joint_index):
        """
        Returns a linear gp
        """
        if self.f_transform is None:
            f_transform = lambda x: x
            f_add_par_list = []
        else:
            f_transform = self.f_transform[joint_index]
            f_add_par_list = self.f_add_par_list[joint_index]

        return SGP.Linear_GP(
            self.active_dims_list[joint_index],
            f_transform=f_transform,
            f_add_par_list=f_add_par_list,
            sigma_n_init=self.sigma_n_init_list[joint_index],
            flg_train_sigma_n=self.flg_train_sigma_n,
            Sigma_function=self.Sigma_function_list[joint_index],
            Sigma_f_additional_par_list=self.Sigma_f_additional_par_list[joint_index],
            Sigma_pos_par_init=self.Sigma_pos_par_init_list[joint_index],
            flg_train_Sigma_pos_par=self.flg_train_Sigma_pos_par,
            Sigma_free_par_init=self.Sigma_free_par_init_list[joint_index],
            flg_train_Sigma_free_par=self.flg_train_Sigma_free_par,
            sigma_n_num=self.sigma_n_num,
            name=self.name + "PP_" + str(joint_index),
            dtype=self.dtype,
            device=self.device,
        )


class m_ind_GIP(m_indep_GP):
    """
    Implementation of the robot inverse dynamics estimator based on GIP kernel
    """

    def __init__(
        self,
        num_dof,
        gp_dict_list,
        name="",
        dtype=torch.float64,
        max_input_loc=100000,
        downsampling_mode="Downsampling",
        pos_indices=None,
        vel_indices=None,
        acc_indices=None,
        sin_indices=None,
        cos_indices=None,
        sigma_n_num=None,
        device=None,
    ):
        # initialize the superclass
        super().__init__(
            num_dof=num_dof,
            active_dims_list=None,
            sigma_n_init_list=None,
            flg_train_sigma_n=None,
            pos_indices=pos_indices,
            acc_indices=acc_indices,
            name=name,
            dtype=dtype,
            max_input_loc=max_input_loc,
            downsampling_mode=downsampling_mode,
            sigma_n_num=sigma_n_num,
            device=device,
        )
        self.vel_indices = vel_indices
        self.sin_indices = sin_indices
        self.cos_indices = cos_indices
        # save model par dict
        self.gp_dict_list = gp_dict_list
        # get the GP of each link
        self.models_list = torch.nn.ModuleList(
            [self.init_joint_torque_model(joint_index) for joint_index in range(0, num_dof)]
        )

    def init_joint_torque_model(self, joint_index):
        """
        Returns a gp with kernel given by the product of polynomial kernels
        """
        GP_list = []
        for gp_dict in self.gp_dict_list[joint_index]:
            GP_list = GP_list + self.get_gp_list_from_dict(joint_index, gp_dict)
        GP = GP_prior.Multiply_GP_prior(*GP_list)
        return GP

    def get_gp_list_from_dict(self, joint_index, gp_dict):
        """
        Returns a list of MPK object
        """
        GP_list = []
        num_gp = len(gp_dict["active_dims"])
        for gp_index in range(0, num_gp):

            # VOLTERRA MPK
            GP_list.append(
                SGP.get_Volterra_MPK_GP(
                    active_dims=gp_dict["active_dims"][gp_index],
                    poly_deg=gp_dict["poly_deg"][gp_index],
                    flg_offset=gp_dict["flg_offset"][gp_index],
                    sigma_n_init=gp_dict["sigma_n_init"][gp_index],
                    flg_train_sigma_n=gp_dict["flg_train_sigma_n"][gp_index],
                    Sigma_function_list=[cov.diagonal_covariance_ARD] * gp_dict["poly_deg"][gp_index],
                    Sigma_f_additional_par_list=[[]] * gp_dict["poly_deg"][gp_index],
                    Sigma_pos_par_init_list=gp_dict["Sigma_pos_par_init"][gp_index],
                    flg_train_Sigma_pos_par_list=gp_dict["flg_train_Sigma_pos_par"][gp_index],
                    Sigma_free_par_init_list=gp_dict["Sigma_free_par_init"][gp_index],
                    flg_train_Sigma_free_par_list=gp_dict["flg_train_Sigma_free_par"][gp_index],
                    name=self.name + "_" + str(joint_index + 1) + "_" + gp_dict["name"] + "_" + str(gp_index),
                    dtype=self.dtype,
                    sigma_n_num=gp_dict["sigma_n_num"][gp_index],
                    device=self.device,
                )
            )
        return GP_list


class m_ind_GIP_fast(m_indep_GP):
    """
    Implementation of the robot inverse dynamics estimator based on GIP kernel
    """

    def __init__(
        self,
        num_dof,
        gp_dict_list,
        name="",
        dtype=torch.float64,
        max_input_loc=100000,
        downsampling_mode="Downsampling",
        pos_indices=None,
        vel_indices=None,
        acc_indices=None,
        rev_indices=None,
        prism_indices=None,
        sigma_n_num=None,
        device=None,
    ):
        # initialize the superclass
        super().__init__(
            num_dof=num_dof,
            active_dims_list=None,
            sigma_n_init_list=None,
            flg_train_sigma_n=None,
            pos_indices=pos_indices,
            acc_indices=acc_indices,
            name=name,
            dtype=dtype,
            max_input_loc=max_input_loc,
            downsampling_mode=downsampling_mode,
            sigma_n_num=sigma_n_num,
            device=device,
        )
        self.vel_indices = vel_indices
        self.rev_indices = rev_indices
        self.prism_indices = prism_indices
        # save model par dict
        self.gp_dict_list = gp_dict_list
        # get the GP of each link
        self.models_list = torch.nn.ModuleList(
            [self.init_joint_torque_model(joint_index) for joint_index in range(0, num_dof)]
        )

    def init_joint_torque_model(self, joint_index):
        """
        Returns a gp with kernel given by the product of polynomial kernels
        """
        return GIP.GIP_GP(
            active_dims=self.pos_indices + self.vel_indices + self.acc_indices,
            num_dof=self.num_dof,
            q_dims=self.pos_indices,
            dq_dims=self.vel_indices,
            ddq_dims=self.acc_indices,
            rev_indices=self.rev_indices,
            prism_indices=self.prism_indices,
            sigma_n_init=self.gp_dict_list[joint_index]["sigma_n_init"],
            flg_train_sigma_n=self.gp_dict_list[joint_index]["flg_train_sigma_n"],
            Sigma_par_acc_init=self.gp_dict_list[joint_index]["Sigma_par_acc_init"],
            flg_train_Sigma_par_acc=self.gp_dict_list[joint_index]["flg_train_Sigma_par_acc"],
            Sigma_par_vel1_init=self.gp_dict_list[joint_index]["Sigma_par_vel1_init"],
            Sigma_par_vel2_init=self.gp_dict_list[joint_index]["Sigma_par_vel2_init"],
            Sigma_par_vel3_init=self.gp_dict_list[joint_index]["Sigma_par_vel3_init"],
            flg_train_Sigma_par_vel=self.gp_dict_list[joint_index]["flg_train_Sigma_par_vel"],
            Sigma_par_pos_rev1_init=self.gp_dict_list[joint_index]["Sigma_par_pos_rev1_init"],
            Sigma_par_pos_rev2_init=self.gp_dict_list[joint_index]["Sigma_par_pos_rev2_init"],
            Sigma_par_pos_rev3_init=self.gp_dict_list[joint_index]["Sigma_par_pos_rev3_init"],
            flg_train_Sigma_par_pos_rev=self.gp_dict_list[joint_index]["flg_train_Sigma_par_pos_rev"],
            Sigma_par_pos_prism1_init=self.gp_dict_list[joint_index]["Sigma_par_pos_prism1_init"],
            Sigma_par_pos_prism2_init=self.gp_dict_list[joint_index]["Sigma_par_pos_prism2_init"],
            Sigma_par_pos_prism3_init=self.gp_dict_list[joint_index]["Sigma_par_pos_prism3_init"],
            flg_train_Sigma_par_pos_prism=self.gp_dict_list[joint_index]["flg_train_Sigma_par_pos_prism"],
            scale_init=self.gp_dict_list[joint_index]["scale_init"],
            flg_train_scale=self.gp_dict_list[joint_index]["flg_train_scale"],
            name=self.name + "_" + str(joint_index + 1),
            dtype=self.dtype,
            sigma_n_num=self.sigma_n_num,
            device=self.device,
        )


class m_ind_GIP_with_friction(m_ind_GIP):
    """
    Implementation of the robot inverse dynamics estimator based on GIP kernel
    plus a linear kernel modeling frictions
    """

    def __init__(
        self,
        num_dof,
        gp_dict_list,
        gp_friction_dict_list,
        pos_indices=None,
        vel_indices=None,
        acc_indices=None,
        sin_indices=None,
        cos_indices=None,
        name="",
        dtype=torch.float64,
        max_input_loc=100000,
        downsampling_mode="Downsampling",
        sigma_n_num=None,
        device=None,
        downsampling_threshold_list=[],
    ):
        # save gp_dict of frictions object
        self.gp_friction_dict_list = gp_friction_dict_list
        super().__init__(
            num_dof=num_dof,
            gp_dict_list=gp_dict_list,
            pos_indices=pos_indices,
            vel_indices=vel_indices,
            acc_indices=acc_indices,
            sin_indices=sin_indices,
            cos_indices=cos_indices,
            name=name,
            dtype=dtype,
            max_input_loc=max_input_loc,
            downsampling_mode=downsampling_mode,
            sigma_n_num=sigma_n_num,
            device=device,
        )

    def init_joint_torque_model(self, joint_index):
        return GP_prior.Sum_Independent_GP(
            super().init_joint_torque_model(joint_index),
            SGP.Linear_GP(
                active_dims=self.gp_friction_dict_list[joint_index]["active_dims"],
                sigma_n_init=None,
                flg_train_sigma_n=False,
                Sigma_function=self.gp_friction_dict_list[joint_index]["Sigma_function"],
                Sigma_f_additional_par_list=self.gp_friction_dict_list[joint_index]["Sigma_f_additional_par_list"],
                Sigma_pos_par_init=self.gp_friction_dict_list[joint_index]["Sigma_pos_par_init"],
                flg_train_Sigma_pos_par=self.gp_friction_dict_list[joint_index]["flg_train_Sigma_pos_par"],
                Sigma_free_par_init=self.gp_friction_dict_list[joint_index]["Sigma_free_par_init"],
                flg_train_Sigma_free_par=self.gp_friction_dict_list[joint_index]["flg_train_Sigma_free_par"],
                flg_offset=False,
                name=self.name + "_" + str(joint_index + 1) + "_" + self.gp_friction_dict_list[joint_index]["name"],
                dtype=self.dtype,
                sigma_n_num=self.sigma_n_num,
                device=self.device,
            ),
        )


##############################
# ----------NN MODELS---------#
##############################


class m_indep_NN_sigmoid(m_independent_joint):
    """
    Class that mdoels each joint torque with a two layer NN with
    sigmoids as activation functions
    """

    def __init__(self, num_dof, num_input, num_unit_1_list, num_unit_2_list, name="", dtype=torch.float64, device=None):
        super().__init__(num_dof=num_dof, name=name, dtype=dtype, device=device)
        # save the model parameters
        self.num_input = num_input
        self.num_unit_1_list = num_unit_1_list
        self.num_unit_2_list = num_unit_2_list
        # init the model of each joint torque
        torch.set_default_dtype(dtype)
        self.models_list = torch.nn.ModuleList(
            [self.init_joint_torque_model(joint_index) for joint_index in range(0, num_dof)]
        )

    def init_joint_torque_model(self, joint_index):
        """Init the model of the joint_index torque"""
        return Deep_NN_sigmoid(self.num_input, self.num_unit_1_list[joint_index], self.num_unit_2_list[joint_index])


class m_indep_NN_relu(m_indep_NN_sigmoid):
    """
    Class that mdoels each joint torque with a two layer NN with
    relu as activation functions
    """

    def init_joint_torque_model(self, joint_index):
        """Init the model of the joint_index torque"""
        return Deep_NN_relu(self.num_input, self.num_unit_1_list[joint_index], self.num_unit_2_list[joint_index])


class m_indep_NN(m_indep_NN_sigmoid):
    """
    Class that mdoels each joint torque with a two layer NN.
    Activation functions of the first layer are sigmoid, while
    the activation functions of the second layer are relu
    """

    def init_joint_torque_model(self, joint_index):
        """
        Init the model of the joint_index torque
        """
        return Deep_NN(self.num_input, self.num_unit_1_list[joint_index], self.num_unit_2_list[joint_index])


class Deep_NN_sigmoid(torch.nn.Module):
    """
    Two layer fully connected NN with sigmoid as activation functions
    """

    def __init__(self, num_input, num_unit_1, num_unit_2):
        super().__init__()
        self.f1 = torch.nn.Linear(num_input, num_unit_1, bias=True)
        self.f2 = torch.nn.Linear(num_unit_1, num_unit_2, bias=True)
        self.f3 = torch.nn.Linear(num_unit_2, 1, bias=True)

    def forward(self, x):
        layer_1 = torch.sigmoid(self.f1(x))
        layer_2 = torch.sigmoid(self.f2(layer_1))
        return self.f3(layer_2)


class Deep_NN_relu(torch.nn.Module):
    """
    Two layer fully connected NN with relu as activation functions
    """

    def __init__(self, num_input, num_unit_1, num_unit_2):
        super().__init__()
        self.f1 = torch.nn.Linear(num_input, num_unit_1, bias=True)
        self.f2 = torch.nn.Linear(num_unit_1, num_unit_2, bias=True)
        self.f3 = torch.nn.Linear(num_unit_2, 1, bias=True)

    def forward(self, x):
        layer_1 = torch.relu(self.f1(x))
        layer_2 = torch.relu(self.f2(layer_1))
        return self.f3(layer_2)


class Deep_NN(torch.nn.Module):
    """
    Two layer fully connected NN.
    The activation functions of the first layer are sigmoids,
    while the second layer is composed of relu functions
    """

    def __init__(self, num_input, num_unit_1, num_unit_2):
        super().__init__()
        self.f1 = torch.nn.Linear(num_input, num_unit_1, bias=True)
        self.f2 = torch.nn.Linear(num_unit_1, num_unit_2, bias=True)
        self.f3 = torch.nn.Linear(num_unit_2, 1, bias=True)

    def forward(self, x):
        layer_1 = torch.sigmoid(self.f1(x))
        layer_2 = torch.relu(self.f2(layer_1))
        return self.f3(layer_2)
