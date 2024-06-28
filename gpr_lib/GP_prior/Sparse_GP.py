"""
Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
This file contains the definition of linear and polynomial GP.
"""


import numpy as np
import torch

from .. import Utils
from . import GP_prior


class Linear_GP(GP_prior.GP_prior):
    """
    Implementation of the GP with linear kernel (dot product covariance).
    f(X) = phi(X)*w, where the w is defined as a gaussian variable (mean_w and Sigma_w)
    """

    def __init__(
        self,
        active_dims,
        f_transform=lambda x: x,
        f_add_par_list=[],
        sigma_n_init=None,
        flg_train_sigma_n=False,
        f_mean=None,
        f_mean_add_par_dict={},
        pos_par_mean_init=None,
        flg_train_pos_par_mean=False,
        free_par_mean_init=None,
        flg_train_free_par_mean=False,
        Sigma_function=None,
        Sigma_f_additional_par_list=None,
        Sigma_pos_par_init=None,
        flg_train_Sigma_pos_par=True,
        Sigma_free_par_init=None,
        flg_train_Sigma_free_par=True,
        flg_offset=False,
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
        # save f_transform
        self.f_transform = f_transform
        self.f_add_par_list = f_add_par_list
        # check active dims
        if active_dims is None:
            raise RuntimeError("Active_dims are needed")
        self.num_features = active_dims.size
        # save flg_offset (flg_offset=True => ones added to the phi)
        self.flg_offset = flg_offset
        # # check mean init
        # self.check_mean(mean_init, flg_mean_trainable, flg_no_mean)
        # check Sigma init
        self.check_sigma_function(
            Sigma_function,
            Sigma_f_additional_par_list,
            Sigma_pos_par_init,
            flg_train_Sigma_pos_par,
            Sigma_free_par_init,
            flg_train_Sigma_free_par,
        )

    # def check_mean(self, mean_init, flg_mean_trainable, flg_no_mean):
    #     if mean_init is None:
    #         mean_init = np.zeros(1)
    #         flg_no_mean = True
    #     self.flg_no_mean = flg_no_mean
    #     self.mean_par = torch.nn.Parameter(torch.tensor(mean_init, dtype=self.dtype, device=self.device), requires_grad=flg_mean_trainable)

    def check_sigma_function(
        self,
        Sigma_function,
        Sigma_f_additional_par_list,
        Sigma_pos_par_init,
        flg_train_Sigma_pos_par,
        Sigma_free_par_init,
        flg_train_Sigma_free_par,
    ):
        if Sigma_function is None:
            raise RuntimeError("Specify a Sigma function")
        self.Sigma_function = Sigma_function
        self.Sigma_f_additional_par_list = Sigma_f_additional_par_list
        if Sigma_pos_par_init is None:
            self.Sigma_pos_par = None
        else:
            self.Sigma_pos_par = torch.nn.Parameter(
                torch.tensor(np.log(Sigma_pos_par_init), dtype=self.dtype, device=self.device),
                requires_grad=flg_train_Sigma_pos_par,
            )
        if Sigma_free_par_init is None:
            self.Sigma_free_par = None
        else:
            self.Sigma_free_par = torch.nn.Parameter(
                torch.tensor(Sigma_free_par_init, dtype=self.dtype, device=self.device),
                requires_grad=flg_train_Sigma_free_par,
            )

    def get_phi(self, X):
        """
        Returns the regression matrix associated to the inputs X
        """
        num_samples = X.shape[0]
        # apply transform
        phi = self.f_transform(X, *self.f_add_par_list)
        if self.flg_offset:
            return torch.cat(
                [phi[:, self.active_dims], torch.ones(num_samples, 1, dtype=self.dtype, device=self.device)], 1
            )
        else:
            return phi[:, self.active_dims]

    def get_Sigma(self):
        """
        Computes the Sigma matrix
        """
        if self.Sigma_pos_par is None:
            return self.Sigma_function(self.Sigma_pos_par, self.Sigma_free_par, *self.Sigma_f_additional_par_list)
        else:
            return self.Sigma_function(
                torch.exp(self.Sigma_pos_par), self.Sigma_free_par, *self.Sigma_f_additional_par_list
            )

    # def get_mean(self,X):
    #     """
    #     Returns phi(X)*w_mean
    #     """
    #     if self.flg_no_mean:
    #         N = X.size()[0]
    #         return torch.zeros(N, 1, dtype=self.dtype, device=self.device)
    #     else:
    #         return torch.matmul(self.get_phi(X), self.mean_par)

    def get_covariance(self, X1, X2=None, flg_noise=False, p_drop=0.0):
        """
        Returns phi(X)^T*Sigma*phi(X)
        """
        # get the parameters variance
        Sigma = self.get_Sigma()
        # get the covariance
        phi_X1 = self.get_phi(X1)
        if X2 is None:
            K_X = torch.exp(self.scale_log) * torch.matmul(phi_X1, torch.matmul(Sigma, phi_X1.transpose(0, 1)))
            # check if we need to add the noise
            if flg_noise & self.GP_with_noise:
                N = X1.size()[0]
                return K_X + self.get_sigma_n_2() * torch.eye(N, dtype=self.dtype, device=self.device)
            else:
                return K_X
        else:
            return torch.exp(self.scale_log) * torch.matmul(
                phi_X1, torch.matmul(Sigma, self.get_phi(X2).transpose(0, 1))
            )

    def get_diag_covariance(self, X, flg_noise=False):
        """
        Returns the diag of the cov matrix
        """
        # Get the parameters the variance
        Sigma = self.get_Sigma()
        # get the diag of the covariance
        phi_X = self.get_phi(X)
        diag = torch.exp(self.scale_log) * torch.sum(torch.matmul(phi_X, Sigma) * (phi_X), dim=1)
        if flg_noise & self.GP_with_noise:
            return diag + self.get_sigma_n_2()
        else:
            return diag

    def get_parameters(self, X, Y, flg_print=False):
        """
        Returns the posterior estimate of w, the parameters of the regression.
        NB: the parameters returned are correct only if this is the only GP
        """
        # get the mean and the inverse of the kernel matrix using forward
        m_X, _, K_X_inv, _ = self.forward(X)
        Y = Y - m_X
        # get sigma and phi
        Sigma = self.get_Sigma()
        phi_X_T = torch.transpose(self.get_phi(X), 0, 1)
        # get the parameters
        w_hat = torch.matmul(Sigma, torch.matmul(phi_X_T, torch.matmul(K_X_inv, Y)))
        if flg_print:
            print(self.name + " linear parameters estimated: ", w_hat.data)
        return w_hat

    def get_parameters_inv_lemma(self, X, Y, flg_print=False):
        """
        Returns the estimate of w, the parameters of the regression.
        This implementation exploit the sparsity of the covariance and uses
        the matrix inversion lemma.
        NB: the parameters returned are correct only if this is the only GP
        """
        # get the mean and the phi and the covariance of the parameters
        m_X = self.get_mean(X)
        Y = Y - m_X
        Phi_X = self.get_phi(X)
        Sigma = self.get_Sigma()
        sigma_n_square = self.get_sigma_n_2()
        # return the parameters
        cov = torch.inverse(Sigma) + sigma_n_square * torch.matmul(Phi_X.transpose(0, 1), Phi_X)
        cov = torch.inverse(cov)
        w_hat = sigma_n_square * torch.matmul(torch.matmul(cov, Phi_X.transpose(0, 1)), Y)
        if flg_print:
            print(self.name + " linear parameters estimated: ", w_hat.data)
        return w_hat


# class Linear_GP_f_transform(Linear_GP):
#     """docstring for Linear_GP_F_transform"""
#     def __init__(self, active_dims, f_transform, f_add_par,
#                  sigma_n_init=None, flg_train_sigma_n=False,
#                  f_mean=None, f_mean_add_par_dict={},
#                  pos_par_mean_init=None, flg_train_pos_par_mean=False,
#                  free_par_mean_init=None, flg_train_free_par_mean=False,
#                  Sigma_function=None, Sigma_f_additional_par_list=None,
#                  Sigma_pos_par_init=None, flg_train_Sigma_pos_par=True,
#                  Sigma_free_par_init=None, flg_train_Sigma_free_par=True,
#                  flg_offset=False,
#                  scale_init=np.ones(1), flg_train_scale=False,
#                  name='', dtype=torch.float64, sigma_n_num=None, device=None):
#         super(Linear_GP_F_transform, self).__init__(active_dims=active_dims,
#                                                     sigma_n_init=sigma_n_init, flg_train_sigma_n=flg_train_sigma_n,
#                                                     f_mean=f_mean, f_mean_add_par_dict=f_mean_add_par_dict,
#                                                     pos_par_mean_init=pos_par_mean_init, flg_train_pos_par_mean=flg_train_pos_par_mean,
#                                                     free_par_mean_init=free_par_mean_init, flg_train_free_par_mean=flg_train_free_par_mean,
#                                                     Sigma_function=Sigma_function, Sigma_f_additional_par_list=Sigma_f_additional_par_list,
#                                                     Sigma_pos_par_init=Sigma_pos_par_init, flg_train_Sigma_pos_par=flg_train_Sigma_pos_par,
#                                                     Sigma_free_par_init=Sigma_free_par_init, flg_train_Sigma_free_par=flg_train_Sigma_free_par,
#                                                     flg_offset=flg_offset,
#                                                     scale_init=scale_init, flg_train_scale=flg_train_scale,
#                                                     name=name, dtype=dtype, sigma_n_num=sigma_n_num, device=device)
#         self.f_transform = f_transform
#         self.f_add_par = f_add_par


#     def get_phi(self,X):
#         """
#         Returns the regression matrix associated to the inputs X
#         """
#         num_samples = X.shape[0]
#         phi = self.f_transform(X,self.f_add_par)
#         if self.flg_offset:
#             return torch.cat([phi,torch.ones(num_samples,1, dtype=self.dtype, device=self.device)],1)
#         else:
#             return phi


class Poly_GP(Linear_GP):
    """
    GP with polynomial kernel. Implemented extending the Linear_GP
    """

    def __init__(
        self,
        active_dims,
        poly_deg,
        sigma_n_init=None,
        flg_train_sigma_n=True,
        f_mean=None,
        f_mean_add_par_dict={},
        pos_par_mean_init=None,
        flg_train_pos_par_mean=False,
        free_par_mean_init=None,
        flg_train_free_par_mean=False,
        Sigma_function=None,
        Sigma_f_additional_par_list=None,
        Sigma_pos_par_init=None,
        flg_train_Sigma_pos_par=True,
        Sigma_free_par_init=None,
        flg_train_Sigma_free_par=True,
        flg_offset=True,
        scale_init=np.ones(1),
        flg_train_scale=False,
        name="",
        dtype=torch.float64,
        sigma_n_num=None,
        device=None,
    ):
        # initialize the linear model
        super().__init__(
            active_dims=active_dims,
            sigma_n_init=sigma_n_init,
            flg_train_sigma_n=flg_train_sigma_n,
            f_mean=f_mean,
            f_mean_add_par_dict=f_mean_add_par_dict,
            pos_par_mean_init=pos_par_mean_init,
            flg_train_pos_par_mean=flg_train_pos_par_mean,
            free_par_mean_init=free_par_mean_init,
            flg_train_free_par_mean=flg_train_free_par_mean,
            Sigma_function=Sigma_function,
            Sigma_f_additional_par_list=Sigma_f_additional_par_list,
            Sigma_pos_par_init=Sigma_pos_par_init,
            flg_train_Sigma_pos_par=flg_train_Sigma_pos_par,
            Sigma_free_par_init=Sigma_free_par_init,
            flg_train_Sigma_free_par=flg_train_Sigma_free_par,
            flg_offset=flg_offset,
            scale_init=np.ones(1),
            flg_train_scale=False,
            name=name,
            dtype=dtype,
            sigma_n_num=sigma_n_num,
            device=device,
        )
        # save the poly deg
        self.poly_deg = poly_deg
        # save the scaling parameter of the poly transformation
        self.scale_log = torch.nn.Parameter(
            torch.tensor(np.log(scale_init), dtype=self.dtype, device=self.device), requires_grad=flg_train_scale
        )

    def get_covariance(self, X1, X2=None, flg_noise=False):
        """
        Returns the linear covariance raised to self.poly_deg
        """
        return torch.exp(self.scale_log) * (super().get_covariance(X1, X2, flg_noise)) ** self.poly_deg

    def get_diag_covariance(self, X, flg_noise=False):
        return torch.exp(self.scale_log) * (super().get_diag_covariance(X, flg_noise=flg_noise)) ** self.poly_deg

    def get_parameters(self, X, Y, flg_print=False):
        raise NotImplementedError()

    def get_parameters_inv_lemma(self, X, Y, flg_print=False):
        raise NotImplementedError()


# class MPK_GP(Linear_GP):
#     """
#     Implementation of the Multiplicaive Polynomial Kernel
#     """
#     def __init__(self, active_dims, poly_deg,
#                  sigma_n_init=None, flg_train_sigma_n=True,
#                  Sigma_pos_par_init=None, flg_train_Sigma_pos_par=True,
#                  Sigma_free_par_init=None, flg_train_Sigma_free_par=True,
#                  flg_offset=True,
#                  name='', dtype=torch.float64, sigma_n_num=None, device=None):
#         # init the linear GP object
#         mean_init=None
#         flg_mean_trainable=False
#         flg_no_mean=True
#         Sigma_function = Utils.Parameters_covariance_functions.diagonal_covariance_ARD
#         Sigma_f_additional_par_list = []
#         super(MPK_GP, self).__init__(active_dims=active_dims,
#                                      mean_init=mean_init, flg_mean_trainable=flg_mean_trainable, flg_no_mean=flg_no_mean,
#                                      sigma_n_init=sigma_n_init, flg_train_sigma_n=flg_train_sigma_n,
#                                      Sigma_function=Sigma_function, Sigma_f_additional_par_list=Sigma_f_additional_par_list,
#                                      Sigma_pos_par_init=None, flg_train_Sigma_pos_par=False,
#                                      Sigma_free_par_init=None, flg_train_Sigma_free_par=False,
#                                      flg_offset=flg_offset,
#                                      name=name, dtype=dtype, sigma_n_num=sigma_n_num, device=device)
#         self.poly_deg = poly_deg
#         # get kernel parameters
#         self.Sigma_free_par = torch.nn.Parameter(torch.tensor(np.log(Sigma_free_par_init),
#                                                               dtype=self.dtype,
#                                                               device=self.device),
#                                                 requires_grad=flg_train_Sigma_free_par)
#         self.num_Sigma_free_par = int(Sigma_free_par_init.size/poly_deg)
#         self.Sigma_pos_par = torch.nn.Parameter(torch.tensor(np.log(Sigma_pos_par_init),
#                                                              dtype=self.dtype,
#                                                              device=self.device),
#                                                 requires_grad=flg_train_Sigma_pos_par)
#         self.num_Sigma_pos_par = int(Sigma_pos_par_init.size/poly_deg)


#     def get_Sigma(self):
#         """Computes the Sigma matrix"""
#         Sigma_pos_par = torch.exp(self.Sigma_pos_par[self.current_deg*self.num_Sigma_pos_par:(self.current_deg+1)*self.num_Sigma_pos_par])
#         Sigma_free_par = torch.zeros(self.num_Sigma_free_par,
#                                      dtype=self.dtype,
#                                      device=self.device)
#         for deg in range(self.current_deg, self.poly_deg):
#             Sigma_free_par += torch.exp(self.Sigma_free_par[deg*self.num_Sigma_free_par:(deg+1)*self.num_Sigma_free_par])**2
#         Sigma_free_par = torch.sqrt(Sigma_free_par)
#         return self.Sigma_function(torch.cat([Sigma_free_par,Sigma_pos_par]))


#     def get_covariance(self, X1, X2=None, flg_noise=False):
#         N1 = X1.size()[0]
#         if X2 is None:
#             N2 = N1
#         else:
#             N2 = X2.size()[0]
#         K_X = torch.ones((N1,N2), dtype=self.dtype, device=self.device)
#         for deg in range(0,self.poly_deg):
#             self.current_deg = deg
#             K_X *= super(MPK_GP, self).get_covariance(X1,X2, flg_noise=False)
#         if flg_noise & self.GP_with_noise & N1==N2:
#             K_X += self.get_sigma_n_2()*torch.eye(N1, dtype=self.dtype, device=self.device)
#         return K_X


#     def get_diag_covariance(self, X, flg_noise=False):
#         """ Returns the diag of the cov matrix"""
#         N = X.size()[0]
#         diag = torch.ones(N, dtype=self.dtype, device=self.device)
#         for deg in range(0,self.poly_deg):
#             self.current_deg = deg
#             diag *= super(MPK_GP, self).get_diag_covariance(X, flg_noise=False)
#         if flg_noise & self.GP_with_noise:
#             return diag + self.get_sigma_n_2()
#         else:
#             return diag


class MPK_GP(Linear_GP):
    """
    Implementation of the Multiplicaive Polynomial Kernel
    """

    def __init__(
        self,
        active_dims,
        poly_deg,
        sigma_n_init=None,
        flg_train_sigma_n=True,
        f_mean=None,
        f_mean_add_par_dict={},
        pos_par_mean_init=None,
        flg_train_pos_par_mean=False,
        free_par_mean_init=None,
        flg_train_free_par_mean=False,
        Sigma_pos_par_init=None,
        flg_train_Sigma_pos_par=True,
        flg_offset=True,
        name="",
        dtype=torch.float64,
        sigma_n_num=None,
        device=None,
    ):
        # init the linear GP object
        Sigma_function = Utils.Parameters_covariance_functions.diagonal_covariance_ARD
        Sigma_f_additional_par_list = []
        super().__init__(
            active_dims=active_dims,
            sigma_n_init=sigma_n_init,
            flg_train_sigma_n=flg_train_sigma_n,
            f_mean=f_mean,
            f_mean_add_par_dict=f_mean_add_par_dict,
            pos_par_mean_init=pos_par_mean_init,
            flg_train_pos_par_mean=flg_train_pos_par_mean,
            free_par_mean_init=free_par_mean_init,
            flg_train_free_par_mean=flg_train_free_par_mean,
            Sigma_function=Sigma_function,
            Sigma_f_additional_par_list=Sigma_f_additional_par_list,
            Sigma_pos_par_init=Sigma_pos_par_init,
            flg_train_Sigma_pos_par=flg_train_Sigma_pos_par,
            Sigma_free_par_init=None,
            flg_train_Sigma_free_par=False,
            flg_offset=flg_offset,
            name=name,
            dtype=dtype,
            sigma_n_num=sigma_n_num,
            device=device,
        )
        self.poly_deg = poly_deg

    def get_covariance(self, X1, X2=None, flg_noise=False):
        # get sigma
        Sigma_matrices = torch.stack(
            [
                self.Sigma_function(torch.exp(self.Sigma_pos_par[d, :]), None, *self.Sigma_f_additional_par_list)
                for d in range(0, self.poly_deg)
            ]
        )
        # get the kernel
        phi_X1 = self.get_phi(X1).unsqueeze_(0)
        # print('Sigma_matrices',Sigma_matrices.shape)
        # print('phi_X1',phi_X1.shape)
        if X2 is None:
            K_X = torch.exp(self.scale_log) * torch.prod(
                torch.matmul(phi_X1, torch.matmul(Sigma_matrices, phi_X1.transpose(1, 2))), 0
            )
            # check if we need to add the noise
            if flg_noise & self.GP_with_noise:
                N = X1.size()[0]
                return K_X + self.get_sigma_n_2() * torch.eye(N, dtype=self.dtype, device=self.device)
            else:
                return K_X
        else:
            phi_X2 = self.get_phi(X2).unsqueeze_(0)
            return torch.exp(self.scale_log) * torch.prod(
                torch.matmul(phi_X1, torch.matmul(Sigma_matrices, phi_X2.transpose(1, 2))), 0
            )

    def get_diag_covariance(self, X, flg_noise=False):
        """
        Returns the diag of the cov matrix
        """
        # get sigma
        Sigma_matrices = torch.stack(
            [
                self.Sigma_function(torch.exp(self.Sigma_pos_par[d, :]), None, *self.Sigma_f_additional_par_list)
                for d in range(0, self.poly_deg)
            ]
        )
        phi_X = self.get_phi(X).unsqueeze_(0)
        diag = torch.exp(self.scale_log) * torch.sum(torch.matmul(phi_X, Sigma_matrices) * (phi_X), dim=2).prod(0)
        if flg_noise & self.GP_with_noise:
            return diag + self.get_sigma_n_2()
        else:
            return diag


# class Poly_par_shared_GP(Poly_GP):
#     """
#     GP with polynomial kernel and sharing of the parameters.
#     The kernel hyperparameters depends on the output_channel selected,
#     and are a function of a set of common hyperparameters.
#     In this basic implementation the PK hyperparameters are a subset of
#     the common hyperparameters, selected based on a set of indices
#     """
#     def __init__(self, active_dims, poly_deg,
#                  sigma_n_init=None, flg_train_sigma_n=True,
#                  Sigma_function=None, Sigma_f_additional_par_list=None,
#                  Sigma_pos_par_shared_init=None, flg_train_Sigma_pos_par=True,
#                  Sigma_free_par_shared_init=None, flg_train_Sigma_free_par=True,
#                  flg_offset=True,
#                  Sigma_pos_par_indices_list = None, Sigma_free_par_indices_list=None,
#                  scale_init=np.ones(1), flg_train_scale=False,
#                  name='', dtype=torch.float64, sigma_n_num=None, device=None):
#         # initilize the mean parameters (no mean considered)
#         mean_init=None
#         flg_mean_trainable=False
#         flg_no_mean=True
#         # initialize the polynomial model
#         super(Poly_par_shared_GP, self).__init__(active_dims=active_dims, poly_deg=poly_deg,
#                                                  sigma_n_init=sigma_n_init, flg_train_sigma_n=flg_train_sigma_n,
#                                                  Sigma_function=Sigma_function, Sigma_f_additional_par_list=Sigma_f_additional_par_list,
#                                                  Sigma_pos_par_init=None, flg_train_Sigma_pos_par=None,
#                                                  Sigma_free_par_init=None, flg_train_Sigma_free_par=None,
#                                                  flg_offset=flg_offset,
#                                                  scale_init=scale_init, flg_train_scale=flg_train_scale,
#                                                  name=name, dtype=dtype, sigma_n_num=sigma_n_num, device=device)
#         # save the shared parameters
#         if Sigma_pos_par_shared_init is None:
#             self.Sigma_pos_par_shared = None
#         else:
#             self.Sigma_pos_par_shared = torch.nn.Parameter(torch.tensor(np.log(Sigma_pos_par_shared_init), dtype=self.dtype, device=self.device),
#                                                                 requires_grad=flg_train_Sigma_pos_par)
#         if Sigma_free_par_shared_init is None:
#             self.Sigma_free_par_shared = None
#         else:
#             self.Sigma_free_par_shared = torch.nn.Parameter(torch.tensor(Sigma_free_par_shared_init, dtype=self.dtype, device=self.device),
#                                                                  requires_grad=flg_train_Sigma_free_par)
#         # save the indices that define the Sigma_pos_par and Sigma_free_par dependign on the output_channel
#         self.Sigma_pos_par_indices_list = Sigma_pos_par_indices_list
#         self.Sigma_free_par_indices_list = Sigma_free_par_indices_list


#     def set_kernel_parameters(self, output_channel):
#         """
#         Funciton that sets self.sigma_pos_par and self.Sigma_free_par
#         based on the output channel
#         """
#         if self.Sigma_pos_par_shared is None:
#             self.Sigma_pos_par = None
#         else:
#             self.Sigma_pos_par = self.Sigma_pos_par_shared[self.Sigma_pos_par_indices_list[output_channel]]
#         if self.Sigma_free_par_shared is None:
#             self.Sigma_free_par = None
#         else:
#             self.Sigma_free_par = self.Sigma_free_par_shared[self.Sigma_free_par_indices_list[output_channel]]


#     def print_model(self):
#         """
#         Print the model parameters
#         """
#         print(self.name+' parameters:')
#         if self.Sigma_pos_par is not None:
#             print('-', 'Sigma_pos_par', ':', self.Sigma_pos_par.data)
#         if self.Sigma_free_par is not None:
#             print('-', 'Sigma_free_par', ':', self.Sigma_free_par.data)


def get_SOR_GP(exact_GP_object):
    """
    function that returns a Subset of Regressors GP, given a GP object.
    This model is a low-rank approximation of an exact GP model.
    Given a GP model with kernel function k(x_1,x_2), SOR approximate its covariance
    with k_SOR(x_1,x_2) = k(x_1,U) K(U,U)^-1 k(U,x_2), where:
    - U = {set of inducing inputs}
    - K(U,U) = kernel matrix associated to the inducing inputs
    """

    # create the SOR_GP class dynamically
    class SOR_GP(type(exact_GP_object)):
        """SOR_GP"""

        def __init__(self, exact_GP_object):
            """Initialize the object inheriting all the exact_GP_object parameters"""
            # initialize the GP object randomly
            GP_prior.GP_prior.__init__(
                self,
                active_dims=[0],
                sigma_n_init=None,
                flg_train_sigma_n=False,
                scale_init=np.ones(1),
                flg_train_scale=False,
                name="",
                dtype=torch.float64,
                sigma_n_num=None,
                device=torch.device("cpu"),
            )
            # assign all the variables of the exact_GP_object
            self.__dict__ = exact_GP_object.__dict__
            self.name = "SOR_GP " + self.name

        def init_inducing_inputs(self, inducing_inputs, flg_train_inducing_inputs=False):
            """set the set U, a matrix of dimension (num_inducing_inputs, num_feature)
            which represent the initial value of the inducing inputs.
            If flg_train_inducing_inputs=True U is considered as a trainable hyperparameter
            """
            self.U = torch.nn.Parameter(
                torch.tensor(inducing_inputs, dtype=self.dtype, device=self.device),
                requires_grad=flg_train_inducing_inputs,
            )

        def set_inducing_inputs_from_data(self, X, Y, threshold, flg_trainable):
            """
            Set the inducing inputs with an online procedure
            """
            print("\nSelection of the inducing inputs...")
            # get number of samples
            num_samples = X.shape[0]
            # init the set of inducing inputs with the first sample
            self.U = torch.nn.Parameter(X[0:1, :], requires_grad=flg_trainable)
            inducing_inputs_indices = [0]
            # iterate all the samples
            for sample_index in range(2, num_samples):
                # get the estimate
                # _, var, _ = self.get_SOR_estimate(X[:sample_index-1,:], Y[:sample_index-1,:], X[sample_index:sample_index+1,:])
                _, var, _ = self.get_estimate(
                    X[inducing_inputs_indices, :], Y[inducing_inputs_indices, :], X[sample_index : sample_index + 1, :]
                )
                # check
                # print('torch.sqrt(var)',torch.sqrt(var))
                if torch.sqrt(var) > threshold:
                    self.U.data = torch.cat([self.U.data, X[sample_index : sample_index + 1, :]], 0)
                    inducing_inputs_indices.append(sample_index)
            print("Shape of the inducing inputs selected:", self.U.shape)
            return inducing_inputs_indices

        def get_SOR_alpha(self, X, Y):
            """
            Returns the coefficients that defines the SOR posterior distribution

            inputs:
            - X = training inputs
            - Y = training outputs

            outputs:
            - alpha = vector defining the SOR posterior distribution
            - m_X = prior mean of X
            - Sigma = inverse of (K_UU + sigma_n^-2*K_UX*K_XU)
            """
            # get the mean and the phi and the covariance of the parameters
            m_X = self.get_mean(X)
            Y = Y - m_X
            K_XU = self.get_covariance(X, self.U)
            K_UU = self.get_covariance(self.U)
            sigma_n_square_inv = 1 / self.get_sigma_n_2()
            sigma_n_square = self.get_sigma_n_2()
            # return the parameters
            Sigma_inv = K_UU + sigma_n_square_inv * torch.matmul(K_XU.transpose(0, 1), K_XU)
            # Sigma_inv = sigma_n_square*K_UU + torch.matmul(K_XU.transpose(0,1), K_XU)

            # Sigma = torch.inverse(Sigma_inv)
            U_Sigma_inv = torch.cholesky(Sigma_inv, upper=True)
            U_Sigma = torch.inverse(U_Sigma_inv)
            Sigma = torch.matmul(U_Sigma, U_Sigma.transpose(0, 1))

            SOR_alpha = sigma_n_square_inv * torch.matmul(torch.matmul(Sigma, K_XU.transpose(0, 1)), Y)
            # SOR_alpha = torch.matmul(torch.matmul(Sigma, K_XU.transpose(0,1)), Y)
            return SOR_alpha, m_X, Sigma

        def get_SOR_estimate_from_alpha(self, X, X_test, SOR_alpha, m_X, Sigma=None):
            """
            Compute the SOR posterior distribution in X_test, given the alpha vector.

            input:
            - X = training input locations (used to compute alpha)
            - X_test = test input locations
            - SOR_alpha = vector of coefficients defining the SOR posterior
            - m_X = prior mean of X
            - Sigma = inverse of (K_UU + sigma_n^-2*K_UX*K_XU)

            output:
            - Y_hat = posterior mean
            - var = diagonal elements of the posterior variance

            If Sigma_inv is given the method returns also the confidence intervals (variance of the gaussian)
            """
            # get covariance and prior mean
            K_X_test_U = self.get_covariance(X_test, self.U)
            m_X_test = self.get_mean(X_test)
            # get the estimate
            Y_hat = m_X_test + torch.matmul(K_X_test_U, SOR_alpha)
            # if Sigma_inv is given compute the confidence intervals
            if Sigma is not None:
                var = torch.sum(torch.matmul(K_X_test_U, Sigma) * (K_X_test_U), dim=1)
                # var = self.get_sigma_n_2()*torch.sum(torch.matmul(K_X_test_U, Sigma)*(K_X_test_U), dim=1)
            return Y_hat, var

        def get_SOR_estimate(self, X, Y, X_test, Y_test=None, flg_return_Sigma=False):
            """
            Returns the SOR posterior distribution in X_test, given the training samples X Y and the inducing inputs self.U.

            input:
            - X = training input
            - Y = training output
            - X_test = test input
            - Y_test = value of the output in the test location

            output:
            - Y_hat = mean of the test posterior
            - var = diagonal elements of the variance posterior
            - SOR_alpha = coefficients defining the SOR posterior
            - m_X = prior mean of the training samples
            - Sigma = (K_UU + sigma_n^-2*K_UX*K_XU)
            """
            # get the coefficent and the mean
            SOR_alpha, m_X, Sigma = self.get_SOR_alpha(X, Y)
            # get the estimate and the confidence intervals
            Y_hat, var = self.get_SOR_estimate_from_alpha(X, X_test, SOR_alpha, m_X, Sigma=Sigma)
            # return the opportune values
            if flg_return_Sigma:
                return Y_hat, var, SOR_alpha, m_X, Sigma
            else:
                return Y_hat, var, SOR_alpha

        def SOR_forward(self, X):
            """
            Returns the elements that define the likelihood distribution of the model
            when considering the SOR approximation

            input:
            - X = training inputs (X has dimension [num_samples, num_features])

            output:
            - m_X = mean
            - K_X = None
            - K_X_inv = inverse of (K_XU*K_UU^-1*K_UX+sigma_n^2)
            - log_det = (K_XU*K_UU^-1*K_UX+sigma_n^2)
            """
            # get the mean
            m_X = self.get_mean(X)
            # get kernel matrices
            K_X = None
            N = X.shape[0]
            K_UU = self.get_covariance(self.U)
            K_XU = self.get_covariance(X, self.U)
            sigma_n_square_inv = 1 / self.get_sigma_n_2()
            # compute the K_UU^-1 logdet
            U_K_UU = torch.cholesky(K_UU, upper=True)
            K_UU_inv_log_det = -2 * torch.sum(torch.log(torch.diag(U_K_UU)))
            # compute Sigma
            Sigma_inv = K_UU + sigma_n_square_inv * torch.matmul(K_XU.transpose(0, 1), K_XU)
            # compute Sigma inverse and logdet
            U_Sigma_inv = torch.cholesky(Sigma_inv, upper=True)
            Sigma_inv_log_det = 2 * torch.sum(torch.log(torch.diag(U_Sigma_inv)))
            U_Sigma = torch.inverse(U_Sigma_inv)
            Sigma = torch.matmul(U_Sigma, U_Sigma.transpose(0, 1))
            # compute K_X_inv
            K_X_inv = sigma_n_square_inv * torch.eye(N, dtype=self.dtype, device=self.device)
            K_X_inv -= sigma_n_square_inv**2 * torch.matmul(K_XU, torch.matmul(Sigma, K_XU.transpose(0, 1)))
            # compute the log_det
            log_det = N * torch.log(self.get_sigma_n_2()) + K_UU_inv_log_det + Sigma_inv_log_det
            return m_X, K_X, K_X_inv, log_det

        def fit_SOR_model(
            self,
            trainloader=None,
            optimizer=None,
            criterion=None,
            N_epoch=1,
            N_epoch_print=1,
            f_saving_model=None,
            f_print=None,
        ):
            """
            Optimize the SOR model hyperparameters

            input:
            - trainloader = torch train loader object
            - optimizer = torch optimizer object
            - criterion = loss function
            - N_epoch = number of epochs
            - N_epoch_print = number of epoch between print two prints of the current loss and model parameters
            - f_saving_model = customizable function that save the model
            - f_print_model = customizable function that print the model (eventually with performance)
            """
            # print initial parametes and initial estimates
            print("\nInitial parameters:")
            self.print_model()
            # iterate over the training data for N_epochs
            for epoch in range(0, N_epoch):
                # initialize loss grad and counter
                running_loss = 0.0
                N_btc = 0
                optimizer.zero_grad()
                # iterate over the training set
                for i, data in enumerate(trainloader, 0):
                    # get the training data
                    inputs, labels = data
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    out_SOR_GP_priors = self.SOR_forward(inputs)
                    loss = criterion(out_SOR_GP_priors, labels)
                    loss.backward(retain_graph=False)
                    optimizer.step()
                    # update the running loss
                    running_loss = running_loss + loss.item()
                    N_btc = N_btc + 1
                # print statistics and save the model
                if epoch % N_epoch_print == 0:
                    print("\nEPOCH:", epoch)
                    self.print_model()
                    print("Runnung loss:", running_loss / N_btc)
                    if f_saving_model is not None:
                        f_saving_model(epoch)
                    if f_print is not None:
                        f_print()
            # print the final parameters
            print("\nFinal parameters:")
            self.print_model()

    # init the object and return
    return SOR_GP(exact_GP_object)


def get_Volterra_PK_GP(
    active_dims,
    poly_deg,
    flg_offset,
    sigma_n_init=None,
    flg_train_sigma_n=False,
    Sigma_function_list=None,
    Sigma_f_additional_par_list=None,
    Sigma_pos_par_init_list=[],
    flg_train_Sigma_pos_par_list=[],
    Sigma_free_par_init_list=[],
    flg_train_Sigma_free_par_list=[],
    name="",
    dtype=torch.float64,
    sigma_n_num=None,
    device=None,
):
    """
    Returns a Volterra PK GP:
    the kernel is the sum of poloy GP (one for each deg)
    """
    # init the GP list
    gp_list = []
    # # get the first order contribution (with noise)
    # gp_list.append(Poly_GP(active_dims, poly_deg=1,
    #                        sigma_n_init=sigma_n_init, flg_train_sigma_n=flg_train_sigma_n,
    #                        Sigma_function=Sigma_function_list[0], Sigma_f_additional_par_list=Sigma_f_additional_par_list[0],
    #                        Sigma_pos_par_init=Sigma_pos_par_init_list[0], flg_train_Sigma_pos_par=flg_train_Sigma_pos_par_list[0],
    #                        Sigma_free_par_init=None, flg_train_Sigma_free_par=False,
    #                        flg_offset=flg_offset,
    #                        name=name+'_PK1', dtype=torch.float64, sigma_n_num=sigma_n_num, device=None))
    # # get the higher order contributions
    # for deg in range(1, poly_deg):
    #     gp_list.append(Poly_GP(active_dims, poly_deg=deg+1,
    #                            sigma_n_init=None, flg_train_sigma_n=False,
    #                            Sigma_function=Sigma_function_list[deg], Sigma_f_additional_par_list=Sigma_f_additional_par_list[deg],
    #                            Sigma_pos_par_init=Sigma_pos_par_init_list[deg], flg_train_Sigma_pos_par=flg_train_Sigma_pos_par_list[deg],
    #                            flg_offset=False,
    #                            name=name+'_PK'+str(deg+1), dtype=torch.float64, sigma_n_num=None, device=None))
    # get the first order contribution (with noise)
    gp_list.append(
        Poly_GP(
            active_dims,
            poly_deg=1,
            sigma_n_init=sigma_n_init,
            flg_train_sigma_n=flg_train_sigma_n,
            Sigma_function=Sigma_function_list[0],
            Sigma_f_additional_par_list=Sigma_f_additional_par_list[0],
            Sigma_pos_par_init=Sigma_pos_par_init_list[0],
            flg_train_Sigma_pos_par=flg_train_Sigma_pos_par_list[0],
            Sigma_free_par_init=Sigma_free_par_init_list[0],
            flg_train_Sigma_free_par=flg_train_Sigma_free_par_list[0],
            flg_offset=flg_offset,
            name=name + "_PK1",
            dtype=torch.float64,
            sigma_n_num=sigma_n_num,
            device=None,
        )
    )
    # get the higher order contributions
    for deg in range(1, poly_deg):
        gp_list.append(
            Poly_GP(
                active_dims,
                poly_deg=deg + 1,
                sigma_n_init=None,
                flg_train_sigma_n=False,
                Sigma_function=Sigma_function_list[deg],
                Sigma_f_additional_par_list=Sigma_f_additional_par_list[deg],
                Sigma_pos_par_init=Sigma_pos_par_init_list[deg],
                flg_train_Sigma_pos_par=flg_train_Sigma_pos_par_list[deg],
                Sigma_free_par_init=Sigma_free_par_init_list[deg],
                flg_train_Sigma_free_par=flg_train_Sigma_free_par_list[deg],
                flg_offset=False,
                name=name + "_PK" + str(deg + 1),
                dtype=torch.float64,
                sigma_n_num=None,
                device=None,
            )
        )
    # return the sum of the GPs
    return GP_prior.Sum_Independent_GP(*gp_list)


def get_Volterra_MPK_GP(
    active_dims,
    poly_deg,
    flg_offset,
    sigma_n_init=None,
    flg_train_sigma_n=False,
    Sigma_function_list=None,
    Sigma_f_additional_par_list=None,
    Sigma_pos_par_init_list=[],
    flg_train_Sigma_pos_par_list=[],
    Sigma_free_par_init_list=[],
    flg_train_Sigma_free_par_list=[],
    name="",
    dtype=torch.float64,
    sigma_n_num=None,
    device=None,
):
    """
    Returns a Volterra PK GP:
    the kernel is the sum of poly GP (one for each deg)
    """
    # init the GP list
    gp_list = []
    # gp_list.append(Poly_GP(active_dims, poly_deg=1,
    #                        sigma_n_init=sigma_n_init, flg_train_sigma_n=flg_train_sigma_n,
    #                        Sigma_function=Sigma_function_list[0], Sigma_f_additional_par_list=Sigma_f_additional_par_list[0],
    #                        Sigma_pos_par_init=Sigma_pos_par_init_list[0], flg_train_Sigma_pos_par=flg_train_Sigma_pos_par_list[0],
    #                        Sigma_free_par_init=Sigma_free_par_init_list[0], flg_train_Sigma_free_par=flg_train_Sigma_free_par_list[0],
    #                        flg_offset=flg_offset,
    #                        name=name+'_MPK1', dtype=torch.float64, sigma_n_num=sigma_n_num, device=None))
    gp_list.append(
        Linear_GP(
            active_dims,
            sigma_n_init=sigma_n_init,
            flg_train_sigma_n=flg_train_sigma_n,
            Sigma_function=Sigma_function_list[0],
            Sigma_f_additional_par_list=Sigma_f_additional_par_list[0],
            Sigma_pos_par_init=Sigma_pos_par_init_list[0],
            flg_train_Sigma_pos_par=flg_train_Sigma_pos_par_list[0],
            Sigma_free_par_init=Sigma_free_par_init_list[0],
            flg_train_Sigma_free_par=flg_train_Sigma_free_par_list[0],
            flg_offset=flg_offset,
            scale_init=np.ones(1),
            flg_train_scale=False,
            name=name + "_DEG_1",
            dtype=dtype,
            sigma_n_num=sigma_n_num,
            device=device,
        )
    )
    # get the higher order contributions
    for deg in range(1, poly_deg):
        gp_list.append(
            MPK_GP(
                active_dims,
                poly_deg=deg + 1,
                sigma_n_init=None,
                flg_train_sigma_n=False,
                Sigma_pos_par_init=Sigma_pos_par_init_list[deg],
                flg_train_Sigma_pos_par=flg_train_Sigma_pos_par_list[deg],
                flg_offset=False,
                name=name + "_DEG_" + str(deg + 1),
                dtype=dtype,
                sigma_n_num=None,
                device=device,
            )
        )
    # return the sum of the GPs
    return GP_prior.Sum_Independent_GP(*gp_list)
