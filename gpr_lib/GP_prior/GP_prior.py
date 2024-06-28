"""
Author: Alberto Dalla Libera
This file contains the high level definition of the GP object.

1)GP_prior: superclass of the GP objects:
Define initialization and methods common to all the GP objects
In this way, to define a new GP object we just need to extend GP_prior and define
the methods get_mean(), get_covariance(), get_diag_covariance()
"""

import time

import numpy as np
import torch


class GP_prior(torch.nn.Module):
    """
    Superclass of each GP model (this class extends torch.nn.Module)
    """

    def __init__(
        self,
        active_dims,
        sigma_n_init=None,
        flg_train_sigma_n=False,
        scale_init=np.ones(1),
        flg_train_scale=False,
        f_mean=None,
        f_mean_add_par_dict={},
        pos_par_mean_init=None,
        flg_train_pos_par_mean=False,
        free_par_mean_init=None,
        flg_train_free_par_mean=False,
        name="",
        dtype=torch.float64,
        sigma_n_num=None,
        device=torch.device("cpu"),
    ):
        """
        Initialize the Module object and set flags regarding:
        - noise
         sigma_n_int = initial noise (None = no noise)
         flg_train_sigma_n = set to true to train the noise parameter
        - sigma_n_num = used to compensate numerical noise
        - active dims = indices of the input selected by the GP model to compute mean and covariance
        - data type
        - device
        - name
        - scaling parameters:
          scale_init = initialization of the scaling parameters
          flg_train_scale = set to true to train the scaling parameter
        """
        # initilize the Module object
        super().__init__()
        # set name device and type
        self.name = name
        self.dtype = dtype
        self.device = device
        # active dims
        self.active_dims = torch.tensor(active_dims, requires_grad=False, device=device)
        # set sigma_n_log (the log of the noise standard deviation)
        if sigma_n_init is None:
            self.GP_with_noise = False
        else:
            self.GP_with_noise = True
            self.sigma_n_log = torch.nn.Parameter(
                torch.tensor(np.log(sigma_n_init), dtype=self.dtype, device=self.device),
                requires_grad=flg_train_sigma_n,
            )
        # scaling parameters
        self.scale_log = torch.nn.Parameter(
            torch.tensor(np.log(scale_init), dtype=self.dtype, device=self.device), requires_grad=flg_train_scale
        )
        # standard deviation of the rounding errors compensation
        if sigma_n_num is not None:
            self.sigma_n_num = torch.tensor(sigma_n_num, dtype=self.dtype, device=self.device)
        else:
            self.sigma_n_num = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        # check/set mean parameters
        self.check_mean(
            f_mean,
            f_mean_add_par_dict,
            pos_par_mean_init,
            flg_train_pos_par_mean,
            free_par_mean_init,
            flg_train_free_par_mean,
        )

    def check_mean(
        self,
        f_mean,
        f_mean_add_par_dict,
        pos_par_mean_init,
        flg_train_pos_par_mean,
        free_par_mean_init,
        flg_train_free_par_mean,
    ):
        if f_mean is None:
            self.flg_mean = False
        else:
            self.flg_mean = True
            self.f_mean = f_mean
            self.f_mean_add_par_dict = f_mean_add_par_dict
            self.pos_par_mean_log = torch.nn.Parameter(
                torch.tensor(np.log(pos_par_mean_init), dtype=self.dtype, device=self.device),
                requires_grad=flg_train_pos_par_mean,
            )
            self.free_par_mean = torch.nn.Parameter(
                torch.tensor(free_par_mean_init, dtype=self.dtype, device=self.device),
                requires_grad=flg_train_free_par_mean,
            )

    def to(self, dev):
        """
        Set the device and move the parameters
        """
        # set the new device
        super().to(dev)
        self.device = dev
        # move the model parameters in the new device
        self.sigma_n_num = self.sigma_n_num.to(dev)

    def get_sigma_n_2(self):
        """
        Returns the variance of the noise (measurement noise + rounding noise)
        """
        return torch.exp(self.sigma_n_log) ** 2 + self.sigma_n_num**2

    # def forward_for_estimate(self, X):
    #     """
    #     Returns mean, variance and inverse of the variance.
    #     The inversion is computed with torch.inverse()
    #     """
    #     N = X.size()[0]
    #     if self.GP_with_noise:
    #         K_X = self.get_covariance(X, flg_noise=True)
    #     else:
    #         K_X = self.get_covariance(X)
    #     K_X_inv = torch.inverse(K_X)
    #     m_X = self.get_mean(X)
    #     return m_X, K_X, K_X_inv

    def forward(self, X, p_drop=0):
        """
        Returns the prior distribution and the inverse anf the log_det of the prior covariace

        input:
        - X = training inputs (X has dimension [num_samples, num_features])

        output:
        - m_X = prior mean of X
        - K_X = prior covariance of X
        - K_X_inv = inverse of the prior covariance
        - log_det = log det of the prior covariance
        """
        # get the covariance
        if self.GP_with_noise:
            K_X = self.get_covariance(X, flg_noise=True, p_drop=p_drop)
        else:
            K_X = self.get_covariance(X, p_drop=p_drop)
        # get inverse and log_det with cholesky

        # U = torch.cholesky(K_X, upper=True)
        # log_det = 2*torch.sum(torch.log(torch.diag(U)))
        # K_X_inv = torch.cholesky_inverse(U, upper=True)

        L = torch.linalg.cholesky(K_X)
        log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        K_X_inv = torch.cholesky_inverse(L)

        # get the mean
        m_X = self.get_mean(X)
        # return the values
        return m_X, K_X, K_X_inv, log_det

    def get_mean(self, X):
        """
        Returns the prior mean of X
        X is assumed to be a tenso of dimension [num_samples, num_features]
        """
        if self.flg_mean:
            return self.f_mean(X, torch.exp(self.pos_par_mean_log), self.free_par_mean, **self.f_mean_add_par_dict)
        else:
            return 0.0

    def get_covariance(self, X1, X2=None, flg_noise=False, p_drop=0):
        """
        Returns the covariance betweeen the input locations X1 X2.
        If X2 is None X2 is assumed to be equal to X1
        """
        raise NotImplementedError()

    def get_diag_covariance(self, X, flg_noise=False):
        """
        Returns the diagonal elements of the covariance betweeen the input locations X1
        """
        raise NotImplementedError()

    def get_alpha(self, X, Y):
        """
        Returns alpha, the vector of coefficients defining the posterior distribution

        inputs:
        - X = training input
        - Y = training output

        outputs:
        - alpha = vector defining the posterior distribution
        - m_X = prior mean of X
        - K_X_inv = inverse of the prior covariance of X
        """
        m_X, _, K_X_inv, _ = self.forward(X)
        alpha = torch.matmul(K_X_inv, Y - m_X)
        return alpha, m_X, K_X_inv

    def get_estimate_from_alpha(self, X, X_test, alpha, m_X, K_X_inv=None, Y_test=None):
        """
        Compute the posterior distribution in X_test, given the alpha vector.

        input:
        - X = training input locations (used to compute alpha)
        - X_test = test input locations
        - alpha = vector of coefficients defining the posterior
        - m_X = prior mean of X
        - K_X_inv = inverse of the prior covariance of X
        - Y_test = value of the output in the test location

        output:
        - Y_hat = posterior mean
        - var = diagonal elements of the posterior variance

        If K_X_inv is given the method returns also the confidence intervals (variance of the gaussian)
        If Y_test is given the method prints the MSE
        """
        # get covariance and prior mean
        K_X_test_X = self.get_covariance(X_test, X)
        m_X_test = self.get_mean(X_test)
        # get the estimate
        Y_hat = m_X_test + torch.matmul(K_X_test_X, alpha)
        # print the MSE if Y_test is given
        if not (Y_test is None):
            print("MSE:", torch.sum((Y_test - Y_hat) ** 2) / Y_test.size()[0])
        # if K_X_inv is given compute the confidence intervals
        if K_X_inv is not None:
            # num_test = X_test.size()[0]
            var = self.get_diag_covariance(X_test) - torch.sum(torch.matmul(K_X_test_X, K_X_inv) * (K_X_test_X), dim=1)
        else:
            var = None
        return Y_hat, var, m_X_test

    def get_estimate(self, X, Y, X_test, Y_test=None, flg_return_K_X_inv=False):
        """
        Returns the posterior distribution in X_test, given the training samples X Y.

        input:
        - X = training input
        - Y = training output
        - X_test = test input
        - Y_test = value of the output in the test location

        output:
        - Y_hat = mean of the test posterior
        - var = diagonal elements of the variance posterior
        - alpha = coefficients defining the posterior
        - m_X = prior mean of the training samples
        - K_X_inv = inverse of the training covariance

        The function returns:
           -a vector containing the sigma squared confidence intervals
           -the vector of the coefficient
           -the K_X inverse in case required through flg_return_K_X_inv"""
        # get the coefficent and the mean
        alpha, m_X, K_X_inv = self.get_alpha(X, Y)
        # get the estimate and the confidence intervals
        Y_hat, var, m_X_test = self.get_estimate_from_alpha(X, X_test, alpha, m_X, K_X_inv=K_X_inv, Y_test=Y_test)
        # return the opportune values
        if flg_return_K_X_inv:
            return Y_hat, var, alpha, m_X, K_X_inv
        else:
            return Y_hat, var, alpha, m_X_test

    def print_model(self):
        """
        Print the model parameters
        """
        print(self.name + " parameters:")
        for par_name, par in self.named_parameters():
            if par.requires_grad:
                print("-", par_name, ":", par.data)

    def fit_model(
        self,
        trainloader=None,
        optimizer=None,
        criterion=None,
        N_epoch=1,
        N_epoch_print=1,
        f_saving_model=None,
        f_print=None,
        p_drop=0.0,
    ):
        """
        Optimize the model hyperparameters

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
        t_start = time.time()
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
                out_GP_priors = self(inputs, p_drop=p_drop)
                loss = criterion(out_GP_priors, labels)
                loss.backward(retain_graph=False)
                optimizer.step()
                # update the running loss
                running_loss = running_loss + loss.item()
                N_btc = N_btc + 1
            # print statistics and save the model
            if epoch % N_epoch_print == 0:
                print("\nEPOCH:", epoch)
                self.print_model()
                print("Runnng loss:", running_loss / N_btc)
                t_stop = time.time()
                print("Time elapsed:", t_stop - t_start)
                if f_saving_model is not None:
                    f_saving_model(epoch)
                if f_print is not None:
                    f_print()
                t_start = time.time()
        # print the final parameters
        print("\nFinal parameters:")
        self.print_model()

    def __add__(self, other_GP_prior):
        """
        Returns a new GP given by the sum of two GP
        """
        return Sum_Independent_GP(self, other_GP_prior)

    def __mul__(self, other_GP_prior):
        """
        Returns a new GP with mean given by the product of the means and covariances
        """
        return Multiply_GP_prior(self, other_GP_prior)


class Combine_GP(GP_prior):
    """
    Class that extend GP_prior and provide common utilities to combine GP
    """

    def __init__(self, *gp_priors_obj):
        """
        Initialize the multiple kernel object
        """
        # initialize a new GP object
        super().__init__(
            active_dims=[],
            sigma_n_num=None,
            scale_init=np.ones(1),
            flg_train_scale=False,
            dtype=gp_priors_obj[0].dtype,
            device=gp_priors_obj[0].device,
        )
        # build a list with all the models
        self.gp_list = torch.nn.ModuleList(gp_priors_obj)
        # check the noise flag
        GP_with_noise = False
        for gp in self.gp_list:
            GP_with_noise = GP_with_noise or gp.GP_with_noise
        self.GP_with_noise = GP_with_noise

    def to(self, dev):
        """
        Move all the models to the desired device
        """
        super().to(dev)
        self.device = dev
        for gp in self.gp_list:
            gp.to(dev)

    def print_model(self):
        """
        Print the parameters of all the models in the gp_list
        """
        for gp in self.gp_list:
            gp.print_model()

    def get_sigma_n_2(self):
        """
        Iterate over all the models in the list and returns the global noise variance
        """
        sigma_n_2 = torch.zeros(1, dtype=self.dtype, device=self.device)
        for gp in self.gp_list:
            if gp.GP_with_noise:
                sigma_n_2 += gp.get_sigma_n_2()
        return sigma_n_2


class Sum_Independent_GP(Combine_GP):
    """
    Class that sum GP_priors objects
    """

    def __init__(self, *gp_priors_obj):
        """
        Initialize the gp list
        """
        super().__init__(*gp_priors_obj)

    def get_mean(self, X):
        """
        Returns the sum of the means returned by the models in gp_list
        """
        N = X.size()[0]
        mean = torch.zeros(N, 1, dtype=self.dtype, device=self.device)
        for gp in self.gp_list:
            mean += gp.get_mean(X)
            return mean

    def get_covariance(self, X1, X2=None, flg_noise=False, p_drop=0):
        """
        Returns the sum of the covariances returned by the mdoels in gp_list
        """
        # get dimensions
        N1 = X1.size()[0]
        if X2 is None:
            N2 = N1
        else:
            N2 = X2.size()[0]

        # initialize the covariance
        cov = torch.zeros(N1, N2, dtype=self.dtype, device=self.device)
        # sum all the covariances
        for gp in self.gp_list:
            cov += gp.get_covariance(X1, X2, flg_noise=False)

        # # compute and sum covariances
        # cov = torch.stack([gp.get_covariance(X1,X2, flg_noise=False)
        #                    for gp in self.gp_list]).sum(0)

        # add the noise
        if flg_noise & self.GP_with_noise:
            cov = cov + self.get_sigma_n_2() * torch.eye(N1, dtype=self.dtype, device=self.device)
        return cov

    def get_diag_covariance(self, X, flg_noise=False):
        """
        Returns the sum of the diagonals of the covariances in gp list
        """
        # initialize the vector
        diag = torch.zeros(X.size()[0], dtype=self.dtype, device=self.device)
        # iterate in the list and sum the diagonals
        for gp in self.gp_list:
            diag += gp.get_diag_covariance(X, flg_noise=False)
        # add the noise
        if flg_noise & self.GP_with_noise:
            diag += self.get_sigma_n_2()
        return diag


class Multiply_GP_prior(Combine_GP):
    """
    Class that generates a GP_prior multiplying GP_priors objects
    """

    def __init__(self, *gp_priors_obj):
        """
        Initilize the GP list
        """
        super().__init__(*gp_priors_obj)

    def get_mean(self, X):
        """
        Returns the product of the means returned by the models in gp_list
        """
        # initilize the mean vector
        N = X.size()[0]
        mean = torch.ones(N, 1, dtype=self.dtype, device=self.device)
        # multiply all the means
        for gp in self.gp_list:
            mean = mean * gp.get_mean(X)
        return mean

    def get_covariance(self, X1, X2=None, flg_noise=False, p_drop=0):
        """
        Returns the element-wise product of the covariances returned by the models in gp_list
        """
        # get size
        N1 = X1.size()[0]
        if X2 is None:
            N2 = N1
        else:
            N2 = X2.size()[0]

        # initilize the covariance
        cov = torch.ones(N1, N2, dtype=self.dtype, device=self.device)
        # multiply all the covariances
        for gp in self.gp_list:
            cov *= gp.get_covariance(X1, X2, flg_noise=False)

        # # compute and multiply covariances
        # cov = torch.stack([gp.get_covariance(X1,X2, flg_noise=False)
        #                    for gp in self.gp_list]).prod(0)

        # add the noise
        if flg_noise & self.GP_with_noise:
            cov = cov + self.get_sigma_n_2() * torch.eye(N1, dtype=self.dtype, device=self.device)
        return cov

    def get_diag_covariance(self, X, flg_noise=False):
        """
        Returns the product of the diagonal elements of the covariance returned by the models in gp_list
        """
        # initilize the diagona
        N = X.size()[0]
        diag = torch.ones(N, dtype=self.dtype, device=self.device)
        # multiply all the diagonals
        for gp in self.gp_list:
            diag *= gp.get_diag_covariance(X, flg_noise=False)
        # add the nosie
        if flg_noise & self.GP_with_noise:
            diag += self.get_sigma_n_2()
        return diag


def Scale_GP_prior(
    GP_prior_class,
    GP_prior_par_dict,
    f_scale,
    active_dims_f_scale,
    pos_par_f_init=None,
    flg_train_pos_par_f=True,
    free_par_f_init=None,
    flg_train_free_par_f=True,
    additional_par_f_list=[],
):
    """
    Funciton that returns a GP_prior scaled. This class implement the following model:
    y(x) = a(x)f(x) + e, where f(x) is a GP and a(x) a deterministic function.
    The function a(x) can be parametrize with respect to a set of trainable prameters.
    This class retuns an instance of a new class defined dynamically in the following
    """

    # define dynamically the new class
    class Scaled_GP(GP_prior_class):
        """
        Class that extends the GP_prior_class with the scaling parameters
        """

        def __init__(
            self,
            GP_prior_par_dict,
            f_scale,
            active_dims_f_scale,
            pos_par_f_init,
            flg_train_pos_par_f,
            free_par_f_init,
            flg_train_free_par_f,
            additional_par_f_list,
        ):
            # initialize the object of the superclass
            super().__init__(**GP_prior_par_dict)
            # save the scaling info
            self.f_scale = f_scale
            self.active_dims_f_scale = active_dims_f_scale
            self.additional_par_f_list = additional_par_f_list
            if pos_par_f_init is None:
                self.flg_pos_par = False
                self.pos_par_f_log = None
            else:
                self.flg_pos_par = True
                self.pos_par_f_log = torch.nn.Parameter(
                    torch.tensor(np.log(pos_par_f_init), dtype=self.dtype, device=self.device),
                    requires_grad=flg_train_pos_par_f,
                )
            if free_par_f_init is None:
                self.flg_free_par = False
                self.free_par_f = None
            else:
                self.flg_free_par = True
                self.free_par_f = torch.nn.Parameter(
                    torch.tensor(free_par_f_init, dtype=self.dtype, device=self.device),
                    requires_grad=flg_train_free_par_f,
                )

        def get_scaling(self, X):
            """
            Returns the scaling funciton evaluated in X
            """
            if self.flg_pos_par:
                pos_par = torch.exp(self.pos_par_f_log)
            else:
                pos_par = None
            return self.f_scale(
                X[:, self.active_dims_f_scale], pos_par, self.free_par_f, *self.additional_par_f_list
            ).reshape(-1)

        def get_mean(self, X):
            """
            Calls the get_mean of the superclass and apply the scaling
            """
            # get the supercalss mean and scale the result
            return self.get_scaling(X) * super().get_mean(X)

        def get_covariance(self, X1, X2=None, flg_noise=False, p_drop=0):
            """
            Calls the get_covariance of the superclass and apply the scaling
            """
            # get the scaling functions
            a_X1 = self.get_scaling(X1).reshape([-1, 1])
            # if required evaluate the scaling function in X2 and get the covariance
            if X2 is None:
                K = a_X1 * super().get_covariance(X1, X2, flg_noise=False) * (a_X1.transpose(0, 1))
            else:
                a_X2 = self.get_scaling(X2).reshape([-1, 1])
                K = a_X1 * super().get_covariance(X1, X2, flg_noise=False) * (a_X2.transpose(0, 1))
            # if required add the noise and return the covariance
            if flg_noise & self.GP_with_noise:
                N = K.size()[0]
                return K + self.get_sigma_n_2() * torch.eye(N, dtype=self.dtype, device=self.device)
            else:
                return K

        def get_diag_covariance(self, X, flg_noise=False):
            """
            Calls the get_diag_covariance of the superclass and apply the scaling
            """
            # evaluate the scaling function in X1
            a_X = self.get_scaling(X)
            diag = a_X**2 * super().get_diag_covariance(X, flg_noise=False)
            # if required add the noise and return the covariance
            if flg_noise & self.GP_with_noise:
                return diag + self.get_sigma_n_2()
            else:
                return diag

    # return an object of the new class
    return Scaled_GP(
        GP_prior_par_dict,
        f_scale,
        active_dims_f_scale,
        pos_par_f_init,
        flg_train_pos_par_f,
        free_par_f_init,
        flg_train_free_par_f,
        additional_par_f_list,
    )
