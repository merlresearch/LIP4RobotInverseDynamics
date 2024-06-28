# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import sys

import numpy as np
import pytest
import torch

sys.path.append("../")

from gpr_lib.GP_prior.LK.get_K_blocks_GIP_sum_PANDA_7dof_no_subs_diag_torch import (
    get_K_blocks_GIP_sum_PANDA_7dof_no_subs as f_K_blocks_diag_LIP,
)
from gpr_lib.GP_prior.LK.get_K_blocks_GIP_sum_PANDA_7dof_no_subs_L_Y_cov_torch import (
    get_K_blocks_GIP_sum_PANDA_7dof_no_subs as f_K_blocks_L_Y_cov_LIP,
)
from gpr_lib.GP_prior.LK.get_K_blocks_GIP_sum_PANDA_7dof_no_subs_ltr_torch import (
    get_K_blocks_GIP_sum_PANDA_7dof_no_subs as f_K_blocks_ltr_LIP,
)
from gpr_lib.GP_prior.LK.get_K_blocks_GIP_sum_PANDA_7dof_no_subs_T_Y_cov_torch import (
    get_K_blocks_GIP_sum_PANDA_7dof_no_subs as f_K_blocks_T_Y_cov_LIP,
)
from gpr_lib.GP_prior.LK.get_K_blocks_GIP_sum_PANDA_7dof_no_subs_torch import (
    get_K_blocks_GIP_sum_PANDA_7dof_no_subs as f_K_blocks_LIP,
)
from gpr_lib.GP_prior.LK.get_K_blocks_GIP_sum_PANDA_7dof_no_subs_U_Y_cov_torch import (
    get_K_blocks_GIP_sum_PANDA_7dof_no_subs as f_K_blocks_U_Y_cov_LIP,
)
from gpr_lib.GP_prior.LK.get_K_blocks_RBF_1_sc_7dof_no_subs_diag_torch import (
    get_K_blocks_RBF_1_sc_7dof_no_subs as f_K_blocks_diag_LSE,
)
from gpr_lib.GP_prior.LK.get_K_blocks_RBF_1_sc_7dof_no_subs_L_Y_cov_torch import (
    get_K_blocks_RBF_1_sc_7dof_no_subs as f_K_blocks_L_Y_cov_LSE,
)
from gpr_lib.GP_prior.LK.get_K_blocks_RBF_1_sc_7dof_no_subs_ltr_torch import (
    get_K_blocks_RBF_1_sc_7dof_no_subs as f_K_blocks_ltr_LSE,
)
from gpr_lib.GP_prior.LK.get_K_blocks_RBF_1_sc_7dof_no_subs_T_Y_cov_torch import (
    get_K_blocks_RBF_1_sc_7dof_no_subs as f_K_blocks_T_Y_cov_LSE,
)
from gpr_lib.GP_prior.LK.get_K_blocks_RBF_1_sc_7dof_no_subs_torch import (
    get_K_blocks_RBF_1_sc_7dof_no_subs as f_K_blocks_LSE,
)
from gpr_lib.GP_prior.LK.get_K_blocks_RBF_1_sc_7dof_no_subs_U_Y_cov_torch import (
    get_K_blocks_RBF_1_sc_7dof_no_subs as f_K_blocks_U_Y_cov_LSE,
)
from Models_Lagrangian_kernel import m_GP_LK_GIP_sum, m_GP_LK_RBF_1


def get_dummy_model(kernel_name: str):
    if kernel_name == "LIP":
        return get_dummy_LIP_model()
    elif kernel_name == "LSE":
        return get_dummy_LSE_model()
    else:
        raise Exception("kernel name must be either 'LIP' or 'LSE'.")


def get_dummy_LIP_model():
    n = 7
    robot_structure = [0] * n
    pos_indices = list(range(n))
    vel_indices = list(range(n, 2 * n))
    acc_indices = list(range(2 * n, 3 * n))

    init_param_dict = {}
    init_param_dict["sigma_kin_vel_list_init"] = [np.ones(joint_index_1) for joint_index_1 in range(1, n + 1)]
    init_param_dict["flg_train_sigma_kin_vel"] = True
    sigma_kin_pos_list = []
    sigma_pot_list = []
    for joint_index_1 in range(n):
        for joint_index_2 in range(joint_index_1 + 1):
            if robot_structure[joint_index_2] == 1:
                sigma_kin_pos_list.append(np.ones(2))
            else:
                sigma_kin_pos_list.append(np.ones(3))
        if robot_structure[joint_index_1] == 1:
            sigma_pot_list.append(np.ones(2))
        else:
            sigma_pot_list.append(np.ones(3))
    init_param_dict["sigma_kin_pos_list_init"] = sigma_kin_pos_list
    init_param_dict["flg_train_sigma_kin_pos"] = True
    init_param_dict["sigma_pot_list_init"] = sigma_pot_list
    init_param_dict["flg_train_sigma_pot"] = True

    m = m_GP_LK_GIP_sum(
        num_dof=n,
        pos_indices=pos_indices,
        vel_indices=vel_indices,
        acc_indices=acc_indices,
        init_param_dict=init_param_dict,
        f_K_blocks=f_K_blocks_LIP,
        f_K_blocks_ltr=f_K_blocks_ltr_LIP,
        f_K_blocks_diag=f_K_blocks_diag_LIP,
        f_K_L_Y_blocks=f_K_blocks_L_Y_cov_LIP,
        f_K_T_Y_blocks=f_K_blocks_T_Y_cov_LIP,
        f_K_U_Y_blocks=f_K_blocks_U_Y_cov_LIP,
        name="test_LIP",
        device=torch.device("cpu"),
        sigma_n_init=0.1,
        sigma_n_num=1e-4,
        flg_norm_noise=False,
    )
    return m


def get_dummy_LSE_model():
    n = 7
    pos_indices = list(range(n))
    vel_indices = list(range(n, 2 * n))
    acc_indices = list(range(2 * n, 3 * n))

    init_param_dict = {}
    init_param_dict["lengthscales_L_init"] = np.ones(n * 3)
    init_param_dict["flg_train_lengthscales_L"] = True
    init_param_dict["scale_L_init"] = np.ones(1)
    init_param_dict["flg_train_scale_L"] = True

    m = m_GP_LK_RBF_1(
        num_dof=n,
        pos_indices=pos_indices,
        vel_indices=vel_indices,
        acc_indices=acc_indices,
        init_param_dict=init_param_dict,
        f_K_blocks=f_K_blocks_LSE,
        f_K_blocks_ltr=f_K_blocks_ltr_LSE,
        f_K_blocks_diag=f_K_blocks_diag_LSE,
        f_K_L_Y_blocks=f_K_blocks_L_Y_cov_LSE,
        f_K_T_Y_blocks=f_K_blocks_T_Y_cov_LSE,
        f_K_U_Y_blocks=f_K_blocks_U_Y_cov_LSE,
        name="test_LIP",
        device=torch.device("cpu"),
        sigma_n_init=0.1,
        sigma_n_num=1e-4,
        flg_norm_noise=False,
    )
    return m


def test_inertia_matrix_shape():
    m_LIP = get_dummy_LIP_model()
    m_LSE = get_dummy_LSE_model()

    n = m_LIP.num_dof
    N1 = 10
    N2 = 20
    X1 = np.random.randn(N1, 3 * n)
    Y = np.random.randn(N1, n)
    X2 = np.random.randn(N2, 3 * n)

    with torch.no_grad():
        alpha_LIP, _ = m_LIP.get_alpha(
            torch.tensor(X1, dtype=m_LIP.dtype, device=m_LIP.device),
            torch.tensor(Y, dtype=m_LIP.dtype, device=m_LIP.device),
        )

        alpha_LSE, _ = m_LIP.get_alpha(
            torch.tensor(X1, dtype=m_LSE.dtype, device=m_LSE.device),
            torch.tensor(Y, dtype=m_LSE.dtype, device=m_LSE.device),
        )

        M_LIP = m_LIP.get_M_estimates_T(
            torch.tensor(X1, dtype=m_LIP.dtype, device=m_LIP.device),
            torch.tensor(X2, dtype=m_LIP.dtype, device=m_LIP.device),
            alpha_LIP,
        )
        M_LSE = m_LSE.get_M_estimates(
            torch.tensor(X1, dtype=m_LSE.dtype, device=m_LSE.device),
            torch.tensor(X2, dtype=m_LSE.dtype, device=m_LSE.device),
            alpha_LSE,
        )

    expected_shape = (7, 7)
    assert M_LIP[0, :, :].shape == expected_shape
    assert M_LSE[0, :, :].shape == expected_shape


def test_inertia_matrix_simmetry():
    m_LIP = get_dummy_LIP_model()
    m_LSE = get_dummy_LSE_model()

    n = m_LIP.num_dof
    N1 = 10
    N2 = 20
    X1 = np.random.randn(N1, 3 * n)
    Y = np.random.randn(N1, n)
    X2 = np.random.randn(N2, 3 * n)

    with torch.no_grad():
        alpha_LIP, _ = m_LIP.get_alpha(
            torch.tensor(X1, dtype=m_LIP.dtype, device=m_LIP.device),
            torch.tensor(Y, dtype=m_LIP.dtype, device=m_LIP.device),
        )

        alpha_LSE, _ = m_LIP.get_alpha(
            torch.tensor(X1, dtype=m_LSE.dtype, device=m_LSE.device),
            torch.tensor(Y, dtype=m_LSE.dtype, device=m_LSE.device),
        )

        M_LIP = m_LIP.get_M_estimates_T(
            torch.tensor(X1, dtype=m_LIP.dtype, device=m_LIP.device),
            torch.tensor(X2, dtype=m_LIP.dtype, device=m_LIP.device),
            alpha_LIP,
        )
        M_LSE = m_LSE.get_M_estimates(
            torch.tensor(X1, dtype=m_LSE.dtype, device=m_LSE.device),
            torch.tensor(X2, dtype=m_LSE.dtype, device=m_LSE.device),
            alpha_LSE,
        )
    assert M_LIP[0, :, :] == pytest.approx(torch.transpose(M_LIP[0, :, :], 0, 1))
    assert M_LSE[0, :, :] == pytest.approx(torch.transpose(M_LSE[0, :, :], 0, 1))
    # assert (M_LSE[0,:,:].shape == expected_shape)


def test_inertial_component():
    m_LIP = get_dummy_LIP_model()
    m_LSE = get_dummy_LSE_model()

    n = m_LIP.num_dof
    N1 = 10
    N2 = 20
    X1 = np.random.randn(N1, 3 * n)
    Y = np.random.randn(N1, n)
    X2 = np.random.randn(N2, 3 * n)
    with torch.no_grad():
        alpha_LIP, _ = m_LIP.get_alpha(
            torch.tensor(X1, dtype=m_LIP.dtype, device=m_LIP.device),
            torch.tensor(Y, dtype=m_LIP.dtype, device=m_LIP.device),
        )
        alpha_LSE, _ = m_LIP.get_alpha(
            torch.tensor(X1, dtype=m_LSE.dtype, device=m_LSE.device),
            torch.tensor(Y, dtype=m_LSE.dtype, device=m_LSE.device),
        )
        Y_hat_LIP, _ = m_LIP.get_estimate_from_alpha(
            torch.tensor(X1, dtype=m_LIP.dtype, device=m_LIP.device),
            torch.tensor(X2, dtype=m_LIP.dtype, device=m_LIP.device),
            alpha_LIP,
        )
        Y_hat_LSE, _ = m_LSE.get_estimate_from_alpha(
            torch.tensor(X1, dtype=m_LSE.dtype, device=m_LSE.device),
            torch.tensor(X2, dtype=m_LSE.dtype, device=m_LSE.device),
            alpha_LSE,
        )
        Y_m_LIP = m_LIP.get_m_estimates(
            torch.tensor(X1, dtype=m_LIP.dtype, device=m_LIP.device),
            torch.tensor(X2, dtype=m_LIP.dtype, device=m_LIP.device),
            alpha_LIP,
        )
        Y_c_LIP = m_LIP.get_c_estimates(
            torch.tensor(X1, dtype=m_LIP.dtype, device=m_LIP.device),
            torch.tensor(X2, dtype=m_LIP.dtype, device=m_LIP.device),
            alpha_LIP,
        )
        Y_g_LIP = m_LIP.get_g_estimates(
            torch.tensor(X1, dtype=m_LIP.dtype, device=m_LIP.device),
            torch.tensor(X2, dtype=m_LIP.dtype, device=m_LIP.device),
            alpha_LIP,
        )
        Y_m_LSE = m_LSE.get_m_estimates(
            torch.tensor(X1, dtype=m_LSE.dtype, device=m_LSE.device),
            torch.tensor(X2, dtype=m_LSE.dtype, device=m_LSE.device),
            alpha_LSE,
        )
        Y_c_LSE = m_LSE.get_c_estimates(
            torch.tensor(X1, dtype=m_LSE.dtype, device=m_LSE.device),
            torch.tensor(X2, dtype=m_LSE.dtype, device=m_LSE.device),
            alpha_LSE,
        )
        Y_g_LSE = m_LSE.get_g_estimates(
            torch.tensor(X1, dtype=m_LSE.dtype, device=m_LSE.device),
            torch.tensor(X2, dtype=m_LSE.dtype, device=m_LSE.device),
            alpha_LSE,
        )

    assert Y_m_LIP == pytest.approx(Y_hat_LIP - Y_c_LIP - Y_g_LIP)
    assert Y_m_LSE == pytest.approx(Y_hat_LSE - Y_c_LSE - Y_g_LSE, abs=1e-5)


def test_coriolis_component():
    m_LIP = get_dummy_LIP_model()
    m_LSE = get_dummy_LSE_model()

    n = m_LIP.num_dof
    N1 = 10
    N2 = 20
    X1 = np.random.randn(N1, 3 * n)
    Y = np.random.randn(N1, n)
    X2 = np.random.randn(N2, 3 * n)
    with torch.no_grad():
        alpha_LIP, _ = m_LIP.get_alpha(
            torch.tensor(X1, dtype=m_LIP.dtype, device=m_LIP.device),
            torch.tensor(Y, dtype=m_LIP.dtype, device=m_LIP.device),
        )
        alpha_LSE, _ = m_LIP.get_alpha(
            torch.tensor(X1, dtype=m_LSE.dtype, device=m_LSE.device),
            torch.tensor(Y, dtype=m_LSE.dtype, device=m_LSE.device),
        )
        Y_hat_LIP, _ = m_LIP.get_estimate_from_alpha(
            torch.tensor(X1, dtype=m_LIP.dtype, device=m_LIP.device),
            torch.tensor(X2, dtype=m_LIP.dtype, device=m_LIP.device),
            alpha_LIP,
        )
        Y_hat_LSE, _ = m_LSE.get_estimate_from_alpha(
            torch.tensor(X1, dtype=m_LSE.dtype, device=m_LSE.device),
            torch.tensor(X2, dtype=m_LSE.dtype, device=m_LSE.device),
            alpha_LSE,
        )
        Y_m_LIP = m_LIP.get_m_estimates(
            torch.tensor(X1, dtype=m_LIP.dtype, device=m_LIP.device),
            torch.tensor(X2, dtype=m_LIP.dtype, device=m_LIP.device),
            alpha_LIP,
        )
        Y_c_LIP = m_LIP.get_c_estimates(
            torch.tensor(X1, dtype=m_LIP.dtype, device=m_LIP.device),
            torch.tensor(X2, dtype=m_LIP.dtype, device=m_LIP.device),
            alpha_LIP,
        )
        Y_g_LIP = m_LIP.get_g_estimates(
            torch.tensor(X1, dtype=m_LIP.dtype, device=m_LIP.device),
            torch.tensor(X2, dtype=m_LIP.dtype, device=m_LIP.device),
            alpha_LIP,
        )
        Y_m_LSE = m_LSE.get_m_estimates(
            torch.tensor(X1, dtype=m_LSE.dtype, device=m_LSE.device),
            torch.tensor(X2, dtype=m_LSE.dtype, device=m_LSE.device),
            alpha_LSE,
        )
        Y_c_LSE = m_LSE.get_c_estimates(
            torch.tensor(X1, dtype=m_LSE.dtype, device=m_LSE.device),
            torch.tensor(X2, dtype=m_LSE.dtype, device=m_LSE.device),
            alpha_LSE,
        )
        Y_g_LSE = m_LSE.get_g_estimates(
            torch.tensor(X1, dtype=m_LSE.dtype, device=m_LSE.device),
            torch.tensor(X2, dtype=m_LSE.dtype, device=m_LSE.device),
            alpha_LSE,
        )

    assert Y_c_LIP == pytest.approx(
        Y_hat_LIP - Y_m_LIP - Y_g_LIP,
    )
    assert Y_c_LSE == pytest.approx(Y_hat_LSE - Y_m_LSE - Y_g_LSE, abs=1e-5)


def test_gravity_component():
    m_LIP = get_dummy_LIP_model()
    m_LSE = get_dummy_LSE_model()

    n = m_LIP.num_dof
    N1 = 10
    N2 = 20
    X1 = np.random.randn(N1, 3 * n)
    Y = np.random.randn(N1, n)
    X2 = np.random.randn(N2, 3 * n)
    with torch.no_grad():
        alpha_LIP, _ = m_LIP.get_alpha(
            torch.tensor(X1, dtype=m_LIP.dtype, device=m_LIP.device),
            torch.tensor(Y, dtype=m_LIP.dtype, device=m_LIP.device),
        )
        alpha_LSE, _ = m_LIP.get_alpha(
            torch.tensor(X1, dtype=m_LSE.dtype, device=m_LSE.device),
            torch.tensor(Y, dtype=m_LSE.dtype, device=m_LSE.device),
        )
        Y_hat_LIP, _ = m_LIP.get_estimate_from_alpha(
            torch.tensor(X1, dtype=m_LIP.dtype, device=m_LIP.device),
            torch.tensor(X2, dtype=m_LIP.dtype, device=m_LIP.device),
            alpha_LIP,
        )
        Y_hat_LSE, _ = m_LSE.get_estimate_from_alpha(
            torch.tensor(X1, dtype=m_LSE.dtype, device=m_LSE.device),
            torch.tensor(X2, dtype=m_LSE.dtype, device=m_LSE.device),
            alpha_LSE,
        )
        Y_m_LIP = m_LIP.get_m_estimates(
            torch.tensor(X1, dtype=m_LIP.dtype, device=m_LIP.device),
            torch.tensor(X2, dtype=m_LIP.dtype, device=m_LIP.device),
            alpha_LIP,
        )
        Y_c_LIP = m_LIP.get_c_estimates(
            torch.tensor(X1, dtype=m_LIP.dtype, device=m_LIP.device),
            torch.tensor(X2, dtype=m_LIP.dtype, device=m_LIP.device),
            alpha_LIP,
        )
        Y_g_LIP = m_LIP.get_g_estimates(
            torch.tensor(X1, dtype=m_LIP.dtype, device=m_LIP.device),
            torch.tensor(X2, dtype=m_LIP.dtype, device=m_LIP.device),
            alpha_LIP,
        )
        Y_m_LSE = m_LSE.get_m_estimates(
            torch.tensor(X1, dtype=m_LSE.dtype, device=m_LSE.device),
            torch.tensor(X2, dtype=m_LSE.dtype, device=m_LSE.device),
            alpha_LSE,
        )
        Y_c_LSE = m_LSE.get_c_estimates(
            torch.tensor(X1, dtype=m_LSE.dtype, device=m_LSE.device),
            torch.tensor(X2, dtype=m_LSE.dtype, device=m_LSE.device),
            alpha_LSE,
        )
        Y_g_LSE = m_LSE.get_g_estimates(
            torch.tensor(X1, dtype=m_LSE.dtype, device=m_LSE.device),
            torch.tensor(X2, dtype=m_LSE.dtype, device=m_LSE.device),
            alpha_LSE,
        )

    assert Y_g_LIP == pytest.approx(Y_hat_LIP - Y_m_LIP - Y_c_LIP)
    assert Y_g_LSE == pytest.approx(Y_hat_LSE - Y_m_LSE - Y_c_LSE, abs=1e-5)


if __name__ == "__main__":
    print("Testing the inertia estimation...")
    test_inertia_matrix_shape()
    test_inertia_matrix_simmetry()
    print("Testing the inertia torque component...")
    test_inertial_component()
    print("Testing the coriolis torque component...")
    test_coriolis_component()
    print("Testing the gravitational torque component...")
    test_gravity_component()
