"""
Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
Utils functions to generate Lagrangian-equations-kernel from K_T and K_U:
K_T = kinetic-energy kernel
K_U = potential-energy kernel
"""

import re

import sympy
from sympy.printing.numpy import NumPyPrinter

# ---------------------------- GENERIC KERNELS ----------------------------------


def RBF(x1, x2, l, s):
    """
    RBF kernel single element
    """
    return s * sympy.exp(-sum([((x1[i] - x2[i]) / l[i]) ** 2 for i in range(len(l))]))


def POLY(x1, x2, sigma, d, flg_offset=True):
    """
    Polynomial kernel with degree d and offset single element
    """
    if flg_offset:
        return (sum([x1[i] * x2[i] * sigma[i] for i in range(len(x1))] + [sigma[-1]])) ** d
    else:
        return (sum([x1[i] * x2[i] * sigma[i] for i in range(len(x1))])) ** d


def MPK(x1, x2, sigma, d, flg_offset=True):
    """
    Polynomial kernel with degree d and offset single element
    """
    out = 1.0
    for deg in range(d):
        if flg_offset:
            out = out * (sum([x1[i] * x2[i] * sigma[deg][i] for i in range(len(x1))] + [sigma[deg][-1]]))
        else:
            out = out * (sum([x1[i] * x2[i] * sigma[deg][i] for i in range(len(x1))]))
    return out


def GIP_lagrangian(
    x1, x2, robot_structure, sigma_kin_vel, sigma_kin_pos_prism, sigma_pot_prism, sigma_kin_pos_rev, sigma_pot_rev
):
    """
    GIP kernel
    """
    num_dof = len(robot_structure)
    K_kin_vel = POLY(x1[num_dof : 2 * num_dof], x2[num_dof : 2 * num_dof], sigma=sigma_kin_vel, d=2, flg_offset=False)
    K_kin_pos = 1.0
    K_pot = 1.0
    index_prism = 0
    index_rev = 0
    for joint_index in range(num_dof):
        if robot_structure[joint_index] == 1:
            K_kin_pos = K_kin_pos * POLY(
                [x1[joint_index]],
                [x2[joint_index]],
                sigma=sigma_kin_pos_prism[index_prism],
                # s=s_kin_vel_prism[index_prism],
                d=2,
                flg_offset=True,
            )
            K_pot = K_pot * POLY(
                [x1[joint_index]],
                [x2[joint_index]],
                sigma=sigma_pot_prism[index_prism],
                # s=s_pot_prism[index_prism],
                d=1,
                flg_offset=True,
            )
            index_prism = index_prism + 1
        else:
            K_kin_pos = K_kin_pos * POLY(
                [sympy.sin(x1[joint_index]), sympy.cos(x1[joint_index])],
                [sympy.sin(x2[joint_index]), sympy.cos(x2[joint_index])],
                sigma=sigma_kin_pos_rev[index_rev],
                # s=s_kin_pos_rev[index_rev],
                d=2,
                flg_offset=True,
            )
            K_pot = K_pot * POLY(
                [sympy.sin(x1[joint_index]), sympy.cos(x1[joint_index])],
                [sympy.sin(x2[joint_index]), sympy.cos(x2[joint_index])],
                sigma=sigma_pot_rev[index_rev],
                # s=s_pot_rev[index_rev],
                d=1,
                flg_offset=True,
            )
            index_rev = index_rev + 1
    return K_kin_vel * K_kin_pos + K_pot, K_kin_vel * K_kin_pos, K_pot


def GIP_vel_lagrangian(
    x1, x2, robot_structure, sigma_kin_vel, sigma_kin_pos_prism, sigma_pot_prism, sigma_kin_pos_rev, sigma_pot_rev
):
    """
    GIP kernel with vel monomials explicit
    """
    num_dof = len(robot_structure)
    x1_vel = [x1[num_dof + i] * x1[num_dof + j] for i in range(num_dof) for j in range(i, num_dof)]
    x2_vel = [x2[num_dof + i] * x2[num_dof + j] for i in range(num_dof) for j in range(i, num_dof)]
    K_kin_vel = POLY(x1_vel, x2_vel, sigma=sigma_kin_vel, d=1, flg_offset=False)
    K_kin_pos = 1.0
    K_pot = 1.0
    index_prism = 0
    index_rev = 0
    for joint_index in range(num_dof):
        if robot_structure[joint_index] == 1:
            K_kin_pos = K_kin_pos * POLY(
                [x1[joint_index]],
                [x2[joint_index]],
                sigma=sigma_kin_pos_prism[index_prism],
                # s=s_kin_vel_prism[index_prism],
                d=2,
                flg_offset=True,
            )
            K_pot = K_pot * POLY(
                [x1[joint_index]],
                [x2[joint_index]],
                sigma=sigma_pot_prism[index_prism],
                # s=s_pot_prism[index_prism],
                d=1,
                flg_offset=True,
            )
            index_prism = index_prism + 1
        else:
            K_kin_pos = K_kin_pos * POLY(
                [sympy.sin(x1[joint_index]), sympy.cos(x1[joint_index])],
                [sympy.sin(x2[joint_index]), sympy.cos(x2[joint_index])],
                sigma=sigma_kin_pos_rev[index_rev],
                # s=s_kin_pos_rev[index_rev],
                d=2,
                flg_offset=True,
            )
            K_pot = K_pot * POLY(
                [sympy.sin(x1[joint_index]), sympy.cos(x1[joint_index])],
                [sympy.sin(x2[joint_index]), sympy.cos(x2[joint_index])],
                sigma=sigma_pot_rev[index_rev],
                # s=s_pot_rev[index_rev],
                d=1,
                flg_offset=True,
            )
            index_rev = index_rev + 1
    return K_kin_vel * K_kin_pos + K_pot, K_kin_vel * K_kin_pos, K_pot


def GIP_sum_lagrangian(x1, x2, robot_structure, sigma_kin_vel_list, sigma_kin_pos_list, sigma_pot_list):
    """
    GIP kernel
    """
    num_dof = len(robot_structure)
    K_kin_list = []
    K_pot = 1.0
    pos_index = 0
    for joint_index in range(0, num_dof):
        # kinetik energy
        K_kin_i = POLY(
            x1[num_dof : num_dof + joint_index + 1],
            x2[num_dof : num_dof + joint_index + 1],
            sigma=sigma_kin_vel_list[joint_index],
            d=2,
            flg_offset=False,
        )
        for joint_index_ in range(0, joint_index + 1):
            if robot_structure[joint_index_] == 1:
                K_kin_i = K_kin_i * POLY(
                    [x1[joint_index_]], [x2[joint_index_]], sigma=sigma_kin_pos_list[pos_index], d=2, flg_offset=True
                )
            else:
                K_kin_i = K_kin_i * POLY(
                    [sympy.sin(x1[joint_index_]), sympy.cos(x1[joint_index_])],
                    [sympy.sin(x2[joint_index_]), sympy.cos(x2[joint_index_])],
                    sigma=sigma_kin_pos_list[pos_index],
                    d=2,
                    flg_offset=True,
                )
            pos_index += 1
        K_kin_list.append(K_kin_i)
        # potential energy
        if robot_structure[joint_index] == 1:
            K_pot = K_pot * POLY(
                [x1[joint_index]], [x2[joint_index]], sigma=sigma_pot_list[joint_index], d=1, flg_offset=True
            )
        else:
            K_pot = K_pot * POLY(
                [sympy.sin(x1[joint_index]), sympy.cos(x1[joint_index])],
                [sympy.sin(x2[joint_index]), sympy.cos(x2[joint_index])],
                sigma=sigma_pot_list[joint_index],
                d=1,
                flg_offset=True,
            )
    return sum(K_kin_list) + K_pot, sum(K_kin_list), K_pot


def RBF_VEL(x1, x2, l, s, dq_11, dq_12, dq_21, dq_22, c_dq):
    """
    RBF kernel scaled by square of velocities
    """
    return (
        c_dq
        * dq_11
        * dq_12
        * dq_21
        * dq_22
        * s
        * sympy.exp(-sum([((x1[i] - x2[i]) / l[i]) ** 2 for i in range(len(l))]))
    )


# ---------------------------- LAGRANGIAN KERNELS ----------------------------------


def gen_symbolic_inputs(n):
    q_names_i = ["q_i" + str(joint_index + 1) for joint_index in range(n)]
    dq_names_i = ["dq_i" + str(joint_index + 1) for joint_index in range(n)]
    ddq_names_i = ["ddq_i" + str(joint_index + 1) for joint_index in range(n)]
    q_names_j = ["q_j" + str(joint_index + 1) for joint_index in range(n)]
    dq_names_j = ["dq_j" + str(joint_index + 1) for joint_index in range(n)]
    ddq_names_j = ["ddq_j" + str(joint_index + 1) for joint_index in range(n)]
    q_i = sympy.symbols(q_names_i)
    dq_i = sympy.symbols(dq_names_i)
    ddq_i = sympy.symbols(ddq_names_i)
    q_j = sympy.symbols(q_names_j)
    dq_j = sympy.symbols(dq_names_j)
    ddq_j = sympy.symbols(ddq_names_j)
    return q_i, dq_i, ddq_i, q_j, dq_j, ddq_j


def get_Lagrangian_kernel_RBF(n, flg_subs=True):
    """
    Generate the kinetic and potential energy kernels and derive the
    Lagrangian kernel equations
    Prior on the energies defined by RBF kernels
    """

    # generate inputs
    q_i, dq_i, ddq_i, q_j, dq_j, ddq_j = gen_symbolic_inputs(n)
    # generate symbolicparameters
    lT_names = ["lT" + str(joint_index + 1) for joint_index in range(2 * n)]
    lU_names = ["lU" + str(joint_index + 1) for joint_index in range(n)]
    lT = sympy.symbols(lT_names)
    lU = sympy.symbols(lU_names)
    sT = sympy.symbols("sT")
    sU = sympy.symbols("sU")
    K_T = sympy.symbols("K_T")
    K_U = sympy.symbols("K_U")

    # get the kinetic and potential energy kernel
    f_K_T = RBF(x1=q_i + dq_i, x2=q_j + dq_j, l=lT, s=sT)
    f_K_U = RBF(x1=q_i, x2=q_j, l=lU, s=sU)
    if flg_subs:
        sub_expr_U = f_K_U
        sub_expr_T = f_K_T
    else:
        sub_expr_U = sub_expr_T = []
    f_K_U_L = [
        potential_op(f_K_U=f_K_U, q_i=q_i, q_j=q_j, joint_index_i=joint_index_i, joint_index_j=joint_index_j).subs(
            sub_expr_U, K_U
        )
        for joint_index_i in range(n)
        for joint_index_j in range(n)
    ]
    f_K_U_Y = [
        potential_op(
            f_K_U=f_K_U, q_i=q_i, q_j=q_j, joint_index_i=-1, joint_index_j=joint_index_j, flg_first_index_diff=False
        ).subs(sub_expr_U, K_U)
        for joint_index_j in range(n)
    ]

    f_K_T_L = [
        kinetic_op(
            f_K_T=f_K_T,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=joint_index_i,
            joint_index_j=joint_index_j,
            n=n,
        ).subs(sub_expr_T, K_T)
        for joint_index_i in range(n)
        for joint_index_j in range(n)
    ]
    f_K_T_Y = [
        kinetic_op(
            f_K_T=f_K_T,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=-1,
            joint_index_j=joint_index_j,
            n=n,
            flg_first_index_diff=False,
        ).subs(sub_expr_T, K_T)
        for joint_index_j in range(n)
    ]

    args = [q_i, dq_i, ddq_i, q_j, dq_j, ddq_j, lT, lU, sT, sU, K_T, K_U]

    return f_K_T, f_K_U, f_K_U_L, f_K_T_L, args, f_K_U_Y, f_K_T_Y


def get_Lagrangian_kernel_RBF_1(n, flg_subs=True):
    """
    Generate a kernel for the entire Langrangian energy and derive the
    Lagrangian kernel equations
    Prior on the energies defined by RBF kernels
    """

    # generate inputs
    q_i, dq_i, ddq_i, q_j, dq_j, ddq_j = gen_symbolic_inputs(n)
    # generate symbolicparameters
    lL_names = ["lL" + str(joint_index + 1) for joint_index in range(2 * n)]
    lL = sympy.symbols(lL_names)
    sL = sympy.symbols("sL")
    K_L = sympy.symbols("K_L")

    # get the kinetic and potential energy kernel
    f_K_L = RBF(x1=q_i + dq_i, x2=q_j + dq_j, l=lL, s=sL)
    if flg_subs:
        sub_expr_L = f_K_L
    else:
        sub_expr_L = []
    f_K_L_L = [
        kinetic_op(
            f_K_T=f_K_L,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=joint_index_i,
            joint_index_j=joint_index_j,
            n=n,
        ).subs(sub_expr_L, K_L)
        for joint_index_i in range(n)
        for joint_index_j in range(n)
    ]
    f_K_L_Y = [
        kinetic_op(
            f_K_T=f_K_L,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=-1,
            joint_index_j=joint_index_j,
            n=n,
            flg_first_index_diff=False,
        ).subs(sub_expr_L, K_L)
        for joint_index_j in range(n)
    ]

    args = [q_i, dq_i, ddq_i, q_j, dq_j, ddq_j, lL, sL, K_L]

    return f_K_L, f_K_L_L, args, f_K_L_Y


def get_Lagrangian_kernel_RBF_1_sc(n, flg_subs=True):
    """
    Generate a kernel for the entire Langrangian energy and derive the
    Lagrangian kernel equations
    Prior on the energies defined by RBF kernels
    """

    # generate inputs
    q_i, dq_i, ddq_i, q_j, dq_j, ddq_j = gen_symbolic_inputs(n)
    # generate symbolicparameters
    lL_names = ["lL" + str(joint_index + 1) for joint_index in range(3 * n)]
    lL = sympy.symbols(lL_names)
    sL = sympy.symbols("sL")
    K_L = sympy.symbols("K_L")

    # get the kinetic and potential energy kernel
    x_i_cs = [sympy.cos(q_ii) for q_ii in q_i] + [sympy.sin(q_ii) for q_ii in q_i] + dq_i
    x_j_cs = [sympy.cos(q_ji) for q_ji in q_j] + [sympy.sin(q_ji) for q_ji in q_j] + dq_j
    f_K_L = RBF(x1=x_i_cs, x2=x_j_cs, l=lL, s=sL)
    if flg_subs:
        sub_expr_L = f_K_L
        f_K_L_L = [
            kinetic_op(
                f_K_T=f_K_L,
                q_i=q_i,
                q_j=q_j,
                dq_i=dq_i,
                dq_j=dq_j,
                ddq_i=ddq_i,
                ddq_j=ddq_j,
                joint_index_i=joint_index_i,
                joint_index_j=joint_index_j,
                n=n,
            ).subs(sub_expr_L, K_L)
            for joint_index_i in range(n)
            for joint_index_j in range(n)
        ]
        f_K_L_Y = [
            kinetic_op(
                f_K_T=f_K_L,
                q_i=q_i,
                q_j=q_j,
                dq_i=dq_i,
                dq_j=dq_j,
                ddq_i=ddq_i,
                ddq_j=ddq_j,
                joint_index_i=-1,
                joint_index_j=joint_index_j,
                n=n,
                flg_first_index_diff=False,
            ).subs(sub_expr_L, K_L)
            for joint_index_j in range(n)
        ]
    else:
        f_K_L_L = [
            kinetic_op(
                f_K_T=f_K_L,
                q_i=q_i,
                q_j=q_j,
                dq_i=dq_i,
                dq_j=dq_j,
                ddq_i=ddq_i,
                ddq_j=ddq_j,
                joint_index_i=joint_index_i,
                joint_index_j=joint_index_j,
                n=n,
            )
            for joint_index_i in range(n)
            for joint_index_j in range(n)
        ]
        f_K_L_Y = [
            kinetic_op(
                f_K_T=f_K_L,
                q_i=q_i,
                q_j=q_j,
                dq_i=dq_i,
                dq_j=dq_j,
                ddq_i=ddq_i,
                ddq_j=ddq_j,
                joint_index_i=-1,
                joint_index_j=joint_index_j,
                n=n,
                flg_first_index_diff=False,
            )
            for joint_index_j in range(n)
        ]

    args = [q_i, dq_i, ddq_i, q_j, dq_j, ddq_j, lL, sL, K_L]

    return f_K_L, f_K_L_L, args, f_K_L_Y


def get_Lagrangian_kernel_POLY_RBF(n, flg_subs=True):
    """
    Generate the kinetic and potential energy kernels and derive the
    Lagrangian kernel equations
    Prior on the kinetic energy: POLY(vel)*RBF(pos)
    Prior on the potential energy: *RBF(pos)
    """

    # generate inputs
    q_i, dq_i, ddq_i, q_j, dq_j, ddq_j = gen_symbolic_inputs(n)
    # generate symbolic parameters
    lT_names = ["lT" + str(joint_index + 1) for joint_index in range(n)]
    sigmaT_names = ["sigmaT" + str(joint_index + 1) for joint_index in range(n + 1)]
    lU_names = ["lU" + str(joint_index + 1) for joint_index in range(n)]
    lT = sympy.symbols(lT_names)
    sigmaT = sympy.symbols(sigmaT_names)
    lU = sympy.symbols(lU_names)
    sT = sympy.symbols("sT")
    sU = sympy.symbols("sU")
    K_T = sympy.symbols("K_T")
    K_U = sympy.symbols("K_U")

    # get the kinetic and potential energy kernel
    f_K_T = RBF(x1=q_i, x2=q_j, l=lT, s=sT) * POLY(x1=dq_i, x2=dq_j, sigma=sigmaT, d=2, flg_offset=False)
    f_K_U = RBF(x1=q_i, x2=q_j, l=lU, s=sU)

    if flg_subs:
        sub_expr_U = f_K_U
        sub_expr_T = f_K_T
    else:
        sub_expr_U = sub_expr_T = []
    f_K_U_L = [
        potential_op(f_K_U=f_K_U, q_i=q_i, q_j=q_j, joint_index_i=joint_index_i, joint_index_j=joint_index_j).subs(
            sub_expr_U, K_U
        )
        for joint_index_i in range(n)
        for joint_index_j in range(n)
    ]

    f_K_U_Y = [
        potential_op(
            f_K_U=f_K_U, q_i=q_i, q_j=q_j, joint_index_i=-1, joint_index_j=joint_index_j, flg_first_index_diff=False
        ).subs(sub_expr_U, K_U)
        for joint_index_j in range(n)
    ]

    f_K_T_L = [
        kinetic_op(
            f_K_T=f_K_T,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=joint_index_i,
            joint_index_j=joint_index_j,
            n=n,
        ).subs(sub_expr_T, K_T)
        for joint_index_i in range(n)
        for joint_index_j in range(n)
    ]

    f_K_T_Y = [
        kinetic_op(
            f_K_T=f_K_T,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=-1,
            joint_index_j=joint_index_j,
            n=n,
            flg_first_index_diff=False,
        ).subs(sub_expr_T, K_T)
        for joint_index_j in range(n)
    ]

    f_K_L_Y = [
        kinetic_op(
            f_K_T=f_K_T + f_K_U,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=-1,
            joint_index_j=joint_index_j,
            n=n,
            flg_first_index_diff=False,
        ).subs(sub_expr_T, K_T)
        for joint_index_j in range(n)
    ]

    args = [q_i, dq_i, ddq_i, q_j, dq_j, ddq_j, lT, lU, sT, sU, K_T, K_U]

    return f_K_T, f_K_U, f_K_U_L, f_K_T_L, args, f_K_L_Y, f_K_U_Y, f_K_T_Y


def get_Lagrangian_kernel_POLY_vel_RBF(n, flg_subs=True):
    """
    Generate the kinetic and potential energy kernels and derive the
    Lagrangian kernel equations
    Prior on the kinetic energy: POLY(vel)*RBF(pos)
    Prior on the potential energy: *RBF(pos)
    """

    # generate inputs
    q_i, dq_i, ddq_i, q_j, dq_j, ddq_j = gen_symbolic_inputs(n)
    # generate symbolic parameters
    lT_names = ["lT" + str(joint_index + 1) for joint_index in range(n)]
    sigmaT_names = [
        "sigmaT" + str(joint_index_1 + 1) + "_" + str(joint_index_2 + 1)
        for joint_index_1 in range(n)
        for joint_index_2 in range(joint_index_1, n)
    ]
    lU_names = ["lU" + str(joint_index + 1) for joint_index in range(n)]
    lT = sympy.symbols(lT_names)
    sigmaT = sympy.symbols(sigmaT_names)
    lU = sympy.symbols(lU_names)
    sT = sympy.symbols("sT")
    sU = sympy.symbols("sU")
    K_T = sympy.symbols("K_T")
    K_U = sympy.symbols("K_U")

    # get the kinetic and potential energy kernel
    x1_vel = [dq_i[i] * dq_i[j] for i in range(n) for j in range(i, n)]
    x2_vel = [dq_j[i] * dq_j[j] for i in range(n) for j in range(i, n)]
    f_K_T = RBF(x1=q_i, x2=q_j, l=lT, s=sT) * POLY(x1=x1_vel, x2=x2_vel, sigma=sigmaT, d=1, flg_offset=False)
    f_K_U = RBF(x1=q_i, x2=q_j, l=lU, s=sU)

    if flg_subs:
        sub_expr_U = f_K_U
        sub_expr_T = f_K_T
    else:
        sub_expr_U = sub_expr_T = []
    f_K_U_L = [
        potential_op(f_K_U=f_K_U, q_i=q_i, q_j=q_j, joint_index_i=joint_index_i, joint_index_j=joint_index_j).subs(
            sub_expr_U, K_U
        )
        for joint_index_i in range(n)
        for joint_index_j in range(n)
    ]

    f_K_U_Y = [
        potential_op(
            f_K_U=f_K_U, q_i=q_i, q_j=q_j, joint_index_i=-1, joint_index_j=joint_index_j, flg_first_index_diff=False
        ).subs(sub_expr_U, K_U)
        for joint_index_j in range(n)
    ]

    f_K_T_L = [
        kinetic_op(
            f_K_T=f_K_T,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=joint_index_i,
            joint_index_j=joint_index_j,
            n=n,
        ).subs(sub_expr_T, K_T)
        for joint_index_i in range(n)
        for joint_index_j in range(n)
    ]

    f_K_T_Y = [
        kinetic_op(
            f_K_T=f_K_T,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=-1,
            joint_index_j=joint_index_j,
            n=n,
            flg_first_index_diff=False,
        ).subs(sub_expr_T, K_T)
        for joint_index_j in range(n)
    ]

    f_K_L_Y = [
        kinetic_op(
            f_K_T=f_K_T + f_K_U,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=-1,
            joint_index_j=joint_index_j,
            n=n,
            flg_first_index_diff=False,
        ).subs(sub_expr_T, K_T)
        for joint_index_j in range(n)
    ]

    args = [q_i, dq_i, ddq_i, q_j, dq_j, ddq_j, lT, lU, sT, sU, K_T, K_U]

    return f_K_T, f_K_U, f_K_U_L, f_K_T_L, args, f_K_L_Y, f_K_U_Y, f_K_T_Y


def get_Lagrangian_kernel_GIP(n, robot_structure, flg_subs=True):
    """
    Models the lagrangian function with a GIP kernel
    and derives the kernel of the Lagrangian kernel equations
    Prior on the energies defined by RBF kernels
    """

    # generate inputs
    q_i, dq_i, ddq_i, q_j, dq_j, ddq_j = gen_symbolic_inputs(n)
    # generate symbolic parameters
    sigma_kin_vel_names = ["sigma_kin_vel" + str(joint_index + 1) for joint_index in range(n)]
    # prism names
    sigma_kin_pos_prism_names = []
    sigma_pot_prism_names = []
    # rev names
    sigma_kin_pos_rev_names = []
    sigma_pot_rev_names = []
    for joint_index in range(n):
        if robot_structure[joint_index] == 1:
            # generate prism names
            sigma_kin_pos_prism_names.append(
                [
                    "sigma_kin_pos_prism_j" + str(joint_index + 1) + "_1",
                    "sigma_kin_pos_prism_j" + str(joint_index + 1) + "_2",
                ]
            )
            sigma_pot_prism_names.append(
                ["sigma_pot_prism_j" + str(joint_index + 1) + "_1", "sigma_pot_prism_j" + str(joint_index + 1) + "_2"]
            )
        else:
            # generate rev names
            sigma_kin_pos_rev_names.append(
                [
                    "sigma_kin_pos_rev_j" + str(joint_index + 1) + "_1",
                    "sigma_kin_pos_rev_j" + str(joint_index + 1) + "_2",
                    "sigma_kin_pos_rev_j" + str(joint_index + 1) + "_3",
                ]
            )
            sigma_pot_rev_names.append(
                [
                    "sigma_pot_rev_j" + str(joint_index + 1) + "_1",
                    "sigma_pot_rev_j" + str(joint_index + 1) + "_2",
                    "sigma_pot_rev_j" + str(joint_index + 1) + "_3",
                ]
            )
    sigma_kin_vel = sympy.symbols(sigma_kin_vel_names)
    sigma_kin_pos_prism = sympy.symbols(sigma_kin_pos_prism_names)
    sigma_pot_prism = sympy.symbols(sigma_pot_prism_names)
    sigma_kin_pos_rev = sympy.symbols(sigma_kin_pos_rev_names)
    sigma_pot_rev = sympy.symbols(sigma_pot_rev_names)
    K_L = sympy.symbols("K_L")
    K_T = sympy.symbols("K_T")
    K_U = sympy.symbols("K_U")

    # get the kinetic and potential energy kernel
    f_K_L, f_K_T, f_K_U = GIP_lagrangian(
        x1=q_i + dq_i,
        x2=q_j + dq_j,
        robot_structure=robot_structure,
        sigma_kin_vel=sigma_kin_vel,
        sigma_kin_pos_prism=sigma_kin_pos_prism,
        sigma_pot_prism=sigma_pot_prism,
        sigma_kin_pos_rev=sigma_kin_pos_rev,
        sigma_pot_rev=sigma_pot_rev,
    )

    if flg_subs:
        sub_expr_L = f_K_L
        sub_expr_U = f_K_U
        sub_expr_T = f_K_T
    else:
        sub_expr_L = []
        sub_expr_U = []
        sub_expr_T = []
    f_K_L_L = [
        kinetic_op(
            f_K_T=f_K_L,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=joint_index_i,
            joint_index_j=joint_index_j,
            n=n,
        ).subs(sub_expr_L, K_L)
        for joint_index_i in range(n)
        for joint_index_j in range(n)
    ]
    f_K_L_Y = [
        kinetic_op(
            f_K_T=f_K_L,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=-1,
            joint_index_j=joint_index_j,
            n=n,
            flg_first_index_diff=False,
        ).subs(sub_expr_L, K_L)
        for joint_index_j in range(n)
    ]
    f_K_T_Y = [
        kinetic_op(
            f_K_T=f_K_T,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=-1,
            joint_index_j=joint_index_j,
            n=n,
            flg_first_index_diff=False,
        ).subs(sub_expr_T, K_T)
        for joint_index_j in range(n)
    ]
    f_K_U_Y = [
        potential_op(
            f_K_U=f_K_U, q_i=q_i, q_j=q_j, joint_index_i=-1, joint_index_j=joint_index_j, flg_first_index_diff=False
        ).subs(sub_expr_U, K_U)
        for joint_index_j in range(n)
    ]

    args = [
        q_i,
        dq_i,
        ddq_i,
        q_j,
        dq_j,
        ddq_j,
        sigma_kin_vel,
        sigma_kin_pos_prism,
        sigma_pot_prism,
        sigma_kin_pos_rev,
        sigma_pot_rev,
    ]

    return f_K_L, f_K_L_L, args, f_K_L_Y, f_K_T_Y, f_K_U_Y


def get_Lagrangian_kernel_GIP_vel(n, robot_structure, flg_subs=True):
    """
    Models the lagrangian function with a GIP kernel
    and derives the kernel of the Lagrangian kernel equations
    """

    # generate inputs
    q_i, dq_i, ddq_i, q_j, dq_j, ddq_j = gen_symbolic_inputs(n)
    # generate symbolic parameters
    sigma_kin_vel_names = [
        "sigma_kin_vel_" + str(joint_index_1 + 1) + "_" + str(joint_index_2 + 1)
        for joint_index_1 in range(n)
        for joint_index_2 in range(joint_index_1, n)
    ]
    # prism names
    sigma_kin_pos_prism_names = []
    sigma_pot_prism_names = []
    # rev names
    sigma_kin_pos_rev_names = []
    sigma_pot_rev_names = []
    for joint_index in range(n):
        if robot_structure[joint_index] == 1:
            # generate prism names
            sigma_kin_pos_prism_names.append(
                [
                    "sigma_kin_pos_prism_j" + str(joint_index + 1) + "_1",
                    "sigma_kin_pos_prism_j" + str(joint_index + 1) + "_2",
                ]
            )
            sigma_pot_prism_names.append(
                ["sigma_pot_prism_j" + str(joint_index + 1) + "_1", "sigma_pot_prism_j" + str(joint_index + 1) + "_2"]
            )
        else:
            # generate rev names
            sigma_kin_pos_rev_names.append(
                [
                    "sigma_kin_pos_rev_j" + str(joint_index + 1) + "_1",
                    "sigma_kin_pos_rev_j" + str(joint_index + 1) + "_2",
                    "sigma_kin_pos_rev_j" + str(joint_index + 1) + "_3",
                ]
            )
            sigma_pot_rev_names.append(
                [
                    "sigma_pot_rev_j" + str(joint_index + 1) + "_1",
                    "sigma_pot_rev_j" + str(joint_index + 1) + "_2",
                    "sigma_pot_rev_j" + str(joint_index + 1) + "_3",
                ]
            )
    sigma_kin_vel = sympy.symbols(sigma_kin_vel_names)
    sigma_kin_pos_prism = sympy.symbols(sigma_kin_pos_prism_names)
    sigma_pot_prism = sympy.symbols(sigma_pot_prism_names)
    sigma_kin_pos_rev = sympy.symbols(sigma_kin_pos_rev_names)
    sigma_pot_rev = sympy.symbols(sigma_pot_rev_names)
    K_L = sympy.symbols("K_L")
    K_T = sympy.symbols("K_T")
    K_U = sympy.symbols("K_U")

    # get the kinetic and potential energy kernel
    f_K_L, f_K_T, f_K_U = GIP_vel_lagrangian(
        x1=q_i + dq_i,
        x2=q_j + dq_j,
        robot_structure=robot_structure,
        sigma_kin_vel=sigma_kin_vel,
        sigma_kin_pos_prism=sigma_kin_pos_prism,
        sigma_pot_prism=sigma_pot_prism,
        sigma_kin_pos_rev=sigma_kin_pos_rev,
        sigma_pot_rev=sigma_pot_rev,
    )

    if flg_subs:
        sub_expr_L = f_K_L
        sub_expr_U = f_K_U
        sub_expr_T = f_K_T
    else:
        sub_expr_L = []
        sub_expr_U = []
        sub_expr_T = []
    f_K_L_L = [
        kinetic_op(
            f_K_T=f_K_L,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=joint_index_i,
            joint_index_j=joint_index_j,
            n=n,
        ).subs(sub_expr_L, K_L)
        for joint_index_i in range(n)
        for joint_index_j in range(n)
    ]
    f_K_L_Y = [
        kinetic_op(
            f_K_T=f_K_L,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=-1,
            joint_index_j=joint_index_j,
            n=n,
            flg_first_index_diff=False,
        ).subs(sub_expr_L, K_L)
        for joint_index_j in range(n)
    ]
    f_K_T_Y = [
        kinetic_op(
            f_K_T=f_K_T,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=-1,
            joint_index_j=joint_index_j,
            n=n,
            flg_first_index_diff=False,
        ).subs(sub_expr_T, K_T)
        for joint_index_j in range(n)
    ]
    f_K_U_Y = [
        potential_op(
            f_K_U=f_K_U, q_i=q_i, q_j=q_j, joint_index_i=-1, joint_index_j=joint_index_j, flg_first_index_diff=False
        ).subs(sub_expr_U, K_U)
        for joint_index_j in range(n)
    ]

    args = [
        q_i,
        dq_i,
        ddq_i,
        q_j,
        dq_j,
        ddq_j,
        sigma_kin_vel,
        sigma_kin_pos_prism,
        sigma_pot_prism,
        sigma_kin_pos_rev,
        sigma_pot_rev,
    ]

    return f_K_L, f_K_L_L, args, f_K_L_Y, f_K_T_Y, f_K_U_Y


def get_Lagrangian_kernel_GIP_sum(n, robot_structure, flg_subs=True):
    """
    Models the lagrangian function with a sum of GIP kernel
    and derives the kernel of the Lagrangian kernel equations
    """

    # generate inputs
    q_i, dq_i, ddq_i, q_j, dq_j, ddq_j = gen_symbolic_inputs(n)
    # generate symbolic parameters
    sigma_kin_vel_list_names = [
        [
            "sigma_kin_v_" + str(joint_index_1) + "_" + str(joint_index_2)
            for joint_index_2 in range(1, joint_index_1 + 1)
        ]
        for joint_index_1 in range(1, n + 1)
    ]
    sigma_kin_pos_list_names = []
    sigma_pot_list_names = []
    for joint_index_1 in range(n):
        # kin energy par names
        # sigma_kin_pos_i = []
        for joint_index_2 in range(joint_index_1 + 1):
            if robot_structure[joint_index_2] == 1:
                kin_pos_par = [
                    "sigma_kin_p_" + str(joint_index_1 + 1) + "_" + str(joint_index_2 + 1) + "_l",
                    "sigma_kin_p_" + str(joint_index_1 + 1) + "_" + str(joint_index_2 + 1) + "_off",
                ]
            else:
                kin_pos_par = [
                    "sigma_kin_p_" + str(joint_index_1 + 1) + "_" + str(joint_index_2 + 1) + "_s",
                    "sigma_kin_p_" + str(joint_index_1 + 1) + "_" + str(joint_index_2 + 1) + "_c",
                    "sigma_kin_p_" + str(joint_index_1 + 1) + "_" + str(joint_index_2 + 1) + "_off",
                ]
            sigma_kin_pos_list_names.append(kin_pos_par)
        # pot energy par names
        if robot_structure[joint_index_1] == 1:
            sigma_pot_list_names.append(
                ["sigma_pot_" + str(joint_index_1 + 1) + "_l", "sigma_pot_" + str(joint_index_1 + 1) + "_off"]
            )
        else:
            sigma_pot_list_names.append(
                [
                    "sigma_pot_" + str(joint_index_1 + 1) + "_s",
                    "sigma_pot_" + str(joint_index_1 + 1) + "_c",
                    "sigma_pot_" + str(joint_index_1 + 1) + "_off",
                ]
            )
    # generate symbols
    sigma_kin_vel_list = sympy.symbols(sigma_kin_vel_list_names)
    sigma_kin_pos_list = sympy.symbols(sigma_kin_pos_list_names)
    sigma_pot_list = sympy.symbols(sigma_pot_list_names)
    K_L = sympy.symbols("K_L")
    K_T = sympy.symbols("K_T")
    K_U = sympy.symbols("K_U")

    # get the kinetic and potential energy kernel
    f_K_L, f_K_T, f_K_U = GIP_sum_lagrangian(
        x1=q_i + dq_i,
        x2=q_j + dq_j,
        robot_structure=robot_structure,
        sigma_kin_vel_list=sigma_kin_vel_list,
        sigma_kin_pos_list=sigma_kin_pos_list,
        sigma_pot_list=sigma_pot_list,
    )

    if flg_subs:
        sub_expr_L = f_K_L
        sub_expr_U = f_K_U
        sub_expr_T = f_K_T
        f_K_L_L = [
            kinetic_op(
                f_K_T=f_K_L,
                q_i=q_i,
                q_j=q_j,
                dq_i=dq_i,
                dq_j=dq_j,
                ddq_i=ddq_i,
                ddq_j=ddq_j,
                joint_index_i=joint_index_i,
                joint_index_j=joint_index_j,
                n=n,
            ).subs(sub_expr_L, K_L)
            for joint_index_i in range(n)
            for joint_index_j in range(n)
        ]
        f_K_L_Y = [
            kinetic_op(
                f_K_T=f_K_L,
                q_i=q_i,
                q_j=q_j,
                dq_i=dq_i,
                dq_j=dq_j,
                ddq_i=ddq_i,
                ddq_j=ddq_j,
                joint_index_i=-1,
                joint_index_j=joint_index_j,
                n=n,
                flg_first_index_diff=False,
            ).subs(sub_expr_L, K_L)
            for joint_index_j in range(n)
        ]
        f_K_T_Y = [
            kinetic_op(
                f_K_T=f_K_T,
                q_i=q_i,
                q_j=q_j,
                dq_i=dq_i,
                dq_j=dq_j,
                ddq_i=ddq_i,
                ddq_j=ddq_j,
                joint_index_i=-1,
                joint_index_j=joint_index_j,
                n=n,
                flg_first_index_diff=False,
            ).subs(sub_expr_T, K_T)
            for joint_index_j in range(n)
        ]
        f_K_U_Y = [
            potential_op(
                f_K_U=f_K_U, q_i=q_i, q_j=q_j, joint_index_i=-1, joint_index_j=joint_index_j, flg_first_index_diff=False
            ).subs(sub_expr_U, K_U)
            for joint_index_j in range(n)
        ]

    else:
        f_K_L_L = [
            kinetic_op(
                f_K_T=f_K_L,
                q_i=q_i,
                q_j=q_j,
                dq_i=dq_i,
                dq_j=dq_j,
                ddq_i=ddq_i,
                ddq_j=ddq_j,
                joint_index_i=joint_index_i,
                joint_index_j=joint_index_j,
                n=n,
            )
            for joint_index_i in range(n)
            for joint_index_j in range(n)
        ]
        f_K_L_Y = [
            kinetic_op(
                f_K_T=f_K_L,
                q_i=q_i,
                q_j=q_j,
                dq_i=dq_i,
                dq_j=dq_j,
                ddq_i=ddq_i,
                ddq_j=ddq_j,
                joint_index_i=-1,
                joint_index_j=joint_index_j,
                n=n,
                flg_first_index_diff=False,
            )
            for joint_index_j in range(n)
        ]
        f_K_T_Y = [
            kinetic_op(
                f_K_T=f_K_T,
                q_i=q_i,
                q_j=q_j,
                dq_i=dq_i,
                dq_j=dq_j,
                ddq_i=ddq_i,
                ddq_j=ddq_j,
                joint_index_i=-1,
                joint_index_j=joint_index_j,
                n=n,
                flg_first_index_diff=False,
            )
            for joint_index_j in range(n)
        ]
        f_K_U_Y = [
            potential_op(
                f_K_U=f_K_U, q_i=q_i, q_j=q_j, joint_index_i=-1, joint_index_j=joint_index_j, flg_first_index_diff=False
            )
            for joint_index_j in range(n)
        ]

    args = [q_i, dq_i, ddq_i, q_j, dq_j, ddq_j, sigma_kin_vel_list, sigma_kin_pos_list, sigma_pot_list]

    return f_K_L, f_K_L_L, args, f_K_L_Y, f_K_T_Y, f_K_U_Y


def get_Lagrangian_kernel_RBF_M(n, flg_subs=True):
    """
    Generate the kinetic and potential energy kernels and derive the
    Lagrangian kernel equations
    Prior on the kinetic energy: dq^T*M(q)*dq approximated as a RBF(pos) on each element of M(q) scaled by velocities
    Prior on the potential energy: RBF(pos)
    """

    # generate inputs
    q_i, dq_i, ddq_i, q_j, dq_j, ddq_j = gen_symbolic_inputs(n)
    # generate symbolic parameters
    lT_list_names = [
        ["lT" + str(joint_index_1 + 1) + str(joint_index_2 + 1) + "_" + str(j + 1) for j in range(n)]
        for joint_index_1 in range(n)
        for joint_index_2 in range(joint_index_1 + 1)
    ]
    sT_list_names = [
        "sT" + str(joint_index_1 + 1) + str(joint_index_2 + 1)
        for joint_index_1 in range(n)
        for joint_index_2 in range(joint_index_1 + 1)
    ]
    lU_names = ["lU" + str(joint_index + 1) for joint_index in range(n)]
    lT_list = sympy.symbols(lT_list_names)
    sT_list = sympy.symbols(sT_list_names)
    lU = sympy.symbols(lU_names)
    sU = sympy.symbols("sU")
    K_T = sympy.symbols("K_T")
    K_U = sympy.symbols("K_U")

    # get the kinetic kernel
    f_K_T = 0
    current_index = 0
    for joint_index_1 in range(n):
        for joint_index_2 in range(joint_index_1 + 1):
            if joint_index_1 == joint_index_2:
                c = 1
            else:
                c = 4
            f_K_T = f_K_T + RBF_VEL(
                x1=q_i,
                x2=q_j,
                l=lT_list[current_index],
                s=sT_list[current_index],
                dq_11=dq_i[joint_index_1],
                dq_12=dq_i[joint_index_2],
                dq_21=dq_j[joint_index_1],
                dq_22=dq_j[joint_index_2],
                c_dq=c,
            )
            current_index = current_index + 1
    # get the potential kernel
    f_K_U = RBF(x1=q_i, x2=q_j, l=lU, s=sU)

    if flg_subs:
        sub_expr_U = f_K_U
        sub_expr_T = f_K_T
    else:
        sub_expr_U = sub_expr_T = []
    f_K_U_L = [
        potential_op(f_K_U=f_K_U, q_i=q_i, q_j=q_j, joint_index_i=joint_index_i, joint_index_j=joint_index_j).subs(
            sub_expr_U, K_U
        )
        for joint_index_i in range(n)
        for joint_index_j in range(n)
    ]

    f_K_U_Y = [
        potential_op(
            f_K_U=f_K_U, q_i=q_i, q_j=q_j, joint_index_i=-1, joint_index_j=joint_index_j, flg_first_index_diff=False
        ).subs(sub_expr_U, K_U)
        for joint_index_j in range(n)
    ]

    f_K_T_L = [
        kinetic_op(
            f_K_T=f_K_T,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=joint_index_i,
            joint_index_j=joint_index_j,
            n=n,
        ).subs(sub_expr_T, K_T)
        for joint_index_i in range(n)
        for joint_index_j in range(n)
    ]

    f_K_T_Y = [
        kinetic_op(
            f_K_T=f_K_T,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=-1,
            joint_index_j=joint_index_j,
            n=n,
            flg_first_index_diff=False,
        ).subs(sub_expr_T, K_T)
        for joint_index_j in range(n)
    ]

    f_K_L_Y = [
        kinetic_op(
            f_K_T=f_K_T + f_K_U,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=-1,
            joint_index_j=joint_index_j,
            n=n,
            flg_first_index_diff=False,
        ).subs(sub_expr_T, K_T)
        for joint_index_j in range(n)
    ]

    args = [q_i, dq_i, ddq_i, q_j, dq_j, ddq_j, lT_list, lU, sT_list, sU, K_T, K_U]

    return f_K_T, f_K_U, f_K_U_L, f_K_T_L, args, f_K_L_Y, f_K_U_Y, f_K_T_Y


def get_Lagrangian_kernel_POLY_vel_RBF_sum(n, flg_subs=True):
    """
    Generate the kinetic and potential energy kernels and derive the
    Lagrangian kernel equations
    Prior on the kinetic energy: sum of POLY(vel)*RBF(pos)
    Prior on the potential energy: *RBF(pos)
    """

    # generate inputs
    q_i, dq_i, ddq_i, q_j, dq_j, ddq_j = gen_symbolic_inputs(n)
    # generate symbolic parameters
    lT_names_list = [
        ["lT" + str(joint_index + 1) + "_" + str(joint_index_ + 1) for joint_index_ in range(joint_index + 1)]
        for joint_index in range(n)
    ]
    sigmaT_names_list = [
        ["sigmaT" + str(joint_index + 1) + "_" + str(joint_index_1 + 1) for joint_index_1 in range(joint_index + 1)]
        for joint_index in range(n)
    ]
    lU_names = ["lU" + str(joint_index + 1) for joint_index in range(n)]
    sT_names = ["sT" + str(joint_index + 1) for joint_index in range(n)]
    lT_list = sympy.symbols(lT_names_list)
    sigmaT_list = sympy.symbols(sigmaT_names_list)
    lU = sympy.symbols(lU_names)
    sT = sympy.symbols(sT_names)
    sU = sympy.symbols("sU")
    K_T = sympy.symbols("K_T")
    K_U = sympy.symbols("K_U")

    f_K_T = 0.0
    for joint_index in range(n):
        # get the kinetic and potential energy kernel
        f_K_T += RBF(
            x1=q_i[0 : joint_index + 1], x2=q_j[0 : joint_index + 1], l=lT_list[joint_index], s=sT[joint_index]
        ) * POLY(
            x1=dq_i[0 : joint_index + 1],
            x2=dq_j[0 : joint_index + 1],
            sigma=sigmaT_list[joint_index],
            d=2,
            flg_offset=False,
        )
    f_K_U = RBF(x1=q_i, x2=q_j, l=lU, s=sU)

    if flg_subs:
        sub_expr_U = f_K_U
        sub_expr_T = f_K_T
    else:
        sub_expr_U = sub_expr_T = []
    f_K_U_L = [
        potential_op(f_K_U=f_K_U, q_i=q_i, q_j=q_j, joint_index_i=joint_index_i, joint_index_j=joint_index_j).subs(
            sub_expr_U, K_U
        )
        for joint_index_i in range(n)
        for joint_index_j in range(n)
    ]

    f_K_U_Y = [
        potential_op(
            f_K_U=f_K_U, q_i=q_i, q_j=q_j, joint_index_i=-1, joint_index_j=joint_index_j, flg_first_index_diff=False
        ).subs(sub_expr_U, K_U)
        for joint_index_j in range(n)
    ]

    f_K_T_L = [
        kinetic_op(
            f_K_T=f_K_T,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=joint_index_i,
            joint_index_j=joint_index_j,
            n=n,
        ).subs(sub_expr_T, K_T)
        for joint_index_i in range(n)
        for joint_index_j in range(n)
    ]

    f_K_T_Y = [
        kinetic_op(
            f_K_T=f_K_T,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=-1,
            joint_index_j=joint_index_j,
            n=n,
            flg_first_index_diff=False,
        ).subs(sub_expr_T, K_T)
        for joint_index_j in range(n)
    ]

    f_K_L_Y = [
        kinetic_op(
            f_K_T=f_K_T + f_K_U,
            q_i=q_i,
            q_j=q_j,
            dq_i=dq_i,
            dq_j=dq_j,
            ddq_i=ddq_i,
            ddq_j=ddq_j,
            joint_index_i=-1,
            joint_index_j=joint_index_j,
            n=n,
            flg_first_index_diff=False,
        ).subs(sub_expr_T, K_T)
        for joint_index_j in range(n)
    ]

    args = [q_i, dq_i, ddq_i, q_j, dq_j, ddq_j, sigmaT_list, lT_list, lU, sT, sU, K_T, K_U]

    return f_K_T, f_K_U, f_K_U_L, f_K_T_L, args, f_K_L_Y, f_K_U_Y, f_K_T_Y


def potential_op(f_K_U, q_i, q_j, joint_index_i, joint_index_j, flg_first_index_diff=True):
    """
    get the potential part
    """
    if flg_first_index_diff:
        return sympy.diff(sympy.diff(f_K_U, q_i[joint_index_i]), q_j[joint_index_j])
    else:
        return sympy.diff(f_K_U, q_j[joint_index_j])


def kinetic_op(f_K_T, q_i, q_j, dq_i, dq_j, ddq_i, ddq_j, joint_index_i, joint_index_j, n, flg_first_index_diff=True):
    """
    get the potential part
    """
    if flg_first_index_diff:
        # ### diff w.r.t. the first index ####
        # derive w.r.t the velocity of the first index
        tmp_i = sympy.diff(f_K_T, dq_i[joint_index_i])
        # derive w.r.t. time, first index
        T_dt_i = sum(
            [
                sympy.diff(tmp_i, q_i[joint_index]) * dq_i[joint_index]
                + sympy.diff(tmp_i, dq_i[joint_index]) * ddq_i[joint_index]
                for joint_index in range(n)
            ]
        )
        # derive w.r.t. pos, first index
        T_dq_i = -sympy.diff(f_K_T, q_i[joint_index_i])
        # update the result
        expr = T_dt_i + T_dq_i
    else:
        expr = f_K_T

    # ### diff w.r.t. the second index ####
    # derive w.r.t the velocity of the second index
    tmp_j_0 = sympy.diff(expr, dq_j[joint_index_j])
    # derive w.r.t. time, second index
    tmp_j_1 = sum(
        [
            sympy.diff(tmp_j_0, q_j[joint_index]) * dq_j[joint_index]
            + sympy.diff(tmp_j_0, dq_j[joint_index]) * ddq_j[joint_index]
            for joint_index in range(n)
        ]
    )
    # derive w.r.t. pos, second index
    tmp_j_2 = -sympy.diff(expr, q_j[joint_index_j])
    # update the result
    expr = tmp_j_1 + tmp_j_2

    return expr


# ---------------------------- FUNCTION GENERATORS ----------------------------------


def gen_num_function_RBF(f_K_U_L, f_K_T_L, f_name, n, optimizations=None, flg_subs=True, f_K_T_Y=None, f_K_U_Y=None):
    """
    Generate numerical expression
    """
    # get the expression
    f_K_L = [f_K_U_L[kernel_block_index] + f_K_T_L[kernel_block_index] for kernel_block_index in range(len(f_K_U_L))]

    # header string
    if flg_subs:
        header_str = "def " + f_name + "(K_T, K_U, X1, X2, pos_indices, vel_indices, acc_indices, lT, lU, sT, sU):\n"
    else:
        header_str = "def " + f_name + "(X1, X2, pos_indices, vel_indices, acc_indices, lT, lU, sT, sU):\n"

    # parameter string
    param_def_str = ""
    for joint_index in range(n):
        param_def_str += "    lT" + str(joint_index + 1) + " = lT[" + str(joint_index) + "]\n"
        param_def_str += "    lT" + str(joint_index + 1 + n) + " = lT[" + str(joint_index + n) + "]\n"
        param_def_str += "    lU" + str(joint_index + 1) + " = lU[" + str(joint_index) + "]\n"
    gen_num_function(f_name, f_K_L, header_str, param_def_str, n, optimizations, f_K_T_Y=f_K_T_Y, f_K_U_Y=f_K_U_Y)


def gen_num_function_RBF_1(f_K_L_L, f_name, n, optimizations=None, flg_subs=True, f_K_L_Y=None):
    """
    Generate numerical expression
    """
    # function definition
    if flg_subs:
        header_str = "def " + f_name + "(K_L, X1, X2, pos_indices, vel_indices, acc_indices, lL, sL):\n"
    else:
        header_str = "def " + f_name + "(X1, X2, pos_indices, vel_indices, acc_indices, lL, sL):\n"

    # parameter string
    param_def_str = ""
    for joint_index in range(n):
        param_def_str += "    lL" + str(joint_index + 1) + " = lL[" + str(joint_index) + "]\n"
        param_def_str += "    lL" + str(joint_index + 1 + n) + " = lL[" + str(joint_index + n) + "]\n"
    gen_num_function(
        f_name, f_K_L_L, header_str, param_def_str, n, optimizations, f_K_L_Y=f_K_L_Y, f_K_U_Y=0, f_K_T_Y=0
    )


def gen_num_function_RBF_1_sc(f_K_L_L, f_name, n, optimizations=None, flg_subs=True, f_K_L_Y=None, path=""):
    """
    Generate numerical expression
    """
    # function definition
    if flg_subs:
        header_str = "def " + f_name + "(K_L, X1, X2, pos_indices, vel_indices, acc_indices, lL, sL):\n"
    else:
        header_str = "def " + f_name + "(X1, X2, pos_indices, vel_indices, acc_indices, lL, sL):\n"

    # parameter string
    param_def_str = ""
    for joint_index in range(n):
        param_def_str += "    lL" + str(joint_index + 1) + " = lL[" + str(joint_index) + "]\n"
        param_def_str += "    lL" + str(joint_index + 1 + n) + " = lL[" + str(joint_index + n) + "]\n"
        param_def_str += "    lL" + str(joint_index + 1 + 2 * n) + " = lL[" + str(joint_index + 2 * n) + "]\n"
    gen_num_function(
        f_name, f_K_L_L, header_str, param_def_str, n, optimizations, f_K_L_Y=f_K_L_Y, f_K_U_Y=0, f_K_T_Y=0, path=path
    )


def gen_num_function_POLY_RBF(
    f_K_U_L, f_K_T_L, f_name, n, optimizations=None, flg_subs=True, f_K_L_Y=None, f_K_T_Y=None, f_K_U_Y=None
):
    """
    Generate numerical expression
    """
    # get the expression
    f_K_L = [f_K_U_L[kernel_block_index] + f_K_T_L[kernel_block_index] for kernel_block_index in range(len(f_K_U_L))]

    # header string
    if flg_subs:
        header_str = (
            "def " + f_name + "(K_T, K_U, X1, X2, pos_indices, vel_indices, acc_indices, lT, lU, sT, sU, sigmaT):\n"
        )
    else:
        header_str = "def " + f_name + "(X1, X2, pos_indices, vel_indices, acc_indices, lT, lU, sT, sU, sigmaT):\n"

    # parameter string
    param_def_str = ""
    for joint_index in range(n):
        param_def_str += "    lT" + str(joint_index + 1) + " = lT[" + str(joint_index) + "]\n"
        param_def_str += "    lU" + str(joint_index + 1) + " = lU[" + str(joint_index) + "]\n"
        param_def_str += "    sigmaT" + str(joint_index + 1) + " = sigmaT[" + str(joint_index) + "]\n"
    # param_def_str += '    sigmaT'+str(n+1)+' = sigmaT['+str(n)+']\n'
    gen_num_function(
        f_name, f_K_L, header_str, param_def_str, n, optimizations, f_K_L_Y=f_K_L_Y, f_K_T_Y=f_K_T_Y, f_K_U_Y=f_K_U_Y
    )


def gen_num_function_POLY_vel_RBF(
    f_K_U_L, f_K_T_L, f_name, n, optimizations=None, flg_subs=True, f_K_L_Y=None, f_K_T_Y=None, f_K_U_Y=None
):
    """
    Generate numerical expression
    """
    # get the expression
    f_K_L = [f_K_U_L[kernel_block_index] + f_K_T_L[kernel_block_index] for kernel_block_index in range(len(f_K_U_L))]

    # header string
    if flg_subs:
        header_str = (
            "def " + f_name + "(K_T, K_U, X1, X2, pos_indices, vel_indices, acc_indices, lT, lU, sT, sU, sigmaT):\n"
        )
    else:
        header_str = "def " + f_name + "(X1, X2, pos_indices, vel_indices, acc_indices, lT, lU, sT, sU, sigmaT):\n"

    # parameter string
    param_def_str = ""
    index_kin_vel = 0
    for joint_index in range(n):
        param_def_str += "    lT" + str(joint_index + 1) + " = lT[" + str(joint_index) + "]\n"
        param_def_str += "    lU" + str(joint_index + 1) + " = lU[" + str(joint_index) + "]\n"
        for joint_index_2 in range(joint_index, n):
            param_def_str += (
                "    sigmaT"
                + str(joint_index + 1)
                + "_"
                + str(joint_index_2 + 1)
                + " = sigmaT["
                + str(index_kin_vel)
                + "]\n"
            )
            index_kin_vel = index_kin_vel + 1
    gen_num_function(
        f_name, f_K_L, header_str, param_def_str, n, optimizations, f_K_L_Y=f_K_L_Y, f_K_T_Y=f_K_T_Y, f_K_U_Y=f_K_U_Y
    )


def gen_num_function_GIP(
    f_K_L_L, f_name, n, robot_structure, optimizations=None, flg_subs=True, f_K_L_Y=None, f_K_T_Y=None, f_K_U_Y=None
):
    """
    Generate numerical expression
    """
    # function definition
    if flg_subs:
        header_str = (
            "def "
            + f_name
            + "(K_L, X1, X2, pos_indices, vel_indices, acc_indices, sigma_kin_vel, sigma_kin_pos_prism, sigma_pot_prism, sigma_kin_pos_rev, sigma_pot_rev):\n"
        )
    else:
        header_str = (
            "def "
            + f_name
            + "(X1, X2, pos_indices, vel_indices, acc_indices, sigma_kin_vel, sigma_kin_pos_prism, sigma_pot_prism, sigma_kin_pos_rev, sigma_pot_rev):\n"
        )

    # parameter string
    param_def_str = ""
    index_prism = 0
    index_rev = 0
    for joint_index in range(n):
        param_def_str += "    sigma_kin_vel" + str(joint_index + 1) + " = sigma_kin_vel[" + str(joint_index) + "]\n"
        if robot_structure[joint_index] == 1:
            param_def_str += (
                "    sigma_kin_pos_prism_j"
                + str(joint_index + 1)
                + "_1 = sigma_kin_pos_prism["
                + str(index_prism)
                + ", 0]\n"
            )
            param_def_str += (
                "    sigma_kin_pos_prism_j"
                + str(joint_index + 1)
                + "_2 = sigma_kin_pos_prism["
                + str(index_prism)
                + ", 1]\n"
            )
            param_def_str += (
                "    sigma_pot_prism_j" + str(joint_index + 1) + "_1 = sigma_pot_prism[" + str(index_prism) + ", 0]\n"
            )
            param_def_str += (
                "    sigma_pot_prism_j" + str(joint_index + 1) + "_2 = sigma_pot_prism[" + str(index_prism) + ", 1]\n"
            )
            index_prism = index_prism + 1
        else:
            param_def_str += (
                "    sigma_kin_pos_rev_j" + str(joint_index + 1) + "_1 = sigma_kin_pos_rev[" + str(index_rev) + ", 0]\n"
            )
            param_def_str += (
                "    sigma_kin_pos_rev_j" + str(joint_index + 1) + "_2 = sigma_kin_pos_rev[" + str(index_rev) + ", 1]\n"
            )
            param_def_str += (
                "    sigma_kin_pos_rev_j" + str(joint_index + 1) + "_3 = sigma_kin_pos_rev[" + str(index_rev) + ", 2]\n"
            )
            param_def_str += (
                "    sigma_pot_rev_j" + str(joint_index + 1) + "_1 = sigma_pot_rev[" + str(index_rev) + ", 0]\n"
            )
            param_def_str += (
                "    sigma_pot_rev_j" + str(joint_index + 1) + "_2 = sigma_pot_rev[" + str(index_rev) + ", 1]\n"
            )
            param_def_str += (
                "    sigma_pot_rev_j" + str(joint_index + 1) + "_3 = sigma_pot_rev[" + str(index_rev) + ", 2]\n"
            )
            index_rev = index_rev + 1
    gen_num_function(
        f_name, f_K_L_L, header_str, param_def_str, n, optimizations, f_K_L_Y=f_K_L_Y, f_K_T_Y=f_K_T_Y, f_K_U_Y=f_K_U_Y
    )


def gen_num_function_GIP_vel(
    f_K_L_L, f_name, n, robot_structure, optimizations=None, flg_subs=True, f_K_L_Y=None, f_K_T_Y=None, f_K_U_Y=None
):
    """
    Generate numerical expression
    """
    # function definition
    if flg_subs:
        header_str = (
            "def "
            + f_name
            + "(K_L, X1, X2, pos_indices, vel_indices, acc_indices, sigma_kin_vel, sigma_kin_pos_prism, sigma_pot_prism, sigma_kin_pos_rev, sigma_pot_rev):\n"
        )
    else:
        header_str = (
            "def "
            + f_name
            + "(X1, X2, pos_indices, vel_indices, acc_indices, sigma_kin_vel, sigma_kin_pos_prism, sigma_pot_prism, sigma_kin_pos_rev, sigma_pot_rev):\n"
        )

    # parameter string
    param_def_str = ""
    index_prism = 0
    index_rev = 0
    index_kin_vel = 0
    for joint_index in range(n):
        for joint_index_2 in range(joint_index, n):
            param_def_str += (
                "    sigma_kin_vel_"
                + str(joint_index + 1)
                + "_"
                + str(joint_index_2 + 1)
                + " = sigma_kin_vel["
                + str(index_kin_vel)
                + "]\n"
            )
            index_kin_vel = index_kin_vel + 1
        if robot_structure[joint_index] == 1:
            param_def_str += (
                "    sigma_kin_pos_prism_j"
                + str(joint_index + 1)
                + "_1 = sigma_kin_pos_prism["
                + str(index_prism)
                + ", 0]\n"
            )
            param_def_str += (
                "    sigma_kin_pos_prism_j"
                + str(joint_index + 1)
                + "_2 = sigma_kin_pos_prism["
                + str(index_prism)
                + ", 1]\n"
            )
            param_def_str += (
                "    sigma_pot_prism_j" + str(joint_index + 1) + "_1 = sigma_pot_prism[" + str(index_prism) + ", 0]\n"
            )
            param_def_str += (
                "    sigma_pot_prism_j" + str(joint_index + 1) + "_2 = sigma_pot_prism[" + str(index_prism) + ", 1]\n"
            )
            index_prism = index_prism + 1
        else:
            param_def_str += (
                "    sigma_kin_pos_rev_j" + str(joint_index + 1) + "_1 = sigma_kin_pos_rev[" + str(index_rev) + ", 0]\n"
            )
            param_def_str += (
                "    sigma_kin_pos_rev_j" + str(joint_index + 1) + "_2 = sigma_kin_pos_rev[" + str(index_rev) + ", 1]\n"
            )
            param_def_str += (
                "    sigma_kin_pos_rev_j" + str(joint_index + 1) + "_3 = sigma_kin_pos_rev[" + str(index_rev) + ", 2]\n"
            )
            param_def_str += (
                "    sigma_pot_rev_j" + str(joint_index + 1) + "_1 = sigma_pot_rev[" + str(index_rev) + ", 0]\n"
            )
            param_def_str += (
                "    sigma_pot_rev_j" + str(joint_index + 1) + "_2 = sigma_pot_rev[" + str(index_rev) + ", 1]\n"
            )
            param_def_str += (
                "    sigma_pot_rev_j" + str(joint_index + 1) + "_3 = sigma_pot_rev[" + str(index_rev) + ", 2]\n"
            )
            index_rev = index_rev + 1
    gen_num_function(
        f_name, f_K_L_L, header_str, param_def_str, n, optimizations, f_K_L_Y=f_K_L_Y, f_K_T_Y=f_K_T_Y, f_K_U_Y=f_K_U_Y
    )


def gen_num_function_GIP_sum(
    f_K_L_L,
    f_name,
    n,
    robot_structure,
    optimizations=None,
    flg_subs=True,
    f_K_L_Y=None,
    f_K_T_Y=None,
    f_K_U_Y=None,
    path="",
):
    """
    Generate numerical expression
    """
    # function definition
    if flg_subs:
        header_str = (
            "def "
            + f_name
            + "(K_L, X1, X2, pos_indices, vel_indices, acc_indices, sigma_kin_vel_list, sigma_kin_pos_list, sigma_pot_list):\n"
        )
    else:
        header_str = (
            "def "
            + f_name
            + "(X1, X2, pos_indices, vel_indices, acc_indices, sigma_kin_vel_list, sigma_kin_pos_list, sigma_pot_list):\n"
        )

    # parameter string
    param_def_str = ""
    pos_index = 0
    for joint_index_1 in range(n):
        for joint_index_2 in range(joint_index_1 + 1):
            param_def_str += (
                "    sigma_kin_v_"
                + str(joint_index_1 + 1)
                + "_"
                + str(joint_index_2 + 1)
                + " = sigma_kin_vel_list["
                + str(joint_index_1)
                + "]["
                + str(joint_index_2)
                + "]\n"
            )
            if robot_structure[joint_index_2] == 1:
                param_def_str += (
                    "    sigma_kin_p_"
                    + str(joint_index_1 + 1)
                    + "_"
                    + str(joint_index_2 + 1)
                    + "_l = sigma_kin_pos_list["
                    + str(pos_index)
                    + "][0]\n"
                )
                param_def_str += (
                    "    sigma_kin_p_"
                    + str(joint_index_1 + 1)
                    + "_"
                    + str(joint_index_2 + 1)
                    + "_off = sigma_kin_pos_list["
                    + str(pos_index)
                    + "][1]\n"
                )
            else:
                param_def_str += (
                    "    sigma_kin_p_"
                    + str(joint_index_1 + 1)
                    + "_"
                    + str(joint_index_2 + 1)
                    + "_s = sigma_kin_pos_list["
                    + str(pos_index)
                    + "][0]\n"
                )
                param_def_str += (
                    "    sigma_kin_p_"
                    + str(joint_index_1 + 1)
                    + "_"
                    + str(joint_index_2 + 1)
                    + "_c = sigma_kin_pos_list["
                    + str(pos_index)
                    + "][1]\n"
                )
                param_def_str += (
                    "    sigma_kin_p_"
                    + str(joint_index_1 + 1)
                    + "_"
                    + str(joint_index_2 + 1)
                    + "_off = sigma_kin_pos_list["
                    + str(pos_index)
                    + "][2]\n"
                )
            pos_index += 1
        if robot_structure[joint_index_1] == 1:
            param_def_str += (
                "    sigma_pot_" + str(joint_index_1 + 1) + "_l = sigma_pot_list[" + str(joint_index_1) + "][0]\n"
            )
            param_def_str += (
                "    sigma_pot_" + str(joint_index_1 + 1) + "_off = sigma_pot_list[" + str(joint_index_1) + "][1]\n"
            )
        else:
            param_def_str += (
                "    sigma_pot_" + str(joint_index_1 + 1) + "_s = sigma_pot_list[" + str(joint_index_1) + "][0]\n"
            )
            param_def_str += (
                "    sigma_pot_" + str(joint_index_1 + 1) + "_c = sigma_pot_list[" + str(joint_index_1) + "][1]\n"
            )
            param_def_str += (
                "    sigma_pot_" + str(joint_index_1 + 1) + "_off = sigma_pot_list[" + str(joint_index_1) + "][2]\n"
            )
    gen_num_function(
        f_name,
        f_K_L_L,
        header_str,
        param_def_str,
        n,
        optimizations,
        f_K_L_Y=f_K_L_Y,
        f_K_T_Y=f_K_T_Y,
        f_K_U_Y=f_K_U_Y,
        path=path,
    )


def gen_num_function_RBF_M(
    f_K_U_L, f_K_T_L, f_name, n, optimizations=None, flg_subs=True, f_K_L_Y=None, f_K_T_Y=None, f_K_U_Y=None
):
    """
    Generate numerical expression
    """
    # get the expression
    f_K_L = [f_K_U_L[kernel_block_index] + f_K_T_L[kernel_block_index] for kernel_block_index in range(len(f_K_U_L))]

    # header string
    if flg_subs:
        header_str = (
            "def " + f_name + "(K_T, K_U, X1, X2, pos_indices, vel_indices, acc_indices, lT_list, lU, sT_list, sU):\n"
        )
    else:
        header_str = "def " + f_name + "(X1, X2, pos_indices, vel_indices, acc_indices, lT_list, lU, sT_list, sU):\n"

    # parameter string
    param_def_str = ""
    current_index = 0
    # set potential kernel parameters
    for joint_index in range(n):
        param_def_str += "    lU" + str(joint_index + 1) + " = lU[" + str(joint_index) + "]\n"
    # Set kinetic kernel parameters
    for joint_index_1 in range(n):
        for joint_index_2 in range(joint_index_1 + 1):
            for l_index in range(n):
                param_def_str += (
                    "    lT"
                    + str(joint_index_1 + 1)
                    + str(joint_index_2 + 1)
                    + "_"
                    + str(l_index + 1)
                    + " = lT_list["
                    + str(current_index)
                    + "]["
                    + str(l_index)
                    + "]\n"
                )
            param_def_str += (
                "    sT" + str(joint_index_1 + 1) + str(joint_index_2 + 1) + " = sT_list[" + str(current_index) + "]\n"
            )
            current_index = current_index + 1
    gen_num_function(
        f_name, f_K_L, header_str, param_def_str, n, optimizations, f_K_L_Y=f_K_L_Y, f_K_T_Y=f_K_T_Y, f_K_U_Y=f_K_U_Y
    )


def gen_num_function_POLY_vel_RBF_sum(
    f_K_U_L, f_K_T_L, f_name, n, optimizations=None, flg_subs=True, f_K_L_Y=None, f_K_T_Y=None, f_K_U_Y=None
):
    """
    Generate numerical expression
    """
    # get the expression
    f_K_L = [f_K_U_L[kernel_block_index] + f_K_T_L[kernel_block_index] for kernel_block_index in range(len(f_K_U_L))]

    # header string
    if flg_subs:
        header_str = (
            "def "
            + f_name
            + "(K_T, K_U, X1, X2, pos_indices, vel_indices, acc_indices, lT_list, lU, sT, sU, sigmaT_list):\n"
        )
    else:
        header_str = (
            "def " + f_name + "(X1, X2, pos_indices, vel_indices, acc_indices, lT_list, lU, sT, sU, sigmaT_list):\n"
        )

    # parameter string
    param_def_str = ""
    # potential param
    for joint_index in range(n):
        param_def_str += "    lU" + str(joint_index + 1) + " = lU[" + str(joint_index) + "]\n"
    # kin param
    for joint_index in range(n):
        param_def_str += "    sT" + str(joint_index + 1) + " = sT[" + str(joint_index) + "]\n"
        # index_kin_vel = 0
        for joint_index_1 in range(joint_index + 1):
            param_def_str += (
                "    lT"
                + str(joint_index + 1)
                + "_"
                + str(joint_index_1 + 1)
                + " = lT_list["
                + str(joint_index)
                + "]["
                + str(joint_index_1)
                + "]\n"
            )
            param_def_str += (
                "    sigmaT"
                + str(joint_index + 1)
                + "_"
                + str(joint_index_1 + 1)
                + " = sigmaT_list["
                + str(joint_index)
                + "]["
                + str(joint_index_1)
                + "]\n"
            )
    gen_num_function(
        f_name, f_K_L, header_str, param_def_str, n, optimizations, f_K_L_Y=f_K_L_Y, f_K_T_Y=f_K_T_Y, f_K_U_Y=f_K_U_Y
    )


def gen_num_function(
    f_name, f_K_L, header_str, param_def_str, n, optimizations, f_K_T_Y=None, f_K_U_Y=None, f_K_L_Y=None, path=""
):
    """
    Generate numpy/torch code from cse
    """

    # generate input assignment
    input_def_full_str, input_def_diag_str = gen_input_assignment(n)

    print("\nGenerate full kernel function...")
    gen_code(
        expressions=f_K_L,
        f_name=f_name,
        header_str=header_str,
        input_def_str=input_def_full_str,
        param_def_str=param_def_str,
        optimizations=optimizations,
        path=path,
    )

    print("\nGenerate lower-triangular kernel function...")
    # diag kernel matrix generation
    ltr_indices = []
    current_index = 0
    for joint_index_1 in range(n):
        for joint_index_2 in range(n):
            if joint_index_2 <= joint_index_1:
                ltr_indices.append(current_index)
            current_index += 1
    print(ltr_indices)
    gen_code(
        expressions=[f_K_L[i] for i in ltr_indices],
        f_name=f_name + "_ltr",
        header_str=header_str,
        input_def_str=input_def_full_str,
        param_def_str=param_def_str,
        optimizations=optimizations,
        path=path,
    )

    print("\nGenerate diagonal kernel function...")
    # diag kernel matrix generation
    diag_indices = []
    for joint_index in range(0, n):
        diag_indices.append(n * joint_index + joint_index)
    gen_code(
        expressions=[f_K_L[i] for i in diag_indices],
        f_name=f_name + "_diag",
        header_str=header_str,
        input_def_str=input_def_diag_str,
        param_def_str=param_def_str,
        optimizations=optimizations,
        path=path,
    )

    if f_K_T_Y is not None:
        print("\nGenerate covariance matrix between kinetic energy and measures...")
        gen_code(
            expressions=f_K_T_Y,
            f_name=f_name + "_T_Y_cov",
            header_str=header_str,
            input_def_str=input_def_full_str,
            param_def_str=param_def_str,
            optimizations=optimizations,
            path=path,
        )

    if f_K_U_Y is not None:
        print("\nGenerate covariance matrix between potential energy and measures...")
        gen_code(
            expressions=f_K_U_Y,
            f_name=f_name + "_U_Y_cov",
            header_str=header_str,
            input_def_str=input_def_full_str,
            param_def_str=param_def_str,
            optimizations=optimizations,
            path=path,
        )

    if f_K_L_Y is not None:
        print("\nGenerate covariance matrix between lagrangian energy and measures...")
        gen_code(
            expressions=f_K_L_Y,
            f_name=f_name + "_L_Y_cov",
            header_str=header_str,
            input_def_str=input_def_full_str,
            param_def_str=param_def_str,
            optimizations=optimizations,
            path=path,
        )


def gen_code(expressions, f_name, header_str, input_def_str, param_def_str, optimizations, path=""):
    f_str = "import numpy\n\n"
    f_str += header_str + "\n"
    f_str += input_def_str + "\n"
    f_str += param_def_str + "\n"
    f_str += gen_main_str(*sympy.cse(expressions, optimizations=optimizations)) + "\n"
    # write the function
    # file_f = open(f_name + ".py", "w")
    # file_f.write(f_str)
    # file_f.close()
    # convert the numpy code in torch
    f_str_torch = np2torch(f_str)
    # write the function
    file_f = open(path + f_name + "_torch.py", "w")
    file_f.write(f_str_torch)
    file_f.close()


def gen_input_assignment(n):
    # generate the string for input assignment
    input_def_full_str = ""  # full kernel matrix computation
    input_def_diag_str = ""  # diagonal kernel matrix computation
    for joint_index in range(n):
        # full
        input_def_full_str += (
            "    q_i"
            + str(joint_index + 1)
            + " = X1[:,pos_indices["
            + str(joint_index)
            + "]:pos_indices["
            + str(joint_index)
            + "]+1]\n"
        )
        input_def_full_str += (
            "    dq_i"
            + str(joint_index + 1)
            + " = X1[:,vel_indices["
            + str(joint_index)
            + "]:vel_indices["
            + str(joint_index)
            + "]+1]\n"
        )
        input_def_full_str += (
            "    ddq_i"
            + str(joint_index + 1)
            + " = X1[:,acc_indices["
            + str(joint_index)
            + "]:acc_indices["
            + str(joint_index)
            + "]+1]\n"
        )
        input_def_full_str += (
            "    q_j"
            + str(joint_index + 1)
            + " = X2[:,pos_indices["
            + str(joint_index)
            + "]:pos_indices["
            + str(joint_index)
            + "]+1].transpose(1,0)\n"
        )
        input_def_full_str += (
            "    dq_j"
            + str(joint_index + 1)
            + " = X2[:,vel_indices["
            + str(joint_index)
            + "]:vel_indices["
            + str(joint_index)
            + "]+1].transpose(1,0)\n"
        )
        input_def_full_str += (
            "    ddq_j"
            + str(joint_index + 1)
            + " = X2[:,acc_indices["
            + str(joint_index)
            + "]:acc_indices["
            + str(joint_index)
            + "]+1].transpose(1,0)\n"
        )
        # diag
        input_def_diag_str += (
            "    q_i"
            + str(joint_index + 1)
            + " = X1[:,pos_indices["
            + str(joint_index)
            + "]:pos_indices["
            + str(joint_index)
            + "]+1]\n"
        )
        input_def_diag_str += (
            "    dq_i"
            + str(joint_index + 1)
            + " = X1[:,vel_indices["
            + str(joint_index)
            + "]:vel_indices["
            + str(joint_index)
            + "]+1]\n"
        )
        input_def_diag_str += (
            "    ddq_i"
            + str(joint_index + 1)
            + " = X1[:,acc_indices["
            + str(joint_index)
            + "]:acc_indices["
            + str(joint_index)
            + "]+1]\n"
        )
        input_def_diag_str += (
            "    q_j"
            + str(joint_index + 1)
            + " = X2[:,pos_indices["
            + str(joint_index)
            + "]:pos_indices["
            + str(joint_index)
            + "]+1]\n"
        )
        input_def_diag_str += (
            "    dq_j"
            + str(joint_index + 1)
            + " = X2[:,vel_indices["
            + str(joint_index)
            + "]:vel_indices["
            + str(joint_index)
            + "]+1]\n"
        )
        input_def_diag_str += (
            "    ddq_j"
            + str(joint_index + 1)
            + " = X2[:,acc_indices["
            + str(joint_index)
            + "]:acc_indices["
            + str(joint_index)
            + "]+1]\n"
        )
    return input_def_full_str, input_def_diag_str


def gen_main_str(sub_expr, expr):
    # sub expressions computation str
    str_main = "    \n"
    for s_expr in sub_expr:
        str_main += "    " + str(s_expr[0]) + " = " + NumPyPrinter().doprint(s_expr[1]) + "\n"
    # kernel block computation
    str_main += "    \n"
    str_main += "    K_block_list = []\n"
    for K_block_expr in expr:
        str_main += "    K_block_list.append(" + NumPyPrinter().doprint(K_block_expr) + ")\n"
    # return kernel
    str_main += "    \n"
    str_main += "    return K_block_list\n"
    return str_main


def np2torch(str):
    """
    Convert numpy file string in torch string
    """
    p_np = re.compile("numpy")
    str_torch = p_np.sub("torch", str)
    return str_torch
