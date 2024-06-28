# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Panda data collection (set the reference trajectory and compute torques with inverse dynamics)
Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
        Giulio Giacomuzzo (giulio.giacomuzzo@gmail.com)
for i in {0..40};do python Get_PANDA_data_inv_dyn.py -seed $i -saving_path 'data/data_no_noise/'; done;
"""

import argparse
import pickle as pkl

import H
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from c import c as f_c
from g import g as f_g
from g_U import g_U as f_g_U

# import PANDA equations
from inv_dyn import inv_dyn
from M import M as f_M

# import time
from U import U as f_U

p = argparse.ArgumentParser("Panda data inverse dynamics")
p.add_argument("-saving_path", type=str, default="./data/data_no_noise/", help="saving folder")
p.add_argument("-seed", type=int, default=0, help="seed")
p.add_argument("-sim_interval", type=int, default=50, help="sim_interval")
p.add_argument("-pos_range", type=float, default=1.0, help="pos_range percentage")
p.add_argument("-w_f", type=float, default=0.02, help="Fundamental pulsation")
p.add_argument("-num_sin", type=int, default=5, help="number of sinusoids")
p.add_argument("-c_init", type=float, default=1.0, help="initial range of the sinusoids coefficient distribution")
p.add_argument("-flg_frict", type=bool, default=False, help="if true use friction")
p.add_argument("-p_dyn_perc_error", type=float, default=0.0, help="percentage multiplicative error")
p.add_argument("-p_dyn_add_std", type=float, default=0.0, help="std additive error")
p.add_argument(
    "-sampling_time",
    type=float,
    default=0.001,
    help="number of non fixed joints -- to simulate lower dof configurations ",
)
p.add_argument(
    "-num_active_joints",
    type=int,
    default=7,
    help="number of non fixed joints -- to simulate lower dof configurations ",
)
p.add_argument("-std_noise", type=float, default=0.0, help="torque noise std")
p.add_argument("-plot", type=int, default=1, help="if 1 plot results")

locals().update(vars(p.parse_known_args()[0]))
np.random.seed(seed)

flg_plot = plot == 1
# p_dyn = pkl.load(open('friction_PANDA_dyn_par.pkl', 'rb'))

# generate parameters from nominal
p_dyn_nominal = pkl.load(open("PANDA_dyn_par.pkl", "rb"))
num_p_dyn = p_dyn_nominal.size
# multiplicative error (uniform in +-p_dyn_perc_error)
p_dyn = p_dyn_nominal + p_dyn_perc_error * (2 * np.random.rand(num_p_dyn) - 1) * p_dyn_nominal
# additive gaussian noise (with standard deviation p_dyn_add_std) not on inertia tensors mass an centers of mass
# p_dyn = p_dyn + p_dyn_add_std*np.random.randn(num_p_dyn)*np.array([0,0,0,0,0,0,1,1,1,1,1]*6)
p_dyn = p_dyn + p_dyn_add_std * np.random.randn(num_p_dyn) * np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1] * 7)

# # pkl.dump(p_dyn, open(saving_path+'PANDA_dyn_par_'+str(seed)+'.pkl', 'wb'))


def get_sum_of_sin_trj(
    num_dof, q_ranges, dq_ranges, ddq_ranges, w_f, num_sin, c_init, T_sampling, sim_interval, num_active_joints=-1
):
    """
    Return q q_dot q_ddot
    """
    if num_active_joints == -1:
        num_active_joints = num_dof

    q_list = []
    dq_list = []
    ddq_list = []
    t = np.arange(0, sim_interval + T_sampling, T_sampling)
    num_samples = t.size
    for joint_index in range(0, num_dof):
        # print('\nJoint index '+str(joint_index+1)+':')
        # Generate q (add samples discarded with numerical diff)
        if joint_index < num_active_joints:
            flg_feasible = False
            num_attempt = 0
            current_c = c_init * q_ranges[joint_index]
            while not (flg_feasible):
                # print('Joint index '+str(joint_index+1)+' attempt',num_attempt)
                # sample trj coefficients
                a_k = current_c * (2 * np.random.rand(num_sin) - 1)
                b_k = current_c * (2 * np.random.rand(num_sin) - 1)
                # get q dq ddq
                pos = 0
                vel = 0
                acc = 0
                for sin_index in range(num_sin):
                    l = sin_index + 1
                    pos = (
                        pos
                        + a_k[sin_index] / (w_f * l) * np.sin(w_f * l * t)
                        - b_k[sin_index] / (w_f * l) * np.cos(w_f * l * t)
                    )
                    vel = vel + a_k[sin_index] * np.cos(w_f * l * t) + b_k[sin_index] * np.sin(w_f * l * t)
                    acc = (
                        acc
                        - a_k[sin_index] * (w_f * l) * np.sin(w_f * l * t)
                        + b_k[sin_index] * (w_f * l) * np.cos(w_f * l * t)
                    )
                # pos = np.sum(np.concatenate([a_k[sin_index]/(w_f*(sin_index+1))*np.sin(w_f*(sin_index+1)*t).reshape([1,-1])\
                #                             - b_k[sin_index]/(w_f*(sin_index+1))*np.cos(w_f*(sin_index+1)*t).reshape([1,-1])
                #                       for sin_index in range(num_sin)], 0), 0)
                # vel = np.sum(np.concatenate([a_k[sin_index]*np.cos(w_f*(sin_index+1)*t).reshape([1,-1]) \
                #                              + b_k[sin_index]*np.sin(w_f*(sin_index+1)*t).reshape([1,-1])
                #                       for sin_index in range(num_sin)], 0), 0)
                # acc = np.sum(np.concatenate([- a_k[sin_index]*(w_f*(sin_index+1))*np.sin(w_f*(sin_index+1)*t).reshape([1,-1])\
                #                              + b_k[sin_index]*(w_f*(sin_index+1))*np.cos(w_f*(sin_index+1)*t).reshape([1,-1])
                #                       for sin_index in range(num_sin)], 0), 0)
                # check feasibility
                if np.max(np.abs(pos)) < q_ranges[joint_index]:
                    if np.max(np.abs(vel)) < dq_ranges[joint_index]:
                        if np.max(np.abs(acc)) < ddq_ranges[joint_index]:
                            flg_feasible = True
                if flg_feasible:
                    break
                if num_attempt == 10:
                    # change sampling distribution
                    current_c = current_c * 0.8
                    # print('current_c', current_c)
                    num_attempt = 0
                else:
                    num_attempt += 1
        else:
            pos = vel = acc = np.zeros((num_samples, 1))
        q_list.append(pos.reshape([-1, 1]))
        dq_list.append(vel.reshape([-1, 1]))
        ddq_list.append(acc.reshape([-1, 1]))
    q = np.concatenate(q_list, 1)
    dq = np.concatenate(dq_list, 1)
    ddq = np.concatenate(ddq_list, 1)
    return q, dq, ddq


def get_tau(q, q_dot, q_ddot, damping_coef, friction_coef, offset_coef):
    """
    Computes the PANDA inverse dynamics
    """
    num_samples, num_dof = q.shape
    tau = np.zeros([num_samples, num_dof])
    for t in range(0, num_samples):
        # print(inv_dyn(p_dyn, q[t,:], q_dot[t,:], q_ddot[t,:]))
        tau[t, :] = (
            inv_dyn(p_dyn, q[t, :], q_dot[t, :], q_ddot[t, :])
            + damping_coef * q_dot[t, :]
            + friction_coef * np.sign(q_dot[t, :])
            + offset_coef
        )
    return tau


def get_M(q):
    """
    Computes the PANDA inertia matrix
    """
    num_samples, num_dof = q.shape
    M = np.zeros([num_samples, num_dof, num_dof])
    for t in range(0, num_samples):
        M[t] = np.array(f_M(p_dyn, q[t, :])).reshape([num_dof, num_dof])
    return M


def get_U(q):
    num_samples = q.shape[0]
    U = np.zeros([num_samples, 1])
    for t in range(num_samples):
        U[t] = f_U(q[t, :])
    return U


def get_g(parms, q):
    num_samples, num_dof = q.shape
    g = np.zeros([num_samples, num_dof])
    for t in range(num_samples):
        g[t, :] = np.array(f_g(parms, q[t, :])).reshape(1, -1)
    return g


def get_c(parms, q, dq):
    num_samples, num_dof = q.shape
    c = np.zeros([num_samples, num_dof])
    for t in range(num_samples):
        c[t, :] = np.array(f_c(parms, q[t, :], dq[t, :])).reshape(1, -1)
    return c


def get_m(parms, q, ddq):
    num_samples, num_dof = q.shape
    m = np.zeros([num_samples, num_dof])
    for t in range(num_samples):
        m[t, :] = M[t, :, :] @ ddq[t, :]
    return m


def get_g_U(q):
    num_samples, num_dof = q.shape
    g = np.zeros([num_samples, num_dof])
    for t in range(num_samples):
        g[t, :] = np.array(f_g_U(q[t, :])).reshape(1, -1)
    return g


def get_phi(q, q_dot, q_ddot):
    """
    Computes the PANDA phi basis
    """
    num_samples, num_dof = q.shape
    phi = np.zeros([num_samples, num_dof * 60])
    for t in range(0, num_samples):
        phi[t, :] = np.array(H.H(q[t, :], q_dot[t, :], q_ddot[t, :])).reshape([1, -1])
    return phi


# saving names
saving_path += "panda_" + str(num_active_joints) + "dof_"
if p_dyn_perc_error == 0.0 and p_dyn_add_std == 0.0:
    if flg_frict:
        saving_name_data = (
            saving_path
            + "data_inv_dyn_sum_of_sin__noise_std"
            + str(std_noise)
            + "_friction__w_f"
            + str(w_f)
            + "__num_sin"
            + str(num_sin)
            + "__pos_range_"
            + str(pos_range)
            + "__"
            + str(sim_interval)
            + "seconds"
            + "__Ts_"
            + str(sampling_time)
            + "__seed"
            + str(seed)
            + ".pkl"
        )
        # saving_name_data_test = saving_path+'data_inv_dyn_sum_of_sin_friction__w_f'+str(w_f)+'__num_sin'+str(num_sin)+'__pos_range_'+str(pos_range)+'__'+str(sim_interval)+'seconds__seed'+str(seed)+'_test.pkl'
        saving_name_data_M = (
            saving_path
            + "data_M_inv_dyn_sum_of_sin__noise_std"
            + str(std_noise)
            + "_friction__w_f"
            + str(w_f)
            + "__num_sin"
            + str(num_sin)
            + "__pos_range_"
            + str(pos_range)
            + "__"
            + str(sim_interval)
            + "seconds"
            + "__Ts_"
            + str(sampling_time)
            + "__seed"
            + str(seed)
            + ".pkl"
        )
        # saving_name_data_M_test = saving_path+'data_M_inv_dyn_sum_of_sin_friction__w_f'+str(w_f)+'__num_sin'+str(num_sin)+'__pos_range_'+str(pos_range)+'__'+str(sim_interval)+'seconds__seed'+str(seed)+'_test.pkl'
    else:
        saving_name_data = (
            saving_path
            + "data_inv_dyn_sum_of_sin__noise_std"
            + str(std_noise)
            + "__w_f"
            + str(w_f)
            + "__num_sin"
            + str(num_sin)
            + "__pos_range_"
            + str(pos_range)
            + "__"
            + str(sim_interval)
            + "seconds"
            + "__Ts_"
            + str(sampling_time)
            + "__seed"
            + str(seed)
            + ".pkl"
        )
        # saving_name_data_test = saving_path+'data_inv_dyn_sum_of_sin__w_f'+str(w_f)+'__num_sin'+str(num_sin)+'__pos_range_'+str(pos_range)+'__'+str(sim_interval)+'seconds__seed'+str(seed)+'_test.pkl'
        saving_name_data_M = (
            saving_path
            + "data_M_inv_dyn_sum_of_sin__noise_std"
            + str(std_noise)
            + "__w_f"
            + str(w_f)
            + "__num_sin"
            + str(num_sin)
            + "__pos_range_"
            + str(pos_range)
            + "__"
            + str(sim_interval)
            + "seconds"
            + "__Ts_"
            + str(sampling_time)
            + "__seed"
            + str(seed)
            + ".pkl"
        )
        # saving_name_data_M_test = saving_path+'data_M_inv_dyn_sum_of_sin__w_f'+str(w_f)+'__num_sin'+str(num_sin)+'__pos_range_'+str(pos_range)+'__'+str(sim_interval)+'seconds__seed'+str(seed)+'_test.pkl'
else:
    if flg_frict:
        saving_name_data = (
            saving_path
            + "data_inv_dyn_sum_of_sin__noise_std"
            + str(std_noise)
            + "_friction__p_dyn_add_std_"
            + str(p_dyn_add_std)
            + "__p_dyn_add_std_"
            + str(p_dyn_add_std)
            + "__w_f"
            + str(w_f)
            + "__num_sin"
            + str(num_sin)
            + "__pos_range_"
            + str(pos_range)
            + "__"
            + str(sim_interval)
            + "seconds"
            + "__Ts_"
            + str(sampling_time)
            + "__seed"
            + str(seed)
            + ".pkl"
        )
        # saving_name_data_test = saving_path+'data_inv_dyn_sum_of_sin_friction__p_dyn_add_std_'+str(p_dyn_add_std)+'__p_dyn_add_std_'+str(p_dyn_add_std)+'__w_f'+str(w_f)+'__num_sin'+str(num_sin)+'__pos_range_'+str(pos_range)+'__'+str(sim_interval)+'seconds__seed'+str(seed)+'_test.pkl'
        saving_name_data_M = (
            saving_path
            + "data_M_inv_dyn_sum_of_sin__noise_std"
            + str(std_noise)
            + "_friction__p_dyn_add_std_"
            + str(p_dyn_add_std)
            + "__p_dyn_add_std_"
            + str(p_dyn_add_std)
            + "__w_f"
            + str(w_f)
            + "__num_sin"
            + str(num_sin)
            + "__pos_range_"
            + str(pos_range)
            + "__"
            + str(sim_interval)
            + "seconds"
            + "__Ts_"
            + str(sampling_time)
            + "__seed"
            + str(seed)
            + ".pkl"
        )
        # saving_name_data_M_test = saving_path+'data_M_inv_dyn_sum_of_sin_friction__p_dyn_add_std_'+str(p_dyn_add_std)+'__p_dyn_add_std_'+str(p_dyn_add_std)+'__w_f'+str(w_f)+'__num_sin'+str(num_sin)+'__pos_range_'+str(pos_range)+'__'+str(sim_interval)+'seconds__seed'+str(seed)+'_test.pkl'
    else:
        saving_name_data = (
            saving_path
            + "data_inv_dyn_sum_of_sin__noise_std"
            + str(std_noise)
            + "__p_dyn_add_std_"
            + str(p_dyn_add_std)
            + "__p_dyn_add_std_"
            + str(p_dyn_add_std)
            + "__w_f"
            + str(w_f)
            + "__num_sin"
            + str(num_sin)
            + "__pos_range_"
            + str(pos_range)
            + "__"
            + str(sim_interval)
            + "seconds"
            + "__Ts_"
            + str(sampling_time)
            + "__seed"
            + str(seed)
            + ".pkl"
        )
        # saving_name_data_test = saving_path+'data_inv_dyn_sum_of_sin__p_dyn_add_std_'+str(p_dyn_add_std)+'__p_dyn_add_std_'+str(p_dyn_add_std)+'__w_f'+str(w_f)+'__num_sin'+str(num_sin)+'__pos_range_'+str(pos_range)+'__'+str(sim_interval)+'seconds__seed'+str(seed)+'_test.pkl'
        saving_name_data_M = (
            saving_path
            + "data_M_inv_dyn_sum_of_sin__noise_std"
            + str(std_noise)
            + "__p_dyn_add_std_"
            + str(p_dyn_add_std)
            + "__p_dyn_add_std_"
            + str(p_dyn_add_std)
            + "__w_f"
            + str(w_f)
            + "__num_sin"
            + str(num_sin)
            + "__pos_range_"
            + str(pos_range)
            + "__"
            + str(sim_interval)
            + "seconds"
            + "__Ts_"
            + str(sampling_time)
            + "__seed"
            + str(seed)
            + ".pkl"
        )
        # saving_name_data_M_test = saving_path+'data_M_inv_dyn_sum_of_sin__p_dyn_add_std_'+str(p_dyn_add_std)+'__p_dyn_add_std_'+str(p_dyn_add_std)+'__w_f'+str(w_f)+'__num_sin'+str(num_sin)+'__pos_range_'+str(pos_range)+'__'+str(sim_interval)+'seconds__seed'+str(seed)+'_test.pkl'

# ### Generate joint trajectories
# print('\nGENERATE TRAJECTORIES')
# trj generation parameters
T_sampling = sampling_time  # 0.1 #0.001
num_dof = 7
# Define joint operational ranges
ll = [-2.9, -1.8, -2.9, -3.0, -2.9, 0.0, -2.9]  # [rad]
ul = [2.9, 1.8, 2.9, 0.0, 2.9, 3.8, 2.9]  # [rad]
q_ranges = np.array([pos_range * (ul[i] - ll[i]) / 2 for i in range(num_dof)])
mean_ranges = np.array([(ul[i] + ll[i]) / 2 for i in range(num_dof)])
dq_ranges = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
ddq_ranges = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
omega_f = 2 * np.pi * w_f

# gen trj
q, q_dot, q_ddot = get_sum_of_sin_trj(
    num_dof,
    q_ranges,
    dq_ranges,
    ddq_ranges,
    omega_f,
    num_sin,
    c_init,
    T_sampling,
    sim_interval,
    num_active_joints=num_active_joints,
)

# add position offset
q = q + mean_ranges

# ### Compute torques
# print('\nCOMPUTE TORQUES')
if flg_frict:
    damping_coef = [0.0628, 0.2088, 0.0361, 0.2174, 0.1021, 0.0001, 0.0632]
    friction_coef = [0.2549, 0.1413, 0.1879, 0.3625, 0.2728, 0.1529, 0.2097]
    offset_coef = [-0.1069, -0.1601, -0.0718, -0.2562, 0.0079, 0.0935, -0.0070]
else:
    damping_coef = 0.0
    friction_coef = 0.0
    offset_coef = 0.0
tau_noiseless = get_tau(q, q_dot, q_ddot, damping_coef, friction_coef, offset_coef)
tau = tau_noiseless + std_noise * np.random.randn(*tau_noiseless.shape)

# ### Compute M
# print('\nCOMPUTE INERTIA MATRICES')
M = get_M(q)

# ### Compute U
# print('\nCOMPUTE POTENTIAL ENERGY')
U = get_U(q)

# ### Compute g
# print('\nCOMPUTE GRAVITY TORQUE')
g = get_g(p_dyn, q)
gU = get_g_U(q)

# ### Compute c
# print('\nCOMPUTE CORIOLIS TORQUE')
c = get_c(p_dyn, q, q_dot)

# ### Compute m
m = get_m(p_dyn, q, q_ddot)

# #### PRINT DATA
if flg_plot:
    # print('\nPRINT DATA')

    plt.figure()
    for joint_index in range(0, num_dof):
        plt.subplot(3, 3, joint_index + 1)
        plt.grid()
        plt.plot(q[:, joint_index], label="q_" + str(joint_index + 1))
        plt.plot(q_dot[:, joint_index], label="dq_" + str(joint_index + 1))
        plt.plot(q_ddot[:, joint_index], label="ddq_" + str(joint_index + 1))
        #    plt.plot(tau[:,joint_index], label='tau_'+str(joint_index+1))
        plt.legend()
    plt.suptitle("TRAJECTORIES")

    plt.figure()
    for joint_index in range(0, num_dof):
        plt.subplot(3, 3, joint_index + 1)
        plt.grid()
        plt.plot(q[:, joint_index], label="q_" + str(joint_index + 1))
        plt.plot(np.ones_like(q[:, joint_index]) * ul[joint_index], "--r")
        plt.plot(np.ones_like(q[:, joint_index]) * ll[joint_index], "--r")
        plt.legend()
    plt.suptitle("POSITIONS")

    plt.figure()
    for joint_index in range(0, num_dof):
        plt.subplot(3, 3, joint_index + 1)
        plt.grid()
        plt.plot(q_dot[:, joint_index], label="q_" + str(joint_index + 1))
        plt.plot(np.ones_like(q_dot[:, joint_index]) * dq_ranges[joint_index], "--r")
        plt.plot(-np.ones_like(q_dot[:, joint_index]) * dq_ranges[joint_index], "--r")
        plt.legend()
    plt.suptitle("VELOCITIES")

    plt.figure()
    for joint_index in range(0, num_dof):
        plt.subplot(3, 3, joint_index + 1)
        plt.grid()
        plt.plot(q_ddot[:, joint_index], label="q_" + str(joint_index + 1))
        plt.plot(np.ones_like(q_ddot[:, joint_index]) * ddq_ranges[joint_index], "--r")
        plt.plot(-np.ones_like(q_ddot[:, joint_index]) * ddq_ranges[joint_index], "--r")
        plt.legend()
    plt.suptitle("ACCELERATIONS")

    plt.figure()
    for joint_index in range(0, num_dof):
        plt.subplot(3, 3, joint_index + 1)
        plt.grid()
        #    plt.plot(q[:,joint_index], label='q_'+str(joint_index+1))
        #    plt.plot(q_dot[:,joint_index], label='dq_'+str(joint_index+1))
        #    plt.plot(q_ddot[:,joint_index], label='ddq_'+str(joint_index+1))
        plt.plot(tau[:, joint_index], label="tau_" + str(joint_index + 1))
        plt.plot(tau_noiseless[:, joint_index], label="tau_noiseless" + str(joint_index + 1))
        plt.legend()
    plt.suptitle("TAU")

    plt.figure()
    for joint_index in range(0, num_dof):
        plt.subplot(3, 3, joint_index + 1)
        plt.grid()
        plt.plot(g[:, joint_index], label="Newton Euler")
        plt.plot(gU[:, joint_index], label="diff U")
        plt.legend()
    plt.suptitle("Gravity")

    plt.figure()
    for joint_index in range(0, num_dof):
        plt.subplot(3, 3, joint_index + 1)
        plt.grid()
        plt.plot(tau[:, joint_index], label="True")
        plt.plot(m[:, joint_index] + c[:, joint_index] + g[:, joint_index], label="sum of comp")
        plt.legend()
    plt.suptitle("torque decomposition")

    plt.figure()
    for joint_index in range(0, num_dof):

        plt.grid()
        plt.plot(U)

    plt.suptitle("POTENTIAL ENERGY")

    # N = int(1/(T_sampling*downsampling)*sim_interval)
    # xf = fftfreq(N, T_sampling*downsampling)
    # plt.figure()
    # for joint_index in range(0,num_dof):
    #     plt.subplot(3,3,joint_index+1)
    #     plt.grid()
    #     plt.plot(xf, np.abs(fft(q_tr[:,joint_index], N)), label='FFT q_'+str(joint_index+1))
    #     # plt.plot(xf, np.abs(fft(q_dot_tr[:,joint_index], N)), label='FFT dq_'+str(joint_index+1))
    #     # plt.plot(xf, np.abs(fft(q_ddot_tr[:,joint_index], N)), label='FFT ddq_'+str(joint_index+1))
    #     # plt.plot(xf, np.abs(fft(tau_tr[:,joint_index], N)), label='FFT tau_'+str(joint_index+1))
    #     plt.legend()
    # plt.suptitle('TRAINING FFT')

    # plt.figure()
    # for joint_index in range(0,num_dof):
    #    plt.subplot(3,3,joint_index+1)
    #    plt.grid()
    #    plt.plot(q_test[:,joint_index], label='q_'+str(joint_index+1))
    #    # plt.plot(q_ref_test[:,joint_index], label='q_ref_'+str(joint_index+1))
    #    # plt.plot(states_test[:,joint_index+4], label='dq_'+str(joint_index+1))
    #    # plt.plot(q_ddot_test[:,joint_index], label='ddq_'+str(joint_index+1))
    #    plt.plot(tau_test[:,joint_index], label='tau_'+str(joint_index+1))
    #    plt.legend()
    # plt.suptitle('TEST')

    plt.show()

# SAVE DATA
# print('\nSAVE DATA')
# get pandas dataset
q_names = ["q_" + str(k + 1) for k in range(0, num_active_joints)]
dq_names = ["dq_" + str(k + 1) for k in range(0, num_active_joints)]
ddq_names = ["ddq_" + str(k + 1) for k in range(0, num_active_joints)]
tau_names = ["tau_" + str(k + 1) for k in range(0, num_active_joints)]
tau_noiseless_names = ["tau_noiseless_" + str(k + 1) for k in range(0, num_active_joints)]
m_names = ["m_" + str(k + 1) for k in range(0, num_active_joints)]
c_names = ["c_" + str(k + 1) for k in range(0, num_active_joints)]
g_names = ["g_" + str(k + 1) for k in range(0, num_active_joints)]
U_name = ["U"]
# phi_names = ['phi_'+str(k+1)+'_'+str(j+1) for k in range(0,num_dof) for j in range(0, 66) ]
# phi_names = ['phi_'+str(k+1)+'_'+str(j+1) for k in range(0,self.dof) for j in range(0, self.numeric_dyn_par.size)s ]
# data_names = q_names + dq_names + ddq_names + tau_names + phi_names
# data_tr = pd.DataFrame(data=np.concatenate([q_tr, q_dot_tr, q_ddot_tr, tau_tr, phi_tr], axis=1),
#                        columns=data_names)
# data_test = pd.DataFrame(data=np.concatenate([q_test, q_dot_test, q_ddot_test, tau_test, phi_test], axis=1),
#                          columns=data_names)
data_names = q_names + dq_names + ddq_names + tau_names + tau_noiseless_names + m_names + c_names + g_names + U_name
data = pd.DataFrame(
    data=np.concatenate(
        [
            q[:, :num_active_joints],
            q_dot[:, :num_active_joints],
            q_ddot[:, :num_active_joints],
            tau[:, :num_active_joints],
            tau_noiseless[:, :num_active_joints],
            m[:, :num_active_joints],
            c[:, :num_active_joints],
            g[:, :num_active_joints],
            U,
        ],
        axis=1,
    ),
    columns=data_names,
)

pkl.dump(data, open(saving_name_data, "wb"))
pkl.dump(M[:, :num_active_joints, :num_active_joints], open(saving_name_data_M, "wb"))
