# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympybotics
from scipy import integrate, signal
from sympy import *
from sympy.utilities import lambdify


class Simulated_Robot:
    """Class that generates robot symbolic expression (exploiting sympybotics) and
    performs simulations.
    """

    def __init__(self, rbtdef, numeric_dyn_par_list=[], saving_path=""):

        # save the parameters
        self.saving_path = saving_path
        self.dof = rbtdef.dof

        # get the robot symbolic equations
        self.rbt = sympybotics.RobotAllSymb(rbtdef)
        self.rbt_code = sympybotics.RobotDynCode(rbtdef, verbose=False)

        # save the dynamics parameters
        self.numeric_dyn_par_list = numeric_dyn_par_list
        numeric_dyn_par = []
        for par in numeric_dyn_par_list:
            numeric_dyn_par = numeric_dyn_par + list(par)
        self.numeric_dyn_par = np.array(numeric_dyn_par)

    def get_filt_rnd_input_trj(
        self, sampling_time, num_samples, downsampling, fc, std_vect, mean_vect=None, saving_name=None
    ):
        """method that returns an input trajectory object.
        input trajectories are defined as the filtered gaussian gaussian noise"""

        # get the time vector
        t = np.arange(0, sampling_time * (num_samples * downsampling), sampling_time).reshape(-1, 1)
        # generate the white noise trj
        initial_rest = int(0.5 / sampling_time)
        pos = np.concatenate([np.zeros([initial_rest, self.dof]), np.random.randn(t.size, self.dof) * std_vect], 0)

        b, a = signal.butter(2, fc)
        pos_trj_list = []
        vel_trj_list = []
        acc_trj_list = []
        for joint_index in range(0, self.dof):
            # filt the positions
            # pos_filt = signal.filtfilt(b, a, pos[:,joint_index])[:-(initial_rest)]
            pos_filt = (signal.lfilter(b, a, signal.lfilter(b, a, signal.lfilter(b, a, pos[:, joint_index]))))[
                :-(initial_rest)
            ]
            # pos_filt = signal.filtfilt(b, a, pos[:,joint_index])
            # get velocities and accelerations
            vel = signal.filtfilt(
                b,
                a,
                (
                    pos_filt[
                        2:,
                    ]
                    - pos_filt[
                        :-2,
                    ]
                )
                / (2 * sampling_time),
            )
            acc = signal.filtfilt(
                b,
                a,
                (
                    vel[
                        2:,
                    ]
                    - vel[
                        :-2,
                    ]
                )
                / (2 * sampling_time),
            )
            # vel = signal.filtfilt(b, a, (pos_filt[2:,]-pos_filt[:-2,])/(2*sampling_time))
            # acc = signal.filtfilt(b, a, (vel[2:,]-vel[:-2,])/(2*sampling_time))
            # cut positions and velocities to obtain the same number of samples
            pos_filt = pos_filt[2:-2]
            vel = vel[1:-1]
            # append the results
            pos_trj_list.append(pos_filt[::downsampling].reshape([-1, 1]))
            vel_trj_list.append(vel[::downsampling].reshape([-1, 1]))
            acc_trj_list.append(acc[::downsampling].reshape([-1, 1]))
        trj_obj = Input_trj(
            np.concatenate(pos_trj_list, axis=1),
            np.concatenate(vel_trj_list, axis=1),
            np.concatenate(acc_trj_list, axis=1),
            t[::downsampling, :],
        )
        if saving_name is not None:
            pkl.dump(trj_obj, open(saving_name, "wb"))
        # return the trj object
        return trj_obj

    def get_rand_sin_input_trj(
        self,
        sampling_time,
        num_samples,
        num_sin,
        max_ampl,
        max_HZ,
        min_HZ,
        flg_pos=False,
        verbose=False,
        saving_name=None,
    ):
        """method that returns an input trajectory object.
        input trajectories are defined by a sum of random sinusoid (for each joint)"""

        # get the time vector
        t = np.arange(0, sampling_time * (num_samples), sampling_time).reshape(-1, 1)

        # get the pos, vect and acc functions
        get_pos = lambda t, ampl, omega: np.sum(
            ampl * np.sin(t * omega), axis=1
        )  # t*omega is array (num_samples x num_sin)
        get_vel = lambda t, ampl, omega: np.sum((omega * ampl) * np.cos(t * omega), axis=1)
        get_acc = lambda t, ampl, omega: np.sum(-(omega**2 * ampl) * np.sin(t * omega), axis=1)

        # generate rand par (ampl, omega)
        # ampl = max_ampl/num_sin
        ampl = max_ampl / num_sin
        omega = (
            2
            * np.pi
            * np.random.choice([-1, 1], [self.dof, num_sin])
            * (min_HZ + (max_HZ - min_HZ) * np.random.rand(self.dof, num_sin))
        )

        # #get trj object
        if flg_pos:
            pos_trj_list = [get_pos(t, ampl, omega[k, :]).reshape(-1, 1) for k in range(0, self.dof)]
            vel_trj_list = [get_vel(t, ampl, omega[k, :]).reshape(-1, 1) for k in range(0, self.dof)]
            acc_trj_list = [get_acc(t, ampl, omega[k, :]).reshape(-1, 1) for k in range(0, self.dof)]
        else:
            ampl = np.abs(ampl / omega)
            # ampl = np.abs(ampl/omega)*np.random.rand(self.dof, num_sin)
            # ampl = np.abs(ampl)*np.random.rand(self.dof, num_sin)
            pos_trj_list = [get_pos(t, ampl[k, :], omega[k, :]).reshape(-1, 1) for k in range(0, self.dof)]
            vel_trj_list = [get_vel(t, ampl[k, :], omega[k, :]).reshape(-1, 1) for k in range(0, self.dof)]
            acc_trj_list = [get_acc(t, ampl[k, :], omega[k, :]).reshape(-1, 1) for k in range(0, self.dof)]
        trj_obj = Input_trj(
            np.concatenate(pos_trj_list, axis=1),
            np.concatenate(vel_trj_list, axis=1),
            np.concatenate(acc_trj_list, axis=1),
            t,
        )

        if verbose:
            trj_obj.print_trj()

        if saving_name is not None:
            pkl.dump(trj_obj, open(saving_name, "wb"))

        # return the trj object
        return trj_obj

    def get_sin_3_joint(self, sampling_time, omega, ampl, num_samples, verbose=False, saving_name=None):
        """method that returns an input trajectory object considering the motion
        of the single third joint"""

        # get the time vector
        t = np.arange(0, sampling_time * (num_samples), sampling_time).reshape(-1, 1)
        pos = (ampl * (np.cos(omega * t) - 1)).reshape(num_samples)
        vel = (-ampl * np.sin(omega * t) * omega).reshape(num_samples)
        acc = (-ampl * np.cos(omega * t) * (omega**2)).reshape(num_samples)

        pos_tot = np.ones([num_samples, 1]) * np.array([0, np.pi / 2, 0, 0, 0, 0])
        pos_tot[:, 2] = pos
        vel_tot = np.zeros([num_samples, self.dof])
        vel_tot[:, 2] = vel
        acc_tot = np.zeros([num_samples, self.dof])
        acc_tot[:, 2] = acc
        trj_obj = Input_trj(pos_tot, vel_tot, acc_tot, t)

        if verbose:
            trj_obj.print_trj()

        if saving_name is not None:
            pkl.dump(trj_obj, open(saving_name, "wb"))

        # return the trj object
        return trj_obj

    def get_sin_3_joint_v(self, sampling_time, omega, ampl, num_samples, verbose=False, saving_name=None):
        """method that returns an input trajectory object considering the motion
        of the single third joint"""

        # get the time vector
        t = np.arange(0, sampling_time * (num_samples), sampling_time).reshape(-1, 1)
        pos = (ampl * (np.cos(omega * t) - 1)).reshape(num_samples)
        vel = (-ampl * np.sin(omega * t) * omega).reshape(num_samples)
        acc = (-ampl * np.cos(omega * t) * (omega**2)).reshape(num_samples)

        pos_tot = np.ones([num_samples, 1]) * np.array([0, 0, 0, 0, 0, 0])
        pos_tot[:, 2] = pos
        vel_tot = np.zeros([num_samples, self.dof])
        vel_tot[:, 2] = vel
        acc_tot = np.zeros([num_samples, self.dof])
        acc_tot[:, 2] = acc
        trj_obj = Input_trj(pos_tot, vel_tot, acc_tot, t)

        if verbose:
            trj_obj.print_trj()

        if saving_name is not None:
            pkl.dump(trj_obj, open(saving_name, "wb"))

        # return the trj object
        return trj_obj

    def get_zero_input_trj(self, sampling_time, num_samples, verbose=False, saving_name=None):
        """method that returns an input trajectory object.
        It's supposed to be still the robot"""

        # get the time vector
        t = np.arange(0, sampling_time * (num_samples), sampling_time).reshape(-1, 1)
        trj_obj = Input_trj(
            np.zeros([int(num_samples), self.dof]),
            np.zeros([int(num_samples), self.dof]),
            np.zeros([int(num_samples), self.dof]),
            t,
        )

        if verbose:
            trj_obj.print_trj()

        if saving_name is not None:
            pkl.dump(trj_obj, open(saving_name, "wb"))
        return trj_obj

    def get_inv_dyn_eq(self):
        """Generates the robot inverse dynamics function"""

        inv_dyn_str = sympybotics.robotcodegen.robot_code_to_func(
            "py", self.rbt_code.invdyn_code, "inv_dyn_num", "inv_dyn", self.rbt_code.rbtdef
        )
        inv_dyn_str = "import math\n" + "from math import sin, cos\n" + inv_dyn_str
        saving_name = self.saving_path + "inv_dyn.py"
        f = open(saving_name, "w")
        f.write(inv_dyn_str)
        f.close()

    def get_M_eq(self):
        """Generates the robot generalized inertia matrix function"""

        M_str = sympybotics.robotcodegen.robot_code_to_func(
            "py", self.rbt_code.M_code, "M_num", "M", self.rbt_code.rbtdef
        )
        M_str = "import math\n" + "from math import sin, cos\n" + M_str
        saving_name = self.saving_path + "M.py"
        f = open(saving_name, "w")
        f.write(M_str)
        f.close()

    def get_g_eq(self):
        """Generates the robot gravitational force function"""

        g_str = sympybotics.robotcodegen.robot_code_to_func(
            "py", self.rbt_code.g_code, "g_num", "g", self.rbt_code.rbtdef
        )
        g_str = "import math\n" + "from math import sin, cos\n" + g_str
        saving_name = self.saving_path + "g.py"
        # print('saving_name',saving_name)
        f = open(saving_name, "w")
        f.write(g_str)
        f.close()

    def get_c_eq(self):
        """Generates the robot coriolis force function"""

        c_str = sympybotics.robotcodegen.robot_code_to_func(
            "py", self.rbt_code.c_code, "c_num", "c", self.rbt_code.rbtdef
        )
        c_str = "import math\n" + "from math import sin, cos\n" + c_str
        saving_name = self.saving_path + "c.py"
        f = open(saving_name, "w")
        f.write(c_str)
        f.close()

    def get_f_eq(self):
        """Generates the robot friction force"""

        f_str = sympybotics.robotcodegen.robot_code_to_func(
            "py", self.rbt_code.f_code, "f_num", "f", self.rbt_code.rbtdef
        )
        f_str = "import math\n" + "from math import sin, cos\n" + f_str
        saving_name = self.saving_path + "f.py"
        f = open(saving_name, "w")
        f.write(f_str)
        f.close()

    def get_H_eq(self):
        """Generates the robot H function"""

        H_str = sympybotics.robotcodegen.robot_code_to_func(
            "py", self.rbt_code.H_code, "H_num", "H", self.rbt_code.rbtdef
        )
        H_str = "import math\n" + "from math import sin, cos\n" + H_str
        saving_name = self.saving_path + "H.py"
        f = open(saving_name, "w")
        f.write(H_str)
        f.close()

    def get_U_symb(self):
        """Returns symbolic expression for robot Potential energy"""
        T = self.rbt_code.geo.T
        l = [Matrix([*self.numeric_dyn_par_list[i][6:9], self.numeric_dyn_par_list[i][-1]]) for i in range(self.dof)]
        p = [Matrix(T[i].dot(l[i]))[:-1, :] for i in range(self.dof)]
        # p = [T[i][:3, -1] for i in range(self.dof)]
        g0 = self.rbt_code.rbtdef.gravityacc
        u = [g0.dot(p[i]) for i in range(self.dof)]
        return -sum(u)

    def get_U_eq(self):
        """Generates the robot potential energy function"""

        def gen_U_code(U_exp, optimizations):
            f_str = "import math\n" + "from math import sin, cos\n\n"
            f_str += "def U(q):\n"
            for joint_index in range(self.dof):
                f_str += "    q" + str(joint_index + 1) + " = q[" + str(joint_index) + "]\n"
            f_str += gen_main_str(*cse(U_exp, optimizations=optimizations)) + "\n"
            # write the function
            file_f = open(self.saving_path + "U.py", "w")
            file_f.write(f_str)
            file_f.close()

        def gen_main_str(sub_expr, expr):
            # sub expressions computation str
            str_main = "    \n"
            for s_expr in sub_expr:
                str_main += "    " + str(s_expr[0]) + " = " + printing.lambdarepr.lambdarepr(s_expr[1]) + "\n"
            # U computation
            str_main += "    \n"
            str_main += "    return " + printing.lambdarepr.lambdarepr(expr[0]) + "\n"
            return str_main

        U = self.get_U_symb()
        gen_U_code(U, None)

    def get_g_from_U_eq(self):
        """Generates the gravitational torque deriving potential energy"""
        U = self.get_U_symb()
        q = symbols(["q" + str(i) for i in range(1, 8)])
        g = [diff(U, q[i]) for i in range(len(q))]

        def gen_g_code(g_exp, f_name):
            f_str = "import math\n" + "from math import sin, cos\n\n"
            f_str += "def " + f_name + "(q):\n"
            for joint_index in range(self.dof):
                f_str += "    q" + str(joint_index + 1) + " = q[" + str(joint_index) + "]\n"
            f_str += gen_main_str_list(*cse(g_exp, optimizations=None)) + "\n"
            # write the function
            file_f = open(f_name + ".py", "w")
            file_f.write(f_str)
            file_f.close()

        def gen_main_str_list(sub_expr, expr):
            # sub expressions computation str
            str_main = "    \n"
            for s_expr in sub_expr:
                str_main += "    " + str(s_expr[0]) + " = " + printing.lambdarepr.lambdarepr(s_expr[1]) + "\n"
            # U computation
            str_main += "    \n"
            str_main += "    g = []\n"
            for e in expr:
                str_main += "    g.append(" + printing.lambdarepr.lambdarepr(e) + ")\n"
            str_main += "    return g"
            return str_main

        gen_g_code(g, "g_U")

    #    def get_C_eq(self):

    def get_Hb_eq(self):
        """Generates the robot Hb function"""

        self.rbt_code.calc_base_parms()
        Hb_str = sympybotics.robotcodegen.robot_code_to_func(
            "py", self.rbt_code.Hb_code, "Hb_num", "Hb", self.rbt_code.rbtdef
        )
        Hb_str = "import math\n" + "from math import sin, cos\n" + Hb_str
        saving_name = self.saving_path + "Hb.py"
        f = open(saving_name, "w")
        f.write(Hb_str)
        f.close()

    def get_sim_no_contact(self, trj, inv_dyn, Hf, verbose=False, saving_name=None):
        """Generates a matrix (num.samples x dof) representing the motor torque
        vector that is actuating the robot in all the instants"""
        """without contact tau_ext is null, i.e. tau_tot = tau_m"""

        # initialize the vectors of motor torques
        tau = np.zeros((trj.num_samples, self.dof))
        tau_h = np.zeros((trj.num_samples, self.dof))
        phi = np.zeros([trj.num_samples, self.dof * (self.numeric_dyn_par.size)])

        # for each sample compute the motor torque
        for sample_index in range(0, trj.num_samples):
            # motor torque
            tau[sample_index, :] = np.array(
                inv_dyn(
                    self.numeric_dyn_par,
                    trj.position[sample_index, :],
                    trj.velocity[sample_index, :],
                    trj.acceleration[sample_index, :],
                )
            )
            # phi features
            phi[sample_index, :] = np.array(
                Hf(trj.position[sample_index, :], trj.velocity[sample_index, :], trj.acceleration[sample_index, :])
            ).reshape([1, -1])
            # compute the tau with the linear model
            tau_h[sample_index : sample_index + 1, :] = np.matmul(
                phi[sample_index : sample_index + 1, :].reshape(self.dof, self.numeric_dyn_par.size),
                self.numeric_dyn_par.reshape([-1, 1]),
            ).reshape([1, -1])

        if saving_name is not None:
            # pkl.dump(tau, open(saving_name, 'wb') )

            # return the pandas dataframe
            q_names = ["q_" + str(k + 1) for k in range(0, self.dof)]
            dq_names = ["dq_" + str(k + 1) for k in range(0, self.dof)]
            ddq_names = ["ddq_" + str(k + 1) for k in range(0, self.dof)]
            tau_names = ["tau_" + str(k + 1) for k in range(0, self.dof)]
            phi_names = [
                "phi_" + str(k + 1) + "_" + str(j + 1)
                for k in range(0, self.dof)
                for j in range(0, self.numeric_dyn_par.size)
            ]
            data_names = q_names + dq_names + ddq_names + phi_names + tau_names
            data = pd.DataFrame(
                data=np.concatenate([trj.position, trj.velocity, trj.acceleration, phi, tau], axis=1),
                columns=data_names,
            )
            pkl.dump(data, open(saving_name, "wb"))
        return tau, tau_h, data

    def get_W(self, trj, Hb):
        """Returns an array composed by a vertical pile of the Hb matrices"""

        W = []
        for sample_index in range(0, trj.num_samples):
            W.append(
                np.array(
                    Hb(trj.position[sample_index, :], trj.velocity[sample_index, :], trj.acceleration[sample_index, :])
                ).reshape([self.dof, -1])
            )
        W = np.concatenate(W, axis=0)
        return W

    def get_ident_par(self, trj, tau, H_b):
        """Returns the estimate of the base parameters vector for the direct
        estimation exploiting the linearity property (Lagrange)"""

        beta = np.linalg.inv(np.dot(np.transpose(H_b), H_b))
        beta = np.dot(beta, np.transpose(H_b))
        beta = np.dot(beta, tau.reshape([trj.num_samples * self.dof, -1]))
        return beta

    def get_f_J(self, link_index):
        """Returns the Jacobian as a symbolic function of the joint vector"""

        return lambdify(tuple(self.rbt.rbtdef.q[:]), self.rbt.kin.J[link_index])

    def get_J_stacked(self, trj):
        """Returns a matrix composed by a pile of Jacobian matrices"""

        # get symbolic function J(q) (like function get_f_J)
        f = lambdify(tuple(self.rbt.rbtdef.q[:]), self.rbt.kin.J[-1])

        # get the positions
        q = trj.position

        # evaluates J(q) in all position and stacks the matrices in row
        result = [f(*q[i, :]) for i in range(q.shape[0])]  # @me chk q.shape[0]
        Jac_stack = np.concatenate(result, axis=1)
        return Jac_stack

    def get_sim_with_contact(self, trj, force, f_inv_dyn, verbose=False, saving_name=None):
        """Performs simulation with contact and returns the motor torque and
        external torque"""

        # get the Jacobian function
        f_J = self.get_f_J(5)

        # initialize the output
        tau_m = np.zeros([trj.num_samples, self.dof])
        tau_ext = np.zeros([trj.num_samples, self.dof])

        # computes simulated motor torque for each instant
        for sample_index in range(0, trj.num_samples):
            # computes inv_dyn (total) tau
            tau_tot = np.array(
                f_inv_dyn(
                    self.numeric_dyn_par,
                    trj.position[sample_index, :],
                    trj.velocity[sample_index, :],
                    trj.acceleration[sample_index, :],
                )
            )
            # computes external tau, i.e. tau_ext = J.T force
            tau_ext[sample_index, :] = np.dot(f_J(*trj.position[sample_index, :]).T, force[sample_index, :]).T
            # computes motor torque
            tau_m[sample_index, :] = tau_tot - tau_ext[sample_index, :]

        if saving_name is not None:
            pkl.dump(tau_m, open(saving_name, "wb"))
        return tau_m, tau_ext

    def get_force_costant(self, trj, indexes, init, final):
        """Returns a ndarray (num_samples x 6) representing the contact forces
        that are characterized by an amplitude of 'indexes'. The forces are
        constant in the interval [init,final[ and null elsewhere"""

        gamma = np.zeros([trj.num_samples, 6])  # force matrices
        gamma[init:final, :] = np.ones([final - init, 6]) * np.array(indexes).reshape(1, -1)
        force_obj = Force_trj(gamma, trj.time)
        return force_obj

    def get_force_rand_sin(self, sampling_time, num_samples, num_sin, max_ampl, max_omega, saving_name=None):
        """Method that returns a ndarray (num_samples x 6).
        Forces are defined by a sum of random sinusoid (for each variable)"""

        # get the time vector
        t = np.arange(0, sampling_time * (num_samples), sampling_time).reshape(-1, 1)

        # get the force functions
        get_f = lambda t, ampl, omega: np.sum(
            ampl * np.sin(t * omega), axis=1
        )  # t*omega is array (num_samples x num_sin)

        # generate rand par (ampl, omega)
        ampl = -max_ampl + 2 * max_ampl * np.random.rand(6, num_sin)
        omega = -max_omega + 2 * max_omega * np.random.rand(6, num_sin)

        # get the forces
        f_trj_list = [get_f(t, ampl[k, :], omega[k, :]).reshape(-1, 1) for k in range(0, 6)]

        # squeeze functions
        max_abs_f = [np.max(np.abs(f_trj)) for f_trj in f_trj_list]
        correction_factor = [max_ampl / max_value for max_value in max_abs_f]
        f_trj_list_squeezed = [correction_factor[k] * f_trj_list[k] for k in range(0, 6)]

        # get force matrix
        force = np.concatenate(f_trj_list_squeezed, axis=1)
        force_obj = Force_trj(force, t)

        if saving_name is not None:
            pkl.dump(force_obj, open(saving_name, "wb"))

        # return the force
        return force_obj


class Force_trj:
    """class that constains the force trajectories"""

    def __init__(self, force, time):

        # force is a numpy array of shape (num_samples, 6)
        self.force = force
        self.time = time
        self.num_samples = force.shape[0]

    def get_force_matrix(self):
        """Returns a numpy array with forces (num_samples, 6)"""

        return self.force

    def print_trj(self):
        """Prints the forces"""

        for trj_index in range(0, 6):
            plt.figure()
            plt.plot(self.time, self.force[:, trj_index], label="force component " + str(trj_index + 1))
            plt.ylabel("Force " + str(trj_index + 1))
            plt.xlabel("Time")
            plt.grid()
            plt.legend()
        plt.show()


class Input_trj:
    """class that contains the input trj"""

    def __init__(self, position, velocity, acceleration, time):
        # position, velocity and acc are numpy array of shape [num_joint, num_samples]
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.time = time
        self.num_samples, self.num_trj = position.shape

    def get_input_matrix(self):
        """Returns a ndarray with position, vel and acc of dim (num_samples x 3dof)"""

        return np.concatenate([self.position, self.velocity, self.acceleration], axis=1)

    def print_trj(self, check_num_int=False):
        """Print position velocity and accelerations"""

        for trj_index in range(0, self.num_trj):
            plt.figure()

            plt.subplot(3, 1, 1)
            plt.plot(self.time, self.position[:, trj_index], label="pos")
            plt.ylabel("Pos " + str(trj_index + 1))
            plt.xlabel("Time")
            plt.grid()
            if check_num_int:
                plt.plot(
                    self.time,
                    integrate.cumtrapz(
                        self.velocity[:, trj_index], self.time[:, 0], initial=self.position[0, trj_index]
                    )
                    + np.concatenate([np.zeros(1), self.position[0, trj_index] * np.ones(self.num_samples - 1)]),
                )
            plt.legend()

            plt.subplot(3, 1, 2)
            plt.plot(self.time, self.velocity[:, trj_index], label="vel")
            plt.ylabel("Vel " + str(trj_index + 1))
            plt.xlabel("Time")
            plt.grid()
            if check_num_int:
                plt.plot(
                    self.time,
                    integrate.cumtrapz(
                        self.acceleration[:, trj_index], self.time[:, 0], initial=self.velocity[0, trj_index]
                    )
                    + np.concatenate([np.zeros(1), self.velocity[0, trj_index] * np.ones(self.num_samples - 1)]),
                )
            plt.legend()

            plt.subplot(3, 1, 3)
            plt.plot(self.time, self.acceleration[:, trj_index], label="acc")
            plt.ylabel("Acc " + str(trj_index + 1))
            plt.xlabel("Time")
            plt.grid()
            plt.legend()


def get_cylinder_inertia(m, h, r, principal_axis="x"):
    """Computes the cylinder inertia parameters of a cylinder (principal axis along the x axis)
    Parameters required are:
    - m --> mass of the cylinder
    - h --> height of the cylinder
    - r --> radius
    Function returns:
    (L_xx, L_xy, L_xz, L_yy, L_yz, L_zz)
    """

    L_xx = m / 2 * r**2
    L_zz = L_yy = m * (3 * r**2 + h**2)
    L_xy = L_xz = L_yz = 0.0
    if principal_axis == "x":
        return (L_xx, L_xy, L_xz, L_yy, L_yz, L_zz)
    elif principal_axis == "y":
        return (L_yy, L_xy, L_yz, L_xx, L_xz, L_zz)
    elif principal_axis == "z":
        return (L_zz, L_yz, L_xz, L_yy, L_xy, L_xx)


def print_tau_trj(mat):
    """Given the matrix of shape (num_samples, dof) representing the torques,
    prints the trajectories"""

    for i in range(mat.shape[1]):
        plt.figure()
        plt.plot(mat[:, i], label="tau " + str(i + 1))
        plt.grid()
        plt.legend()
    plt.show()


def get_trj_quantized(trj, Q, a, saving_name=None):
    """Get the quantized version of an input trajectory trj with quantization
    step size Q. a indicates the choice of step for numerical differentiation"""

    T = trj.time[1] - trj.time[0]
    q = (2 * np.pi / Q) * np.round(trj.position / (2 * np.pi / Q))
    dq = np.zeros([trj.num_samples, trj.position.shape[1]])
    ddq = np.zeros([trj.num_samples, trj.position.shape[1]])
    if a == 1:
        num = np.array([1.0, -1.0])
        den = np.array([T, 0])
    elif a == 2:
        num = np.array([1.0, 0, -1.0])
        den = np.array([2 * T, 0])
    else:
        return print("not implemented")
    for sample_index in range(0, trj.position.shape[1]):
        dq[:, sample_index] = signal.lfilter(num, den, q[:, sample_index])
    for sample_index in range(0, trj.velocity.shape[1]):
        ddq[:, sample_index] = signal.lfilter(num, den, dq[:, sample_index])
    trj_obj = Input_trj(q, dq, ddq, trj.time)
    if saving_name is not None:
        pkl.dump(trj_obj, open(saving_name, "wb"))
    return trj_obj


def get_trj_filtered(trj, fcp, fcv, fca, saving_name=None):
    """Get the filtered version of an input trajectory trj with different cut off
    frequencies for position, velocity, acceleration. If fcp=1 then no filtering of position"""

    if fcp != 1:
        b, a = signal.butter(4, fcp, btype="low", analog=False)
    d, c = signal.butter(4, fcv, btype="low", analog=False)
    f, e = signal.butter(4, fca, btype="low", analog=False)
    trj_obj = Input_trj(
        np.zeros(trj.position.shape), np.zeros(trj.position.shape), np.zeros(trj.position.shape), trj.time
    )
    for i in range(0, trj.position.shape[1]):
        if fcp == 1:
            trj_obj.position[:, i] = signal.lfilter(np.array([1.0, 0]), np.array([1.0, 0]), trj.position[:, i])
        else:
            trj_obj.position[:, i] = signal.lfilter(b, a, trj.position[:, i])
        trj_obj.velocity[:, i] = signal.lfilter(d, c, trj.velocity[:, i])
        trj_obj.acceleration[:, i] = signal.lfilter(f, e, trj.acceleration[:, i])
    if saving_name is not None:
        pkl.dump(trj_obj, open(saving_name, "wb"))
    return trj_obj


def get_tau_filtered(tau, fc, saving_name=None):
    """Get the filtered version of torque signals"""

    if fc == 1:
        if saving_name is not None:
            pkl.dump(tau, open(saving_name, "wb"))
        return tau
    tau_new = np.zeros(tau.shape)
    b, a = signal.butter(4, fc, "low", analog=False)
    for i in range(tau.shape[1]):
        tau_new[:, i] = signal.lfilter(b, a, tau[:, i], axis=-1, zi=None)
    if saving_name is not None:
        pkl.dump(tau_new, open(saving_name, "wb"))
    return tau_new
