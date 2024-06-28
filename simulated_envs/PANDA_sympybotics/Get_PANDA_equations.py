# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Computations of the PANDA robot dynamics equations
Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
        Giulio Giacomuzzo (giulio.giacomuzzo@gmail.com)
"""
import pickle as pkl
import sys

import sympy

sys.path.insert(0, "..")
import sympybotics
from Simulated_Robot import (
    Input_trj,
    Simulated_Robot,
    get_cylinder_inertia,
    get_tau_filtered,
    get_trj_filtered,
    get_trj_quantized,
)

# ### ROBOT DEF ####
print("--- Define the kinematic and the dynamic ---")

# Kin def
dof = 7
rbtdef = sympybotics.RobotDef(
    "PANDA",  # robot name
    [
        (0.0, 0.0, 0.333, "q"),
        ("-pi/2", 0.0, 0.0, "q"),
        ("pi/2", 0.0, 0.316, "q"),
        ("pi/2", 0.0825, 0.0, "q"),
        ("-pi/2", -0.0825, 0.384, "q"),
        ("pi/2", 0.0, 0.0, "q"),
        (" pi/2", 0.088, 0.0, "q"),
    ],
    dh_convention="modified",  # either 'standard' or 'modified' (alpha, a, d, theta)
)


# Set frictions and grav
# rbtdef.frictionmodel = {'Coulomb', 'viscous'} # options are None or a combination of 'Coulomb', 'viscous' and 'offset'
rbtdef.frictionmodel = None
rbtdef.gravityacc = sympy.Matrix([0.0, 0.0, -9.81])  # optional, this is the default value
# dyn par numerical values ([(L_1xx,L_1xy,L_1xz,L_1yy,L_1yz,L_1zz,l_1x,l_1y,l_1z,m_1),...])

# Set dyn params
m = [4.970684, 0.646926, 3.228604, 3.587895, 1.225946, 1.666555, 0.735522]  # list of masses of the links form 1 to 7

com = [
    (3.875e-03, 2.081e-03, -0.1750),  # list of centers of mass of the links from 1 to 7
    (-3.141e-03, -2.872e-02, 3.495e-03),
    (2.7518e-02, 3.9252e-02, -6.6502e-02),
    (-5.317e-02, 1.04419e-01, 2.7454e-02),
    (-1.1953e-02, 4.1065e-02, -3.8437e-02),
    (6.0149e-02, -1.4117e-02, -1.0517e-02),
    (1.0517e-02, -4.252e-03, 6.1597e-02),
]

l = [
    tuple(
        v * m[k] for v in com[k]
    )  # list of first moments of inertia of each link: CoM * mass. See https://github.com/cdsousa/SymPyBotics/issues/25
    for k in range(0, rbtdef.dof)
]

L = [
    (7.0337e-01, -1.3900e-04, 6.7720e-03, 7.0661e-01, 1.9169e-02, 9.1170e-03),  # [Lxx Lxy Lxz Lyy Lyz Lzz]
    (7.9620e-03, -3.9250e-03, 1.0254e-02, 2.8110e-02, 7.0400e-04, 2.5995e-02),
    (3.7242e-02, -4.7610e-03, -1.1396e-02, 3.6155e-02, -1.2805e-02, 1.0830e-02),
    (2.5853e-02, 7.7960e-03, -1.3320e-03, 1.9552e-02, 8.6410e-03, 2.8323e-02),
    (3.5549e-02, -2.1170e-03, -4.0370e-03, 2.9474e-02, 2.2900e-04, 8.6270e-03),
    (1.9640e-03, 1.0900e-04, -1.1580e-03, 4.3540e-03, 3.4100e-04, 5.4330e-03),
    (1.2516e-02, -4.2800e-04, -1.1960e-03, 1.0027e-02, -7.4100e-04, 4.8150e-03),
]

num_dyn_params_nominal = [
    (*L[k], *l[k], m[k]) for k in range(0, rbtdef.dof)
]  # dyn par numerical values ([(L_1xx,L_1xy,L_1xz,L_1yy,L_1yz,L_1zz,l_1x,l_1y,l_1z,m_1),...])


# #### GET THE SIMULATED ROBOT OBJECT ####
print("--- Get the Simulated robot object ---")
rbt_sim = Simulated_Robot(rbtdef, num_dyn_params_nominal, "")
saving_name = "rbt_sim.pkl"
pkl.dump(rbt_sim, open(saving_name, "wb"))


# #### GET THE EQUATIONS ####
print("--- Get the equations ---")
rbt_sim.get_inv_dyn_eq()
rbt_sim.get_H_eq()
rbt_sim.get_Hb_eq()
rbt_sim.get_M_eq()
pkl.dump(rbt_sim.numeric_dyn_par, open("PANDA_dyn_par.pkl", "wb"))
rbt_sim.get_g_eq()
rbt_sim.get_c_eq()
rbt_sim.get_U_eq()
rbt_sim.get_g_from_U_eq()
# rbt_sim.get_f_eq()
