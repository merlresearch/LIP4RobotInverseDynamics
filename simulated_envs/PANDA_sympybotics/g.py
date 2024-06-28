# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import math
from math import cos, sin


def g(parms, q):
    #
    g_num = [0] * 7
    #
    x0 = cos(q[1])
    x1 = sin(q[1])
    x2 = -9.81 * x1
    x3 = cos(q[2])
    x4 = x2 * x3
    x5 = -x4
    x6 = -x2
    x7 = sin(q[2])
    x8 = x6 * x7
    x9 = -x8
    x10 = -x9
    x11 = cos(q[3])
    x12 = sin(q[3])
    x13 = -9.81 * x0
    x14 = -x13
    x15 = x11 * x4 + x12 * x14
    x16 = sin(q[6])
    x17 = x11 * x14 + x12 * x5
    x18 = sin(q[5])
    x19 = cos(q[4])
    x20 = sin(q[4])
    x21 = -x20
    x22 = x15 * x19 + x21 * x9
    x23 = cos(q[5])
    x24 = x17 * x18 + x22 * x23
    x25 = cos(q[6])
    x26 = x10 * x19 + x15 * x21
    x27 = -x26
    x28 = x16 * x27 + x24 * x25
    x29 = parms[69] * x28
    x30 = x16 * x29
    x31 = -x24
    x32 = x16 * x31 + x25 * x27
    x33 = parms[69] * x32
    x34 = x25 * x33
    x35 = parms[49] * x26 - parms[59] * x27 - x30 - x34
    x36 = x19 * x35
    x37 = -x22
    x38 = parms[66] * x32 - parms[67] * x28
    x39 = -parms[56] * x27 + parms[58] * x24 - 0.088 * x30 - 0.088 * x34 - x38
    x40 = x17 * x23 + x18 * x37
    x41 = -x40
    x42 = -parms[66] * x41 + parms[68] * x28
    x43 = -x16
    x44 = parms[67] * x41 - parms[68] * x32
    x45 = parms[57] * x27 + parms[58] * x41 + x25 * x44 + x42 * x43
    x46 = parms[46] * x26 + parms[47] * x37 + x18 * x45 + x23 * x39
    x47 = -x18
    x48 = parms[69] * x41
    x49 = parms[59] * x40 - x48
    x50 = parms[59] * x24 + x25 * x29 + x33 * x43
    x51 = parms[49] * x22 + x23 * x50 + x47 * x49
    x52 = x20 * x51
    x53 = parms[36] * x10 + parms[38] * x15 - 0.0825 * x36 + x46 - 0.0825 * x52
    x54 = parms[47] * x17 + parms[48] * x27 + x23 * x45 + x39 * x47
    x55 = -x17
    x56 = parms[56] * x40 + parms[57] * x31 + x16 * x44 + x25 * x42 - 0.088 * x48
    x57 = parms[46] * x55 + parms[48] * x22 - x56
    x58 = parms[37] * x9 + parms[38] * x55 + x19 * x54 + x21 * x57 - 0.384 * x36 - 0.384 * x52
    x59 = parms[39] * x9 + x21 * x51 - x36
    x60 = parms[26] * x8 + parms[27] * x5 + x11 * x53 + x12 * x58 - 0.0825 * x59
    x61 = parms[29] * x8 - x59
    x62 = -x12
    x63 = parms[27] * x14 + parms[28] * x9 + x11 * x58 + x53 * x62
    x64 = parms[49] * x17 + x18 * x50 + x23 * x49
    x65 = parms[39] * x17 + x64
    x66 = x19 * x51
    x67 = parms[39] * x15 + x21 * x35 + x66
    x68 = parms[36] * x17 - parms[37] * x15 - x19 * x57 + 0.384 * x20 * x35 + x21 * x54 - 0.0825 * x64 - 0.384 * x66
    x69 = -parms[26] * x14 + parms[28] * x4 - 0.0825 * x11 * x65 - 0.0825 * x12 * x67 - x68
    x70 = parms[29] * x4 + x11 * x67 + x62 * x65
    #
    g_num[0] = -x0 * (parms[18] * x2 - x60) - x1 * (
        parms[18] * x14 - 0.316 * x3 * x61 + x3 * x63 - x69 * x7 - 0.316 * x7 * x70
    )
    g_num[1] = parms[16] * x13 + parms[17] * x6 + x3 * x69 + 0.316 * x3 * x70 - 0.316 * x61 * x7 + x63 * x7
    g_num[2] = x60
    g_num[3] = x68
    g_num[4] = x46
    g_num[5] = x56
    g_num[6] = x38
    #
    return g_num
