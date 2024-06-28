# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import math
from math import cos, sin


def g_U(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    q5 = q[4]
    q6 = q[5]
    q7 = q[6]

    x0 = sin(q2)
    x1 = cos(q2)
    x2 = sin(q4)
    x3 = x0 * x2
    x4 = 4.8076874943165 * x3
    x5 = cos(q4)
    x6 = x0 * x5
    x7 = 16.8799056608454 * x6
    x8 = sin(q3)
    x9 = x1 * x8
    x10 = cos(q3)
    x11 = sin(q5)
    x12 = x11 * x9
    x13 = cos(q5)
    x14 = x13 * x9
    x15 = x1 * x2
    x16 = 16.8799056608454 * x15
    x17 = x1 * x5
    x18 = 4.8076874943165 * x17
    x19 = sin(q6)
    x20 = x10 * x15 - x6
    x21 = x19 * x20
    x22 = cos(q6)
    x23 = 0.67524884163189 * x22
    x24 = 0.66581089427925 * x11
    x25 = -x10 * x17 - x3
    x26 = 0.14375311619778 * x13
    x27 = 0.67524884163189 * x19
    x28 = x12 + x13 * x25
    x29 = x22 * x28
    x30 = sin(q7)
    x31 = 0.07588510661394 * x30
    x32 = x11 * x25 - x14
    x33 = cos(q7)
    x34 = 0.03068018192664 * x33
    x35 = 0.03068018192664 * x30
    x36 = x21 + x29
    x37 = 0.07588510661394 * x33
    x38 = x0 * x8
    x39 = x0 * x10
    x40 = x0 * x2 * x8
    x41 = x0 * x5 * x8
    x42 = x11 * x39
    x43 = x13 * x39
    x44 = x19 * x40
    x45 = x11 * x41
    x46 = x13 * x41
    x47 = x42 + x46
    x48 = x22 * x47
    x49 = -x43 + x45
    x50 = -x44 + x48
    x51 = x10 * x3 + x17
    x52 = x10 * x6
    x53 = -x15 + x52
    x54 = x19 * x53
    x55 = x11 * x51
    x56 = x19 * x51
    x57 = 0.67524884163189 * x56
    x58 = x22 * x51
    x59 = 1.61833169193795 * x58
    x60 = x13 * x58 + x54
    x61 = x11 * x38
    x62 = x13 * x38
    x63 = x15 - x52
    x64 = x11 * x63
    x65 = x13 * x63
    x66 = x61 + x65
    x67 = x62 - x64
    x68 = x22 * x67
    x69 = x19 * x66
    x70 = x58 - x69
    x71 = -x62 + x64
    x72 = x22 * x66 + x56

    g = []
    g.append(0)
    g.append(
        -30.4535758268527 * x0
        - 6.71158958634432 * x1 * x10
        + 0.01993386669246 * x1
        + x10 * x16
        + x10 * x18
        - 0.14375311619778 * x12
        + 0.66581089427925 * x14
        - x20 * x23
        + 1.61833169193795 * x21
        - x24 * x25
        - x25 * x26
        + x27 * x28
        + 1.61833169193795 * x29
        + x31 * x32
        - x32 * x34
        + x35 * x36
        + x36 * x37
        + x4
        - x7
        + 0.27690780075318 * x9
    )
    g.append(
        x23 * x40
        + x27 * x47
        + x31 * x49
        - x34 * x49
        + x35 * x50
        + x37 * x50
        + 6.71158958634432 * x38
        + 0.27690780075318 * x39
        - 16.8799056608454 * x40
        - 4.8076874943165 * x41
        - 0.14375311619778 * x42
        + 0.66581089427925 * x43
        - 1.61833169193795 * x44
        - 0.66581089427925 * x45
        - 0.14375311619778 * x46
        + 1.61833169193795 * x48
    )
    g.append(
        -x10 * x4
        + x10 * x7
        + x13 * x57
        + x13 * x59
        - x16
        - x18
        - x23 * x53
        - x24 * x51
        - x26 * x51
        + x31 * x55
        - x34 * x55
        + x35 * x60
        + x37 * x60
        + 1.61833169193795 * x54
    )
    g.append(
        x27 * x67
        + x31 * x66
        - x34 * x66
        + x35 * x68
        + x37 * x68
        - 0.66581089427925 * x61
        - 0.14375311619778 * x62
        + 0.14375311619778 * x64
        - 0.66581089427925 * x65
        + 1.61833169193795 * x68
    )
    g.append(x23 * x66 + x35 * x70 + x37 * x70 + x57 + x59 - 1.61833169193795 * x69)
    g.append(-x31 * x72 + x34 * x72 + x35 * x71 + x37 * x71)
    return g
