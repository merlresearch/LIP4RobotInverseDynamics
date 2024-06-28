# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import math
from math import cos, sin


def c(parms, q, dq):
    #
    c_num = [0] * 7
    #
    x0 = cos(q[2])
    x1 = sin(q[3])
    x2 = sin(q[1])
    x3 = -dq[0]
    x4 = x2 * x3
    x5 = -0.316 * dq[1] ** 2 - 0.316 * x4**2
    x6 = sin(q[2])
    x7 = dq[1] * x6 + x0 * x4
    x8 = -x7
    x9 = cos(q[1])
    x10 = x3 * x9
    x11 = dq[1] * x10
    x12 = dq[2] * x8 - x11 * x6
    x13 = -x12
    x14 = -x10
    x15 = dq[2] + x14
    x16 = x15 * x7
    x17 = 0.0825 * x13 + 0.0825 * x16 + x5
    x18 = cos(q[3])
    x19 = -0.632 * x11
    x20 = x10 * x4
    x21 = -0.316 * x0 * x20 + x19 * x6
    x22 = -x4
    x23 = dq[1] * x0 + x22 * x6
    x24 = -(x15**2)
    x25 = -(x23**2) + x24
    x26 = x21 + 0.0825 * x25
    x27 = x1 * x17 + x18 * x26
    x28 = x23 * x7
    x29 = dq[0] * dq[1] * x2
    x30 = -x29
    x31 = x28 + x30
    x32 = x0 * x19 + 0.316 * x20 * x6
    x33 = -x32
    x34 = -0.0825 * x31 + x33
    x35 = dq[2] * x23 + x0 * x11
    x36 = x1 * x8 + x15 * x18
    x37 = dq[3] * x36 + x1 * x30 + x18 * x35
    x38 = x1 * x15 + x18 * x7
    x39 = -x38
    x40 = -x23
    x41 = dq[3] + x40
    x42 = parms[32] * x38 + parms[34] * x36 + parms[35] * x41
    x43 = -x35
    x44 = dq[3] * x39 + x1 * x43 + x18 * x30
    x45 = parms[30] * x38 + parms[31] * x36 + parms[32] * x41
    x46 = cos(q[4])
    x47 = sin(q[6])
    x48 = sin(q[4])
    x49 = -x41
    x50 = x38 * x46 + x48 * x49
    x51 = cos(q[5])
    x52 = dq[4] + x36
    x53 = sin(q[5])
    x54 = x50 * x51 + x52 * x53
    x55 = -x54
    x56 = x39 * x48 + x46 * x49
    x57 = -x13
    x58 = dq[4] * x56 + x37 * x46 + x48 * x57
    x59 = -x58
    x60 = dq[5] * x55 + x44 * x51 + x53 * x59
    x61 = -x60
    x62 = cos(q[6])
    x63 = -x56
    x64 = dq[5] + x63
    x65 = x47 * x64 + x54 * x62
    x66 = x47 * x55 + x62 * x64
    x67 = x65 * x66
    x68 = x54 * x64
    x69 = x61 + x68
    x70 = -(x41**2)
    x71 = -(x36**2)
    x72 = x70 + x71
    x73 = x36 * x38
    x74 = x57 + x73
    x75 = x27 - 0.0825 * x72 + 0.384 * x74
    x76 = -x48
    x77 = -x44
    x78 = x38 * x41
    x79 = x77 + x78
    x80 = x36 * x41
    x81 = x37 + x80
    x82 = x34 - 0.0825 * x79 + 0.384 * x81
    x83 = -x46
    x84 = x75 * x76 + x82 * x83
    x85 = -x84
    x86 = 0.088 * x69 + x85
    x87 = -x50
    x88 = x51 * x52 + x53 * x87
    x89 = -(x88**2)
    x90 = -(x64**2)
    x91 = x89 + x90
    x92 = -(x38**2)
    x93 = x70 + x92
    x94 = -x1
    x95 = x17 * x18 + x26 * x94
    x96 = x13 + x73
    x97 = 0.384 * x93 + x95 - 0.0825 * x96
    x98 = x46 * x75 + x76 * x82
    x99 = x51 * x98 + x53 * x97
    x100 = 0.088 * x91 + x99
    x101 = x100 * x62 + x47 * x86
    x102 = -x88
    x103 = dq[6] + x102
    x104 = -(x103**2)
    x105 = -(x66**2)
    x106 = -x37
    x107 = dq[4] * x87 + x106 * x48 + x46 * x57
    x108 = -x107
    x109 = -x65
    x110 = dq[5] * x88 + x44 * x53 + x51 * x58
    x111 = -x110
    x112 = dq[6] * x109 + x108 * x62 + x111 * x47
    x113 = x103 * x65
    x114 = parms[66] * (x104 + x105) + parms[67] * (-x61 + x67) + parms[68] * (x112 + x113) + parms[69] * x101
    x115 = x114 * x47
    x116 = -(x65**2)
    x117 = dq[6] * x66 + x108 * x47 + x110 * x62
    x118 = x103 * x66
    x119 = -x47
    x120 = x100 * x119 + x62 * x86
    x121 = parms[66] * (x61 + x67) + parms[67] * (x104 + x116) + parms[68] * (-x117 + x118) + parms[69] * x120
    x122 = x121 * x62
    x123 = x64 * x88
    x124 = -(x54**2)
    x125 = x50 * x56
    x126 = x52 * x56
    x127 = -(x50**2)
    x128 = -(x52**2)
    x129 = (
        parms[46] * (x125 + x44)
        + parms[47] * (x127 + x128)
        + parms[48] * (x126 + x59)
        + parms[49] * x84
        - parms[56] * x69
        - parms[57] * (x110 + x123)
        - parms[58] * (x124 + x89)
        - parms[59] * x85
        - x115
        - x122
    )
    x130 = x129 * x46
    x131 = parms[52] * x54 + parms[54] * x88 + parms[55] * x64
    x132 = -x98
    x133 = x132 * x53 + x51 * x97
    x134 = -x133
    x135 = parms[51] * x54 + parms[53] * x88 + parms[54] * x64
    x136 = parms[62] * x65 + parms[64] * x66 + parms[65] * x103
    x137 = x54 * x88
    x138 = x108 + x137
    x139 = x134 - 0.088 * x138
    x140 = parms[61] * x65 + parms[63] * x66 + parms[64] * x103
    x141 = (
        parms[60] * x117
        + parms[61] * x112
        + parms[62] * x61
        + parms[67] * x139
        - parms[68] * x120
        - x103 * x140
        + x136 * x66
    )
    x142 = parms[60] * x65 + parms[61] * x66 + parms[62] * x103
    x143 = (
        parms[61] * x117
        + parms[63] * x112
        + parms[64] * x61
        - parms[66] * x139
        + parms[68] * x101
        + x103 * x142
        + x109 * x136
    )
    x144 = (
        parms[50] * x110
        + parms[51] * x60
        + parms[52] * x108
        + parms[57] * x85
        + parms[58] * x134
        + x119 * x143
        + x131 * x88
        - x135 * x64
        + x141 * x62
    )
    x145 = parms[40] * x50 + parms[41] * x56 + parms[42] * x52
    x146 = parms[41] * x50 + parms[43] * x56 + parms[44] * x52
    x147 = parms[50] * x54 + parms[51] * x88 + parms[52] * x64
    x148 = (
        parms[62] * x117
        + parms[64] * x112
        + parms[65] * x61
        + parms[66] * x120
        - parms[67] * x101
        + x140 * x65
        - x142 * x66
    )
    x149 = (
        parms[51] * x110
        + parms[53] * x60
        + parms[54] * x108
        - parms[56] * x85
        + parms[58] * x99
        - 0.088 * x115
        - 0.088 * x122
        + x131 * x55
        + x147 * x64
        - x148
    )
    x150 = (
        parms[42] * x58
        + parms[44] * x107
        + parms[45] * x44
        + parms[46] * x84
        + parms[47] * x132
        + x144 * x53
        + x145 * x63
        + x146 * x50
        + x149 * x51
    )
    x151 = parms[66] * (-x112 + x113) + parms[67] * (x117 + x118) + parms[68] * (x105 + x116) + parms[69] * x139
    x152 = parms[56] * x138 + parms[57] * (x124 + x90) + parms[58] * (x111 + x123) + parms[59] * x133 - x151
    x153 = -x53
    x154 = -(x56**2)
    x155 = x50 * x52
    x156 = (
        parms[56] * x91
        + parms[57] * (-x108 + x137)
        + parms[58] * (x60 + x68)
        + parms[59] * x99
        + x114 * x62
        + x119 * x121
    )
    x157 = (
        parms[46] * (x128 + x154)
        + parms[47] * (x125 + x77)
        + parms[48] * (x107 + x155)
        + parms[49] * x98
        + x152 * x153
        + x156 * x51
    )
    x158 = x157 * x48
    x159 = (
        parms[31] * x37
        + parms[33] * x44
        + parms[34] * x13
        - parms[36] * x34
        + parms[38] * x27
        - 0.0825 * x130
        + x150
        - 0.0825 * x158
        + x39 * x42
        + x41 * x45
    )
    x160 = parms[21] * x7 + parms[23] * x23 + parms[24] * x15
    x161 = parms[22] * x7 + parms[24] * x23 + parms[25] * x15
    x162 = parms[42] * x50 + parms[44] * x56 + parms[45] * x52
    x163 = (
        parms[40] * x58
        + parms[41] * x107
        + parms[42] * x44
        + parms[47] * x97
        + parms[48] * x85
        + x144 * x51
        - x146 * x52
        + x149 * x153
        + x162 * x56
    )
    x164 = parms[31] * x38 + parms[33] * x36 + parms[34] * x41
    x165 = (
        parms[52] * x110
        + parms[54] * x60
        + parms[55] * x108
        + parms[56] * x133
        - parms[57] * x99
        + x102 * x147
        + x135 * x54
        + x141 * x47
        + x143 * x62
        - 0.088 * x151
    )
    x166 = (
        parms[41] * x58
        + parms[43] * x107
        + parms[44] * x44
        - parms[46] * x97
        + parms[48] * x98
        + x145 * x52
        + x162 * x87
        - x165
    )
    x167 = (
        parms[30] * x37
        + parms[31] * x44
        + parms[32] * x13
        + parms[37] * x34
        - parms[38] * x95
        - 0.384 * x130
        - 0.384 * x158
        + x163 * x46
        + x164 * x49
        + x166 * x76
        + x36 * x42
    )
    x168 = (
        parms[20] * x35
        + parms[21] * x12
        + parms[22] * x30
        + parms[27] * x5
        + parms[28] * x33
        - x15 * x160
        + x159 * x94
        + x161 * x23
        + x167 * x18
    )
    x169 = dq[1] * parms[15] + parms[12] * x4 + parms[14] * x10
    x170 = dq[1] * parms[14] + parms[11] * x4 + parms[13] * x10
    x171 = (
        parms[46] * (x108 + x155)
        + parms[47] * (x126 + x58)
        + parms[48] * (x127 + x154)
        + parms[49] * x97
        + x152 * x51
        + x156 * x53
    )
    x172 = parms[36] * x96 + parms[37] * x93 + parms[38] * (x106 + x80) + parms[39] * x95 + x171
    x173 = x157 * x46
    x174 = parms[36] * x72 + parms[37] * x74 + parms[38] * (x44 + x78) + parms[39] * x27 + x129 * x76 + x173
    x175 = (
        parms[26] * x25 + parms[27] * (x28 - x30) + parms[28] * (x12 + x16) + parms[29] * x21 + x172 * x94 + x174 * x18
    )
    x176 = (
        parms[32] * x37
        + parms[34] * x44
        + parms[35] * x13
        + parms[36] * x95
        - parms[37] * x27
        + 0.384 * x129 * x48
        + x163 * x76
        + x164 * x38
        + x166 * x83
        - 0.0825 * x171
        - 0.384 * x173
        - x36 * x45
    )
    x177 = parms[20] * x7 + parms[21] * x23 + parms[22] * x15
    x178 = (
        parms[21] * x35
        + parms[23] * x12
        + parms[24] * x30
        - parms[26] * x5
        + parms[28] * x21
        - 0.0825 * x1 * x174
        + x15 * x177
        + x161 * x8
        - 0.0825 * x172 * x18
        - x176
    )
    x179 = parms[36] * x79 + parms[37] * x81 + parms[38] * (x71 + x92) + parms[39] * x34 + x129 * x83 + x157 * x76
    x180 = parms[26] * x31 + parms[27] * (x24 - x7**2) + parms[28] * (x15 * x23 + x43) + parms[29] * x32 - x179
    x181 = (
        parms[22] * x35
        + parms[24] * x12
        + parms[25] * x30
        + parms[26] * x32
        - parms[27] * x21
        + x1 * x167
        + x159 * x18
        + x160 * x7
        + x177 * x40
        - 0.0825 * x179
    )
    x182 = dq[1] * parms[12] + parms[10] * x4 + parms[11] * x10
    #
    c_num[0] = -x2 * (
        -dq[1] * x170
        + parms[10] * x11
        + parms[11] * x29
        + x0 * x168
        - 0.316 * x0 * x180
        + x10 * x169
        - 0.316 * x175 * x6
        - x178 * x6
    ) - x9 * (dq[1] * x182 + parms[11] * x11 + parms[13] * x29 + x169 * x22 - x181)
    c_num[1] = (
        parms[12] * x11
        + parms[14] * x29
        + 0.316 * x0 * x175
        + x0 * x178
        + x14 * x182
        + x168 * x6
        + x170 * x4
        - 0.316 * x180 * x6
    )
    c_num[2] = x181
    c_num[3] = x176
    c_num[4] = x150
    c_num[5] = x165
    c_num[6] = x148
    #
    return c_num
