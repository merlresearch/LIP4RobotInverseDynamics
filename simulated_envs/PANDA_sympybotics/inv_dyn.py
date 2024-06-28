# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import math
from math import cos, sin


def inv_dyn(parms, q, dq, ddq):
    #
    inv_dyn_num = [0] * 7
    #
    x0 = cos(q[4])
    x1 = cos(q[5])
    x2 = sin(q[5])
    x3 = sin(q[4])
    x4 = cos(q[2])
    x5 = sin(q[1])
    x6 = -dq[0]
    x7 = x5 * x6
    x8 = sin(q[2])
    x9 = -x8
    x10 = dq[1] * x4 + x7 * x9
    x11 = -x10
    x12 = dq[3] + x11
    x13 = -x12
    x14 = cos(q[1])
    x15 = x14 * x6
    x16 = -x15
    x17 = dq[2] + x16
    x18 = sin(q[3])
    x19 = dq[1] * x8 + x4 * x7
    x20 = cos(q[3])
    x21 = x17 * x18 + x19 * x20
    x22 = x0 * x21 + x13 * x3
    x23 = -x22
    x24 = -x19
    x25 = x17 * x20 + x18 * x24
    x26 = dq[4] + x25
    x27 = x1 * x26 + x2 * x23
    x28 = dq[1] * x15
    x29 = -ddq[0]
    x30 = x28 + x29 * x5
    x31 = ddq[1] * x4 + dq[2] * x24 - x30 * x8
    x32 = -x31
    x33 = ddq[3] + x32
    x34 = -x33
    x35 = ddq[1] * x8 + dq[2] * x10 + x30 * x4
    x36 = dq[0] * dq[1] * x5 + x14 * x29
    x37 = ddq[2] - x36
    x38 = dq[3] * x25 + x18 * x37 + x20 * x35
    x39 = -x21
    x40 = x0 * x13 + x3 * x39
    x41 = dq[4] * x40 + x0 * x38 + x3 * x34
    x42 = -x35
    x43 = dq[3] * x39 + x18 * x42 + x20 * x37
    x44 = ddq[4] + x43
    x45 = dq[5] * x27 + x1 * x41 + x2 * x44
    x46 = x21 * x25
    x47 = x34 + x46
    x48 = -(x12**2)
    x49 = -(x25**2)
    x50 = x48 + x49
    x51 = -9.81 * x14
    x52 = -x51
    x53 = -0.316 * dq[1] ** 2 + x52 - 0.316 * x7**2
    x54 = x17 * x19
    x55 = 0.0825 * x32 + x53 + 0.0825 * x54
    x56 = -(x17**2)
    x57 = -(x10**2) + x56
    x58 = -9.81 * x5
    x59 = 0.316 * ddq[1] - 0.316 * x15 * x7 + x58
    x60 = -0.316 * x28 - 0.316 * x30
    x61 = x4 * x59 + x60 * x8
    x62 = 0.0825 * x57 + x61
    x63 = x18 * x55 + x20 * x62
    x64 = 0.384 * x47 - 0.0825 * x50 + x63
    x65 = -x3
    x66 = x12 * x21
    x67 = -x43 + x66
    x68 = x12 * x25
    x69 = x38 + x68
    x70 = x4 * x60 + x59 * x9
    x71 = -x70
    x72 = x10 * x19
    x73 = x37 + x72
    x74 = x71 - 0.0825 * x73
    x75 = -0.0825 * x67 + 0.384 * x69 + x74
    x76 = -x0
    x77 = x64 * x65 + x75 * x76
    x78 = -x77
    x79 = -x40
    x80 = dq[5] + x79
    x81 = x1 * x22 + x2 * x26
    x82 = parms[52] * x81 + parms[54] * x27 + parms[55] * x80
    x83 = parms[51] * x81 + parms[53] * x27 + parms[54] * x80
    x84 = x33 + x46
    x85 = -(x21**2)
    x86 = x48 + x85
    x87 = -x18
    x88 = x20 * x55 + x62 * x87
    x89 = -0.0825 * x84 + 0.384 * x86 + x88
    x90 = x0 * x64 + x65 * x75
    x91 = -x90
    x92 = x1 * x89 + x2 * x91
    x93 = -x92
    x94 = -x41
    x95 = -x81
    x96 = dq[5] * x95 + x1 * x44 + x2 * x94
    x97 = -x38
    x98 = dq[4] * x23 + x0 * x34 + x3 * x97
    x99 = -x98
    x100 = ddq[5] + x99
    x101 = cos(q[6])
    x102 = x80 * x81
    x103 = -x96
    x104 = x102 + x103
    x105 = 0.088 * x104 + x78
    x106 = x1 * x90 + x2 * x89
    x107 = -(x27**2)
    x108 = -(x80**2)
    x109 = x107 + x108
    x110 = x106 + 0.088 * x109
    x111 = sin(q[6])
    x112 = -x111
    x113 = x101 * x105 + x110 * x112
    x114 = -x45
    x115 = x101 * x81 + x111 * x80
    x116 = -x115
    x117 = dq[6] * x116 + x100 * x101 + x111 * x114
    x118 = x101 * x80 + x111 * x95
    x119 = -x27
    x120 = dq[6] + x119
    x121 = parms[62] * x115 + parms[64] * x118 + parms[65] * x120
    x122 = dq[6] * x118 + x100 * x111 + x101 * x45
    x123 = x27 * x81
    x124 = x100 + x123
    x125 = -0.088 * x124 + x93
    x126 = ddq[6] + x103
    x127 = parms[61] * x115 + parms[63] * x118 + parms[64] * x120
    x128 = (
        parms[60] * x122
        + parms[61] * x117
        + parms[62] * x126
        + parms[67] * x125
        - parms[68] * x113
        + x118 * x121
        - x120 * x127
    )
    x129 = x101 * x110 + x105 * x111
    x130 = parms[60] * x115 + parms[61] * x118 + parms[62] * x120
    x131 = (
        parms[61] * x122
        + parms[63] * x117
        + parms[64] * x126
        - parms[66] * x125
        + parms[68] * x129
        + x116 * x121
        + x120 * x130
    )
    x132 = (
        parms[50] * x45
        + parms[51] * x96
        + parms[52] * x100
        + parms[57] * x78
        + parms[58] * x93
        + x101 * x128
        + x112 * x131
        + x27 * x82
        - x80 * x83
    )
    x133 = (
        parms[62] * x122
        + parms[64] * x117
        + parms[65] * x126
        + parms[66] * x113
        - parms[67] * x129
        + x115 * x127
        - x118 * x130
    )
    x134 = x115 * x118
    x135 = -(x120**2)
    x136 = -(x118**2)
    x137 = x115 * x120
    x138 = parms[66] * (x135 + x136) + parms[67] * (-x126 + x134) + parms[68] * (x117 + x137) + parms[69] * x129
    x139 = x111 * x138
    x140 = parms[50] * x81 + parms[51] * x27 + parms[52] * x80
    x141 = x118 * x120
    x142 = -(x115**2)
    x143 = parms[66] * (x126 + x134) + parms[67] * (x135 + x142) + parms[68] * (-x122 + x141) + parms[69] * x113
    x144 = x101 * x143
    x145 = (
        parms[51] * x45
        + parms[53] * x96
        + parms[54] * x100
        - parms[56] * x78
        + parms[58] * x106
        - x133
        - 0.088 * x139
        + x140 * x80
        - 0.088 * x144
        + x82 * x95
    )
    x146 = -x2
    x147 = parms[42] * x22 + parms[44] * x40 + parms[45] * x26
    x148 = parms[41] * x22 + parms[43] * x40 + parms[44] * x26
    x149 = (
        parms[40] * x41
        + parms[41] * x98
        + parms[42] * x44
        + parms[47] * x89
        + parms[48] * x78
        + x1 * x132
        + x145 * x146
        + x147 * x40
        - x148 * x26
    )
    x150 = x27 * x80
    x151 = -(x81**2)
    x152 = -(x22**2)
    x153 = -(x26**2)
    x154 = x22 * x40
    x155 = x26 * x40
    x156 = (
        parms[46] * (x154 + x44)
        + parms[47] * (x152 + x153)
        + parms[48] * (x155 + x94)
        + parms[49] * x77
        - parms[56] * x104
        - parms[57] * (x150 + x45)
        - parms[58] * (x107 + x151)
        - parms[59] * x78
        - x139
        - x144
    )
    x157 = x0 * x156
    x158 = (
        parms[56] * x109
        + parms[57] * (-x100 + x123)
        + parms[58] * (x102 + x96)
        + parms[59] * x106
        + x101 * x138
        + x112 * x143
    )
    x159 = parms[66] * (-x117 + x137) + parms[67] * (x122 + x141) + parms[68] * (x136 + x142) + parms[69] * x125
    x160 = parms[56] * x124 + parms[57] * (x108 + x151) + parms[58] * (x114 + x150) + parms[59] * x92 - x159
    x161 = -(x40**2)
    x162 = x22 * x26
    x163 = (
        parms[46] * (x153 + x161)
        + parms[47] * (x154 - x44)
        + parms[48] * (x162 + x98)
        + parms[49] * x90
        + x1 * x158
        + x146 * x160
    )
    x164 = x163 * x3
    x165 = parms[31] * x21 + parms[33] * x25 + parms[34] * x12
    x166 = parms[32] * x21 + parms[34] * x25 + parms[35] * x12
    x167 = parms[40] * x22 + parms[41] * x40 + parms[42] * x26
    x168 = (
        parms[52] * x45
        + parms[54] * x96
        + parms[55] * x100
        + parms[56] * x92
        - parms[57] * x106
        + x101 * x131
        + x111 * x128
        + x119 * x140
        - 0.088 * x159
        + x81 * x83
    )
    x169 = (
        parms[41] * x41
        + parms[43] * x98
        + parms[44] * x44
        - parms[46] * x89
        + parms[48] * x90
        + x147 * x23
        + x167 * x26
        - x168
    )
    x170 = (
        parms[30] * x38
        + parms[31] * x43
        + parms[32] * x33
        + parms[37] * x74
        - parms[38] * x88
        + x0 * x149
        + x13 * x165
        - 0.384 * x157
        - 0.384 * x164
        + x166 * x25
        + x169 * x65
    )
    x171 = parms[36] * x67 + parms[37] * x69 + parms[38] * (x49 + x85) + parms[39] * x74 + x156 * x76 + x163 * x65
    x172 = parms[20] * x19 + parms[21] * x10 + parms[22] * x17
    x173 = parms[21] * x19 + parms[23] * x10 + parms[24] * x17
    x174 = parms[30] * x21 + parms[31] * x25 + parms[32] * x12
    x175 = (
        parms[42] * x41
        + parms[44] * x98
        + parms[45] * x44
        + parms[46] * x77
        + parms[47] * x91
        + x1 * x145
        + x132 * x2
        + x148 * x22
        + x167 * x79
    )
    x176 = (
        parms[31] * x38
        + parms[33] * x43
        + parms[34] * x33
        - parms[36] * x74
        + parms[38] * x63
        + x12 * x174
        - 0.0825 * x157
        - 0.0825 * x164
        + x166 * x39
        + x175
    )
    x177 = (
        parms[22] * x35
        + parms[24] * x31
        + parms[25] * x37
        + parms[26] * x70
        - parms[27] * x61
        + x11 * x172
        + x170 * x18
        - 0.0825 * x171
        + x173 * x19
        + x176 * x20
    )
    x178 = dq[1] * parms[12] + parms[10] * x7 + parms[11] * x15
    x179 = dq[1] * parms[15] + parms[12] * x7 + parms[14] * x15
    x180 = parms[22] * x19 + parms[24] * x10 + parms[25] * x17
    x181 = (
        parms[20] * x35
        + parms[21] * x31
        + parms[22] * x37
        + parms[27] * x53
        + parms[28] * x71
        + x10 * x180
        - x17 * x173
        + x170 * x20
        + x176 * x87
    )
    x182 = x0 * x163
    x183 = parms[36] * x50 + parms[37] * x47 + parms[38] * (x43 + x66) + parms[39] * x63 + x156 * x65 + x182
    x184 = (
        parms[46] * (x162 + x99)
        + parms[47] * (x155 + x41)
        + parms[48] * (x152 + x161)
        + parms[49] * x89
        + x1 * x160
        + x158 * x2
    )
    x185 = parms[36] * x84 + parms[37] * x86 + parms[38] * (x68 + x97) + parms[39] * x88 + x184
    x186 = (
        parms[26] * x57 + parms[27] * (-x37 + x72) + parms[28] * (x31 + x54) + parms[29] * x61 + x183 * x20 + x185 * x87
    )
    x187 = parms[26] * x73 + parms[27] * (-(x19**2) + x56) + parms[28] * (x10 * x17 + x42) + parms[29] * x70 - x171
    x188 = (
        parms[32] * x38
        + parms[34] * x43
        + parms[35] * x33
        + parms[36] * x88
        - parms[37] * x63
        + x149 * x65
        + 0.384 * x156 * x3
        + x165 * x21
        + x169 * x76
        - x174 * x25
        - 0.384 * x182
        - 0.0825 * x184
    )
    x189 = (
        parms[21] * x35
        + parms[23] * x31
        + parms[24] * x37
        - parms[26] * x53
        + parms[28] * x61
        + x17 * x172
        - 0.0825 * x18 * x183
        + x180 * x24
        - 0.0825 * x185 * x20
        - x188
    )
    x190 = dq[1] * parms[14] + parms[11] * x7 + parms[13] * x15
    #
    inv_dyn_num[0] = (
        ddq[0] * parms[5]
        - x14
        * (ddq[1] * parms[14] + dq[1] * x178 + parms[11] * x30 + parms[13] * x36 + parms[18] * x58 - x177 - x179 * x7)
        - x5
        * (
            ddq[1] * parms[12]
            - dq[1] * x190
            + parms[10] * x30
            + parms[11] * x36
            + parms[18] * x52
            + x15 * x179
            + x181 * x4
            - 0.316 * x186 * x8
            - 0.316 * x187 * x4
            + x189 * x9
        )
    )
    inv_dyn_num[1] = (
        ddq[1] * parms[15]
        + parms[12] * x30
        + parms[14] * x36
        + parms[16] * x51
        - parms[17] * x58
        + x16 * x178
        + x181 * x8
        + 0.316 * x186 * x4
        - 0.316 * x187 * x8
        + x189 * x4
        + x190 * x7
    )
    inv_dyn_num[2] = x177
    inv_dyn_num[3] = x188
    inv_dyn_num[4] = x175
    inv_dyn_num[5] = x168
    inv_dyn_num[6] = x133
    #
    return inv_dyn_num
