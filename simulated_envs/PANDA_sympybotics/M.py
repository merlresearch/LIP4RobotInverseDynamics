# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import math
from math import cos, sin


def M(parms, q):
    #
    M_num = [0] * 49
    #
    x0 = -sin(q[1])
    x1 = cos(q[3])
    x2 = cos(q[2])
    x3 = x0 * x2
    x4 = -x3
    x5 = sin(q[3])
    x6 = -cos(q[1])
    x7 = -x6
    x8 = x1 * x7 + x4 * x5
    x9 = -x8
    x10 = x1 * x3 + x5 * x7
    x11 = -0.316 * x3
    x12 = -x11
    x13 = x12 - 0.0825 * x7
    x14 = 0.384 * x10 + x13 - 0.0825 * x9
    x15 = sin(q[4])
    x16 = -x15
    x17 = cos(q[4])
    x18 = sin(q[2])
    x19 = -x0 * x18
    x20 = -x19
    x21 = 0.0825 * x20
    x22 = -0.316 * x18
    x23 = x0 * x22
    x24 = x1 * x23 + x21 * x5
    x25 = -x20
    x26 = x24 + 0.384 * x25
    x27 = x14 * x16 + x17 * x26
    x28 = -x10
    x29 = x15 * x28 + x17 * x25
    x30 = x10 * x17 + x15 * x25
    x31 = -x30
    x32 = sin(q[5])
    x33 = cos(q[5])
    x34 = x31 * x32 + x33 * x8
    x35 = -x29
    x36 = -x23
    x37 = x1 * x21 + x36 * x5
    x38 = -0.0825 * x20 + x37
    x39 = x27 * x33 + x32 * x38
    x40 = sin(q[6])
    x41 = x30 * x33 + x32 * x8
    x42 = cos(q[6])
    x43 = x35 * x40 + x41 * x42
    x44 = -x34
    x45 = -x39
    x46 = -x17
    x47 = x14 * x46 + x16 * x26
    x48 = -x47
    x49 = 0.088 * x44 + x48
    x50 = x40 * x45 + x42 * x49
    x51 = parms[66] * x44 - parms[68] * x43 + parms[69] * x50
    x52 = -x40
    x53 = x39 * x42 + x40 * x49
    x54 = -x41
    x55 = x35 * x42 + x40 * x54
    x56 = -parms[67] * x44 + parms[68] * x55 + parms[69] * x53
    x57 = -parms[57] * x35 + parms[58] * x34 + parms[59] * x39 + x42 * x56 + x51 * x52
    x58 = -x27
    x59 = x32 * x58 + x33 * x38
    x60 = -x59
    x61 = -0.088 * x35 + x60
    x62 = -parms[66] * x55 + parms[67] * x43 + parms[69] * x61
    x63 = parms[56] * x35 + parms[58] * x54 + parms[59] * x59 - x62
    x64 = -x32
    x65 = parms[47] * x9 + parms[48] * x29 + parms[49] * x27 + x33 * x57 + x63 * x64
    x66 = x17 * x65
    x67 = x42 * x51
    x68 = x40 * x56
    x69 = (
        parms[46] * x8
        + parms[48] * x31
        + parms[49] * x47
        - parms[56] * x44
        - parms[57] * x41
        - parms[59] * x48
        - x67
        - x68
    )
    x70 = parms[37] * x25 + parms[38] * x8 + parms[39] * x24 + x16 * x69 + x66
    x71 = parms[46] * x35 + parms[47] * x30 + parms[49] * x38 + x32 * x57 + x33 * x63
    x72 = parms[36] * x20 + parms[38] * x28 + parms[39] * x37 + x71
    x73 = -x5
    x74 = -parms[27] * x7 + parms[28] * x19 + parms[29] * x23 + x1 * x70 + x72 * x73
    x75 = x15 * x65
    x76 = -parms[36]
    x77 = x17 * x69
    x78 = -parms[68]
    x79 = parms[60] * x43 + parms[61] * x55 + parms[62] * x44 + parms[67] * x61 + x50 * x78
    x80 = -parms[66]
    x81 = parms[61] * x43 + parms[63] * x55 + parms[64] * x44 + parms[68] * x53 + x61 * x80
    x82 = (
        parms[50] * x41 + parms[51] * x34 + parms[52] * x35 + parms[57] * x48 + parms[58] * x60 + x42 * x79 + x52 * x81
    )
    x83 = -parms[67]
    x84 = parms[62] * x43 + parms[64] * x55 + parms[65] * x44 + parms[66] * x50 + x53 * x83
    x85 = -parms[56]
    x86 = (
        parms[51] * x41
        + parms[53] * x34
        + parms[54] * x35
        + parms[58] * x39
        + x48 * x85
        - 0.088 * x67
        - 0.088 * x68
        - x84
    )
    x87 = parms[42] * x30 + parms[44] * x29 + parms[45] * x8 + parms[46] * x47 + parms[47] * x58 + x32 * x82 + x33 * x86
    x88 = (
        parms[31] * x10
        + parms[33] * x8
        + parms[34] * x20
        + parms[38] * x24
        + x13 * x76
        - 0.0825 * x75
        - 0.0825 * x77
        + x87
    )
    x89 = -parms[38]
    x90 = parms[40] * x30 + parms[41] * x29 + parms[42] * x8 + parms[47] * x38 + parms[48] * x48 + x33 * x82 + x64 * x86
    x91 = (
        parms[52] * x41
        + parms[54] * x34
        + parms[55] * x35
        + parms[56] * x59
        + parms[57] * x45
        + x40 * x79
        + x42 * x81
        - 0.088 * x62
    )
    x92 = -parms[46]
    x93 = parms[41] * x30 + parms[43] * x29 + parms[44] * x8 + parms[48] * x27 + x38 * x92 - x91
    x94 = (
        parms[30] * x10
        + parms[31] * x8
        + parms[32] * x20
        + parms[37] * x13
        + x16 * x93
        + x17 * x90
        + x37 * x89
        - 0.384 * x75
        - 0.384 * x77
    )
    x95 = parms[20] * x3 + parms[21] * x19 + parms[22] * x7 + parms[28] * x12 + x1 * x94 + x73 * x88
    x96 = parms[36] * x9 + parms[37] * x10 + parms[39] * x13 + x16 * x65 + x46 * x69
    x97 = parms[26] * x7 + parms[28] * x4 + parms[29] * x11 - x96
    x98 = -parms[37]
    x99 = 0.384 * x15
    x100 = (
        parms[32] * x10
        + parms[34] * x8
        + parms[35] * x20
        + parms[36] * x37
        + x16 * x90
        + x24 * x98
        + x46 * x93
        - 0.384 * x66
        + x69 * x99
        - 0.0825 * x71
    )
    x101 = (
        parms[21] * x3
        + parms[23] * x19
        + parms[24] * x7
        + parms[28] * x23
        - 0.0825 * x1 * x72
        - x100
        - 0.0825 * x5 * x70
    )
    x102 = -x18
    x103 = (
        parms[22] * x3
        + parms[24] * x19
        + parms[25] * x7
        + parms[26] * x11
        + parms[27] * x36
        + x1 * x88
        + x5 * x94
        - 0.0825 * x96
    )
    x104 = parms[12] * x0 + parms[14] * x6 + x101 * x2 + x18 * x95 + 0.316 * x2 * x74 + x22 * x97
    x105 = x18 * x73
    x106 = -x2
    x107 = x1 * x18
    x108 = x106 * x16 + x107 * x17
    x109 = -x22
    x110 = -x105
    x111 = 0.384 * x107 + x109 - 0.0825 * x110
    x112 = -x106
    x113 = 0.0825 * x106
    x114 = 0.316 * x2
    x115 = x1 * x114 + x113 * x5
    x116 = 0.384 * x112 + x115
    x117 = x111 * x46 + x116 * x16
    x118 = -x117
    x119 = x105 * x32 + x108 * x33
    x120 = x105 * x33 + x108 * x64
    x121 = -x120
    x122 = x106 * x46 + x107 * x16
    x123 = -x122
    x124 = x119 * x42 + x123 * x40
    x125 = x111 * x16 + x116 * x17
    x126 = x1 * x113 + x114 * x73
    x127 = -0.0825 * x106 + x126
    x128 = x125 * x33 + x127 * x32
    x129 = x118 + 0.088 * x121
    x130 = x128 * x52 + x129 * x42
    x131 = parms[66] * x121 - parms[68] * x124 + parms[69] * x130
    x132 = x131 * x42
    x133 = x128 * x42 + x129 * x40
    x134 = x119 * x52 + x123 * x42
    x135 = -parms[67] * x121 + parms[68] * x134 + parms[69] * x133
    x136 = x135 * x40
    x137 = (
        parms[46] * x105
        - parms[48] * x108
        + parms[49] * x117
        - parms[56] * x121
        - parms[57] * x119
        - parms[59] * x118
        - x132
        - x136
    )
    x138 = -parms[57] * x123 + parms[58] * x120 + parms[59] * x128 + x131 * x52 + x135 * x42
    x139 = x125 * x64 + x127 * x33
    x140 = -x139
    x141 = -0.088 * x123 + x140
    x142 = -parms[66] * x134 + parms[67] * x124 + parms[69] * x141
    x143 = parms[56] * x123 - parms[58] * x119 + parms[59] * x139 - x142
    x144 = parms[47] * x110 + parms[48] * x122 + parms[49] * x125 + x138 * x33 + x143 * x64
    x145 = x144 * x17
    x146 = parms[37] * x112 + parms[38] * x105 + parms[39] * x115 + x137 * x16 + x145
    x147 = parms[46] * x123 + parms[47] * x108 + parms[49] * x127 + x138 * x32 + x143 * x33
    x148 = parms[36] * x106 - parms[38] * x107 + parms[39] * x126 + x147
    x149 = parms[62] * x124 + parms[64] * x134 + parms[65] * x121 + parms[66] * x130 + x133 * x83
    x150 = (
        parms[51] * x119
        + parms[53] * x120
        + parms[54] * x123
        + parms[58] * x128
        + x118 * x85
        - 0.088 * x132
        - 0.088 * x136
        - x149
    )
    x151 = parms[61] * x124 + parms[63] * x134 + parms[64] * x121 + parms[68] * x133 + x141 * x80
    x152 = parms[60] * x124 + parms[61] * x134 + parms[62] * x121 + parms[67] * x141 + x130 * x78
    x153 = (
        parms[50] * x119
        + parms[51] * x120
        + parms[52] * x123
        + parms[57] * x118
        + parms[58] * x140
        + x151 * x52
        + x152 * x42
    )
    x154 = (
        parms[40] * x108
        + parms[41] * x122
        + parms[42] * x105
        + parms[47] * x127
        + parms[48] * x118
        + x150 * x64
        + x153 * x33
    )
    x155 = -parms[57]
    x156 = (
        parms[52] * x119
        + parms[54] * x120
        + parms[55] * x123
        + parms[56] * x139
        + x128 * x155
        - 0.088 * x142
        + x151 * x42
        + x152 * x40
    )
    x157 = parms[41] * x108 + parms[43] * x122 + parms[44] * x105 + parms[48] * x125 + x127 * x92 - x156
    x158 = (
        parms[32] * x107
        + parms[34] * x105
        + parms[35] * x106
        + parms[36] * x126
        + x115 * x98
        + x137 * x99
        - 0.384 * x145
        - 0.0825 * x147
        + x154 * x16
        + x157 * x46
    )
    x159 = -parms[47]
    x160 = (
        parms[42] * x108
        + parms[44] * x122
        + parms[45] * x105
        + parms[46] * x117
        + x125 * x159
        + x150 * x33
        + x153 * x32
    )
    x161 = x137 * x17
    x162 = x144 * x15
    x163 = (
        parms[31] * x107
        + parms[33] * x105
        + parms[34] * x106
        + parms[38] * x115
        + x109 * x76
        + x160
        - 0.0825 * x161
        - 0.0825 * x162
    )
    x164 = (
        parms[30] * x107
        + parms[31] * x105
        + parms[32] * x106
        + parms[37] * x109
        + x126 * x89
        + x154 * x17
        + x157 * x16
        - 0.384 * x161
        - 0.384 * x162
    )
    x165 = parms[36] * x110 + parms[37] * x107 + parms[39] * x109 + x137 * x46 + x144 * x16
    x166 = parms[22] * x18 + parms[24] * x2 + parms[26] * x22 - parms[27] * x114 + x1 * x163 + x164 * x5 - 0.0825 * x165
    x167 = x17 * x5
    x168 = x1 * x33 + x167 * x64
    x169 = x16 * x5
    x170 = -x169
    x171 = -x1
    x172 = -0.0825 * x171 + 0.384 * x5 - 0.0825
    x173 = x16 * x172
    x174 = x173 * x33
    x175 = -x168
    x176 = x172 * x46
    x177 = -x176
    x178 = 0.088 * x175 + x177
    x179 = x174 * x42 + x178 * x40
    x180 = x1 * x32 + x167 * x33
    x181 = x170 * x42 + x180 * x52
    x182 = -parms[67] * x175 + parms[68] * x181 + parms[69] * x179
    x183 = x170 * x40 + x180 * x42
    x184 = x174 * x52 + x178 * x42
    x185 = parms[66] * x175 - parms[68] * x183 + parms[69] * x184
    x186 = -parms[57] * x170 + parms[58] * x168 + parms[59] * x174 + x182 * x42 + x185 * x52
    x187 = x173 * x64
    x188 = -x187
    x189 = -0.088 * x170 + x188
    x190 = -parms[66] * x181 + parms[67] * x183 + parms[69] * x189
    x191 = parms[56] * x170 - parms[58] * x180 + parms[59] * x187 - x190
    x192 = parms[47] * x171 + parms[48] * x169 + parms[49] * x173 + x186 * x33 + x191 * x64
    x193 = x182 * x40
    x194 = x185 * x42
    x195 = (
        parms[46] * x1
        - parms[48] * x167
        + parms[49] * x176
        - parms[56] * x175
        - parms[57] * x180
        - parms[59] * x177
        - x193
        - x194
    )
    x196 = x17 * x195
    x197 = x15 * x192
    x198 = parms[61] * x183 + parms[63] * x181 + parms[64] * x175 + parms[68] * x179 + x189 * x80
    x199 = parms[60] * x183 + parms[61] * x181 + parms[62] * x175 + parms[67] * x189 + x184 * x78
    x200 = (
        parms[52] * x180
        + parms[54] * x168
        + parms[55] * x170
        + parms[56] * x187
        + x155 * x174
        - 0.088 * x190
        + x198 * x42
        + x199 * x40
    )
    x201 = parms[41] * x167 + parms[43] * x169 + parms[44] * x1 + parms[48] * x173 - x200
    x202 = (
        parms[50] * x180
        + parms[51] * x168
        + parms[52] * x170
        + parms[57] * x177
        + parms[58] * x188
        + x198 * x52
        + x199 * x42
    )
    x203 = parms[62] * x183 + parms[64] * x181 + parms[65] * x175 + parms[66] * x184 + x179 * x83
    x204 = (
        parms[51] * x180
        + parms[53] * x168
        + parms[54] * x170
        + parms[58] * x174
        + x177 * x85
        - 0.088 * x193
        - 0.088 * x194
        - x203
    )
    x205 = parms[40] * x167 + parms[41] * x169 + parms[42] * x1 + parms[48] * x177 + x202 * x33 + x204 * x64
    x206 = (
        parms[42] * x167 + parms[44] * x169 + parms[45] * x1 + parms[46] * x176 + x159 * x173 + x202 * x32 + x204 * x33
    )
    x207 = (
        parms[32] * x5
        + parms[34] * x1
        - 0.0825 * parms[46] * x170
        - 0.0825 * parms[47] * x167
        + x16 * x205
        - 0.384 * x17 * x192
        - 0.0825 * x186 * x32
        - 0.0825 * x191 * x33
        + x195 * x99
        + x201 * x46
    )
    x208 = -0.384 * x17
    x209 = x208 * x33 - 0.0825 * x32
    x210 = x16 * x64
    x211 = -x210
    x212 = -x99
    x213 = 0.088 * x211 + x212
    x214 = x209 * x52 + x213 * x42
    x215 = x16 * x33
    x216 = -x46
    x217 = x215 * x42 + x216 * x40
    x218 = parms[66] * x211 - parms[68] * x217 + parms[69] * x214
    x219 = x218 * x42
    x220 = x215 * x52 + x216 * x42
    x221 = x209 * x42 + x213 * x40
    x222 = parms[62] * x217 + parms[64] * x220 + parms[65] * x211 + parms[66] * x214 + x221 * x83
    x223 = -parms[67] * x211 + parms[68] * x220 + parms[69] * x221
    x224 = x223 * x40
    x225 = (
        parms[51] * x215
        + parms[53] * x210
        + parms[54] * x216
        + parms[58] * x209
        + x212 * x85
        - 0.088 * x219
        - x222
        - 0.088 * x224
    )
    x226 = x208 * x64 - 0.0825 * x33
    x227 = -x226
    x228 = -0.088 * x216 + x227
    x229 = parms[60] * x217 + parms[61] * x220 + parms[62] * x211 + parms[67] * x228 + x214 * x78
    x230 = parms[61] * x217 + parms[63] * x220 + parms[64] * x211 + parms[68] * x221 + x228 * x80
    x231 = (
        parms[50] * x215
        + parms[51] * x210
        + parms[52] * x216
        + parms[57] * x212
        + parms[58] * x227
        + x229 * x42
        + x230 * x52
    )
    x232 = -parms[66] * x220 + parms[67] * x217 + parms[69] * x228
    x233 = (
        parms[52] * x215
        + parms[54] * x210
        + parms[55] * x216
        + parms[56] * x226
        + x155 * x209
        + x229 * x40
        + x230 * x42
        - 0.088 * x232
    )
    x234 = -parms[57] * x216 + parms[58] * x210 + parms[59] * x209 + x218 * x52 + x223 * x42
    x235 = parms[56] * x216 - parms[58] * x215 + parms[59] * x226 - x232
    x236 = parms[42] * x16 + parms[44] * x46 + parms[46] * x99 + x159 * x208 + x225 * x33 + x231 * x32
    x237 = x32 * x42
    x238 = x32 * x52
    x239 = -x33
    x240 = 0.088 * x239
    x241 = x240 * x42
    x242 = parms[60] * x237 + parms[61] * x238 + parms[62] * x239 + x241 * x78
    x243 = x240 * x40
    x244 = parms[61] * x237 + parms[63] * x238 + parms[64] * x239 + parms[68] * x243
    x245 = parms[62] * x237 + parms[64] * x238 + parms[65] * x239 + parms[66] * x241 + x243 * x83
    x246 = (
        parms[52] * x32
        + parms[54] * x33
        + 0.088 * parms[66] * x238
        - 0.088 * parms[67] * x237
        + x242 * x40
        + x244 * x42
    )
    x247 = parms[62] * x40 + parms[64] * x42
    #
    M_num[0] = (
        parms[5]
        + x0 * (parms[10] * x0 + parms[11] * x6 + x101 * x102 - 0.316 * x18 * x74 + x2 * x95 - 0.316 * x2 * x97)
        + x6 * (parms[11] * x0 + parms[13] * x6 - x103)
    )
    M_num[1] = x104
    M_num[2] = x103
    M_num[3] = x100
    M_num[4] = x87
    M_num[5] = x91
    M_num[6] = x84
    M_num[7] = x104
    M_num[8] = (
        parms[15]
        + x114 * (parms[28] * x2 + parms[29] * x114 + x1 * x146 + x148 * x73)
        + x18 * (parms[20] * x18 + parms[21] * x2 + parms[28] * x109 + x1 * x164 + x163 * x73)
        + x2 * (parms[21] * x18 + parms[23] * x2 + parms[28] * x114 - 0.0825 * x1 * x148 - 0.0825 * x146 * x5 - x158)
        + x22 * (parms[28] * x102 + parms[29] * x22 - x165)
    )
    M_num[9] = x166
    M_num[10] = x158
    M_num[11] = x160
    M_num[12] = x156
    M_num[13] = x149
    M_num[14] = x103
    M_num[15] = x166
    M_num[16] = (
        parms[25]
        - 0.0825 * parms[36] * x171
        - 0.0825 * parms[37] * x5
        + 0.00680625 * parms[39]
        + x1 * (parms[31] * x5 + parms[33] * x1 + 0.0825 * parms[36] - 0.0825 * x196 - 0.0825 * x197 + x206)
        - 0.0825 * x16 * x192
        - 0.0825 * x195 * x46
        + x5
        * (parms[30] * x5 + parms[31] * x1 - 0.0825 * parms[37] + x16 * x201 + x17 * x205 - 0.384 * x196 - 0.384 * x197)
    )
    M_num[17] = x207
    M_num[18] = x206
    M_num[19] = x200
    M_num[20] = x203
    M_num[21] = x100
    M_num[22] = x158
    M_num[23] = x207
    M_num[24] = (
        parms[35]
        - 0.0825 * parms[46] * x216
        - 0.0825 * parms[47] * x16
        + 0.00680625 * parms[49]
        + x16 * (parms[40] * x16 + parms[41] * x46 - 0.0825 * parms[47] + parms[48] * x212 + x225 * x64 + x231 * x33)
        + x208 * (parms[48] * x46 + parms[49] * x208 + x234 * x33 + x235 * x64)
        - 0.0825 * x234 * x32
        - 0.0825 * x235 * x33
        + x46 * (parms[41] * x16 + parms[43] * x46 + 0.0825 * parms[46] + parms[48] * x208 - x233)
        + x99
        * (-parms[48] * x16 + parms[49] * x99 - parms[56] * x211 - parms[57] * x215 - parms[59] * x212 - x219 - x224)
    )
    M_num[25] = x236
    M_num[26] = x233
    M_num[27] = x222
    M_num[28] = x87
    M_num[29] = x160
    M_num[30] = x206
    M_num[31] = x236
    M_num[32] = (
        parms[45]
        + x32 * (parms[50] * x32 + parms[51] * x33 + x242 * x42 + x244 * x52)
        + x33
        * (
            parms[51] * x32
            + parms[53] * x33
            - x245
            - 0.088 * x40 * (-parms[67] * x239 + parms[68] * x238 + parms[69] * x243)
            - 0.088 * x42 * (parms[66] * x239 - parms[68] * x237 + parms[69] * x241)
        )
    )
    M_num[33] = x246
    M_num[34] = x245
    M_num[35] = x91
    M_num[36] = x156
    M_num[37] = x200
    M_num[38] = x233
    M_num[39] = x246
    M_num[40] = (
        parms[55]
        + 0.088 * parms[66] * x42
        - 0.088 * parms[67] * x40
        + 0.007744 * parms[69]
        + x40 * (parms[60] * x40 + parms[61] * x42 - 0.088 * parms[67])
        + x42 * (parms[61] * x40 + parms[63] * x42 + 0.088 * parms[66])
    )
    M_num[41] = x247
    M_num[42] = x84
    M_num[43] = x149
    M_num[44] = x203
    M_num[45] = x222
    M_num[46] = x245
    M_num[47] = x247
    M_num[48] = parms[65]
    #
    return M_num
