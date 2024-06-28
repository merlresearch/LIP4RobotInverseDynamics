import torch


def get_K_blocks_GIP_sum_PANDA_7dof_no_subs(
    X1, X2, pos_indices, vel_indices, acc_indices, sigma_kin_vel_list, sigma_kin_pos_list, sigma_pot_list
):

    q_i1 = X1[:, pos_indices[0] : pos_indices[0] + 1]
    dq_i1 = X1[:, vel_indices[0] : vel_indices[0] + 1]
    ddq_i1 = X1[:, acc_indices[0] : acc_indices[0] + 1]
    q_j1 = X2[:, pos_indices[0] : pos_indices[0] + 1].transpose(1, 0)
    dq_j1 = X2[:, vel_indices[0] : vel_indices[0] + 1].transpose(1, 0)
    ddq_j1 = X2[:, acc_indices[0] : acc_indices[0] + 1].transpose(1, 0)
    q_i2 = X1[:, pos_indices[1] : pos_indices[1] + 1]
    dq_i2 = X1[:, vel_indices[1] : vel_indices[1] + 1]
    ddq_i2 = X1[:, acc_indices[1] : acc_indices[1] + 1]
    q_j2 = X2[:, pos_indices[1] : pos_indices[1] + 1].transpose(1, 0)
    dq_j2 = X2[:, vel_indices[1] : vel_indices[1] + 1].transpose(1, 0)
    ddq_j2 = X2[:, acc_indices[1] : acc_indices[1] + 1].transpose(1, 0)
    q_i3 = X1[:, pos_indices[2] : pos_indices[2] + 1]
    dq_i3 = X1[:, vel_indices[2] : vel_indices[2] + 1]
    ddq_i3 = X1[:, acc_indices[2] : acc_indices[2] + 1]
    q_j3 = X2[:, pos_indices[2] : pos_indices[2] + 1].transpose(1, 0)
    dq_j3 = X2[:, vel_indices[2] : vel_indices[2] + 1].transpose(1, 0)
    ddq_j3 = X2[:, acc_indices[2] : acc_indices[2] + 1].transpose(1, 0)
    q_i4 = X1[:, pos_indices[3] : pos_indices[3] + 1]
    dq_i4 = X1[:, vel_indices[3] : vel_indices[3] + 1]
    ddq_i4 = X1[:, acc_indices[3] : acc_indices[3] + 1]
    q_j4 = X2[:, pos_indices[3] : pos_indices[3] + 1].transpose(1, 0)
    dq_j4 = X2[:, vel_indices[3] : vel_indices[3] + 1].transpose(1, 0)
    ddq_j4 = X2[:, acc_indices[3] : acc_indices[3] + 1].transpose(1, 0)
    q_i5 = X1[:, pos_indices[4] : pos_indices[4] + 1]
    dq_i5 = X1[:, vel_indices[4] : vel_indices[4] + 1]
    ddq_i5 = X1[:, acc_indices[4] : acc_indices[4] + 1]
    q_j5 = X2[:, pos_indices[4] : pos_indices[4] + 1].transpose(1, 0)
    dq_j5 = X2[:, vel_indices[4] : vel_indices[4] + 1].transpose(1, 0)
    ddq_j5 = X2[:, acc_indices[4] : acc_indices[4] + 1].transpose(1, 0)
    q_i6 = X1[:, pos_indices[5] : pos_indices[5] + 1]
    dq_i6 = X1[:, vel_indices[5] : vel_indices[5] + 1]
    ddq_i6 = X1[:, acc_indices[5] : acc_indices[5] + 1]
    q_j6 = X2[:, pos_indices[5] : pos_indices[5] + 1].transpose(1, 0)
    dq_j6 = X2[:, vel_indices[5] : vel_indices[5] + 1].transpose(1, 0)
    ddq_j6 = X2[:, acc_indices[5] : acc_indices[5] + 1].transpose(1, 0)
    q_i7 = X1[:, pos_indices[6] : pos_indices[6] + 1]
    dq_i7 = X1[:, vel_indices[6] : vel_indices[6] + 1]
    ddq_i7 = X1[:, acc_indices[6] : acc_indices[6] + 1]
    q_j7 = X2[:, pos_indices[6] : pos_indices[6] + 1].transpose(1, 0)
    dq_j7 = X2[:, vel_indices[6] : vel_indices[6] + 1].transpose(1, 0)
    ddq_j7 = X2[:, acc_indices[6] : acc_indices[6] + 1].transpose(1, 0)

    sigma_kin_v_1_1 = sigma_kin_vel_list[0][0]
    sigma_kin_p_1_1_s = sigma_kin_pos_list[0][0]
    sigma_kin_p_1_1_c = sigma_kin_pos_list[0][1]
    sigma_kin_p_1_1_off = sigma_kin_pos_list[0][2]
    sigma_pot_1_s = sigma_pot_list[0][0]
    sigma_pot_1_c = sigma_pot_list[0][1]
    sigma_pot_1_off = sigma_pot_list[0][2]
    sigma_kin_v_2_1 = sigma_kin_vel_list[1][0]
    sigma_kin_p_2_1_s = sigma_kin_pos_list[1][0]
    sigma_kin_p_2_1_c = sigma_kin_pos_list[1][1]
    sigma_kin_p_2_1_off = sigma_kin_pos_list[1][2]
    sigma_kin_v_2_2 = sigma_kin_vel_list[1][1]
    sigma_kin_p_2_2_s = sigma_kin_pos_list[2][0]
    sigma_kin_p_2_2_c = sigma_kin_pos_list[2][1]
    sigma_kin_p_2_2_off = sigma_kin_pos_list[2][2]
    sigma_pot_2_s = sigma_pot_list[1][0]
    sigma_pot_2_c = sigma_pot_list[1][1]
    sigma_pot_2_off = sigma_pot_list[1][2]
    sigma_kin_v_3_1 = sigma_kin_vel_list[2][0]
    sigma_kin_p_3_1_s = sigma_kin_pos_list[3][0]
    sigma_kin_p_3_1_c = sigma_kin_pos_list[3][1]
    sigma_kin_p_3_1_off = sigma_kin_pos_list[3][2]
    sigma_kin_v_3_2 = sigma_kin_vel_list[2][1]
    sigma_kin_p_3_2_s = sigma_kin_pos_list[4][0]
    sigma_kin_p_3_2_c = sigma_kin_pos_list[4][1]
    sigma_kin_p_3_2_off = sigma_kin_pos_list[4][2]
    sigma_kin_v_3_3 = sigma_kin_vel_list[2][2]
    sigma_kin_p_3_3_s = sigma_kin_pos_list[5][0]
    sigma_kin_p_3_3_c = sigma_kin_pos_list[5][1]
    sigma_kin_p_3_3_off = sigma_kin_pos_list[5][2]
    sigma_pot_3_s = sigma_pot_list[2][0]
    sigma_pot_3_c = sigma_pot_list[2][1]
    sigma_pot_3_off = sigma_pot_list[2][2]
    sigma_kin_v_4_1 = sigma_kin_vel_list[3][0]
    sigma_kin_p_4_1_s = sigma_kin_pos_list[6][0]
    sigma_kin_p_4_1_c = sigma_kin_pos_list[6][1]
    sigma_kin_p_4_1_off = sigma_kin_pos_list[6][2]
    sigma_kin_v_4_2 = sigma_kin_vel_list[3][1]
    sigma_kin_p_4_2_s = sigma_kin_pos_list[7][0]
    sigma_kin_p_4_2_c = sigma_kin_pos_list[7][1]
    sigma_kin_p_4_2_off = sigma_kin_pos_list[7][2]
    sigma_kin_v_4_3 = sigma_kin_vel_list[3][2]
    sigma_kin_p_4_3_s = sigma_kin_pos_list[8][0]
    sigma_kin_p_4_3_c = sigma_kin_pos_list[8][1]
    sigma_kin_p_4_3_off = sigma_kin_pos_list[8][2]
    sigma_kin_v_4_4 = sigma_kin_vel_list[3][3]
    sigma_kin_p_4_4_s = sigma_kin_pos_list[9][0]
    sigma_kin_p_4_4_c = sigma_kin_pos_list[9][1]
    sigma_kin_p_4_4_off = sigma_kin_pos_list[9][2]
    sigma_pot_4_s = sigma_pot_list[3][0]
    sigma_pot_4_c = sigma_pot_list[3][1]
    sigma_pot_4_off = sigma_pot_list[3][2]
    sigma_kin_v_5_1 = sigma_kin_vel_list[4][0]
    sigma_kin_p_5_1_s = sigma_kin_pos_list[10][0]
    sigma_kin_p_5_1_c = sigma_kin_pos_list[10][1]
    sigma_kin_p_5_1_off = sigma_kin_pos_list[10][2]
    sigma_kin_v_5_2 = sigma_kin_vel_list[4][1]
    sigma_kin_p_5_2_s = sigma_kin_pos_list[11][0]
    sigma_kin_p_5_2_c = sigma_kin_pos_list[11][1]
    sigma_kin_p_5_2_off = sigma_kin_pos_list[11][2]
    sigma_kin_v_5_3 = sigma_kin_vel_list[4][2]
    sigma_kin_p_5_3_s = sigma_kin_pos_list[12][0]
    sigma_kin_p_5_3_c = sigma_kin_pos_list[12][1]
    sigma_kin_p_5_3_off = sigma_kin_pos_list[12][2]
    sigma_kin_v_5_4 = sigma_kin_vel_list[4][3]
    sigma_kin_p_5_4_s = sigma_kin_pos_list[13][0]
    sigma_kin_p_5_4_c = sigma_kin_pos_list[13][1]
    sigma_kin_p_5_4_off = sigma_kin_pos_list[13][2]
    sigma_kin_v_5_5 = sigma_kin_vel_list[4][4]
    sigma_kin_p_5_5_s = sigma_kin_pos_list[14][0]
    sigma_kin_p_5_5_c = sigma_kin_pos_list[14][1]
    sigma_kin_p_5_5_off = sigma_kin_pos_list[14][2]
    sigma_pot_5_s = sigma_pot_list[4][0]
    sigma_pot_5_c = sigma_pot_list[4][1]
    sigma_pot_5_off = sigma_pot_list[4][2]
    sigma_kin_v_6_1 = sigma_kin_vel_list[5][0]
    sigma_kin_p_6_1_s = sigma_kin_pos_list[15][0]
    sigma_kin_p_6_1_c = sigma_kin_pos_list[15][1]
    sigma_kin_p_6_1_off = sigma_kin_pos_list[15][2]
    sigma_kin_v_6_2 = sigma_kin_vel_list[5][1]
    sigma_kin_p_6_2_s = sigma_kin_pos_list[16][0]
    sigma_kin_p_6_2_c = sigma_kin_pos_list[16][1]
    sigma_kin_p_6_2_off = sigma_kin_pos_list[16][2]
    sigma_kin_v_6_3 = sigma_kin_vel_list[5][2]
    sigma_kin_p_6_3_s = sigma_kin_pos_list[17][0]
    sigma_kin_p_6_3_c = sigma_kin_pos_list[17][1]
    sigma_kin_p_6_3_off = sigma_kin_pos_list[17][2]
    sigma_kin_v_6_4 = sigma_kin_vel_list[5][3]
    sigma_kin_p_6_4_s = sigma_kin_pos_list[18][0]
    sigma_kin_p_6_4_c = sigma_kin_pos_list[18][1]
    sigma_kin_p_6_4_off = sigma_kin_pos_list[18][2]
    sigma_kin_v_6_5 = sigma_kin_vel_list[5][4]
    sigma_kin_p_6_5_s = sigma_kin_pos_list[19][0]
    sigma_kin_p_6_5_c = sigma_kin_pos_list[19][1]
    sigma_kin_p_6_5_off = sigma_kin_pos_list[19][2]
    sigma_kin_v_6_6 = sigma_kin_vel_list[5][5]
    sigma_kin_p_6_6_s = sigma_kin_pos_list[20][0]
    sigma_kin_p_6_6_c = sigma_kin_pos_list[20][1]
    sigma_kin_p_6_6_off = sigma_kin_pos_list[20][2]
    sigma_pot_6_s = sigma_pot_list[5][0]
    sigma_pot_6_c = sigma_pot_list[5][1]
    sigma_pot_6_off = sigma_pot_list[5][2]
    sigma_kin_v_7_1 = sigma_kin_vel_list[6][0]
    sigma_kin_p_7_1_s = sigma_kin_pos_list[21][0]
    sigma_kin_p_7_1_c = sigma_kin_pos_list[21][1]
    sigma_kin_p_7_1_off = sigma_kin_pos_list[21][2]
    sigma_kin_v_7_2 = sigma_kin_vel_list[6][1]
    sigma_kin_p_7_2_s = sigma_kin_pos_list[22][0]
    sigma_kin_p_7_2_c = sigma_kin_pos_list[22][1]
    sigma_kin_p_7_2_off = sigma_kin_pos_list[22][2]
    sigma_kin_v_7_3 = sigma_kin_vel_list[6][2]
    sigma_kin_p_7_3_s = sigma_kin_pos_list[23][0]
    sigma_kin_p_7_3_c = sigma_kin_pos_list[23][1]
    sigma_kin_p_7_3_off = sigma_kin_pos_list[23][2]
    sigma_kin_v_7_4 = sigma_kin_vel_list[6][3]
    sigma_kin_p_7_4_s = sigma_kin_pos_list[24][0]
    sigma_kin_p_7_4_c = sigma_kin_pos_list[24][1]
    sigma_kin_p_7_4_off = sigma_kin_pos_list[24][2]
    sigma_kin_v_7_5 = sigma_kin_vel_list[6][4]
    sigma_kin_p_7_5_s = sigma_kin_pos_list[25][0]
    sigma_kin_p_7_5_c = sigma_kin_pos_list[25][1]
    sigma_kin_p_7_5_off = sigma_kin_pos_list[25][2]
    sigma_kin_v_7_6 = sigma_kin_vel_list[6][5]
    sigma_kin_p_7_6_s = sigma_kin_pos_list[26][0]
    sigma_kin_p_7_6_c = sigma_kin_pos_list[26][1]
    sigma_kin_p_7_6_off = sigma_kin_pos_list[26][2]
    sigma_kin_v_7_7 = sigma_kin_vel_list[6][6]
    sigma_kin_p_7_7_s = sigma_kin_pos_list[27][0]
    sigma_kin_p_7_7_c = sigma_kin_pos_list[27][1]
    sigma_kin_p_7_7_off = sigma_kin_pos_list[27][2]
    sigma_pot_7_s = sigma_pot_list[6][0]
    sigma_pot_7_c = sigma_pot_list[6][1]
    sigma_pot_7_off = sigma_pot_list[6][2]

    x0 = torch.sin(q_j1)
    x1 = torch.sin(q_i1)
    x2 = x0 * x1
    x3 = torch.cos(q_i1)
    x4 = torch.cos(q_j1)
    x5 = x3 * x4
    x6 = sigma_kin_v_1_1**2
    x7 = sigma_kin_p_1_1_c * x5 + sigma_kin_p_1_1_off + sigma_kin_p_1_1_s * x2
    x8 = x6 * x7
    x9 = x8 * (sigma_kin_p_1_1_c * x2 + sigma_kin_p_1_1_s * x5)
    x10 = dq_j1**2
    x11 = dq_i1**2
    x12 = 2 * x11
    x13 = x10 * x12
    x14 = x0 * x3
    x15 = x1 * x4
    x16 = sigma_kin_p_1_1_c * x14 - sigma_kin_p_1_1_s * x15
    x17 = sigma_kin_p_1_1_c * x15 - sigma_kin_p_1_1_s * x14
    x18 = x16 * x17 * x6
    x19 = sigma_kin_p_2_1_c * x2 + sigma_kin_p_2_1_s * x5
    x20 = torch.cos(q_i2)
    x21 = torch.cos(q_j2)
    x22 = x20 * x21
    x23 = torch.sin(q_i2)
    x24 = torch.sin(q_j2)
    x25 = x23 * x24
    x26 = sigma_kin_p_2_2_c * x22 + sigma_kin_p_2_2_off + sigma_kin_p_2_2_s * x25
    x27 = x26**2
    x28 = sigma_kin_p_2_1_c * x5 + sigma_kin_p_2_1_off + sigma_kin_p_2_1_s * x2
    x29 = x27 * x28
    x30 = x19 * x29
    x31 = dq_i1 * dq_j1
    x32 = dq_i2 * dq_j2
    x33 = sigma_kin_v_2_1 * x31 + sigma_kin_v_2_2 * x32
    x34 = x33**2
    x35 = 2 * x34
    x36 = sigma_kin_p_2_1_c * x14 - sigma_kin_p_2_1_s * x15
    x37 = sigma_kin_p_2_1_c * x15 - sigma_kin_p_2_1_s * x14
    x38 = x27 * x36 * x37
    x39 = sigma_kin_p_3_1_c * x2 + sigma_kin_p_3_1_s * x5
    x40 = sigma_kin_p_3_1_c * x5 + sigma_kin_p_3_1_off + sigma_kin_p_3_1_s * x2
    x41 = sigma_kin_p_3_2_c * x22 + sigma_kin_p_3_2_off + sigma_kin_p_3_2_s * x25
    x42 = x41**2
    x43 = torch.cos(q_i3)
    x44 = torch.cos(q_j3)
    x45 = x43 * x44
    x46 = torch.sin(q_i3)
    x47 = torch.sin(q_j3)
    x48 = x46 * x47
    x49 = sigma_kin_p_3_3_c * x45 + sigma_kin_p_3_3_off + sigma_kin_p_3_3_s * x48
    x50 = x49**2
    x51 = x42 * x50
    x52 = x40 * x51
    x53 = x39 * x52
    x54 = dq_i3 * dq_j3
    x55 = sigma_kin_v_3_1 * x31 + sigma_kin_v_3_2 * x32 + sigma_kin_v_3_3 * x54
    x56 = x55**2
    x57 = 2 * x56
    x58 = sigma_kin_p_3_1_c * x14 - sigma_kin_p_3_1_s * x15
    x59 = sigma_kin_p_3_1_c * x15 - sigma_kin_p_3_1_s * x14
    x60 = x51 * x58 * x59
    x61 = dq_i4 * dq_j4
    x62 = sigma_kin_v_4_1 * x31 + sigma_kin_v_4_2 * x32 + sigma_kin_v_4_3 * x54 + sigma_kin_v_4_4 * x61
    x63 = x62**2
    x64 = 2 * x63
    x65 = sigma_kin_p_4_1_c * x2 + sigma_kin_p_4_1_s * x5
    x66 = sigma_kin_p_4_3_c * x45 + sigma_kin_p_4_3_off + sigma_kin_p_4_3_s * x48
    x67 = x66**2
    x68 = torch.cos(q_i4)
    x69 = torch.cos(q_j4)
    x70 = x68 * x69
    x71 = torch.sin(q_i4)
    x72 = torch.sin(q_j4)
    x73 = x71 * x72
    x74 = sigma_kin_p_4_4_c * x70 + sigma_kin_p_4_4_off + sigma_kin_p_4_4_s * x73
    x75 = x74**2
    x76 = x67 * x75
    x77 = sigma_kin_p_4_2_c * x22 + sigma_kin_p_4_2_off + sigma_kin_p_4_2_s * x25
    x78 = x77**2
    x79 = sigma_kin_p_4_1_c * x5 + sigma_kin_p_4_1_off + sigma_kin_p_4_1_s * x2
    x80 = x78 * x79
    x81 = x76 * x80
    x82 = x65 * x81
    x83 = sigma_kin_p_4_1_c * x14 - sigma_kin_p_4_1_s * x15
    x84 = x76 * x83
    x85 = sigma_kin_p_4_1_c * x15 - sigma_kin_p_4_1_s * x14
    x86 = x78 * x85
    x87 = x84 * x86
    x88 = sigma_pot_4_c * x70 + sigma_pot_4_off + sigma_pot_4_s * x73
    x89 = torch.cos(q_i5)
    x90 = torch.cos(q_j5)
    x91 = x89 * x90
    x92 = torch.sin(q_i5)
    x93 = torch.sin(q_j5)
    x94 = x92 * x93
    x95 = sigma_pot_5_c * x91 + sigma_pot_5_off + sigma_pot_5_s * x94
    x96 = torch.cos(q_i6)
    x97 = torch.cos(q_j6)
    x98 = x96 * x97
    x99 = torch.sin(q_i6)
    x100 = torch.sin(q_j6)
    x101 = x100 * x99
    x102 = sigma_pot_6_c * x98 + sigma_pot_6_off + sigma_pot_6_s * x101
    x103 = torch.cos(q_i7)
    x104 = torch.cos(q_j7)
    x105 = x103 * x104
    x106 = torch.sin(q_i7)
    x107 = torch.sin(q_j7)
    x108 = x106 * x107
    x109 = sigma_pot_7_c * x105 + sigma_pot_7_off + sigma_pot_7_s * x108
    x110 = 1.0 * x102 * x109 * x88 * x95
    x111 = sigma_pot_2_c * x22 + sigma_pot_2_off + sigma_pot_2_s * x25
    x112 = sigma_pot_3_c * x45 + sigma_pot_3_off + sigma_pot_3_s * x48
    x113 = x111 * x112
    x114 = dq_i5 * dq_j5
    x115 = (
        sigma_kin_v_5_1 * x31
        + sigma_kin_v_5_2 * x32
        + sigma_kin_v_5_3 * x54
        + sigma_kin_v_5_4 * x61
        + sigma_kin_v_5_5 * x114
    )
    x116 = x115**2
    x117 = 2 * x116
    x118 = sigma_kin_p_5_3_c * x45 + sigma_kin_p_5_3_off + sigma_kin_p_5_3_s * x48
    x119 = x118**2
    x120 = sigma_kin_p_5_4_c * x70 + sigma_kin_p_5_4_off + sigma_kin_p_5_4_s * x73
    x121 = x120**2
    x122 = sigma_kin_p_5_5_c * x91 + sigma_kin_p_5_5_off + sigma_kin_p_5_5_s * x94
    x123 = x122**2
    x124 = x119 * x121 * x123
    x125 = sigma_kin_p_5_1_c * x2 + sigma_kin_p_5_1_s * x5
    x126 = sigma_kin_p_5_2_c * x22 + sigma_kin_p_5_2_off + sigma_kin_p_5_2_s * x25
    x127 = x126**2
    x128 = sigma_kin_p_5_1_c * x5 + sigma_kin_p_5_1_off + sigma_kin_p_5_1_s * x2
    x129 = x127 * x128
    x130 = x125 * x129
    x131 = x124 * x130
    x132 = sigma_kin_p_5_1_c * x14 - sigma_kin_p_5_1_s * x15
    x133 = sigma_kin_p_5_1_c * x15 - sigma_kin_p_5_1_s * x14
    x134 = x127 * x132 * x133
    x135 = x124 * x134
    x136 = 4 * dq_j1
    x137 = sigma_kin_p_7_1_c * x14 - sigma_kin_p_7_1_s * x15
    x138 = sigma_kin_p_7_1_c * x5 + sigma_kin_p_7_1_off + sigma_kin_p_7_1_s * x2
    x139 = sigma_kin_p_7_2_c * x22 + sigma_kin_p_7_2_off + sigma_kin_p_7_2_s * x25
    x140 = x139**2
    x141 = x138 * x140
    x142 = x137 * x141
    x143 = sigma_kin_p_7_3_c * x45 + sigma_kin_p_7_3_off + sigma_kin_p_7_3_s * x48
    x144 = x143**2
    x145 = sigma_kin_p_7_4_c * x70 + sigma_kin_p_7_4_off + sigma_kin_p_7_4_s * x73
    x146 = x145**2
    x147 = sigma_kin_p_7_5_c * x91 + sigma_kin_p_7_5_off + sigma_kin_p_7_5_s * x94
    x148 = x147**2
    x149 = sigma_kin_p_7_6_c * x98 + sigma_kin_p_7_6_off + sigma_kin_p_7_6_s * x101
    x150 = x149**2
    x151 = sigma_kin_p_7_7_c * x104
    x152 = sigma_kin_p_7_7_s * x107
    x153 = sigma_kin_p_7_7_off + x103 * x151 + x106 * x152
    x154 = x153**2
    x155 = x144 * x146 * x148 * x150 * x154
    x156 = sigma_kin_v_7_1 * x155
    x157 = x142 * x156
    x158 = dq_j7 * sigma_kin_v_7_7
    x159 = ddq_i7 * x158
    x160 = x157 * x159
    x161 = dq_i6 * dq_j6
    x162 = (
        sigma_kin_v_6_1 * x31
        + sigma_kin_v_6_2 * x32
        + sigma_kin_v_6_3 * x54
        + sigma_kin_v_6_4 * x61
        + sigma_kin_v_6_5 * x114
        + sigma_kin_v_6_6 * x161
    )
    x163 = x162**2
    x164 = 2 * x163
    x165 = sigma_kin_p_6_3_c * x45 + sigma_kin_p_6_3_off + sigma_kin_p_6_3_s * x48
    x166 = x165**2
    x167 = sigma_kin_p_6_4_c * x70 + sigma_kin_p_6_4_off + sigma_kin_p_6_4_s * x73
    x168 = x167**2
    x169 = sigma_kin_p_6_5_c * x91 + sigma_kin_p_6_5_off + sigma_kin_p_6_5_s * x94
    x170 = x169**2
    x171 = sigma_kin_p_6_6_c * x98 + sigma_kin_p_6_6_off + sigma_kin_p_6_6_s * x101
    x172 = x171**2
    x173 = x166 * x168 * x170 * x172
    x174 = sigma_kin_p_6_1_c * x2 + sigma_kin_p_6_1_s * x5
    x175 = sigma_kin_p_6_2_c * x22 + sigma_kin_p_6_2_off + sigma_kin_p_6_2_s * x25
    x176 = x175**2
    x177 = sigma_kin_p_6_1_c * x5 + sigma_kin_p_6_1_off + sigma_kin_p_6_1_s * x2
    x178 = x176 * x177
    x179 = x174 * x178
    x180 = x173 * x179
    x181 = sigma_kin_p_6_1_c * x14 - sigma_kin_p_6_1_s * x15
    x182 = sigma_kin_p_6_1_c * x15 - sigma_kin_p_6_1_s * x14
    x183 = x176 * x181 * x182
    x184 = x173 * x183
    x185 = sigma_kin_p_7_1_c * x2 + sigma_kin_p_7_1_s * x5
    x186 = x141 * x185
    x187 = sigma_kin_v_7_1 * x31
    x188 = sigma_kin_v_7_2 * x32
    x189 = sigma_kin_v_7_3 * x54
    x190 = sigma_kin_v_7_4 * x61
    x191 = sigma_kin_v_7_5 * x114
    x192 = sigma_kin_v_7_6 * x161
    x193 = dq_i7 * x158
    x194 = x188 + x189 + x190 + x191 + x192 + x193
    x195 = x187 + x194
    x196 = x195**2
    x197 = 2 * x196
    x198 = x155 * x197
    x199 = sigma_kin_p_7_1_c * x15 - sigma_kin_p_7_1_s * x14
    x200 = x137 * x140
    x201 = x199 * x200
    x202 = x142 * x146
    x203 = -x103 * x152 + x106 * x151
    x204 = x144 * x148
    x205 = x203 * x204
    x206 = x202 * x205
    x207 = 8 * dq_j1
    x208 = dq_i7 * x153
    x209 = sigma_kin_v_7_1 * x195
    x210 = x150 * x209
    x211 = x208 * x210
    x212 = x207 * x211
    x213 = x178 * x181
    x214 = sigma_kin_v_6_1 * x173
    x215 = x213 * x214
    x216 = sigma_kin_v_6_6 * x215 + sigma_kin_v_7_6 * x157
    x217 = ddq_i6 * dq_j6
    x218 = x216 * x217
    x219 = x97 * x99
    x220 = x100 * x96
    x221 = sigma_kin_p_6_6_c * x219 - sigma_kin_p_6_6_s * x220
    x222 = x170 * x171
    x223 = x221 * x222
    x224 = x168 * x213
    x225 = sigma_kin_v_6_1 * x162
    x226 = x166 * x225
    x227 = x224 * x226
    x228 = x223 * x227
    x229 = sigma_kin_p_7_6_c * x219 - sigma_kin_p_7_6_s * x220
    x230 = x148 * x229
    x231 = x209 * x230
    x232 = x138 * x149
    x233 = x144 * x154
    x234 = x146 * x233
    x235 = x232 * x234
    x236 = x200 * x235
    x237 = x231 * x236
    x238 = x228 + x237
    x239 = dq_i6 * x207
    x240 = x129 * x132
    x241 = sigma_kin_v_5_1 * x124
    x242 = x240 * x241
    x243 = sigma_kin_v_5_5 * x242 + sigma_kin_v_6_5 * x215 + sigma_kin_v_7_5 * x157
    x244 = ddq_i5 * dq_j5
    x245 = x243 * x244
    x246 = sigma_kin_v_4_1 * x76
    x247 = x246 * x80
    x248 = x247 * x83
    x249 = sigma_kin_v_4_4 * x248 + sigma_kin_v_5_4 * x242 + sigma_kin_v_6_4 * x215 + sigma_kin_v_7_4 * x157
    x250 = ddq_i4 * dq_j4
    x251 = x249 * x250
    x252 = x119 * x122
    x253 = x90 * x92
    x254 = x89 * x93
    x255 = sigma_kin_p_5_5_c * x253 - sigma_kin_p_5_5_s * x254
    x256 = sigma_kin_v_5_1 * x115
    x257 = x121 * x256
    x258 = x255 * x257
    x259 = x252 * x258
    x260 = x240 * x259
    x261 = sigma_kin_p_6_5_c * x253 - sigma_kin_p_6_5_s * x254
    x262 = x169 * x172
    x263 = x261 * x262
    x264 = x227 * x263
    x265 = sigma_kin_p_7_5_c * x253 - sigma_kin_p_7_5_s * x254
    x266 = x147 * x265
    x267 = x210 * x266
    x268 = x233 * x267
    x269 = x202 * x268
    x270 = x260 + x264 + x269
    x271 = dq_i5 * x207
    x272 = x52 * x58
    x273 = sigma_kin_v_3_1 * sigma_kin_v_3_3
    x274 = (
        sigma_kin_v_4_3 * x248 + sigma_kin_v_5_3 * x242 + sigma_kin_v_6_3 * x215 + sigma_kin_v_7_3 * x157 + x272 * x273
    )
    x275 = ddq_i3 * dq_j3
    x276 = x274 * x275
    x277 = sigma_kin_v_2_1 * sigma_kin_v_2_2
    x278 = x277 * x29
    x279 = sigma_kin_v_3_1 * sigma_kin_v_3_2
    x280 = (
        sigma_kin_v_4_2 * x248
        + sigma_kin_v_5_2 * x242
        + sigma_kin_v_6_2 * x215
        + sigma_kin_v_7_2 * x157
        + x272 * x279
        + x278 * x36
    )
    x281 = ddq_i2 * dq_j2
    x282 = x280 * x281
    x283 = x62 * x83
    x284 = x283 * x80
    x285 = x69 * x71
    x286 = x68 * x72
    x287 = sigma_kin_p_4_4_c * x285 - sigma_kin_p_4_4_s * x286
    x288 = x67 * x74
    x289 = x287 * x288
    x290 = sigma_kin_v_4_1 * x289
    x291 = x284 * x290
    x292 = sigma_kin_p_5_4_c * x285 - sigma_kin_p_5_4_s * x286
    x293 = x120 * x292
    x294 = x256 * x293
    x295 = x123 * x240
    x296 = x119 * x295
    x297 = x294 * x296
    x298 = x166 * x167
    x299 = sigma_kin_p_6_4_c * x285 - sigma_kin_p_6_4_s * x286
    x300 = x170 * x172
    x301 = x225 * x300
    x302 = x299 * x301
    x303 = x298 * x302
    x304 = x213 * x303
    x305 = x144 * x145
    x306 = sigma_kin_p_7_4_c * x285 - sigma_kin_p_7_4_s * x286
    x307 = x148 * x150 * x154
    x308 = x209 * x307
    x309 = x306 * x308
    x310 = x305 * x309
    x311 = x142 * x310
    x312 = x291 + x297 + x304 + x311
    x313 = dq_i4 * x207
    x314 = sigma_kin_v_2_1**2
    x315 = x29 * x314
    x316 = sigma_kin_v_3_1**2
    x317 = x316 * x52
    x318 = sigma_kin_v_4_1**2
    x319 = x318 * x76
    x320 = x319 * x80
    x321 = sigma_kin_v_5_1**2
    x322 = x124 * x321
    x323 = x129 * x322
    x324 = sigma_kin_v_6_1**2
    x325 = x173 * x324
    x326 = x178 * x325
    x327 = sigma_kin_v_7_1**2
    x328 = x155 * x327
    x329 = x141 * x328
    x330 = x132 * x323 + x137 * x329 + x16 * x8 + x181 * x326 + x315 * x36 + x317 * x58 + x320 * x83
    x331 = 4 * ddq_i1 * x10
    x332 = x44 * x46
    x333 = x43 * x47
    x334 = sigma_kin_p_3_3_c * x332 - sigma_kin_p_3_3_s * x333
    x335 = x42 * x49
    x336 = x334 * x335
    x337 = sigma_kin_v_3_1 * x55
    x338 = x337 * x40
    x339 = x338 * x58
    x340 = x336 * x339
    x341 = sigma_kin_p_4_3_c * x332 - sigma_kin_p_4_3_s * x333
    x342 = x66 * x75
    x343 = x341 * x342
    x344 = sigma_kin_v_4_1 * x343
    x345 = x284 * x344
    x346 = sigma_kin_p_5_3_c * x332 - sigma_kin_p_5_3_s * x333
    x347 = x118 * x346
    x348 = x257 * x347
    x349 = x295 * x348
    x350 = sigma_kin_p_6_3_c * x332 - sigma_kin_p_6_3_s * x333
    x351 = x165 * x350
    x352 = x301 * x351
    x353 = x224 * x352
    x354 = sigma_kin_p_7_3_c * x332 - sigma_kin_p_7_3_s * x333
    x355 = x143 * x354
    x356 = x308 * x355
    x357 = x202 * x356
    x358 = x340 + x345 + x349 + x353 + x357
    x359 = dq_i3 * x207
    x360 = x21 * x23
    x361 = x20 * x24
    x362 = sigma_kin_p_2_2_c * x360 - sigma_kin_p_2_2_s * x361
    x363 = sigma_kin_v_2_1 * x33
    x364 = x362 * x363
    x365 = x26 * x28
    x366 = x36 * x365
    x367 = x364 * x366
    x368 = sigma_kin_p_3_2_c * x360 - sigma_kin_p_3_2_s * x361
    x369 = x41 * x50
    x370 = x368 * x369
    x371 = x339 * x370
    x372 = sigma_kin_p_4_2_c * x360 - sigma_kin_p_4_2_s * x361
    x373 = x246 * x372
    x374 = x77 * x79
    x375 = x373 * x374
    x376 = x283 * x375
    x377 = sigma_kin_p_5_2_c * x360 - sigma_kin_p_5_2_s * x361
    x378 = x115 * x241
    x379 = x377 * x378
    x380 = x126 * x128
    x381 = x132 * x380
    x382 = x379 * x381
    x383 = sigma_kin_p_6_2_c * x360 - sigma_kin_p_6_2_s * x361
    x384 = x162 * x214
    x385 = x383 * x384
    x386 = x175 * x177
    x387 = x181 * x386
    x388 = x385 * x387
    x389 = sigma_kin_p_7_2_c * x360 - sigma_kin_p_7_2_s * x361
    x390 = x156 * x195
    x391 = x389 * x390
    x392 = x138 * x139
    x393 = x137 * x392
    x394 = x391 * x393
    x395 = x367 + x371 + x376 + x382 + x388 + x394
    x396 = dq_i2 * x207
    x397 = x139 * x143
    x398 = x145 * x147
    x399 = x397 * x398
    x400 = x149 * x153
    x401 = ddq_i7 * x400
    x402 = x399 * x401
    x403 = x143 * x398
    x404 = x389 * x403
    x405 = 2 * dq_i2
    x406 = dq_i7 * x400
    x407 = x405 * x406
    x408 = x404 * x407
    x409 = x139 * x398
    x410 = x354 * x409
    x411 = 2 * dq_i3
    x412 = x406 * x411
    x413 = x410 * x412
    x414 = x397 * x406
    x415 = 2 * dq_i4
    x416 = x147 * x306
    x417 = x415 * x416
    x418 = x414 * x417
    x419 = 2 * dq_i5
    x420 = x145 * x265
    x421 = x419 * x420
    x422 = x414 * x421
    x423 = 2 * dq_i6
    x424 = x399 * x423
    x425 = x208 * x229
    x426 = x149 * x399
    x427 = dq_i7**2
    x428 = x203 * x427
    x429 = 2 * x428
    x430 = -x402 + x408 + x413 + x418 + x422 + x424 * x425 + x426 * x429
    x431 = sigma_kin_v_7_7 * x400
    x432 = 2 * ddq_j7
    x433 = x431 * x432
    x434 = x138**2
    x435 = sigma_kin_v_7_1 * x399 * x434
    x436 = x18 * x31
    x437 = x31 * x9
    x438 = x363 * x38
    x439 = x30 * x363
    x440 = x337 * x60
    x441 = x337 * x53
    x442 = x246 * x85
    x443 = x442 * x78
    x444 = x283 * x443
    x445 = x247 * x65
    x446 = x445 * x62
    x447 = x134 * x378
    x448 = x130 * x378
    x449 = x183 * x384
    x450 = x179 * x384
    x451 = x195 * x199
    x452 = x156 * x451
    x453 = x200 * x452
    x454 = x186 * x390
    x455 = x438 + x439 + x440 + x441 + x444 + x446 + x447 + x448 + x449 + x450 + x453 + x454
    x456 = sigma_kin_v_7_1 * sigma_kin_v_7_6
    x457 = x150 * x456
    x458 = x140 * x434
    x459 = x146 * x458
    x460 = x204 * x459
    x461 = x203 * x460
    x462 = x208 * x461
    x463 = x423 * x462
    x464 = x177**2
    x465 = x176 * x464
    x466 = x173 * x465
    x467 = sigma_kin_v_6_1 * x466
    x468 = x155 * x458
    x469 = sigma_kin_v_7_1 * x468
    x470 = sigma_kin_v_6_6 * x467 + sigma_kin_v_7_6 * x469
    x471 = x178 * x182
    x472 = sigma_kin_v_6_6 * x214
    x473 = x141 * x199
    x474 = sigma_kin_v_7_6 * x156
    x475 = x471 * x472 + x473 * x474
    x476 = dq_i1 * x423
    x477 = x175 * x464
    x478 = x383 * x477
    x479 = x139 * x434
    x480 = x389 * x479
    x481 = x472 * x478 + x474 * x480
    x482 = dq_i6 * x405
    x483 = x168 * x465
    x484 = x165 * x483
    x485 = x350 * x484
    x486 = x300 * x485
    x487 = sigma_kin_v_6_1 * sigma_kin_v_6_6
    x488 = x143 * x459
    x489 = x354 * x488
    x490 = x307 * x489
    x491 = x456 * x490 + x486 * x487
    x492 = dq_i6 * x411
    x493 = x298 * x465
    x494 = x299 * x493
    x495 = x300 * x494
    x496 = x305 * x458
    x497 = x306 * x496
    x498 = x307 * x497
    x499 = x456 * x498 + x487 * x495
    x500 = dq_i6 * x415
    x501 = x166 * x483
    x502 = x263 * x501
    x503 = x233 * x459
    x504 = x266 * x503
    x505 = x150 * x504
    x506 = x456 * x505 + x487 * x502
    x507 = dq_i6 * x419
    x508 = x223 * x501
    x509 = x230 * x503
    x510 = x149 * x509
    x511 = x456 * x510 + x487 * x508
    x512 = dq_i6**2
    x513 = 2 * x512
    x514 = (
        2 * ddq_i6 * x470
        + 4 * dq_i1 * dq_i6 * sigma_kin_v_6_1 * sigma_kin_v_6_6 * x166 * x168 * x170 * x172 * x176 * x177 * x182
        + 4 * dq_i1 * dq_i6 * sigma_kin_v_7_1 * sigma_kin_v_7_6 * x138 * x140 * x144 * x146 * x148 * x150 * x154 * x199
        - 2 * x457 * x463
        - 2 * x475 * x476
        - 2 * x481 * x482
        - 2 * x491 * x492
        - 2 * x499 * x500
        - 2 * x506 * x507
        - 2 * x511 * x513
    )
    x515 = 2 * dq_i1
    x516 = x199 * x515
    x517 = x399 * x516
    x518 = x103 * x107
    x519 = x104 * x106
    x520 = sigma_kin_p_7_7_c * x518 - sigma_kin_p_7_7_s * x519
    x521 = x400 * x520
    x522 = x517 * x521
    x523 = x195 * x522
    x524 = 2 * x187
    x525 = x194 + x524
    x526 = x404 * x405
    x527 = x521 * x526
    x528 = x138 * x525
    x529 = x410 * x411
    x530 = x521 * x528
    x531 = x397 * x530
    x532 = x138 * x399
    x533 = x229 * x532
    x534 = x153 * x533
    x535 = x520 * x534
    x536 = x423 * x535
    x537 = ddq_i1 * dq_j1
    x538 = 2 * x537
    x539 = sigma_kin_v_7_1 * x400
    x540 = x520 * x532
    x541 = x539 * x540
    x542 = x149 * x532
    x543 = x203 * x520
    x544 = dq_i7 * x543
    x545 = x542 * x544
    x546 = x187 * x406
    x547 = sigma_kin_p_7_7_c * x108 + sigma_kin_p_7_7_s * x105
    x548 = x532 * x547
    x549 = sigma_kin_v_7_2 * x400
    x550 = x540 * x549
    x551 = -x281 * x550
    x552 = sigma_kin_v_7_3 * x400
    x553 = x540 * x552
    x554 = -x275 * x553
    x555 = sigma_kin_v_7_4 * x400
    x556 = x540 * x555
    x557 = -x250 * x556
    x558 = sigma_kin_v_7_5 * x400
    x559 = x540 * x558
    x560 = -x244 * x559
    x561 = sigma_kin_v_7_6 * x400
    x562 = x540 * x561
    x563 = -x217 * x562
    x564 = x551 + x554 + x557 + x560 + x563
    x565 = x195 * x542
    x566 = x195 * x532
    x567 = x406 * x566
    x568 = x138 * x402
    x569 = x520 * x568
    x570 = x158 * x569
    x571 = x544 * x565 + x547 * x567 - x570
    x572 = 4 * x542
    x573 = dq_j7 * x572
    x574 = sigma_kin_v_7_1 * x573
    x575 = sigma_kin_v_7_1 * sigma_kin_v_7_5
    x576 = x150 * x575
    x577 = x419 * x462
    x578 = sigma_kin_v_6_1 * sigma_kin_v_6_5
    x579 = x508 * x578 + x510 * x575
    x580 = x128**2
    x581 = x127 * x580
    x582 = x124 * x581
    x583 = sigma_kin_v_5_1 * x582
    x584 = sigma_kin_v_5_5 * x583 + sigma_kin_v_6_5 * x467 + sigma_kin_v_7_5 * x469
    x585 = x129 * x133
    x586 = sigma_kin_v_5_5 * x241
    x587 = sigma_kin_v_6_5 * x214
    x588 = sigma_kin_v_7_5 * x156
    x589 = x471 * x587 + x473 * x588 + x585 * x586
    x590 = dq_i1 * x419
    x591 = x126 * x580
    x592 = x377 * x591
    x593 = x478 * x587 + x480 * x588 + x586 * x592
    x594 = dq_i5 * x405
    x595 = x123 * x581
    x596 = x118 * x595
    x597 = x346 * x596
    x598 = x121 * x597
    x599 = sigma_kin_v_5_1 * sigma_kin_v_5_5
    x600 = x486 * x578 + x490 * x575 + x598 * x599
    x601 = dq_i5 * x411
    x602 = x119 * x595
    x603 = x293 * x602
    x604 = x495 * x578 + x498 * x575 + x599 * x603
    x605 = dq_i5 * x415
    x606 = x252 * x581
    x607 = x255 * x606
    x608 = x121 * x607
    x609 = x502 * x578 + x505 * x575 + x599 * x608
    x610 = dq_i5**2
    x611 = 2 * x610
    x612 = (
        2 * ddq_i5 * x584
        + 4 * dq_i1 * dq_i5 * sigma_kin_v_5_1 * sigma_kin_v_5_5 * x119 * x121 * x123 * x127 * x128 * x133
        + 4 * dq_i1 * dq_i5 * sigma_kin_v_6_1 * sigma_kin_v_6_5 * x166 * x168 * x170 * x172 * x176 * x177 * x182
        + 4 * dq_i1 * dq_i5 * sigma_kin_v_7_1 * sigma_kin_v_7_5 * x138 * x140 * x144 * x146 * x148 * x150 * x154 * x199
        - 2 * x507 * x579
        - 2 * x576 * x577
        - 2 * x589 * x590
        - 2 * x593 * x594
        - 2 * x600 * x601
        - 2 * x604 * x605
        - 2 * x609 * x611
    )
    x613 = sigma_kin_v_7_1 * sigma_kin_v_7_4
    x614 = x150 * x613
    x615 = x415 * x462
    x616 = sigma_kin_v_6_1 * sigma_kin_v_6_4
    x617 = x501 * x616
    x618 = x223 * x617 + x510 * x613
    x619 = sigma_kin_v_5_1 * sigma_kin_v_5_4
    x620 = x263 * x617 + x505 * x613 + x608 * x619
    x621 = x79**2
    x622 = x621 * x78
    x623 = x622 * x76
    x624 = sigma_kin_v_4_1 * x623
    x625 = sigma_kin_v_4_4 * x624 + sigma_kin_v_5_4 * x583 + sigma_kin_v_6_4 * x467 + sigma_kin_v_7_4 * x469
    x626 = x442 * x80
    x627 = sigma_kin_v_5_4 * x241
    x628 = sigma_kin_v_6_4 * x214
    x629 = sigma_kin_v_7_4 * x156
    x630 = sigma_kin_v_4_4 * x626 + x471 * x628 + x473 * x629 + x585 * x627
    x631 = dq_i1 * x415
    x632 = x621 * x77
    x633 = x373 * x632
    x634 = sigma_kin_v_4_4 * x633 + x478 * x628 + x480 * x629 + x592 * x627
    x635 = dq_i4 * x405
    x636 = sigma_kin_v_4_1 * x622
    x637 = sigma_kin_v_4_4 * x636
    x638 = x343 * x637 + x486 * x616 + x490 * x613 + x598 * x619
    x639 = dq_i4 * x411
    x640 = x289 * x637 + x495 * x616 + x498 * x613 + x603 * x619
    x641 = dq_i4**2
    x642 = 2 * x641
    x643 = (
        2 * ddq_i4 * x625
        + 4 * dq_i1 * dq_i4 * sigma_kin_v_4_1 * sigma_kin_v_4_4 * x67 * x75 * x78 * x79 * x85
        + 4 * dq_i1 * dq_i4 * sigma_kin_v_5_1 * sigma_kin_v_5_4 * x119 * x121 * x123 * x127 * x128 * x133
        + 4 * dq_i1 * dq_i4 * sigma_kin_v_6_1 * sigma_kin_v_6_4 * x166 * x168 * x170 * x172 * x176 * x177 * x182
        + 4 * dq_i1 * dq_i4 * sigma_kin_v_7_1 * sigma_kin_v_7_4 * x138 * x140 * x144 * x146 * x148 * x150 * x154 * x199
        - 2 * x500 * x618
        - 2 * x605 * x620
        - 2 * x614 * x615
        - 2 * x630 * x631
        - 2 * x634 * x635
        - 2 * x638 * x639
        - 2 * x640 * x642
    )
    x644 = sigma_kin_v_7_1 * sigma_kin_v_7_3
    x645 = x150 * x644
    x646 = x411 * x462
    x647 = sigma_kin_v_6_1 * sigma_kin_v_6_3
    x648 = x501 * x647
    x649 = x223 * x648 + x510 * x644
    x650 = sigma_kin_v_5_1 * sigma_kin_v_5_3
    x651 = x263 * x648 + x505 * x644 + x608 * x650
    x652 = sigma_kin_v_4_3 * x636
    x653 = x289 * x652 + x495 * x647 + x498 * x644 + x603 * x650
    x654 = x40**2
    x655 = x51 * x654
    x656 = sigma_kin_v_3_1 * x655
    x657 = (
        sigma_kin_v_3_3 * x656
        + sigma_kin_v_4_3 * x624
        + sigma_kin_v_5_3 * x583
        + sigma_kin_v_6_3 * x467
        + sigma_kin_v_7_3 * x469
    )
    x658 = x52 * x59
    x659 = sigma_kin_v_5_3 * x241
    x660 = sigma_kin_v_6_3 * x214
    x661 = sigma_kin_v_7_3 * x156
    x662 = sigma_kin_v_4_3 * x626 + x273 * x658 + x471 * x660 + x473 * x661 + x585 * x659
    x663 = dq_i1 * x411
    x664 = x273 * x654
    x665 = sigma_kin_v_4_3 * x633 + x370 * x664 + x478 * x660 + x480 * x661 + x592 * x659
    x666 = dq_i3 * x405
    x667 = x336 * x664 + x343 * x652 + x486 * x647 + x490 * x644 + x598 * x650
    x668 = dq_i3**2
    x669 = 2 * x668
    x670 = (
        2 * ddq_i3 * x657
        + 4 * dq_i1 * dq_i3 * sigma_kin_v_3_1 * sigma_kin_v_3_3 * x40 * x42 * x50 * x59
        + 4 * dq_i1 * dq_i3 * sigma_kin_v_4_1 * sigma_kin_v_4_3 * x67 * x75 * x78 * x79 * x85
        + 4 * dq_i1 * dq_i3 * sigma_kin_v_5_1 * sigma_kin_v_5_3 * x119 * x121 * x123 * x127 * x128 * x133
        + 4 * dq_i1 * dq_i3 * sigma_kin_v_6_1 * sigma_kin_v_6_3 * x166 * x168 * x170 * x172 * x176 * x177 * x182
        + 4 * dq_i1 * dq_i3 * sigma_kin_v_7_1 * sigma_kin_v_7_3 * x138 * x140 * x144 * x146 * x148 * x150 * x154 * x199
        - 2 * x492 * x649
        - 2 * x601 * x651
        - 2 * x639 * x653
        - 2 * x645 * x646
        - 2 * x662 * x663
        - 2 * x665 * x666
        - 2 * x667 * x669
    )
    x671 = dq_i1 * x405
    x672 = x278 * x37
    x673 = x279 * x658
    x674 = sigma_kin_v_4_2 * x626
    x675 = sigma_kin_v_5_2 * x241
    x676 = x585 * x675
    x677 = sigma_kin_v_6_2 * x214
    x678 = x471 * x677
    x679 = sigma_kin_v_7_2 * x156
    x680 = x473 * x679
    x681 = sigma_kin_v_7_1 * sigma_kin_v_7_2
    x682 = x150 * x681
    x683 = x405 * x462
    x684 = sigma_kin_v_6_1 * sigma_kin_v_6_2
    x685 = x501 * x684
    x686 = x223 * x685 + x510 * x681
    x687 = sigma_kin_v_5_1 * sigma_kin_v_5_2
    x688 = x121 * x687
    x689 = x263 * x685 + x505 * x681 + x607 * x688
    x690 = sigma_kin_v_4_2 * x636
    x691 = x300 * x684
    x692 = x307 * x681
    x693 = x289 * x690 + x494 * x691 + x497 * x692 + x603 * x687
    x694 = x28**2
    x695 = x27 * x694
    x696 = (
        sigma_kin_v_3_2 * x656
        + sigma_kin_v_4_2 * x624
        + sigma_kin_v_5_2 * x583
        + sigma_kin_v_6_2 * x467
        + sigma_kin_v_7_2 * x469
        + x277 * x695
    )
    x697 = x279 * x654
    x698 = sigma_kin_v_4_2 * x343
    x699 = x336 * x697 + x486 * x684 + x490 * x681 + x598 * x687 + x636 * x698
    x700 = x672 + x673 + x674 + x676 + x678 + x680
    x701 = x26 * x694
    x702 = x277 * x701
    x703 = x362 * x702
    x704 = x370 * x697
    x705 = sigma_kin_v_4_2 * x632
    x706 = x373 * x705
    x707 = x592 * x675
    x708 = x478 * x677
    x709 = x480 * x679
    x710 = x703 + x704 + x706 + x707 + x708 + x709
    x711 = dq_i2**2
    x712 = 2 * x711
    x713 = (
        2 * ddq_i2 * x696
        - 2 * x482 * x686
        - 2 * x594 * x689
        - 2 * x635 * x693
        - 2 * x666 * x699
        + 2 * x671 * x672
        + 2 * x671 * x673
        + 2 * x671 * x674
        + 2 * x671 * x676
        + 2 * x671 * x678
        + 2 * x671 * x680
        - 2 * x671 * x700
        - 2 * x682 * x683
        - 2 * x710 * x712
    )
    x714 = x17 * x8
    x715 = x315 * x37
    x716 = x317 * x59
    x717 = x320 * x85
    x718 = x133 * x323
    x719 = x182 * x326
    x720 = x199 * x329
    x721 = x150 * x327
    x722 = x462 * x515
    x723 = x324 * x501
    x724 = x222 * x723
    x725 = x149 * x327
    x726 = x121 * x321
    x727 = x606 * x726
    x728 = x318 * x622
    x729 = x288 * x728
    x730 = x293 * x321
    x731 = x300 * x324
    x732 = x493 * x731
    x733 = x307 * x327
    x734 = x496 * x733
    x735 = x316 * x654
    x736 = x335 * x735
    x737 = x314 * x701
    x738 = x369 * x735
    x739 = x319 * x632
    x740 = x322 * x591
    x741 = x325 * x477
    x742 = x328 * x479
    x743 = 4 * ddq_j1
    x744 = sigma_kin_p_7_6_c * x220 - sigma_kin_p_7_6_s * x219
    x745 = x148 * x503
    x746 = x744 * x745
    x747 = x149 * x746
    x748 = sigma_kin_v_7_1 * x159
    x749 = x747 * x748
    x750 = sigma_kin_p_6_6_c * x220 - sigma_kin_p_6_6_s * x219
    x751 = x222 * x750
    x752 = x168 * x226
    x753 = x751 * x752
    x754 = x471 * x753
    x755 = x148 * x744
    x756 = x140 * x235
    x757 = x755 * x756
    x758 = x681 * x747 + x685 * x751
    x759 = x281 * x758
    x760 = x644 * x747 + x648 * x751
    x761 = x275 * x760
    x762 = x613 * x747 + x617 * x751
    x763 = x250 * x762
    x764 = x501 * x751
    x765 = x575 * x747 + x578 * x764
    x766 = x244 * x765
    x767 = x456 * x747 + x487 * x764
    x768 = x217 * x767
    x769 = x725 * x744
    x770 = x724 * x750 + x745 * x769
    x771 = x166 * x168
    x772 = x471 * x771
    x773 = x751 * x772
    x774 = x31 * x324
    x775 = x31 * x327
    x776 = x199 * x757
    x777 = x451 * x757
    x778 = sigma_kin_v_7_1 * x777 + x754
    x779 = x478 * x771
    x780 = x751 * x779
    x781 = x31 * x769
    x782 = x234 * x480
    x783 = x148 * x782
    x784 = x478 * x753
    x785 = x149 * x209
    x786 = x755 * x785
    x787 = x782 * x786
    x788 = x784 + x787
    x789 = x485 * x751
    x790 = x154 * x489
    x791 = x148 * x781
    x792 = x225 * x751
    x793 = x485 * x792
    x794 = x786 * x790
    x795 = x793 + x794
    x796 = x494 * x751
    x797 = x154 * x497
    x798 = x494 * x792
    x799 = x786 * x797
    x800 = x798 + x799
    x801 = x31 * x723
    x802 = x169 * x171
    x803 = x261 * x750 * x802
    x804 = x225 * x501
    x805 = x803 * x804
    x806 = x504 * x744
    x807 = x785 * x806
    x808 = x805 + x807
    x809 = x170 * x750
    x810 = x221 * x809
    x811 = sigma_kin_p_6_6_c * x101 + sigma_kin_p_6_6_s * x98
    x812 = x509 * x744
    x813 = sigma_kin_p_7_6_c * x101 + sigma_kin_p_7_6_s * x98
    x814 = x745 * x813
    x815 = x31 * x725
    x816 = x222 * x811
    x817 = x804 * x816
    x818 = x221 * x804
    x819 = x809 * x818
    x820 = x785 * x814
    x821 = x209 * x812
    x822 = x817 + x819 + x820 + x821
    x823 = 4 * dq_j6
    x824 = sigma_kin_p_5_5_c * x254 - sigma_kin_p_5_5_s * x253
    x825 = x252 * x824
    x826 = x257 * x825
    x827 = x585 * x826
    x828 = sigma_kin_p_7_5_c * x254 - sigma_kin_p_7_5_s * x253
    x829 = x147 * x503
    x830 = x828 * x829
    x831 = x150 * x830
    x832 = x748 * x831
    x833 = sigma_kin_p_6_5_c * x254 - sigma_kin_p_6_5_s * x253
    x834 = x262 * x833
    x835 = x752 * x834
    x836 = x471 * x835
    x837 = x210 * x828
    x838 = x147 * x837
    x839 = x234 * x838
    x840 = x141 * x516
    x841 = x501 * x834
    x842 = x456 * x831 + x487 * x841
    x843 = x217 * x842
    x844 = x606 * x824
    x845 = x681 * x831 + x685 * x834 + x688 * x844
    x846 = x281 * x845
    x847 = x121 * x844
    x848 = x644 * x831 + x648 * x834 + x650 * x847
    x849 = x275 * x848
    x850 = x613 * x831 + x617 * x834 + x619 * x847
    x851 = x250 * x850
    x852 = x575 * x831 + x578 * x841 + x599 * x847
    x853 = x244 * x852
    x854 = x721 * x828
    x855 = x723 * x834 + x727 * x824 + x829 * x854
    x856 = x802 * x833
    x857 = x221 * x856
    x858 = x229 * x830
    x859 = x818 * x856
    x860 = x785 * x858
    x861 = x859 + x860
    x862 = x31 * x726
    x863 = x585 * x825
    x864 = x774 * x834
    x865 = x31 * x854
    x866 = x147 * x865
    x867 = x473 * x839 + x827 + x836
    x868 = x592 * x825
    x869 = x592 * x826
    x870 = x478 * x835
    x871 = x480 * x839
    x872 = x869 + x870 + x871
    x873 = x581 * x824
    x874 = x122 * x873
    x875 = x347 * x874
    x876 = x348 * x874
    x877 = x225 * x834
    x878 = x485 * x877
    x879 = x790 * x838
    x880 = x876 + x878 + x879
    x881 = x31 * x730
    x882 = x233 * x398
    x883 = x458 * x882
    x884 = x306 * x883
    x885 = x294 * x844
    x886 = x494 * x877
    x887 = x837 * x884
    x888 = x885 + x886 + x887
    x889 = x119 * x873
    x890 = x255 * x889
    x891 = sigma_kin_p_5_5_c * x94 + sigma_kin_p_5_5_s * x91
    x892 = x172 * x261 * x833
    x893 = x262 * (sigma_kin_p_6_5_c * x94 + sigma_kin_p_6_5_s * x91)
    x894 = x265 * x503
    x895 = sigma_kin_p_7_5_c * x94 + sigma_kin_p_7_5_s * x91
    x896 = x829 * x895
    x897 = x31 * x721
    x898 = x606 * x891
    x899 = x257 * x898
    x900 = x258 * x889
    x901 = x804 * x893
    x902 = x804 * x892
    x903 = x210 * x896
    x904 = x837 * x894
    x905 = x899 + x900 + x901 + x902 + x903 + x904
    x906 = 4 * dq_j5
    x907 = x80 * x85
    x908 = sigma_kin_p_4_4_c * x286 - sigma_kin_p_4_4_s * x285
    x909 = x288 * x908
    x910 = sigma_kin_v_4_1 * x62
    x911 = x909 * x910
    x912 = x907 * x911
    x913 = x123 * x585
    x914 = sigma_kin_p_5_4_c * x286 - sigma_kin_p_5_4_s * x285
    x915 = x256 * x914
    x916 = x120 * x915
    x917 = x119 * x916
    x918 = x913 * x917
    x919 = sigma_kin_p_7_4_c * x286 - sigma_kin_p_7_4_s * x285
    x920 = x496 * x919
    x921 = x307 * x920
    x922 = x748 * x921
    x923 = x301 * x471
    x924 = sigma_kin_p_6_4_c * x286 - sigma_kin_p_6_4_s * x285
    x925 = x298 * x924
    x926 = x923 * x925
    x927 = x305 * x919
    x928 = x308 * x927
    x929 = x493 * x924
    x930 = x300 * x929
    x931 = x456 * x921 + x487 * x930
    x932 = x217 * x931
    x933 = x120 * x602
    x934 = x914 * x933
    x935 = x575 * x921 + x578 * x930 + x599 * x934
    x936 = x244 * x935
    x937 = x687 * x934 + x690 * x909 + x691 * x929 + x692 * x920
    x938 = x281 * x937
    x939 = x644 * x921 + x647 * x930 + x650 * x934 + x652 * x909
    x940 = x275 * x939
    x941 = x613 * x921 + x616 * x930 + x619 * x934 + x637 * x909
    x942 = x250 * x941
    x943 = x321 * x934 + x729 * x908 + x732 * x924 + x734 * x919
    x944 = x774 * x929
    x945 = x154 * x920
    x946 = x230 * x945
    x947 = x225 * x929
    x948 = x223 * x947
    x949 = x230 * x785
    x950 = x945 * x949
    x951 = x948 + x950
    x952 = x31 * x321
    x953 = x120 * x914
    x954 = x607 * x953
    x955 = x458 * x919
    x956 = x882 * x955
    x957 = x265 * x956
    x958 = x607 * x916
    x959 = x263 * x947
    x960 = x210 * x957
    x961 = x958 + x959 + x960
    x962 = x31 * x318
    x963 = x909 * x962
    x964 = x119 * x953
    x965 = x913 * x964
    x966 = x31 * x731
    x967 = x471 * x925
    x968 = x31 * x733
    x969 = x473 * x927
    x970 = x473 * x928 + x912 + x918 + x926
    x971 = x372 * x632
    x972 = x123 * x592
    x973 = x964 * x972
    x974 = x478 * x925
    x975 = x480 * x927
    x976 = x911 * x971
    x977 = x917 * x972
    x978 = x301 * x478
    x979 = x925 * x978
    x980 = x480 * x928
    x981 = x976 + x977 + x979 + x980
    x982 = x31 * x728
    x983 = x66 * x74
    x984 = x341 * x908 * x983
    x985 = x597 * x953
    x986 = x465 * x924
    x987 = x167 * x986
    x988 = x351 * x987
    x989 = x145 * x955
    x990 = x355 * x989
    x991 = x62 * x636
    x992 = x984 * x991
    x993 = x597 * x916
    x994 = x352 * x987
    x995 = x356 * x989
    x996 = x992 + x993 + x994 + x995
    x997 = x67 * x908
    x998 = x287 * x997
    x999 = sigma_kin_p_4_4_c * x73 + sigma_kin_p_4_4_s * x70
    x1000 = x292 * x602
    x1001 = x1000 * x914
    x1002 = x933 * (sigma_kin_p_5_4_c * x73 + sigma_kin_p_5_4_s * x70)
    x1003 = x166 * x986
    x1004 = x1003 * x299
    x1005 = sigma_kin_p_6_4_c * x73 + sigma_kin_p_6_4_s * x70
    x1006 = x144 * x955
    x1007 = x1006 * x306
    x1008 = sigma_kin_p_7_4_c * x73 + sigma_kin_p_7_4_s * x70
    x1009 = x288 * x999
    x1010 = x1009 * x991
    x1011 = x287 * x991
    x1012 = x1011 * x997
    x1013 = x1002 * x256
    x1014 = x1000 * x915
    x1015 = x1005 * x493
    x1016 = x1015 * x301
    x1017 = x1003 * x302
    x1018 = x1008 * x496
    x1019 = x1018 * x308
    x1020 = x1006 * x309
    x1021 = x1010 + x1012 + x1013 + x1014 + x1016 + x1017 + x1019 + x1020
    x1022 = 4 * dq_j4
    x1023 = sigma_kin_p_3_3_c * x333 - sigma_kin_p_3_3_s * x332
    x1024 = x1023 * x335
    x1025 = x338 * x59
    x1026 = x1024 * x1025
    x1027 = sigma_kin_p_4_3_c * x333 - sigma_kin_p_4_3_s * x332
    x1028 = x1027 * x342
    x1029 = x1028 * x910
    x1030 = x1029 * x907
    x1031 = sigma_kin_p_5_3_c * x333 - sigma_kin_p_5_3_s * x332
    x1032 = x1031 * x118
    x1033 = x1032 * x257
    x1034 = x1033 * x913
    x1035 = sigma_kin_p_7_3_c * x333 - sigma_kin_p_7_3_s * x332
    x1036 = x1035 * x488
    x1037 = x1036 * x307
    x1038 = x1037 * x748
    x1039 = sigma_kin_p_6_3_c * x333 - sigma_kin_p_6_3_s * x332
    x1040 = x1039 * x165
    x1041 = x1040 * x168
    x1042 = x1041 * x923
    x1043 = x146 * x308
    x1044 = x1035 * x143
    x1045 = x1039 * x484
    x1046 = x1045 * x300
    x1047 = x1037 * x456 + x1046 * x487
    x1048 = x1047 * x217
    x1049 = x1031 * x596
    x1050 = x1049 * x121
    x1051 = x1037 * x575 + x1046 * x578 + x1050 * x599
    x1052 = x1051 * x244
    x1053 = x1028 * x637 + x1037 * x613 + x1046 * x616 + x1050 * x619
    x1054 = x1053 * x250
    x1055 = x1024 * x697 + x1028 * x690 + x1036 * x692 + x1045 * x691 + x1049 * x688
    x1056 = x1055 * x281
    x1057 = x1024 * x664 + x1028 * x652 + x1037 * x644 + x1046 * x647 + x1050 * x650
    x1058 = x1057 * x275
    x1059 = x1023 * x736 + x1028 * x728 + x1036 * x733 + x1045 * x731 + x1049 * x726
    x1060 = x1045 * x774
    x1061 = x1036 * x154
    x1062 = x1061 * x230
    x1063 = x1045 * x225
    x1064 = x1063 * x223
    x1065 = x1061 * x949
    x1066 = x1064 + x1065
    x1067 = x122 * x581
    x1068 = x1032 * x1067
    x1069 = x1068 * x255
    x1070 = x1061 * x266
    x1071 = x1068 * x258
    x1072 = x1063 * x263
    x1073 = x1061 * x267
    x1074 = x1071 + x1072 + x1073
    x1075 = x1027 * x983
    x1076 = x1075 * x287
    x1077 = x167 * x465
    x1078 = x1040 * x1077
    x1079 = x1078 * x299
    x1080 = x145 * x458
    x1081 = x1044 * x1080
    x1082 = x1081 * x306
    x1083 = x1011 * x1075
    x1084 = x1049 * x294
    x1085 = x1078 * x302
    x1086 = x1081 * x309
    x1087 = x1083 + x1084 + x1085 + x1086
    x1088 = x31 * x316
    x1089 = x40 * x59
    x1090 = x1024 * x1089
    x1091 = x1028 * x962
    x1092 = x1032 * x913
    x1093 = x1041 * x471
    x1094 = x1044 * x473
    x1095 = x1094 * x146
    x1096 = x1026 + x1030 + x1034 + x1042 + x1043 * x1094
    x1097 = x31 * x735
    x1098 = x41 * x49
    x1099 = x1023 * x1098 * x368
    x1100 = x1032 * x972
    x1101 = x1041 * x478
    x1102 = x1035 * x389 * x434
    x1103 = x146 * x397
    x1104 = x1102 * x1103
    x1105 = x337 * x654
    x1106 = x1099 * x1105
    x1107 = x1029 * x971
    x1108 = x1033 * x972
    x1109 = x1041 * x978
    x1110 = x1043 * x397
    x1111 = x1102 * x1110
    x1112 = x1106 + x1107 + x1108 + x1109 + x1111
    x1113 = x1023 * x42
    x1114 = x1113 * x334
    x1115 = sigma_kin_p_3_3_c * x48 + sigma_kin_p_3_3_s * x45
    x1116 = x1027 * x341 * x75
    x1117 = x342 * (sigma_kin_p_4_3_c * x48 + sigma_kin_p_4_3_s * x45)
    x1118 = x1031 * x346
    x1119 = x1118 * x595
    x1120 = x596 * (sigma_kin_p_5_3_c * x48 + sigma_kin_p_5_3_s * x45)
    x1121 = x1039 * x350
    x1122 = x1121 * x483
    x1123 = x484 * (sigma_kin_p_6_3_c * x48 + sigma_kin_p_6_3_s * x45)
    x1124 = x1035 * x354
    x1125 = x1124 * x459
    x1126 = sigma_kin_p_7_3_c * x48 + sigma_kin_p_7_3_s * x45
    x1127 = x1126 * x488
    x1128 = x1115 * x335
    x1129 = x1105 * x1128
    x1130 = x1105 * x334
    x1131 = x1117 * x991
    x1132 = x1120 * x257
    x1133 = x121 * x595
    x1134 = x1118 * x1133
    x1135 = x1123 * x301
    x1136 = x300 * x483
    x1137 = x1121 * x1136
    x1138 = x1127 * x308
    x1139 = x307 * x459
    x1140 = x1124 * x1139
    x1141 = (
        x1113 * x1130
        + x1116 * x991
        + x1129
        + x1131
        + x1132
        + x1134 * x256
        + x1135
        + x1137 * x225
        + x1138
        + x1140 * x209
    )
    x1142 = 4 * dq_j3
    x1143 = sigma_kin_p_2_2_c * x361 - sigma_kin_p_2_2_s * x360
    x1144 = x1143 * x365 * x37
    x1145 = x1144 * x363
    x1146 = sigma_kin_p_3_2_c * x361 - sigma_kin_p_3_2_s * x360
    x1147 = x1146 * x369
    x1148 = x1025 * x1147
    x1149 = sigma_kin_p_4_2_c * x361 - sigma_kin_p_4_2_s * x360
    x1150 = x1149 * x62
    x1151 = x374 * x442
    x1152 = x1150 * x1151
    x1153 = sigma_kin_p_5_2_c * x361 - sigma_kin_p_5_2_s * x360
    x1154 = x1153 * x133 * x380
    x1155 = x1154 * x378
    x1156 = sigma_kin_p_7_2_c * x361 - sigma_kin_p_7_2_s * x360
    x1157 = x1156 * x479
    x1158 = x1157 * x159
    x1159 = x1158 * x156
    x1160 = sigma_kin_p_6_2_c * x361 - sigma_kin_p_6_2_s * x360
    x1161 = x1160 * x182 * x386
    x1162 = x1161 * x384
    x1163 = x1156 * x392
    x1164 = x1160 * x477
    x1165 = x1157 * x474 + x1164 * x472
    x1166 = x1165 * x217
    x1167 = x1153 * x591
    x1168 = x1157 * x588 + x1164 * x587 + x1167 * x586
    x1169 = x1168 * x244
    x1170 = x246 * x632
    x1171 = x1149 * x1170
    x1172 = sigma_kin_v_4_4 * x1171 + x1157 * x629 + x1164 * x628 + x1167 * x627
    x1173 = x1172 * x250
    x1174 = sigma_kin_v_4_3 * x1171 + x1147 * x664 + x1157 * x661 + x1164 * x660 + x1167 * x659
    x1175 = x1174 * x275
    x1176 = x246 * x705
    x1177 = x1143 * x702 + x1147 * x697 + x1149 * x1176 + x1157 * x679 + x1164 * x677 + x1167 * x675
    x1178 = x1177 * x281
    x1179 = x1143 * x737 + x1146 * x738 + x1149 * x739 + x1153 * x740 + x1156 * x742 + x1160 * x741
    x1180 = x1164 * x168
    x1181 = x1180 * x223
    x1182 = x166 * x774
    x1183 = x1157 * x234
    x1184 = x1183 * x230
    x1185 = x1180 * x226
    x1186 = x1185 * x223
    x1187 = x1183 * x949
    x1188 = x1186 + x1187
    x1189 = x1167 * x252
    x1190 = x1189 * x255
    x1191 = x1180 * x263
    x1192 = x1183 * x266
    x1193 = x1167 * x259
    x1194 = x1185 * x263
    x1195 = x1157 * x146
    x1196 = x1195 * x268
    x1197 = x1193 + x1194 + x1196
    x1198 = x1149 * x632
    x1199 = x1198 * x289
    x1200 = x1167 * x123
    x1201 = x119 * x1200
    x1202 = x1164 * x298
    x1203 = x1202 * x299
    x1204 = x1157 * x306
    x1205 = x1204 * x305
    x1206 = x1150 * x632
    x1207 = x1206 * x290
    x1208 = x1201 * x294
    x1209 = x1164 * x303
    x1210 = x1157 * x310
    x1211 = x1207 + x1208 + x1209 + x1210
    x1212 = x1098 * x1146
    x1213 = x1212 * x334
    x1214 = x1198 * x343
    x1215 = x1200 * x347
    x1216 = x1180 * x351
    x1217 = x1156 * x434
    x1218 = x1217 * x354
    x1219 = x1103 * x1218
    x1220 = x1130 * x1212
    x1221 = x1206 * x344
    x1222 = x1200 * x348
    x1223 = x1180 * x352
    x1224 = x1110 * x1218
    x1225 = x1220 + x1221 + x1222 + x1223 + x1224
    x1226 = x31 * x314
    x1227 = x1089 * x1147
    x1228 = x31 * x319
    x1229 = x1149 * x85
    x1230 = x1229 * x374
    x1231 = x31 * x322
    x1232 = x31 * x325
    x1233 = x31 * x328
    x1234 = x1163 * x199
    x1235 = x1145 + x1148 + x1152 + x1155 + x1162 + x1163 * x452
    x1236 = x1143 * x694
    x1237 = x1236 * x362
    x1238 = sigma_kin_p_2_2_c * x25 + sigma_kin_p_2_2_s * x22
    x1239 = x1146 * x368 * x50
    x1240 = sigma_kin_p_3_2_c * x25 + sigma_kin_p_3_2_s * x22
    x1241 = x1149 * x621
    x1242 = x1241 * x372
    x1243 = sigma_kin_p_4_2_c * x25 + sigma_kin_p_4_2_s * x22
    x1244 = x1153 * x580
    x1245 = x1244 * x377
    x1246 = sigma_kin_p_5_2_c * x25 + sigma_kin_p_5_2_s * x22
    x1247 = x1160 * x464
    x1248 = x1247 * x383
    x1249 = sigma_kin_p_6_2_c * x25 + sigma_kin_p_6_2_s * x22
    x1250 = x1217 * x389
    x1251 = sigma_kin_p_7_2_c * x25 + sigma_kin_p_7_2_s * x22
    x1252 = x1238 * x701
    x1253 = x1252 * x363
    x1254 = x1240 * x369
    x1255 = x1105 * x1254
    x1256 = x1243 * x62
    x1257 = x1170 * x1256
    x1258 = x1241 * x373
    x1259 = x1246 * x591
    x1260 = x1259 * x378
    x1261 = x1249 * x477
    x1262 = x1261 * x384
    x1263 = x1251 * x479
    x1264 = x1263 * x390
    x1265 = (
        x1105 * x1239
        + x1217 * x391
        + x1236 * x364
        + x1244 * x379
        + x1247 * x385
        + x1253
        + x1255
        + x1257
        + x1258 * x62
        + x1260
        + x1262
        + x1264
    )
    x1266 = 4 * dq_j2
    x1267 = dq_j1 * x11
    x1268 = x223 * x224
    x1269 = x230 * x236
    x1270 = x240 * x252
    x1271 = x1270 * x255
    x1272 = x224 * x263
    x1273 = x1272 * x166
    x1274 = x202 * x233
    x1275 = x1274 * x266
    x1276 = x80 * x83
    x1277 = x1276 * x289
    x1278 = x213 * x298
    x1279 = x1278 * x299
    x1280 = x142 * x306
    x1281 = x1280 * x305
    x1282 = x40 * x58
    x1283 = x1282 * x336
    x1284 = x1276 * x343
    x1285 = x295 * x347
    x1286 = x224 * x351
    x1287 = x202 * x355
    x1288 = x362 * x366
    x1289 = x1282 * x370
    x1290 = x372 * x374
    x1291 = x1290 * x83
    x1292 = x377 * x381
    x1293 = x383 * x387
    x1294 = x389 * x393
    x1295 = x83 * x86
    x1296 = 4 * x34
    x1297 = 4 * x56
    x1298 = sigma_pot_1_c * x15 - sigma_pot_1_s * x14
    x1299 = x110 * x1298
    x1300 = sigma_pot_2_c * x361 - sigma_pot_2_s * x360
    x1301 = x112 * x1300
    x1302 = 4 * x63
    x1303 = x374 * x76
    x1304 = 4 * x116
    x1305 = x124 * x1304
    x1306 = ddq_j7 * x473
    x1307 = sigma_kin_v_7_2 * x155
    x1308 = 4 * x1307
    x1309 = dq_i7 * sigma_kin_v_7_7
    x1310 = dq_i2 * x1309
    x1311 = 4 * x163
    x1312 = x1311 * x173
    x1313 = 4 * x196
    x1314 = x1313 * x155
    x1315 = x1195 * x205
    x1316 = 4 * ddq_j6
    x1317 = dq_i6 * x1316
    x1318 = -x182
    x1319 = x1318 * x178
    x1320 = x1319 * x173
    x1321 = sigma_kin_v_6_2 * x1320
    x1322 = -x199
    x1323 = x1322 * x141
    x1324 = x1323 * x155
    x1325 = sigma_kin_v_7_2 * x1324
    x1326 = dq_i2 * (sigma_kin_v_6_6 * x1321 + sigma_kin_v_7_6 * x1325)
    x1327 = 4 * ddq_j5
    x1328 = dq_i5 * x1327
    x1329 = -x133
    x1330 = x129 * x1329
    x1331 = x124 * x1330
    x1332 = sigma_kin_v_5_2 * x1331
    x1333 = dq_i2 * (sigma_kin_v_5_5 * x1332 + sigma_kin_v_6_5 * x1321 + sigma_kin_v_7_5 * x1325)
    x1334 = 4 * ddq_j4
    x1335 = dq_i4 * x1334
    x1336 = -x85
    x1337 = x1336 * x80
    x1338 = x1337 * x76
    x1339 = sigma_kin_v_4_2 * x1338
    x1340 = dq_i2 * (
        sigma_kin_v_4_4 * x1339 + sigma_kin_v_5_4 * x1332 + sigma_kin_v_6_4 * x1321 + sigma_kin_v_7_4 * x1325
    )
    x1341 = 4 * ddq_j3
    x1342 = dq_i3 * x1341
    x1343 = -x59
    x1344 = x1343 * x52
    x1345 = sigma_kin_v_3_2 * sigma_kin_v_3_3
    x1346 = dq_i2 * (
        sigma_kin_v_4_3 * x1339
        + sigma_kin_v_5_3 * x1332
        + sigma_kin_v_6_3 * x1321
        + sigma_kin_v_7_3 * x1325
        + x1344 * x1345
    )
    x1347 = -x37
    x1348 = sigma_kin_v_2_2**2
    x1349 = x1348 * x29
    x1350 = sigma_kin_v_3_2**2
    x1351 = sigma_kin_v_4_2**2
    x1352 = x1351 * x76
    x1353 = sigma_kin_v_5_2**2
    x1354 = x124 * x1353
    x1355 = sigma_kin_v_6_2**2
    x1356 = x1355 * x173
    x1357 = sigma_kin_v_7_2**2
    x1358 = x1357 * x155
    x1359 = 4 * ddq_j2
    x1360 = x1359 * x711
    x1361 = 8 * x31
    x1362 = dq_i2 * dq_j1
    x1363 = sigma_kin_v_7_1 * x1362
    x1364 = x406 * x548
    x1365 = ddq_i2 * dq_j1
    x1366 = x405 * x521
    x1367 = x199 * x399
    x1368 = x1367 * x187
    x1369 = dq_j1 * x666
    x1370 = x138 * x520
    x1371 = x1370 * x539
    x1372 = x1371 * x410
    x1373 = dq_j1 * x635
    x1374 = x397 * x539
    x1375 = x1370 * x1374
    x1376 = x1375 * x416
    x1377 = dq_j1 * x594
    x1378 = x1375 * x420
    x1379 = dq_j1 * x482
    x1380 = sigma_kin_v_7_1 * x535
    x1381 = dq_j1 * x712
    x1382 = x1371 * x404
    x1383 = x399 * x451
    x1384 = x1383 * x521
    x1385 = sigma_kin_v_7_2 * x573
    x1386 = x461 * x744
    x1387 = dq_j1 * x1386
    x1388 = sigma_kin_v_6_2 * x162
    x1389 = x1388 * x773
    x1390 = sigma_kin_v_7_2 * x777
    x1391 = x166 * x684
    x1392 = x1391 * x168
    x1393 = x1392 * x751
    x1394 = x1393 * x471 + x681 * x776
    x1395 = x31 * x405
    x1396 = x684 * x751
    x1397 = x149 * x681
    x1398 = x1397 * x755
    x1399 = x1396 * x485 + x1398 * x790
    x1400 = x1396 * x494 + x1398 * x797
    x1401 = x1397 * x806 + x685 * x803
    x1402 = x1393 * x478 + x1398 * x782
    x1403 = x149 * x814
    x1404 = x1403 * x681 + x681 * x812 + x685 * x810 + x685 * x816
    x1405 = dq_i6 * x1362
    x1406 = sigma_kin_v_5_2 * x115
    x1407 = x121 * x1406
    x1408 = x1407 * x863
    x1409 = dq_j1 * x405
    x1410 = x147 * x682
    x1411 = x1410 * x828
    x1412 = x144 * x459
    x1413 = x203 * x208
    x1414 = x1412 * x1413
    x1415 = x1411 * x1414
    x1416 = x1388 * x834
    x1417 = x1416 * x772
    x1418 = sigma_kin_v_7_2 * x195
    x1419 = x1418 * x150
    x1420 = x473 * x828
    x1421 = x1420 * x234
    x1422 = x1421 * x147
    x1423 = x1419 * x1422
    x1424 = x1397 * x858 + x685 * x857
    x1425 = x688 * x825
    x1426 = x1392 * x834
    x1427 = x1410 * x234
    x1428 = x1420 * x1427 + x1425 * x585 + x1426 * x471
    x1429 = x347 * x688
    x1430 = x684 * x834
    x1431 = x1411 * x790 + x1429 * x874 + x1430 * x485
    x1432 = x293 * x687
    x1433 = x828 * x884
    x1434 = x1430 * x494 + x1432 * x844 + x1433 * x682
    x1435 = x1411 * x782 + x1425 * x592 + x1426 * x478
    x1436 = x828 * x894
    x1437 = x1436 * x682 + x682 * x896 + x685 * x892 + x685 * x893 + x688 * x890 + x688 * x898
    x1438 = dq_i5 * x1362
    x1439 = sigma_kin_v_4_2 * x62
    x1440 = x1439 * x907
    x1441 = x1440 * x909
    x1442 = x1406 * x965
    x1443 = x148 * x203
    x1444 = x1443 * x920
    x1445 = x208 * x682
    x1446 = x1409 * x1445
    x1447 = x1388 * x300
    x1448 = x1447 * x967
    x1449 = x1418 * x307
    x1450 = x1449 * x969
    x1451 = x684 * x929
    x1452 = x1397 * x230
    x1453 = x1451 * x223 + x1452 * x945
    x1454 = x120 * x687
    x1455 = x1454 * x914
    x1456 = x1451 * x263 + x1455 * x607 + x682 * x957
    x1457 = sigma_kin_v_4_2 * x907
    x1458 = sigma_kin_v_4_1 * x909
    x1459 = x119 * x1454
    x1460 = x1459 * x914
    x1461 = x691 * x925
    x1462 = x692 * x927
    x1463 = x1457 * x1458 + x1460 * x913 + x1461 * x471 + x1462 * x473
    x1464 = x351 * x691
    x1465 = x1455 * x597 + x1464 * x987 + x690 * x984 + x692 * x990
    x1466 = x372 * x705
    x1467 = x1458 * x1466 + x1460 * x972 + x1461 * x478 + x1462 * x480
    x1468 = (
        x1001 * x687
        + x1002 * x687
        + x1004 * x691
        + x1007 * x692
        + x1009 * x690
        + x1015 * x691
        + x1018 * x692
        + x690 * x998
    )
    x1469 = dq_i4 * x1362
    x1470 = sigma_kin_v_3_2 * x55
    x1471 = x1090 * x1470
    x1472 = x1028 * x1440
    x1473 = x1092 * x1407
    x1474 = x1036 * x1443
    x1475 = x1093 * x1447
    x1476 = x1095 * x1449
    x1477 = x1045 * x684
    x1478 = x1061 * x1452 + x1477 * x223
    x1479 = x255 * x688
    x1480 = x266 * x682
    x1481 = x1061 * x1480 + x1068 * x1479 + x1477 * x263
    x1482 = x299 * x691
    x1483 = x1049 * x1432 + x1076 * x690 + x1078 * x1482 + x1082 * x692
    x1484 = x1089 * x279
    x1485 = sigma_kin_v_4_1 * x1028
    x1486 = x1032 * x688
    x1487 = x1041 * x691
    x1488 = x146 * x692
    x1489 = x1024 * x1484 + x1094 * x1488 + x1457 * x1485 + x1486 * x913 + x1487 * x471
    x1490 = x1488 * x397
    x1491 = x1099 * x697 + x1102 * x1490 + x1466 * x1485 + x1486 * x972 + x1487 * x478
    x1492 = dq_i3 * (
        x1114 * x697
        + x1116 * x690
        + x1117 * x690
        + x1119 * x688
        + x1120 * x688
        + x1122 * x691
        + x1123 * x691
        + x1125 * x692
        + x1127 * x692
        + x1128 * x697
    )
    x1493 = sigma_kin_v_2_2 * x33
    x1494 = x1144 * x1493
    x1495 = x1227 * x1470
    x1496 = sigma_kin_v_4_2 * x76
    x1497 = x1496 * x62
    x1498 = x1230 * x1497
    x1499 = x124 * x1406
    x1500 = x1154 * x1499
    x1501 = x1388 * x173
    x1502 = x1161 * x1501
    x1503 = x1163 * x451
    x1504 = x1307 * x1503
    x1505 = x1180 * x1391
    x1506 = x1149 * x705
    x1507 = x305 * x692
    x1508 = x1149 * x1151
    x1509 = x1176 * x1243 + x1238 * x702 + x1254 * x697 + x1259 * x675 + x1261 * x677 + x1263 * x679
    x1510 = x1493 * x30
    x1511 = -x36
    x1512 = x1347 * x27
    x1513 = x1470 * x53
    x1514 = -x58
    x1515 = x1343 * x51
    x1516 = x1514 * x1515
    x1517 = x1439 * x82
    x1518 = -x83
    x1519 = sigma_kin_v_4_2 * x1518
    x1520 = x1336 * x78
    x1521 = x62 * x76
    x1522 = x1520 * x1521
    x1523 = x131 * x1406
    x1524 = -x132
    x1525 = x127 * x1329
    x1526 = x1524 * x1525
    x1527 = x1388 * x180
    x1528 = -x181
    x1529 = x1318 * x176
    x1530 = x1528 * x1529
    x1531 = -x137
    x1532 = x141 * x1531
    x1533 = -x203
    x1534 = x146 * x204
    x1535 = x1533 * x1534
    x1536 = x1532 * x1535
    x1537 = x1307 * x195
    x1538 = x1537 * x186
    x1539 = x1322 * x140
    x1540 = x1531 * x1539
    x1541 = -x221
    x1542 = x1541 * x222
    x1543 = x1528 * x178
    x1544 = x1392 * x1543
    x1545 = -x229
    x1546 = x148 * x1545
    x1547 = x1546 * x681
    x1548 = x1531 * x756
    x1549 = x129 * x1524
    x1550 = x1549 * x688
    x1551 = -x255
    x1552 = x1551 * x252
    x1553 = -x261 * x262
    x1554 = -x265
    x1555 = x1532 * x1554
    x1556 = -x287
    x1557 = x1556 * x288
    x1558 = sigma_kin_v_4_1 * x80
    x1559 = x1519 * x1558
    x1560 = -x292
    x1561 = x123 * x1560
    x1562 = x1549 * x1561
    x1563 = x1543 * x691
    x1564 = -x299
    x1565 = x1564 * x298
    x1566 = -x306
    x1567 = x1532 * x1566
    x1568 = x1514 * x279
    x1569 = x1524 * x675
    x1570 = x1528 * x677
    x1571 = x1531 * x679
    x1572 = -x334
    x1573 = x1572 * x335
    x1574 = x1568 * x40
    x1575 = -x341
    x1576 = x1575 * x342
    x1577 = -x346
    x1578 = x118 * x123
    x1579 = x1577 * x1578
    x1580 = -x350
    x1581 = x165 * x168
    x1582 = x1580 * x1581
    x1583 = -x354
    x1584 = x1488 * x1583
    x1585 = x143 * x1532
    x1586 = -x362
    x1587 = x1511 * x277
    x1588 = -x368
    x1589 = x1588 * x369
    x1590 = x1519 * x246
    x1591 = -x372
    x1592 = x1591 * x374
    x1593 = -x377
    x1594 = x1593 * x380
    x1595 = -x383
    x1596 = x1595 * x386
    x1597 = -x389
    x1598 = x1597 * x392
    x1599 = sigma_kin_v_4_2 * x445 + x130 * x675 + x179 * x677 + x186 * x679 + x277 * x30 + x279 * x53
    x1600 = sigma_pot_3_c * x333 - sigma_pot_3_s * x332
    x1601 = x111 * x1600
    x1602 = x1302 * x907
    x1603 = x121 * x1304
    x1604 = 4 * x1306
    x1605 = sigma_kin_v_7_3 * x155
    x1606 = x1309 * x1605
    x1607 = dq_i3 * x1606
    x1608 = x1311 * x300
    x1609 = x1313 * x307
    x1610 = x1443 * x212
    x1611 = sigma_kin_v_6_3 * x1320
    x1612 = sigma_kin_v_7_3 * x1324
    x1613 = sigma_kin_v_6_6 * x1611 + sigma_kin_v_7_6 * x1612
    x1614 = dq_i3 * x1317
    x1615 = sigma_kin_v_5_3 * x1331
    x1616 = sigma_kin_v_5_5 * x1615 + sigma_kin_v_6_5 * x1611 + sigma_kin_v_7_5 * x1612
    x1617 = dq_i3 * x1328
    x1618 = sigma_kin_v_4_3 * sigma_kin_v_4_4
    x1619 = sigma_kin_v_5_4 * x1615 + sigma_kin_v_6_4 * x1611 + sigma_kin_v_7_4 * x1612 + x1338 * x1618
    x1620 = dq_i3 * x1335
    x1621 = dq_i3 * x1359
    x1622 = sigma_kin_v_3_3**2
    x1623 = sigma_kin_v_4_3**2
    x1624 = sigma_kin_v_5_3**2
    x1625 = sigma_kin_v_6_3**2
    x1626 = sigma_kin_v_7_3**2
    x1627 = x1341 * x668
    x1628 = dq_i3 * dq_j1
    x1629 = sigma_kin_v_7_1 * x1628
    x1630 = ddq_i3 * dq_j1
    x1631 = x1368 * x521
    x1632 = dq_j1 * x639
    x1633 = dq_j1 * x601
    x1634 = dq_j1 * x492
    x1635 = dq_j1 * x669
    x1636 = sigma_kin_v_7_3 * x573
    x1637 = sigma_kin_v_6_3 * x162
    x1638 = x1637 * x773
    x1639 = sigma_kin_v_7_3 * x777
    x1640 = x166 * x647
    x1641 = x1640 * x168
    x1642 = x1641 * x751
    x1643 = x1642 * x471 + x644 * x776
    x1644 = x31 * x411
    x1645 = x149 * x644
    x1646 = x1645 * x755
    x1647 = x1642 * x478 + x1646 * x782
    x1648 = x647 * x751
    x1649 = x1646 * x797 + x1648 * x494
    x1650 = x1645 * x806 + x648 * x803
    x1651 = x1646 * x790 + x1648 * x485
    x1652 = x1403 * x644 + x644 * x812 + x648 * x810 + x648 * x816
    x1653 = dq_i6 * x1628
    x1654 = sigma_kin_v_5_3 * x115
    x1655 = x121 * x1654
    x1656 = x1655 * x863
    x1657 = dq_j1 * x411
    x1658 = x147 * x645
    x1659 = x1658 * x828
    x1660 = x1414 * x1659
    x1661 = x1637 * x834
    x1662 = x1661 * x772
    x1663 = sigma_kin_v_7_3 * x195
    x1664 = x150 * x1663
    x1665 = x1422 * x1664
    x1666 = x1645 * x858 + x648 * x857
    x1667 = x121 * x650
    x1668 = x1667 * x825
    x1669 = x1641 * x834
    x1670 = x1658 * x234
    x1671 = x1420 * x1670 + x1668 * x585 + x1669 * x471
    x1672 = x1659 * x782 + x1668 * x592 + x1669 * x478
    x1673 = x293 * x650
    x1674 = x647 * x834
    x1675 = x1433 * x645 + x1673 * x844 + x1674 * x494
    x1676 = x1667 * x347
    x1677 = x1659 * x790 + x1674 * x485 + x1676 * x874
    x1678 = x1436 * x645 + x1667 * x890 + x1667 * x898 + x645 * x896 + x648 * x892 + x648 * x893
    x1679 = dq_i5 * x1628
    x1680 = sigma_kin_v_4_3 * x62
    x1681 = x1680 * x907
    x1682 = x1681 * x909
    x1683 = x1654 * x965
    x1684 = x208 * x645
    x1685 = x1657 * x1684
    x1686 = x1637 * x300
    x1687 = x1686 * x967
    x1688 = x1663 * x307
    x1689 = x1688 * x969
    x1690 = x647 * x929
    x1691 = x1645 * x230
    x1692 = x1690 * x223 + x1691 * x945
    x1693 = x120 * x650
    x1694 = x1693 * x914
    x1695 = x1690 * x263 + x1694 * x607 + x645 * x957
    x1696 = sigma_kin_v_4_3 * x907
    x1697 = x119 * x1693
    x1698 = x1697 * x914
    x1699 = x300 * x647
    x1700 = x1699 * x925
    x1701 = x307 * x644
    x1702 = x1701 * x927
    x1703 = x1458 * x1696 + x1698 * x913 + x1700 * x471 + x1702 * x473
    x1704 = sigma_kin_v_4_3 * x372
    x1705 = x1704 * x632
    x1706 = x1458 * x1705 + x1698 * x972 + x1700 * x478 + x1702 * x480
    x1707 = x1699 * x351
    x1708 = x1694 * x597 + x1701 * x990 + x1707 * x987 + x652 * x984
    x1709 = dq_i4 * (
        x1001 * x650
        + x1002 * x650
        + x1004 * x1699
        + x1007 * x1701
        + x1009 * x652
        + x1015 * x1699
        + x1018 * x1701
        + x652 * x998
    )
    x1710 = sigma_kin_v_3_3 * x55
    x1711 = x1227 * x1710
    x1712 = sigma_kin_v_4_3 * x76
    x1713 = x1150 * x1712 * x374 * x85
    x1714 = x124 * x1654
    x1715 = x1154 * x1714
    x1716 = x1637 * x173
    x1717 = x1161 * x1716
    x1718 = x1503 * x1605
    x1719 = x1180 * x1640
    x1720 = x1184 * x1645 + x1719 * x223
    x1721 = x1667 * x255
    x1722 = x266 * x645
    x1723 = x1183 * x1722 + x1189 * x1721 + x1719 * x263
    x1724 = x1198 * x290
    x1725 = x1699 * x299
    x1726 = x1701 * x305
    x1727 = sigma_kin_v_4_3 * x1724 + x1201 * x1673 + x1202 * x1725 + x1204 * x1726
    x1728 = sigma_kin_v_4_3 * x1508 + x1154 * x659 + x1161 * x660 + x1227 * x273 + x1234 * x661
    x1729 = sigma_kin_v_4_3 * x344
    x1730 = x146 * x1701
    x1731 = x1730 * x397
    x1732 = x1180 * x1707 + x1198 * x1729 + x1200 * x1676 + x1213 * x664 + x1218 * x1731
    x1733 = x1170 * x1243
    x1734 = (
        sigma_kin_v_4_3 * x1258
        + sigma_kin_v_4_3 * x1733
        + x1239 * x664
        + x1245 * x659
        + x1248 * x660
        + x1250 * x661
        + x1254 * x664
        + x1259 * x659
        + x1261 * x660
        + x1263 * x661
    )
    x1735 = x1090 * x1710
    x1736 = x1028 * x1681
    x1737 = x1092 * x1655
    x1738 = x1093 * x1686
    x1739 = x1095 * x1688
    x1740 = x1045 * x647
    x1741 = x1032 * x1667
    x1742 = x1041 * x1699
    x1743 = x1117 * x652 + x1120 * x1667 + x1123 * x1699 + x1127 * x1701 + x1128 * x664
    x1744 = x1710 * x53
    x1745 = x1680 * x82
    x1746 = sigma_kin_v_4_3 * x1518
    x1747 = x131 * x1654
    x1748 = dq_i3 * x1714
    x1749 = x1637 * x180
    x1750 = dq_i3 * x1716
    x1751 = x1605 * x195
    x1752 = x1751 * x186
    x1753 = dq_i3 * x1751
    x1754 = x1543 * x1641
    x1755 = x1546 * x644
    x1756 = x1549 * x1667
    x1757 = x1558 * x1746
    x1758 = x1543 * x1699
    x1759 = x1514 * x273
    x1760 = x1524 * x659
    x1761 = x1528 * x660
    x1762 = x1531 * x661
    x1763 = x1759 * x40
    x1764 = x1746 * x246
    x1765 = sigma_kin_v_4_3 * x445 + x130 * x659 + x179 * x660 + x186 * x661 + x273 * x53
    x1766 = 1.0 * x102 * x109
    x1767 = x1298 * x1766
    x1768 = sigma_pot_4_c * x286 - sigma_pot_4_s * x285
    x1769 = x113 * x95
    x1770 = x1768 * x1769
    x1771 = x1309 * x1604
    x1772 = sigma_kin_v_7_4 * x155
    x1773 = dq_i4 * x1772
    x1774 = sigma_kin_v_6_4 * x1320
    x1775 = sigma_kin_v_7_4 * x1324
    x1776 = dq_i4 * dq_i6
    x1777 = x1776 * (sigma_kin_v_6_6 * x1774 + sigma_kin_v_7_6 * x1775)
    x1778 = sigma_kin_v_5_4 * sigma_kin_v_5_5
    x1779 = dq_i4 * dq_i5
    x1780 = x1779 * (sigma_kin_v_6_5 * x1774 + sigma_kin_v_7_5 * x1775 + x1331 * x1778)
    x1781 = dq_i4 * x1359
    x1782 = dq_i4 * x1342
    x1783 = sigma_kin_v_4_4**2
    x1784 = sigma_kin_v_5_4**2
    x1785 = sigma_kin_v_6_4**2
    x1786 = sigma_kin_v_7_4**2
    x1787 = x1334 * x641
    x1788 = dq_i4 * dq_j1
    x1789 = sigma_kin_v_7_1 * x1788
    x1790 = ddq_i4 * dq_j1
    x1791 = dq_j1 * x605
    x1792 = dq_j1 * x500
    x1793 = dq_j1 * x642
    x1794 = sigma_kin_v_7_4 * x573
    x1795 = dq_j1 * x415
    x1796 = x406 * x744
    x1797 = x1796 * x461
    x1798 = x1797 * x613
    x1799 = sigma_kin_v_6_4 * x162
    x1800 = x1799 * x773
    x1801 = sigma_kin_v_7_4 * x777
    x1802 = x166 * x616
    x1803 = x168 * x1802
    x1804 = x1803 * x751
    x1805 = x1804 * x471 + x613 * x776
    x1806 = x31 * x415
    x1807 = x149 * x613
    x1808 = x1807 * x755
    x1809 = x1804 * x478 + x1808 * x782
    x1810 = x616 * x751
    x1811 = x1808 * x790 + x1810 * x485
    x1812 = x1807 * x806 + x617 * x803
    x1813 = x1808 * x797 + x1810 * x494
    x1814 = x1403 * x613 + x613 * x812 + x617 * x810 + x617 * x816
    x1815 = dq_i6 * x1788
    x1816 = sigma_kin_v_5_4 * x115
    x1817 = x121 * x1816
    x1818 = x1817 * x863
    x1819 = x147 * x614
    x1820 = x1819 * x828
    x1821 = x1414 * x1820
    x1822 = x1799 * x834
    x1823 = x1822 * x772
    x1824 = sigma_kin_v_7_4 * x195
    x1825 = x150 * x1824
    x1826 = x1422 * x1825
    x1827 = x1807 * x858 + x617 * x857
    x1828 = x121 * x619
    x1829 = x1828 * x825
    x1830 = x1803 * x834
    x1831 = x1421 * x1819 + x1829 * x585 + x1830 * x471
    x1832 = x1820 * x782 + x1829 * x592 + x1830 * x478
    x1833 = x1828 * x347
    x1834 = x616 * x834
    x1835 = x1820 * x790 + x1833 * x874 + x1834 * x485
    x1836 = x293 * x619
    x1837 = x1433 * x614 + x1834 * x494 + x1836 * x844
    x1838 = dq_i5 * (x1436 * x614 + x1828 * x890 + x1828 * x898 + x614 * x896 + x617 * x892 + x617 * x893)
    x1839 = sigma_kin_v_4_4 * x1230 * x1521
    x1840 = x124 * x1816
    x1841 = x1154 * x1840
    x1842 = x1315 * x208
    x1843 = x1795 * x614
    x1844 = x173 * x1799
    x1845 = x1161 * x1844
    x1846 = x1503 * x1772
    x1847 = x1802 * x223
    x1848 = x1180 * x1847 + x1184 * x1807
    x1849 = x1828 * x255
    x1850 = x266 * x614
    x1851 = x1183 * x1850 + x1189 * x1849 + x1191 * x1802
    x1852 = sigma_kin_v_4_4 * x1508 + x1154 * x627 + x1161 * x628 + x1234 * x629
    x1853 = x300 * x616
    x1854 = x1853 * x351
    x1855 = x307 * x613
    x1856 = x1103 * x1855
    x1857 = sigma_kin_v_4_4 * x1198 * x344 + x1180 * x1854 + x1200 * x1833 + x1218 * x1856
    x1858 = x1853 * x299
    x1859 = x1855 * x305
    x1860 = sigma_kin_v_4_4 * x1724 + x1201 * x1836 + x1202 * x1858 + x1204 * x1859
    x1861 = (
        sigma_kin_v_4_4 * x1258
        + sigma_kin_v_4_4 * x1733
        + x1245 * x627
        + x1248 * x628
        + x1250 * x629
        + x1259 * x627
        + x1261 * x628
        + x1263 * x629
    )
    x1862 = sigma_kin_v_4_4 * x62
    x1863 = x1028 * x1862
    x1864 = x1863 * x907
    x1865 = x1092 * x1817
    x1866 = x1843 * x208
    x1867 = x1799 * x300
    x1868 = x1093 * x1867
    x1869 = x1824 * x307
    x1870 = x1095 * x1869
    x1871 = x1045 * x616
    x1872 = x230 * x613
    x1873 = x149 * x1872
    x1874 = x1061 * x1873 + x1871 * x223
    x1875 = x1061 * x1850 + x1068 * x1849 + x1871 * x263
    x1876 = sigma_kin_v_4_4 * x1485
    x1877 = x1092 * x1828 + x1093 * x1853 + x1095 * x1855 + x1876 * x907
    x1878 = x1853 * x478
    x1879 = x1041 * x1878 + x1100 * x1828 + x1102 * x1856 + x1876 * x971
    x1880 = x1049 * x1836 + x1076 * x637 + x1078 * x1858 + x1082 * x1855
    x1881 = (
        x1116 * x637
        + x1117 * x637
        + x1120 * x1828
        + x1123 * x1853
        + x1127 * x1855
        + x1134 * x619
        + x1137 * x616
        + x1140 * x613
    )
    x1882 = x1862 * x909
    x1883 = x1882 * x907
    x1884 = x1816 * x965
    x1885 = x1867 * x967
    x1886 = x1869 * x969
    x1887 = x616 * x929
    x1888 = x1873 * x945 + x1887 * x223
    x1889 = x619 * x953
    x1890 = x1887 * x263 + x1889 * x607 + x614 * x957
    x1891 = sigma_kin_v_4_4 * x1458
    x1892 = x119 * x1889
    x1893 = x1855 * x927
    x1894 = x1853 * x967 + x1891 * x907 + x1892 * x913 + x1893 * x473
    x1895 = x1878 * x925 + x1891 * x971 + x1892 * x972 + x1893 * x480
    x1896 = x1854 * x987 + x1855 * x990 + x1889 * x597 + x637 * x984
    x1897 = (
        x1001 * x619
        + x1002 * x619
        + x1004 * x1853
        + x1007 * x1855
        + x1009 * x637
        + x1015 * x1853
        + x1018 * x1855
        + x637 * x998
    )
    x1898 = x1862 * x82
    x1899 = x283 * x86
    x1900 = sigma_kin_v_4_4 * x76
    x1901 = x1899 * x1900
    x1902 = x131 * x1816
    x1903 = x135 * x1816
    x1904 = x1799 * x180
    x1905 = x1799 * x184
    x1906 = x1772 * x195
    x1907 = x186 * x1906
    x1908 = x200 * x451
    x1909 = x1772 * x1908
    x1910 = x1847 * x224 + x1872 * x236
    x1911 = x1270 * x1849 + x1272 * x1802 + x1274 * x1850
    x1912 = sigma_kin_v_4_4 * x83
    x1913 = x1292 * x627 + x1293 * x628 + x1294 * x629 + x1912 * x375
    x1914 = x1912 * x80
    x1915 = x1287 * x1855 + x1833 * x295 + x1854 * x224 + x1914 * x344
    x1916 = x1278 * x1858 + x1280 * x1859 + x1836 * x296 + x1914 * x290
    x1917 = x443 * x83
    x1918 = (
        sigma_kin_v_4_4 * x1917
        + sigma_kin_v_4_4 * x445
        + x130 * x627
        + x134 * x627
        + x179 * x628
        + x183 * x628
        + x186 * x629
        + x201 * x629
    )
    x1919 = sigma_pot_5_c * x254 - sigma_pot_5_s * x253
    x1920 = x1919 * x88
    x1921 = x113 * x1920
    x1922 = sigma_kin_v_7_5 * x155
    x1923 = dq_i5 * x1922
    x1924 = x1311 * x834
    x1925 = x1313 * x150
    x1926 = sigma_kin_v_6_5 * sigma_kin_v_6_6
    x1927 = sigma_kin_v_7_5 * sigma_kin_v_7_6
    x1928 = dq_i5 * dq_i6
    x1929 = x1928 * (x1320 * x1926 + x1324 * x1927)
    x1930 = dq_i5 * x1359
    x1931 = dq_i5 * x1342
    x1932 = sigma_kin_v_5_5**2
    x1933 = sigma_kin_v_6_5**2
    x1934 = sigma_kin_v_7_5**2
    x1935 = x1327 * x610
    x1936 = dq_j1 * sigma_kin_v_7_1
    x1937 = dq_i5 * x1936
    x1938 = ddq_i5 * dq_j1
    x1939 = dq_j1 * x507
    x1940 = dq_j1 * x611
    x1941 = sigma_kin_v_7_5 * x573
    x1942 = dq_j1 * x419
    x1943 = x1797 * x575
    x1944 = sigma_kin_v_6_5 * x162
    x1945 = x1944 * x773
    x1946 = sigma_kin_v_7_5 * x777
    x1947 = x166 * x578
    x1948 = x168 * x1947
    x1949 = x1948 * x751
    x1950 = x1949 * x471 + x575 * x776
    x1951 = x31 * x419
    x1952 = x149 * x575
    x1953 = x1952 * x755
    x1954 = x1949 * x478 + x1953 * x782
    x1955 = x578 * x751
    x1956 = x1953 * x790 + x1955 * x485
    x1957 = x1953 * x797 + x1955 * x494
    x1958 = x501 * x578
    x1959 = x1952 * x806 + x1958 * x803
    x1960 = x1403 * x575 + x1958 * x810 + x1958 * x816 + x575 * x812
    x1961 = dq_j1 * x1928
    x1962 = sigma_kin_v_5_5 * x115
    x1963 = x124 * x1962
    x1964 = x1154 * x1963
    x1965 = x1942 * x576
    x1966 = x173 * x1944
    x1967 = x1161 * x1966
    x1968 = x1503 * x1922
    x1969 = x1947 * x223
    x1970 = x1180 * x1969 + x1184 * x1952
    x1971 = x1154 * x586 + x1161 * x587 + x1234 * x588
    x1972 = x121 * x599
    x1973 = x1972 * x347
    x1974 = x300 * x578
    x1975 = x1974 * x351
    x1976 = x307 * x575
    x1977 = x1103 * x1976
    x1978 = x1180 * x1975 + x1200 * x1973 + x1218 * x1977
    x1979 = x293 * x599
    x1980 = x1974 * x299
    x1981 = x1976 * x305
    x1982 = x1201 * x1979 + x1202 * x1980 + x1204 * x1981
    x1983 = x1972 * x255
    x1984 = x266 * x576
    x1985 = x1183 * x1984 + x1189 * x1983 + x1191 * x1947
    x1986 = x1245 * x586 + x1248 * x587 + x1250 * x588 + x1259 * x586 + x1261 * x587 + x1263 * x588
    x1987 = x121 * x1962
    x1988 = x1092 * x1987
    x1989 = x1965 * x208
    x1990 = x1944 * x300
    x1991 = x1093 * x1990
    x1992 = sigma_kin_v_7_5 * x195
    x1993 = x1992 * x307
    x1994 = x1095 * x1993
    x1995 = x1045 * x578
    x1996 = x230 * x575
    x1997 = x149 * x1996
    x1998 = x1061 * x1997 + x1995 * x223
    x1999 = x1092 * x1972 + x1093 * x1974 + x1095 * x1976
    x2000 = x1974 * x478
    x2001 = x1041 * x2000 + x1100 * x1972 + x1102 * x1977
    x2002 = x1049 * x1979 + x1078 * x1980 + x1082 * x1976
    x2003 = x1061 * x1984 + x1068 * x1983 + x1995 * x263
    x2004 = x1120 * x1972 + x1123 * x1974 + x1127 * x1976 + x1134 * x599 + x1137 * x578 + x1140 * x575
    x2005 = x1962 * x965
    x2006 = x1990 * x967
    x2007 = x1993 * x969
    x2008 = x578 * x929
    x2009 = x1997 * x945 + x2008 * x223
    x2010 = x1974 * x967 + x1976 * x969 + x599 * x965
    x2011 = x1976 * x975 + x2000 * x925 + x599 * x973
    x2012 = x599 * x953
    x2013 = x1975 * x987 + x1976 * x990 + x2012 * x597
    x2014 = x2008 * x263 + x2012 * x607 + x576 * x957
    x2015 = x1001 * x599 + x1002 * x599 + x1004 * x1974 + x1007 * x1976 + x1015 * x1974 + x1018 * x1976
    x2016 = x1987 * x863
    x2017 = x147 * x576
    x2018 = x2017 * x828
    x2019 = x1944 * x834
    x2020 = x2019 * x772
    x2021 = x150 * x1992
    x2022 = x1422 * x2021
    x2023 = x1952 * x858 + x1958 * x857
    x2024 = x1972 * x825
    x2025 = x1948 * x834
    x2026 = x1421 * x2017 + x2024 * x585 + x2025 * x471
    x2027 = x2018 * x782 + x2024 * x592 + x2025 * x478
    x2028 = x578 * x834
    x2029 = x1973 * x874 + x2018 * x790 + x2028 * x485
    x2030 = x1433 * x576 + x1979 * x844 + x2028 * x494
    x2031 = x1436 * x576 + x1958 * x892 + x1958 * x893 + x1972 * x890 + x1972 * x898 + x576 * x896
    x2032 = x131 * x1962
    x2033 = x135 * x1962
    x2034 = x180 * x1944
    x2035 = x184 * x1944
    x2036 = x1922 * x195
    x2037 = x186 * x2036
    x2038 = x1908 * x1922
    x2039 = x1969 * x224 + x1996 * x236
    x2040 = x1292 * x586 + x1293 * x587 + x1294 * x588
    x2041 = x1287 * x1976 + x1973 * x295 + x1975 * x224
    x2042 = x1278 * x1980 + x1280 * x1981 + x1979 * x296
    x2043 = x1270 * x1983 + x1272 * x1947 + x1274 * x1984
    x2044 = x130 * x586 + x134 * x586 + x179 * x587 + x183 * x587 + x186 * x588 + x201 * x588
    x2045 = x1298 * x1769
    x2046 = sigma_pot_6_c * x220 - sigma_pot_6_s * x219
    x2047 = 1.0 * x88
    x2048 = x109 * x2047
    x2049 = x2046 * x2048
    x2050 = sigma_kin_v_7_6 * x155
    x2051 = dq_i6 * x2050
    x2052 = dq_i6 * x1359
    x2053 = dq_i6 * x1342
    x2054 = sigma_kin_v_6_6**2
    x2055 = sigma_kin_v_7_6**2
    x2056 = x1316 * x512
    x2057 = dq_i6 * x1936
    x2058 = ddq_i6 * dq_j1
    x2059 = x424 * x521
    x2060 = dq_j1 * x513
    x2061 = sigma_kin_v_7_6 * x573
    x2062 = dq_j1 * x423
    x2063 = x2062 * x457
    x2064 = sigma_kin_v_6_6 * x162
    x2065 = x173 * x2064
    x2066 = x1161 * x2065
    x2067 = x1503 * x2050
    x2068 = x1161 * x472 + x1234 * x474
    x2069 = x31 * x423
    x2070 = x300 * x487
    x2071 = x2070 * x351
    x2072 = x307 * x456
    x2073 = x1103 * x2072
    x2074 = x1180 * x2071 + x1218 * x2073
    x2075 = x2070 * x299
    x2076 = x2072 * x305
    x2077 = x1202 * x2075 + x1204 * x2076
    x2078 = x166 * x487
    x2079 = x266 * x457
    x2080 = x1183 * x2079 + x1191 * x2078
    x2081 = x149 * x456
    x2082 = x1181 * x2078 + x1184 * x2081
    x2083 = x1248 * x472 + x1250 * x474 + x1261 * x472 + x1263 * x474
    x2084 = x2063 * x208
    x2085 = x2064 * x300
    x2086 = x1093 * x2085
    x2087 = sigma_kin_v_7_6 * x195
    x2088 = x2087 * x307
    x2089 = x1095 * x2088
    x2090 = x1093 * x2070 + x1095 * x2072
    x2091 = x2070 * x478
    x2092 = x1041 * x2091 + x1102 * x2073
    x2093 = x1078 * x2075 + x1082 * x2072
    x2094 = x1045 * x487
    x2095 = x1061 * x2079 + x2094 * x263
    x2096 = x230 * x456
    x2097 = x149 * x2096
    x2098 = x1061 * x2097 + x2094 * x223
    x2099 = x1123 * x2070 + x1127 * x2072 + x1137 * x487 + x1140 * x456
    x2100 = x2085 * x967
    x2101 = x2088 * x969
    x2102 = x2070 * x967 + x2072 * x969
    x2103 = x2072 * x975 + x2091 * x925
    x2104 = x2071 * x987 + x2072 * x990
    x2105 = x487 * x929
    x2106 = x2105 * x263 + x457 * x957
    x2107 = x2097 * x945 + x2105 * x223
    x2108 = x1004 * x2070 + x1007 * x2072 + x1015 * x2070 + x1018 * x2072
    x2109 = x147 * x828
    x2110 = x2109 * x457
    x2111 = x1414 * x2110
    x2112 = x2064 * x834
    x2113 = x2112 * x772
    x2114 = x150 * x2087
    x2115 = x1422 * x2114
    x2116 = x487 * x834
    x2117 = x1422 * x457 + x2116 * x772
    x2118 = x2110 * x782 + x2116 * x779
    x2119 = x2110 * x790 + x2116 * x485
    x2120 = x1433 * x457 + x2116 * x494
    x2121 = x487 * x501
    x2122 = x2081 * x858 + x2121 * x857
    x2123 = x1436 * x457 + x2121 * x892 + x2121 * x893 + x457 * x896
    x2124 = x1797 * x456
    x2125 = x2064 * x773
    x2126 = sigma_kin_v_7_6 * x777
    x2127 = x487 * x751
    x2128 = x2127 * x772 + x456 * x776
    x2129 = x2081 * x755
    x2130 = x2127 * x779 + x2129 * x782
    x2131 = x2127 * x485 + x2129 * x790
    x2132 = x2127 * x494 + x2129 * x797
    x2133 = x2081 * x806 + x2121 * x803
    x2134 = x1403 * x456 + x2121 * x810 + x2121 * x816 + x456 * x812
    x2135 = x180 * x2064
    x2136 = x184 * x2064
    x2137 = x206 * x208
    x2138 = x2137 * x457
    x2139 = x195 * x2050
    x2140 = x186 * x2139
    x2141 = x1908 * x2050
    x2142 = x1293 * x472 + x1294 * x474
    x2143 = x1287 * x2072 + x2071 * x224
    x2144 = x1278 * x2075 + x1280 * x2076
    x2145 = x1272 * x2078 + x1274 * x2079
    x2146 = x1268 * x2078 + x2096 * x236
    x2147 = x179 * x472 + x183 * x472 + x186 * x474 + x201 * x474
    x2148 = sigma_pot_7_c * x518 - sigma_pot_7_s * x519
    x2149 = x102 * x2148
    x2150 = x2047 * x2149
    x2151 = x460 * x520
    x2152 = x153 * x2151
    x2153 = x136 * x2152
    x2154 = x2153 * x614
    x2155 = x2153 * x576
    x2156 = x2153 * x457
    x2157 = ddq_i7 * x153
    x2158 = x150 * x2151 * x2157
    x2159 = x158 * x2158
    x2160 = sigma_kin_v_7_1 * x136
    x2161 = x1310 * x1359
    x2162 = x1342 * x1606
    x2163 = x1309 * x473
    x2164 = x1335 * x1772
    x2165 = x1328 * x1922
    x2166 = x1317 * x2050
    x2167 = x153 * x721
    x2168 = 4 * sigma_kin_v_7_7**2
    x2169 = x155 * x2168
    x2170 = x2169 * x427
    x2171 = x136 * x460
    x2172 = x153 * x1925
    x2173 = x2172 * x520
    x2174 = x1534 * x2173
    x2175 = x1534 * x520
    x2176 = x2175 * x473
    x2177 = x150 * x153
    x2178 = 8 * x195
    x2179 = x2177 * x2178
    x2180 = x2176 * x2179
    x2181 = x153 * x210
    x2182 = x2175 * x480
    x2183 = x148 * x520
    x2184 = x2181 * x2183
    x2185 = x153 * x520
    x2186 = x1412 * x2185
    x2187 = x1412 * x521
    x2188 = 2 * ddq_j1
    x2189 = x1251 * x532
    x2190 = x2189 * x406
    x2191 = x1936 * x568
    x2192 = x138 * x406
    x2193 = x1156 * x404
    x2194 = x2192 * x2193
    x2195 = x1367 * x524
    x2196 = x2195 * x406
    x2197 = x138 * x413
    x2198 = x1156 * x1936
    x2199 = x138 * x418
    x2200 = x138 * x422
    x2201 = x208 * x533
    x2202 = x2201 * x423
    x2203 = x429 * x542
    x2204 = x1936 * x2203
    x2205 = 2 * x1156
    x2206 = x1383 * x406
    x2207 = x138 * x403
    x2208 = x1266 * x2207
    x2209 = x2208 * x431
    x2210 = x1126 * x532
    x2211 = x2210 * x406
    x2212 = x1035 * x410
    x2213 = x2192 * x2212
    x2214 = x138 * x408
    x2215 = x1035 * x1936
    x2216 = 2 * x2206
    x2217 = x1142 * x138 * x409
    x2218 = x2217 * x431
    x2219 = x1008 * x532
    x2220 = x2219 * x406
    x2221 = x138 * x414
    x2222 = x416 * x919
    x2223 = x2221 * x2222
    x2224 = x1936 * x919
    x2225 = x397 * x431
    x2226 = x138 * x147
    x2227 = x1022 * x2226
    x2228 = x2225 * x2227
    x2229 = x532 * x895
    x2230 = x2229 * x406
    x2231 = x420 * x828
    x2232 = x2221 * x2231
    x2233 = x1936 * x828
    x2234 = x138 * x145
    x2235 = x2234 * x906
    x2236 = x2225 * x2235
    x2237 = x532 * x813
    x2238 = x2237 * x406
    x2239 = x2201 * x744
    x2240 = x1936 * x744
    x2241 = x153 * x823
    x2242 = x2241 * x532
    x2243 = sigma_kin_v_7_7 * x2242
    x2244 = x532 * x539
    x2245 = x427 * x547
    x2246 = x428 * x542
    x2247 = x2246 * x520
    x2248 = x1367 * x520
    x2249 = x1936 * x520
    x2250 = x158 * x572
    x2251 = x185 * x532
    x2252 = x137 * x568
    x2253 = x1367 * x137
    x2254 = x137 * x1936
    x2255 = x137 * x1383
    x2256 = x136 * x399
    x2257 = x2256 * x431
    x2258 = sigma_pot_1_c * x14 - sigma_pot_1_s * x15
    x2259 = sigma_pot_2_c * x360 - sigma_pot_2_s * x361
    x2260 = x110 * x112
    x2261 = x1290 * x84
    x2262 = x1307 * x142
    x2263 = x159 * x2262
    x2264 = 4 * dq_i1
    x2265 = ddq_j7 * x480
    x2266 = x1309 * x2265
    x2267 = 8 * dq_j2
    x2268 = x1419 * x2267
    x2269 = x173 * x213
    x2270 = sigma_kin_v_6_2 * x2269
    x2271 = sigma_kin_v_6_6 * x2270 + sigma_kin_v_7_6 * x2262
    x2272 = x217 * x2271
    x2273 = x1595 * x477
    x2274 = x173 * x2273
    x2275 = x1597 * x479
    x2276 = x155 * x2275
    x2277 = dq_i1 * (x2274 * x487 + x2276 * x456)
    x2278 = x1388 * x166
    x2279 = x1268 * x2278
    x2280 = x1269 * x1418
    x2281 = x2279 + x2280
    x2282 = dq_i6 * x2267
    x2283 = x124 * x240
    x2284 = sigma_kin_v_5_2 * x2283
    x2285 = sigma_kin_v_5_5 * x2284 + sigma_kin_v_6_5 * x2270 + sigma_kin_v_7_5 * x2262
    x2286 = x2285 * x244
    x2287 = x1593 * x591
    x2288 = x124 * x2287
    x2289 = dq_i1 * (x2274 * x578 + x2276 * x575 + x2288 * x599)
    x2290 = x80 * x84
    x2291 = sigma_kin_v_4_2 * sigma_kin_v_4_4
    x2292 = sigma_kin_v_5_4 * x2284 + sigma_kin_v_6_4 * x2270 + sigma_kin_v_7_4 * x2262 + x2290 * x2291
    x2293 = x2292 * x250
    x2294 = x1591 * x632
    x2295 = x2294 * x246
    x2296 = dq_i1 * (sigma_kin_v_4_4 * x2295 + x2274 * x616 + x2276 * x613 + x2288 * x619)
    x2297 = x1407 * x255
    x2298 = x1270 * x2297
    x2299 = x1272 * x2278
    x2300 = x1419 * x266
    x2301 = x1274 * x2300
    x2302 = x2298 + x2299 + x2301
    x2303 = dq_i5 * x2267
    x2304 = sigma_kin_v_4_2 * sigma_kin_v_4_3
    x2305 = sigma_kin_v_5_2 * sigma_kin_v_5_3
    x2306 = sigma_kin_v_6_2 * sigma_kin_v_6_3
    x2307 = sigma_kin_v_7_3 * x2262 + x1345 * x272 + x2269 * x2306 + x2283 * x2305 + x2290 * x2304
    x2308 = x2307 * x275
    x2309 = dq_i1 * (sigma_kin_v_4_3 * x2295 + x1589 * x664 + x2274 * x647 + x2276 * x644 + x2288 * x650)
    x2310 = ddq_i1 * dq_j2
    x2311 = x2310 * x280
    x2312 = x1350 * x52
    x2313 = x1352 * x80
    x2314 = x129 * x1354
    x2315 = x1356 * x178
    x2316 = x1358 * x141
    x2317 = x132 * x2314 + x1349 * x36 + x137 * x2316 + x181 * x2315 + x2312 * x58 + x2313 * x83
    x2318 = 4 * ddq_i2 * dq_j2**2
    x2319 = x11 * x743
    x2320 = x284 * x289
    x2321 = sigma_kin_v_4_2 * x2320
    x2322 = x1406 * x293
    x2323 = x2322 * x296
    x2324 = x1447 * x299
    x2325 = x1278 * x2324
    x2326 = x1449 * x305
    x2327 = x1280 * x2326
    x2328 = x2321 + x2323 + x2325 + x2327
    x2329 = dq_i4 * x2267
    x2330 = x1282 * x1470
    x2331 = x2330 * x336
    x2332 = x284 * x698
    x2333 = x1407 * x347
    x2334 = x2333 * x295
    x2335 = x1447 * x351
    x2336 = x224 * x2335
    x2337 = x1287 * x1449
    x2338 = x2331 + x2332 + x2334 + x2336 + x2337
    x2339 = dq_i3 * x2267
    x2340 = x1288 * x1493
    x2341 = x2330 * x370
    x2342 = x1290 * x283
    x2343 = x1496 * x2342
    x2344 = x1292 * x1499
    x2345 = x1293 * x1501
    x2346 = x1294 * x1537
    x2347 = x2340 + x2341 + x2343 + x2344 + x2345 + x2346
    x2348 = 8 * x32
    x2349 = (
        x1307 * x1908
        + x135 * x1406
        + x1388 * x184
        + x1470 * x60
        + x1493 * x38
        + x1496 * x1899
        + x1510
        + x1513
        + x1517
        + x1523
        + x1527
        + x1538
    )
    x2350 = dq_i1 * dq_j2
    x2351 = sigma_kin_v_7_2 * x2350
    x2352 = x515 * x521
    x2353 = x138 * x404
    x2354 = x188 * x2353
    x2355 = dq_j2 * x663
    x2356 = x1370 * x410
    x2357 = x2356 * x549
    x2358 = dq_j2 * x631
    x2359 = x397 * x549
    x2360 = x1370 * x2359
    x2361 = x2360 * x416
    x2362 = dq_j2 * x590
    x2363 = x2360 * x420
    x2364 = dq_j2 * x476
    x2365 = sigma_kin_v_7_2 * x535
    x2366 = dq_j2 * x12
    x2367 = x2248 * x549
    x2368 = x138 * x195
    x2369 = x2368 * x521
    x2370 = x2369 * x404
    x2371 = dq_j2 * x515
    x2372 = x2310 * x758
    x2373 = x32 * x515
    x2374 = dq_i6 * x2350
    x2375 = x2310 * x845
    x2376 = dq_i5 * x2350
    x2377 = (
        2 * ddq_i1 * x696
        - 2 * x12 * x700
        - 2 * x476 * x686
        - 2 * x590 * x689
        - 2 * x631 * x693
        - 2 * x663 * x699
        + 2 * x671 * x703
        + 2 * x671 * x704
        + 2 * x671 * x706
        + 2 * x671 * x707
        + 2 * x671 * x708
        + 2 * x671 * x709
        - 2 * x671 * x710
        - 2 * x682 * x722
    )
    x2378 = x1445 * x2371
    x2379 = x2310 * x937
    x2380 = dq_i4 * x2350
    x2381 = x1055 * x2310
    x2382 = sigma_kin_v_4_2 * x83
    x2383 = x290 * x80
    x2384 = x279 * x40
    x2385 = x2384 * x58
    x2386 = x698 * x83
    x2387 = -x1143
    x2388 = x1586 * x694
    x2389 = -x1146
    x2390 = x1588 * x50
    x2391 = x2389 * x2390
    x2392 = dq_i1 * x1105
    x2393 = -x1149
    x2394 = x1591 * x621
    x2395 = x2393 * x2394 * x62
    x2396 = -x1153
    x2397 = x1593 * x580
    x2398 = x2396 * x2397
    x2399 = -x1160
    x2400 = x1595 * x464
    x2401 = x2399 * x2400
    x2402 = -x1156
    x2403 = x2402 * x479
    x2404 = x1535 * x2403
    x2405 = x1597 * x434
    x2406 = x2402 * x2405
    x2407 = x2399 * x477
    x2408 = x1392 * x2407
    x2409 = x149 * x234 * x2403
    x2410 = x2396 * x591
    x2411 = x2410 * x688
    x2412 = x1554 * x2403
    x2413 = x2393 * x705
    x2414 = sigma_kin_v_4_1 * x2413
    x2415 = x1561 * x2410
    x2416 = x2407 * x691
    x2417 = x1566 * x2403
    x2418 = x2389 * x697
    x2419 = x2396 * x675
    x2420 = x2399 * x677
    x2421 = x2402 * x679
    x2422 = x1098 * x1572
    x2423 = x2402 * x434
    x2424 = x2387 * x277
    x2425 = x1343 * x2389 * x369
    x2426 = sigma_kin_v_4_2 * x2393
    x2427 = x2426 * x246
    x2428 = x1336 * x374
    x2429 = x1329 * x380
    x2430 = x1318 * x386
    x2431 = x1322 * x392
    x2432 = x57 * x654
    x2433 = x64 * x76
    x2434 = x1243 * x632
    x2435 = sigma_pot_1_c * x5 + sigma_pot_1_off + sigma_pot_1_s * x2
    x2436 = x117 * x124
    x2437 = x1158 * x1307
    x2438 = x164 * x173
    x2439 = x208 * x2268
    x2440 = sigma_kin_v_6_2 * sigma_kin_v_6_6
    x2441 = x173 * x2440
    x2442 = sigma_kin_v_7_6 * x1307
    x2443 = x1157 * x2442 + x1164 * x2441
    x2444 = x217 * x2443
    x2445 = x1181 * x2278
    x2446 = x1418 * x149
    x2447 = x1184 * x2446
    x2448 = x2445 + x2447
    x2449 = sigma_kin_v_5_2 * sigma_kin_v_5_5
    x2450 = x124 * x2449
    x2451 = sigma_kin_v_6_2 * sigma_kin_v_6_5
    x2452 = x173 * x2451
    x2453 = sigma_kin_v_7_5 * x1307
    x2454 = x1157 * x2453 + x1164 * x2452 + x1167 * x2450
    x2455 = x244 * x2454
    x2456 = x705 * x76
    x2457 = sigma_kin_v_4_4 * x2456
    x2458 = sigma_kin_v_5_2 * sigma_kin_v_5_4
    x2459 = x124 * x2458
    x2460 = sigma_kin_v_6_2 * sigma_kin_v_6_4
    x2461 = x173 * x2460
    x2462 = sigma_kin_v_7_4 * x1307
    x2463 = x1149 * x2457 + x1157 * x2462 + x1164 * x2461 + x1167 * x2459
    x2464 = x2463 * x250
    x2465 = x1189 * x2297
    x2466 = x1191 * x2278
    x2467 = x1183 * x2300
    x2468 = x2465 + x2466 + x2467
    x2469 = x1345 * x654
    x2470 = sigma_kin_v_4_3 * x2456
    x2471 = x124 * x2305
    x2472 = x173 * x2306
    x2473 = sigma_kin_v_7_3 * x1307
    x2474 = x1147 * x2469 + x1149 * x2470 + x1157 * x2473 + x1164 * x2472 + x1167 * x2471
    x2475 = x2474 * x275
    x2476 = x136 * x2310
    x2477 = x1348 * x701
    x2478 = x1350 * x654
    x2479 = x1352 * x632
    x2480 = x1354 * x591
    x2481 = x1356 * x477
    x2482 = x1358 * x479
    x2483 = x1143 * x2477 + x1147 * x2478 + x1149 * x2479 + x1153 * x2480 + x1156 * x2482 + x1160 * x2481
    x2484 = x1150 * x705
    x2485 = x2484 * x289
    x2486 = x1201 * x2322
    x2487 = x1202 * x2324
    x2488 = x1204 * x2326
    x2489 = x2485 + x2486 + x2487 + x2488
    x2490 = x1470 * x654
    x2491 = x1213 * x2490
    x2492 = x2484 * x343
    x2493 = x1200 * x2333
    x2494 = x1180 * x2335
    x2495 = x1103 * x1449
    x2496 = x1218 * x2495
    x2497 = x2491 + x2492 + x2493 + x2494 + x2496
    x2498 = x1494 + x1495 + x1498 + x1500 + x1502 + x1504
    x2499 = 8 * x2350
    x2500 = x138 * x401
    x2501 = x406 * x516
    x2502 = x138 * x398
    x2503 = x354 * x412
    x2504 = x143 * x2192
    x2505 = x423 * x425
    x2506 = x232 * x429
    x2507 = x2207 * x2505 - x2500 * x403 + x2501 * x403 + x2502 * x2503 + x2504 * x417 + x2504 * x421 + x2506 * x403
    x2508 = sigma_kin_v_7_2 * x141 * x403
    x2509 = x1252 * x1493
    x2510 = x1237 * x1493
    x2511 = x1254 * x2490
    x2512 = x1239 * x2490
    x2513 = x1256 * x2456
    x2514 = x1242 * x1497
    x2515 = x1259 * x1499
    x2516 = x1245 * x1499
    x2517 = x1261 * x1501
    x2518 = x1248 * x1501
    x2519 = x1263 * x1537
    x2520 = x1250 * x1537
    x2521 = x2509 + x2510 + x2511 + x2512 + x2513 + x2514 + x2515 + x2516 + x2517 + x2518 + x2519 + x2520
    x2522 = sigma_kin_v_7_2 * sigma_kin_v_7_6
    x2523 = x150 * x2522
    x2524 = sigma_kin_v_6_2 * x466
    x2525 = sigma_kin_v_7_2 * x468
    x2526 = sigma_kin_v_6_6 * x2524 + sigma_kin_v_7_6 * x2525
    x2527 = x2441 * x471 + x2442 * x473
    x2528 = x2441 * x478 + x2442 * x480
    x2529 = x2440 * x486 + x2522 * x490
    x2530 = x2440 * x495 + x2522 * x498
    x2531 = x2440 * x502 + x2522 * x505
    x2532 = x2440 * x508 + x2522 * x510
    x2533 = (
        2 * ddq_i6 * x2526
        + 4 * dq_i2 * dq_i6 * sigma_kin_v_6_2 * sigma_kin_v_6_6 * x166 * x168 * x170 * x172 * x175 * x383 * x464
        + 4 * dq_i2 * dq_i6 * sigma_kin_v_7_2 * sigma_kin_v_7_6 * x139 * x144 * x146 * x148 * x150 * x154 * x389 * x434
        - 2 * x2523 * x463
        - 2 * x2527 * x476
        - 2 * x2528 * x482
        - 2 * x2529 * x492
        - 2 * x2530 * x500
        - 2 * x2531 * x507
        - 2 * x2532 * x513
    )
    x2534 = x2368 * x527
    x2535 = 2 * x188
    x2536 = x187 + x190 + x191 + x192 + x193
    x2537 = x189 + x2535 + x2536
    x2538 = x138 * x2537
    x2539 = x2538 * x521
    x2540 = x2539 * x397
    x2541 = 2 * x281
    x2542 = -x537 * x541
    x2543 = x2542 + x557 + x560 + x563 + x571
    x2544 = sigma_kin_v_7_2 * sigma_kin_v_7_5
    x2545 = x150 * x2544
    x2546 = x2451 * x508 + x2544 * x510
    x2547 = sigma_kin_v_5_2 * x582
    x2548 = sigma_kin_v_5_5 * x2547 + sigma_kin_v_6_5 * x2524 + sigma_kin_v_7_5 * x2525
    x2549 = x2450 * x585 + x2452 * x471 + x2453 * x473
    x2550 = x2450 * x592 + x2452 * x478 + x2453 * x480
    x2551 = x2449 * x598 + x2451 * x486 + x2544 * x490
    x2552 = x2449 * x603 + x2451 * x495 + x2544 * x498
    x2553 = x2449 * x608 + x2451 * x502 + x2544 * x505
    x2554 = (
        2 * ddq_i5 * x2548
        + 4 * dq_i2 * dq_i5 * sigma_kin_v_5_2 * sigma_kin_v_5_5 * x119 * x121 * x123 * x126 * x377 * x580
        + 4 * dq_i2 * dq_i5 * sigma_kin_v_6_2 * sigma_kin_v_6_5 * x166 * x168 * x170 * x172 * x175 * x383 * x464
        + 4 * dq_i2 * dq_i5 * sigma_kin_v_7_2 * sigma_kin_v_7_5 * x139 * x144 * x146 * x148 * x150 * x154 * x389 * x434
        - 2 * x2545 * x577
        - 2 * x2546 * x507
        - 2 * x2549 * x590
        - 2 * x2550 * x594
        - 2 * x2551 * x601
        - 2 * x2552 * x605
        - 2 * x2553 * x611
    )
    x2555 = sigma_kin_v_7_2 * sigma_kin_v_7_4
    x2556 = x150 * x2555
    x2557 = x2460 * x508 + x2555 * x510
    x2558 = x2458 * x608 + x2460 * x502 + x2555 * x505
    x2559 = sigma_kin_v_4_2 * x623
    x2560 = sigma_kin_v_4_4 * x2559 + sigma_kin_v_5_4 * x2547 + sigma_kin_v_6_4 * x2524 + sigma_kin_v_7_4 * x2525
    x2561 = x81 * x85
    x2562 = x2291 * x2561 + x2459 * x585 + x2461 * x471 + x2462 * x473
    x2563 = x2457 * x372 + x2459 * x592 + x2461 * x478 + x2462 * x480
    x2564 = x622 * x698
    x2565 = sigma_kin_v_4_4 * x2564 + x2458 * x598 + x2460 * x486 + x2555 * x490
    x2566 = x289 * x622
    x2567 = x2291 * x2566 + x2458 * x603 + x2460 * x495 + x2555 * x498
    x2568 = (
        2 * ddq_i4 * x2560
        + 4 * dq_i2 * dq_i4 * sigma_kin_v_4_2 * sigma_kin_v_4_4 * x372 * x621 * x67 * x75 * x77
        + 4 * dq_i2 * dq_i4 * sigma_kin_v_5_2 * sigma_kin_v_5_4 * x119 * x121 * x123 * x126 * x377 * x580
        + 4 * dq_i2 * dq_i4 * sigma_kin_v_6_2 * sigma_kin_v_6_4 * x166 * x168 * x170 * x172 * x175 * x383 * x464
        + 4 * dq_i2 * dq_i4 * sigma_kin_v_7_2 * sigma_kin_v_7_4 * x139 * x144 * x146 * x148 * x150 * x154 * x389 * x434
        - 2 * x2556 * x615
        - 2 * x2557 * x500
        - 2 * x2558 * x605
        - 2 * x2562 * x631
        - 2 * x2563 * x635
        - 2 * x2565 * x639
        - 2 * x2567 * x642
    )
    x2569 = sigma_kin_v_7_2 * sigma_kin_v_7_3
    x2570 = x150 * x2569
    x2571 = x2306 * x508 + x2569 * x510
    x2572 = x2305 * x608 + x2306 * x502 + x2569 * x505
    x2573 = x2304 * x2566 + x2305 * x603 + x2306 * x495 + x2569 * x498
    x2574 = (
        sigma_kin_v_4_3 * x2559
        + sigma_kin_v_5_3 * x2547
        + sigma_kin_v_6_3 * x2524
        + sigma_kin_v_7_3 * x2525
        + x1345 * x655
    )
    x2575 = x1345 * x658 + x2304 * x2561 + x2471 * x585 + x2472 * x471 + x2473 * x473
    x2576 = x1704 * x2456 + x2469 * x370 + x2471 * x592 + x2472 * x478 + x2473 * x480
    x2577 = sigma_kin_v_4_3 * x2564 + x2305 * x598 + x2306 * x486 + x2469 * x336 + x2569 * x490
    x2578 = (
        2 * ddq_i3 * x2574
        + 4 * dq_i2 * dq_i3 * sigma_kin_v_3_2 * sigma_kin_v_3_3 * x368 * x41 * x50 * x654
        + 4 * dq_i2 * dq_i3 * sigma_kin_v_4_2 * sigma_kin_v_4_3 * x372 * x621 * x67 * x75 * x77
        + 4 * dq_i2 * dq_i3 * sigma_kin_v_5_2 * sigma_kin_v_5_3 * x119 * x121 * x123 * x126 * x377 * x580
        + 4 * dq_i2 * dq_i3 * sigma_kin_v_6_2 * sigma_kin_v_6_3 * x166 * x168 * x170 * x172 * x175 * x383 * x464
        + 4 * dq_i2 * dq_i3 * sigma_kin_v_7_2 * sigma_kin_v_7_3 * x139 * x144 * x146 * x148 * x150 * x154 * x389 * x434
        - 2 * x2570 * x646
        - 2 * x2571 * x492
        - 2 * x2572 * x601
        - 2 * x2573 * x639
        - 2 * x2575 * x663
        - 2 * x2576 * x666
        - 2 * x2577 * x669
    )
    x2579 = x2477 * x362
    x2580 = x2478 * x370
    x2581 = x2479 * x372
    x2582 = x2480 * x377
    x2583 = x2481 * x383
    x2584 = x2482 * x389
    x2585 = x1357 * x150
    x2586 = x1355 * x501
    x2587 = x222 * x2586
    x2588 = x1357 * x149
    x2589 = x121 * x1353
    x2590 = x2589 * x606
    x2591 = x1351 * x622
    x2592 = x2591 * x288
    x2593 = x1355 * x300
    x2594 = x2593 * x493
    x2595 = x1357 * x307
    x2596 = x2595 * x496
    x2597 = x2478 * x335
    x2598 = sigma_kin_v_7_2 * x159
    x2599 = x2598 * x747
    x2600 = x1388 * x751
    x2601 = x2600 * x779
    x2602 = x2446 * x755
    x2603 = x2602 * x782
    x2604 = x2306 * x764 + x2569 * x747
    x2605 = x2604 * x275
    x2606 = x2460 * x764 + x2555 * x747
    x2607 = x250 * x2606
    x2608 = x2451 * x764 + x2544 * x747
    x2609 = x244 * x2608
    x2610 = x2440 * x764 + x2522 * x747
    x2611 = x217 * x2610
    x2612 = x2587 * x750 + x2588 * x746
    x2613 = x1355 * x32
    x2614 = x1357 * x32
    x2615 = x1389 + x1390
    x2616 = x2588 * x32
    x2617 = x2616 * x755
    x2618 = x2601 + x2603
    x2619 = x2600 * x485
    x2620 = x2602 * x790
    x2621 = x2619 + x2620
    x2622 = x2600 * x494
    x2623 = x2602 * x797
    x2624 = x2622 + x2623
    x2625 = x2586 * x32
    x2626 = x1388 * x501
    x2627 = x2626 * x803
    x2628 = x2446 * x806
    x2629 = x2627 + x2628
    x2630 = x2626 * x816
    x2631 = x2626 * x810
    x2632 = x1403 * x1418
    x2633 = x1418 * x812
    x2634 = x2630 + x2631 + x2632 + x2633
    x2635 = x1407 * x868
    x2636 = x2598 * x831
    x2637 = x1416 * x779
    x2638 = x1419 * x2109
    x2639 = x2638 * x782
    x2640 = x2440 * x841 + x2522 * x831
    x2641 = x217 * x2640
    x2642 = x2305 * x847 + x2306 * x841 + x2569 * x831
    x2643 = x2642 * x275
    x2644 = x2458 * x847 + x2460 * x841 + x2555 * x831
    x2645 = x250 * x2644
    x2646 = x2449 * x847 + x2451 * x841 + x2544 * x831
    x2647 = x244 * x2646
    x2648 = x2585 * x830 + x2586 * x834 + x2590 * x824
    x2649 = x2626 * x857
    x2650 = x2446 * x858
    x2651 = x2649 + x2650
    x2652 = x2589 * x32
    x2653 = x2613 * x834
    x2654 = x2585 * x32
    x2655 = x1408 + x1417 + x1423
    x2656 = x2109 * x782
    x2657 = x2635 + x2637 + x2639
    x2658 = x2109 * x790
    x2659 = x2333 * x874
    x2660 = x1416 * x485
    x2661 = x2638 * x790
    x2662 = x2659 + x2660 + x2661
    x2663 = x1353 * x32
    x2664 = x293 * x844
    x2665 = x2322 * x844
    x2666 = x1416 * x494
    x2667 = x1419 * x1433
    x2668 = x2665 + x2666 + x2667
    x2669 = x1407 * x898
    x2670 = x1407 * x890
    x2671 = x2626 * x893
    x2672 = x2626 * x892
    x2673 = x1419 * x896
    x2674 = x1419 * x1436
    x2675 = x2669 + x2670 + x2671 + x2672 + x2673 + x2674
    x2676 = x1466 * x62
    x2677 = x2676 * x909
    x2678 = x1406 * x973
    x2679 = x2598 * x921
    x2680 = x1447 * x478
    x2681 = x2680 * x925
    x2682 = x1449 * x975
    x2683 = x2440 * x930 + x2522 * x921
    x2684 = x217 * x2683
    x2685 = x2449 * x934 + x2451 * x930 + x2544 * x921
    x2686 = x244 * x2685
    x2687 = x622 * x909
    x2688 = x2304 * x2687 + x2305 * x934 + x2306 * x930 + x2569 * x921
    x2689 = x2688 * x275
    x2690 = x2291 * x2687 + x2458 * x934 + x2460 * x930 + x2555 * x921
    x2691 = x250 * x2690
    x2692 = x1353 * x934 + x2592 * x908 + x2594 * x924 + x2596 * x919
    x2693 = x2613 * x929
    x2694 = x1388 * x929
    x2695 = x223 * x2694
    x2696 = x230 * x2446
    x2697 = x2696 * x945
    x2698 = x2695 + x2697
    x2699 = x1406 * x953
    x2700 = x2699 * x607
    x2701 = x263 * x2694
    x2702 = x1419 * x957
    x2703 = x2700 + x2701 + x2702
    x2704 = x1351 * x32
    x2705 = x2704 * x909
    x2706 = x2593 * x32
    x2707 = x2595 * x32
    x2708 = x1441 + x1442 + x1448 + x1450
    x2709 = x2677 + x2678 + x2681 + x2682
    x2710 = x2591 * x32
    x2711 = x1439 * x622
    x2712 = x2711 * x984
    x2713 = x2699 * x597
    x2714 = x2335 * x987
    x2715 = x1449 * x990
    x2716 = x2712 + x2713 + x2714 + x2715
    x2717 = x1009 * x2711
    x2718 = x2711 * x998
    x2719 = x1002 * x1406
    x2720 = x1001 * x1406
    x2721 = x1015 * x1447
    x2722 = x1004 * x1447
    x2723 = x1018 * x1449
    x2724 = x1007 * x1449
    x2725 = x2717 + x2718 + x2719 + x2720 + x2721 + x2722 + x2723 + x2724
    x2726 = x1099 * x2490
    x2727 = x1028 * x2676
    x2728 = x1100 * x1407
    x2729 = x1037 * x2598
    x2730 = x1041 * x2680
    x2731 = x1102 * x2495
    x2732 = x1037 * x2522 + x1046 * x2440
    x2733 = x217 * x2732
    x2734 = x1037 * x2544 + x1046 * x2451 + x1050 * x2449
    x2735 = x244 * x2734
    x2736 = x1028 * x622
    x2737 = x1037 * x2555 + x1046 * x2460 + x1050 * x2458 + x2291 * x2736
    x2738 = x250 * x2737
    x2739 = x1024 * x2469 + x1037 * x2569 + x1046 * x2306 + x1050 * x2305 + x2304 * x2736
    x2740 = x2739 * x275
    x2741 = x1023 * x2597 + x1028 * x2591 + x1036 * x2595 + x1045 * x2593 + x1049 * x2589
    x2742 = x1045 * x2613
    x2743 = x1045 * x1388
    x2744 = x223 * x2743
    x2745 = x1061 * x2696
    x2746 = x2744 + x2745
    x2747 = x1068 * x2297
    x2748 = x263 * x2743
    x2749 = x1061 * x2300
    x2750 = x2747 + x2748 + x2749
    x2751 = x1049 * x293
    x2752 = x1076 * x2711
    x2753 = x1049 * x2322
    x2754 = x1078 * x2324
    x2755 = x1082 * x1449
    x2756 = x2752 + x2753 + x2754 + x2755
    x2757 = x1350 * x32
    x2758 = x1028 * x2704
    x2759 = x1471 + x1472 + x1473 + x1475 + x1476
    x2760 = x2478 * x32
    x2761 = x2726 + x2727 + x2728 + x2730 + x2731
    x2762 = x1128 * x2490
    x2763 = x1117 * x2711
    x2764 = x1120 * x1407
    x2765 = x1123 * x1447
    x2766 = x1127 * x1449
    x2767 = (
        x1114 * x2490
        + x1116 * x2711
        + x1134 * x1406
        + x1137 * x1388
        + x1140 * x1418
        + x2762
        + x2763
        + x2764
        + x2765
        + x2766
    )
    x2768 = x166 * x2613
    x2769 = x293 * x296
    x2770 = x1348 * x32
    x2771 = x1352 * x32
    x2772 = x1354 * x32
    x2773 = x1356 * x32
    x2774 = x1358 * x32
    x2775 = x1201 * x293
    x2776 = x1297 * x654
    x2777 = x2259 * x2435
    x2778 = x1302 * x971
    x2779 = x1443 * x2439
    x2780 = sigma_kin_v_6_3 * x2274
    x2781 = sigma_kin_v_7_3 * x2276
    x2782 = sigma_kin_v_6_6 * x2780 + sigma_kin_v_7_6 * x2781
    x2783 = sigma_kin_v_5_3 * x2288
    x2784 = sigma_kin_v_5_5 * x2783 + sigma_kin_v_6_5 * x2780 + sigma_kin_v_7_5 * x2781
    x2785 = x1618 * x76
    x2786 = sigma_kin_v_5_4 * x2783 + sigma_kin_v_6_4 * x2780 + sigma_kin_v_7_4 * x2781 + x2294 * x2785
    x2787 = dq_i3 * x743
    x2788 = x1622 * x654
    x2789 = x1623 * x76
    x2790 = x124 * x1624
    x2791 = x1625 * x173
    x2792 = x155 * x1626
    x2793 = dq_i3 * dq_j2
    x2794 = sigma_kin_v_7_2 * x2793
    x2795 = ddq_i3 * dq_j2
    x2796 = x2354 * x521
    x2797 = dq_j2 * x639
    x2798 = dq_j2 * x601
    x2799 = dq_j2 * x492
    x2800 = dq_j2 * x669
    x2801 = x1386 * x2569
    x2802 = x1637 * x780
    x2803 = x149 * x755
    x2804 = x2803 * x782
    x2805 = x1663 * x2804
    x2806 = x2306 * x773 + x2569 * x776
    x2807 = x2306 * x751
    x2808 = x149 * x2569
    x2809 = x2808 * x755
    x2810 = x2807 * x779 + x2809 * x782
    x2811 = x32 * x411
    x2812 = x2807 * x494 + x2809 * x797
    x2813 = x2306 * x501
    x2814 = x2808 * x806 + x2813 * x803
    x2815 = x2807 * x485 + x2809 * x790
    x2816 = x1403 * x2569 + x2569 * x812 + x2813 * x810 + x2813 * x816
    x2817 = dq_i6 * x2793
    x2818 = x1655 * x868
    x2819 = dq_j2 * x411
    x2820 = x2109 * x2570
    x2821 = x1414 * x2820
    x2822 = x1661 * x779
    x2823 = x1664 * x2656
    x2824 = x2808 * x858 + x2813 * x857
    x2825 = x121 * x2305
    x2826 = x2306 * x834
    x2827 = x1422 * x2570 + x2825 * x863 + x2826 * x772
    x2828 = x2820 * x782 + x2825 * x868 + x2826 * x779
    x2829 = x2305 * x293
    x2830 = x1433 * x2570 + x2826 * x494 + x2829 * x844
    x2831 = x2825 * x347
    x2832 = x2820 * x790 + x2826 * x485 + x2831 * x874
    x2833 = x1436 * x2570 + x2570 * x896 + x2813 * x892 + x2813 * x893 + x2825 * x890 + x2825 * x898
    x2834 = dq_i5 * x2793
    x2835 = x1705 * x62
    x2836 = x2835 * x909
    x2837 = x1654 * x973
    x2838 = x208 * x2570
    x2839 = x2819 * x2838
    x2840 = x1686 * x974
    x2841 = x1688 * x975
    x2842 = x2306 * x929
    x2843 = x223 * x2842 + x2808 * x946
    x2844 = x2305 * x953
    x2845 = x2570 * x957 + x263 * x2842 + x2844 * x607
    x2846 = x2304 * x907
    x2847 = x2306 * x300
    x2848 = x2569 * x307
    x2849 = x2305 * x965 + x2846 * x909 + x2847 * x967 + x2848 * x969
    x2850 = x1704 * x705
    x2851 = x2847 * x478
    x2852 = x2305 * x973 + x2848 * x975 + x2850 * x909 + x2851 * x925
    x2853 = x2304 * x622
    x2854 = x2847 * x351
    x2855 = x2844 * x597 + x2848 * x990 + x2853 * x984 + x2854 * x987
    x2856 = dq_i4 * (
        x1001 * x2305
        + x1002 * x2305
        + x1004 * x2847
        + x1007 * x2848
        + x1009 * x2853
        + x1015 * x2847
        + x1018 * x2848
        + x2853 * x998
    )
    x2857 = x1289 * x1710
    x2858 = x1303 * x1704 * x283
    x2859 = x1292 * x1714
    x2860 = x2137 * x2570
    x2861 = x1293 * x1716
    x2862 = x1294 * x1751
    x2863 = x166 * x2306
    x2864 = x1268 * x2863 + x1269 * x2569
    x2865 = x255 * x2825
    x2866 = x2570 * x266
    x2867 = x1270 * x2865 + x1272 * x2863 + x1274 * x2866
    x2868 = x2847 * x299
    x2869 = x2848 * x305
    x2870 = x1277 * x2304 + x1278 * x2868 + x1280 * x2869 + x2829 * x296
    x2871 = x1289 * x1345 + x1292 * x2471 + x1293 * x2472 + x1294 * x2473 + x2261 * x2304
    x2872 = sigma_kin_v_4_3 * x2386 * x80 + x1283 * x1345 + x1287 * x2848 + x224 * x2854 + x2831 * x295
    x2873 = (
        x131 * x2305
        + x1345 * x53
        + x1345 * x60
        + x135 * x2305
        + x180 * x2306
        + x184 * x2306
        + x186 * x2473
        + x201 * x2473
        + x2304 * x82
        + x2304 * x87
    )
    x2874 = x1710 * x654
    x2875 = x1099 * x2874
    x2876 = x1028 * x2835
    x2877 = x1100 * x1655
    x2878 = x1101 * x1686
    x2879 = x1104 * x1688
    x2880 = x1045 * x2306
    x2881 = x1103 * x2848
    x2882 = x1117 * x2853 + x1120 * x2825 + x1123 * x2847 + x1127 * x2848 + x1128 * x2469
    x2883 = x1254 * x2874
    x2884 = x1256 * x632
    x2885 = x1712 * x2884
    x2886 = x1259 * x1714
    x2887 = x1261 * x1716
    x2888 = x1263 * x1751
    x2889 = x2407 * x771
    x2890 = x1542 * x2306
    x2891 = x1546 * x2569
    x2892 = x2410 * x2825
    x2893 = x1553 * x2306
    x2894 = x147 * x2570
    x2895 = sigma_kin_v_4_3 * x2413
    x2896 = x120 * x2305
    x2897 = x2407 * x2847
    x2898 = x2389 * x2469
    x2899 = x2396 * x2471
    x2900 = x2399 * x2472
    x2901 = x2402 * x2473
    x2902 = x1345 * x40
    x2903 = x1712 * x2426
    x2904 = x1243 * x2470 + x1254 * x2469 + x1259 * x2471 + x1261 * x2472 + x1263 * x2473
    x2905 = x112 * x2777
    x2906 = x1766 * x95
    x2907 = x1768 * x2906
    x2908 = 4 * x2266
    x2909 = sigma_kin_v_6_4 * x2274
    x2910 = sigma_kin_v_7_4 * x2276
    x2911 = x1776 * (sigma_kin_v_6_6 * x2909 + sigma_kin_v_7_6 * x2910)
    x2912 = x1779 * (sigma_kin_v_6_5 * x2909 + sigma_kin_v_7_5 * x2910 + x1778 * x2288)
    x2913 = dq_i4 * x743
    x2914 = x1783 * x76
    x2915 = dq_i4 * dq_j2
    x2916 = sigma_kin_v_7_2 * x2915
    x2917 = ddq_i4 * dq_j2
    x2918 = dq_j2 * x605
    x2919 = dq_j2 * x500
    x2920 = dq_j2 * x642
    x2921 = dq_j2 * x415
    x2922 = x1799 * x780
    x2923 = x1824 * x2804
    x2924 = x2460 * x773 + x2555 * x776
    x2925 = x2460 * x751
    x2926 = x149 * x2555
    x2927 = x2926 * x755
    x2928 = x2925 * x779 + x2927 * x782
    x2929 = x32 * x415
    x2930 = x2925 * x485 + x2927 * x790
    x2931 = x2460 * x501
    x2932 = x2926 * x806 + x2931 * x803
    x2933 = x2925 * x494 + x2927 * x797
    x2934 = x1403 * x2555 + x2555 * x812 + x2931 * x810 + x2931 * x816
    x2935 = dq_i6 * x2915
    x2936 = x1817 * x868
    x2937 = x2109 * x2556
    x2938 = x1414 * x2937
    x2939 = x1822 * x779
    x2940 = x1825 * x2656
    x2941 = x2926 * x858 + x2931 * x857
    x2942 = x121 * x2458
    x2943 = x2460 * x834
    x2944 = x1422 * x2556 + x2942 * x863 + x2943 * x772
    x2945 = x2937 * x782 + x2942 * x868 + x2943 * x779
    x2946 = x2942 * x347
    x2947 = x2937 * x790 + x2943 * x485 + x2946 * x874
    x2948 = x2458 * x293
    x2949 = x1433 * x2556 + x2943 * x494 + x2948 * x844
    x2950 = dq_i5 * (x1436 * x2556 + x2556 * x896 + x2931 * x892 + x2931 * x893 + x2942 * x890 + x2942 * x898)
    x2951 = x1900 * x2342
    x2952 = x1292 * x1840
    x2953 = x2137 * x2556
    x2954 = x1293 * x1844
    x2955 = x1294 * x1906
    x2956 = x166 * x2460
    x2957 = x1268 * x2956 + x1269 * x2555
    x2958 = x255 * x2942
    x2959 = x2556 * x266
    x2960 = x1270 * x2958 + x1272 * x2956 + x1274 * x2959
    x2961 = x1292 * x2459 + x1293 * x2461 + x1294 * x2462 + x2261 * x2291
    x2962 = x2460 * x300
    x2963 = x2962 * x351
    x2964 = x2555 * x307
    x2965 = x1287 * x2964 + x1914 * x698 + x224 * x2963 + x2946 * x295
    x2966 = x2962 * x299
    x2967 = x2964 * x305
    x2968 = x1277 * x2291 + x1278 * x2966 + x1280 * x2967 + x2948 * x296
    x2969 = (
        x131 * x2458
        + x135 * x2458
        + x180 * x2460
        + x184 * x2460
        + x186 * x2462
        + x201 * x2462
        + x2291 * x82
        + x2291 * x87
    )
    x2970 = x1863 * x971
    x2971 = x1100 * x1817
    x2972 = x208 * x2556 * x2921
    x2973 = x1101 * x1867
    x2974 = x1104 * x1869
    x2975 = x1045 * x2460
    x2976 = x1062 * x2926 + x223 * x2975
    x2977 = x1061 * x2959 + x1068 * x2958 + x263 * x2975
    x2978 = x2291 * x907
    x2979 = x1028 * x2978 + x1092 * x2942 + x1093 * x2962 + x1095 * x2964
    x2980 = sigma_kin_v_4_4 * x1466
    x2981 = x1028 * x2980 + x1100 * x2942 + x1101 * x2962 + x1104 * x2964
    x2982 = x2291 * x622
    x2983 = x1049 * x2948 + x1076 * x2982 + x1078 * x2966 + x1082 * x2964
    x2984 = (
        x1116 * x2982
        + x1117 * x2982
        + x1120 * x2942
        + x1123 * x2962
        + x1127 * x2964
        + x1134 * x2458
        + x1137 * x2460
        + x1140 * x2555
    )
    x2985 = x1882 * x971
    x2986 = x1816 * x973
    x2987 = x1867 * x974
    x2988 = x1869 * x975
    x2989 = x2460 * x929
    x2990 = x223 * x2989 + x2926 * x946
    x2991 = x2458 * x953
    x2992 = x2556 * x957 + x263 * x2989 + x2991 * x607
    x2993 = x2458 * x965 + x2962 * x967 + x2964 * x969 + x2978 * x909
    x2994 = x2458 * x973 + x2962 * x974 + x2964 * x975 + x2980 * x909
    x2995 = x2963 * x987 + x2964 * x990 + x2982 * x984 + x2991 * x597
    x2996 = (
        x1001 * x2458
        + x1002 * x2458
        + x1004 * x2962
        + x1007 * x2964
        + x1009 * x2982
        + x1015 * x2962
        + x1018 * x2964
        + x2982 * x998
    )
    x2997 = x1900 * x2884
    x2998 = sigma_kin_v_4_4 * x1242
    x2999 = x1521 * x2998
    x3000 = x1259 * x1840
    x3001 = x1245 * x1840
    x3002 = x1261 * x1844
    x3003 = x1248 * x1844
    x3004 = x1263 * x1906
    x3005 = x1250 * x1906
    x3006 = x1181 * x2956 + x1184 * x2926
    x3007 = x1183 * x2959 + x1189 * x2958 + x1191 * x2956
    x3008 = x1230 * x1496
    x3009 = sigma_kin_v_4_4 * x3008 + x1154 * x2459 + x1161 * x2461 + x1234 * x2462
    x3010 = sigma_kin_v_4_4 * x1506
    x3011 = x1180 * x2963 + x1200 * x2946 + x1219 * x2964 + x3010 * x343
    x3012 = x1201 * x2948 + x1202 * x2966 + x1204 * x2967 + x289 * x3010
    x3013 = (
        x1243 * x2457
        + x1245 * x2459
        + x1248 * x2461
        + x1250 * x2462
        + x1259 * x2459
        + x1261 * x2461
        + x1263 * x2462
        + x1496 * x2998
    )
    x3014 = x1766 * x1920
    x3015 = x1928 * (x1926 * x2274 + x1927 * x2276)
    x3016 = dq_i5 * x743
    x3017 = dq_j2 * sigma_kin_v_7_2
    x3018 = dq_i5 * x3017
    x3019 = ddq_i5 * dq_j2
    x3020 = dq_j2 * x507
    x3021 = dq_j2 * x611
    x3022 = dq_j2 * x419
    x3023 = x1944 * x780
    x3024 = x1992 * x2804
    x3025 = x2451 * x773 + x2544 * x776
    x3026 = x2451 * x751
    x3027 = x149 * x2544
    x3028 = x3027 * x755
    x3029 = x3026 * x779 + x3028 * x782
    x3030 = x32 * x419
    x3031 = x3026 * x485 + x3028 * x790
    x3032 = x3026 * x494 + x3028 * x797
    x3033 = x2451 * x501
    x3034 = x3027 * x806 + x3033 * x803
    x3035 = x1403 * x2544 + x2544 * x812 + x3033 * x810 + x3033 * x816
    x3036 = dq_j2 * x1928
    x3037 = x1292 * x1963
    x3038 = x2137 * x2545
    x3039 = x1293 * x1966
    x3040 = x1294 * x2036
    x3041 = x166 * x2451
    x3042 = x1268 * x3041 + x1269 * x2544
    x3043 = x1292 * x2450 + x1293 * x2452 + x1294 * x2453
    x3044 = x121 * x2449
    x3045 = x3044 * x347
    x3046 = x2451 * x300
    x3047 = x3046 * x351
    x3048 = x2544 * x307
    x3049 = x1287 * x3048 + x224 * x3047 + x295 * x3045
    x3050 = x2449 * x293
    x3051 = x299 * x3046
    x3052 = x3048 * x305
    x3053 = x1278 * x3051 + x1280 * x3052 + x296 * x3050
    x3054 = x255 * x3044
    x3055 = x2545 * x266
    x3056 = x1270 * x3054 + x1272 * x3041 + x1274 * x3055
    x3057 = x131 * x2449 + x135 * x2449 + x180 * x2451 + x184 * x2451 + x186 * x2453 + x201 * x2453
    x3058 = x1100 * x1987
    x3059 = x208 * x2545 * x3022
    x3060 = x1101 * x1990
    x3061 = x1104 * x1993
    x3062 = x1045 * x2451
    x3063 = x1062 * x3027 + x223 * x3062
    x3064 = x1092 * x3044 + x1093 * x3046 + x1095 * x3048
    x3065 = x1100 * x3044 + x1101 * x3046 + x1104 * x3048
    x3066 = x1049 * x3050 + x1078 * x3051 + x1082 * x3048
    x3067 = x1061 * x3055 + x1068 * x3054 + x263 * x3062
    x3068 = x1120 * x3044 + x1123 * x3046 + x1127 * x3048 + x1134 * x2449 + x1137 * x2451 + x1140 * x2544
    x3069 = x1962 * x973
    x3070 = x1990 * x974
    x3071 = x1993 * x975
    x3072 = x2451 * x929
    x3073 = x223 * x3072 + x3027 * x946
    x3074 = x2449 * x965 + x3046 * x967 + x3048 * x969
    x3075 = x2449 * x973 + x3046 * x974 + x3048 * x975
    x3076 = x2449 * x953
    x3077 = x3047 * x987 + x3048 * x990 + x3076 * x597
    x3078 = x2545 * x957 + x263 * x3072 + x3076 * x607
    x3079 = x1001 * x2449 + x1002 * x2449 + x1004 * x3046 + x1007 * x3048 + x1015 * x3046 + x1018 * x3048
    x3080 = x1987 * x868
    x3081 = x2109 * x2545
    x3082 = x2019 * x779
    x3083 = x2021 * x2656
    x3084 = x3027 * x858 + x3033 * x857
    x3085 = x2451 * x834
    x3086 = x1422 * x2545 + x3044 * x863 + x3085 * x772
    x3087 = x3044 * x868 + x3081 * x782 + x3085 * x779
    x3088 = x3045 * x874 + x3081 * x790 + x3085 * x485
    x3089 = x1433 * x2545 + x3050 * x844 + x3085 * x494
    x3090 = x1436 * x2545 + x2545 * x896 + x3033 * x892 + x3033 * x893 + x3044 * x890 + x3044 * x898
    x3091 = x1259 * x1963
    x3092 = x1245 * x1963
    x3093 = x1261 * x1966
    x3094 = x1248 * x1966
    x3095 = x1263 * x2036
    x3096 = x1250 * x2036
    x3097 = x1181 * x3041 + x1184 * x3027
    x3098 = x1154 * x2450 + x1161 * x2452 + x1234 * x2453
    x3099 = x1180 * x3047 + x1200 * x3045 + x1219 * x3048
    x3100 = x1201 * x3050 + x1202 * x3051 + x1204 * x3052
    x3101 = x1183 * x3055 + x1189 * x3054 + x1191 * x3041
    x3102 = x1245 * x2450 + x1248 * x2452 + x1250 * x2453 + x1259 * x2450 + x1261 * x2452 + x1263 * x2453
    x3103 = x2905 * x95
    x3104 = dq_i6 * x743
    x3105 = dq_i6 * x3017
    x3106 = ddq_i6 * dq_j2
    x3107 = dq_j2 * x513
    x3108 = dq_j2 * x423
    x3109 = x2137 * x2523
    x3110 = x1293 * x2065
    x3111 = x1294 * x2139
    x3112 = x1293 * x2441 + x1294 * x2442
    x3113 = x32 * x423
    x3114 = x2440 * x300
    x3115 = x3114 * x351
    x3116 = x2522 * x307
    x3117 = x1287 * x3116 + x224 * x3115
    x3118 = x299 * x3114
    x3119 = x305 * x3116
    x3120 = x1278 * x3118 + x1280 * x3119
    x3121 = x166 * x2440
    x3122 = x2523 * x266
    x3123 = x1272 * x3121 + x1274 * x3122
    x3124 = x1268 * x3121 + x1269 * x2522
    x3125 = x180 * x2440 + x184 * x2440 + x186 * x2442 + x201 * x2442
    x3126 = x2523 * x3108
    x3127 = x208 * x3126
    x3128 = x1101 * x2085
    x3129 = x1104 * x2088
    x3130 = x1093 * x3114 + x1095 * x3116
    x3131 = x1101 * x3114 + x1104 * x3116
    x3132 = x1078 * x3118 + x1082 * x3116
    x3133 = x1045 * x2440
    x3134 = x1061 * x3122 + x263 * x3133
    x3135 = x149 * x2522
    x3136 = x1062 * x3135 + x223 * x3133
    x3137 = x1123 * x3114 + x1127 * x3116 + x1137 * x2440 + x1140 * x2522
    x3138 = x2085 * x974
    x3139 = x2088 * x975
    x3140 = x3114 * x967 + x3116 * x969
    x3141 = x3114 * x974 + x3116 * x975
    x3142 = x3115 * x987 + x3116 * x990
    x3143 = x2440 * x929
    x3144 = x2523 * x957 + x263 * x3143
    x3145 = x223 * x3143 + x3135 * x946
    x3146 = x1004 * x3114 + x1007 * x3116 + x1015 * x3114 + x1018 * x3116
    x3147 = x1412 * x2109
    x3148 = x1413 * x3147
    x3149 = x2112 * x779
    x3150 = x2114 * x2656
    x3151 = x2440 * x834
    x3152 = x1422 * x2523 + x3151 * x772
    x3153 = x2523 * x2656 + x3151 * x779
    x3154 = x2523 * x2658 + x3151 * x485
    x3155 = x1433 * x2523 + x3151 * x494
    x3156 = x2440 * x501
    x3157 = x3135 * x858 + x3156 * x857
    x3158 = x1436 * x2523 + x2523 * x896 + x3156 * x892 + x3156 * x893
    x3159 = x2064 * x780
    x3160 = x2087 * x2804
    x3161 = x2440 * x773 + x2522 * x776
    x3162 = x2440 * x751
    x3163 = x3135 * x755
    x3164 = x3162 * x779 + x3163 * x782
    x3165 = x3162 * x485 + x3163 * x790
    x3166 = x3162 * x494 + x3163 * x797
    x3167 = x3135 * x806 + x3156 * x803
    x3168 = x1403 * x2522 + x2522 * x812 + x3156 * x810 + x3156 * x816
    x3169 = x1261 * x2065
    x3170 = x1248 * x2065
    x3171 = x1263 * x2139
    x3172 = x1250 * x2139
    x3173 = x1161 * x2441 + x1234 * x2442
    x3174 = x1180 * x3115 + x1219 * x3116
    x3175 = x1202 * x3118 + x1204 * x3119
    x3176 = x1183 * x3122 + x1191 * x3121
    x3177 = x1181 * x3121 + x1184 * x3135
    x3178 = x1248 * x2441 + x1250 * x2442 + x1261 * x2441 + x1263 * x2442
    x3179 = x1266 * x2152
    x3180 = x1309 * x480
    x3181 = dq_i1 * x743
    x3182 = x1266 * x1419
    x3183 = x208 * x460 * x547
    x3184 = x460 * x544
    x3185 = x1419 * x153
    x3186 = x2179 * x2182
    x3187 = x2183 * x3185
    x3188 = x2187 * x230
    x3189 = 2 * ddq_j2
    x3190 = x2351 * x406
    x3191 = x137 * x404
    x3192 = x2192 * x2535
    x3193 = x137 * x3017
    x3194 = x2203 * x3017
    x3195 = x195 * x2192
    x3196 = 2 * x3195
    x3197 = x3017 * x568
    x3198 = x2501 * x399
    x3199 = x1035 * x3017
    x3200 = x3192 * x404
    x3201 = x3196 * x404
    x3202 = x3017 * x919
    x3203 = x3017 * x828
    x3204 = x3017 * x744
    x3205 = x2245 * x532
    x3206 = x3017 * x520
    x3207 = x1156 * x3017
    x3208 = sigma_pot_3_c * x332 - sigma_pot_3_s * x333
    x3209 = x110 * x3208
    x3210 = x1302 * x343
    x3211 = x142 * x1605
    x3212 = x159 * x3211
    x3213 = ddq_j7 * x490
    x3214 = dq_i7 * x3213
    x3215 = sigma_kin_v_7_1 * sigma_kin_v_7_7
    x3216 = x2264 * x3215
    x3217 = 8 * dq_j3
    x3218 = x1664 * x3217
    x3219 = sigma_kin_v_6_3 * x2269
    x3220 = sigma_kin_v_6_6 * x3219 + sigma_kin_v_7_6 * x3211
    x3221 = x217 * x3220
    x3222 = x1580 * x484
    x3223 = x300 * x3222
    x3224 = x1583 * x488
    x3225 = x307 * x3224
    x3226 = dq_i1 * (x3223 * x487 + x3225 * x456)
    x3227 = x1637 * x166
    x3228 = x1268 * x3227
    x3229 = x1269 * x1663
    x3230 = x3228 + x3229
    x3231 = dq_i6 * x3217
    x3232 = sigma_kin_v_5_3 * x2283
    x3233 = sigma_kin_v_5_5 * x3232 + sigma_kin_v_6_5 * x3219 + sigma_kin_v_7_5 * x3211
    x3234 = x244 * x3233
    x3235 = x1577 * x596
    x3236 = x121 * x3235
    x3237 = dq_i1 * (x3223 * x578 + x3225 * x575 + x3236 * x599)
    x3238 = sigma_kin_v_5_4 * x3232 + sigma_kin_v_6_4 * x3219 + sigma_kin_v_7_4 * x3211 + x1618 * x2290
    x3239 = x250 * x3238
    x3240 = dq_i1 * (x1576 * x637 + x3223 * x616 + x3225 * x613 + x3236 * x619)
    x3241 = x1655 * x255
    x3242 = x1270 * x3241
    x3243 = x1272 * x3227
    x3244 = x1664 * x266
    x3245 = x1274 * x3244
    x3246 = x3242 + x3243 + x3245
    x3247 = dq_i5 * x3217
    x3248 = ddq_i1 * dq_j3
    x3249 = x274 * x3248
    x3250 = x2307 * x281
    x3251 = dq_i1 * dq_i2
    x3252 = x3251 * (x1573 * x697 + x1576 * x690 + x3223 * x684 + x3225 * x681 + x3236 * x687)
    x3253 = x1622 * x52
    x3254 = x2789 * x80
    x3255 = x129 * x2790
    x3256 = x178 * x2791
    x3257 = x141 * x2792
    x3258 = x132 * x3255 + x137 * x3257 + x181 * x3256 + x3253 * x58 + x3254 * x83
    x3259 = 4 * ddq_i3 * dq_j3**2
    x3260 = sigma_kin_v_4_3 * x2320
    x3261 = x1654 * x293
    x3262 = x296 * x3261
    x3263 = x1686 * x299
    x3264 = x1278 * x3263
    x3265 = x1688 * x305
    x3266 = x1280 * x3265
    x3267 = x3260 + x3262 + x3264 + x3266
    x3268 = dq_i4 * x3217
    x3269 = x2857 + x2858 + x2859 + x2861 + x2862
    x3270 = dq_i2 * x3217
    x3271 = x1283 * x1710
    x3272 = sigma_kin_v_4_3 * x343
    x3273 = x284 * x3272
    x3274 = x1285 * x1655
    x3275 = x1286 * x1686
    x3276 = x1287 * x1688
    x3277 = x3271 + x3273 + x3274 + x3275 + x3276
    x3278 = 8 * x54
    x3279 = (
        x135 * x1654
        + x1605 * x1908
        + x1637 * x184
        + x1710 * x60
        + x1712 * x1899
        + x1744
        + x1745
        + x1747
        + x1749
        + x1752
    )
    x3280 = dq_i1 * dq_j3
    x3281 = sigma_kin_v_7_3 * x3280
    x3282 = dq_j3 * x671
    x3283 = x1370 * x404
    x3284 = x3283 * x552
    x3285 = x138 * x410
    x3286 = x189 * x3285
    x3287 = dq_j3 * x631
    x3288 = x397 * x552
    x3289 = x1370 * x3288
    x3290 = x3289 * x416
    x3291 = dq_j3 * x590
    x3292 = x3289 * x420
    x3293 = dq_j3 * x476
    x3294 = sigma_kin_v_7_3 * x535
    x3295 = dq_j3 * x12
    x3296 = x2248 * x552
    x3297 = x2369 * x410
    x3298 = dq_j3 * x515
    x3299 = x3248 * x760
    x3300 = x515 * x54
    x3301 = dq_i6 * x3280
    x3302 = x3248 * x848
    x3303 = dq_i5 * x3280
    x3304 = (
        ddq_i1 * x657
        + 2 * dq_i1 * dq_i3 * sigma_kin_v_3_1 * sigma_kin_v_3_3 * x334 * x42 * x49 * x654
        + 2 * dq_i1 * dq_i3 * sigma_kin_v_4_1 * sigma_kin_v_4_3 * x341 * x621 * x66 * x75 * x78
        + 2 * dq_i1 * dq_i3 * sigma_kin_v_5_1 * sigma_kin_v_5_3 * x118 * x121 * x123 * x127 * x346 * x580
        + 2 * dq_i1 * dq_i3 * sigma_kin_v_6_1 * sigma_kin_v_6_3 * x165 * x168 * x170 * x172 * x176 * x350 * x464
        + 2 * dq_i1 * dq_i3 * sigma_kin_v_7_1 * sigma_kin_v_7_3 * x140 * x143 * x146 * x148 * x150 * x154 * x354 * x434
        - x12 * x662
        - x476 * x649
        - x590 * x651
        - x631 * x653
        - x645 * x722
        - x663 * x667
        - x665 * x671
    )
    x3305 = 2 * ddq_j3
    x3306 = x1684 * x3298
    x3307 = x3248 * x939
    x3308 = sigma_kin_v_4_3 * x83
    x3309 = x1174 * x3248
    x3310 = dq_i2 * x3280
    x3311 = -x1023
    x3312 = x1572 * x42
    x3313 = x3311 * x3312
    x3314 = -x1027
    x3315 = x1575 * x75
    x3316 = x3314 * x3315
    x3317 = -x1031
    x3318 = x1133 * x1577 * x3317
    x3319 = -x1039
    x3320 = x1136 * x1580 * x3319
    x3321 = -x1035
    x3322 = x3321 * x488
    x3323 = x148 * x1533 * x3322
    x3324 = x1139 * x1583 * x3321
    x3325 = x3319 * x484
    x3326 = x3325 * x647
    x3327 = x154 * x3322
    x3328 = x149 * x3327
    x3329 = x1667 * x3317
    x3330 = x1067 * x118 * x1551
    x3331 = x1554 * x3327
    x3332 = x3314 * x652
    x3333 = x1556 * x983
    x3334 = x3317 * x596
    x3335 = x1560 * x3334
    x3336 = x1699 * x3319
    x3337 = x1077 * x1564 * x165
    x3338 = x143 * x3321
    x3339 = x1080 * x1566 * x3338
    x3340 = x3311 * x664
    x3341 = x1098 * x1588
    x3342 = sigma_kin_v_4_1 * sigma_kin_v_4_3
    x3343 = x3314 * x342
    x3344 = x2294 * x3343
    x3345 = x1578 * x3329
    x3346 = x1581 * x3336
    x3347 = x2405 * x3321
    x3348 = x1343 * x3311 * x335
    x3349 = x1337 * x3343
    x3350 = x1323 * x3338
    x3351 = x1157 * x1605
    x3352 = x159 * x3351
    x3353 = 4 * x3213
    x3354 = sigma_kin_v_7_2 * x1310
    x3355 = x208 * x3218
    x3356 = x1164 * x173
    x3357 = sigma_kin_v_6_3 * x3356
    x3358 = sigma_kin_v_6_6 * x3357 + sigma_kin_v_7_6 * x3351
    x3359 = x217 * x3358
    x3360 = dq_i2 * (x2440 * x3223 + x2522 * x3225)
    x3361 = x1181 * x3227
    x3362 = x149 * x1663
    x3363 = x1184 * x3362
    x3364 = x3361 + x3363
    x3365 = x1167 * x124
    x3366 = sigma_kin_v_5_3 * sigma_kin_v_5_5
    x3367 = sigma_kin_v_6_5 * x3357 + sigma_kin_v_7_5 * x3351 + x3365 * x3366
    x3368 = x244 * x3367
    x3369 = dq_i2 * (x2449 * x3236 + x2451 * x3223 + x2544 * x3225)
    x3370 = x2785 * x632
    x3371 = sigma_kin_v_5_3 * sigma_kin_v_5_4
    x3372 = sigma_kin_v_6_3 * sigma_kin_v_6_4
    x3373 = sigma_kin_v_7_4 * x3351 + x1149 * x3370 + x3356 * x3372 + x3365 * x3371
    x3374 = x250 * x3373
    x3375 = dq_i2 * (x1576 * x2982 + x2458 * x3236 + x2460 * x3223 + x2555 * x3225)
    x3376 = x1189 * x3241
    x3377 = x1191 * x3227
    x3378 = x1183 * x3244
    x3379 = x3376 + x3377 + x3378
    x3380 = x2474 * x281
    x3381 = x2789 * x632
    x3382 = x2790 * x591
    x3383 = x2791 * x477
    x3384 = x2792 * x479
    x3385 = x1147 * x2788 + x1149 * x3381 + x1153 * x3382 + x1156 * x3384 + x1160 * x3383
    x3386 = x1198 * x1680
    x3387 = x289 * x3386
    x3388 = x1201 * x3261
    x3389 = x1202 * x3263
    x3390 = x1204 * x3265
    x3391 = x3387 + x3388 + x3389 + x3390
    x3392 = x1711 + x1713 + x1715 + x1717 + x1718
    x3393 = 8 * x3280
    x3394 = x1213 * x2874
    x3395 = x3386 * x343
    x3396 = x1215 * x1655
    x3397 = x1216 * x1686
    x3398 = x1219 * x1688
    x3399 = x3394 + x3395 + x3396 + x3397 + x3398
    x3400 = x1241 * x1704
    x3401 = (
        x1239 * x2874
        + x1245 * x1714
        + x1248 * x1716
        + x1250 * x1751
        + x1521 * x3400
        + x2883
        + x2885
        + x2886
        + x2887
        + x2888
    )
    x3402 = dq_i2 * dq_j3
    x3403 = sigma_kin_v_7_3 * x3402
    x3404 = ddq_i2 * dq_j3
    x3405 = dq_j3 * x635
    x3406 = dq_j3 * x594
    x3407 = dq_j3 * x482
    x3408 = dq_j3 * x712
    x3409 = x405 * x54
    x3410 = dq_i6 * x3402
    x3411 = dq_j3 * x405
    x3412 = dq_i5 * x3402
    x3413 = (
        ddq_i2 * x2574
        + 2 * dq_i2 * dq_i3 * sigma_kin_v_3_2 * sigma_kin_v_3_3 * x334 * x42 * x49 * x654
        + 2 * dq_i2 * dq_i3 * sigma_kin_v_4_2 * sigma_kin_v_4_3 * x341 * x621 * x66 * x75 * x78
        + 2 * dq_i2 * dq_i3 * sigma_kin_v_5_2 * sigma_kin_v_5_3 * x118 * x121 * x123 * x127 * x346 * x580
        + 2 * dq_i2 * dq_i3 * sigma_kin_v_6_2 * sigma_kin_v_6_3 * x165 * x168 * x170 * x172 * x176 * x350 * x464
        + 2 * dq_i2 * dq_i3 * sigma_kin_v_7_2 * sigma_kin_v_7_3 * x140 * x143 * x146 * x148 * x150 * x154 * x354 * x434
        - x2570 * x683
        - x2571 * x482
        - x2572 * x594
        - x2573 * x635
        - x2575 * x671
        - x2576 * x712
        - x2577 * x666
    )
    x3414 = x2838 * x3411
    x3415 = x2825 * x3317
    x3416 = x2853 * x3314
    x3417 = x2847 * x3319
    x3418 = x2469 * x3311
    x3419 = x1578 * x3415
    x3420 = x1581 * x3417
    x3421 = x622 * x64
    x3422 = x111 * x2435
    x3423 = x1120 * x121
    x3424 = sigma_kin_v_7_3 * x159
    x3425 = x1037 * x3424
    x3426 = x1123 * x300
    x3427 = x1127 * x307
    x3428 = x1443 * x3355
    x3429 = sigma_kin_v_6_3 * sigma_kin_v_6_6
    x3430 = sigma_kin_v_7_3 * sigma_kin_v_7_6
    x3431 = x1037 * x3430 + x1046 * x3429
    x3432 = x217 * x3431
    x3433 = x1045 * x1637
    x3434 = x223 * x3433
    x3435 = x1062 * x3362
    x3436 = x3434 + x3435
    x3437 = sigma_kin_v_6_3 * sigma_kin_v_6_5
    x3438 = sigma_kin_v_7_3 * sigma_kin_v_7_5
    x3439 = x1037 * x3438 + x1046 * x3437 + x1050 * x3366
    x3440 = x244 * x3439
    x3441 = sigma_kin_v_7_3 * sigma_kin_v_7_4
    x3442 = x1037 * x3441 + x1046 * x3372 + x1050 * x3371 + x1618 * x2736
    x3443 = x250 * x3442
    x3444 = x1068 * x3241
    x3445 = x263 * x3433
    x3446 = x1061 * x3244
    x3447 = x3444 + x3445 + x3446
    x3448 = x136 * x3248
    x3449 = x2739 * x281
    x3450 = x2788 * x335
    x3451 = x1623 * x622
    x3452 = x121 * x1624
    x3453 = x1625 * x300
    x3454 = x1626 * x307
    x3455 = x1023 * x3450 + x1028 * x3451 + x1036 * x3454 + x1045 * x3453 + x1049 * x3452
    x3456 = x1680 * x622
    x3457 = x1076 * x3456
    x3458 = x1049 * x3261
    x3459 = x1078 * x3263
    x3460 = x1082 * x1688
    x3461 = x3457 + x3458 + x3459 + x3460
    x3462 = x1735 + x1736 + x1737 + x1738 + x1739
    x3463 = x2875 + x2876 + x2877 + x2878 + x2879
    x3464 = x389 * x407
    x3465 = x139 * x2192
    x3466 = x138 * x2505
    x3467 = -x2500 * x409 + x2501 * x409 + x2502 * x3464 + x2506 * x409 + x3465 * x417 + x3465 * x421 + x3466 * x409
    x3468 = sigma_kin_v_7_3 * x138 * x144 * x409
    x3469 = x1128 * x2874
    x3470 = x1114 * x2874
    x3471 = x1117 * x3456
    x3472 = x1116 * x3456
    x3473 = x1120 * x1655
    x3474 = x1134 * x1654
    x3475 = x1123 * x1686
    x3476 = x1137 * x1637
    x3477 = x1127 * x1688
    x3478 = x1140 * x1663
    x3479 = x3469 + x3470 + x3471 + x3472 + x3473 + x3474 + x3475 + x3476 + x3477 + x3478
    x3480 = x150 * x3430
    x3481 = sigma_kin_v_6_3 * x466
    x3482 = sigma_kin_v_7_3 * x468
    x3483 = sigma_kin_v_6_6 * x3481 + sigma_kin_v_7_6 * x3482
    x3484 = x173 * x3429
    x3485 = sigma_kin_v_7_6 * x1605
    x3486 = x3484 * x471 + x3485 * x473
    x3487 = x3484 * x478 + x3485 * x480
    x3488 = x3429 * x486 + x3430 * x490
    x3489 = x3429 * x495 + x3430 * x498
    x3490 = x3429 * x502 + x3430 * x505
    x3491 = x3429 * x508 + x3430 * x510
    x3492 = (
        ddq_i6 * x3483
        + 2 * dq_i3 * dq_i6 * sigma_kin_v_6_3 * sigma_kin_v_6_6 * x165 * x168 * x170 * x172 * x176 * x350 * x464
        + 2 * dq_i3 * dq_i6 * sigma_kin_v_7_3 * sigma_kin_v_7_6 * x140 * x143 * x146 * x148 * x150 * x154 * x354 * x434
        - x3480 * x463
        - x3486 * x476
        - x3487 * x482
        - x3488 * x492
        - x3489 * x500
        - x3490 * x507
        - x3491 * x513
    )
    x3493 = 2 * ddq_j6
    x3494 = x2369 * x529
    x3495 = 2 * x189
    x3496 = x188 + x2536 + x3495
    x3497 = x138 * x3496
    x3498 = x3497 * x521
    x3499 = x3498 * x397
    x3500 = 2 * x275
    x3501 = x150 * x3438
    x3502 = x3437 * x508 + x3438 * x510
    x3503 = sigma_kin_v_5_3 * x582
    x3504 = sigma_kin_v_5_5 * x3503 + sigma_kin_v_6_5 * x3481 + sigma_kin_v_7_5 * x3482
    x3505 = x124 * x3366
    x3506 = x173 * x3437
    x3507 = sigma_kin_v_7_5 * x1605
    x3508 = x3505 * x585 + x3506 * x471 + x3507 * x473
    x3509 = x3505 * x592 + x3506 * x478 + x3507 * x480
    x3510 = x3366 * x598 + x3437 * x486 + x3438 * x490
    x3511 = x3366 * x603 + x3437 * x495 + x3438 * x498
    x3512 = x3366 * x608 + x3437 * x502 + x3438 * x505
    x3513 = (
        ddq_i5 * x3504
        + 2 * dq_i3 * dq_i5 * sigma_kin_v_5_3 * sigma_kin_v_5_5 * x118 * x121 * x123 * x127 * x346 * x580
        + 2 * dq_i3 * dq_i5 * sigma_kin_v_6_3 * sigma_kin_v_6_5 * x165 * x168 * x170 * x172 * x176 * x350 * x464
        + 2 * dq_i3 * dq_i5 * sigma_kin_v_7_3 * sigma_kin_v_7_5 * x140 * x143 * x146 * x148 * x150 * x154 * x354 * x434
        - x3501 * x577
        - x3502 * x507
        - x3508 * x590
        - x3509 * x594
        - x3510 * x601
        - x3511 * x605
        - x3512 * x611
    )
    x3514 = 2 * ddq_j5
    x3515 = x150 * x3441
    x3516 = x3372 * x508 + x3441 * x510
    x3517 = x3371 * x608 + x3372 * x502 + x3441 * x505
    x3518 = sigma_kin_v_5_4 * x3503 + sigma_kin_v_6_4 * x3481 + sigma_kin_v_7_4 * x3482 + x1618 * x623
    x3519 = x124 * x3371
    x3520 = x173 * x3372
    x3521 = sigma_kin_v_7_4 * x1605
    x3522 = x1618 * x2561 + x3519 * x585 + x3520 * x471 + x3521 * x473
    x3523 = x3370 * x372 + x3519 * x592 + x3520 * x478 + x3521 * x480
    x3524 = x1618 * x622
    x3525 = x3371 * x598 + x3372 * x486 + x343 * x3524 + x3441 * x490
    x3526 = x1618 * x2566 + x3371 * x603 + x3372 * x495 + x3441 * x498
    x3527 = (
        ddq_i4 * x3518
        + 2 * dq_i3 * dq_i4 * sigma_kin_v_4_3 * sigma_kin_v_4_4 * x341 * x621 * x66 * x75 * x78
        + 2 * dq_i3 * dq_i4 * sigma_kin_v_5_3 * sigma_kin_v_5_4 * x118 * x121 * x123 * x127 * x346 * x580
        + 2 * dq_i3 * dq_i4 * sigma_kin_v_6_3 * sigma_kin_v_6_4 * x165 * x168 * x170 * x172 * x176 * x350 * x464
        + 2 * dq_i3 * dq_i4 * sigma_kin_v_7_3 * sigma_kin_v_7_4 * x140 * x143 * x146 * x148 * x150 * x154 * x354 * x434
        - x3515 * x615
        - x3516 * x500
        - x3517 * x605
        - x3522 * x631
        - x3523 * x635
        - x3525 * x639
        - x3526 * x642
    )
    x3528 = 2 * ddq_j4
    x3529 = x150 * x1626
    x3530 = x1625 * x501
    x3531 = x222 * x3530
    x3532 = x149 * x1626
    x3533 = x3452 * x606
    x3534 = x288 * x3451
    x3535 = x3453 * x493
    x3536 = x3454 * x496
    x3537 = x3424 * x747
    x3538 = x1637 * x751
    x3539 = x3538 * x485
    x3540 = x1663 * x2803
    x3541 = x3540 * x790
    x3542 = x2604 * x281
    x3543 = x3372 * x764 + x3441 * x747
    x3544 = x250 * x3543
    x3545 = x3437 * x764 + x3438 * x747
    x3546 = x244 * x3545
    x3547 = x3429 * x764 + x3430 * x747
    x3548 = x217 * x3547
    x3549 = x3531 * x750 + x3532 * x746
    x3550 = x1625 * x54
    x3551 = x1626 * x54
    x3552 = x1638 + x1639
    x3553 = x3532 * x54
    x3554 = x3553 * x755
    x3555 = x2802 + x2805
    x3556 = x3539 + x3541
    x3557 = x3538 * x494
    x3558 = x3540 * x797
    x3559 = x3557 + x3558
    x3560 = x3530 * x54
    x3561 = x1637 * x501
    x3562 = x3561 * x803
    x3563 = x3362 * x806
    x3564 = x3562 + x3563
    x3565 = x3561 * x816
    x3566 = x3561 * x810
    x3567 = x1403 * x1663
    x3568 = x1663 * x812
    x3569 = x3565 + x3566 + x3567 + x3568
    x3570 = x1655 * x875
    x3571 = x3424 * x831
    x3572 = x1661 * x485
    x3573 = x1664 * x2658
    x3574 = x3429 * x841 + x3430 * x831
    x3575 = x217 * x3574
    x3576 = x2642 * x281
    x3577 = x3371 * x847 + x3372 * x841 + x3441 * x831
    x3578 = x250 * x3577
    x3579 = x3366 * x847 + x3437 * x841 + x3438 * x831
    x3580 = x244 * x3579
    x3581 = x3529 * x830 + x3530 * x834 + x3533 * x824
    x3582 = x3561 * x857
    x3583 = x3362 * x858
    x3584 = x3582 + x3583
    x3585 = x3452 * x54
    x3586 = x3550 * x834
    x3587 = x3529 * x54
    x3588 = x1656 + x1662 + x1665
    x3589 = x2818 + x2822 + x2823
    x3590 = x3570 + x3572 + x3573
    x3591 = x1624 * x54
    x3592 = x3261 * x844
    x3593 = x1661 * x494
    x3594 = x1433 * x1664
    x3595 = x3592 + x3593 + x3594
    x3596 = x1655 * x898
    x3597 = x1655 * x890
    x3598 = x3561 * x893
    x3599 = x3561 * x892
    x3600 = x1664 * x896
    x3601 = x1436 * x1664
    x3602 = x3596 + x3597 + x3598 + x3599 + x3600 + x3601
    x3603 = x3456 * x984
    x3604 = x1654 * x953
    x3605 = x3604 * x597
    x3606 = x3424 * x921
    x3607 = x1686 * x988
    x3608 = x1688 * x990
    x3609 = x3429 * x930 + x3430 * x921
    x3610 = x217 * x3609
    x3611 = x3366 * x934 + x3437 * x930 + x3438 * x921
    x3612 = x244 * x3611
    x3613 = x2688 * x281
    x3614 = x1618 * x2687 + x3371 * x934 + x3372 * x930 + x3441 * x921
    x3615 = x250 * x3614
    x3616 = x1624 * x934 + x3534 * x908 + x3535 * x924 + x3536 * x919
    x3617 = x3550 * x929
    x3618 = x1637 * x929
    x3619 = x223 * x3618
    x3620 = x3362 * x946
    x3621 = x3619 + x3620
    x3622 = x3604 * x607
    x3623 = x263 * x3618
    x3624 = x1664 * x957
    x3625 = x3622 + x3623 + x3624
    x3626 = x1623 * x54
    x3627 = x3626 * x909
    x3628 = x3453 * x54
    x3629 = x3454 * x54
    x3630 = x1682 + x1683 + x1687 + x1689
    x3631 = x2836 + x2837 + x2840 + x2841
    x3632 = x3451 * x54
    x3633 = x3603 + x3605 + x3607 + x3608
    x3634 = x1009 * x3456
    x3635 = x3456 * x998
    x3636 = x1002 * x1654
    x3637 = x1001 * x1654
    x3638 = x1015 * x1686
    x3639 = x1004 * x1686
    x3640 = x1018 * x1688
    x3641 = x1007 * x1688
    x3642 = x3634 + x3635 + x3636 + x3637 + x3638 + x3639 + x3640 + x3641
    x3643 = x166 * x3550
    x3644 = x1622 * x54
    x3645 = x2789 * x54
    x3646 = x2790 * x54
    x3647 = x2791 * x54
    x3648 = x2792 * x54
    x3649 = x2788 * x54
    x3650 = x1045 * x3550
    x3651 = x1028 * x3626
    x3652 = x3208 * x3422
    x3653 = x1302 * x622
    x3654 = x1309 * x3353
    x3655 = dq_i4 * sigma_kin_v_7_4
    x3656 = sigma_kin_v_6_4 * x3223
    x3657 = sigma_kin_v_7_4 * x3225
    x3658 = x1776 * (sigma_kin_v_6_6 * x3656 + sigma_kin_v_7_6 * x3657)
    x3659 = x1779 * (sigma_kin_v_6_5 * x3656 + sigma_kin_v_7_5 * x3657 + x1778 * x3236)
    x3660 = x1783 * x622
    x3661 = dq_i4 * dq_j3
    x3662 = sigma_kin_v_7_3 * x3661
    x3663 = ddq_i4 * dq_j3
    x3664 = x3286 * x521
    x3665 = dq_j3 * x605
    x3666 = dq_j3 * x500
    x3667 = dq_j3 * x642
    x3668 = dq_j3 * x415
    x3669 = x1799 * x789
    x3670 = x2803 * x790
    x3671 = x1824 * x3670
    x3672 = x3372 * x773 + x3441 * x776
    x3673 = x2804 * x3441 + x3372 * x780
    x3674 = x3372 * x751
    x3675 = x2803 * x3441
    x3676 = x3674 * x485 + x3675 * x790
    x3677 = x415 * x54
    x3678 = x3372 * x501
    x3679 = x149 * x3441
    x3680 = x3678 * x803 + x3679 * x806
    x3681 = x3674 * x494 + x3675 * x797
    x3682 = x1403 * x3441 + x3441 * x812 + x3678 * x810 + x3678 * x816
    x3683 = dq_i6 * x3661
    x3684 = x1817 * x875
    x3685 = x3515 * x3668
    x3686 = x1822 * x485
    x3687 = x1825 * x2658
    x3688 = x3678 * x857 + x3679 * x858
    x3689 = x121 * x3371
    x3690 = x3372 * x834
    x3691 = x1422 * x3515 + x3689 * x863 + x3690 * x772
    x3692 = x2656 * x3515 + x3689 * x868 + x3690 * x779
    x3693 = x2658 * x3515 + x3689 * x875 + x3690 * x485
    x3694 = x293 * x3371
    x3695 = x1433 * x3515 + x3690 * x494 + x3694 * x844
    x3696 = dq_i5 * (x1436 * x3515 + x3515 * x896 + x3678 * x892 + x3678 * x893 + x3689 * x890 + x3689 * x898)
    x3697 = sigma_kin_v_4_4 * x284 * x343
    x3698 = x1285 * x1817
    x3699 = x2137 * x3515
    x3700 = x1286 * x1867
    x3701 = x1287 * x1869
    x3702 = x166 * x3372
    x3703 = x1268 * x3702 + x1269 * x3441
    x3704 = x255 * x3689
    x3705 = x266 * x3515
    x3706 = x1270 * x3704 + x1272 * x3702 + x1274 * x3705
    x3707 = x1292 * x3519 + x1293 * x3520 + x1294 * x3521 + x1618 * x2261
    x3708 = x1618 * x343
    x3709 = x300 * x3372
    x3710 = x307 * x3441
    x3711 = x1276 * x3708 + x1285 * x3689 + x1286 * x3709 + x1287 * x3710
    x3712 = x299 * x3709
    x3713 = x305 * x3710
    x3714 = x1277 * x1618 + x1278 * x3712 + x1280 * x3713 + x296 * x3694
    x3715 = (
        x131 * x3371
        + x135 * x3371
        + x1618 * x82
        + x1618 * x87
        + x180 * x3372
        + x184 * x3372
        + x186 * x3521
        + x201 * x3521
    )
    x3716 = x1214 * x1862
    x3717 = x1215 * x1817
    x3718 = x1216 * x1867
    x3719 = x1219 * x1869
    x3720 = x1181 * x3702 + x1184 * x3679
    x3721 = x1183 * x3705 + x1189 * x3704 + x1191 * x3702
    x3722 = x1154 * x3519 + x1161 * x3520 + x1230 * x2785 + x1234 * x3521
    x3723 = x1198 * x3708 + x1215 * x3689 + x1216 * x3709 + x1219 * x3710
    x3724 = x1199 * x1618 + x1201 * x3694 + x1202 * x3712 + x1204 * x3713
    x3725 = (
        x1242 * x2785
        + x1245 * x3519
        + x1248 * x3520
        + x1250 * x3521
        + x1259 * x3519
        + x1261 * x3520
        + x1263 * x3521
        + x2434 * x2785
    )
    x3726 = x1862 * x622
    x3727 = x3726 * x984
    x3728 = x1816 * x985
    x3729 = x1867 * x988
    x3730 = x1869 * x990
    x3731 = x3372 * x929
    x3732 = x223 * x3731 + x3679 * x946
    x3733 = x3371 * x953
    x3734 = x263 * x3731 + x3515 * x957 + x3733 * x607
    x3735 = x1618 * x909
    x3736 = x3371 * x965 + x3709 * x967 + x3710 * x969 + x3735 * x907
    x3737 = x3371 * x973 + x3709 * x974 + x3710 * x975 + x3735 * x971
    x3738 = x3524 * x984 + x3709 * x988 + x3710 * x990 + x3733 * x597
    x3739 = (
        x1001 * x3371
        + x1002 * x3371
        + x1004 * x3709
        + x1007 * x3710
        + x1009 * x3524
        + x1015 * x3709
        + x1018 * x3710
        + x3524 * x998
    )
    x3740 = x1117 * x3726
    x3741 = x1116 * x3726
    x3742 = x1816 * x3423
    x3743 = x1134 * x1816
    x3744 = x1799 * x3426
    x3745 = x1137 * x1799
    x3746 = x1824 * x3427
    x3747 = x1140 * x1824
    x3748 = x1045 * x3372
    x3749 = x1062 * x3679 + x223 * x3748
    x3750 = x1061 * x3705 + x1068 * x3704 + x263 * x3748
    x3751 = x1028 * x1618
    x3752 = x1092 * x3689 + x1093 * x3709 + x1095 * x3710 + x3751 * x907
    x3753 = x1100 * x3689 + x1101 * x3709 + x1104 * x3710 + x3751 * x971
    x3754 = x1049 * x3694 + x1076 * x3524 + x1078 * x3712 + x1082 * x3710
    x3755 = (
        x1116 * x3524
        + x1117 * x3524
        + x1134 * x3371
        + x1137 * x3372
        + x1140 * x3441
        + x3371 * x3423
        + x3372 * x3426
        + x3427 * x3441
    )
    x3756 = dq_i5 * sigma_kin_v_7_5
    x3757 = x1928 * (x1926 * x3223 + x1927 * x3225)
    x3758 = dq_j3 * sigma_kin_v_7_3
    x3759 = dq_i5 * x3758
    x3760 = ddq_i5 * dq_j3
    x3761 = dq_j3 * x507
    x3762 = dq_j3 * x611
    x3763 = dq_j3 * x419
    x3764 = x1944 * x789
    x3765 = x1992 * x3670
    x3766 = x3437 * x773 + x3438 * x776
    x3767 = x2804 * x3438 + x3437 * x780
    x3768 = x3437 * x751
    x3769 = x2803 * x3438
    x3770 = x3768 * x485 + x3769 * x790
    x3771 = x419 * x54
    x3772 = x3768 * x494 + x3769 * x797
    x3773 = x3437 * x501
    x3774 = x149 * x3438
    x3775 = x3773 * x803 + x3774 * x806
    x3776 = x1403 * x3438 + x3438 * x812 + x3773 * x810 + x3773 * x816
    x3777 = dq_j3 * x1928
    x3778 = x1285 * x1987
    x3779 = x2137 * x3501
    x3780 = x1286 * x1990
    x3781 = x1287 * x1993
    x3782 = x166 * x3437
    x3783 = x1268 * x3782 + x1269 * x3438
    x3784 = x1292 * x3505 + x1293 * x3506 + x1294 * x3507
    x3785 = x121 * x3366
    x3786 = x300 * x3437
    x3787 = x307 * x3438
    x3788 = x1285 * x3785 + x1286 * x3786 + x1287 * x3787
    x3789 = x293 * x3366
    x3790 = x299 * x3786
    x3791 = x305 * x3787
    x3792 = x1278 * x3790 + x1280 * x3791 + x296 * x3789
    x3793 = x255 * x3785
    x3794 = x266 * x3501
    x3795 = x1270 * x3793 + x1272 * x3782 + x1274 * x3794
    x3796 = x131 * x3366 + x135 * x3366 + x180 * x3437 + x184 * x3437 + x186 * x3507 + x201 * x3507
    x3797 = x1215 * x1987
    x3798 = x3501 * x3763
    x3799 = x1216 * x1990
    x3800 = x1219 * x1993
    x3801 = x1181 * x3782 + x1184 * x3774
    x3802 = x1154 * x3505 + x1161 * x3506 + x1234 * x3507
    x3803 = x1215 * x3785 + x1216 * x3786 + x1219 * x3787
    x3804 = x1201 * x3789 + x1202 * x3790 + x1204 * x3791
    x3805 = x1183 * x3794 + x1189 * x3793 + x1191 * x3782
    x3806 = x1245 * x3505 + x1248 * x3506 + x1250 * x3507 + x1259 * x3505 + x1261 * x3506 + x1263 * x3507
    x3807 = x1962 * x985
    x3808 = x1990 * x988
    x3809 = x1993 * x990
    x3810 = x3437 * x929
    x3811 = x223 * x3810 + x3774 * x946
    x3812 = x3366 * x965 + x3786 * x967 + x3787 * x969
    x3813 = x3366 * x973 + x3786 * x974 + x3787 * x975
    x3814 = x3366 * x985 + x3786 * x988 + x3787 * x990
    x3815 = x263 * x3810 + x3366 * x954 + x3501 * x957
    x3816 = x1001 * x3366 + x1002 * x3366 + x1004 * x3786 + x1007 * x3787 + x1015 * x3786 + x1018 * x3787
    x3817 = x1987 * x875
    x3818 = x2019 * x485
    x3819 = x2021 * x2658
    x3820 = x3773 * x857 + x3774 * x858
    x3821 = x3437 * x834
    x3822 = x1422 * x3501 + x3785 * x863 + x3821 * x772
    x3823 = x2656 * x3501 + x3785 * x868 + x3821 * x779
    x3824 = x2658 * x3501 + x3785 * x875 + x3821 * x485
    x3825 = x1433 * x3501 + x3789 * x844 + x3821 * x494
    x3826 = x1436 * x3501 + x3501 * x896 + x3773 * x892 + x3773 * x893 + x3785 * x890 + x3785 * x898
    x3827 = x1962 * x3423
    x3828 = x1134 * x1962
    x3829 = x1944 * x3426
    x3830 = x1137 * x1944
    x3831 = x1992 * x3427
    x3832 = x1140 * x1992
    x3833 = x1045 * x3437
    x3834 = x1062 * x3774 + x223 * x3833
    x3835 = x1092 * x3785 + x1093 * x3786 + x1095 * x3787
    x3836 = x1100 * x3785 + x1101 * x3786 + x1104 * x3787
    x3837 = x1049 * x3789 + x1078 * x3790 + x1082 * x3787
    x3838 = x1061 * x3794 + x1068 * x3793 + x263 * x3833
    x3839 = x1134 * x3366 + x1137 * x3437 + x1140 * x3438 + x3366 * x3423 + x3426 * x3437 + x3427 * x3438
    x3840 = x3652 * x95
    x3841 = sigma_kin_v_7_6 * sigma_kin_v_7_7
    x3842 = 4 * dq_i6 * x3841
    x3843 = dq_i6 * x3758
    x3844 = ddq_i6 * dq_j3
    x3845 = dq_j3 * x513
    x3846 = dq_j3 * x423
    x3847 = x2137 * x3480
    x3848 = x1286 * x2085
    x3849 = x1287 * x2088
    x3850 = x1293 * x3484 + x1294 * x3485
    x3851 = x300 * x3429
    x3852 = x307 * x3430
    x3853 = x1286 * x3851 + x1287 * x3852
    x3854 = x423 * x54
    x3855 = x299 * x3851
    x3856 = x305 * x3852
    x3857 = x1278 * x3855 + x1280 * x3856
    x3858 = x166 * x3429
    x3859 = x266 * x3480
    x3860 = x1272 * x3858 + x1274 * x3859
    x3861 = x1268 * x3858 + x1269 * x3430
    x3862 = x180 * x3429 + x184 * x3429 + x186 * x3485 + x201 * x3485
    x3863 = x3480 * x3846
    x3864 = x1216 * x2085
    x3865 = x1219 * x2088
    x3866 = x1161 * x3484 + x1234 * x3485
    x3867 = x1216 * x3851 + x1219 * x3852
    x3868 = x1202 * x3855 + x1204 * x3856
    x3869 = x1183 * x3859 + x1191 * x3858
    x3870 = x149 * x3430
    x3871 = x1181 * x3858 + x1184 * x3870
    x3872 = x1248 * x3484 + x1250 * x3485 + x1261 * x3484 + x1263 * x3485
    x3873 = x208 * x3863
    x3874 = x2085 * x988
    x3875 = x2088 * x990
    x3876 = x3851 * x967 + x3852 * x969
    x3877 = x3851 * x974 + x3852 * x975
    x3878 = x3851 * x988 + x3852 * x990
    x3879 = x3429 * x929
    x3880 = x263 * x3879 + x3480 * x957
    x3881 = x223 * x3879 + x3870 * x946
    x3882 = x1004 * x3851 + x1007 * x3852 + x1015 * x3851 + x1018 * x3852
    x3883 = x2112 * x485
    x3884 = x2114 * x2658
    x3885 = x3429 * x834
    x3886 = x1422 * x3480 + x3885 * x772
    x3887 = x2656 * x3480 + x3885 * x779
    x3888 = x2658 * x3480 + x3885 * x485
    x3889 = x1433 * x3480 + x3885 * x494
    x3890 = x3429 * x501
    x3891 = x3870 * x858 + x3890 * x857
    x3892 = x1436 * x3480 + x3480 * x896 + x3890 * x892 + x3890 * x893
    x3893 = x2064 * x789
    x3894 = x2087 * x3670
    x3895 = x3429 * x773 + x3430 * x776
    x3896 = x2804 * x3430 + x3429 * x780
    x3897 = x3429 * x751
    x3898 = x2803 * x3430
    x3899 = x3897 * x485 + x3898 * x790
    x3900 = x3897 * x494 + x3898 * x797
    x3901 = x3870 * x806 + x3890 * x803
    x3902 = x1403 * x3430 + x3430 * x812 + x3890 * x810 + x3890 * x816
    x3903 = x2064 * x3426
    x3904 = x1137 * x2064
    x3905 = x2087 * x3427
    x3906 = x1140 * x2087
    x3907 = x1093 * x3851 + x1095 * x3852
    x3908 = x1101 * x3851 + x1104 * x3852
    x3909 = x1078 * x3855 + x1082 * x3852
    x3910 = x1045 * x3429
    x3911 = x1061 * x3859 + x263 * x3910
    x3912 = x1062 * x3870 + x223 * x3910
    x3913 = x1137 * x3429 + x1140 * x3430 + x3426 * x3429 + x3427 * x3430
    x3914 = x1142 * x2152
    x3915 = dq_i7 * x490
    x3916 = x3181 * x3215
    x3917 = sigma_kin_v_7_2 * x2161
    x3918 = x1309 * x490
    x3919 = sigma_kin_v_7_4 * x1335
    x3920 = sigma_kin_v_7_5 * x1328
    x3921 = x1317 * x3841
    x3922 = x2168 * x427
    x3923 = x1142 * x1664
    x3924 = x148 * x2173
    x3925 = x153 * x1664
    x3926 = x2183 * x489
    x3927 = x2179 * x3926
    x3928 = x2183 * x497
    x3929 = x3281 * x406
    x3930 = x137 * x3758
    x3931 = x2192 * x3495 * x410
    x3932 = x2203 * x3758
    x3933 = x3196 * x410
    x3934 = x3758 * x568
    x3935 = x1156 * x3758
    x3936 = x3758 * x919
    x3937 = x3758 * x828
    x3938 = x3758 * x744
    x3939 = x3758 * x520
    x3940 = x1035 * x3758
    x3941 = sigma_pot_4_c * x285 - sigma_pot_4_s * x286
    x3942 = x1766 * x1769
    x3943 = x142 * x1772
    x3944 = x159 * x3943
    x3945 = ddq_j7 * x498
    x3946 = dq_i7 * x3945
    x3947 = 8 * dq_j4
    x3948 = x1825 * x3947
    x3949 = sigma_kin_v_6_4 * x2269
    x3950 = sigma_kin_v_6_6 * x3949 + sigma_kin_v_7_6 * x3943
    x3951 = x217 * x3950
    x3952 = x1564 * x493
    x3953 = x1566 * x496
    x3954 = dq_i1 * (x2070 * x3952 + x2072 * x3953)
    x3955 = x166 * x1799
    x3956 = x1268 * x3955
    x3957 = x1269 * x1824
    x3958 = x3956 + x3957
    x3959 = dq_i6 * x3947
    x3960 = sigma_kin_v_6_5 * x3949 + sigma_kin_v_7_5 * x3943 + x1778 * x2283
    x3961 = x244 * x3960
    x3962 = x1560 * x933
    x3963 = dq_i1 * (x1974 * x3952 + x1976 * x3953 + x3962 * x599)
    x3964 = ddq_i1 * dq_j4
    x3965 = x249 * x3964
    x3966 = x2292 * x281
    x3967 = x275 * x3238
    x3968 = x3251 * (x1557 * x690 + x3952 * x691 + x3953 * x692 + x3962 * x687)
    x3969 = dq_i1 * (x1557 * x652 + x1699 * x3952 + x1701 * x3953 + x3962 * x650)
    x3970 = x2914 * x80
    x3971 = x124 * x1784
    x3972 = x173 * x1785
    x3973 = x155 * x1786
    x3974 = x142 * x3973 + x213 * x3972 + x240 * x3971 + x3970 * x83
    x3975 = 4 * ddq_i4 * dq_j4**2
    x3976 = x1817 * x255
    x3977 = x1270 * x3976
    x3978 = x1272 * x3955
    x3979 = x1825 * x266
    x3980 = x1274 * x3979
    x3981 = x3977 + x3978 + x3980
    x3982 = dq_i5 * x3947
    x3983 = x2951 + x2952 + x2954 + x2955
    x3984 = dq_i2 * x3947
    x3985 = x3697 + x3698 + x3700 + x3701
    x3986 = dq_i3 * x3947
    x3987 = sigma_kin_v_4_4 * x2320
    x3988 = x1816 * x2769
    x3989 = x1279 * x1867
    x3990 = x1281 * x1869 + x3987 + x3988 + x3989
    x3991 = 8 * x61
    x3992 = x1898 + x1901 + x1902 + x1903 + x1904 + x1905 + x1907 + x1909
    x3993 = dq_i1 * dq_j4
    x3994 = sigma_kin_v_7_4 * x3993
    x3995 = dq_j4 * x671
    x3996 = x3283 * x555
    x3997 = dq_j4 * x663
    x3998 = x2356 * x555
    x3999 = x138 * x397
    x4000 = x2352 * x3999
    x4001 = x190 * x416
    x4002 = dq_j4 * x590
    x4003 = x397 * x555
    x4004 = x1370 * x420
    x4005 = x4003 * x4004
    x4006 = dq_j4 * x476
    x4007 = sigma_kin_v_7_4 * x535
    x4008 = dq_j4 * x12
    x4009 = x2248 * x555
    x4010 = x2368 * x397
    x4011 = x4010 * x521
    x4012 = x4011 * x416
    x4013 = dq_j4 * x515
    x4014 = x3964 * x762
    x4015 = x515 * x61
    x4016 = dq_i6 * x3993
    x4017 = (
        ddq_i1 * x625
        + 2 * dq_i1 * dq_i4 * sigma_kin_v_4_1 * sigma_kin_v_4_4 * x287 * x621 * x67 * x74 * x78
        + 2 * dq_i1 * dq_i4 * sigma_kin_v_5_1 * sigma_kin_v_5_4 * x119 * x120 * x123 * x127 * x292 * x580
        + 2 * dq_i1 * dq_i4 * sigma_kin_v_6_1 * sigma_kin_v_6_4 * x166 * x167 * x170 * x172 * x176 * x299 * x464
        + 2 * dq_i1 * dq_i4 * sigma_kin_v_7_1 * sigma_kin_v_7_4 * x140 * x144 * x145 * x148 * x150 * x154 * x306 * x434
        - x12 * x630
        - x476 * x618
        - x590 * x620
        - x614 * x722
        - x631 * x640
        - x634 * x671
        - x638 * x663
    )
    x4018 = x3964 * x850
    x4019 = x4013 * x614
    x4020 = x1172 * x3964
    x4021 = dq_i2 * x3993
    x4022 = x1053 * x3964
    x4023 = dq_i3 * x3993
    x4024 = x3964 * x941
    x4025 = x2435 * x3941
    x4026 = x2906 * x4025
    x4027 = x1157 * x1772
    x4028 = x159 * x4027
    x4029 = 4 * x3945
    x4030 = x208 * x3948
    x4031 = sigma_kin_v_6_4 * x3356
    x4032 = sigma_kin_v_6_6 * x4031 + sigma_kin_v_7_6 * x4027
    x4033 = x217 * x4032
    x4034 = dq_i2 * (x3114 * x3952 + x3116 * x3953)
    x4035 = x1181 * x3955
    x4036 = x149 * x1824
    x4037 = x1184 * x4036
    x4038 = x4035 + x4037
    x4039 = sigma_kin_v_6_5 * x4031 + sigma_kin_v_7_5 * x4027 + x1778 * x3365
    x4040 = x244 * x4039
    x4041 = dq_i2 * (x2449 * x3962 + x3046 * x3952 + x3048 * x3953)
    x4042 = x2463 * x281
    x4043 = x275 * x3373
    x4044 = dq_i2 * (x1557 * x2853 + x2305 * x3962 + x2847 * x3952 + x2848 * x3953)
    x4045 = x2914 * x632
    x4046 = x3971 * x591
    x4047 = x3972 * x477
    x4048 = x3973 * x479
    x4049 = x1149 * x4045 + x1153 * x4046 + x1156 * x4048 + x1160 * x4047
    x4050 = x1189 * x3976
    x4051 = x1191 * x3955
    x4052 = x1183 * x3979
    x4053 = x4050 + x4051 + x4052
    x4054 = x1839 + x1841 + x1845 + x1846
    x4055 = 8 * x3993
    x4056 = x3716 + x3717 + x3718 + x3719
    x4057 = x1199 * x1862
    x4058 = x1816 * x2775
    x4059 = x1203 * x1867
    x4060 = x1205 * x1869 + x4057 + x4058 + x4059
    x4061 = x2997 + x2999 + x3000 + x3001 + x3002 + x3003 + x3004 + x3005
    x4062 = dq_i2 * dq_j4
    x4063 = sigma_kin_v_7_4 * x4062
    x4064 = ddq_i2 * dq_j4
    x4065 = dq_j4 * x666
    x4066 = x3999 * x4001
    x4067 = dq_j4 * x594
    x4068 = dq_j4 * x482
    x4069 = dq_j4 * x712
    x4070 = dq_j4 * x1386
    x4071 = x405 * x61
    x4072 = dq_i6 * x4062
    x4073 = (
        ddq_i2 * x2560
        + 2 * dq_i2 * dq_i4 * sigma_kin_v_4_2 * sigma_kin_v_4_4 * x287 * x621 * x67 * x74 * x78
        + 2 * dq_i2 * dq_i4 * sigma_kin_v_5_2 * sigma_kin_v_5_4 * x119 * x120 * x123 * x127 * x292 * x580
        + 2 * dq_i2 * dq_i4 * sigma_kin_v_6_2 * sigma_kin_v_6_4 * x166 * x167 * x170 * x172 * x176 * x299 * x464
        + 2 * dq_i2 * dq_i4 * sigma_kin_v_7_2 * sigma_kin_v_7_4 * x140 * x144 * x145 * x148 * x150 * x154 * x306 * x434
        - x2556 * x683
        - x2557 * x482
        - x2558 * x594
        - x2562 * x671
        - x2563 * x712
        - x2565 * x666
        - x2567 * x635
    )
    x4074 = dq_j4 * x405
    x4075 = x2556 * x4074
    x4076 = dq_i3 * x4062
    x4077 = sigma_kin_v_7_4 * x1037
    x4078 = x159 * x4077
    x4079 = x1309 * x4029
    x4080 = dq_i3 * sigma_kin_v_7_3
    x4081 = sigma_kin_v_6_4 * x1046
    x4082 = sigma_kin_v_6_6 * x4081 + sigma_kin_v_7_6 * x4077
    x4083 = x217 * x4082
    x4084 = x3851 * x3952 + x3852 * x3953
    x4085 = x1045 * x1799
    x4086 = x223 * x4085
    x4087 = x1062 * x4036
    x4088 = x4086 + x4087
    x4089 = sigma_kin_v_6_5 * x4081 + sigma_kin_v_7_5 * x4077 + x1050 * x1778
    x4090 = x244 * x4089
    x4091 = x3366 * x3962 + x3786 * x3952 + x3787 * x3953
    x4092 = x2737 * x281
    x4093 = x275 * x3442
    x4094 = x121 * x1784
    x4095 = x1785 * x300
    x4096 = x1786 * x307
    x4097 = x1028 * x3660 + x1036 * x4096 + x1045 * x4095 + x1049 * x4094
    x4098 = x1068 * x3976
    x4099 = x263 * x4085
    x4100 = x1061 * x3979
    x4101 = x4098 + x4099 + x4100
    x4102 = x1864 + x1865 + x1868 + x1870
    x4103 = x2970 + x2971 + x2973 + x2974
    x4104 = x1076 * x3726
    x4105 = x1816 * x2751
    x4106 = x1079 * x1867
    x4107 = x1082 * x1869 + x4104 + x4105 + x4106
    x4108 = x3740 + x3741 + x3742 + x3743 + x3744 + x3745 + x3746 + x3747
    x4109 = dq_i3 * dq_j4
    x4110 = sigma_kin_v_7_4 * x4109
    x4111 = ddq_i3 * dq_j4
    x4112 = x4066 * x521
    x4113 = dq_j4 * x601
    x4114 = dq_j4 * x492
    x4115 = dq_j4 * x669
    x4116 = x411 * x61
    x4117 = dq_i6 * x4109
    x4118 = (
        ddq_i3 * x3518
        + 2 * dq_i3 * dq_i4 * sigma_kin_v_4_3 * sigma_kin_v_4_4 * x287 * x621 * x67 * x74 * x78
        + 2 * dq_i3 * dq_i4 * sigma_kin_v_5_3 * sigma_kin_v_5_4 * x119 * x120 * x123 * x127 * x292 * x580
        + 2 * dq_i3 * dq_i4 * sigma_kin_v_6_3 * sigma_kin_v_6_4 * x166 * x167 * x170 * x172 * x176 * x299 * x464
        + 2 * dq_i3 * dq_i4 * sigma_kin_v_7_3 * sigma_kin_v_7_4 * x140 * x144 * x145 * x148 * x150 * x154 * x306 * x434
        - x3515 * x646
        - x3516 * x492
        - x3517 * x601
        - x3522 * x663
        - x3523 * x666
        - x3525 * x669
        - x3526 * x639
    )
    x4119 = dq_j4 * x411
    x4120 = x3515 * x4119
    x4121 = sigma_kin_v_7_4 * x159
    x4122 = x4121 * x921
    x4123 = x164 * x300
    x4124 = x197 * x307
    x4125 = sigma_kin_v_6_4 * sigma_kin_v_6_6
    x4126 = sigma_kin_v_7_4 * sigma_kin_v_7_6
    x4127 = x4125 * x930 + x4126 * x921
    x4128 = x217 * x4127
    x4129 = x1799 * x929
    x4130 = x223 * x4129
    x4131 = x4036 * x946
    x4132 = x4130 + x4131
    x4133 = sigma_kin_v_6_4 * sigma_kin_v_6_5
    x4134 = sigma_kin_v_7_4 * sigma_kin_v_7_5
    x4135 = x1778 * x934 + x4133 * x930 + x4134 * x921
    x4136 = x244 * x4135
    x4137 = x2690 * x281
    x4138 = x275 * x3614
    x4139 = x288 * x3660
    x4140 = x4095 * x493
    x4141 = x4096 * x496
    x4142 = x1784 * x934 + x4139 * x908 + x4140 * x924 + x4141 * x919
    x4143 = x1816 * x954
    x4144 = x263 * x4129
    x4145 = x1825 * x957
    x4146 = x4143 + x4144 + x4145
    x4147 = x1883 + x1884 + x1885 + x1886
    x4148 = x2985 + x2986 + x2987 + x2988
    x4149 = x3727 + x3728 + x3729 + x3730
    x4150 = x2500 * x397
    x4151 = x414 * x516
    x4152 = x143 * x3464
    x4153 = x139 * x2503
    x4154 = x265 * x419
    x4155 = x3466 * x397
    x4156 = x2506 * x397
    x4157 = -x147 * x4150 + x147 * x4151 + x147 * x4155 + x147 * x4156 + x2221 * x4154 + x2226 * x4152 + x2226 * x4153
    x4158 = x2225 * x432
    x4159 = sigma_kin_v_7_4 * x146 * x2226
    x4160 = x1009 * x3726
    x4161 = x3726 * x998
    x4162 = x1002 * x1816
    x4163 = x1001 * x1816
    x4164 = x1015 * x1867
    x4165 = x1004 * x1867
    x4166 = x1018 * x1869
    x4167 = x1007 * x1869
    x4168 = x4160 + x4161 + x4162 + x4163 + x4164 + x4165 + x4166 + x4167
    x4169 = x150 * x4126
    x4170 = sigma_kin_v_6_4 * x466
    x4171 = sigma_kin_v_7_4 * x468
    x4172 = sigma_kin_v_6_6 * x4170 + sigma_kin_v_7_6 * x4171
    x4173 = x173 * x4125
    x4174 = sigma_kin_v_7_6 * x1772
    x4175 = x4173 * x471 + x4174 * x473
    x4176 = x4173 * x478 + x4174 * x480
    x4177 = x4125 * x486 + x4126 * x490
    x4178 = x4125 * x495 + x4126 * x498
    x4179 = x4125 * x502 + x4126 * x505
    x4180 = x4125 * x508 + x4126 * x510
    x4181 = (
        ddq_i6 * x4172
        + 2 * dq_i4 * dq_i6 * sigma_kin_v_6_4 * sigma_kin_v_6_6 * x166 * x167 * x170 * x172 * x176 * x299 * x464
        + 2 * dq_i4 * dq_i6 * sigma_kin_v_7_4 * sigma_kin_v_7_6 * x140 * x144 * x145 * x148 * x150 * x154 * x306 * x434
        - x4169 * x463
        - x4175 * x476
        - x4176 * x482
        - x4177 * x492
        - x4178 * x500
        - x4179 * x507
        - x4180 * x513
    )
    x4182 = x4011 * x417
    x4183 = 2 * x190
    x4184 = x187 + x188 + x189 + x192 + x193
    x4185 = x191 + x4183 + x4184
    x4186 = x138 * x4185
    x4187 = x4186 * x521
    x4188 = x397 * x4187
    x4189 = 2 * x250
    x4190 = x2542 + x551 + x554 + x563 + x571
    x4191 = x150 * x4134
    x4192 = x4133 * x508 + x4134 * x510
    x4193 = sigma_kin_v_6_5 * x4170 + sigma_kin_v_7_5 * x4171 + x1778 * x582
    x4194 = x124 * x1778
    x4195 = x173 * x4133
    x4196 = sigma_kin_v_7_5 * x1772
    x4197 = x4194 * x585 + x4195 * x471 + x4196 * x473
    x4198 = x4194 * x592 + x4195 * x478 + x4196 * x480
    x4199 = x1778 * x598 + x4133 * x486 + x4134 * x490
    x4200 = x1778 * x603 + x4133 * x495 + x4134 * x498
    x4201 = x1778 * x608 + x4133 * x502 + x4134 * x505
    x4202 = (
        ddq_i5 * x4193
        + 2 * dq_i4 * dq_i5 * sigma_kin_v_5_4 * sigma_kin_v_5_5 * x119 * x120 * x123 * x127 * x292 * x580
        + 2 * dq_i4 * dq_i5 * sigma_kin_v_6_4 * sigma_kin_v_6_5 * x166 * x167 * x170 * x172 * x176 * x299 * x464
        + 2 * dq_i4 * dq_i5 * sigma_kin_v_7_4 * sigma_kin_v_7_5 * x140 * x144 * x145 * x148 * x150 * x154 * x306 * x434
        - x4191 * x577
        - x4192 * x507
        - x4197 * x590
        - x4198 * x594
        - x4199 * x601
        - x4200 * x605
        - x4201 * x611
    )
    x4203 = x150 * x1786
    x4204 = x1785 * x501
    x4205 = x222 * x4204
    x4206 = x149 * x1786
    x4207 = x4094 * x606
    x4208 = x4121 * x747
    x4209 = x1799 * x796
    x4210 = x306 * x415
    x4211 = x2606 * x281
    x4212 = x275 * x3543
    x4213 = x4133 * x764 + x4134 * x747
    x4214 = x244 * x4213
    x4215 = x4125 * x764 + x4126 * x747
    x4216 = x217 * x4215
    x4217 = x4205 * x750 + x4206 * x746
    x4218 = x1785 * x61
    x4219 = x1786 * x61
    x4220 = x1800 + x1801
    x4221 = x4206 * x61
    x4222 = x4221 * x755
    x4223 = x2922 + x2923
    x4224 = x3669 + x3671
    x4225 = x2803 * x797
    x4226 = x1824 * x4225 + x4209
    x4227 = x4204 * x61
    x4228 = x1799 * x501
    x4229 = x4228 * x803
    x4230 = x4036 * x806
    x4231 = x4229 + x4230
    x4232 = x4228 * x816
    x4233 = x4228 * x810
    x4234 = x1403 * x1824
    x4235 = x1824 * x812
    x4236 = x4232 + x4233 + x4234 + x4235
    x4237 = x1816 * x2664
    x4238 = x4121 * x831
    x4239 = x1822 * x494
    x4240 = x4125 * x841 + x4126 * x831
    x4241 = x217 * x4240
    x4242 = x2644 * x281
    x4243 = x275 * x3577
    x4244 = x1778 * x847 + x4133 * x841 + x4134 * x831
    x4245 = x244 * x4244
    x4246 = x4203 * x830 + x4204 * x834 + x4207 * x824
    x4247 = x4228 * x857
    x4248 = x4036 * x858
    x4249 = x4247 + x4248
    x4250 = x4094 * x61
    x4251 = x4218 * x834
    x4252 = x4203 * x61
    x4253 = x1818 + x1823 + x1826
    x4254 = x2936 + x2939 + x2940
    x4255 = x3684 + x3686 + x3687
    x4256 = x1784 * x61
    x4257 = x1433 * x1825 + x4237 + x4239
    x4258 = x1817 * x898
    x4259 = x1817 * x890
    x4260 = x4228 * x893
    x4261 = x4228 * x892
    x4262 = x1825 * x896
    x4263 = x1436 * x1825
    x4264 = x4258 + x4259 + x4260 + x4261 + x4262 + x4263
    x4265 = x1869 * x4210
    x4266 = x305 * x4265
    x4267 = x166 * x4218
    x4268 = x2914 * x61
    x4269 = x3971 * x61
    x4270 = x3972 * x61
    x4271 = x3973 * x61
    x4272 = x1783 * x61
    x4273 = x4095 * x61
    x4274 = x4096 * x61
    x4275 = x1045 * x4218
    x4276 = x1028 * x4272
    x4277 = x3660 * x61
    x4278 = x4218 * x929
    x4279 = x4272 * x909
    x4280 = x113 * x1766
    x4281 = x1926 * x300
    x4282 = x1927 * x307
    x4283 = x1928 * (x3952 * x4281 + x3953 * x4282)
    x4284 = x1933 * x300
    x4285 = x1934 * x307
    x4286 = dq_j4 * sigma_kin_v_7_4
    x4287 = dq_i5 * x4286
    x4288 = ddq_i5 * dq_j4
    x4289 = dq_j4 * x507
    x4290 = dq_j4 * x611
    x4291 = dq_j4 * x419
    x4292 = x1797 * x4134
    x4293 = x1944 * x796
    x4294 = x1992 * x4225
    x4295 = x4133 * x773 + x4134 * x776
    x4296 = x2804 * x4134 + x4133 * x780
    x4297 = x3670 * x4134 + x4133 * x789
    x4298 = x4133 * x796 + x4134 * x4225
    x4299 = x419 * x61
    x4300 = x4133 * x501
    x4301 = x149 * x4134
    x4302 = x4300 * x803 + x4301 * x806
    x4303 = x1403 * x4134 + x4134 * x812 + x4300 * x810 + x4300 * x816
    x4304 = dq_j4 * x1928
    x4305 = x1962 * x2769
    x4306 = x2137 * x4191
    x4307 = x1279 * x1990
    x4308 = x1281 * x1993
    x4309 = x166 * x4133
    x4310 = x1268 * x4309 + x1269 * x4134
    x4311 = x1292 * x4194 + x1293 * x4195 + x1294 * x4196
    x4312 = x121 * x1778
    x4313 = x300 * x4133
    x4314 = x307 * x4134
    x4315 = x1285 * x4312 + x1286 * x4313 + x1287 * x4314
    x4316 = x1279 * x4313 + x1281 * x4314 + x1778 * x2769
    x4317 = x255 * x4312
    x4318 = x266 * x4191
    x4319 = x1270 * x4317 + x1272 * x4309 + x1274 * x4318
    x4320 = x131 * x1778 + x135 * x1778 + x180 * x4133 + x184 * x4133 + x186 * x4196 + x201 * x4196
    x4321 = x1962 * x2775
    x4322 = x4191 * x4291
    x4323 = x1203 * x1990
    x4324 = x1205 * x1993
    x4325 = x1181 * x4309 + x1184 * x4301
    x4326 = x1154 * x4194 + x1161 * x4195 + x1234 * x4196
    x4327 = x1215 * x4312 + x1216 * x4313 + x1219 * x4314
    x4328 = x1203 * x4313 + x1205 * x4314 + x1778 * x2775
    x4329 = x1183 * x4318 + x1189 * x4317 + x1191 * x4309
    x4330 = x1245 * x4194 + x1248 * x4195 + x1250 * x4196 + x1259 * x4194 + x1261 * x4195 + x1263 * x4196
    x4331 = x1962 * x2751
    x4332 = x1079 * x1990
    x4333 = x1082 * x1993
    x4334 = x1045 * x4133
    x4335 = x1062 * x4301 + x223 * x4334
    x4336 = x1092 * x4312 + x1093 * x4313 + x1095 * x4314
    x4337 = x1100 * x4312 + x1101 * x4313 + x1104 * x4314
    x4338 = x1079 * x4313 + x1082 * x4314 + x1778 * x2751
    x4339 = x1061 * x4318 + x1068 * x4317 + x263 * x4334
    x4340 = x1134 * x1778 + x1137 * x4133 + x1140 * x4134 + x1778 * x3423 + x3426 * x4133 + x3427 * x4134
    x4341 = x1962 * x2664
    x4342 = x2019 * x494
    x4343 = x1433 * x2021
    x4344 = x4300 * x857 + x4301 * x858
    x4345 = x4133 * x834
    x4346 = x1422 * x4191 + x4312 * x863 + x4345 * x772
    x4347 = x2656 * x4191 + x4312 * x868 + x4345 * x779
    x4348 = x2658 * x4191 + x4312 * x875 + x4345 * x485
    x4349 = x1433 * x4191 + x1778 * x2664 + x4345 * x494
    x4350 = x1436 * x4191 + x4191 * x896 + x4300 * x892 + x4300 * x893 + x4312 * x890 + x4312 * x898
    x4351 = x1002 * x1962
    x4352 = x1001 * x1962
    x4353 = x1015 * x1990
    x4354 = x1004 * x1990
    x4355 = x1018 * x1993
    x4356 = x1007 * x1993
    x4357 = x4133 * x929
    x4358 = x223 * x4357 + x4301 * x946
    x4359 = x1778 * x965 + x4313 * x967 + x4314 * x969
    x4360 = x1778 * x973 + x4313 * x974 + x4314 * x975
    x4361 = x1778 * x985 + x4313 * x988 + x4314 * x990
    x4362 = x1778 * x954 + x263 * x4357 + x4191 * x957
    x4363 = x1001 * x1778 + x1002 * x1778 + x1004 * x4313 + x1007 * x4314 + x1015 * x4313 + x1018 * x4314
    x4364 = 1.0 * x109
    x4365 = x1769 * x2046
    x4366 = x2054 * x300
    x4367 = x2055 * x307
    x4368 = dq_i6 * x4286
    x4369 = ddq_i6 * dq_j4
    x4370 = dq_j4 * x513
    x4371 = dq_j4 * x423
    x4372 = x2137 * x4169
    x4373 = x1279 * x2085
    x4374 = x1281 * x2088
    x4375 = x1293 * x4173 + x1294 * x4174
    x4376 = x300 * x4125
    x4377 = x307 * x4126
    x4378 = x1286 * x4376 + x1287 * x4377
    x4379 = x1279 * x4376 + x1281 * x4377
    x4380 = x423 * x61
    x4381 = x166 * x4125
    x4382 = x266 * x4169
    x4383 = x1272 * x4381 + x1274 * x4382
    x4384 = x1268 * x4381 + x1269 * x4126
    x4385 = x180 * x4125 + x184 * x4125 + x186 * x4174 + x201 * x4174
    x4386 = x4169 * x4371
    x4387 = x1203 * x2085
    x4388 = x1205 * x2088
    x4389 = x1161 * x4173 + x1234 * x4174
    x4390 = x1216 * x4376 + x1219 * x4377
    x4391 = x1203 * x4376 + x1205 * x4377
    x4392 = x1183 * x4382 + x1191 * x4381
    x4393 = x149 * x4126
    x4394 = x1181 * x4381 + x1184 * x4393
    x4395 = x1248 * x4173 + x1250 * x4174 + x1261 * x4173 + x1263 * x4174
    x4396 = x208 * x4386
    x4397 = x1079 * x2085
    x4398 = x1082 * x2088
    x4399 = x1093 * x4376 + x1095 * x4377
    x4400 = x1101 * x4376 + x1104 * x4377
    x4401 = x1079 * x4376 + x1082 * x4377
    x4402 = x1045 * x4125
    x4403 = x1061 * x4382 + x263 * x4402
    x4404 = x1062 * x4393 + x223 * x4402
    x4405 = x1137 * x4125 + x1140 * x4126 + x3426 * x4125 + x3427 * x4126
    x4406 = x2112 * x494
    x4407 = x1433 * x2114
    x4408 = x4125 * x834
    x4409 = x1422 * x4169 + x4408 * x772
    x4410 = x2656 * x4169 + x4408 * x779
    x4411 = x2658 * x4169 + x4408 * x485
    x4412 = x1433 * x4169 + x4408 * x494
    x4413 = x4125 * x501
    x4414 = x4393 * x858 + x4413 * x857
    x4415 = x1436 * x4169 + x4169 * x896 + x4413 * x892 + x4413 * x893
    x4416 = x1797 * x4126
    x4417 = x2064 * x796
    x4418 = x2087 * x4225
    x4419 = x4125 * x773 + x4126 * x776
    x4420 = x2804 * x4126 + x4125 * x780
    x4421 = x3670 * x4126 + x4125 * x789
    x4422 = x4125 * x796 + x4126 * x4225
    x4423 = x4393 * x806 + x4413 * x803
    x4424 = x1403 * x4126 + x4126 * x812 + x4413 * x810 + x4413 * x816
    x4425 = x1015 * x2085
    x4426 = x1004 * x2085
    x4427 = x1018 * x2088
    x4428 = x1007 * x2088
    x4429 = x4376 * x967 + x4377 * x969
    x4430 = x4376 * x974 + x4377 * x975
    x4431 = x4376 * x988 + x4377 * x990
    x4432 = x4125 * x929
    x4433 = x263 * x4432 + x4169 * x957
    x4434 = x223 * x4432 + x4393 * x946
    x4435 = x1004 * x4376 + x1007 * x4377 + x1015 * x4376 + x1018 * x4377
    x4436 = x1022 * x2152
    x4437 = dq_i7 * x498
    x4438 = x1309 * x498
    x4439 = sigma_kin_v_7_3 * x1342
    x4440 = ddq_j7 * x3922
    x4441 = x1022 * x1825
    x4442 = x153 * x1825
    x4443 = x2179 * x3928
    x4444 = x3994 * x406
    x4445 = x137 * x4286
    x4446 = x137 * x2221
    x4447 = x416 * x4183
    x4448 = x2203 * x4286
    x4449 = x195 * x2221
    x4450 = x416 * x4449
    x4451 = 2 * x137
    x4452 = x4286 * x568
    x4453 = x1156 * x4286
    x4454 = x2221 * x4447
    x4455 = x1035 * x4286
    x4456 = 2 * x4450
    x4457 = x4286 * x828
    x4458 = x4286 * x744
    x4459 = x4286 * x520
    x4460 = x4286 * x919
    x4461 = sigma_pot_5_c * x253 - sigma_pot_5_s * x254
    x4462 = x4461 * x88
    x4463 = x142 * x159
    x4464 = x1922 * x4463
    x4465 = ddq_j7 * x505
    x4466 = dq_i7 * x4465
    x4467 = x1311 * x166
    x4468 = 8 * dq_j5
    x4469 = x2021 * x4468
    x4470 = x155 * x1927
    x4471 = x142 * x4470 + x1926 * x2269
    x4472 = x217 * x4471
    x4473 = x1554 * x829
    x4474 = dq_i1 * (x1553 * x2121 + x4473 * x457)
    x4475 = x166 * x1944
    x4476 = x1268 * x4475
    x4477 = x1269 * x1992
    x4478 = x4476 + x4477
    x4479 = dq_i6 * x4468
    x4480 = ddq_i1 * dq_j5
    x4481 = x243 * x4480
    x4482 = x2285 * x281
    x4483 = x275 * x3233
    x4484 = x250 * x3960
    x4485 = x1551 * x606
    x4486 = x3251 * (x1553 * x685 + x4473 * x682 + x4485 * x688)
    x4487 = dq_i1 * (x1553 * x648 + x1667 * x4485 + x4473 * x645)
    x4488 = dq_i1 * (x1553 * x617 + x1828 * x4485 + x4473 * x614)
    x4489 = x124 * x1932
    x4490 = x173 * x1933
    x4491 = x155 * x1934
    x4492 = x142 * x4491 + x213 * x4490 + x240 * x4489
    x4493 = 4 * ddq_i5 * dq_j5**2
    x4494 = x3037 + x3039 + x3040
    x4495 = dq_i2 * x4468
    x4496 = x3778 + x3780 + x3781
    x4497 = dq_i3 * x4468
    x4498 = x4305 + x4307 + x4308
    x4499 = dq_i4 * x4468
    x4500 = x1271 * x1987
    x4501 = x1272 * x4475
    x4502 = x1275 * x2021
    x4503 = x4500 + x4501 + x4502
    x4504 = 8 * x114
    x4505 = x2032 + x2033 + x2034 + x2035 + x2037 + x2038
    x4506 = dq_i1 * dq_j5
    x4507 = sigma_kin_v_7_5 * x4506
    x4508 = dq_j5 * x671
    x4509 = x3283 * x558
    x4510 = dq_j5 * x663
    x4511 = x2356 * x558
    x4512 = dq_j5 * x631
    x4513 = x397 * x558
    x4514 = x1370 * x416
    x4515 = x4513 * x4514
    x4516 = x191 * x420
    x4517 = dq_j5 * x476
    x4518 = sigma_kin_v_7_5 * x535
    x4519 = dq_j5 * x12
    x4520 = x2248 * x558
    x4521 = x4011 * x420
    x4522 = dq_j5 * x515
    x4523 = x4480 * x765
    x4524 = x114 * x515
    x4525 = dq_i6 * x4506
    x4526 = (
        ddq_i1 * x584
        + 2 * dq_i1 * dq_i5 * sigma_kin_v_5_1 * sigma_kin_v_5_5 * x119 * x121 * x122 * x127 * x255 * x580
        + 2 * dq_i1 * dq_i5 * sigma_kin_v_6_1 * sigma_kin_v_6_5 * x166 * x168 * x169 * x172 * x176 * x261 * x464
        + 2 * dq_i1 * dq_i5 * sigma_kin_v_7_1 * sigma_kin_v_7_5 * x140 * x144 * x146 * x147 * x150 * x154 * x265 * x434
        - x12 * x589
        - x476 * x579
        - x576 * x722
        - x590 * x609
        - x593 * x671
        - x600 * x663
        - x604 * x631
    )
    x4527 = x4522 * x576
    x4528 = x1168 * x4480
    x4529 = dq_i2 * x4506
    x4530 = x208 * x4527
    x4531 = x1051 * x4480
    x4532 = dq_i3 * x4506
    x4533 = x4480 * x935
    x4534 = dq_i4 * x4506
    x4535 = x4480 * x852
    x4536 = x1301 * x2435
    x4537 = x1766 * x4462
    x4538 = x1158 * x1922
    x4539 = 4 * x4465
    x4540 = x208 * x4469
    x4541 = x1157 * x4470 + x1926 * x3356
    x4542 = x217 * x4541
    x4543 = dq_i2 * (x1553 * x3156 + x2523 * x4473)
    x4544 = x1181 * x4475
    x4545 = x149 * x1992
    x4546 = x1184 * x4545
    x4547 = x4544 + x4546
    x4548 = x2454 * x281
    x4549 = x275 * x3367
    x4550 = x250 * x4039
    x4551 = dq_i2 * (x1553 * x2813 + x2570 * x4473 + x2825 * x4485)
    x4552 = dq_i2 * (x1553 * x2931 + x2556 * x4473 + x2942 * x4485)
    x4553 = x4489 * x591
    x4554 = x4490 * x477
    x4555 = x4491 * x479
    x4556 = x1153 * x4553 + x1156 * x4555 + x1160 * x4554
    x4557 = x1964 + x1967 + x1968
    x4558 = 8 * x4506
    x4559 = x3797 + x3799 + x3800
    x4560 = x4321 + x4323 + x4324
    x4561 = x1190 * x1987
    x4562 = x1191 * x4475
    x4563 = x1192 * x2021
    x4564 = x4561 + x4562 + x4563
    x4565 = x3091 + x3092 + x3093 + x3094 + x3095 + x3096
    x4566 = dq_i2 * dq_j5
    x4567 = sigma_kin_v_7_5 * x4566
    x4568 = ddq_i2 * dq_j5
    x4569 = dq_j5 * x666
    x4570 = dq_j5 * x635
    x4571 = x3999 * x4516
    x4572 = dq_j5 * x482
    x4573 = dq_j5 * x712
    x4574 = dq_j5 * x1386
    x4575 = x114 * x405
    x4576 = dq_i6 * x4566
    x4577 = (
        ddq_i2 * x2548
        + 2 * dq_i2 * dq_i5 * sigma_kin_v_5_2 * sigma_kin_v_5_5 * x119 * x121 * x122 * x127 * x255 * x580
        + 2 * dq_i2 * dq_i5 * sigma_kin_v_6_2 * sigma_kin_v_6_5 * x166 * x168 * x169 * x172 * x176 * x261 * x464
        + 2 * dq_i2 * dq_i5 * sigma_kin_v_7_2 * sigma_kin_v_7_5 * x140 * x144 * x146 * x147 * x150 * x154 * x265 * x434
        - x2545 * x683
        - x2546 * x482
        - x2549 * x671
        - x2550 * x712
        - x2551 * x666
        - x2552 * x635
        - x2553 * x594
    )
    x4578 = dq_j5 * x405
    x4579 = x2545 * x4578
    x4580 = x208 * x4579
    x4581 = dq_i3 * x4566
    x4582 = dq_i4 * x4566
    x4583 = x1601 * x2435
    x4584 = sigma_kin_v_7_5 * x159
    x4585 = x1037 * x4584
    x4586 = x1309 * x4539
    x4587 = x1311 * x263
    x4588 = x1037 * x1927 + x1046 * x1926
    x4589 = x217 * x4588
    x4590 = x1553 * x3890 + x3480 * x4473
    x4591 = x1045 * x1944
    x4592 = x223 * x4591
    x4593 = x1062 * x4545
    x4594 = x4592 + x4593
    x4595 = x2734 * x281
    x4596 = x275 * x3439
    x4597 = x250 * x4089
    x4598 = x1553 * x3678 + x3515 * x4473 + x3689 * x4485
    x4599 = x121 * x1932
    x4600 = x1036 * x4285 + x1045 * x4284 + x1049 * x4599
    x4601 = x1988 + x1991 + x1994
    x4602 = x3058 + x3060 + x3061
    x4603 = x4331 + x4332 + x4333
    x4604 = x1069 * x1987
    x4605 = x263 * x4591
    x4606 = x1070 * x2021
    x4607 = x4604 + x4605 + x4606
    x4608 = x3827 + x3828 + x3829 + x3830 + x3831 + x3832
    x4609 = dq_i3 * dq_j5
    x4610 = sigma_kin_v_7_5 * x4609
    x4611 = ddq_i3 * dq_j5
    x4612 = dq_j5 * x639
    x4613 = x4571 * x521
    x4614 = dq_j5 * x492
    x4615 = dq_j5 * x669
    x4616 = x114 * x411
    x4617 = dq_i6 * x4609
    x4618 = (
        ddq_i3 * x3504
        + 2 * dq_i3 * dq_i5 * sigma_kin_v_5_3 * sigma_kin_v_5_5 * x119 * x121 * x122 * x127 * x255 * x580
        + 2 * dq_i3 * dq_i5 * sigma_kin_v_6_3 * sigma_kin_v_6_5 * x166 * x168 * x169 * x172 * x176 * x261 * x464
        + 2 * dq_i3 * dq_i5 * sigma_kin_v_7_3 * sigma_kin_v_7_5 * x140 * x144 * x146 * x147 * x150 * x154 * x265 * x434
        - x3501 * x646
        - x3502 * x492
        - x3508 * x663
        - x3509 * x666
        - x3510 * x669
        - x3511 * x639
        - x3512 * x601
    )
    x4619 = dq_j5 * x411
    x4620 = x3501 * x4619
    x4621 = x208 * x4620
    x4622 = dq_i4 * x4609
    x4623 = x2435 * x4280
    x4624 = x4584 * x921
    x4625 = x1926 * x930 + x1927 * x921
    x4626 = x217 * x4625
    x4627 = x1776 * (x1553 * x4413 + x4169 * x4473)
    x4628 = x1944 * x929
    x4629 = x223 * x4628
    x4630 = x4545 * x946
    x4631 = x4629 + x4630
    x4632 = x2685 * x281
    x4633 = x275 * x3611
    x4634 = x250 * x4135
    x4635 = x4284 * x493
    x4636 = x4285 * x496
    x4637 = x1932 * x934 + x4635 * x924 + x4636 * x919
    x4638 = x2005 + x2006 + x2007
    x4639 = x3069 + x3070 + x3071
    x4640 = x3807 + x3808 + x3809
    x4641 = x1962 * x954
    x4642 = x263 * x4628
    x4643 = x2021 * x957 + x4641 + x4642
    x4644 = x4351 + x4352 + x4353 + x4354 + x4355 + x4356
    x4645 = dq_j5 * sigma_kin_v_7_5
    x4646 = dq_i4 * x4645
    x4647 = ddq_i4 * dq_j5
    x4648 = dq_j5 * x500
    x4649 = dq_j5 * x642
    x4650 = dq_j5 * x415
    x4651 = x114 * x415
    x4652 = dq_j5 * x1776
    x4653 = (
        ddq_i4 * x4193
        + 2 * dq_i4 * dq_i5 * sigma_kin_v_5_4 * sigma_kin_v_5_5 * x119 * x121 * x122 * x127 * x255 * x580
        + 2 * dq_i4 * dq_i5 * sigma_kin_v_6_4 * sigma_kin_v_6_5 * x166 * x168 * x169 * x172 * x176 * x261 * x464
        + 2 * dq_i4 * dq_i5 * sigma_kin_v_7_4 * sigma_kin_v_7_5 * x140 * x144 * x146 * x147 * x150 * x154 * x265 * x434
        - x4191 * x615
        - x4192 * x500
        - x4197 * x631
        - x4198 * x635
        - x4199 * x639
        - x4200 * x642
        - x4201 * x605
    )
    x4654 = x4191 * x4650
    x4655 = x208 * x4654
    x4656 = x117 * x121
    x4657 = x4584 * x831
    x4658 = x164 * x501
    x4659 = x150 * x197
    x4660 = x1926 * x841 + x1927 * x831
    x4661 = x217 * x4660
    x4662 = x1944 * x501
    x4663 = x4662 * x857
    x4664 = x4545 * x858
    x4665 = x4663 + x4664
    x4666 = x2646 * x281
    x4667 = x275 * x3579
    x4668 = x250 * x4244
    x4669 = x4599 * x606
    x4670 = x1933 * x501
    x4671 = x150 * x1934
    x4672 = x4669 * x824 + x4670 * x834 + x4671 * x830
    x4673 = x2016 + x2020 + x2022
    x4674 = x3080 + x3082 + x3083
    x4675 = x3817 + x3818 + x3819
    x4676 = x4341 + x4342 + x4343
    x4677 = -x145 * x4150 + x145 * x4151 + x145 * x4155 + x145 * x4156 + x2221 * x4210 + x2234 * x4152 + x2234 * x4153
    x4678 = sigma_kin_v_7_5 * x148 * x2234
    x4679 = x1987 * x898
    x4680 = x1987 * x890
    x4681 = x4662 * x893
    x4682 = x4662 * x892
    x4683 = x2021 * x896
    x4684 = x1436 * x2021
    x4685 = x4679 + x4680 + x4681 + x4682 + x4683 + x4684
    x4686 = x150 * x1927
    x4687 = x1926 * x466 + x1927 * x468
    x4688 = x173 * x1926
    x4689 = x4470 * x473 + x4688 * x471
    x4690 = x4470 * x480 + x4688 * x478
    x4691 = x1926 * x486 + x1927 * x490
    x4692 = x1926 * x495 + x1927 * x498
    x4693 = x1926 * x502 + x1927 * x505
    x4694 = x1926 * x508 + x1927 * x510
    x4695 = (
        ddq_i6 * x4687
        + 2 * dq_i5 * dq_i6 * sigma_kin_v_6_5 * sigma_kin_v_6_6 * x166 * x168 * x169 * x172 * x176 * x261 * x464
        + 2 * dq_i5 * dq_i6 * sigma_kin_v_7_5 * sigma_kin_v_7_6 * x140 * x144 * x146 * x147 * x150 * x154 * x265 * x434
        - x463 * x4686
        - x4689 * x476
        - x4690 * x482
        - x4691 * x492
        - x4692 * x500
        - x4693 * x507
        - x4694 * x513
    )
    x4696 = x4011 * x421
    x4697 = 2 * x191
    x4698 = x190 + x4184 + x4697
    x4699 = x138 * x4698
    x4700 = x4699 * x521
    x4701 = x397 * x4700
    x4702 = 2 * x244
    x4703 = x222 * x4670
    x4704 = x149 * x1934
    x4705 = x4584 * x747
    x4706 = x4662 * x803
    x4707 = x4545 * x806
    x4708 = x2608 * x281
    x4709 = x275 * x3545
    x4710 = x250 * x4213
    x4711 = x1926 * x764 + x1927 * x747
    x4712 = x217 * x4711
    x4713 = x4703 * x750 + x4704 * x746
    x4714 = x114 * x1933
    x4715 = x114 * x1934
    x4716 = x1945 + x1946
    x4717 = x114 * x4704
    x4718 = x4717 * x755
    x4719 = x3023 + x3024
    x4720 = x3764 + x3765
    x4721 = x4293 + x4294
    x4722 = x114 * x4670
    x4723 = x4706 + x4707
    x4724 = x4662 * x816
    x4725 = x4662 * x810
    x4726 = x1403 * x1992
    x4727 = x1992 * x812
    x4728 = x4724 + x4725 + x4726 + x4727
    x4729 = x166 * x4714
    x4730 = x114 * x4489
    x4731 = x114 * x4490
    x4732 = x114 * x4491
    x4733 = x114 * x4599
    x4734 = x114 * x4284
    x4735 = x114 * x4285
    x4736 = x114 * x1932
    x4737 = x114 * x4671
    x4738 = x1045 * x4714
    x4739 = x4714 * x929
    x4740 = x4714 * x834
    x4741 = x113 * x2435 * x4461
    x4742 = x1311 * x501
    x4743 = x1313 * x149
    x4744 = x2054 * x501
    x4745 = x150 * x2055
    x4746 = dq_i6 * x4645
    x4747 = ddq_i6 * dq_j5
    x4748 = dq_j5 * x513
    x4749 = dq_j5 * x423
    x4750 = x2137 * x4686
    x4751 = x166 * x2064
    x4752 = x1272 * x4751
    x4753 = x1275 * x2114
    x4754 = x1293 * x4688 + x1294 * x4470
    x4755 = x1286 * x4281 + x1287 * x4282
    x4756 = x1279 * x4281 + x1281 * x4282
    x4757 = x166 * x1926
    x4758 = x1272 * x4757 + x1275 * x4686
    x4759 = x114 * x423
    x4760 = x1268 * x4757 + x1269 * x1927
    x4761 = x180 * x1926 + x184 * x1926 + x186 * x4470 + x201 * x4470
    x4762 = x4686 * x4749
    x4763 = x1191 * x4751
    x4764 = x1192 * x2114
    x4765 = x1161 * x4688 + x1234 * x4470
    x4766 = x1216 * x4281 + x1219 * x4282
    x4767 = x1203 * x4281 + x1205 * x4282
    x4768 = x1191 * x4757 + x1192 * x4686
    x4769 = x149 * x1927
    x4770 = x1181 * x4757 + x1184 * x4769
    x4771 = x1248 * x4688 + x1250 * x4470 + x1261 * x4688 + x1263 * x4470
    x4772 = x208 * x4762
    x4773 = x1045 * x2064
    x4774 = x263 * x4773
    x4775 = x1070 * x2114
    x4776 = x1093 * x4281 + x1095 * x4282
    x4777 = x1101 * x4281 + x1104 * x4282
    x4778 = x1079 * x4281 + x1082 * x4282
    x4779 = x1045 * x1926
    x4780 = x1070 * x4686 + x263 * x4779
    x4781 = x1062 * x4769 + x223 * x4779
    x4782 = x1137 * x1926 + x1140 * x1927 + x1926 * x3426 + x1927 * x3427
    x4783 = x2064 * x929
    x4784 = x263 * x4783
    x4785 = x2114 * x957
    x4786 = x4281 * x967 + x4282 * x969
    x4787 = x4281 * x974 + x4282 * x975
    x4788 = x4281 * x988 + x4282 * x990
    x4789 = x1926 * x929
    x4790 = x263 * x4789 + x4686 * x957
    x4791 = x223 * x4789 + x4769 * x946
    x4792 = x1004 * x4281 + x1007 * x4282 + x1015 * x4281 + x1018 * x4282
    x4793 = x1797 * x1927
    x4794 = x2064 * x501
    x4795 = x4794 * x803
    x4796 = x149 * x2087
    x4797 = x4796 * x806
    x4798 = x1926 * x773 + x1927 * x776
    x4799 = x1926 * x780 + x1927 * x2804
    x4800 = x1926 * x789 + x1927 * x3670
    x4801 = x1926 * x796 + x1927 * x4225
    x4802 = x1926 * x501
    x4803 = x4769 * x806 + x4802 * x803
    x4804 = x1403 * x1927 + x1927 * x812 + x4802 * x810 + x4802 * x816
    x4805 = x4794 * x893
    x4806 = x4794 * x892
    x4807 = x2114 * x896
    x4808 = x1436 * x2114
    x4809 = x1926 * x834
    x4810 = x1422 * x4686 + x4809 * x772
    x4811 = x2656 * x4686 + x4809 * x779
    x4812 = x2658 * x4686 + x4809 * x485
    x4813 = x1433 * x4686 + x4809 * x494
    x4814 = x4769 * x858 + x4802 * x857
    x4815 = x1436 * x4686 + x4686 * x896 + x4802 * x892 + x4802 * x893
    x4816 = x2152 * x906
    x4817 = dq_i7 * x505
    x4818 = x1309 * x505
    x4819 = x2021 * x906
    x4820 = x1412 * x266
    x4821 = x153 * x2021
    x4822 = x2179 * x4820 * x520
    x4823 = x406 * x4507
    x4824 = x137 * x4645
    x4825 = x420 * x4697
    x4826 = x2203 * x4645
    x4827 = x420 * x4449
    x4828 = x4645 * x568
    x4829 = x1156 * x4645
    x4830 = x2221 * x4825
    x4831 = x1035 * x4645
    x4832 = 2 * x4827
    x4833 = x4645 * x919
    x4834 = x4645 * x744
    x4835 = x4645 * x520
    x4836 = x4645 * x828
    x4837 = sigma_pot_6_c * x219 - sigma_pot_6_s * x220
    x4838 = x2048 * x4837
    x4839 = x1769 * x2258
    x4840 = x2050 * x4463
    x4841 = ddq_j7 * x510
    x4842 = 8 * dq_j6
    x4843 = x2114 * x4842
    x4844 = ddq_i1 * dq_j6
    x4845 = x216 * x4844
    x4846 = x2271 * x281
    x4847 = x275 * x3220
    x4848 = x250 * x3950
    x4849 = x244 * x4471
    x4850 = x1545 * x745
    x4851 = x149 * x4850
    x4852 = x3251 * (x1542 * x685 + x4851 * x681)
    x4853 = dq_i1 * (x1542 * x648 + x4851 * x644)
    x4854 = dq_i1 * (x1542 * x617 + x4851 * x613)
    x4855 = dq_i1 * (x1542 * x1958 + x4851 * x575)
    x4856 = x173 * x2054
    x4857 = x155 * x2055
    x4858 = x142 * x4857 + x213 * x4856
    x4859 = 4 * ddq_i6 * dq_j6**2
    x4860 = x3110 + x3111
    x4861 = dq_i2 * x4842
    x4862 = x3848 + x3849
    x4863 = dq_i3 * x4842
    x4864 = x4373 + x4374
    x4865 = dq_i4 * x4842
    x4866 = x4752 + x4753
    x4867 = dq_i5 * x4842
    x4868 = x1268 * x4751
    x4869 = x1269 * x2087
    x4870 = x4868 + x4869
    x4871 = 8 * x161
    x4872 = x2135 + x2136 + x2140 + x2141
    x4873 = dq_i1 * dq_j6
    x4874 = sigma_kin_v_7_6 * x4873
    x4875 = dq_j6 * x671
    x4876 = x3283 * x561
    x4877 = dq_j6 * x663
    x4878 = x2356 * x561
    x4879 = dq_j6 * x631
    x4880 = x397 * x561
    x4881 = x4514 * x4880
    x4882 = dq_j6 * x590
    x4883 = x4004 * x4880
    x4884 = x192 * x535
    x4885 = dq_j6 * x12
    x4886 = x2248 * x561
    x4887 = x195 * x535
    x4888 = (
        ddq_i1 * x470
        + 2 * dq_i1 * dq_i6 * sigma_kin_v_6_1 * sigma_kin_v_6_6 * x166 * x168 * x170 * x171 * x176 * x221 * x464
        + 2 * dq_i1 * dq_i6 * sigma_kin_v_7_1 * sigma_kin_v_7_6 * x140 * x144 * x146 * x148 * x149 * x154 * x229 * x434
        - x12 * x475
        - x457 * x722
        - x476 * x511
        - x481 * x671
        - x491 * x663
        - x499 * x631
        - x506 * x590
    )
    x4889 = dq_j6 * x515
    x4890 = x161 * x515
    x4891 = x457 * x4889
    x4892 = x1165 * x4844
    x4893 = dq_i2 * x4873
    x4894 = x208 * x4891
    x4895 = x1047 * x4844
    x4896 = dq_i3 * x4873
    x4897 = x4844 * x931
    x4898 = dq_i4 * x4873
    x4899 = x4844 * x842
    x4900 = dq_i5 * x4873
    x4901 = x4844 * x767
    x4902 = x4838 * x95
    x4903 = x1158 * x2050
    x4904 = 4 * x4841
    x4905 = x208 * x4843
    x4906 = x2443 * x281
    x4907 = x275 * x3358
    x4908 = x250 * x4032
    x4909 = x244 * x4541
    x4910 = dq_i2 * (x1542 * x2813 + x2569 * x4851)
    x4911 = dq_i2 * (x1542 * x2931 + x2555 * x4851)
    x4912 = dq_i2 * (x1542 * x3033 + x2544 * x4851)
    x4913 = x477 * x4856
    x4914 = x479 * x4857
    x4915 = x1156 * x4914 + x1160 * x4913
    x4916 = x2066 + x2067
    x4917 = 8 * x4873
    x4918 = x3864 + x3865
    x4919 = x4387 + x4388
    x4920 = x4763 + x4764
    x4921 = x1181 * x4751
    x4922 = x1184 * x4796
    x4923 = x4921 + x4922
    x4924 = x3169 + x3170 + x3171 + x3172
    x4925 = dq_i2 * dq_j6
    x4926 = sigma_kin_v_7_6 * x4925
    x4927 = ddq_i2 * dq_j6
    x4928 = dq_j6 * x666
    x4929 = dq_j6 * x635
    x4930 = dq_j6 * x594
    x4931 = dq_j6 * x712
    x4932 = (
        ddq_i2 * x2526
        + 2 * dq_i2 * dq_i6 * sigma_kin_v_6_2 * sigma_kin_v_6_6 * x166 * x168 * x170 * x171 * x176 * x221 * x464
        + 2 * dq_i2 * dq_i6 * sigma_kin_v_7_2 * sigma_kin_v_7_6 * x140 * x144 * x146 * x148 * x149 * x154 * x229 * x434
        - x2523 * x683
        - x2527 * x671
        - x2528 * x712
        - x2529 * x666
        - x2530 * x635
        - x2531 * x594
        - x2532 * x482
    )
    x4933 = dq_j6 * x405
    x4934 = x161 * x405
    x4935 = x2523 * x4933
    x4936 = x208 * x4935
    x4937 = dq_i3 * x4925
    x4938 = dq_i4 * x4925
    x4939 = dq_i5 * x4925
    x4940 = dq_j6 * x1386
    x4941 = sigma_kin_v_7_6 * x159
    x4942 = x1037 * x4941
    x4943 = x1309 * x4904
    x4944 = x1311 * x223
    x4945 = x2732 * x281
    x4946 = x275 * x3431
    x4947 = x250 * x4082
    x4948 = x244 * x4588
    x4949 = x1542 * x3678 + x3441 * x4851
    x4950 = x1542 * x3773 + x3438 * x4851
    x4951 = x1036 * x4367 + x1045 * x4366
    x4952 = x2086 + x2089
    x4953 = x3128 + x3129
    x4954 = x4397 + x4398
    x4955 = x4774 + x4775
    x4956 = x223 * x4773
    x4957 = x1062 * x4796
    x4958 = x4956 + x4957
    x4959 = x3903 + x3904 + x3905 + x3906
    x4960 = dq_i3 * dq_j6
    x4961 = sigma_kin_v_7_6 * x4960
    x4962 = ddq_i3 * dq_j6
    x4963 = dq_j6 * x639
    x4964 = dq_j6 * x601
    x4965 = dq_j6 * x669
    x4966 = (
        ddq_i3 * x3483
        + 2 * dq_i3 * dq_i6 * sigma_kin_v_6_3 * sigma_kin_v_6_6 * x166 * x168 * x170 * x171 * x176 * x221 * x464
        + 2 * dq_i3 * dq_i6 * sigma_kin_v_7_3 * sigma_kin_v_7_6 * x140 * x144 * x146 * x148 * x149 * x154 * x229 * x434
        - x3480 * x646
        - x3486 * x663
        - x3487 * x666
        - x3488 * x669
        - x3489 * x639
        - x3490 * x601
        - x3491 * x492
    )
    x4967 = dq_j6 * x411
    x4968 = x161 * x411
    x4969 = x3480 * x4967
    x4970 = x208 * x4969
    x4971 = dq_i4 * x4960
    x4972 = dq_i5 * x4960
    x4973 = x2435 * x4837
    x4974 = x4364 * x4973
    x4975 = x4941 * x921
    x4976 = x2683 * x281
    x4977 = x275 * x3609
    x4978 = x250 * x4127
    x4979 = x244 * x4625
    x4980 = x1779 * (x1542 * x4300 + x4134 * x4851)
    x4981 = x4366 * x493
    x4982 = x4367 * x496
    x4983 = x4981 * x924 + x4982 * x919
    x4984 = x2100 + x2101
    x4985 = x3138 + x3139
    x4986 = x3874 + x3875
    x4987 = x4784 + x4785
    x4988 = x223 * x4783
    x4989 = x4796 * x946
    x4990 = x4988 + x4989
    x4991 = x4425 + x4426 + x4427 + x4428
    x4992 = dq_j6 * sigma_kin_v_7_6
    x4993 = dq_i4 * x4992
    x4994 = ddq_i4 * dq_j6
    x4995 = dq_j6 * x605
    x4996 = dq_j6 * x642
    x4997 = (
        ddq_i4 * x4172
        + 2 * dq_i4 * dq_i6 * sigma_kin_v_6_4 * sigma_kin_v_6_6 * x166 * x168 * x170 * x171 * x176 * x221 * x464
        + 2 * dq_i4 * dq_i6 * sigma_kin_v_7_4 * sigma_kin_v_7_6 * x140 * x144 * x146 * x148 * x149 * x154 * x229 * x434
        - x4169 * x615
        - x4175 * x631
        - x4176 * x635
        - x4177 * x639
        - x4178 * x642
        - x4179 * x605
        - x4180 * x500
    )
    x4998 = dq_j6 * x415
    x4999 = x161 * x415
    x5000 = x4169 * x4998
    x5001 = x208 * x5000
    x5002 = dq_j6 * x1779
    x5003 = x4941 * x831
    x5004 = x2640 * x281
    x5005 = x275 * x3574
    x5006 = x250 * x4240
    x5007 = x244 * x4660
    x5008 = x4744 * x834 + x4745 * x830
    x5009 = x2113 + x2115
    x5010 = x3149 + x3150
    x5011 = x3883 + x3884
    x5012 = x4406 + x4407
    x5013 = x4794 * x857
    x5014 = x4796 * x858
    x5015 = x5013 + x5014
    x5016 = x4805 + x4806 + x4807 + x4808
    x5017 = dq_i5 * x4992
    x5018 = ddq_i5 * dq_j6
    x5019 = dq_j6 * x611
    x5020 = (
        ddq_i5 * x4687
        + 2 * dq_i5 * dq_i6 * sigma_kin_v_6_5 * sigma_kin_v_6_6 * x166 * x168 * x170 * x171 * x176 * x221 * x464
        + 2 * dq_i5 * dq_i6 * sigma_kin_v_7_5 * sigma_kin_v_7_6 * x140 * x144 * x146 * x148 * x149 * x154 * x229 * x434
        - x4686 * x577
        - x4689 * x590
        - x4690 * x594
        - x4691 * x601
        - x4692 * x605
        - x4693 * x611
        - x4694 * x507
    )
    x5021 = dq_j6 * x419
    x5022 = x161 * x419
    x5023 = x4686 * x5021
    x5024 = x208 * x5023
    x5025 = x1769 * x2435
    x5026 = x4941 * x747
    x5027 = x2610 * x281
    x5028 = x275 * x3547
    x5029 = x250 * x4215
    x5030 = x244 * x4711
    x5031 = x222 * x4744
    x5032 = x149 * x2055
    x5033 = x5031 * x750 + x5032 * x746
    x5034 = x2125 + x2126
    x5035 = x3159 + x3160
    x5036 = x3893 + x3894
    x5037 = x4417 + x4418
    x5038 = x4795 + x4797
    x5039 = x4794 * x816
    x5040 = x4794 * x810
    x5041 = x1403 * x2087
    x5042 = x2087 * x812
    x5043 = x5039 + x5040 + x5041 + x5042
    x5044 = x138 * x208
    x5045 = x397 * x5044
    x5046 = x208 * x517 - x2157 * x532 + x417 * x5045 + x421 * x5045 + x429 * x532 + x5044 * x526 + x5044 * x529
    x5047 = ddq_j7 * x532
    x5048 = x2177 * x3841
    x5049 = x195 * x536
    x5050 = 2 * x192
    x5051 = x187 + x188 + x189 + x190 + x191 + x193 + x5050
    x5052 = x138 * x5051
    x5053 = x5052 * x521
    x5054 = x397 * x5053
    x5055 = 2 * x217
    x5056 = x161 * x4856
    x5057 = x161 * x4857
    x5058 = x161 * x4366
    x5059 = x161 * x4367
    x5060 = x161 * x2054
    x5061 = x161 * x4745
    x5062 = x166 * x5060
    x5063 = x161 * x2055
    x5064 = x161 * x5032
    x5065 = x1045 * x5060
    x5066 = x5060 * x929
    x5067 = x5060 * x834
    x5068 = x161 * x4744
    x5069 = x5064 * x755
    x5070 = x2151 * x2241
    x5071 = x1309 * x510
    x5072 = x2114 * x823
    x5073 = x153 * x2114
    x5074 = x2178 * x3188
    x5075 = x406 * x4874
    x5076 = x137 * x4992
    x5077 = x2201 * x5050
    x5078 = x2203 * x4992
    x5079 = x195 * x2201
    x5080 = x4992 * x568
    x5081 = x1156 * x4992
    x5082 = x1035 * x4992
    x5083 = 2 * x5079
    x5084 = x4992 * x919
    x5085 = x4992 * x828
    x5086 = x4992 * x520
    x5087 = x4992 * x744
    x5088 = x195 * x744
    x5089 = sigma_pot_7_c * x519 - sigma_pot_7_s * x518
    x5090 = x102 * x5089
    x5091 = x2047 * x5090
    x5092 = ddq_i1 * x158
    x5093 = x136 * x5092
    x5094 = 4 * x158
    x5095 = x281 * x5094
    x5096 = x275 * x5094
    x5097 = x250 * x5094
    x5098 = x142 * x5094
    x5099 = x1922 * x244
    x5100 = x2050 * x217
    x5101 = x153 * x461
    x5102 = x3251 * x5101 * x682
    x5103 = dq_i1 * x5101
    x5104 = x5103 * x645
    x5105 = x5103 * x614
    x5106 = x5103 * x576
    x5107 = x457 * x5103
    x5108 = dq_j7**2
    x5109 = ddq_i7 * x5108
    x5110 = x2169 * x5109
    x5111 = x155 * x158
    x5112 = x2264 * x5111
    x5113 = x203 * x2172
    x5114 = x204 * x5113
    x5115 = x137 * x158
    x5116 = 8 * dq_i2
    x5117 = dq_i3 * x158
    x5118 = x2178 * x307
    x5119 = x5117 * x5118
    x5120 = dq_i4 * x158
    x5121 = x307 * x5120
    x5122 = x2178 * x5121
    x5123 = dq_i5 * x158
    x5124 = x150 * x2178
    x5125 = x5123 * x5124
    x5126 = dq_i6 * x158
    x5127 = x2178 * x5126
    x5128 = x193 * x2179
    x5129 = x232 * x404
    x5130 = x232 * x410
    x5131 = x232 * x397
    x5132 = x416 * x5131
    x5133 = x420 * x5131
    x5134 = x476 * x533
    x5135 = x199 * x426
    x5136 = (
        -ddq_i1 * x138 * x139 * x143 * x145 * x147 * x149
        + x12 * x5135
        + x5129 * x671
        + x5130 * x663
        + x5132 * x631
        + x5133 * x590
        + x5134
    )
    x5137 = x154 * x542
    x5138 = x432 * x5137
    x5139 = x158 * x400
    x5140 = x3251 * x5139
    x5141 = ddq_i1 * x532
    x5142 = x5139 * x5141
    x5143 = x138 * x2193
    x5144 = x1156 * x5139
    x5145 = x3285 * x663
    x5146 = x416 * x631
    x5147 = x3999 * x5144
    x5148 = x420 * x590
    x5149 = x153 * x5134
    x5150 = x158 * x5149
    x5151 = x193 * x542
    x5152 = x203 * x5151
    x5153 = x515 * x5152
    x5154 = x12 * x1367
    x5155 = x203 * x565
    x5156 = x515 * x5155
    x5157 = dq_i1 * x400
    x5158 = x158 * x5157
    x5159 = dq_i3 * x5158
    x5160 = x138 * x2212
    x5161 = x5139 * x671
    x5162 = x2353 * x5161
    x5163 = x3999 * x5139
    x5164 = x1035 * x5163
    x5165 = x5139 * x5154
    x5166 = dq_i4 * x5158
    x5167 = x2222 * x3999
    x5168 = x5139 * x5145
    x5169 = x5148 * x5163
    x5170 = dq_i5 * x5158
    x5171 = x2231 * x3999
    x5172 = x5146 * x5163
    x5173 = dq_i6 * x2237
    x5174 = dq_i6 * x534
    x5175 = x158 * x744
    x5176 = x5174 * x5175
    x5177 = x137 * x5139
    x5178 = x11 * x5139
    x5179 = x138 * x3191
    x5180 = x3999 * x5177
    x5181 = x193 * x548
    x5182 = x158 * x521
    x5183 = x5151 * x543
    x5184 = x404 * x5182
    x5185 = x138 * x5184
    x5186 = x3999 * x5182
    x5187 = x547 * x566
    x5188 = x543 * x565
    x5189 = x5091 * x95
    x5190 = x1157 * x5094
    x5191 = dq_i2 * x5101
    x5192 = x2570 * x5191
    x5193 = x2556 * x5191
    x5194 = x2545 * x5191
    x5195 = x2523 * x5191
    x5196 = 4 * x195
    x5197 = dq_i2 * x5111 * x5196
    x5198 = 8 * dq_i1
    x5199 = x149 * x5127
    x5200 = x482 * x533
    x5201 = (
        -ddq_i2 * x138 * x139 * x143 * x145 * x147 * x149
        + x5129 * x712
        + x5130 * x666
        + x5132 * x635
        + x5133 * x594
        + x5135 * x671
        + x5200
    )
    x5202 = sigma_kin_v_7_7 * x5138
    x5203 = ddq_i2 * x532
    x5204 = x3285 * x666
    x5205 = x416 * x635
    x5206 = x420 * x594
    x5207 = x153 * x5200
    x5208 = x405 * x5152
    x5209 = x138 * x712
    x5210 = x5139 * x5209
    x5211 = x405 * x5155
    x5212 = dq_i2 * x400
    x5213 = x158 * x5212
    x5214 = dq_i3 * x5213
    x5215 = x5139 * x5203
    x5216 = x1367 * x5161
    x5217 = x158 * x5207
    x5218 = x404 * x5210
    x5219 = dq_i4 * x5213
    x5220 = x5139 * x5204
    x5221 = x5163 * x5206
    x5222 = dq_i5 * x5213
    x5223 = x5163 * x5205
    x5224 = x5139 * x711
    x5225 = x1367 * x671
    x5226 = x400 * x5187
    x5227 = x2160 * x5092
    x5228 = sigma_kin_v_7_2 * x5095
    x5229 = sigma_kin_v_7_3 * x5096
    x5230 = x1037 * x5094
    x5231 = sigma_kin_v_7_5 * x244
    x5232 = sigma_kin_v_7_6 * x217
    x5233 = x3515 * x5101
    x5234 = x3501 * x5101
    x5235 = x3480 * x5101
    x5236 = x2168 * x5109
    x5237 = x5117 * x5196
    x5238 = x148 * x5113
    x5239 = x158 * x5118
    x5240 = dq_i1 * x5239
    x5241 = dq_i2 * x5239
    x5242 = x492 * x533
    x5243 = (
        -ddq_i3 * x138 * x139 * x143 * x145 * x147 * x149
        + x5129 * x666
        + x5130 * x669
        + x5132 * x639
        + x5133 * x601
        + x5135 * x663
        + x5242
    )
    x5244 = x185 * x5157
    x5245 = x5244 * x532
    x5246 = ddq_i3 * x532
    x5247 = x5139 * x666
    x5248 = x416 * x639
    x5249 = x420 * x601
    x5250 = x153 * x5242
    x5251 = x411 * x5152
    x5252 = x3285 * x669
    x5253 = x411 * x5155
    x5254 = x1251 * x5212
    x5255 = x5254 * x532
    x5256 = x5139 * x5246
    x5257 = x2193 * x5212
    x5258 = x138 * x5257
    x5259 = x1367 * x663
    x5260 = x158 * x5250
    x5261 = dq_i3 * x400
    x5262 = x158 * x5261
    x5263 = dq_i4 * x5262
    x5264 = x5139 * x5259
    x5265 = x2353 * x5247
    x5266 = x5163 * x5249
    x5267 = x5139 * x5252
    x5268 = dq_i5 * x5262
    x5269 = x5163 * x5248
    x5270 = x5139 * x668
    x5271 = 1.0 * x2435 * x5090
    x5272 = sigma_kin_v_7_4 * x5097
    x5273 = x5094 * x921
    x5274 = x1327 * x5101
    x5275 = x1779 * x4191
    x5276 = x1316 * x5101
    x5277 = x1776 * x4169
    x5278 = x5121 * x5196
    x5279 = x500 * x533
    x5280 = (
        -ddq_i4 * x138 * x139 * x143 * x145 * x147 * x149
        + x5129 * x635
        + x5130 * x639
        + x5132 * x642
        + x5133 * x605
        + x5135 * x631
        + x5279
    )
    x5281 = ddq_i4 * x532
    x5282 = x5139 * x635
    x5283 = x3285 * x639
    x5284 = x420 * x605
    x5285 = x153 * x5279
    x5286 = x415 * x5152
    x5287 = x416 * x642
    x5288 = x415 * x5155
    x5289 = x5139 * x5281
    x5290 = x1367 * x631
    x5291 = x158 * x5285
    x5292 = x3999 * x5287
    x5293 = x1126 * x5261
    x5294 = x5293 * x532
    x5295 = x2212 * x5261
    x5296 = x138 * x5295
    x5297 = x5139 * x5290
    x5298 = x2353 * x5282
    x5299 = x5139 * x5292
    x5300 = x1779 * x5139
    x5301 = x5139 * x5283
    x5302 = x400 * x813
    x5303 = x5302 * x532
    x5304 = x158 * x1776
    x5305 = x534 * x744
    x5306 = x5163 * x5284
    x5307 = x5139 * x641
    x5308 = dq_i4 * x400
    x5309 = x5094 * x831
    x5310 = x1334 * x5101
    x5311 = x1928 * x4686
    x5312 = x150 * x5196
    x5313 = x5123 * x5312
    x5314 = x158 * x5124
    x5315 = x507 * x533
    x5316 = (
        -ddq_i5 * x138 * x139 * x143 * x145 * x147 * x149
        + x5129 * x594
        + x5130 * x601
        + x5132 * x605
        + x5133 * x611
        + x5135 * x590
        + x5315
    )
    x5317 = ddq_i5 * x532
    x5318 = x5139 * x594
    x5319 = x3285 * x601
    x5320 = x416 * x605
    x5321 = x153 * x5315
    x5322 = x419 * x5152
    x5323 = x420 * x611
    x5324 = x419 * x5155
    x5325 = x5139 * x5317
    x5326 = x1367 * x590
    x5327 = x158 * x5321
    x5328 = x3999 * x5323
    x5329 = x5139 * x5326
    x5330 = x2353 * x5318
    x5331 = x5139 * x5328
    x5332 = x5139 * x5319
    x5333 = x158 * x1928
    x5334 = x5163 * x5320
    x5335 = x5139 * x610
    x5336 = dq_i5 * x400
    x5337 = x5094 * x747
    x5338 = x5126 * x5196
    x5339 = x400 * x744
    x5340 = x149 * x5088
    x5341 = 8 * x5340
    x5342 = x148 * x5341
    x5343 = x400 * x5088
    x5344 = x513 * x533
    x5345 = (
        -ddq_i6 * x138 * x139 * x143 * x145 * x147 * x149
        + x476 * x5135
        + x482 * x5129
        + x492 * x5130
        + x500 * x5132
        + x507 * x5133
        + x5344
    )
    x5346 = ddq_i6 * x532
    x5347 = x482 * x5139
    x5348 = x3285 * x492
    x5349 = x416 * x500
    x5350 = x420 * x507
    x5351 = x137 * x423
    x5352 = x153 * x5344
    x5353 = x423 * x5155
    x5354 = x5139 * x5346
    x5355 = x1367 * x476
    x5356 = x423 * x5152
    x5357 = x158 * x5352
    x5358 = x5139 * x5355
    x5359 = x2353 * x5347
    x5360 = x1776 * x5139
    x5361 = x5139 * x5348
    x5362 = x5163 * x5350
    x5363 = x1928 * x5139
    x5364 = x5163 * x5349
    x5365 = x158 * x512
    x5366 = x150 * x2152
    x5367 = x5094 * x5366
    x5368 = x460 * x4659
    x5369 = x153 * x547
    x5370 = x193 * x460 * x5312
    x5371 = sigma_kin_v_7_7 * x5137
    x5372 = x138 * x193
    x5373 = x529 * x5372
    x5374 = x193 * x3999
    x5375 = x521 * x5374
    x5376 = x2244 * x537
    x5377 = x137 * x532
    x5378 = x281 * x549
    x5379 = x275 * x552
    x5380 = x250 * x555
    x5381 = x244 * x558
    x5382 = x217 * x561
    x5383 = 2 * x158
    x5384 = x137 * x400
    x5385 = x5372 * x5384
    x5386 = x5374 * x5384
    x5387 = x193 * x534
    x5388 = x2368 * x5384
    x5389 = x4010 * x5384
    x5390 = x195 * x534
    x5391 = x1156 * x532
    x5392 = x1156 * x400
    x5393 = x193 * x517
    x5394 = x5374 * x5392
    x5395 = x423 * x5387
    x5396 = x158 * x2203
    x5397 = x195 * x517
    x5398 = x2368 * x529
    x5399 = x4010 * x5392
    x5400 = x423 * x5390
    x5401 = x1035 * x532
    x5402 = x5383 * x568
    x5403 = x1035 * x400
    x5404 = x526 * x5372
    x5405 = x5374 * x5403
    x5406 = x2368 * x526
    x5407 = x4010 * x5403
    x5408 = x1008 * x5308
    x5409 = x193 * x532
    x5410 = x532 * x919
    x5411 = x2222 * x5308
    x5412 = x400 * x919
    x5413 = x421 * x5412
    x5414 = x5336 * x895
    x5415 = x532 * x828
    x5416 = x2231 * x5336
    x5417 = x400 * x828
    x5418 = x417 * x5417
    x5419 = dq_i6 * x5302
    x5420 = x532 * x744
    x5421 = x5339 * x5374
    x5422 = x138 * x5343
    x5423 = x3999 * x5343

    K_block_list = []
    K_block_list.append(
        ddq_j2 * x713
        + ddq_j3 * x670
        + ddq_j4 * x643
        + ddq_j5 * x612
        + ddq_j6 * x514
        + x1022
        * (
            2 * dq_i1 * dq_i7 * dq_j1 * x140 * x144 * x145 * x148 * x150 * x153 * x203 * x327 * x434 * x919
            + 2 * dq_i1 * (x907 * x963 + x952 * x965 + x966 * x967 + x968 * x969 + x970)
            + 2 * dq_i2 * (x952 * x973 + x963 * x971 + x966 * x974 + x968 * x975 + x981)
            + 2 * dq_i3 * (x952 * x985 + x966 * x988 + x968 * x990 + x982 * x984 + x996)
            + dq_i4
            * (
                x1001 * x952
                + x1002 * x952
                + x1004 * x966
                + x1005 * x31 * x732
                + x1007 * x968
                + x1008 * x31 * x734
                + x1021
                + x31 * x729 * x999
                + x982 * x998
            )
            + 2 * dq_i5 * (x263 * x944 + x897 * x957 + x952 * x954 + x961)
            + 2 * dq_i6 * (x223 * x944 + x815 * x946 + x951)
            + 2 * dq_i7 * sigma_kin_v_7_1 * x140 * x144 * x145 * x148 * x150 * x153 * x195 * x203 * x434 * x919
            - x515 * x912
            - x515 * x918
            - x515 * x926
            - x538 * x943
            - x840 * x928
            - x922
            - x932
            - x936
            - x938
            - x940
            - x942
        )
        + x110 * x113 * (sigma_pot_1_c * x2 + sigma_pot_1_s * x5)
        + x1142
        * (
            2 * dq_i1 * dq_i7 * dq_j1 * x1035 * x140 * x143 * x146 * x148 * x150 * x153 * x203 * x327 * x434
            + 2 * dq_i1 * (x1088 * x1090 + x1091 * x907 + x1092 * x862 + x1093 * x966 + x1095 * x968 + x1096)
            + 2 * dq_i2 * (x1091 * x971 + x1097 * x1099 + x1100 * x862 + x1101 * x966 + x1104 * x968 + x1112)
            + dq_i3
            * (
                x1097 * x1114
                + x1115 * x31 * x736
                + x1116 * x982
                + x1117 * x982
                + x1119 * x862
                + x1120 * x862
                + x1122 * x966
                + x1123 * x966
                + x1125 * x968
                + x1127 * x968
                + x1141
            )
            + 2 * dq_i4 * (x1049 * x881 + x1076 * x982 + x1079 * x966 + x1082 * x968 + x1087)
            + 2 * dq_i5 * (x1060 * x263 + x1069 * x862 + x1070 * x897 + x1074)
            + 2 * dq_i6 * (x1060 * x223 + x1062 * x815 + x1066)
            + 2 * dq_i7 * sigma_kin_v_7_1 * x1035 * x140 * x143 * x146 * x148 * x150 * x153 * x195 * x203 * x434
            - x1026 * x515
            - x1030 * x515
            - x1034 * x515
            - x1038
            - x1042 * x515
            - x1043 * x1044 * x840
            - x1048
            - x1052
            - x1054
            - x1056
            - x1058
            - x1059 * x538
        )
        + x117 * x131
        + x117 * x135
        + x1266
        * (
            2 * dq_i1 * dq_i7 * dq_j1 * x1156 * x139 * x144 * x146 * x148 * x150 * x153 * x203 * x327 * x434
            + 2
            * dq_i1
            * (x1088 * x1227 + x1144 * x1226 + x1154 * x1231 + x1161 * x1232 + x1228 * x1230 + x1233 * x1234 + x1235)
            + dq_i2
            * (
                x1097 * x1239
                + x1226 * x1237
                + x1228 * x1242
                + x1231 * x1245
                + x1232 * x1248
                + x1233 * x1250
                + x1238 * x31 * x737
                + x1240 * x31 * x738
                + x1243 * x31 * x739
                + x1246 * x31 * x740
                + x1249 * x31 * x741
                + x1251 * x31 * x742
                + x1265
            )
            + 2 * dq_i3 * (x1097 * x1213 + x1214 * x962 + x1215 * x862 + x1216 * x966 + x1219 * x968 + x1225)
            + 2 * dq_i4 * (x1199 * x962 + x1201 * x881 + x1203 * x966 + x1205 * x968 + x1211)
            + 2 * dq_i5 * (x1182 * x1191 + x1190 * x862 + x1192 * x897 + x1197)
            + 2 * dq_i6 * (x1181 * x1182 + x1184 * x815 + x1188)
            + 2 * dq_i7 * sigma_kin_v_7_1 * x1156 * x139 * x144 * x146 * x148 * x150 * x153 * x195 * x203 * x434
            - x1145 * x515
            - x1148 * x515
            - x1152 * x515
            - x1155 * x515
            - x1159
            - x1162 * x515
            - x1163 * x390 * x516
            - x1166
            - x1169
            - x1173
            - x1175
            - x1178
            - x1179 * x538
        )
        + x13 * x18
        + x13 * x9
        + x136 * x160
        + x136 * x218
        + x136 * x245
        + x136 * x251
        + x136 * x276
        + x136 * x282
        - x136
        * (
            dq_i1 * x438
            + dq_i1 * x439
            + dq_i1 * x440
            + dq_i1 * x441
            + dq_i1 * x444
            + dq_i1 * x446
            + dq_i1 * x447
            + dq_i1 * x448
            + dq_i1 * x449
            + dq_i1 * x450
            + dq_i1 * x453
            + dq_i1 * x454
            - dq_i1
            * (
                x1088 * x60
                + x1226 * x38
                + x1228 * x1295
                + x1231 * x134
                + x1232 * x183
                + x1233 * x201
                + x125 * x31 * x323
                + x174 * x31 * x326
                + x185 * x31 * x329
                + x19 * x31 * x315
                + x31 * x317 * x39
                + x31 * x320 * x65
                + 2 * x436
                + 2 * x437
                + x455
            )
            + x1267 * x18
            + x1267 * x9
            + x160
            - 2 * x206 * x208 * x897
            - 2 * x206 * x211
            + x218
            + x245
            + x251
            + x276
            + x282
            + x330 * x538
            - x405
            * (x1088 * x1289 + x1226 * x1288 + x1228 * x1291 + x1231 * x1292 + x1232 * x1293 + x1233 * x1294 + x395)
            - x411 * (x1088 * x1283 + x1284 * x962 + x1285 * x862 + x1286 * x966 + x1287 * x968 + x358)
            - x415 * (x1277 * x962 + x1279 * x966 + x1281 * x968 + x296 * x881 + x312)
            - x419 * (x1271 * x862 + x1273 * x774 + x1275 * x897 + x270)
            - x423 * (x1182 * x1268 + x1269 * x775 + x238)
        )
        + x164 * x180
        + x164 * x184
        + x186 * x198
        + x198 * x201
        - x206 * x212
        - x238 * x239
        - x270 * x271
        + x30 * x35
        - 4 * x31 * (x436 + x437 + x455)
        - x312 * x313
        + x330 * x331
        + x35 * x38
        - x358 * x359
        - x395 * x396
        - x430 * x433 * x435
        + x53 * x57
        + x57 * x60
        + x574
        * (
            x187 * x545
            + x417 * x531
            + x421 * x531
            + x522 * x525
            - x523
            + x525 * x536
            + x527 * x528
            + x529 * x530
            - x538 * x541
            + x546 * x548
            + x564
            + x571
        )
        + x64 * x82
        + x64 * x87
        + x743
        * (
            ddq_i1 * (x314 * x695 + x319 * x622 + x322 * x581 + x325 * x465 + x328 * x458 + x51 * x735 + x6 * x7**2)
            + x11 * x714
            + x11 * x715
            + x11 * x716
            + x11 * x717
            + x11 * x718
            + x11 * x719
            + x11 * x720
            - x12 * (x714 + x715 + x716 + x717 + x718 + x719 + x720)
            - x476 * (x221 * x724 + x509 * x725)
            - x590 * (x255 * x727 + x263 * x723 + x504 * x721)
            - x631 * (x287 * x729 + x299 * x732 + x306 * x734 + x602 * x730)
            - x663 * (x334 * x736 + x343 * x728 + x485 * x731 + x489 * x733 + x597 * x726)
            - x671 * (x362 * x737 + x368 * x738 + x372 * x739 + x377 * x740 + x383 * x741 + x389 * x742)
            - x721 * x722
        )
        + x823
        * (
            2 * dq_i1 * dq_i7 * dq_j1 * x140 * x144 * x146 * x148 * x149 * x153 * x203 * x327 * x434 * x744
            + 2 * dq_i1 * (x773 * x774 + x775 * x776 + x778)
            + 2 * dq_i2 * (x774 * x780 + x781 * x783 + x788)
            + 2 * dq_i3 * (x774 * x789 + x790 * x791 + x795)
            + 2 * dq_i4 * (x774 * x796 + x791 * x797 + x800)
            + 2 * dq_i5 * (x504 * x781 + x801 * x803 + x808)
            + dq_i6 * (x31 * x724 * x811 + x775 * x812 + x801 * x810 + x814 * x815 + x822)
            + 2 * dq_i7 * sigma_kin_v_7_1 * x140 * x144 * x146 * x148 * x149 * x153 * x195 * x203 * x434 * x744
            - x209 * x516 * x757
            - x515 * x754
            - x538 * x770
            - x749
            - x759
            - x761
            - x763
            - x766
            - x768
        )
        + x906
        * (
            2 * dq_i1 * dq_i7 * dq_j1 * x140 * x144 * x146 * x147 * x150 * x153 * x203 * x327 * x434 * x828
            + 2 * dq_i1 * (x234 * x473 * x866 + x772 * x864 + x862 * x863 + x867)
            + 2 * dq_i2 * (x779 * x864 + x782 * x866 + x862 * x868 + x872)
            + 2 * dq_i3 * (x485 * x864 + x790 * x866 + x862 * x875 + x880)
            + 2 * dq_i4 * (x494 * x864 + x844 * x881 + x865 * x884 + x888)
            + dq_i5 * (x31 * x727 * x891 + x801 * x892 + x801 * x893 + x862 * x890 + x865 * x894 + x896 * x897 + x905)
            + 2 * dq_i6 * (x801 * x857 + x815 * x858 + x861)
            + 2 * dq_i7 * sigma_kin_v_7_1 * x140 * x144 * x146 * x147 * x150 * x153 * x195 * x203 * x434 * x828
            - x515 * x827
            - x515 * x836
            - x538 * x855
            - x832
            - x839 * x840
            - x843
            - x846
            - x849
            - x851
            - x853
        )
    )
    K_block_list.append(
        ddq_j1 * x713
        - dq_i2 * x1265 * x136
        + x1022
        * (
            -x1365 * x937
            + x1369 * x1465
            + x1377 * x1456
            + x1379 * x1453
            + x1381 * x1467
            + x1395 * x1463
            - x1441 * x405
            - x1442 * x405
            + x1444 * x1446
            - x1448 * x405
            - x1450 * x405
            + x1468 * x1469
        )
        + x1142
        * (
            -x1055 * x1365
            + x1362 * x1492
            + x1373 * x1483
            + x1377 * x1481
            + x1379 * x1478
            + x1381 * x1491
            + x1395 * x1489
            + x1446 * x1474
            - x1471 * x405
            - x1472 * x405
            - x1473 * x405
            - x1475 * x405
            - x1476 * x405
        )
        + x1144 * x1296
        + x1154 * x1305
        + x1159 * x136
        + x1161 * x1312
        + x1166 * x136
        + x1169 * x136
        + x1173 * x136
        + x1175 * x136
        + x1178 * x136
        + x1179 * x331
        - x1188 * x239
        - x1197 * x271
        - x1211 * x313
        - x1225 * x359
        + x1227 * x1297
        + x1229 * x1302 * x1303
        + x1234 * x1314
        - x1235 * x1361
        + x1266
        * (
            2
            * dq_i1
            * dq_i2
            * dq_j1
            * (sigma_kin_v_4_2 * x1508 + x1144 * x277 + x1147 * x1484 + x1154 * x675 + x1161 * x677 + x1234 * x679)
            + 2 * dq_i2 * dq_i3 * dq_j1 * (x1180 * x1464 + x1200 * x1429 + x1213 * x697 + x1218 * x1490 + x1506 * x344)
            + 2 * dq_i2 * dq_i4 * dq_j1 * (x1201 * x1432 + x1202 * x1482 + x1204 * x1507 + x1506 * x290)
            + 2 * dq_i2 * dq_i5 * dq_j1 * (x1183 * x1480 + x1189 * x1479 + x1505 * x263)
            + 2 * dq_i2 * dq_i6 * dq_j1 * (x1184 * x1397 + x1505 * x223)
            + 2
            * dq_i2
            * dq_i7
            * dq_j1
            * sigma_kin_v_7_1
            * sigma_kin_v_7_2
            * x1156
            * x139
            * x144
            * x146
            * x148
            * x150
            * x153
            * x203
            * x434
            + dq_j1
            * x711
            * (
                sigma_kin_v_4_2 * x1258
                + x1237 * x277
                + x1239 * x697
                + x1245 * x675
                + x1248 * x677
                + x1250 * x679
                + x1509
            )
            - x1177 * x1365
            - x1494 * x405
            - x1495 * x405
            - x1498 * x405
            - x1500 * x405
            - x1502 * x405
            - x1504 * x405
        )
        + x1299 * x1301
        + x1306 * x1308 * x1310
        - x1315 * x212
        - x1317 * x1326
        - x1328 * x1333
        - x1335 * x1340
        - x1342 * x1346
        - x136
        * (
            dq_i2 * x1470 * x1516
            + dq_i2 * x1493 * x1511 * x1512
            + dq_i2 * x1499 * x1526
            + dq_i2 * x1501 * x1530
            + dq_i2 * x1510
            + dq_i2 * x1513
            + dq_i2 * x1517
            + dq_i2 * x1519 * x1522
            + dq_i2 * x1523
            + dq_i2 * x1527
            + dq_i2 * x1537 * x1540
            + dq_i2 * x1538
            - dq_i2
            * x31
            * (x1512 * x1587 + x1515 * x1568 + x1520 * x1590 + x1525 * x1569 + x1529 * x1570 + x1539 * x1571 + x1599)
            - x1365 * (x129 * x1569 + x141 * x1571 + x1511 * x278 + x1519 * x247 + x1568 * x52 + x1570 * x178)
            - x1369 * (x1550 * x1579 + x1559 * x1576 + x1563 * x1582 + x1573 * x1574 + x1584 * x1585)
            - x1373 * (x1459 * x1562 + x1507 * x1567 + x1557 * x1559 + x1563 * x1565)
            - x1377 * (x1427 * x1555 + x1544 * x1553 + x1550 * x1552)
            - x1379 * (x1542 * x1544 + x1547 * x1548)
            - x1381
            * (x1569 * x1594 + x1570 * x1596 + x1571 * x1598 + x1574 * x1589 + x1586 * x1587 * x365 + x1590 * x1592)
            - x1446 * x1536
        )
        - x1360 * (x1319 * x1356 + x1323 * x1358 + x1330 * x1354 + x1337 * x1352 + x1344 * x1350 + x1347 * x1349)
        + x1385
        * (
            x1363 * x1364
            + x1363 * x545
            - x1365 * x541
            + x1366 * x1368
            + x1369 * x1372
            + x1373 * x1376
            + x1377 * x1378
            + x1379 * x1380
            + x1381 * x1382
            - x1384 * x405
        )
        + x823
        * (
            -x1365 * x758
            + x1369 * x1399
            + x1373 * x1400
            + x1377 * x1401
            + x1381 * x1402
            + x1387 * x407 * x681
            - x1389 * x405
            - x1390 * x405
            + x1394 * x1395
            + x1404 * x1405
        )
        + x906
        * (
            -x1365 * x845
            + x1369 * x1431
            + x1373 * x1434
            + x1379 * x1424
            + x1381 * x1435
            + x1395 * x1428
            - x1408 * x405
            + x1409 * x1415
            - x1417 * x405
            - x1423 * x405
            + x1437 * x1438
        )
    )
    K_block_list.append(
        ddq_j1 * x670
        - dq_i3 * x1141 * x136
        + x1022
        * (
            x1369 * x1706
            + x1444 * x1685
            + x1628 * x1709
            - x1630 * x939
            + x1633 * x1695
            + x1634 * x1692
            + x1635 * x1708
            + x1644 * x1703
            - x1682 * x411
            - x1683 * x411
            - x1687 * x411
            - x1689 * x411
        )
        + x1028 * x1602
        - x1036 * x1610
        + x1038 * x136
        + x1048 * x136
        + x1052 * x136
        + x1054 * x136
        + x1056 * x136
        + x1058 * x136
        + x1059 * x331
        - x1066 * x239
        - x1074 * x271
        - x1087 * x313
        + x1090 * x1297
        + x1092 * x1603
        + x1093 * x1608
        + x1095 * x1609
        - x1096 * x1361
        - x1112 * x396
        + x1142
        * (
            dq_j1 * x668 * (x1114 * x664 + x1116 * x652 + x1134 * x650 + x1137 * x647 + x1140 * x644 + x1743)
            - x1057 * x1630
            + x1369 * (x1099 * x664 + x1102 * x1731 + x1485 * x1705 + x1741 * x972 + x1742 * x478)
            + x1474 * x1685
            + x1632 * (x1049 * x1673 + x1076 * x652 + x1078 * x1725 + x1082 * x1701)
            + x1633 * (x1061 * x1722 + x1068 * x1721 + x1740 * x263)
            + x1634 * (x1061 * x1691 + x1740 * x223)
            + x1644 * (x1090 * x273 + x1094 * x1730 + x1485 * x1696 + x1741 * x913 + x1742 * x471)
            - x1735 * x411
            - x1736 * x411
            - x1737 * x411
            - x1738 * x411
            - x1739 * x411
        )
        + x1266
        * (
            dq_i3 * x1362 * x1734
            - x1174 * x1630
            + x1315 * x1685
            + x1632 * x1727
            + x1633 * x1723
            + x1634 * x1720
            + x1635 * x1732
            + x1644 * x1728
            - x1711 * x411
            - x1713 * x411
            - x1715 * x411
            - x1717 * x411
            - x1718 * x411
        )
        + x1299 * x1601
        - x1346 * x1621
        - x136
        * (
            dq_i3 * x1516 * x1710
            + dq_i3 * x1522 * x1746
            + dq_i3 * x1744
            + dq_i3 * x1745
            + dq_i3 * x1747
            + dq_i3 * x1749
            + dq_i3 * x1752
            - dq_i3 * x31 * (x1515 * x1759 + x1520 * x1764 + x1525 * x1760 + x1529 * x1761 + x1539 * x1762 + x1765)
            - x1369 * (x1589 * x1763 + x1592 * x1764 + x1594 * x1760 + x1596 * x1761 + x1598 * x1762)
            + x1526 * x1748
            + x1530 * x1750
            - x1536 * x1685
            + x1540 * x1753
            - x1630 * (x129 * x1760 + x141 * x1762 + x1746 * x247 + x1759 * x52 + x1761 * x178)
            - x1632 * (x1557 * x1757 + x1562 * x1697 + x1565 * x1758 + x1567 * x1726)
            - x1633 * (x1552 * x1756 + x1553 * x1754 + x1555 * x1670)
            - x1634 * (x1542 * x1754 + x1548 * x1755)
            - x1635 * (x1573 * x1763 + x1576 * x1757 + x1579 * x1756 + x1582 * x1758 + x1583 * x1585 * x1730)
        )
        + x1604 * x1607
        - x1613 * x1614
        - x1616 * x1617
        - x1619 * x1620
        - x1627 * (x1320 * x1625 + x1324 * x1626 + x1331 * x1624 + x1338 * x1623 + x1344 * x1622)
        + x1636
        * (
            x1364 * x1629
            + x1369 * x1382
            + x1372 * x1635
            + x1376 * x1632
            + x1378 * x1633
            + x1380 * x1634
            - x1384 * x411
            + x1629 * x545
            - x1630 * x541
            + x1631 * x411
        )
        + x823
        * (
            x1369 * x1647
            + x1387 * x412 * x644
            - x1630 * x760
            + x1632 * x1649
            + x1633 * x1650
            + x1635 * x1651
            - x1638 * x411
            - x1639 * x411
            + x1643 * x1644
            + x1652 * x1653
        )
        + x906
        * (
            x1369 * x1672
            - x1630 * x848
            + x1632 * x1675
            + x1634 * x1666
            + x1635 * x1677
            + x1644 * x1671
            - x1656 * x411
            + x1657 * x1660
            - x1662 * x411
            - x1665 * x411
            + x1678 * x1679
        )
    )
    K_block_list.append(
        ddq_j1 * x643
        - dq_i4 * x1021 * x136
        + x1022
        * (
            dq_j1 * x1897 * x641
            + x1373 * x1895
            + x1444 * x1866
            + x1632 * x1896
            - x1790 * x941
            + x1791 * x1890
            + x1792 * x1888
            + x1806 * x1894
            - x1883 * x415
            - x1884 * x415
            - x1885 * x415
            - x1886 * x415
        )
        + x1142
        * (
            dq_i4 * x1628 * x1881
            - x1053 * x1790
            + x1373 * x1879
            + x1474 * x1866
            + x1791 * x1875
            + x1792 * x1874
            + x1793 * x1880
            + x1806 * x1877
            - x1864 * x415
            - x1865 * x415
            - x1868 * x415
            - x1870 * x415
        )
        + x1266
        * (
            -x1172 * x1790
            + x1469 * x1861
            + x1632 * x1857
            + x1791 * x1851
            + x1792 * x1848
            + x1793 * x1860
            + x1806 * x1852
            - x1839 * x415
            - x1841 * x415
            + x1842 * x1843
            - x1845 * x415
            - x1846 * x415
        )
        + x1304 * x965
        - x1316 * x1777
        - x1327 * x1780
        - x1340 * x1781
        + x136 * x922
        + x136 * x932
        + x136 * x936
        + x136 * x938
        + x136 * x940
        + x136 * x942
        + x136
        * (
            dq_i1 * dq_i4 * dq_j1 * x1918
            + 2 * dq_i2 * dq_i4 * dq_j1 * x1913
            + 2 * dq_i3 * dq_i4 * dq_j1 * x1915
            + 2 * dq_i4 * dq_i5 * dq_j1 * x1911
            + 2 * dq_i4 * dq_i6 * dq_j1 * x1910
            + 2
            * dq_i4
            * dq_i7
            * dq_j1
            * sigma_kin_v_7_1
            * sigma_kin_v_7_4
            * x137
            * x138
            * x140
            * x144
            * x146
            * x148
            * x150
            * x153
            * x203
            - dq_i4 * x1898
            - dq_i4 * x1901
            - dq_i4 * x1902
            - dq_i4 * x1903
            - dq_i4 * x1904
            - dq_i4 * x1905
            - dq_i4 * x1907
            - dq_i4 * x1909
            + 2 * dq_j1 * x1916 * x641
            - x1790 * x249
        )
        - x1361 * x970
        + x1602 * x909
        + x1608 * x967
        + x1609 * x969
        - x1610 * x920
        - x1619 * x1782
        + x1767 * x1770
        + x1771 * x1773
        - x1787 * (x1320 * x1785 + x1324 * x1786 + x1331 * x1784 + x1338 * x1783)
        + x1794
        * (
            x1364 * x1789
            + x1372 * x1632
            + x1373 * x1382
            + x1376 * x1793
            + x1378 * x1791
            + x1380 * x1792
            - x1384 * x415
            + x1631 * x415
            + x1789 * x545
            - x1790 * x541
        )
        - x239 * x951
        - x271 * x961
        + x331 * x943
        - x359 * x996
        - x396 * x981
        + x823
        * (
            x1373 * x1809
            + x1632 * x1811
            - x1790 * x762
            + x1791 * x1812
            + x1793 * x1813
            + x1795 * x1798
            - x1800 * x415
            - x1801 * x415
            + x1805 * x1806
            + x1814 * x1815
        )
        + x906
        * (
            x1373 * x1832
            + x1632 * x1835
            + x1788 * x1838
            - x1790 * x850
            + x1792 * x1827
            + x1793 * x1837
            + x1795 * x1821
            + x1806 * x1831
            - x1818 * x415
            - x1823 * x415
            - x1826 * x415
        )
    )
    K_block_list.append(
        ddq_j1 * x612
        - dq_i5 * x136 * x905
        + x1022
        * (
            dq_i5 * x1788 * x2015
            + x1377 * x2011
            + x1444 * x1989
            + x1633 * x2013
            - x1938 * x935
            + x1939 * x2009
            + x1940 * x2014
            + x1951 * x2010
            - x2005 * x419
            - x2006 * x419
            - x2007 * x419
        )
        + x1142
        * (
            -x1051 * x1938
            + x1377 * x2001
            + x1474 * x1989
            + x1679 * x2004
            + x1791 * x2002
            + x1939 * x1998
            + x1940 * x2003
            + x1951 * x1999
            - x1988 * x419
            - x1991 * x419
            - x1994 * x419
        )
        + x1266
        * (
            -x1168 * x1938
            + x1438 * x1986
            + x1633 * x1978
            + x1791 * x1982
            + x1842 * x1965
            + x1939 * x1970
            + x1940 * x1985
            + x1951 * x1971
            - x1964 * x419
            - x1967 * x419
            - x1968 * x419
        )
        - x1316 * x1929
        - x1333 * x1930
        - x1334 * x1780
        + x136 * x832
        + x136 * x843
        + x136 * x846
        + x136 * x849
        + x136 * x851
        + x136 * x853
        + x136
        * (
            dq_i1 * dq_i5 * dq_j1 * x2044
            + 2 * dq_i2 * dq_i5 * dq_j1 * x2040
            + 2 * dq_i3 * dq_i5 * dq_j1 * x2041
            + 2 * dq_i4 * dq_i5 * dq_j1 * x2042
            + 2 * dq_i5 * dq_i6 * dq_j1 * x2039
            + 2
            * dq_i5
            * dq_i7
            * dq_j1
            * sigma_kin_v_7_1
            * sigma_kin_v_7_5
            * x137
            * x138
            * x140
            * x144
            * x146
            * x148
            * x150
            * x153
            * x203
            - dq_i5 * x2032
            - dq_i5 * x2033
            - dq_i5 * x2034
            - dq_i5 * x2035
            - dq_i5 * x2037
            - dq_i5 * x2038
            + 2 * dq_j1 * x2043 * x610
            - x1938 * x243
        )
        - x1361 * x867
        - x1414 * x207 * x838
        + x1422 * x1925
        + x1603 * x863
        - x1616 * x1931
        + x1767 * x1921
        + x1771 * x1923
        + x1924 * x772
        - x1935 * (x1320 * x1933 + x1324 * x1934 + x1331 * x1932)
        + x1941
        * (
            x1364 * x1937
            + x1372 * x1633
            + x1376 * x1791
            + x1377 * x1382
            + x1378 * x1940
            + x1380 * x1939
            - x1384 * x419
            + x1631 * x419
            + x1937 * x545
            - x1938 * x541
        )
        - x239 * x861
        - x313 * x888
        + x331 * x855
        - x359 * x880
        - x396 * x872
        + x823
        * (
            x1377 * x1954
            + x1633 * x1956
            + x1791 * x1957
            - x1938 * x765
            + x1940 * x1959
            + x1942 * x1943
            - x1945 * x419
            - x1946 * x419
            + x1950 * x1951
            + x1960 * x1961
        )
        + x906
        * (
            dq_j1 * x2031 * x610
            + x1377 * x2027
            + x1414 * x1942 * x2018
            + x1633 * x2029
            + x1791 * x2030
            - x1938 * x852
            + x1939 * x2023
            + x1951 * x2026
            - x2016 * x419
            - x2020 * x419
            - x2022 * x419
        )
    )
    K_block_list.append(
        ddq_j1 * x514
        - dq_i6 * x136 * x822
        + x1022
        * (
            x1379 * x2103
            + x1444 * x2084
            + x1634 * x2104
            + x1815 * x2108
            + x1939 * x2106
            - x2058 * x931
            + x2060 * x2107
            + x2069 * x2102
            - x2100 * x423
            - x2101 * x423
        )
        + x1142
        * (
            -x1047 * x2058
            + x1379 * x2092
            + x1474 * x2084
            + x1653 * x2099
            + x1792 * x2093
            + x1939 * x2095
            + x2060 * x2098
            + x2069 * x2090
            - x2086 * x423
            - x2089 * x423
        )
        + x1266
        * (
            -x1165 * x2058
            + x1405 * x2083
            + x1634 * x2074
            + x1792 * x2077
            + x1842 * x2063
            + x1939 * x2080
            + x2060 * x2082
            - x2066 * x423
            - x2067 * x423
            + x2068 * x2069
        )
        + x1311 * x773
        + x1313 * x776
        - x1326 * x2052
        - x1327 * x1929
        - x1334 * x1777
        + x136 * x749
        + x136 * x759
        + x136 * x761
        + x136 * x763
        + x136 * x766
        + x136 * x768
        + x136
        * (
            -dq_i6 * x2135
            - dq_i6 * x2136
            - dq_i6 * x2140
            - dq_i6 * x2141
            + dq_i6 * x2147 * x31
            + x1379 * x2142
            + x1634 * x2143
            + x1792 * x2144
            + x1939 * x2145
            - x2058 * x216
            + x2060 * x2146
            + x2062 * x2138
        )
        - x1361 * x778
        - x1613 * x2053
        + x1771 * x2051
        - x1797 * x207 * x209
        + x2045 * x2049
        - x2056 * (x1320 * x2054 + x1324 * x2055)
        + x2061
        * (
            x1364 * x2057
            + x1372 * x1634
            + x1376 * x1792
            + x1378 * x1939
            + x1379 * x1382
            + x1380 * x2060
            + x187 * x199 * x2059
            + x2057 * x545
            - x2058 * x541
            - x2059 * x451
        )
        - x271 * x808
        - x313 * x800
        + x331 * x770
        - x359 * x795
        - x396 * x788
        + x823
        * (
            dq_j1 * x2134 * x512
            + x1379 * x2130
            + x1634 * x2131
            + x1792 * x2132
            + x1939 * x2133
            - x2058 * x767
            + x2062 * x2124
            + x2069 * x2128
            - x2125 * x423
            - x2126 * x423
        )
        + x906
        * (
            x1379 * x2118
            + x1634 * x2119
            + x1792 * x2120
            + x1961 * x2123
            - x2058 * x842
            + x2060 * x2122
            + x2062 * x2111
            + x2069 * x2117
            - x2113 * x423
            - x2115 * x423
        )
    )
    K_block_list.append(
        x1306 * x2170
        + x1307 * x2161 * x473
        - x187 * x2180
        + x2045 * x2150
        - x210 * x2171 * x544
        - x211 * x2171 * x547
        + x2151 * x2167 * x331
        + x2153 * x275 * x645
        + x2153 * x281 * x682
        + x2154 * x250
        + x2155 * x244
        + x2156 * x217
        + x2159 * x2160
        + x2162 * x473
        + x2163 * x2164
        + x2163 * x2165
        + x2163 * x2166
        + x2174 * x473
        - x2181 * x2182 * x396
        - x2184 * x313 * x497
        - x2184 * x359 * x489
        - x2186 * x267 * x271
        - x2187 * x231 * x239
        - x2188 * x430 * x431 * x435
        + x2209
        * (
            -x1156 * x2191
            + x1156 * x2196
            + x1156 * x2204
            + x1363 * x2190
            + x1363 * x2194
            + x2197 * x2198
            + x2198 * x2199
            + x2198 * x2200
            + x2198 * x2202
            - x2205 * x2206
        )
        + x2218
        * (
            -x1035 * x2191
            + x1035 * x2196
            + x1035 * x2204
            - x1035 * x2216
            + x1629 * x2211
            + x1629 * x2213
            + x2199 * x2215
            + x2200 * x2215
            + x2202 * x2215
            + x2214 * x2215
        )
        + x2228
        * (
            x1789 * x2220
            + x1789 * x2223
            - x2191 * x919
            + x2196 * x919
            + x2197 * x2224
            + x2200 * x2224
            + x2202 * x2224
            + x2204 * x919
            + x2214 * x2224
            - x2216 * x919
        )
        + x2236
        * (
            x1937 * x2230
            + x1937 * x2232
            - x2191 * x828
            + x2196 * x828
            + x2197 * x2233
            + x2199 * x2233
            + x2202 * x2233
            + x2204 * x828
            + x2214 * x2233
            - x2216 * x828
        )
        + x2243
        * (
            x1796 * x2195
            + x2057 * x2238
            + x2057 * x2239
            - x2191 * x744
            + x2197 * x2240
            + x2199 * x2240
            + x2200 * x2240
            + x2204 * x744
            + x2214 * x2240
            - x2216 * x744
        )
        + x2250
        * (
            dq_j1 * x2244 * x2245
            + x1936 * x2247
            - x1936 * x569
            + x2197 * x2249
            + x2199 * x2249
            + x2200 * x2249
            + x2202 * x2249
            + x2214 * x2249
            - x2216 * x520
            + x2248 * x406 * x524
        )
        + x2257
        * (
            -x185 * x567
            - x1936 * x2252
            + x2197 * x2254
            + x2199 * x2254
            + x2200 * x2254
            + x2202 * x2254
            + x2203 * x2254
            + x2214 * x2254
            + x2251 * x546
            + x2253 * x546
            - x2255 * x406
        )
    )
    K_block_list.append(
        ddq_j2 * x2377
        - dq_i1 * x1266 * x2349
        + x1022
        * (
            x1444 * x2378
            + x1453 * x2364
            + x1456 * x2362
            + x1463 * x2366
            + x1465 * x2355
            + x1467 * x2373
            + x1468 * x2380
            - x2379
            - x515 * x976
            - x515 * x977
            - x515 * x979
            - x515 * x980
        )
        + x1142
        * (
            -x1106 * x515
            - x1107 * x515
            - x1108 * x515
            - x1109 * x515
            - x1111 * x515
            + x1474 * x2378
            + x1478 * x2364
            + x1481 * x2362
            + x1483 * x2358
            + x1489 * x2366
            + x1491 * x2373
            + x1492 * x2350
            - x2381
        )
        + x1266 * x2263
        + x1266 * x2272
        + x1266 * x2286
        + x1266 * x2293
        + x1266 * x2308
        - x1266
        * (
            dq_i1 * x1253
            + dq_i1 * x1255
            + dq_i1 * x1257
            + dq_i1 * x1260
            + dq_i1 * x1262
            + dq_i1 * x1264
            + dq_i1 * x2387 * x2388 * x363
            + dq_i1 * x2395 * x246
            + dq_i1 * x2398 * x378
            + dq_i1 * x2401 * x384
            + dq_i1 * x2406 * x390
            - dq_i1
            * x32
            * (x1509 + x2388 * x2424 + x2390 * x2418 + x2394 * x2427 + x2397 * x2419 + x2400 * x2420 + x2405 * x2421)
            - x2310 * (x1176 * x2393 + x2387 * x702 + x2418 * x369 + x2419 * x591 + x2420 * x477 + x2421 * x479)
            - x2355 * (x1576 * x2414 + x1579 * x2411 + x1582 * x2416 + x1584 * x2423 * x397 + x2418 * x2422)
            - x2358 * (x1459 * x2415 + x1507 * x2417 + x1557 * x2414 + x1565 * x2416)
            - x2362 * (x1427 * x2412 + x1552 * x2411 + x1553 * x2408)
            - x2364 * (x1542 * x2408 + x1547 * x2409)
            - x2366
            * (x1347 * x2424 * x365 + x2384 * x2425 + x2419 * x2429 + x2420 * x2430 + x2421 * x2431 + x2427 * x2428)
            - x2378 * x2404
            + x2391 * x2392
        )
        + x1288 * x1296
        + x1289 * x1297
        + x1292 * x1305
        + x1293 * x1312
        + x1294 * x1314
        + x1302 * x2261
        - x1317 * x2277
        - x1328 * x2289
        - x1335 * x2296
        - x1342 * x2309
        + x136 * x2311
        + x136
        * (
            2
            * dq_i1
            * dq_i2
            * dq_j2
            * (x1288 * x277 + x1292 * x675 + x1293 * x677 + x1294 * x679 + x2382 * x375 + x2385 * x370)
            + 2 * dq_i1 * dq_i3 * dq_j2 * (x1287 * x692 + x1429 * x295 + x1464 * x224 + x1558 * x2386 + x2385 * x336)
            + 2 * dq_i1 * dq_i4 * dq_j2 * (x1278 * x1482 + x1280 * x1507 + x1432 * x296 + x2382 * x2383)
            + 2 * dq_i1 * dq_i5 * dq_j2 * (x1270 * x1479 + x1272 * x1391 + x1274 * x1480)
            + 2 * dq_i1 * dq_i6 * dq_j2 * (x1268 * x1391 + x1269 * x681)
            + 2
            * dq_i1
            * dq_i7
            * dq_j2
            * sigma_kin_v_7_1
            * sigma_kin_v_7_2
            * x137
            * x138
            * x140
            * x144
            * x146
            * x148
            * x150
            * x153
            * x203
            + dq_j2
            * x11
            * (sigma_kin_v_4_2 * x1917 + x134 * x675 + x1599 + x183 * x677 + x201 * x679 + x277 * x38 + x279 * x60)
            - x2311
            - x367 * x515
            - x371 * x515
            - x376 * x515
            - x382 * x515
            - x388 * x515
            - x394 * x515
        )
        + x156 * x2264 * x2266
        - x2137 * x2268
        + x2258 * x2259 * x2260
        - x2281 * x2282
        - x2302 * x2303
        + x2317 * x2318
        - x2319 * (x1586 * x737 + x1588 * x738 + x1591 * x739 + x1593 * x740 + x1595 * x741 + x1597 * x742)
        - x2328 * x2329
        - x2338 * x2339
        - x2347 * x2348
        + x574
        * (
            x1364 * x2351
            - x2310 * x550
            + x2351 * x545
            + x2352 * x2354
            + x2355 * x2357
            + x2358 * x2361
            + x2362 * x2363
            + x2364 * x2365
            + x2366 * x2367
            - x2370 * x515
        )
        + x823
        * (
            x1394 * x2366
            + x1399 * x2355
            + x1400 * x2358
            + x1401 * x2362
            + x1402 * x2373
            + x1404 * x2374
            + x1797 * x2371 * x681
            - x2372
            - x515 * x784
            - x515 * x787
        )
        + x906
        * (
            x1415 * x2371
            + x1424 * x2364
            + x1428 * x2366
            + x1431 * x2355
            + x1434 * x2358
            + x1435 * x2373
            + x1437 * x2376
            - x2375
            - x515 * x869
            - x515 * x870
            - x515 * x871
        )
    )
    K_block_list.append(
        ddq_j1 * x2377
        + ddq_j3 * x2578
        + ddq_j4 * x2568
        + ddq_j5 * x2554
        + ddq_j6 * x2533
        + x1022
        * (
            2 * dq_i1 * (x2663 * x965 + x2705 * x907 + x2706 * x967 + x2707 * x969 + x2708)
            + 2 * dq_i2 * dq_i7 * dq_j2 * x1357 * x140 * x144 * x145 * x148 * x150 * x153 * x203 * x434 * x919
            + 2 * dq_i2 * (x2663 * x973 + x2705 * x971 + x2706 * x974 + x2707 * x975 + x2709)
            + 2 * dq_i3 * (x2663 * x985 + x2706 * x988 + x2707 * x990 + x2710 * x984 + x2716)
            + dq_i4
            * (
                x1001 * x2663
                + x1002 * x2663
                + x1004 * x2706
                + x1005 * x2594 * x32
                + x1007 * x2707
                + x1008 * x2596 * x32
                + x2592 * x32 * x999
                + x2710 * x998
                + x2725
            )
            + 2 * dq_i5 * (x263 * x2693 + x2654 * x957 + x2663 * x954 + x2703)
            + 2 * dq_i6 * (x223 * x2693 + x2616 * x946 + x2698)
            + 2 * dq_i7 * sigma_kin_v_7_2 * x140 * x144 * x145 * x148 * x150 * x153 * x195 * x203 * x434 * x919
            - x2541 * x2692
            - x2677 * x405
            - x2678 * x405
            - x2679
            - x2681 * x405
            - x2682 * x405
            - x2684
            - x2686
            - x2689
            - x2691
            - x537 * x937
        )
        + x1142
        * (
            2 * dq_i1 * (x1090 * x2757 + x1092 * x2652 + x1093 * x2706 + x1095 * x2707 + x2758 * x907 + x2759)
            + 2 * dq_i2 * dq_i7 * dq_j2 * x1035 * x1357 * x140 * x143 * x146 * x148 * x150 * x153 * x203 * x434
            + 2 * dq_i2 * (x1099 * x2760 + x1100 * x2652 + x1101 * x2706 + x1104 * x2707 + x2758 * x971 + x2761)
            + dq_i3
            * (
                x1114 * x2760
                + x1115 * x2597 * x32
                + x1116 * x2710
                + x1117 * x2710
                + x1119 * x2652
                + x1120 * x2652
                + x1122 * x2706
                + x1123 * x2706
                + x1125 * x2707
                + x1127 * x2707
                + x2767
            )
            + 2 * dq_i4 * (x1076 * x2710 + x1079 * x2706 + x1082 * x2707 + x2663 * x2751 + x2756)
            + 2 * dq_i5 * (x1069 * x2652 + x1070 * x2654 + x263 * x2742 + x2750)
            + 2 * dq_i6 * (x1062 * x2616 + x223 * x2742 + x2746)
            + 2 * dq_i7 * sigma_kin_v_7_2 * x1035 * x140 * x143 * x146 * x148 * x150 * x153 * x195 * x203 * x434
            - x1055 * x537
            - x2541 * x2741
            - x2726 * x405
            - x2727 * x405
            - x2728 * x405
            - x2729
            - x2730 * x405
            - x2731 * x405
            - x2733
            - x2735
            - x2738
            - x2740
        )
        + x1177 * x2476
        + x1237 * x35
        + x1239 * x2432
        + x1242 * x2433
        + x1245 * x2436
        + x1248 * x2438
        + x1250 * x198
        + x1252 * x35
        + x1254 * x2432
        + x1259 * x2436
        + x1261 * x2438
        + x1263 * x198
        + x1266 * x2437
        + x1266 * x2444
        + x1266 * x2455
        + x1266 * x2464
        + x1266 * x2475
        + x1266
        * (
            2
            * dq_i1
            * (x1144 * x2770 + x1154 * x2772 + x1161 * x2773 + x1227 * x2757 + x1230 * x2771 + x1234 * x2774 + x2498)
            + 2 * dq_i2 * dq_i7 * dq_j2 * x1156 * x1357 * x139 * x144 * x146 * x148 * x150 * x153 * x203 * x434
            - dq_i2 * x2509
            - dq_i2 * x2510
            - dq_i2 * x2511
            - dq_i2 * x2512
            - dq_i2 * x2513
            - dq_i2 * x2514
            - dq_i2 * x2515
            - dq_i2 * x2516
            - dq_i2 * x2517
            - dq_i2 * x2518
            - dq_i2 * x2519
            - dq_i2 * x2520
            + dq_i2
            * (
                x1237 * x2770
                + x1238 * x2477 * x32
                + x1239 * x2760
                + x1242 * x2771
                + x1243 * x2479 * x32
                + x1245 * x2772
                + x1246 * x2480 * x32
                + x1248 * x2773
                + x1249 * x2481 * x32
                + x1250 * x2774
                + x1251 * x2482 * x32
                + x1254 * x2760
                + x2521
            )
            + 2 * dq_i3 * (x1213 * x2760 + x1214 * x2704 + x1215 * x2652 + x1216 * x2706 + x1219 * x2707 + x2497)
            + 2 * dq_i4 * (x1199 * x2704 + x1203 * x2706 + x1205 * x2707 + x2489 + x2663 * x2775)
            + 2 * dq_i5 * (x1190 * x2652 + x1191 * x2768 + x1192 * x2654 + x2468)
            + 2 * dq_i6 * (x1181 * x2768 + x1184 * x2616 + x2448)
            + 2 * dq_i7 * sigma_kin_v_7_2 * x1156 * x139 * x144 * x146 * x148 * x150 * x153 * x195 * x203 * x434
            - x1177 * x537
            - x2437
            - x2444
            - x2455
            - x2464
            - x2475
            - x2483 * x2541
        )
        - x1315 * x2439
        + x1359
        * (
            ddq_i2 * (x1348 * x695 + x1350 * x655 + x1351 * x623 + x1353 * x582 + x1355 * x466 + x1357 * x468)
            + x2579 * x711
            + x2580 * x711
            + x2581 * x711
            + x2582 * x711
            + x2583 * x711
            + x2584 * x711
            - x2585 * x683
            - x482 * (x221 * x2587 + x2588 * x509)
            - x594 * (x255 * x2590 + x2585 * x504 + x2586 * x263)
            - x635 * (x1353 * x603 + x2592 * x287 + x2594 * x299 + x2596 * x306)
            - x666 * (x2589 * x597 + x2591 * x343 + x2593 * x485 + x2595 * x489 + x2597 * x334)
            - x671 * (x133 * x2314 + x1349 * x37 + x182 * x2315 + x199 * x2316 + x2312 * x59 + x2313 * x85)
            - x712 * (x2579 + x2580 + x2581 + x2582 + x2583 + x2584)
        )
        + x136
        * (
            dq_i1
            * (
                x125 * x2314 * x32
                + x1295 * x2771
                + x134 * x2772
                + x1349 * x19 * x32
                + x174 * x2315 * x32
                + x183 * x2773
                + x185 * x2316 * x32
                + x201 * x2774
                + x2312 * x32 * x39
                + x2313 * x32 * x65
                + x2349
                + x2757 * x60
                + x2770 * x38
            )
            + 2 * dq_i2 * dq_i7 * dq_j2 * x1357 * x137 * x138 * x140 * x144 * x146 * x148 * x150 * x153 * x203
            + 2
            * dq_i2
            * (x1288 * x2770 + x1289 * x2757 + x1291 * x2771 + x1292 * x2772 + x1293 * x2773 + x1294 * x2774 + x2347)
            + 2 * dq_i3 * (x1283 * x2757 + x1284 * x2704 + x1285 * x2652 + x1286 * x2706 + x1287 * x2707 + x2338)
            + 2 * dq_i4 * (x1277 * x2704 + x1279 * x2706 + x1281 * x2707 + x2328 + x2663 * x2769)
            + 2 * dq_i5 * (x1271 * x2652 + x1273 * x2613 + x1275 * x2654 + x2302)
            + 2 * dq_i6 * (x1268 * x2768 + x1269 * x2614 + x2281)
            + 2 * dq_i7 * sigma_kin_v_7_2 * x137 * x138 * x140 * x144 * x146 * x148 * x150 * x153 * x195 * x203
            - x2263
            - x2272
            - x2286
            - x2293
            - x2308
            - x2317 * x2541
            - x2340 * x405
            - x2341 * x405
            - x2343 * x405
            - x2344 * x405
            - x2345 * x405
            - x2346 * x405
            - x280 * x537
        )
        + x1385
        * (
            x1364 * x188
            + x188 * x545
            - x2534
            + x2537 * x522
            + x2537 * x536
            + x2538 * x527
            + x2539 * x529
            + x2540 * x417
            + x2540 * x421
            - x2541 * x550
            + x2543
            + x554
        )
        + x2260 * x2435 * (sigma_pot_2_c * x25 + sigma_pot_2_s * x22)
        - x2282 * x2448
        - x2303 * x2468
        + x2318 * x2483
        - x2329 * x2489
        - x2339 * x2497
        + x2433 * x2434
        - x2498 * x2499
        - x2507 * x2508 * x433
        - 4 * x2521 * x32
        + x823
        * (
            2 * dq_i1 * (x2613 * x773 + x2614 * x776 + x2615)
            + 2 * dq_i2 * dq_i7 * dq_j2 * x1357 * x140 * x144 * x146 * x148 * x149 * x153 * x203 * x434 * x744
            + 2 * dq_i2 * (x2613 * x780 + x2617 * x782 + x2618)
            + 2 * dq_i3 * (x2613 * x789 + x2617 * x790 + x2621)
            + 2 * dq_i4 * (x2613 * x796 + x2617 * x797 + x2624)
            + 2 * dq_i5 * (x2616 * x806 + x2625 * x803 + x2629)
            + dq_i6 * (x2587 * x32 * x811 + x2614 * x812 + x2616 * x814 + x2625 * x810 + x2634)
            + 2 * dq_i7 * sigma_kin_v_7_2 * x140 * x144 * x146 * x148 * x149 * x153 * x195 * x203 * x434 * x744
            - x2541 * x2612
            - x2599
            - x2601 * x405
            - x2603 * x405
            - x2605
            - x2607
            - x2609
            - x2611
            - x537 * x758
        )
        + x906
        * (
            2 * dq_i1 * (x1422 * x2654 + x2652 * x863 + x2653 * x772 + x2655)
            + 2 * dq_i2 * dq_i7 * dq_j2 * x1357 * x140 * x144 * x146 * x147 * x150 * x153 * x203 * x434 * x828
            + 2 * dq_i2 * (x2652 * x868 + x2653 * x779 + x2654 * x2656 + x2657)
            + 2 * dq_i3 * (x2652 * x875 + x2653 * x485 + x2654 * x2658 + x2662)
            + 2 * dq_i4 * (x1433 * x2654 + x2653 * x494 + x2663 * x2664 + x2668)
            + dq_i5
            * (x1436 * x2654 + x2590 * x32 * x891 + x2625 * x892 + x2625 * x893 + x2652 * x890 + x2654 * x896 + x2675)
            + 2 * dq_i6 * (x2616 * x858 + x2625 * x857 + x2651)
            + 2 * dq_i7 * sigma_kin_v_7_2 * x140 * x144 * x146 * x147 * x150 * x153 * x195 * x203 * x434 * x828
            - x2541 * x2648
            - x2635 * x405
            - x2636
            - x2637 * x405
            - x2639 * x405
            - x2641
            - x2643
            - x2645
            - x2647
            - x537 * x845
        )
    )
    K_block_list.append(
        ddq_j2 * x2578
        - dq_i3 * x1266 * x2767
        + x1022
        * (
            x1444 * x2839
            + x2355 * x2849
            - x2688 * x2795
            + x2793 * x2856
            + x2798 * x2845
            + x2799 * x2843
            + x2800 * x2855
            + x2811 * x2852
            - x2836 * x411
            - x2837 * x411
            - x2840 * x411
            - x2841 * x411
        )
        + x1028 * x2778
        - x1036 * x2779
        + x1099 * x2776
        + x110 * x1600 * x2777
        + x1100 * x1603
        + x1101 * x1608
        + x1104 * x1609
        + x1142
        * (
            dq_j2 * x668 * (x1114 * x2469 + x1116 * x2853 + x1134 * x2305 + x1137 * x2306 + x1140 * x2569 + x2882)
            + x1474 * x2839
            + x2355 * (x1028 * x2846 + x1090 * x1345 + x1092 * x2825 + x1093 * x2847 + x1095 * x2848)
            - x2739 * x2795
            + x2797 * (x1049 * x2829 + x1076 * x2853 + x1078 * x2868 + x1082 * x2848)
            + x2798 * (x1061 * x2866 + x1068 * x2865 + x263 * x2880)
            + x2799 * (x1062 * x2808 + x223 * x2880)
            + x2811 * (x1028 * x2850 + x1041 * x2851 + x1099 * x2469 + x1100 * x2825 + x1102 * x2881)
            - x2875 * x411
            - x2876 * x411
            - x2877 * x411
            - x2878 * x411
            - x2879 * x411
        )
        + x1266 * x2729
        + x1266 * x2733
        + x1266 * x2735
        + x1266 * x2738
        + x1266 * x2740
        - x1266
        * (
            dq_i3 * x1712 * x2395
            + dq_i3 * x2391 * x2874
            + dq_i3 * x2883
            + dq_i3 * x2885
            + dq_i3 * x2886
            + dq_i3 * x2887
            + dq_i3 * x2888
            - dq_i3 * x32 * (x2390 * x2898 + x2394 * x2903 + x2397 * x2899 + x2400 * x2900 + x2405 * x2901 + x2904)
            + x1748 * x2398
            + x1750 * x2401
            + x1753 * x2406
            - x2355 * (x2425 * x2902 + x2428 * x2903 + x2429 * x2899 + x2430 * x2900 + x2431 * x2901)
            - x2404 * x2839
            - x2795 * (x2393 * x2470 + x2898 * x369 + x2899 * x591 + x2900 * x477 + x2901 * x479)
            - x2797 * (x119 * x2415 * x2896 + x1557 * x2895 + x1565 * x2897 + x2417 * x2869)
            - x2798 * (x1552 * x2892 + x234 * x2412 * x2894 + x2889 * x2893)
            - x2799 * (x2409 * x2891 + x2889 * x2890)
            - x2800 * (x1576 * x2895 + x1579 * x2892 + x1582 * x2897 + x1583 * x2423 * x2881 + x2422 * x2898)
        )
        + x136 * x2381
        + x136
        * (
            dq_i3 * x2350 * x2873
            - x2307 * x2795
            + x2797 * x2870
            + x2798 * x2867
            + x2799 * x2864
            + x2800 * x2872
            + x2811 * x2871
            + x2819 * x2860
            - x2857 * x411
            - x2858 * x411
            - x2859 * x411
            - x2861 * x411
            - x2862 * x411
        )
        + 4 * x1607 * x2265
        - x1614 * x2782
        - x1617 * x2784
        - x1620 * x2786
        - x1627 * (x1589 * x2788 + x2273 * x2791 + x2275 * x2792 + x2287 * x2790 + x2294 * x2789)
        + x1636
        * (
            x1364 * x2794
            + x2355 * x2367
            + x2357 * x2800
            + x2361 * x2797
            + x2363 * x2798
            + x2365 * x2799
            - x2370 * x411
            + x2794 * x545
            - x2795 * x550
            + x2796 * x411
        )
        - x2282 * x2746
        - x2303 * x2750
        - x2309 * x2787
        + x2318 * x2741
        - x2329 * x2756
        - x2348 * x2761
        - x2499 * x2759
        + x823
        * (
            dq_j2 * x2801 * x412
            + x2355 * x2806
            - x2604 * x2795
            + x2797 * x2812
            + x2798 * x2814
            + x2800 * x2815
            - x2802 * x411
            - x2805 * x411
            + x2810 * x2811
            + x2816 * x2817
        )
        + x906
        * (
            x2355 * x2827
            - x2642 * x2795
            + x2797 * x2830
            + x2799 * x2824
            + x2800 * x2832
            + x2811 * x2828
            - x2818 * x411
            + x2819 * x2821
            - x2822 * x411
            - x2823 * x411
            + x2833 * x2834
        )
    )
    K_block_list.append(
        ddq_j2 * x2568
        - dq_i4 * x1266 * x2725
        + x1022
        * (
            dq_j2 * x2996 * x641
            + x1444 * x2972
            + x2358 * x2993
            - x2690 * x2917
            + x2797 * x2995
            + x2918 * x2992
            + x2919 * x2990
            + x2929 * x2994
            - x2985 * x415
            - x2986 * x415
            - x2987 * x415
            - x2988 * x415
        )
        + x1142
        * (
            dq_i4 * x2793 * x2984
            + x1474 * x2972
            + x2358 * x2979
            - x2737 * x2917
            + x2918 * x2977
            + x2919 * x2976
            + x2920 * x2983
            + x2929 * x2981
            - x2970 * x415
            - x2971 * x415
            - x2973 * x415
            - x2974 * x415
        )
        + x1266 * x2679
        + x1266 * x2684
        + x1266 * x2686
        + x1266 * x2689
        + x1266 * x2691
        + x1266
        * (
            2 * dq_i1 * dq_i4 * dq_j2 * x3009
            + dq_i2 * dq_i4 * dq_j2 * x3013
            + 2 * dq_i3 * dq_i4 * dq_j2 * x3011
            + 2 * dq_i4 * dq_i5 * dq_j2 * x3007
            + 2 * dq_i4 * dq_i6 * dq_j2 * x3006
            + 2
            * dq_i4
            * dq_i7
            * dq_j2
            * sigma_kin_v_7_2
            * sigma_kin_v_7_4
            * x1156
            * x139
            * x144
            * x146
            * x148
            * x150
            * x153
            * x203
            * x434
            - dq_i4 * x2997
            - dq_i4 * x2999
            - dq_i4 * x3000
            - dq_i4 * x3001
            - dq_i4 * x3002
            - dq_i4 * x3003
            - dq_i4 * x3004
            - dq_i4 * x3005
            + 2 * dq_j2 * x3012 * x641
            - x2463 * x2917
        )
        + x1304 * x973
        - x1316 * x2911
        - x1327 * x2912
        + x136 * x2379
        + x136
        * (
            -x2292 * x2917
            + x2380 * x2969
            + x2797 * x2965
            + x2918 * x2960
            + x2919 * x2957
            + x2920 * x2968
            + x2921 * x2953
            + x2929 * x2961
            - x2951 * x415
            - x2952 * x415
            - x2954 * x415
            - x2955 * x415
        )
        + x1608 * x974
        + x1609 * x975
        + x1773 * x2908
        - x1782 * x2786
        - x1787 * (x1784 * x2288 + x1785 * x2274 + x1786 * x2276 + x2294 * x2914)
        + x1794
        * (
            x1364 * x2916
            + x2357 * x2797
            + x2358 * x2367
            + x2361 * x2920
            + x2363 * x2918
            + x2365 * x2919
            - x2370 * x415
            + x2796 * x415
            + x2916 * x545
            - x2917 * x550
        )
        - x2282 * x2698
        - x2296 * x2913
        - x2303 * x2703
        + x2318 * x2692
        - x2339 * x2716
        - x2348 * x2709
        - x2499 * x2708
        + x2778 * x909
        - x2779 * x920
        + x2905 * x2907
        + x823
        * (
            x1797 * x2555 * x2921
            + x2358 * x2924
            - x2606 * x2917
            + x2797 * x2930
            + x2918 * x2932
            + x2920 * x2933
            - x2922 * x415
            - x2923 * x415
            + x2928 * x2929
            + x2934 * x2935
        )
        + x906
        * (
            x2358 * x2944
            - x2644 * x2917
            + x2797 * x2947
            + x2915 * x2950
            + x2919 * x2941
            + x2920 * x2949
            + x2921 * x2938
            + x2929 * x2945
            - x2936 * x415
            - x2939 * x415
            - x2940 * x415
        )
    )
    K_block_list.append(
        ddq_j2 * x2554
        - dq_i5 * x1266 * x2675
        + x1022
        * (
            dq_i5 * x2915 * x3079
            + x1444 * x3059
            + x2362 * x3074
            - x2685 * x3019
            + x2798 * x3077
            + x3020 * x3073
            + x3021 * x3078
            + x3030 * x3075
            - x3069 * x419
            - x3070 * x419
            - x3071 * x419
        )
        + x1142
        * (
            x1474 * x3059
            + x2362 * x3064
            - x2734 * x3019
            + x2834 * x3068
            + x2918 * x3066
            + x3020 * x3063
            + x3021 * x3067
            + x3030 * x3065
            - x3058 * x419
            - x3060 * x419
            - x3061 * x419
        )
        + x1266 * x2636
        + x1266 * x2641
        + x1266 * x2643
        + x1266 * x2645
        + x1266 * x2647
        + x1266
        * (
            2 * dq_i1 * dq_i5 * dq_j2 * x3098
            + dq_i2 * dq_i5 * dq_j2 * x3102
            + 2 * dq_i3 * dq_i5 * dq_j2 * x3099
            + 2 * dq_i4 * dq_i5 * dq_j2 * x3100
            + 2 * dq_i5 * dq_i6 * dq_j2 * x3097
            + 2
            * dq_i5
            * dq_i7
            * dq_j2
            * sigma_kin_v_7_2
            * sigma_kin_v_7_5
            * x1156
            * x139
            * x144
            * x146
            * x148
            * x150
            * x153
            * x203
            * x434
            - dq_i5 * x3091
            - dq_i5 * x3092
            - dq_i5 * x3093
            - dq_i5 * x3094
            - dq_i5 * x3095
            - dq_i5 * x3096
            + 2 * dq_j2 * x3101 * x610
            - x2454 * x3019
        )
        - x1316 * x3015
        - x1334 * x2912
        + x136 * x2375
        + x136
        * (
            -x2285 * x3019
            + x2376 * x3057
            + x2798 * x3049
            + x2918 * x3053
            + x3020 * x3042
            + x3021 * x3056
            + x3022 * x3038
            + x3030 * x3043
            - x3037 * x419
            - x3039 * x419
            - x3040 * x419
        )
        - x1414 * x2267 * x2638
        + x1603 * x868
        + x1923 * x2908
        + x1924 * x779
        + x1925 * x2656
        - x1931 * x2784
        - x1935 * (x1932 * x2288 + x1933 * x2274 + x1934 * x2276)
        + x1941
        * (
            x1364 * x3018
            + x2357 * x2798
            + x2361 * x2918
            + x2362 * x2367
            + x2363 * x3021
            + x2365 * x3020
            - x2370 * x419
            + x2796 * x419
            + x3018 * x545
            - x3019 * x550
        )
        - x2282 * x2651
        - x2289 * x3016
        + x2318 * x2648
        - x2329 * x2668
        - x2339 * x2662
        - x2348 * x2657
        - x2499 * x2655
        + x2905 * x3014
        + x823
        * (
            x1797 * x2544 * x3022
            + x2362 * x3025
            - x2608 * x3019
            + x2798 * x3031
            + x2918 * x3032
            + x3021 * x3034
            - x3023 * x419
            - x3024 * x419
            + x3029 * x3030
            + x3035 * x3036
        )
        + x906
        * (
            dq_j2 * x3090 * x610
            + x1414 * x3022 * x3081
            + x2362 * x3086
            - x2646 * x3019
            + x2798 * x3088
            + x2918 * x3089
            + x3020 * x3084
            + x3030 * x3087
            - x3080 * x419
            - x3082 * x419
            - x3083 * x419
        )
    )
    K_block_list.append(
        ddq_j2 * x2533
        - dq_i6 * x1266 * x2634
        + x1022
        * (
            x1444 * x3127
            + x2364 * x3140
            - x2683 * x3106
            + x2799 * x3142
            + x2935 * x3146
            + x3020 * x3144
            + x3107 * x3145
            + x3113 * x3141
            - x3138 * x423
            - x3139 * x423
        )
        + x1142
        * (
            x1474 * x3127
            + x2364 * x3130
            - x2732 * x3106
            + x2817 * x3137
            + x2919 * x3132
            + x3020 * x3134
            + x3107 * x3136
            + x3113 * x3131
            - x3128 * x423
            - x3129 * x423
        )
        + x1266 * x2599
        + x1266 * x2605
        + x1266 * x2607
        + x1266 * x2609
        + x1266 * x2611
        + x1266
        * (
            -dq_i6 * x3169
            - dq_i6 * x3170
            - dq_i6 * x3171
            - dq_i6 * x3172
            + dq_i6 * x3178 * x32
            + x1842 * x3126
            + x2364 * x3173
            - x2443 * x3106
            + x2799 * x3174
            + x2919 * x3175
            + x3020 * x3176
            + x3107 * x3177
        )
        + x1311 * x780
        + x1313 * x2804
        - x1327 * x3015
        - x1334 * x2911
        + x136 * x2372
        + x136
        * (
            -x2271 * x3106
            + x2374 * x3125
            + x2799 * x3117
            + x2919 * x3120
            + x3020 * x3123
            + x3107 * x3124
            + x3108 * x3109
            - x3110 * x423
            - x3111 * x423
            + x3112 * x3113
        )
        - x1418 * x1797 * x2267
        + x2049 * x3103
        + x2051 * x2908
        - x2053 * x2782
        - x2056 * (x2054 * x2274 + x2055 * x2276)
        + x2061
        * (
            x1364 * x3105
            + x2357 * x2799
            + x2361 * x2919
            + x2363 * x3020
            + x2364 * x2367
            + x2365 * x3107
            - x2370 * x423
            + x2796 * x423
            + x3105 * x545
            - x3106 * x550
        )
        - x2277 * x3104
        - x2303 * x2629
        + x2318 * x2612
        - x2329 * x2624
        - x2339 * x2621
        - x2348 * x2618
        - x2499 * x2615
        + x823
        * (
            dq_j2 * x3168 * x512
            + x1797 * x2522 * x3108
            + x2364 * x3161
            - x2610 * x3106
            + x2799 * x3165
            + x2919 * x3166
            + x3020 * x3167
            + x3113 * x3164
            - x3159 * x423
            - x3160 * x423
        )
        + x906
        * (
            x2364 * x3152
            - x2640 * x3106
            + x2799 * x3154
            + x2919 * x3155
            + x3036 * x3158
            + x3107 * x3157
            + x3113 * x3153
            + x3126 * x3148
            - x3149 * x423
            - x3150 * x423
        )
    )
    K_block_list.append(
        sigma_kin_v_7_2 * x1266 * x2159
        - x1418 * x2282 * x3188
        + x156 * x3180 * x3181
        - x188 * x3186
        + x2150 * x3103
        + x2152 * x2318 * x2585
        + x2152 * x2476 * x682
        + x2162 * x480
        + x2164 * x3180
        + x2165 * x3180
        + x2166 * x3180
        + x217 * x2523 * x3179
        + x2170 * x2265
        + x2174 * x480
        - x2176 * x2499 * x3185
        - x2186 * x2300 * x2303
        + x2209
        * (
            x1156 * x3194
            - x1156 * x3197
            - x1251 * x567
            + x188 * x2190
            + x188 * x2194
            - x2193 * x3195
            + x2197 * x3207
            + x2199 * x3207
            + x2200 * x3207
            + x2202 * x3207
            + x3198 * x3207
        )
        + x2218
        * (
            x1035 * x3194
            - x1035 * x3197
            + x1035 * x3200
            - x1035 * x3201
            + x2199 * x3199
            + x2200 * x3199
            + x2202 * x3199
            + x2211 * x2794
            + x2213 * x2794
            + x3198 * x3199
        )
        + x2228
        * (
            x2197 * x3202
            + x2200 * x3202
            + x2202 * x3202
            + x2220 * x2916
            + x2223 * x2916
            + x3194 * x919
            - x3197 * x919
            + x3198 * x3202
            + x3200 * x919
            - x3201 * x919
        )
        + x2236
        * (
            x2197 * x3203
            + x2199 * x3203
            + x2202 * x3203
            + x2230 * x3018
            + x2232 * x3018
            + x3194 * x828
            - x3197 * x828
            + x3198 * x3203
            + x3200 * x828
            - x3201 * x828
        )
        + x2243
        * (
            x2197 * x3204
            + x2199 * x3204
            + x2200 * x3204
            + x2238 * x3105
            + x2239 * x3105
            + x3194 * x744
            - x3197 * x744
            + x3198 * x3204
            + x3200 * x744
            - x3201 * x744
        )
        + x2250
        * (
            dq_j2 * x3205 * x549
            + x2197 * x3206
            + x2199 * x3206
            + x2200 * x3206
            + x2202 * x3206
            + x2247 * x3017
            - x3017 * x569
            + x3198 * x3206
            + x3200 * x520
            - x3201 * x520
        )
        + x2257
        * (
            x137 * x3194
            + x2197 * x3193
            + x2199 * x3193
            + x2200 * x3193
            + x2202 * x3193
            + x2251 * x3190
            - x2252 * x3017
            + x2253 * x3190
            + x3191 * x3192
            - x3191 * x3196
        )
        - x2329 * x3187 * x497
        - x2339 * x3187 * x489
        + x244 * x2545 * x3179
        + x250 * x2556 * x3179
        - x2507 * x2508 * x3189 * x431
        + x2570 * x275 * x3179
        - x3182 * x3183
        - x3182 * x3184
    )
    K_block_list.append(
        -dq_i1 * x1142 * x3279
        + x1022
        * (
            x1444 * x3306
            + x1692 * x3293
            + x1695 * x3291
            + x1703 * x3295
            + x1706 * x3282
            + x1708 * x3300
            + x1709 * x3280
            - x3307
            - x515 * x992
            - x515 * x993
            - x515 * x994
            - x515 * x995
        )
        + x111 * x2258 * x3209
        + x1142 * x3212
        + x1142 * x3221
        + x1142 * x3234
        + x1142 * x3239
        + x1142 * x3250
        - x1142
        * (
            dq_i1 * x1129
            + dq_i1 * x1131
            + dq_i1 * x1132
            + dq_i1 * x1135
            + dq_i1 * x1138
            + dq_i1 * x209 * x3324
            + dq_i1 * x225 * x3320
            + dq_i1 * x256 * x3318
            + dq_i1 * x3316 * x991
            - dq_i1 * x54 * (x1743 + x3312 * x3340 + x3315 * x3332 + x3318 * x650 + x3320 * x647 + x3324 * x644)
            + x2392 * x3313
            - x3248 * (x1667 * x3334 + x1699 * x3325 + x1701 * x3322 + x3332 * x342 + x3340 * x335)
            - x3282 * (x1731 * x3347 + x2273 * x3346 + x2287 * x3345 + x3340 * x3341 + x3342 * x3344)
            - x3287 * (x1693 * x3335 + x1701 * x3339 + x3332 * x3333 + x3336 * x3337)
            - x3291 * (x1553 * x3326 + x1658 * x3331 + x3329 * x3330)
            - x3293 * (x1542 * x3326 + x1755 * x3328)
            - x3295 * (x1319 * x3346 + x1330 * x3345 + x1730 * x3350 + x273 * x3348 * x40 + x3342 * x3349)
            - x3306 * x3323
        )
        + x1266
        * (
            -x1220 * x515
            - x1221 * x515
            - x1222 * x515
            - x1223 * x515
            - x1224 * x515
            + x1315 * x3306
            + x1720 * x3293
            + x1723 * x3291
            + x1727 * x3287
            + x1728 * x3295
            + x1732 * x3300
            + x1734 * x3310
            - x3309
        )
        + x1276 * x3210
        + x1283 * x1297
        + x1285 * x1603
        + x1286 * x1608
        + x1287 * x1609
        - x1317 * x3226
        - x1328 * x3237
        - x1335 * x3240
        - x1359 * x3252
        + x136 * x3249
        + x136
        * (
            dq_j3 * x11 * (sigma_kin_v_4_3 * x1917 + x134 * x659 + x1765 + x183 * x660 + x201 * x661 + x273 * x60)
            + x2137 * x3298 * x645
            - x3249
            + x3282 * (x1289 * x273 + x1292 * x659 + x1293 * x660 + x1294 * x661 + x3308 * x375)
            + x3287 * (x1278 * x1725 + x1280 * x1726 + x1673 * x296 + x2383 * x3308)
            + x3291 * (x1270 * x1721 + x1272 * x1640 + x1274 * x1722)
            + x3293 * (x1268 * x1640 + x1269 * x644)
            + x3300 * (x1276 * x1729 + x1283 * x273 + x1287 * x1701 + x1676 * x295 + x1707 * x224)
            - x340 * x515
            - x345 * x515
            - x349 * x515
            - x353 * x515
            - x357 * x515
        )
        - x2137 * x3218
        - x2319 * (x1572 * x736 + x1576 * x728 + x3222 * x731 + x3224 * x733 + x3235 * x726)
        + x3214 * x3216
        - x3230 * x3231
        - x3246 * x3247
        + x3258 * x3259
        - x3267 * x3268
        - x3269 * x3270
        - x3277 * x3278
        + x3304 * x3305
        + x574
        * (
            x1364 * x3281
            + x2352 * x3286
            - x3248 * x553
            + x3281 * x545
            + x3282 * x3284
            + x3287 * x3290
            + x3291 * x3292
            + x3293 * x3294
            + x3295 * x3296
            - x3297 * x515
        )
        + x823
        * (
            x1643 * x3295
            + x1647 * x3282
            + x1649 * x3287
            + x1650 * x3291
            + x1651 * x3300
            + x1652 * x3301
            + x1797 * x3298 * x644
            - x3299
            - x515 * x793
            - x515 * x794
        )
        + x906
        * (
            x1660 * x3298
            + x1666 * x3293
            + x1671 * x3295
            + x1672 * x3282
            + x1675 * x3287
            + x1677 * x3300
            + x1678 * x3303
            - x3302
            - x515 * x876
            - x515 * x878
            - x515 * x879
        )
    )
    K_block_list.append(
        -dq_i2 * x1142 * x3401
        + x1022
        * (
            x1444 * x3414
            - x2688 * x3404
            - x2712 * x405
            - x2713 * x405
            - x2714 * x405
            - x2715 * x405
            + x2843 * x3407
            + x2845 * x3406
            + x2849 * x3282
            + x2852 * x3408
            + x2855 * x3409
            + x2856 * x3402
        )
        + x1142 * x3352
        + x1142 * x3359
        + x1142 * x3368
        + x1142 * x3374
        + x1142 * x3380
        - x1142
        * (
            dq_i2 * x1388 * x3320
            + dq_i2 * x1406 * x3318
            + dq_i2 * x1418 * x3324
            + dq_i2 * x2490 * x3313
            + dq_i2 * x2711 * x3316
            + dq_i2 * x2762
            + dq_i2 * x2763
            + dq_i2 * x2764
            + dq_i2 * x2765
            + dq_i2 * x2766
            - dq_i2 * x54 * (x2305 * x3318 + x2306 * x3320 + x2569 * x3324 + x2882 + x3312 * x3418 + x3315 * x3416)
            - x3282 * (x1319 * x3420 + x1330 * x3419 + x146 * x2848 * x3350 + x2304 * x3349 + x2902 * x3348)
            - x3323 * x3414
            - x3404 * (x2825 * x3334 + x2847 * x3325 + x2848 * x3322 + x335 * x3418 + x3416 * x342)
            - x3405 * (x2848 * x3339 + x2896 * x3335 + x3333 * x3416 + x3337 * x3417)
            - x3406 * (x2893 * x3325 + x2894 * x3331 + x3330 * x3415)
            - x3407 * (x2890 * x3325 + x2891 * x3328)
            - x3408 * (x2273 * x3420 + x2287 * x3419 + x2304 * x3344 + x2881 * x3347 + x3341 * x3418)
        )
        + x1198 * x3210
        + x1213 * x2776
        + x1215 * x1603
        + x1216 * x1608
        + x1219 * x1609
        + x1266
        * (
            dq_j3 * x711 * (x1239 * x2469 + x1245 * x2471 + x1248 * x2472 + x1250 * x2473 + x1496 * x3400 + x2904)
            + x1842 * x2570 * x3411
            - x2474 * x3404
            - x2491 * x405
            - x2492 * x405
            - x2493 * x405
            - x2494 * x405
            - x2496 * x405
            + x3282 * (sigma_kin_v_4_3 * x3008 + x1154 * x2471 + x1161 * x2472 + x1227 * x1345 + x1234 * x2473)
            + x3405 * (sigma_kin_v_4_3 * x1506 * x289 + x1201 * x2829 + x1202 * x2868 + x1204 * x2869)
            + x3406 * (x1183 * x2866 + x1189 * x2865 + x1191 * x2863)
            + x3407 * (x1181 * x2863 + x1184 * x2808)
            + x3409 * (x1180 * x2854 + x1200 * x2831 + x1213 * x2469 + x1218 * x2881 + x1506 * x3272)
        )
        + x1300 * x2435 * x3209
        - x1315 * x3355
        - x1317 * x3360
        - x1328 * x3369
        - x1335 * x3375
        + x136 * x3309
        + x136
        * (
            -x2307 * x3404
            - x2331 * x405
            - x2332 * x405
            - x2334 * x405
            - x2336 * x405
            - x2337 * x405
            + x2860 * x3411
            + x2864 * x3407
            + x2867 * x3406
            + x2870 * x3405
            + x2871 * x3408
            + x2872 * x3409
            + x2873 * x3310
        )
        - x1360 * (x1572 * x2597 + x1576 * x2591 + x2589 * x3235 + x2593 * x3222 + x2595 * x3224)
        + x1385
        * (
            x1364 * x3403
            + x1366 * x3286
            + x3282 * x3296
            + x3284 * x3408
            + x3290 * x3405
            + x3292 * x3406
            + x3294 * x3407
            - x3297 * x405
            + x3403 * x545
            - x3404 * x553
        )
        - x3231 * x3364
        - x3247 * x3379
        - x3252 * x743
        + x3259 * x3385
        - x3268 * x3391
        - x3278 * x3399
        + x3305 * x3413
        + x3353 * x3354
        - x3392 * x3393
        + x823
        * (
            dq_j3 * x2801 * x407
            - x2604 * x3404
            - x2619 * x405
            - x2620 * x405
            + x2806 * x3282
            + x2810 * x3408
            + x2812 * x3405
            + x2814 * x3406
            + x2815 * x3409
            + x2816 * x3410
        )
        + x906
        * (
            -x2642 * x3404
            - x2659 * x405
            - x2660 * x405
            - x2661 * x405
            + x2821 * x3411
            + x2824 * x3407
            + x2827 * x3282
            + x2828 * x3408
            + x2830 * x3405
            + x2832 * x3409
            + x2833 * x3412
        )
    )
    K_block_list.append(
        x1022
        * (
            2 * dq_i1 * (x3591 * x965 + x3627 * x907 + x3628 * x967 + x3629 * x969 + x3630)
            + 2 * dq_i2 * (x3591 * x973 + x3627 * x971 + x3628 * x974 + x3629 * x975 + x3631)
            + 2 * dq_i3 * dq_i7 * dq_j3 * x140 * x144 * x145 * x148 * x150 * x153 * x1626 * x203 * x434 * x919
            + 2 * dq_i3 * (x3591 * x985 + x3628 * x988 + x3629 * x990 + x3632 * x984 + x3633)
            + dq_i4
            * (
                x1001 * x3591
                + x1002 * x3591
                + x1004 * x3628
                + x1005 * x3535 * x54
                + x1007 * x3629
                + x1008 * x3536 * x54
                + x3534 * x54 * x999
                + x3632 * x998
                + x3642
            )
            + 2 * dq_i5 * (x263 * x3617 + x3587 * x957 + x3591 * x954 + x3625)
            + 2 * dq_i6 * (x223 * x3617 + x3553 * x946 + x3621)
            + 2 * dq_i7 * sigma_kin_v_7_3 * x140 * x144 * x145 * x148 * x150 * x153 * x195 * x203 * x434 * x919
            - x3500 * x3616
            - x3603 * x411
            - x3605 * x411
            - x3606
            - x3607 * x411
            - x3608 * x411
            - x3610
            - x3612
            - x3613
            - x3615
            - x537 * x939
        )
        - x1036 * x3428
        + x1057 * x3448
        + x110 * x3422 * (sigma_pot_3_c * x48 + sigma_pot_3_s * x45)
        + x1114 * x2432
        + x1116 * x3421
        + x1117 * x3421
        + x1128 * x2432
        + x1134 * x117
        + x1137 * x164
        + x1140 * x197
        + x1142 * x3425
        + x1142 * x3432
        + x1142 * x3440
        + x1142 * x3443
        + x1142 * x3449
        + x1142
        * (
            2 * dq_i1 * (x1090 * x3644 + x1092 * x3585 + x1093 * x3628 + x1095 * x3629 + x3462 + x3651 * x907)
            + 2 * dq_i2 * (x1099 * x3649 + x1100 * x3585 + x1101 * x3628 + x1104 * x3629 + x3463 + x3651 * x971)
            + 2 * dq_i3 * dq_i7 * dq_j3 * x1035 * x140 * x143 * x146 * x148 * x150 * x153 * x1626 * x203 * x434
            - dq_i3 * x3469
            - dq_i3 * x3470
            - dq_i3 * x3471
            - dq_i3 * x3472
            - dq_i3 * x3473
            - dq_i3 * x3474
            - dq_i3 * x3475
            - dq_i3 * x3476
            - dq_i3 * x3477
            - dq_i3 * x3478
            + dq_i3
            * (
                x1114 * x3649
                + x1115 * x3450 * x54
                + x1116 * x3632
                + x1117 * x3632
                + x1119 * x3585
                + x1120 * x3585
                + x1122 * x3628
                + x1123 * x3628
                + x1125 * x3629
                + x1127 * x3629
                + x3479
            )
            + 2 * dq_i4 * (x1076 * x3632 + x1079 * x3628 + x1082 * x3629 + x2751 * x3591 + x3461)
            + 2 * dq_i5 * (x1069 * x3585 + x1070 * x3587 + x263 * x3650 + x3447)
            + 2 * dq_i6 * (x1062 * x3553 + x223 * x3650 + x3436)
            + 2 * dq_i7 * sigma_kin_v_7_3 * x1035 * x140 * x143 * x146 * x148 * x150 * x153 * x195 * x203 * x434
            - x1057 * x537
            - x3425
            - x3432
            - x3440
            - x3443
            - x3449
            - x3455 * x3500
        )
        + x117 * x3423
        + x1266
        * (
            2 * dq_i1 * (x1154 * x3646 + x1161 * x3647 + x1227 * x3644 + x1230 * x3645 + x1234 * x3648 + x3392)
            + dq_i2
            * (
                x1239 * x3649
                + x1242 * x3645
                + x1243 * x3381 * x54
                + x1245 * x3646
                + x1246 * x3382 * x54
                + x1248 * x3647
                + x1249 * x3383 * x54
                + x1250 * x3648
                + x1251 * x3384 * x54
                + x1254 * x3649
                + x3401
            )
            + 2 * dq_i3 * dq_i7 * dq_j3 * x1156 * x139 * x144 * x146 * x148 * x150 * x153 * x1626 * x203 * x434
            + 2 * dq_i3 * (x1213 * x3649 + x1214 * x3626 + x1215 * x3585 + x1216 * x3628 + x1219 * x3629 + x3399)
            + 2 * dq_i4 * (x1199 * x3626 + x1203 * x3628 + x1205 * x3629 + x2775 * x3591 + x3391)
            + 2 * dq_i5 * (x1190 * x3585 + x1191 * x3643 + x1192 * x3587 + x3379)
            + 2 * dq_i6 * (x1181 * x3643 + x1184 * x3553 + x3364)
            + 2 * dq_i7 * sigma_kin_v_7_3 * x1156 * x139 * x144 * x146 * x148 * x150 * x153 * x195 * x203 * x434
            - x1174 * x537
            - x3352
            - x3359
            - x3368
            - x3374
            - x3380
            - x3385 * x3500
            - x3394 * x411
            - x3395 * x411
            - x3396 * x411
            - x3397 * x411
            - x3398 * x411
        )
        + x1341
        * (
            ddq_i3 * (x1622 * x655 + x1623 * x623 + x1624 * x582 + x1625 * x466 + x1626 * x468)
            + x118 * x121 * x123 * x127 * x1624 * x346 * x580 * x668
            + x140 * x143 * x146 * x148 * x150 * x154 * x1626 * x354 * x434 * x668
            + x1622 * x334 * x42 * x49 * x654 * x668
            + x1623 * x341 * x621 * x66 * x668 * x75 * x78
            + x1625 * x165 * x168 * x170 * x172 * x176 * x350 * x464 * x668
            - x3529 * x646
            - x492 * (x221 * x3531 + x3532 * x509)
            - x601 * (x255 * x3533 + x263 * x3530 + x3529 * x504)
            - x639 * (x1624 * x603 + x287 * x3534 + x299 * x3535 + x306 * x3536)
            - x663 * (x133 * x3255 + x182 * x3256 + x199 * x3257 + x3253 * x59 + x3254 * x85)
            - x666 * (x2788 * x370 + x3381 * x372 + x3382 * x377 + x3383 * x383 + x3384 * x389)
            - x669 * (x334 * x3450 + x343 * x3451 + x3452 * x597 + x3453 * x485 + x3454 * x489)
        )
        + x136
        * (
            dq_i1
            * (
                x125 * x3255 * x54
                + x1295 * x3645
                + x134 * x3646
                + x174 * x3256 * x54
                + x183 * x3647
                + x185 * x3257 * x54
                + x201 * x3648
                + x3253 * x39 * x54
                + x3254 * x54 * x65
                + x3279
                + x3644 * x60
            )
            + 2 * dq_i2 * (x1289 * x3644 + x1291 * x3645 + x1292 * x3646 + x1293 * x3647 + x1294 * x3648 + x3269)
            + 2 * dq_i3 * dq_i7 * dq_j3 * x137 * x138 * x140 * x144 * x146 * x148 * x150 * x153 * x1626 * x203
            + 2 * dq_i3 * (x1283 * x3644 + x1284 * x3626 + x1285 * x3585 + x1286 * x3628 + x1287 * x3629 + x3277)
            + 2 * dq_i4 * (x1277 * x3626 + x1279 * x3628 + x1281 * x3629 + x2769 * x3591 + x3267)
            + 2 * dq_i5 * (x1271 * x3585 + x1273 * x3550 + x1275 * x3587 + x3246)
            + 2 * dq_i6 * (x1268 * x3643 + x1269 * x3551 + x3230)
            + 2 * dq_i7 * sigma_kin_v_7_3 * x137 * x138 * x140 * x144 * x146 * x148 * x150 * x153 * x195 * x203
            - x274 * x537
            - x3212
            - x3221
            - x3234
            - x3239
            - x3250
            - x3258 * x3500
            - x3271 * x411
            - x3273 * x411
            - x3274 * x411
            - x3275 * x411
            - x3276 * x411
        )
        + x1636
        * (
            x1364 * x189
            + x189 * x545
            + x2543
            - x3494
            + x3496 * x522
            + x3496 * x536
            + x3497 * x527
            + x3498 * x529
            + x3499 * x417
            + x3499 * x421
            - x3500 * x553
            + x551
        )
        + x164 * x3426
        + x197 * x3427
        + x2188 * x3304
        + x3189 * x3413
        - x3231 * x3436
        - x3247 * x3447
        + x3259 * x3455
        - x3268 * x3461
        - x3270 * x3463
        - x3393 * x3462
        - x3467 * x3468 * x433
        - 4 * x3479 * x54
        + x3492 * x3493
        + x3513 * x3514
        + x3527 * x3528
        + x823
        * (
            2 * dq_i1 * (x3550 * x773 + x3551 * x776 + x3552)
            + 2 * dq_i2 * (x3550 * x780 + x3554 * x782 + x3555)
            + 2 * dq_i3 * dq_i7 * dq_j3 * x140 * x144 * x146 * x148 * x149 * x153 * x1626 * x203 * x434 * x744
            + 2 * dq_i3 * (x3550 * x789 + x3554 * x790 + x3556)
            + 2 * dq_i4 * (x3550 * x796 + x3554 * x797 + x3559)
            + 2 * dq_i5 * (x3553 * x806 + x3560 * x803 + x3564)
            + dq_i6 * (x3531 * x54 * x811 + x3551 * x812 + x3553 * x814 + x3560 * x810 + x3569)
            + 2 * dq_i7 * sigma_kin_v_7_3 * x140 * x144 * x146 * x148 * x149 * x153 * x195 * x203 * x434 * x744
            - x3500 * x3549
            - x3537
            - x3539 * x411
            - x3541 * x411
            - x3542
            - x3544
            - x3546
            - x3548
            - x537 * x760
        )
        + x906
        * (
            2 * dq_i1 * (x1422 * x3587 + x3585 * x863 + x3586 * x772 + x3588)
            + 2 * dq_i2 * (x2656 * x3587 + x3585 * x868 + x3586 * x779 + x3589)
            + 2 * dq_i3 * dq_i7 * dq_j3 * x140 * x144 * x146 * x147 * x150 * x153 * x1626 * x203 * x434 * x828
            + 2 * dq_i3 * (x2658 * x3587 + x3585 * x875 + x3586 * x485 + x3590)
            + 2 * dq_i4 * (x1433 * x3587 + x2664 * x3591 + x3586 * x494 + x3595)
            + dq_i5
            * (x1436 * x3587 + x3533 * x54 * x891 + x3560 * x892 + x3560 * x893 + x3585 * x890 + x3587 * x896 + x3602)
            + 2 * dq_i6 * (x3553 * x858 + x3560 * x857 + x3584)
            + 2 * dq_i7 * sigma_kin_v_7_3 * x140 * x144 * x146 * x147 * x150 * x153 * x195 * x203 * x434 * x828
            - x3500 * x3581
            - x3570 * x411
            - x3571
            - x3572 * x411
            - x3573 * x411
            - x3575
            - x3576
            - x3578
            - x3580
            - x537 * x848
        )
    )
    K_block_list.append(
        -dq_i4 * x1142 * x3642
        + x1022
        * (
            dq_j3 * x3739 * x641
            + x1444 * x208 * x3685
            + x3287 * x3736
            + x3405 * x3737
            - x3614 * x3663
            + x3665 * x3734
            + x3666 * x3732
            + x3677 * x3738
            - x3727 * x415
            - x3728 * x415
            - x3729 * x415
            - x3730 * x415
        )
        + x1142 * x3606
        + x1142 * x3610
        + x1142 * x3612
        + x1142 * x3613
        + x1142 * x3615
        + x1142
        * (
            2 * dq_i1 * dq_i4 * dq_j3 * x3752
            + 2 * dq_i2 * dq_i4 * dq_j3 * x3753
            + dq_i3 * dq_i4 * dq_j3 * x3755
            + 2 * dq_i4 * dq_i5 * dq_j3 * x3750
            + 2 * dq_i4 * dq_i6 * dq_j3 * x3749
            + 2
            * dq_i4
            * dq_i7
            * dq_j3
            * sigma_kin_v_7_3
            * sigma_kin_v_7_4
            * x1035
            * x140
            * x143
            * x146
            * x148
            * x150
            * x153
            * x203
            * x434
            - dq_i4 * x3740
            - dq_i4 * x3741
            - dq_i4 * x3742
            - dq_i4 * x3743
            - dq_i4 * x3744
            - dq_i4 * x3745
            - dq_i4 * x3746
            - dq_i4 * x3747
            + 2 * dq_j3 * x3754 * x641
            - x3442 * x3663
        )
        + x1266
        * (
            dq_i4 * x3402 * x3725
            + x1842 * x3685
            + x3287 * x3722
            - x3373 * x3663
            + x3665 * x3721
            + x3666 * x3720
            + x3667 * x3724
            + x3677 * x3723
            - x3716 * x415
            - x3717 * x415
            - x3718 * x415
            - x3719 * x415
        )
        + x1304 * x985
        - x1316 * x3658
        - x1327 * x3659
        + x136 * x3307
        + x136
        * (
            dq_i4 * x3280 * x3715
            - x3238 * x3663
            + x3405 * x3707
            + x3665 * x3706
            + x3666 * x3703
            + x3667 * x3714
            + x3668 * x3699
            + x3677 * x3711
            - x3697 * x415
            - x3698 * x415
            - x3700 * x415
            - x3701 * x415
        )
        + x1608 * x988
        + x1609 * x990
        - x1781 * x3375
        - x1787 * (x1576 * x3660 + x1784 * x3236 + x1785 * x3223 + x1786 * x3225)
        + x1794
        * (
            x1364 * x3662
            + x3284 * x3405
            + x3287 * x3296
            + x3290 * x3667
            + x3292 * x3665
            + x3294 * x3666
            - x3297 * x415
            + x3662 * x545
            - x3663 * x553
            + x3664 * x415
        )
        + x2907 * x3652
        - x2913 * x3240
        - x3231 * x3621
        - x3247 * x3625
        + x3259 * x3616
        - x3270 * x3631
        - x3278 * x3633
        + x3305 * x3527
        - x3393 * x3630
        - x3428 * x920
        + x3653 * x984
        + x3654 * x3655
        + x823
        * (
            x1797 * x3441 * x3668
            + x3287 * x3672
            + x3405 * x3673
            - x3543 * x3663
            + x3665 * x3680
            + x3667 * x3681
            - x3669 * x415
            - x3671 * x415
            + x3676 * x3677
            + x3682 * x3683
        )
        + x906
        * (
            x3148 * x3685
            + x3287 * x3691
            + x3405 * x3692
            - x3577 * x3663
            + x3661 * x3696
            + x3666 * x3688
            + x3667 * x3695
            + x3677 * x3693
            - x3684 * x415
            - x3686 * x415
            - x3687 * x415
        )
    )
    K_block_list.append(
        -dq_i5 * x1142 * x3602
        + x1022
        * (
            dq_i5 * x3661 * x3816
            + x1444 * x208 * x3798
            + x3291 * x3812
            + x3406 * x3813
            - x3611 * x3760
            + x3761 * x3811
            + x3762 * x3815
            + x3771 * x3814
            - x3807 * x419
            - x3808 * x419
            - x3809 * x419
        )
        + x1142 * x3571
        + x1142 * x3575
        + x1142 * x3576
        + x1142 * x3578
        + x1142 * x3580
        + x1142
        * (
            2 * dq_i1 * dq_i5 * dq_j3 * x3835
            + 2 * dq_i2 * dq_i5 * dq_j3 * x3836
            + dq_i3 * dq_i5 * dq_j3 * x3839
            + 2 * dq_i4 * dq_i5 * dq_j3 * x3837
            + 2 * dq_i5 * dq_i6 * dq_j3 * x3834
            + 2
            * dq_i5
            * dq_i7
            * dq_j3
            * sigma_kin_v_7_3
            * sigma_kin_v_7_5
            * x1035
            * x140
            * x143
            * x146
            * x148
            * x150
            * x153
            * x203
            * x434
            - dq_i5 * x3827
            - dq_i5 * x3828
            - dq_i5 * x3829
            - dq_i5 * x3830
            - dq_i5 * x3831
            - dq_i5 * x3832
            + 2 * dq_j3 * x3838 * x610
            - x3439 * x3760
        )
        + x1266
        * (
            x1842 * x3798
            + x3291 * x3802
            - x3367 * x3760
            + x3412 * x3806
            + x3665 * x3804
            + x3761 * x3801
            + x3762 * x3805
            + x3771 * x3803
            - x3797 * x419
            - x3799 * x419
            - x3800 * x419
        )
        - x1316 * x3757
        - x1334 * x3659
        + x136 * x3302
        + x136
        * (
            -x3233 * x3760
            + x3303 * x3796
            + x3406 * x3784
            + x3665 * x3792
            + x3761 * x3783
            + x3762 * x3795
            + x3763 * x3779
            + x3771 * x3788
            - x3778 * x419
            - x3780 * x419
            - x3781 * x419
        )
        + x1603 * x875
        + x1924 * x485
        + x1925 * x2658
        - x1930 * x3369
        - x1935 * (x1932 * x3236 + x1933 * x3223 + x1934 * x3225)
        + x1941
        * (
            x1364 * x3759
            + x3284 * x3406
            + x3290 * x3665
            + x3291 * x3296
            + x3292 * x3762
            + x3294 * x3761
            - x3297 * x419
            + x3664 * x419
            + x3759 * x545
            - x3760 * x553
        )
        + x3014 * x3652
        - x3016 * x3237
        - x3148 * x3218
        - x3231 * x3584
        + x3259 * x3581
        - x3268 * x3595
        - x3270 * x3589
        - x3278 * x3590
        + x3305 * x3513
        - x3393 * x3588
        + x3654 * x3756
        + x823
        * (
            x1797 * x3438 * x3763
            + x3291 * x3766
            + x3406 * x3767
            - x3545 * x3760
            + x3665 * x3772
            + x3762 * x3775
            - x3764 * x419
            - x3765 * x419
            + x3770 * x3771
            + x3776 * x3777
        )
        + x906
        * (
            dq_j3 * x3826 * x610
            + x3148 * x3798
            + x3291 * x3822
            + x3406 * x3823
            - x3579 * x3760
            + x3665 * x3825
            + x3761 * x3820
            + x3771 * x3824
            - x3817 * x419
            - x3818 * x419
            - x3819 * x419
        )
    )
    K_block_list.append(
        -dq_i6 * x1142 * x3569
        + x1022
        * (
            x1444 * x3873
            + x3293 * x3876
            + x3407 * x3877
            - x3609 * x3844
            + x3683 * x3882
            + x3761 * x3880
            + x3845 * x3881
            + x3854 * x3878
            - x3874 * x423
            - x3875 * x423
        )
        + x1142 * x3537
        + x1142 * x3542
        + x1142 * x3544
        + x1142 * x3546
        + x1142 * x3548
        + x1142
        * (
            -dq_i6 * x3903
            - dq_i6 * x3904
            - dq_i6 * x3905
            - dq_i6 * x3906
            + dq_i6 * x3913 * x54
            + x1474 * x3873
            + x3293 * x3907
            + x3407 * x3908
            - x3431 * x3844
            + x3666 * x3909
            + x3761 * x3911
            + x3845 * x3912
        )
        + x1266
        * (
            x1842 * x3863
            + x3293 * x3866
            - x3358 * x3844
            + x3410 * x3872
            + x3666 * x3868
            + x3761 * x3869
            + x3845 * x3871
            + x3854 * x3867
            - x3864 * x423
            - x3865 * x423
        )
        + x1311 * x789
        + x1313 * x3670
        - x1327 * x3757
        - x1334 * x3658
        + x136 * x3299
        + x136
        * (
            -x3220 * x3844
            + x3301 * x3862
            + x3407 * x3850
            + x3666 * x3857
            + x3761 * x3860
            + x3845 * x3861
            + x3846 * x3847
            - x3848 * x423
            - x3849 * x423
            + x3853 * x3854
        )
        - x1663 * x1797 * x3217
        + x2049 * x3840
        - x2052 * x3360
        - x2056 * (x2054 * x3223 + x2055 * x3225)
        + x2061
        * (
            x1364 * x3843
            + x3284 * x3407
            + x3290 * x3666
            + x3292 * x3761
            + x3293 * x3296
            + x3294 * x3845
            - x3297 * x423
            + x3664 * x423
            + x3843 * x545
            - x3844 * x553
        )
        - x3104 * x3226
        + x3214 * x3842
        - x3247 * x3564
        + x3259 * x3549
        - x3268 * x3559
        - x3270 * x3555
        - x3278 * x3556
        + x3305 * x3492
        - x3393 * x3552
        + x823
        * (
            dq_j3 * x3902 * x512
            + x1797 * x3430 * x3846
            + x3293 * x3895
            + x3407 * x3896
            - x3547 * x3844
            + x3666 * x3900
            + x3761 * x3901
            + x3854 * x3899
            - x3893 * x423
            - x3894 * x423
        )
        + x906
        * (
            x3148 * x3863
            + x3293 * x3886
            + x3407 * x3887
            - x3574 * x3844
            + x3666 * x3889
            + x3777 * x3892
            + x3845 * x3891
            + x3854 * x3888
            - x3883 * x423
            - x3884 * x423
        )
    )
    K_block_list.append(
        sigma_kin_v_7_3 * x1142 * x2159
        - x1663 * x3188 * x3231
        - x189 * x3927
        + x2150 * x3840
        + x2152 * x3259 * x3529
        + x2152 * x3448 * x645
        + x217 * x3480 * x3914
        - x2176 * x3393 * x3925
        - x2182 * x3270 * x3925
        - x2186 * x3244 * x3247
        + x2209
        * (
            x1156 * x3931
            + x1156 * x3932
            - x1156 * x3934
            + x2190 * x3403
            + x2194 * x3403
            + x2199 * x3935
            + x2200 * x3935
            + x2202 * x3935
            - x2205 * x3195 * x410
            + x3198 * x3935
        )
        + x2218
        * (
            x1035 * x3932
            - x1035 * x3934
            - x1126 * x567
            + x189 * x2211
            + x189 * x2213
            + x2199 * x3940
            + x2200 * x3940
            + x2202 * x3940
            - x2212 * x3195
            + x2214 * x3940
            + x3198 * x3940
        )
        + x2228
        * (
            x2200 * x3936
            + x2202 * x3936
            + x2214 * x3936
            + x2220 * x3662
            + x2223 * x3662
            + x3198 * x3936
            + x3931 * x919
            + x3932 * x919
            - x3933 * x919
            - x3934 * x919
        )
        + x2236
        * (
            x2199 * x3937
            + x2202 * x3937
            + x2214 * x3937
            + x2230 * x3759
            + x2232 * x3759
            + x3198 * x3937
            + x3931 * x828
            + x3932 * x828
            - x3933 * x828
            - x3934 * x828
        )
        + x2243
        * (
            x2199 * x3938
            + x2200 * x3938
            + x2214 * x3938
            + x2238 * x3843
            + x2239 * x3843
            + x3198 * x3938
            + x3931 * x744
            + x3932 * x744
            - x3933 * x744
            - x3934 * x744
        )
        + x2250
        * (
            dq_j3 * x3205 * x552
            + x2199 * x3939
            + x2200 * x3939
            + x2202 * x3939
            + x2214 * x3939
            + x2247 * x3758
            + x3198 * x3939
            - x3758 * x569
            + x3931 * x520
            - x3933 * x520
        )
        + x2257
        * (
            x137 * x3931
            + x137 * x3932
            - x137 * x3933
            + x2199 * x3930
            + x2200 * x3930
            + x2202 * x3930
            + x2214 * x3930
            + x2251 * x3929
            - x2252 * x3758
            + x2253 * x3929
        )
        + x244 * x3501 * x3914
        + x250 * x3515 * x3914
        + x2570 * x281 * x3914
        - x3183 * x3923
        - x3184 * x3923
        + x3213 * x3922
        - x3268 * x3925 * x3928
        - x3305 * x3467 * x3468 * x431
        + x3915 * x3916
        + x3915 * x3921
        + x3917 * x490
        + x3918 * x3919
        + x3918 * x3920
        + x3924 * x489
    )
    K_block_list.append(
        -dq_i1 * x1022 * x3992
        + x1022 * x3944
        + x1022 * x3951
        + x1022 * x3961
        + x1022 * x3966
        + x1022 * x3967
        + x1022
        * (
            2 * dq_i1 * dq_i2 * dq_j4 * x1895
            + 2 * dq_i1 * dq_i3 * dq_j4 * x1896
            + dq_i1 * dq_i4 * dq_j4 * x1897
            + 2 * dq_i1 * dq_i5 * dq_j4 * x1890
            + 2 * dq_i1 * dq_i6 * dq_j4 * x1888
            + 2
            * dq_i1
            * dq_i7
            * dq_j4
            * sigma_kin_v_7_1
            * sigma_kin_v_7_4
            * x140
            * x144
            * x145
            * x148
            * x150
            * x153
            * x203
            * x434
            * x919
            - dq_i1 * x1010
            - dq_i1 * x1012
            - dq_i1 * x1013
            - dq_i1 * x1014
            - dq_i1 * x1016
            - dq_i1 * x1017
            - dq_i1 * x1019
            - dq_i1 * x1020
            + 2 * dq_j4 * x11 * x1894
            - x4024
        )
        + x1142
        * (
            -x1083 * x515
            - x1084 * x515
            - x1085 * x515
            - x1086 * x515
            + x1474 * x208 * x4019
            + x1874 * x4006
            + x1875 * x4002
            + x1877 * x4008
            + x1879 * x3995
            + x1880 * x4015
            + x1881 * x4023
            - x4022
        )
        + x1266
        * (
            -x1207 * x515
            - x1208 * x515
            - x1209 * x515
            - x1210 * x515
            + x1842 * x4019
            + x1848 * x4006
            + x1851 * x4002
            + x1852 * x4008
            + x1857 * x3997
            + x1860 * x4015
            + x1861 * x4021
            - x4020
        )
        + x1277 * x1302
        + x1279 * x1608
        + x1281 * x1609
        + x1304 * x2769
        - x1317 * x3954
        - x1328 * x3963
        - x1342 * x3969
        - x1359 * x3968
        + x136 * x3965
        + x136
        * (
            dq_j4 * x11 * x1918
            + x1910 * x4006
            + x1911 * x4002
            + x1913 * x3995
            + x1915 * x3997
            + x1916 * x4015
            + x2137 * x4013 * x614
            - x291 * x515
            - x297 * x515
            - x304 * x515
            - x311 * x515
            - x3965
        )
        - x2137 * x3948
        + x2258 * x3941 * x3942
        - x2319 * (x1556 * x729 + x1564 * x732 + x1566 * x734 + x321 * x3962)
        + x3216 * x3946
        + x3528 * x4017
        - x3958 * x3959
        + x3974 * x3975
        - x3981 * x3982
        - x3983 * x3984
        - x3985 * x3986
        - x3990 * x3991
        + x574
        * (
            x1364 * x3994
            - x3964 * x556
            + x3994 * x545
            + x3995 * x3996
            + x3997 * x3998
            + x4000 * x4001
            + x4002 * x4005
            + x4006 * x4007
            + x4008 * x4009
            - x4012 * x515
        )
        + x823
        * (
            x1798 * x4013
            + x1805 * x4008
            + x1809 * x3995
            + x1811 * x3997
            + x1812 * x4002
            + x1813 * x4015
            + x1814 * x4016
            - x4014
            - x515 * x798
            - x515 * x799
        )
        + x906
        * (
            x1821 * x4013
            + x1827 * x4006
            + x1831 * x4008
            + x1832 * x3995
            + x1835 * x3997
            + x1837 * x4015
            + x1838 * x3993
            - x4018
            - x515 * x885
            - x515 * x886
            - x515 * x887
        )
    )
    K_block_list.append(
        -dq_i2 * x1022 * x4061
        + x1022 * x4028
        + x1022 * x4033
        + x1022 * x4040
        + x1022 * x4042
        + x1022 * x4043
        + x1022
        * (
            2 * dq_i1 * dq_i2 * dq_j4 * x2993
            + 2 * dq_i2 * dq_i3 * dq_j4 * x2995
            + dq_i2 * dq_i4 * dq_j4 * x2996
            + 2 * dq_i2 * dq_i5 * dq_j4 * x2992
            + 2 * dq_i2 * dq_i6 * dq_j4 * x2990
            + 2
            * dq_i2
            * dq_i7
            * dq_j4
            * sigma_kin_v_7_2
            * sigma_kin_v_7_4
            * x140
            * x144
            * x145
            * x148
            * x150
            * x153
            * x203
            * x434
            * x919
            - dq_i2 * x2717
            - dq_i2 * x2718
            - dq_i2 * x2719
            - dq_i2 * x2720
            - dq_i2 * x2721
            - dq_i2 * x2722
            - dq_i2 * x2723
            - dq_i2 * x2724
            + 2 * dq_j4 * x2994 * x711
            - x2690 * x4064
        )
        + x1142
        * (
            x1474 * x208 * x4075
            - x2737 * x4064
            - x2752 * x405
            - x2753 * x405
            - x2754 * x405
            - x2755 * x405
            + x2976 * x4068
            + x2977 * x4067
            + x2979 * x3995
            + x2981 * x4069
            + x2983 * x4071
            + x2984 * x4076
        )
        + x1199 * x1302
        + x1203 * x1608
        + x1205 * x1609
        + x1266
        * (
            dq_j4 * x3013 * x711
            + x1842 * x4075
            - x2463 * x4064
            - x2485 * x405
            - x2486 * x405
            - x2487 * x405
            - x2488 * x405
            + x3006 * x4068
            + x3007 * x4067
            + x3009 * x3995
            + x3011 * x4065
            + x3012 * x4071
        )
        + x1301 * x4026
        + x1304 * x2775
        - x1315 * x4030
        - x1317 * x4034
        - x1328 * x4041
        - x1342 * x4044
        + x136 * x4020
        + x136
        * (
            -x2292 * x4064
            - x2321 * x405
            - x2323 * x405
            - x2325 * x405
            - x2327 * x405
            + x2953 * x4074
            + x2957 * x4068
            + x2960 * x4067
            + x2961 * x4069
            + x2965 * x4065
            + x2968 * x4071
            + x2969 * x4021
        )
        - x1360 * (x1353 * x3962 + x1556 * x2592 + x1564 * x2594 + x1566 * x2596)
        + x1385
        * (
            x1364 * x4063
            + x1366 * x4066
            + x3995 * x4009
            + x3996 * x4069
            + x3998 * x4065
            + x4005 * x4067
            + x4007 * x4068
            - x4012 * x405
            + x4063 * x545
            - x4064 * x556
        )
        + x3354 * x4029
        + x3528 * x4073
        - x3959 * x4038
        - x3968 * x743
        + x3975 * x4049
        - x3982 * x4053
        - x3986 * x4056
        - x3991 * x4060
        - x4054 * x4055
        + x823
        * (
            x2555 * x407 * x4070
            - x2606 * x4064
            - x2622 * x405
            - x2623 * x405
            + x2924 * x3995
            + x2928 * x4069
            + x2930 * x4065
            + x2932 * x4067
            + x2933 * x4071
            + x2934 * x4072
        )
        + x906
        * (
            -x2644 * x4064
            - x2665 * x405
            - x2666 * x405
            - x2667 * x405
            + x2938 * x4074
            + x2941 * x4068
            + x2944 * x3995
            + x2945 * x4069
            + x2947 * x4065
            + x2949 * x4071
            + x2950 * x4062
        )
    )
    K_block_list.append(
        -dq_i3 * x1022 * x4108
        + x1022 * x4078
        + x1022 * x4083
        + x1022 * x4090
        + x1022 * x4092
        + x1022 * x4093
        + x1022
        * (
            2 * dq_i1 * dq_i3 * dq_j4 * x3736
            + 2 * dq_i2 * dq_i3 * dq_j4 * x3737
            + dq_i3 * dq_i4 * dq_j4 * x3739
            + 2 * dq_i3 * dq_i5 * dq_j4 * x3734
            + 2 * dq_i3 * dq_i6 * dq_j4 * x3732
            + 2
            * dq_i3
            * dq_i7
            * dq_j4
            * sigma_kin_v_7_3
            * sigma_kin_v_7_4
            * x140
            * x144
            * x145
            * x148
            * x150
            * x153
            * x203
            * x434
            * x919
            - dq_i3 * x3634
            - dq_i3 * x3635
            - dq_i3 * x3636
            - dq_i3 * x3637
            - dq_i3 * x3638
            - dq_i3 * x3639
            - dq_i3 * x3640
            - dq_i3 * x3641
            + 2 * dq_j4 * x3738 * x668
            - x3614 * x4111
        )
        + x1076 * x3653
        + x1079 * x1608
        + x1082 * x1609
        + x1142
        * (
            dq_j4 * x3755 * x668
            + x1474 * x208 * x4120
            - x3442 * x4111
            - x3457 * x411
            - x3458 * x411
            - x3459 * x411
            - x3460 * x411
            + x3749 * x4114
            + x3750 * x4113
            + x3752 * x3997
            + x3753 * x4065
            + x3754 * x4116
        )
        + x1266
        * (
            x1842 * x4120
            - x3373 * x4111
            - x3387 * x411
            - x3388 * x411
            - x3389 * x411
            - x3390 * x411
            + x3720 * x4114
            + x3721 * x4113
            + x3722 * x3997
            + x3723 * x4115
            + x3724 * x4116
            + x3725 * x4076
        )
        + x1304 * x2751
        + x136 * x4022
        + x136
        * (
            -x3238 * x4111
            - x3260 * x411
            - x3262 * x411
            - x3264 * x411
            - x3266 * x411
            + x3699 * x4119
            + x3703 * x4114
            + x3706 * x4113
            + x3707 * x4065
            + x3711 * x4115
            + x3714 * x4116
            + x3715 * x4023
        )
        - x1474 * x4030
        + x1601 * x4026
        - x1614 * x4084
        - x1617 * x4091
        - x1621 * x4044
        - x1627 * (x1556 * x3534 + x1564 * x3535 + x1566 * x3536 + x1624 * x3962)
        + x1636
        * (
            x1364 * x4110
            + x3996 * x4065
            + x3997 * x4009
            + x3998 * x4115
            + x4005 * x4113
            + x4007 * x4114
            - x4012 * x411
            + x411 * x4112
            + x4110 * x545
            - x4111 * x556
        )
        - x2787 * x3969
        + x3528 * x4118
        - x3959 * x4088
        + x3975 * x4097
        - x3982 * x4101
        - x3984 * x4103
        - x3991 * x4107
        - x4055 * x4102
        + x4079 * x4080
        + x823
        * (
            x3441 * x4070 * x412
            - x3543 * x4111
            - x3557 * x411
            - x3558 * x411
            + x3672 * x3997
            + x3673 * x4065
            + x3676 * x4115
            + x3680 * x4113
            + x3681 * x4116
            + x3682 * x4117
        )
        + x906
        * (
            x3148 * x4120
            - x3577 * x4111
            - x3592 * x411
            - x3593 * x411
            - x3594 * x411
            + x3688 * x4114
            + x3691 * x3997
            + x3692 * x4065
            + x3693 * x4115
            + x3695 * x4116
            + x3696 * x4109
        )
    )
    K_block_list.append(
        x1001 * x117
        + x1002 * x117
        + x1004 * x4123
        + x1007 * x4124
        + x1009 * x3421
        + x1015 * x4123
        + x1018 * x4124
        + x1022 * x4122
        + x1022 * x4128
        + x1022 * x4136
        + x1022 * x4137
        + x1022 * x4138
        + x1022
        * (
            2 * dq_i1 * (x4147 + x4256 * x965 + x4273 * x967 + x4274 * x969 + x4279 * x907)
            + 2 * dq_i2 * (x4148 + x4256 * x973 + x4273 * x974 + x4274 * x975 + x4279 * x971)
            + 2 * dq_i3 * (x4149 + x4256 * x985 + x4273 * x988 + x4274 * x990 + x4277 * x984)
            + 2 * dq_i4 * dq_i7 * dq_j4 * x140 * x144 * x145 * x148 * x150 * x153 * x1786 * x203 * x434 * x919
            - dq_i4 * x4160
            - dq_i4 * x4161
            - dq_i4 * x4162
            - dq_i4 * x4163
            - dq_i4 * x4164
            - dq_i4 * x4165
            - dq_i4 * x4166
            - dq_i4 * x4167
            + dq_i4
            * (
                x1001 * x4256
                + x1002 * x4256
                + x1004 * x4273
                + x1005 * x4140 * x61
                + x1007 * x4274
                + x1008 * x4141 * x61
                + x4139 * x61 * x999
                + x4168
                + x4277 * x998
            )
            + 2 * dq_i5 * (x263 * x4278 + x4146 + x4252 * x957 + x4256 * x954)
            + 2 * dq_i6 * (x223 * x4278 + x4132 + x4221 * x946)
            + 2 * dq_i7 * sigma_kin_v_7_4 * x140 * x144 * x145 * x148 * x150 * x153 * x195 * x203 * x434 * x919
            - x4122
            - x4128
            - x4136
            - x4137
            - x4138
            - x4142 * x4189
            - x537 * x941
        )
        + x1142
        * (
            2 * dq_i1 * (x1092 * x4250 + x1093 * x4273 + x1095 * x4274 + x4102 + x4276 * x907)
            + 2 * dq_i2 * (x1100 * x4250 + x1101 * x4273 + x1104 * x4274 + x4103 + x4276 * x971)
            + dq_i3
            * (
                x1116 * x4277
                + x1117 * x4277
                + x1119 * x4250
                + x1120 * x4250
                + x1122 * x4273
                + x1123 * x4273
                + x1125 * x4274
                + x1127 * x4274
                + x4108
            )
            + 2 * dq_i4 * dq_i7 * dq_j4 * x1035 * x140 * x143 * x146 * x148 * x150 * x153 * x1786 * x203 * x434
            + 2 * dq_i4 * (x1076 * x4277 + x1079 * x4273 + x1082 * x4274 + x2751 * x4256 + x4107)
            + 2 * dq_i5 * (x1069 * x4250 + x1070 * x4252 + x263 * x4275 + x4101)
            + 2 * dq_i6 * (x1062 * x4221 + x223 * x4275 + x4088)
            + 2 * dq_i7 * sigma_kin_v_7_4 * x1035 * x140 * x143 * x146 * x148 * x150 * x153 * x195 * x203 * x434
            - x1053 * x537
            - x1081 * x4265
            - x4078
            - x4083
            - x4090
            - x4092
            - x4093
            - x4097 * x4189
            - x4104 * x415
            - x4105 * x415
            - x4106 * x415
        )
        + x1266
        * (
            2 * dq_i1 * (x1154 * x4269 + x1161 * x4270 + x1230 * x4268 + x1234 * x4271 + x4054)
            + dq_i2
            * (
                x1242 * x4268
                + x1243 * x4045 * x61
                + x1245 * x4269
                + x1246 * x4046 * x61
                + x1248 * x4270
                + x1249 * x4047 * x61
                + x1250 * x4271
                + x1251 * x4048 * x61
                + x4061
            )
            + 2 * dq_i3 * (x1214 * x4272 + x1215 * x4250 + x1216 * x4273 + x1219 * x4274 + x4056)
            + 2 * dq_i4 * dq_i7 * dq_j4 * x1156 * x139 * x144 * x146 * x148 * x150 * x153 * x1786 * x203 * x434
            + 2 * dq_i4 * (x1199 * x4272 + x1203 * x4273 + x1205 * x4274 + x2775 * x4256 + x4060)
            + 2 * dq_i5 * (x1190 * x4250 + x1191 * x4267 + x1192 * x4252 + x4053)
            + 2 * dq_i6 * (x1181 * x4267 + x1184 * x4221 + x4038)
            + 2 * dq_i7 * sigma_kin_v_7_4 * x1156 * x139 * x144 * x146 * x148 * x150 * x153 * x195 * x203 * x434
            - x1157 * x4266
            - x1172 * x537
            - x4028
            - x4033
            - x4040
            - x4042
            - x4043
            - x4049 * x4189
            - x4057 * x415
            - x4058 * x415
            - x4059 * x415
        )
        + x1334
        * (
            ddq_i4 * (x1783 * x623 + x1784 * x582 + x1785 * x466 + x1786 * x468)
            + x119 * x120 * x123 * x127 * x1784 * x292 * x580 * x641
            + x140 * x144 * x145 * x148 * x150 * x154 * x1786 * x306 * x434 * x641
            + x166 * x167 * x170 * x172 * x176 * x1785 * x299 * x464 * x641
            + x1783 * x287 * x621 * x641 * x67 * x74 * x78
            - x4203 * x615
            - x500 * (x221 * x4205 + x4206 * x509)
            - x605 * (x255 * x4207 + x263 * x4204 + x4203 * x504)
            - x631 * (x3970 * x85 + x3971 * x585 + x3972 * x471 + x3973 * x473)
            - x635 * (x372 * x4045 + x377 * x4046 + x383 * x4047 + x389 * x4048)
            - x639 * (x343 * x3660 + x4094 * x597 + x4095 * x485 + x4096 * x489)
            - x642 * (x1784 * x603 + x287 * x4139 + x299 * x4140 + x306 * x4141)
        )
        + x136 * x4024
        + x136
        * (
            dq_i1
            * (
                x1295 * x4268
                + x130 * x4269
                + x134 * x4269
                + x179 * x4270
                + x183 * x4270
                + x186 * x4271
                + x201 * x4271
                + x3970 * x61 * x65
                + x3992
            )
            + 2 * dq_i2 * (x1291 * x4268 + x1292 * x4269 + x1293 * x4270 + x1294 * x4271 + x3983)
            + 2 * dq_i3 * (x1284 * x4272 + x1285 * x4250 + x1286 * x4273 + x1287 * x4274 + x3985)
            + 2 * dq_i4 * dq_i7 * dq_j4 * x137 * x138 * x140 * x144 * x146 * x148 * x150 * x153 * x1786 * x203
            + 2 * dq_i4 * (x1277 * x4272 + x1279 * x4273 + x1281 * x4274 + x2769 * x4256 + x3990)
            + 2 * dq_i5 * (x1271 * x4250 + x1273 * x4218 + x1275 * x4252 + x3981)
            + 2 * dq_i6 * (x1268 * x4267 + x1269 * x4219 + x3958)
            + 2 * dq_i7 * sigma_kin_v_7_4 * x137 * x138 * x140 * x144 * x146 * x148 * x150 * x153 * x195 * x203
            - x142 * x4266
            - x249 * x537
            - x3944
            - x3951
            - x3961
            - x3966
            - x3967
            - x3974 * x4189
            - x3987 * x415
            - x3988 * x415
            - x3989 * x415
        )
        - x1444 * x4030
        + x1794
        * (
            x1364 * x190
            + x190 * x545
            + x417 * x4188
            - x4182
            + x4185 * x522
            + x4185 * x536
            + x4186 * x527
            + x4187 * x529
            + x4188 * x421
            - x4189 * x556
            + x4190
            + x560
        )
        + x2188 * x4017
        + x2435 * x3942 * (sigma_pot_4_c * x73 + sigma_pot_4_s * x70)
        + x3189 * x4073
        + x3305 * x4118
        + x3421 * x998
        + x3493 * x4181
        + x3514 * x4202
        - x3959 * x4132
        + x3975 * x4142
        - x3982 * x4146
        - x3984 * x4148
        - x3986 * x4149
        - x4055 * x4147
        - x4157 * x4158 * x4159
        - 4 * x4168 * x61
        + x823
        * (
            2 * dq_i1 * (x4218 * x773 + x4219 * x776 + x4220)
            + 2 * dq_i2 * (x4218 * x780 + x4222 * x782 + x4223)
            + 2 * dq_i3 * (x4218 * x789 + x4222 * x790 + x4224)
            + 2 * dq_i4 * dq_i7 * dq_j4 * x140 * x144 * x146 * x148 * x149 * x153 * x1786 * x203 * x434 * x744
            + 2 * dq_i4 * (x4218 * x796 + x4222 * x797 + x4226)
            + 2 * dq_i5 * (x4221 * x806 + x4227 * x803 + x4231)
            + dq_i6 * (x4205 * x61 * x811 + x4219 * x812 + x4221 * x814 + x4227 * x810 + x4236)
            + 2 * dq_i7 * sigma_kin_v_7_4 * x140 * x144 * x146 * x148 * x149 * x153 * x195 * x203 * x434 * x744
            - x154 * x1824 * x2803 * x4210 * x496
            - x415 * x4209
            - x4189 * x4217
            - x4208
            - x4211
            - x4212
            - x4214
            - x4216
            - x537 * x762
        )
        + x906
        * (
            2 * dq_i1 * (x1422 * x4252 + x4250 * x863 + x4251 * x772 + x4253)
            + 2 * dq_i2 * (x2656 * x4252 + x4250 * x868 + x4251 * x779 + x4254)
            + 2 * dq_i3 * (x2658 * x4252 + x4250 * x875 + x4251 * x485 + x4255)
            + 2 * dq_i4 * dq_i7 * dq_j4 * x140 * x144 * x146 * x147 * x150 * x153 * x1786 * x203 * x434 * x828
            + 2 * dq_i4 * (x1433 * x4252 + x2664 * x4256 + x4251 * x494 + x4257)
            + dq_i5
            * (x1436 * x4252 + x4207 * x61 * x891 + x4227 * x892 + x4227 * x893 + x4250 * x890 + x4252 * x896 + x4264)
            + 2 * dq_i6 * (x4221 * x858 + x4227 * x857 + x4249)
            + 2 * dq_i7 * sigma_kin_v_7_4 * x140 * x144 * x146 * x147 * x150 * x153 * x195 * x203 * x434 * x828
            - x1825 * x4210 * x828 * x883
            - x415 * x4237
            - x415 * x4239
            - x4189 * x4246
            - x4238
            - x4241
            - x4242
            - x4243
            - x4245
            - x537 * x850
        )
    )
    K_block_list.append(
        -dq_i5 * x1022 * x4264
        + x1022 * x4238
        + x1022 * x4241
        + x1022 * x4242
        + x1022 * x4243
        + x1022 * x4245
        + x1022
        * (
            2 * dq_i1 * dq_i5 * dq_j4 * x4359
            + 2 * dq_i2 * dq_i5 * dq_j4 * x4360
            + 2 * dq_i3 * dq_i5 * dq_j4 * x4361
            + dq_i4 * dq_i5 * dq_j4 * x4363
            + 2 * dq_i5 * dq_i6 * dq_j4 * x4358
            + 2
            * dq_i5
            * dq_i7
            * dq_j4
            * sigma_kin_v_7_4
            * sigma_kin_v_7_5
            * x140
            * x144
            * x145
            * x148
            * x150
            * x153
            * x203
            * x434
            * x919
            - dq_i5 * x4351
            - dq_i5 * x4352
            - dq_i5 * x4353
            - dq_i5 * x4354
            - dq_i5 * x4355
            - dq_i5 * x4356
            + 2 * dq_j4 * x4362 * x610
            - x4135 * x4288
        )
        + x1142
        * (
            dq_i5 * x4109 * x4340
            + x1474 * x208 * x4322
            + x4002 * x4336
            + x4067 * x4337
            - x4089 * x4288
            - x419 * x4331
            - x419 * x4332
            - x419 * x4333
            + x4289 * x4335
            + x4290 * x4339
            + x4299 * x4338
        )
        + x1266
        * (
            dq_i5 * x4062 * x4330
            + x1842 * x4322
            + x4002 * x4326
            - x4039 * x4288
            + x4113 * x4327
            - x419 * x4321
            - x419 * x4323
            - x419 * x4324
            + x4289 * x4325
            + x4290 * x4329
            + x4299 * x4328
        )
        + x1304 * x2664
        - x1316 * x4283
        + x136 * x4018
        + x136
        * (
            dq_i5 * x3993 * x4320
            - x3960 * x4288
            + x4067 * x4311
            + x4113 * x4315
            - x419 * x4305
            - x419 * x4307
            - x419 * x4308
            + x4289 * x4310
            + x4290 * x4319
            + x4291 * x4306
            + x4299 * x4316
        )
        + x1433 * x1925
        + x1919 * x4025 * x4280
        + x1924 * x494
        - x1930 * x4041
        - x1931 * x4091
        - x1935 * (x1932 * x3962 + x3952 * x4284 + x3953 * x4285)
        + x1941
        * (
            x1364 * x4287
            + x3996 * x4067
            + x3998 * x4113
            + x4002 * x4009
            + x4005 * x4290
            + x4007 * x4289
            - x4012 * x419
            + x4112 * x419
            + x4287 * x545
            - x4288 * x556
        )
        - x3016 * x3963
        - x3148 * x3948
        + x3528 * x4202
        + x3756 * x4079
        - x3959 * x4249
        + x3975 * x4246
        - x3984 * x4254
        - x3986 * x4255
        - x3991 * x4257
        - x4055 * x4253
        + x823
        * (
            x4002 * x4295
            + x4067 * x4296
            + x4113 * x4297
            - x419 * x4293
            - x419 * x4294
            - x4213 * x4288
            + x4290 * x4302
            + x4291 * x4292
            + x4298 * x4299
            + x4303 * x4304
        )
        + x906
        * (
            dq_j4 * x4350 * x610
            + x3148 * x4322
            + x4002 * x4346
            + x4067 * x4347
            + x4113 * x4348
            - x419 * x4341
            - x419 * x4342
            - x419 * x4343
            - x4244 * x4288
            + x4289 * x4344
            + x4299 * x4349
        )
    )
    K_block_list.append(
        -dq_i6 * x1022 * x4236
        + x1022 * x4208
        + x1022 * x4211
        + x1022 * x4212
        + x1022 * x4214
        + x1022 * x4216
        + x1022
        * (
            -dq_i6 * x4425
            - dq_i6 * x4426
            - dq_i6 * x4427
            - dq_i6 * x4428
            + dq_i6 * x4435 * x61
            + x1444 * x4396
            + x4006 * x4429
            + x4068 * x4430
            + x4114 * x4431
            - x4127 * x4369
            + x4289 * x4433
            + x4370 * x4434
        )
        + x1142
        * (
            x1474 * x4396
            + x4006 * x4399
            + x4068 * x4400
            - x4082 * x4369
            + x4117 * x4405
            - x423 * x4397
            - x423 * x4398
            + x4289 * x4403
            + x4370 * x4404
            + x4380 * x4401
        )
        + x1266
        * (
            x1842 * x4386
            + x4006 * x4389
            - x4032 * x4369
            + x4072 * x4395
            + x4114 * x4390
            - x423 * x4387
            - x423 * x4388
            + x4289 * x4392
            + x4370 * x4394
            + x4380 * x4391
        )
        + x1311 * x796
        + x1313 * x4225
        - x1327 * x4283
        + x136 * x4014
        + x136
        * (
            -x3950 * x4369
            + x4016 * x4385
            + x4068 * x4375
            + x4114 * x4378
            - x423 * x4373
            - x423 * x4374
            + x4289 * x4383
            + x4370 * x4384
            + x4371 * x4372
            + x4379 * x4380
        )
        - x1797 * x1824 * x3947
        - x2052 * x4034
        - x2053 * x4084
        - x2056 * (x3952 * x4366 + x3953 * x4367)
        + x2061
        * (
            x1364 * x4368
            + x3996 * x4068
            + x3998 * x4114
            + x4005 * x4289
            + x4006 * x4009
            + x4007 * x4370
            - x4012 * x423
            + x4112 * x423
            + x4368 * x545
            - x4369 * x556
        )
        - x3104 * x3954
        + x3528 * x4181
        + x3842 * x3946
        + x3975 * x4217
        - x3982 * x4231
        - x3984 * x4223
        - x3986 * x4224
        - x3991 * x4226
        + x4025 * x4364 * x4365
        - x4055 * x4220
        + x823
        * (
            dq_j4 * x4424 * x512
            + x4006 * x4419
            + x4068 * x4420
            + x4114 * x4421
            - x4215 * x4369
            - x423 * x4417
            - x423 * x4418
            + x4289 * x4423
            + x4371 * x4416
            + x4380 * x4422
        )
        + x906
        * (
            x3148 * x4386
            + x4006 * x4409
            + x4068 * x4410
            + x4114 * x4411
            - x423 * x4406
            - x423 * x4407
            - x4240 * x4369
            + x4304 * x4415
            + x4370 * x4414
            + x4380 * x4412
        )
    )
    K_block_list.append(
        sigma_kin_v_7_4 * x1022 * x2159
        + 1.0 * x1769 * x2149 * x4025
        - x1824 * x3188 * x3959
        - x190 * x4443
        + x2152 * x3975 * x4203
        + x2154 * x3964
        + x217 * x4169 * x4436
        - x2176 * x4055 * x4442
        - x2182 * x3984 * x4442
        - x2186 * x3979 * x3982
        + x2209
        * (
            x1156 * x4448
            - x1156 * x4452
            + x1156 * x4454
            + x2190 * x4063
            + x2194 * x4063
            + x2197 * x4453
            + x2200 * x4453
            + x2202 * x4453
            - x2205 * x4450
            + x3198 * x4453
        )
        + x2218
        * (
            x1035 * x4448
            - x1035 * x4452
            + x1035 * x4454
            - x1035 * x4456
            + x2200 * x4455
            + x2202 * x4455
            + x2211 * x4110
            + x2213 * x4110
            + x2214 * x4455
            + x3198 * x4455
        )
        - x2225 * x3528 * x4157 * x4159
        + x2228
        * (
            -x1008 * x567
            + x190 * x2220
            + x190 * x2223
            + x2197 * x4460
            + x2200 * x4460
            + x2202 * x4460
            + x2214 * x4460
            - x2222 * x4449
            + x3198 * x4460
            + x4448 * x919
            - x4452 * x919
        )
        + x2236
        * (
            x2197 * x4457
            + x2202 * x4457
            + x2214 * x4457
            + x2230 * x4287
            + x2232 * x4287
            + x3198 * x4457
            + x4448 * x828
            - x4452 * x828
            + x4454 * x828
            - x4456 * x828
        )
        + x2243
        * (
            x2197 * x4458
            + x2200 * x4458
            + x2214 * x4458
            + x2238 * x4368
            + x2239 * x4368
            + x3198 * x4458
            + x4448 * x744
            - x4452 * x744
            + x4454 * x744
            - x4456 * x744
        )
        + x2250
        * (
            dq_j4 * x3205 * x555
            + x2197 * x4459
            + x2200 * x4459
            + x2202 * x4459
            + x2214 * x4459
            + x2247 * x4286
            + x3198 * x4459
            - x4286 * x569
            + x4454 * x520
            - x4456 * x520
        )
        + x2257
        * (
            x137 * x4448
            + x2197 * x4445
            + x2200 * x4445
            + x2202 * x4445
            + x2214 * x4445
            + x2251 * x4444
            - x2252 * x4286
            + x2253 * x4444
            + x4446 * x4447
            - x4450 * x4451
        )
        + x244 * x4191 * x4436
        + x2556 * x281 * x4436
        + x275 * x3515 * x4436
        - x3183 * x4441
        - x3184 * x4441
        + x3916 * x4437
        + x3917 * x498
        + x3920 * x4438
        + x3921 * x4437
        + x3924 * x497
        - x3926 * x3986 * x4442
        + x4438 * x4439
        + x4440 * x498
    )
    K_block_list.append(
        -dq_i1 * x4505 * x906
        + x1022
        * (
            x1444 * x4530
            + x2009 * x4517
            + x2010 * x4519
            + x2011 * x4508
            + x2013 * x4510
            + x2014 * x4524
            + x2015 * x4534
            - x4533
            - x515 * x958
            - x515 * x959
            - x515 * x960
        )
        + x1142
        * (
            -x1071 * x515
            - x1072 * x515
            - x1073 * x515
            + x1474 * x4530
            + x1998 * x4517
            + x1999 * x4519
            + x2001 * x4508
            + x2002 * x4512
            + x2003 * x4524
            + x2004 * x4532
            - x4531
        )
        + x1266
        * (
            -x1193 * x515
            - x1194 * x515
            - x1196 * x515
            + x1842 * x4527
            + x1970 * x4517
            + x1971 * x4519
            + x1978 * x4510
            + x1982 * x4512
            + x1985 * x4524
            + x1986 * x4529
            - x4528
        )
        + x1271 * x1603
        + x1272 * x4467
        + x1275 * x1925
        - x1317 * x4474
        - x1335 * x4488
        - x1342 * x4487
        - x1359 * x4486
        + x136 * x4481
        + x136
        * (
            dq_j5 * x11 * x2044
            + x2039 * x4517
            + x2040 * x4508
            + x2041 * x4510
            + x2042 * x4512
            + x2043 * x4524
            + x2137 * x4522 * x576
            - x260 * x515
            - x264 * x515
            - x269 * x515
            - x4481
        )
        - x2137 * x4469
        + x2258 * x4280 * x4462
        - x2319 * (x1551 * x727 + x1553 * x723 + x4473 * x721)
        + x3216 * x4466
        + x3514 * x4526
        + x4464 * x906
        + x4472 * x906
        - x4478 * x4479
        + x4482 * x906
        + x4483 * x906
        + x4484 * x906
        + x4492 * x4493
        - x4494 * x4495
        - x4496 * x4497
        - x4498 * x4499
        - x4503 * x4504
        + x574
        * (
            x1364 * x4507
            + x4000 * x4516
            - x4480 * x559
            + x4507 * x545
            + x4508 * x4509
            + x4510 * x4511
            + x4512 * x4515
            + x4517 * x4518
            + x4519 * x4520
            - x4521 * x515
        )
        + x823
        * (
            x1943 * x4522
            + x1950 * x4519
            + x1954 * x4508
            + x1956 * x4510
            + x1957 * x4512
            + x1959 * x4524
            + x1960 * x4525
            - x4523
            - x515 * x805
            - x515 * x807
        )
        + x906
        * (
            2 * dq_i1 * dq_i2 * dq_j5 * x2027
            + 2 * dq_i1 * dq_i3 * dq_j5 * x2029
            + 2 * dq_i1 * dq_i4 * dq_j5 * x2030
            + dq_i1 * dq_i5 * dq_j5 * x2031
            + 2 * dq_i1 * dq_i6 * dq_j5 * x2023
            + 2
            * dq_i1
            * dq_i7
            * dq_j5
            * sigma_kin_v_7_1
            * sigma_kin_v_7_5
            * x140
            * x144
            * x146
            * x147
            * x150
            * x153
            * x203
            * x434
            * x828
            - dq_i1 * x899
            - dq_i1 * x900
            - dq_i1 * x901
            - dq_i1 * x902
            - dq_i1 * x903
            - dq_i1 * x904
            + 2 * dq_j5 * x11 * x2026
            - x4535
        )
    )
    K_block_list.append(
        -dq_i2 * x4565 * x906
        + x1022
        * (
            x1444 * x4580
            - x2685 * x4568
            - x2700 * x405
            - x2701 * x405
            - x2702 * x405
            + x3073 * x4572
            + x3074 * x4508
            + x3075 * x4573
            + x3077 * x4569
            + x3078 * x4575
            + x3079 * x4582
        )
        + x1142
        * (
            x1474 * x4580
            - x2734 * x4568
            - x2747 * x405
            - x2748 * x405
            - x2749 * x405
            + x3063 * x4572
            + x3064 * x4508
            + x3065 * x4573
            + x3066 * x4570
            + x3067 * x4575
            + x3068 * x4581
        )
        + x1190 * x1603
        + x1191 * x4467
        + x1192 * x1925
        + x1266
        * (
            dq_j5 * x3102 * x711
            + x1842 * x4579
            - x2454 * x4568
            - x2465 * x405
            - x2466 * x405
            - x2467 * x405
            + x3097 * x4572
            + x3098 * x4508
            + x3099 * x4569
            + x3100 * x4570
            + x3101 * x4575
        )
        - x1315 * x4540
        - x1317 * x4543
        - x1335 * x4552
        - x1342 * x4551
        + x136 * x4528
        + x136
        * (
            -x2285 * x4568
            - x2298 * x405
            - x2299 * x405
            - x2301 * x405
            + x3038 * x4578
            + x3042 * x4572
            + x3043 * x4573
            + x3049 * x4569
            + x3053 * x4570
            + x3056 * x4575
            + x3057 * x4529
        )
        - x1360 * (x1551 * x2590 + x1553 * x2586 + x2585 * x4473)
        + x1385
        * (
            x1364 * x4567
            + x1366 * x4571
            - x405 * x4521
            + x4508 * x4520
            + x4509 * x4573
            + x4511 * x4569
            + x4515 * x4570
            + x4518 * x4572
            + x4567 * x545
            - x4568 * x559
        )
        + x3354 * x4539
        + x3514 * x4577
        - x4479 * x4547
        - x4486 * x743
        + x4493 * x4556
        - x4497 * x4559
        - x4499 * x4560
        - x4504 * x4564
        + x4536 * x4537
        + x4538 * x906
        + x4542 * x906
        + x4548 * x906
        + x4549 * x906
        + x4550 * x906
        - x4557 * x4558
        + x823
        * (
            x2544 * x407 * x4574
            - x2608 * x4568
            - x2627 * x405
            - x2628 * x405
            + x3025 * x4508
            + x3029 * x4573
            + x3031 * x4569
            + x3032 * x4570
            + x3034 * x4575
            + x3035 * x4576
        )
        + x906
        * (
            2 * dq_i1 * dq_i2 * dq_j5 * x3086
            + 2 * dq_i2 * dq_i3 * dq_j5 * x3088
            + 2 * dq_i2 * dq_i4 * dq_j5 * x3089
            + dq_i2 * dq_i5 * dq_j5 * x3090
            + 2 * dq_i2 * dq_i6 * dq_j5 * x3084
            + 2
            * dq_i2
            * dq_i7
            * dq_j5
            * sigma_kin_v_7_2
            * sigma_kin_v_7_5
            * x140
            * x144
            * x146
            * x147
            * x150
            * x153
            * x203
            * x434
            * x828
            - dq_i2 * x2669
            - dq_i2 * x2670
            - dq_i2 * x2671
            - dq_i2 * x2672
            - dq_i2 * x2673
            - dq_i2 * x2674
            + 2 * dq_j5 * x3087 * x711
            - x2646 * x4568
        )
    )
    K_block_list.append(
        -dq_i3 * x4608 * x906
        + x1022
        * (
            x1444 * x4621
            - x3611 * x4611
            - x3622 * x411
            - x3623 * x411
            - x3624 * x411
            + x3811 * x4614
            + x3812 * x4510
            + x3813 * x4569
            + x3814 * x4615
            + x3815 * x4616
            + x3816 * x4622
        )
        + x1045 * x4587
        + x1069 * x1603
        + x1070 * x1925
        + x1142
        * (
            dq_j5 * x3839 * x668
            + x1474 * x4621
            - x3439 * x4611
            - x3444 * x411
            - x3445 * x411
            - x3446 * x411
            + x3834 * x4614
            + x3835 * x4510
            + x3836 * x4569
            + x3837 * x4612
            + x3838 * x4616
        )
        + x1266
        * (
            x1842 * x4620
            - x3367 * x4611
            - x3376 * x411
            - x3377 * x411
            - x3378 * x411
            + x3801 * x4614
            + x3802 * x4510
            + x3803 * x4615
            + x3804 * x4612
            + x3805 * x4616
            + x3806 * x4581
        )
        + x136 * x4531
        + x136
        * (
            -x3233 * x4611
            - x3242 * x411
            - x3243 * x411
            - x3245 * x411
            + x3779 * x4619
            + x3783 * x4614
            + x3784 * x4569
            + x3788 * x4615
            + x3792 * x4612
            + x3795 * x4616
            + x3796 * x4532
        )
        - x1474 * x4540
        - x1614 * x4590
        - x1620 * x4598
        - x1621 * x4551
        - x1627 * (x1551 * x3533 + x1553 * x3530 + x3529 * x4473)
        + x1636
        * (
            x1364 * x4610
            - x411 * x4521
            + x411 * x4613
            + x4509 * x4569
            + x4510 * x4520
            + x4511 * x4615
            + x4515 * x4612
            + x4518 * x4614
            + x4610 * x545
            - x4611 * x559
        )
        - x2787 * x4487
        + x3514 * x4618
        + x4080 * x4586
        - x4479 * x4594
        + x4493 * x4600
        - x4495 * x4602
        - x4499 * x4603
        - x4504 * x4607
        + x4537 * x4583
        - x4558 * x4601
        + x4585 * x906
        + x4589 * x906
        + x4595 * x906
        + x4596 * x906
        + x4597 * x906
        + x823
        * (
            x3438 * x412 * x4574
            - x3545 * x4611
            - x3562 * x411
            - x3563 * x411
            + x3766 * x4510
            + x3767 * x4569
            + x3770 * x4615
            + x3772 * x4612
            + x3775 * x4616
            + x3776 * x4617
        )
        + x906
        * (
            2 * dq_i1 * dq_i3 * dq_j5 * x3822
            + 2 * dq_i2 * dq_i3 * dq_j5 * x3823
            + 2 * dq_i3 * dq_i4 * dq_j5 * x3825
            + dq_i3 * dq_i5 * dq_j5 * x3826
            + 2 * dq_i3 * dq_i6 * dq_j5 * x3820
            + 2
            * dq_i3
            * dq_i7
            * dq_j5
            * sigma_kin_v_7_3
            * sigma_kin_v_7_5
            * x140
            * x144
            * x146
            * x147
            * x150
            * x153
            * x203
            * x434
            * x828
            - dq_i3 * x3596
            - dq_i3 * x3597
            - dq_i3 * x3598
            - dq_i3 * x3599
            - dq_i3 * x3600
            - dq_i3 * x3601
            + 2 * dq_j5 * x3824 * x668
            - x3579 * x4611
        )
    )
    K_block_list.append(
        -dq_i4 * x4644 * x906
        + x1022
        * (
            dq_j5 * x4363 * x641
            + x1444 * x4655
            - x4135 * x4647
            - x4143 * x415
            - x4144 * x415
            - x4145 * x415
            + x4358 * x4648
            + x4359 * x4512
            + x4360 * x4570
            + x4361 * x4612
            + x4362 * x4651
        )
        + x1142
        * (
            x1474 * x4655
            - x4089 * x4647
            - x4098 * x415
            - x4099 * x415
            - x4100 * x415
            + x4335 * x4648
            + x4336 * x4512
            + x4337 * x4570
            + x4338 * x4649
            + x4339 * x4651
            + x4340 * x4622
        )
        + x1266
        * (
            x1842 * x4654
            - x4039 * x4647
            - x4050 * x415
            - x4051 * x415
            - x4052 * x415
            + x4325 * x4648
            + x4326 * x4512
            + x4327 * x4612
            + x4328 * x4649
            + x4329 * x4651
            + x4330 * x4582
        )
        + x1304 * x954
        - x1316 * x4627
        + x136 * x4533
        + x136
        * (
            -x3960 * x4647
            - x3977 * x415
            - x3978 * x415
            - x3980 * x415
            + x4306 * x4650
            + x4310 * x4648
            + x4311 * x4570
            + x4315 * x4612
            + x4316 * x4649
            + x4319 * x4651
            + x4320 * x4534
        )
        - x1444 * x4540
        + x1768 * x4461 * x4623
        - x1781 * x4552
        - x1782 * x4598
        - x1787 * (x1551 * x4207 + x1553 * x4204 + x4203 * x4473)
        + x1794
        * (
            x1364 * x4646
            - x415 * x4521
            + x415 * x4613
            + x4509 * x4570
            + x4511 * x4612
            + x4512 * x4520
            + x4515 * x4649
            + x4518 * x4648
            + x4646 * x545
            - x4647 * x559
        )
        + x1925 * x957
        - x2913 * x4488
        + x3514 * x4653
        + x3655 * x4586
        - x4479 * x4631
        + x4493 * x4637
        - x4495 * x4639
        - x4497 * x4640
        - x4504 * x4643
        - x4558 * x4638
        + x4587 * x929
        + x4624 * x906
        + x4626 * x906
        + x4632 * x906
        + x4633 * x906
        + x4634 * x906
        + x823
        * (
            -x415 * x4229
            - x415 * x4230
            - x4213 * x4647
            + x4292 * x4650
            + x4295 * x4512
            + x4296 * x4570
            + x4297 * x4612
            + x4298 * x4649
            + x4302 * x4651
            + x4303 * x4652
        )
        + x906
        * (
            2 * dq_i1 * dq_i4 * dq_j5 * x4346
            + 2 * dq_i2 * dq_i4 * dq_j5 * x4347
            + 2 * dq_i3 * dq_i4 * dq_j5 * x4348
            + dq_i4 * dq_i5 * dq_j5 * x4350
            + 2 * dq_i4 * dq_i6 * dq_j5 * x4344
            + 2
            * dq_i4
            * dq_i7
            * dq_j5
            * sigma_kin_v_7_4
            * sigma_kin_v_7_5
            * x140
            * x144
            * x146
            * x147
            * x150
            * x153
            * x203
            * x434
            * x828
            - dq_i4 * x4258
            - dq_i4 * x4259
            - dq_i4 * x4260
            - dq_i4 * x4261
            - dq_i4 * x4262
            - dq_i4 * x4263
            + 2 * dq_j5 * x4349 * x641
            - x4244 * x4647
        )
    )
    K_block_list.append(
        x1022
        * (
            2 * dq_i1 * (x4638 + x4734 * x967 + x4735 * x969 + x4736 * x965)
            + 2 * dq_i2 * (x4639 + x4734 * x974 + x4735 * x975 + x4736 * x973)
            + 2 * dq_i3 * (x4640 + x4734 * x988 + x4735 * x990 + x4736 * x985)
            + dq_i4
            * (
                x1001 * x4736
                + x1002 * x4736
                + x1004 * x4734
                + x1005 * x114 * x4635
                + x1007 * x4735
                + x1008 * x114 * x4636
                + x4644
            )
            + 2 * dq_i5 * dq_i7 * dq_j5 * x140 * x144 * x145 * x148 * x150 * x153 * x1934 * x203 * x434 * x919
            + 2 * dq_i5 * (x263 * x4739 + x4643 + x4736 * x954 + x4737 * x957)
            + 2 * dq_i6 * (x223 * x4739 + x4631 + x4717 * x946)
            + 2 * dq_i7 * sigma_kin_v_7_5 * x140 * x144 * x145 * x148 * x150 * x153 * x195 * x203 * x434 * x919
            - x2021 * x4154 * x956
            - x419 * x4641
            - x419 * x4642
            - x4624
            - x4626
            - x4632
            - x4633
            - x4634
            - x4637 * x4702
            - x537 * x935
        )
        - 4 * x114 * x4685
        + x1142
        * (
            2 * dq_i1 * (x1092 * x4733 + x1093 * x4734 + x1095 * x4735 + x4601)
            + 2 * dq_i2 * (x1100 * x4733 + x1101 * x4734 + x1104 * x4735 + x4602)
            + dq_i3
            * (x1119 * x4733 + x1120 * x4733 + x1122 * x4734 + x1123 * x4734 + x1125 * x4735 + x1127 * x4735 + x4608)
            + 2 * dq_i4 * (x1079 * x4734 + x1082 * x4735 + x2751 * x4736 + x4603)
            + 2 * dq_i5 * dq_i7 * dq_j5 * x1035 * x140 * x143 * x146 * x148 * x150 * x153 * x1934 * x203 * x434
            + 2 * dq_i5 * (x1069 * x4733 + x1070 * x4737 + x263 * x4738 + x4607)
            + 2 * dq_i6 * (x1062 * x4717 + x223 * x4738 + x4594)
            + 2 * dq_i7 * sigma_kin_v_7_5 * x1035 * x140 * x143 * x146 * x148 * x150 * x153 * x195 * x203 * x434
            - x1051 * x537
            - x419 * x4604
            - x419 * x4605
            - x419 * x4606
            - x4585
            - x4589
            - x4595
            - x4596
            - x4597
            - x4600 * x4702
        )
        + x1266
        * (
            2 * dq_i1 * (x1154 * x4730 + x1161 * x4731 + x1234 * x4732 + x4557)
            + dq_i2
            * (
                x114 * x1246 * x4553
                + x114 * x1249 * x4554
                + x114 * x1251 * x4555
                + x1245 * x4730
                + x1248 * x4731
                + x1250 * x4732
                + x4565
            )
            + 2 * dq_i3 * (x1215 * x4733 + x1216 * x4734 + x1219 * x4735 + x4559)
            + 2 * dq_i4 * (x1203 * x4734 + x1205 * x4735 + x2775 * x4736 + x4560)
            + 2 * dq_i5 * dq_i7 * dq_j5 * x1156 * x139 * x144 * x146 * x148 * x150 * x153 * x1934 * x203 * x434
            + 2 * dq_i5 * (x1190 * x4733 + x1191 * x4729 + x1192 * x4737 + x4564)
            + 2 * dq_i6 * (x1181 * x4729 + x1184 * x4717 + x4547)
            + 2 * dq_i7 * sigma_kin_v_7_5 * x1156 * x139 * x144 * x146 * x148 * x150 * x153 * x195 * x203 * x434
            - x1168 * x537
            - x419 * x4561
            - x419 * x4562
            - x419 * x4563
            - x4538
            - x4542
            - x4548
            - x4549
            - x4550
            - x4556 * x4702
        )
        + x1327
        * (
            ddq_i5 * (x1932 * x582 + x1933 * x466 + x1934 * x468)
            + x119 * x121 * x122 * x127 * x1932 * x255 * x580 * x610
            + x140 * x144 * x146 * x147 * x150 * x154 * x1934 * x265 * x434 * x610
            + x166 * x168 * x169 * x172 * x176 * x1933 * x261 * x464 * x610
            - x4671 * x577
            - x507 * (x221 * x4703 + x4704 * x509)
            - x590 * (x4489 * x585 + x4490 * x471 + x4491 * x473)
            - x594 * (x377 * x4553 + x383 * x4554 + x389 * x4555)
            - x601 * (x4284 * x485 + x4285 * x489 + x4599 * x597)
            - x605 * (x1932 * x603 + x299 * x4635 + x306 * x4636)
            - x611 * (x255 * x4669 + x263 * x4670 + x4671 * x504)
        )
        + x136 * x4535
        + x136
        * (
            dq_i1 * (x130 * x4730 + x134 * x4730 + x179 * x4731 + x183 * x4731 + x186 * x4732 + x201 * x4732 + x4505)
            + 2 * dq_i2 * (x1292 * x4730 + x1293 * x4731 + x1294 * x4732 + x4494)
            + 2 * dq_i3 * (x1285 * x4733 + x1286 * x4734 + x1287 * x4735 + x4496)
            + 2 * dq_i4 * (x1279 * x4734 + x1281 * x4735 + x2769 * x4736 + x4498)
            + 2 * dq_i5 * dq_i7 * dq_j5 * x137 * x138 * x140 * x144 * x146 * x148 * x150 * x153 * x1934 * x203
            + 2 * dq_i5 * (x1271 * x4733 + x1273 * x4714 + x1275 * x4737 + x4503)
            + 2 * dq_i6 * (x1268 * x4729 + x1269 * x4715 + x4478)
            + 2 * dq_i7 * sigma_kin_v_7_5 * x137 * x138 * x140 * x144 * x146 * x148 * x150 * x153 * x195 * x203
            - x243 * x537
            - x419 * x4500
            - x419 * x4501
            - x419 * x4502
            - x4464
            - x4472
            - x4482
            - x4483
            - x4484
            - x4492 * x4702
        )
        + x1436 * x4659
        + x1941
        * (
            x1364 * x191
            + x191 * x545
            + x417 * x4701
            + x4190
            + x421 * x4701
            - x4696
            + x4698 * x522
            + x4698 * x536
            + x4699 * x527
            + x4700 * x529
            - x4702 * x559
            + x557
        )
        + x2188 * x4526
        - x3148 * x4469
        + x3189 * x4577
        + x3305 * x4618
        + x3493 * x4695
        + x3528 * x4653
        - x4158 * x4677 * x4678
        - x4479 * x4665
        + x4493 * x4672
        - x4495 * x4674
        - x4497 * x4675
        - x4499 * x4676
        - x4558 * x4673
        + x4623 * x88 * (sigma_pot_5_c * x94 + sigma_pot_5_s * x91)
        + x4656 * x890
        + x4656 * x898
        + x4657 * x906
        + x4658 * x892
        + x4658 * x893
        + x4659 * x896
        + x4661 * x906
        + x4666 * x906
        + x4667 * x906
        + x4668 * x906
        + x823
        * (
            2 * dq_i1 * (x4714 * x773 + x4715 * x776 + x4716)
            + 2 * dq_i2 * (x4714 * x780 + x4718 * x782 + x4719)
            + 2 * dq_i3 * (x4714 * x789 + x4718 * x790 + x4720)
            + 2 * dq_i4 * (x4714 * x796 + x4718 * x797 + x4721)
            + 2 * dq_i5 * dq_i7 * dq_j5 * x140 * x144 * x146 * x148 * x149 * x153 * x1934 * x203 * x434 * x744
            + 2 * dq_i5 * (x4717 * x806 + x4722 * x803 + x4723)
            + dq_i6 * (x114 * x4703 * x811 + x4715 * x812 + x4717 * x814 + x4722 * x810 + x4728)
            + 2 * dq_i7 * sigma_kin_v_7_5 * x140 * x144 * x146 * x148 * x149 * x153 * x195 * x203 * x434 * x744
            - x419 * x4706
            - x419 * x4707
            - x4702 * x4713
            - x4705
            - x4708
            - x4709
            - x4710
            - x4712
            - x537 * x765
        )
        + x906
        * (
            2 * dq_i1 * (x1422 * x4737 + x4673 + x4733 * x863 + x4740 * x772)
            + 2 * dq_i2 * (x2656 * x4737 + x4674 + x4733 * x868 + x4740 * x779)
            + 2 * dq_i3 * (x2658 * x4737 + x4675 + x4733 * x875 + x4740 * x485)
            + 2 * dq_i4 * (x1433 * x4737 + x2664 * x4736 + x4676 + x4740 * x494)
            + 2 * dq_i5 * dq_i7 * dq_j5 * x140 * x144 * x146 * x147 * x150 * x153 * x1934 * x203 * x434 * x828
            - dq_i5 * x4679
            - dq_i5 * x4680
            - dq_i5 * x4681
            - dq_i5 * x4682
            - dq_i5 * x4683
            - dq_i5 * x4684
            + dq_i5
            * (x114 * x4669 * x891 + x1436 * x4737 + x4685 + x4722 * x892 + x4722 * x893 + x4733 * x890 + x4737 * x896)
            + 2 * dq_i6 * (x4665 + x4717 * x858 + x4722 * x857)
            + 2 * dq_i7 * sigma_kin_v_7_5 * x140 * x144 * x146 * x147 * x150 * x153 * x195 * x203 * x434 * x828
            - x4657
            - x4661
            - x4666
            - x4667
            - x4668
            - x4672 * x4702
            - x537 * x852
        )
    )
    K_block_list.append(
        -dq_i6 * x4728 * x906
        + x1022
        * (
            x1444 * x4772
            - x423 * x4784
            - x423 * x4785
            + x4517 * x4786
            + x4572 * x4787
            + x4614 * x4788
            - x4625 * x4747
            + x4652 * x4792
            + x4748 * x4791
            + x4759 * x4790
        )
        + x1142
        * (
            x1474 * x4772
            - x423 * x4774
            - x423 * x4775
            + x4517 * x4776
            + x4572 * x4777
            - x4588 * x4747
            + x4617 * x4782
            + x4648 * x4778
            + x4748 * x4781
            + x4759 * x4780
        )
        + x1266
        * (
            x1842 * x4762
            - x423 * x4763
            - x423 * x4764
            + x4517 * x4765
            - x4541 * x4747
            + x4576 * x4771
            + x4614 * x4766
            + x4648 * x4767
            + x4748 * x4770
            + x4759 * x4768
        )
        - x1334 * x4627
        + x136 * x4523
        + x136
        * (
            -x423 * x4752
            - x423 * x4753
            - x4471 * x4747
            + x4525 * x4761
            + x4572 * x4754
            + x4614 * x4755
            + x4648 * x4756
            + x4748 * x4760
            + x4749 * x4750
            + x4758 * x4759
        )
        - x1797 * x1992 * x4468
        + x2049 * x4741
        - x2052 * x4543
        - x2053 * x4590
        - x2056 * (x1553 * x4744 + x4473 * x4745)
        + x2061
        * (
            x1364 * x4746
            - x423 * x4521
            + x423 * x4613
            + x4509 * x4572
            + x4511 * x4614
            + x4515 * x4648
            + x4517 * x4520
            + x4518 * x4748
            + x4746 * x545
            - x4747 * x559
        )
        - x3104 * x4474
        + x3514 * x4695
        + x3842 * x4466
        + x4493 * x4713
        - x4495 * x4719
        - x4497 * x4720
        - x4499 * x4721
        - x4504 * x4723
        - x4558 * x4716
        + x4705 * x906
        + x4708 * x906
        + x4709 * x906
        + x4710 * x906
        + x4712 * x906
        + x4742 * x803
        + x4743 * x806
        + x823
        * (
            dq_j5 * x4804 * x512
            - x423 * x4795
            - x423 * x4797
            + x4517 * x4798
            + x4572 * x4799
            + x4614 * x4800
            + x4648 * x4801
            - x4711 * x4747
            + x4749 * x4793
            + x4759 * x4803
        )
        + x906
        * (
            dq_i6 * x114 * x4815
            - dq_i6 * x4805
            - dq_i6 * x4806
            - dq_i6 * x4807
            - dq_i6 * x4808
            + x3148 * x4762
            + x4517 * x4810
            + x4572 * x4811
            + x4614 * x4812
            + x4648 * x4813
            - x4660 * x4747
            + x4748 * x4814
        )
    )
    K_block_list.append(
        sigma_kin_v_7_5 * x2159 * x906
        - x191 * x4822
        - x1992 * x3188 * x4479
        + x2150 * x4741
        + x2152 * x4493 * x4671
        + x2155 * x4480
        + x217 * x4686 * x4816
        + x2173 * x4820
        - x2176 * x4558 * x4821
        - x2182 * x4495 * x4821
        + x2209
        * (
            x1156 * x4826
            - x1156 * x4828
            + x1156 * x4830
            + x2190 * x4567
            + x2194 * x4567
            + x2197 * x4829
            + x2199 * x4829
            + x2202 * x4829
            - x2205 * x4827
            + x3198 * x4829
        )
        + x2218
        * (
            x1035 * x4826
            - x1035 * x4828
            + x1035 * x4830
            - x1035 * x4832
            + x2199 * x4831
            + x2202 * x4831
            + x2211 * x4610
            + x2213 * x4610
            + x2214 * x4831
            + x3198 * x4831
        )
        - x2225 * x3514 * x4677 * x4678
        + x2228
        * (
            x2197 * x4833
            + x2202 * x4833
            + x2214 * x4833
            + x2220 * x4646
            + x2223 * x4646
            + x3198 * x4833
            + x4826 * x919
            - x4828 * x919
            + x4830 * x919
            - x4832 * x919
        )
        + x2236
        * (
            x191 * x2230
            + x191 * x2232
            + x2197 * x4836
            + x2199 * x4836
            + x2202 * x4836
            + x2214 * x4836
            - x2231 * x4449
            + x3198 * x4836
            + x4826 * x828
            - x4828 * x828
            - x567 * x895
        )
        + x2243
        * (
            x2197 * x4834
            + x2199 * x4834
            + x2214 * x4834
            + x2238 * x4746
            + x2239 * x4746
            + x3198 * x4834
            + x4826 * x744
            - x4828 * x744
            + x4830 * x744
            - x4832 * x744
        )
        + x2250
        * (
            dq_j5 * x3205 * x558
            + x2197 * x4835
            + x2199 * x4835
            + x2202 * x4835
            + x2214 * x4835
            + x2247 * x4645
            + x3198 * x4835
            - x4645 * x569
            + x4830 * x520
            - x4832 * x520
        )
        + x2257
        * (
            x137 * x4826
            + x2197 * x4824
            + x2199 * x4824
            + x2202 * x4824
            + x2214 * x4824
            + x2251 * x4823
            - x2252 * x4645
            + x2253 * x4823
            + x4446 * x4825
            - x4451 * x4827
        )
        + x250 * x4191 * x4816
        + x2545 * x281 * x4816
        + x275 * x3501 * x4816
        - x3183 * x4819
        - x3184 * x4819
        + x3916 * x4817
        + x3917 * x505
        + x3919 * x4818
        + x3921 * x4817
        - x3926 * x4497 * x4821
        - x3928 * x4499 * x4821
        + x4439 * x4818
        + x4440 * x505
    )
    K_block_list.append(
        -dq_i1 * x4872 * x823
        + dq_i7 * x3216 * x4841
        + x1022
        * (
            x1444 * x4894
            + x2102 * x4885
            + x2103 * x4875
            + x2104 * x4877
            + x2106 * x4882
            + x2107 * x4890
            + x2108 * x4898
            - x4897
            - x515 * x948
            - x515 * x950
        )
        + x1142
        * (
            -x1064 * x515
            - x1065 * x515
            + x1474 * x4894
            + x2090 * x4885
            + x2092 * x4875
            + x2093 * x4879
            + x2095 * x4882
            + x2098 * x4890
            + x2099 * x4896
            - x4895
        )
        + x1266
        * (
            -x1186 * x515
            - x1187 * x515
            + x1842 * x4891
            + x2068 * x4885
            + x2074 * x4877
            + x2077 * x4879
            + x2080 * x4882
            + x2082 * x4890
            + x2083 * x4893
            - x4892
        )
        + x1268 * x4467
        + x1269 * x1313
        - x1328 * x4855
        - x1335 * x4854
        - x1342 * x4853
        - x1359 * x4852
        + x136 * x4845
        + x136
        * (
            dq_j6 * x11 * x2147
            + x2138 * x4889
            + x2142 * x4875
            + x2143 * x4877
            + x2144 * x4879
            + x2145 * x4882
            + x2146 * x4890
            - x228 * x515
            - x237 * x515
            - x4845
        )
        - x2137 * x4843
        - x2319 * (x1541 * x724 + x4850 * x725)
        + x3493 * x4888
        + x4838 * x4839
        + x4840 * x823
        + x4846 * x823
        + x4847 * x823
        + x4848 * x823
        + x4849 * x823
        + x4858 * x4859
        - x4860 * x4861
        - x4862 * x4863
        - x4864 * x4865
        - x4866 * x4867
        - x4870 * x4871
        + x574
        * (
            x1364 * x4874
            - x4844 * x562
            + x4874 * x545
            + x4875 * x4876
            + x4877 * x4878
            + x4879 * x4881
            + x4882 * x4883
            + x4884 * x515
            + x4885 * x4886
            - x4887 * x515
        )
        + x823
        * (
            dq_i1 * x161 * x2134
            - dq_i1 * x817
            - dq_i1 * x819
            - dq_i1 * x820
            - dq_i1 * x821
            + x2124 * x4889
            + x2128 * x4885
            + x2130 * x4875
            + x2131 * x4877
            + x2132 * x4879
            + x2133 * x4882
            - x4901
        )
        + x906
        * (
            x2111 * x4889
            + x2117 * x4885
            + x2118 * x4875
            + x2119 * x4877
            + x2120 * x4879
            + x2122 * x4890
            + x2123 * x4900
            - x4899
            - x515 * x859
            - x515 * x860
        )
    )
    K_block_list.append(
        -dq_i2 * x4924 * x823
        + x1022
        * (
            x1444 * x4936
            - x2683 * x4927
            - x2695 * x405
            - x2697 * x405
            + x3140 * x4875
            + x3141 * x4931
            + x3142 * x4928
            + x3144 * x4930
            + x3145 * x4934
            + x3146 * x4938
        )
        + x1142
        * (
            x1474 * x4936
            - x2732 * x4927
            - x2744 * x405
            - x2745 * x405
            + x3130 * x4875
            + x3131 * x4931
            + x3132 * x4929
            + x3134 * x4930
            + x3136 * x4934
            + x3137 * x4937
        )
        + x1181 * x4467
        + x1184 * x4743
        + x1266
        * (
            dq_j6 * x3178 * x711
            + x1842 * x4935
            - x2443 * x4927
            - x2445 * x405
            - x2447 * x405
            + x3173 * x4875
            + x3174 * x4928
            + x3175 * x4929
            + x3176 * x4930
            + x3177 * x4934
        )
        - x1315 * x4905
        - x1328 * x4912
        - x1335 * x4911
        - x1342 * x4910
        + x136 * x4892
        + x136
        * (
            -x2271 * x4927
            - x2279 * x405
            - x2280 * x405
            + x3109 * x4933
            + x3112 * x4931
            + x3117 * x4928
            + x3120 * x4929
            + x3123 * x4930
            + x3124 * x4934
            + x3125 * x4893
        )
        - x1360 * (x1541 * x2587 + x2588 * x4850)
        + x1385
        * (
            x1364 * x4926
            + x405 * x4884
            - x405 * x4887
            + x4875 * x4886
            + x4876 * x4931
            + x4878 * x4928
            + x4881 * x4929
            + x4883 * x4930
            + x4926 * x545
            - x4927 * x562
        )
        + x3354 * x4904
        + x3493 * x4932
        + x4536 * x4902
        - x4852 * x743
        + x4859 * x4915
        - x4863 * x4918
        - x4865 * x4919
        - x4867 * x4920
        - x4871 * x4923
        + x4903 * x823
        + x4906 * x823
        + x4907 * x823
        + x4908 * x823
        + x4909 * x823
        - x4916 * x4917
        + x823
        * (
            dq_i2 * x161 * x3168
            - dq_i2 * x2630
            - dq_i2 * x2631
            - dq_i2 * x2632
            - dq_i2 * x2633
            + x2522 * x407 * x4940
            - x2610 * x4927
            + x3161 * x4875
            + x3164 * x4931
            + x3165 * x4928
            + x3166 * x4929
            + x3167 * x4930
        )
        + x906
        * (
            -x2640 * x4927
            - x2649 * x405
            - x2650 * x405
            + x3148 * x4935
            + x3152 * x4875
            + x3153 * x4931
            + x3154 * x4928
            + x3155 * x4929
            + x3157 * x4934
            + x3158 * x4939
        )
    )
    K_block_list.append(
        -dq_i3 * x4959 * x823
        + x1022
        * (
            x1444 * x4970
            - x3609 * x4962
            - x3619 * x411
            - x3620 * x411
            + x3876 * x4877
            + x3877 * x4928
            + x3878 * x4965
            + x3880 * x4964
            + x3881 * x4968
            + x3882 * x4971
        )
        + x1045 * x4944
        + x1062 * x4743
        + x1142
        * (
            dq_j6 * x3913 * x668
            + x1474 * x4970
            - x3431 * x4962
            - x3434 * x411
            - x3435 * x411
            + x3907 * x4877
            + x3908 * x4928
            + x3909 * x4963
            + x3911 * x4964
            + x3912 * x4968
        )
        + x1266
        * (
            x1842 * x4969
            - x3358 * x4962
            - x3361 * x411
            - x3363 * x411
            + x3866 * x4877
            + x3867 * x4965
            + x3868 * x4963
            + x3869 * x4964
            + x3871 * x4968
            + x3872 * x4937
        )
        + x136 * x4895
        + x136
        * (
            -x3220 * x4962
            - x3228 * x411
            - x3229 * x411
            + x3847 * x4967
            + x3850 * x4928
            + x3853 * x4965
            + x3857 * x4963
            + x3860 * x4964
            + x3861 * x4968
            + x3862 * x4896
        )
        - x1474 * x4905
        - x1617 * x4950
        - x1620 * x4949
        - x1621 * x4910
        - x1627 * (x1541 * x3531 + x3532 * x4850)
        + x1636
        * (
            x1364 * x4961
            + x411 * x4884
            - x411 * x4887
            + x4876 * x4928
            + x4877 * x4886
            + x4878 * x4965
            + x4881 * x4963
            + x4883 * x4964
            + x4961 * x545
            - x4962 * x562
        )
        - x2787 * x4853
        + x3493 * x4966
        + x4080 * x4943
        + x4583 * x4902
        + x4859 * x4951
        - x4861 * x4953
        - x4865 * x4954
        - x4867 * x4955
        - x4871 * x4958
        - x4917 * x4952
        + x4942 * x823
        + x4945 * x823
        + x4946 * x823
        + x4947 * x823
        + x4948 * x823
        + x823
        * (
            dq_i3 * x161 * x3902
            - dq_i3 * x3565
            - dq_i3 * x3566
            - dq_i3 * x3567
            - dq_i3 * x3568
            + x3430 * x412 * x4940
            - x3547 * x4962
            + x3895 * x4877
            + x3896 * x4928
            + x3899 * x4965
            + x3900 * x4963
            + x3901 * x4964
        )
        + x906
        * (
            x3148 * x4969
            - x3574 * x4962
            - x3582 * x411
            - x3583 * x411
            + x3886 * x4877
            + x3887 * x4928
            + x3888 * x4965
            + x3889 * x4963
            + x3891 * x4968
            + x3892 * x4972
        )
    )
    K_block_list.append(
        -dq_i4 * x4991 * x823
        + x1022
        * (
            dq_j6 * x4435 * x641
            + x1444 * x5001
            - x4127 * x4994
            - x4130 * x415
            - x4131 * x415
            + x4429 * x4879
            + x4430 * x4929
            + x4431 * x4963
            + x4433 * x4995
            + x4434 * x4999
        )
        + x1142
        * (
            x1474 * x5001
            - x4082 * x4994
            - x4086 * x415
            - x4087 * x415
            + x4399 * x4879
            + x4400 * x4929
            + x4401 * x4996
            + x4403 * x4995
            + x4404 * x4999
            + x4405 * x4971
        )
        + x1266
        * (
            x1842 * x5000
            - x4032 * x4994
            - x4035 * x415
            - x4037 * x415
            + x4389 * x4879
            + x4390 * x4963
            + x4391 * x4996
            + x4392 * x4995
            + x4394 * x4999
            + x4395 * x4938
        )
        - x1327 * x4980
        + x136 * x4897
        + x136
        * (
            -x3950 * x4994
            - x3956 * x415
            - x3957 * x415
            + x4372 * x4998
            + x4375 * x4929
            + x4378 * x4963
            + x4379 * x4996
            + x4383 * x4995
            + x4384 * x4999
            + x4385 * x4898
        )
        - x1444 * x4905
        + x1770 * x4974
        - x1781 * x4911
        - x1782 * x4949
        - x1787 * (x1541 * x4205 + x4206 * x4850)
        + x1794
        * (
            x1364 * x4993
            + x415 * x4884
            - x415 * x4887
            + x4876 * x4929
            + x4878 * x4963
            + x4879 * x4886
            + x4881 * x4996
            + x4883 * x4995
            + x4993 * x545
            - x4994 * x562
        )
        - x2913 * x4854
        + x3493 * x4997
        + x3655 * x4943
        + x4743 * x946
        + x4859 * x4983
        - x4861 * x4985
        - x4863 * x4986
        - x4867 * x4987
        - x4871 * x4990
        - x4917 * x4984
        + x4944 * x929
        + x4975 * x823
        + x4976 * x823
        + x4977 * x823
        + x4978 * x823
        + x4979 * x823
        + x823
        * (
            dq_i4 * x161 * x4424
            - dq_i4 * x4232
            - dq_i4 * x4233
            - dq_i4 * x4234
            - dq_i4 * x4235
            - x4215 * x4994
            + x4416 * x4998
            + x4419 * x4879
            + x4420 * x4929
            + x4421 * x4963
            + x4422 * x4996
            + x4423 * x4995
        )
        + x906
        * (
            x3148 * x5000
            - x415 * x4247
            - x415 * x4248
            - x4240 * x4994
            + x4409 * x4879
            + x4410 * x4929
            + x4411 * x4963
            + x4412 * x4996
            + x4414 * x4999
            + x4415 * x5002
        )
    )
    K_block_list.append(
        -dq_i5 * x5016 * x823
        + x1022
        * (
            x1444 * x5024
            - x419 * x4629
            - x419 * x4630
            - x4625 * x5018
            + x4786 * x4882
            + x4787 * x4930
            + x4788 * x4964
            + x4790 * x5019
            + x4791 * x5022
            + x4792 * x5002
        )
        + x1142
        * (
            x1474 * x5024
            - x419 * x4592
            - x419 * x4593
            - x4588 * x5018
            + x4776 * x4882
            + x4777 * x4930
            + x4778 * x4995
            + x4780 * x5019
            + x4781 * x5022
            + x4782 * x4972
        )
        + x1266
        * (
            x1842 * x5023
            - x419 * x4544
            - x419 * x4546
            - x4541 * x5018
            + x4765 * x4882
            + x4766 * x4964
            + x4767 * x4995
            + x4768 * x5019
            + x4770 * x5022
            + x4771 * x4939
        )
        - x1334 * x4980
        + x136 * x4899
        + x136
        * (
            -x419 * x4476
            - x419 * x4477
            - x4471 * x5018
            + x4750 * x5021
            + x4754 * x4930
            + x4755 * x4964
            + x4756 * x4995
            + x4758 * x5019
            + x4760 * x5022
            + x4761 * x4900
        )
        + x1921 * x4974
        - x1930 * x4912
        - x1931 * x4950
        - x1935 * (x1541 * x4703 + x4704 * x4850)
        + x1941
        * (
            x1364 * x5017
            + x419 * x4884
            - x419 * x4887
            + x4876 * x4930
            + x4878 * x4964
            + x4881 * x4995
            + x4882 * x4886
            + x4883 * x5019
            + x5017 * x545
            - x5018 * x562
        )
        - x3016 * x4855
        - x3148 * x4843
        + x3493 * x5020
        + x3756 * x4943
        + x4742 * x857
        + x4743 * x858
        + x4859 * x5008
        - x4861 * x5010
        - x4863 * x5011
        - x4865 * x5012
        - x4871 * x5015
        - x4917 * x5009
        + x5003 * x823
        + x5004 * x823
        + x5005 * x823
        + x5006 * x823
        + x5007 * x823
        + x823
        * (
            dq_i5 * x161 * x4804
            - dq_i5 * x4724
            - dq_i5 * x4725
            - dq_i5 * x4726
            - dq_i5 * x4727
            - x4711 * x5018
            + x4793 * x5021
            + x4798 * x4882
            + x4799 * x4930
            + x4800 * x4964
            + x4801 * x4995
            + x4803 * x5019
        )
        + x906
        * (
            dq_j6 * x4815 * x610
            + x3148 * x5023
            - x419 * x4663
            - x419 * x4664
            - x4660 * x5018
            + x4810 * x4882
            + x4811 * x4930
            + x4812 * x4964
            + x4813 * x4995
            + x4814 * x5022
        )
    )
    K_block_list.append(
        x1022
        * (
            2 * dq_i1 * (x4984 + x5058 * x967 + x5059 * x969)
            + 2 * dq_i2 * (x4985 + x5058 * x974 + x5059 * x975)
            + 2 * dq_i3 * (x4986 + x5058 * x988 + x5059 * x990)
            + dq_i4 * (x1004 * x5058 + x1005 * x161 * x4981 + x1007 * x5059 + x1008 * x161 * x4982 + x4991)
            + 2 * dq_i5 * (x263 * x5066 + x4987 + x5061 * x957)
            + 2 * dq_i6 * dq_i7 * dq_j6 * x140 * x144 * x145 * x148 * x150 * x153 * x203 * x2055 * x434 * x919
            + 2 * dq_i6 * (x223 * x5066 + x4990 + x5064 * x946)
            + 2 * dq_i7 * sigma_kin_v_7_6 * x140 * x144 * x145 * x148 * x150 * x153 * x195 * x203 * x434 * x919
            - x423 * x4988
            - x423 * x4989
            - x4975
            - x4976
            - x4977
            - x4978
            - x4979
            - x4983 * x5055
            - x537 * x931
        )
        + x1142
        * (
            2 * dq_i1 * (x1093 * x5058 + x1095 * x5059 + x4952)
            + 2 * dq_i2 * (x1101 * x5058 + x1104 * x5059 + x4953)
            + dq_i3 * (x1122 * x5058 + x1123 * x5058 + x1125 * x5059 + x1127 * x5059 + x4959)
            + 2 * dq_i4 * (x1079 * x5058 + x1082 * x5059 + x4954)
            + 2 * dq_i5 * (x1070 * x5061 + x263 * x5065 + x4955)
            + 2 * dq_i6 * dq_i7 * dq_j6 * x1035 * x140 * x143 * x146 * x148 * x150 * x153 * x203 * x2055 * x434
            + 2 * dq_i6 * (x1062 * x5064 + x223 * x5065 + x4958)
            + 2 * dq_i7 * sigma_kin_v_7_6 * x1035 * x140 * x143 * x146 * x148 * x150 * x153 * x195 * x203 * x434
            - x1047 * x537
            - x423 * x4956
            - x423 * x4957
            - x4942
            - x4945
            - x4946
            - x4947
            - x4948
            - x4951 * x5055
        )
        + x1266
        * (
            2 * dq_i1 * (x1161 * x5056 + x1234 * x5057 + x4916)
            + dq_i2 * (x1248 * x5056 + x1249 * x161 * x4913 + x1250 * x5057 + x1251 * x161 * x4914 + x4924)
            + 2 * dq_i3 * (x1216 * x5058 + x1219 * x5059 + x4918)
            + 2 * dq_i4 * (x1203 * x5058 + x1205 * x5059 + x4919)
            + 2 * dq_i5 * (x1191 * x5062 + x1192 * x5061 + x4920)
            + 2 * dq_i6 * dq_i7 * dq_j6 * x1156 * x139 * x144 * x146 * x148 * x150 * x153 * x203 * x2055 * x434
            + 2 * dq_i6 * (x1181 * x5062 + x1184 * x5064 + x4923)
            + 2 * dq_i7 * sigma_kin_v_7_6 * x1156 * x139 * x144 * x146 * x148 * x150 * x153 * x195 * x203 * x434
            - x1165 * x537
            - x423 * x4921
            - x423 * x4922
            - x4903
            - x4906
            - x4907
            - x4908
            - x4909
            - x4915 * x5055
        )
        + x1316
        * (
            ddq_i6 * (x2054 * x466 + x2055 * x468)
            + x140 * x144 * x146 * x148 * x149 * x154 * x2055 * x229 * x434 * x512
            + x166 * x168 * x170 * x171 * x176 * x2054 * x221 * x464 * x512
            - x463 * x4745
            - x476 * (x471 * x4856 + x473 * x4857)
            - x482 * (x383 * x4913 + x389 * x4914)
            - x492 * (x4366 * x485 + x4367 * x489)
            - x500 * (x299 * x4981 + x306 * x4982)
            - x507 * (x263 * x4744 + x4745 * x504)
            - x513 * (x221 * x5031 + x5032 * x509)
        )
        + x136 * x4901
        + x136
        * (
            dq_i1 * (x179 * x5056 + x183 * x5056 + x186 * x5057 + x201 * x5057 + x4872)
            + 2 * dq_i2 * (x1293 * x5056 + x1294 * x5057 + x4860)
            + 2 * dq_i3 * (x1286 * x5058 + x1287 * x5059 + x4862)
            + 2 * dq_i4 * (x1279 * x5058 + x1281 * x5059 + x4864)
            + 2 * dq_i5 * (x1273 * x5060 + x1275 * x5061 + x4866)
            + 2 * dq_i6 * dq_i7 * dq_j6 * x137 * x138 * x140 * x144 * x146 * x148 * x150 * x153 * x203 * x2055
            + 2 * dq_i6 * (x1268 * x5062 + x1269 * x5063 + x4870)
            + 2 * dq_i7 * sigma_kin_v_7_6 * x137 * x138 * x140 * x144 * x146 * x148 * x150 * x153 * x195 * x203
            - x216 * x537
            - x423 * x4868
            - x423 * x4869
            - x4840
            - x4846
            - x4847
            - x4848
            - x4849
            - x4858 * x5055
        )
        + x1403 * x197
        - 4 * x161 * x5043
        - x1797 * x2087 * x4842
        + x197 * x812
        + x2048 * x5025 * (sigma_pot_6_c * x101 + sigma_pot_6_s * x98)
        + x2061
        * (
            x1364 * x192
            + x192 * x545
            + x2542
            + x417 * x5054
            + x421 * x5054
            - x5049
            + x5051 * x522
            + x5051 * x536
            + x5052 * x527
            + x5053 * x529
            - x5055 * x562
            + x551
            + x554
            + x557
            + x560
            + x571
        )
        + x2188 * x4888
        + x3189 * x4932
        + x3305 * x4966
        + x3514 * x5020
        + x3528 * x4997
        + x4658 * x810
        + x4658 * x816
        + x4859 * x5033
        - x4861 * x5035
        - x4863 * x5036
        - x4865 * x5037
        - x4867 * x5038
        - x4917 * x5034
        + x5026 * x823
        + x5027 * x823
        + x5028 * x823
        + x5029 * x823
        + x5030 * x823
        - 2 * x5046 * x5047 * x5048
        + x823
        * (
            2 * dq_i1 * (x5034 + x5060 * x773 + x5063 * x776)
            + 2 * dq_i2 * (x5035 + x5060 * x780 + x5069 * x782)
            + 2 * dq_i3 * (x5036 + x5060 * x789 + x5069 * x790)
            + 2 * dq_i4 * (x5037 + x5060 * x796 + x5069 * x797)
            + 2 * dq_i5 * (x5038 + x5064 * x806 + x5068 * x803)
            + 2 * dq_i6 * dq_i7 * dq_j6 * x140 * x144 * x146 * x148 * x149 * x153 * x203 * x2055 * x434 * x744
            - dq_i6 * x5039
            - dq_i6 * x5040
            - dq_i6 * x5041
            - dq_i6 * x5042
            + dq_i6 * (x161 * x5031 * x811 + x5043 + x5063 * x812 + x5064 * x814 + x5068 * x810)
            + 2 * dq_i7 * sigma_kin_v_7_6 * x140 * x144 * x146 * x148 * x149 * x153 * x195 * x203 * x434 * x744
            - x5026
            - x5027
            - x5028
            - x5029
            - x5030
            - x5033 * x5055
            - x537 * x767
        )
        + x906
        * (
            2 * dq_i1 * (x1422 * x5061 + x5009 + x5067 * x772)
            + 2 * dq_i2 * (x2656 * x5061 + x5010 + x5067 * x779)
            + 2 * dq_i3 * (x2658 * x5061 + x485 * x5067 + x5011)
            + 2 * dq_i4 * (x1433 * x5061 + x494 * x5067 + x5012)
            + dq_i5 * (x1436 * x5061 + x5016 + x5061 * x896 + x5068 * x892 + x5068 * x893)
            + 2 * dq_i6 * dq_i7 * dq_j6 * x140 * x144 * x146 * x147 * x150 * x153 * x203 * x2055 * x434 * x828
            + 2 * dq_i6 * (x5015 + x5064 * x858 + x5068 * x857)
            + 2 * dq_i7 * sigma_kin_v_7_6 * x140 * x144 * x146 * x147 * x150 * x153 * x195 * x203 * x434 * x828
            - x423 * x5013
            - x423 * x5014
            - x5003
            - x5004
            - x5005
            - x5006
            - x5007
            - x5008 * x5055
            - x537 * x842
        )
    )
    K_block_list.append(
        dq_i7 * x3916 * x510
        + sigma_kin_v_7_6 * x2159 * x823
        + x1313 * x3188
        + x1769 * x2047 * x2148 * x4973
        - x192 * x5074
        - x2114 * x2185 * x4820 * x4867
        + x2152 * x4745 * x4859
        + x2156 * x4844
        - x2176 * x4917 * x5073
        - x2182 * x4861 * x5073
        + x2209
        * (
            x1156 * x5077
            + x1156 * x5078
            - x1156 * x5080
            + x2190 * x4926
            + x2194 * x4926
            + x2197 * x5081
            + x2199 * x5081
            + x2200 * x5081
            - x2205 * x5079
            + x3198 * x5081
        )
        + x2218
        * (
            x1035 * x5077
            + x1035 * x5078
            - x1035 * x5080
            - x1035 * x5083
            + x2199 * x5082
            + x2200 * x5082
            + x2211 * x4961
            + x2213 * x4961
            + x2214 * x5082
            + x3198 * x5082
        )
        + x2228
        * (
            x2197 * x5084
            + x2200 * x5084
            + x2214 * x5084
            + x2220 * x4993
            + x2223 * x4993
            + x3198 * x5084
            + x5077 * x919
            + x5078 * x919
            - x5080 * x919
            - x5083 * x919
        )
        + x2236
        * (
            x2197 * x5085
            + x2199 * x5085
            + x2214 * x5085
            + x2230 * x5017
            + x2232 * x5017
            + x3198 * x5085
            + x5077 * x828
            + x5078 * x828
            - x5080 * x828
            - x5083 * x828
        )
        + x2243
        * (
            x192 * x2238
            + x192 * x2239
            + x2197 * x5087
            + x2199 * x5087
            + x2200 * x5087
            - x2201 * x5088
            + x2214 * x5087
            + x3198 * x5087
            + x5078 * x744
            - x5080 * x744
            - x567 * x813
        )
        + x2250
        * (
            dq_j6 * x3205 * x561
            + x2197 * x5086
            + x2199 * x5086
            + x2200 * x5086
            + x2214 * x5086
            + x2247 * x4992
            + x3198 * x5086
            - x4992 * x569
            + x5077 * x520
            - x5083 * x520
        )
        + x2257
        * (
            x137 * x5077
            + x137 * x5078
            + x2197 * x5076
            + x2199 * x5076
            + x2200 * x5076
            + x2214 * x5076
            + x2251 * x5075
            - x2252 * x4992
            + x2253 * x5075
            - x4451 * x5079
        )
        + x244 * x4686 * x5070
        + x250 * x4169 * x5070
        + x2523 * x281 * x5070
        + x275 * x3480 * x5070
        - x3183 * x5072
        - x3184 * x5072
        - x3493 * x5046 * x5048 * x532
        + x3917 * x510
        + x3919 * x5071
        + x3920 * x5071
        - x3926 * x4863 * x5073
        - x3928 * x4865 * x5073
        + x4439 * x5071
        + x4440 * x510
    )
    K_block_list.append(
        sigma_kin_v_7_1
        * x2242
        * (
            dq_i1 * x5176
            - x5142 * x744
            + x5153 * x744
            - x5156 * x744
            + x5158 * x5173
            + x5162 * x744
            + x5165 * x744
            + x5168 * x744
            + x5169 * x744
            + x5172 * x744
        )
        - x1269 * x5127
        - x1275 * x5125
        - x1281 * x5122
        - x1287 * x5119
        + x1317 * x5107
        + x1328 * x5106
        + x1335 * x5105
        + x1342 * x5104
        + x1359 * x5102
        + x1374
        * x2227
        * (
            x2219 * x5166
            - x5142 * x919
            + x5150 * x919
            + x5153 * x919
            - x5156 * x919
            + x5162 * x919
            + x5165 * x919
            + x5166 * x5167
            + x5168 * x919
            + x5169 * x919
        )
        + x1374
        * x2235
        * (
            x2229 * x5170
            - x5142 * x828
            + x5150 * x828
            + x5153 * x828
            - x5156 * x828
            + x5162 * x828
            + x5165 * x828
            + x5168 * x828
            + x5170 * x5171
            + x5172 * x828
        )
        - x139 * x155 * x2368 * x389 * x5115 * x5116
        + x142 * x5110
        + x157 * x5093
        - x186 * x195 * x5112
        - x1908 * x5112
        + x202 * x5114
        - x206 * x5128
        + x2167 * x2319 * x461
        + x2208
        * x539
        * (
            -x1156 * x5142
            + x1156 * x5150
            + x1156 * x5153
            - x1156 * x5156
            + x2189 * x5140
            + x5140 * x5143
            + x5144 * x5145
            + x5144 * x5154
            + x5146 * x5147
            + x5147 * x5148
        )
        + x2217
        * x539
        * (
            -x1035 * x5142
            + x1035 * x5150
            + x1035 * x5153
            - x1035 * x5156
            + x1035 * x5162
            + x1035 * x5165
            + x2210 * x5159
            + x5146 * x5164
            + x5148 * x5164
            + x5159 * x5160
        )
        + x2256
        * x539
        * (
            x137 * x5153
            - x137 * x5156
            + x2251 * x5178
            + x2253 * x5178
            + x5115 * x5149
            - x5141 * x5177
            + x5145 * x5177
            + x5146 * x5180
            + x5148 * x5180
            + x5161 * x5179
        )
        + x2262 * x5095
        + x3211 * x5096
        - x3215 * x5136 * x5138
        + x3943 * x5097
        + x4839 * x5091
        + x5098 * x5099
        + x5098 * x5100
        + x574
        * (
            dq_i1 * x5183
            - dq_i1 * x5188
            - x5141 * x5182
            + x5145 * x5182
            + x5146 * x5186
            + x5148 * x5186
            + x5150 * x520
            + x5154 * x5182
            + x5157 * x5181
            - x5157 * x5187
            + x5185 * x671
        )
    )
    K_block_list.append(
        sigma_kin_v_7_2
        * x2242
        * (
            dq_i2 * x5176
            + x5173 * x5213
            + x5208 * x744
            - x5211 * x744
            - x5215 * x744
            + x5216 * x744
            + x5218 * x744
            + x5220 * x744
            + x5221 * x744
            + x5223 * x744
        )
        - sigma_kin_v_7_2 * x5201 * x5202
        + x1157 * x1308 * x158 * x281
        + x1157 * x156 * x5093
        + x1157 * x5110
        - x1184 * x5199
        - x1192 * x5125
        + x1195 * x5114
        - x1205 * x5122
        - x1219 * x5119
        - x1250 * x5197
        - x1263 * x5197
        - x1315 * x5128
        + x1317 * x5195
        + x1328 * x5194
        + x1335 * x5193
        + x1342 * x5192
        + x1360 * x2585 * x5101
        + x1385
        * (
            dq_i2 * x5183
            - dq_i2 * x5188
            - dq_i2 * x5226
            + x5181 * x5212
            - x5182 * x5203
            + x5182 * x5204
            + x5182 * x5225
            + x5184 * x5209
            + x5186 * x5205
            + x5186 * x5206
            + x520 * x5217
        )
        - x1503 * x5111 * x5198
        + x2208
        * x549
        * (
            x1156 * x5208
            - x1156 * x5211
            - x1156 * x5215
            + x1156 * x5217
            + x2189 * x5224
            + x5143 * x5224
            + x5144 * x5204
            + x5144 * x5225
            + x5147 * x5205
            + x5147 * x5206
        )
        + x2217
        * x549
        * (
            x1035 * x5208
            - x1035 * x5211
            - x1035 * x5215
            + x1035 * x5216
            + x1035 * x5217
            + x1035 * x5218
            + x2210 * x5214
            + x5160 * x5214
            + x5164 * x5205
            + x5164 * x5206
        )
        + x2227
        * x2359
        * (
            x2219 * x5219
            + x5167 * x5219
            + x5208 * x919
            - x5211 * x919
            - x5215 * x919
            + x5216 * x919
            + x5217 * x919
            + x5218 * x919
            + x5220 * x919
            + x5221 * x919
        )
        + x2235
        * x2359
        * (
            x2229 * x5222
            + x5171 * x5222
            + x5208 * x828
            - x5211 * x828
            - x5215 * x828
            + x5216 * x828
            + x5217 * x828
            + x5218 * x828
            + x5220 * x828
            + x5223 * x828
        )
        + x2256
        * x549
        * (
            x137 * x5208
            - x137 * x5211
            + x2251 * x5140
            + x2253 * x5140
            + x3191 * x5210
            + x5115 * x5207
            - x5177 * x5203
            + x5177 * x5204
            + x5180 * x5205
            + x5180 * x5206
        )
        + x3351 * x5096
        + x4027 * x5097
        + x4536 * x5189
        + x5099 * x5190
        + x5100 * x5190
        + x5102 * x743
    )
    K_block_list.append(
        sigma_kin_v_7_3
        * x2242
        * (
            dq_i3 * x5176
            + x5173 * x5262
            + x5251 * x744
            - x5253 * x744
            - x5256 * x744
            + x5264 * x744
            + x5265 * x744
            + x5266 * x744
            + x5267 * x744
            + x5269 * x744
        )
        - sigma_kin_v_7_3 * x5202 * x5243
        + x1036 * x5238
        + x1037 * x5227
        + x1037 * x5228
        + x1037 * x5229
        + x1037 * x5236
        - x1062 * x5199
        - x1070 * x5125
        - x1082 * x5122
        - x1095 * x5240
        - x1104 * x5241
        - x1140 * x5237
        - x1474 * x5128
        + x1614 * x5235
        + x1617 * x5234
        + x1620 * x5233
        + x1621 * x5192
        + x1627 * x3529 * x5101
        + x1636
        * (
            dq_i3 * x5183
            - dq_i3 * x5188
            - dq_i3 * x5226
            + x5181 * x5261
            - x5182 * x5246
            + x5182 * x5252
            + x5182 * x5259
            + x5185 * x666
            + x5186 * x5248
            + x5186 * x5249
            + x520 * x5260
        )
        + x2208
        * x552
        * (
            x1156 * x5251
            - x1156 * x5253
            - x1156 * x5256
            + x1156 * x5260
            + x5117 * x5255
            + x5117 * x5258
            + x5144 * x5252
            + x5144 * x5259
            + x5147 * x5248
            + x5147 * x5249
        )
        + x2217
        * x552
        * (
            x1035 * x5251
            - x1035 * x5253
            - x1035 * x5256
            + x1035 * x5260
            + x1035 * x5264
            + x1035 * x5265
            + x2210 * x5270
            + x5160 * x5270
            + x5164 * x5248
            + x5164 * x5249
        )
        + x2227
        * x3288
        * (
            x2219 * x5263
            + x5167 * x5263
            + x5251 * x919
            - x5253 * x919
            - x5256 * x919
            + x5260 * x919
            + x5264 * x919
            + x5265 * x919
            + x5266 * x919
            + x5267 * x919
        )
        + x2235
        * x3288
        * (
            x2229 * x5268
            + x5171 * x5268
            + x5251 * x828
            - x5253 * x828
            - x5256 * x828
            + x5260 * x828
            + x5264 * x828
            + x5265 * x828
            + x5267 * x828
            + x5269 * x828
        )
        + x2256
        * x552
        * (
            x137 * x5251
            - x137 * x5253
            + x2253 * x5159
            + x5115 * x5250
            + x5117 * x5245
            - x5177 * x5246
            + x5177 * x5252
            + x5179 * x5247
            + x5180 * x5248
            + x5180 * x5249
        )
        + x2787 * x5104
        - x3427 * x5237
        + x4077 * x5097
        + x4583 * x5189
        + x5230 * x5231
        + x5230 * x5232
    )
    K_block_list.append(
        sigma_kin_v_7_4
        * x2242
        * (
            x5286 * x744
            - x5288 * x744
            - x5289 * x744
            + x5297 * x744
            + x5298 * x744
            + x5299 * x744
            + x5301 * x744
            + x5303 * x5304
            + x5304 * x5305
            + x5306 * x744
        )
        - sigma_kin_v_7_4 * x5202 * x5280
        - x1007 * x5278
        - x1018 * x5278
        - x1444 * x5128
        + x1770 * x5271
        + x1781 * x5193
        + x1782 * x5233
        + x1787 * x4203 * x5101
        + x1794
        * (
            dq_i4 * x5183
            - dq_i4 * x5188
            - dq_i4 * x5226
            + x5181 * x5308
            - x5182 * x5281
            + x5182 * x5283
            + x5182 * x5290
            + x5182 * x5292
            + x5185 * x635
            + x5186 * x5284
            + x520 * x5291
        )
        + x2208
        * x555
        * (
            x1156 * x5286
            - x1156 * x5288
            - x1156 * x5289
            + x1156 * x5291
            + x5120 * x5255
            + x5120 * x5258
            + x5144 * x5283
            + x5144 * x5290
            + x5144 * x5292
            + x5147 * x5284
        )
        + x2217
        * x555
        * (
            x1035 * x5286
            - x1035 * x5288
            - x1035 * x5289
            + x1035 * x5291
            + x1035 * x5297
            + x1035 * x5298
            + x1035 * x5299
            + x5120 * x5294
            + x5120 * x5296
            + x5164 * x5284
        )
        + x2227
        * x4003
        * (
            x2219 * x5307
            + x5167 * x5307
            + x5286 * x919
            - x5288 * x919
            - x5289 * x919
            + x5291 * x919
            + x5297 * x919
            + x5298 * x919
            + x5301 * x919
            + x5306 * x919
        )
        + x2235
        * x4003
        * (
            x2229 * x5300
            + x5171 * x5300
            + x5286 * x828
            - x5288 * x828
            - x5289 * x828
            + x5291 * x828
            + x5297 * x828
            + x5298 * x828
            + x5299 * x828
            + x5301 * x828
        )
        + x2256
        * x555
        * (
            x137 * x5286
            - x137 * x5288
            + x2253 * x5166
            + x5115 * x5285
            + x5120 * x5245
            - x5177 * x5281
            + x5177 * x5283
            + x5179 * x5282
            + x5180 * x5284
            + x5180 * x5287
        )
        + x2913 * x5105
        - x5119 * x990
        - x5125 * x957
        - x5199 * x946
        + x5227 * x921
        + x5228 * x921
        + x5229 * x921
        + x5231 * x5273
        + x5232 * x5273
        + x5236 * x921
        + x5238 * x920
        - x5240 * x969
        - x5241 * x975
        + x5272 * x921
        + x5274 * x5275
        + x5276 * x5277
    )
    K_block_list.append(
        -dq_i1 * x1422 * x5314
        - dq_i2 * x2656 * x5314
        + sigma_kin_v_7_5
        * x2242
        * (
            x5303 * x5333
            + x5305 * x5333
            + x5322 * x744
            - x5324 * x744
            - x5325 * x744
            + x5329 * x744
            + x5330 * x744
            + x5331 * x744
            + x5332 * x744
            + x5334 * x744
        )
        - sigma_kin_v_7_5 * x5202 * x5316
        - x1433 * x5120 * x5124
        - x1436 * x5313
        + x1921 * x5271
        + x1930 * x5194
        + x1931 * x5234
        + x1935 * x4671 * x5101
        + x1941
        * (
            dq_i5 * x5183
            - dq_i5 * x5188
            - dq_i5 * x5226
            + x5181 * x5336
            - x5182 * x5317
            + x5182 * x5319
            + x5182 * x5326
            + x5182 * x5328
            + x5185 * x594
            + x5186 * x5320
            + x520 * x5327
        )
        - x203 * x3147 * x5128
        + x2208
        * x558
        * (
            x1156 * x5322
            - x1156 * x5324
            - x1156 * x5325
            + x1156 * x5327
            + x5123 * x5255
            + x5123 * x5258
            + x5144 * x5319
            + x5144 * x5326
            + x5144 * x5328
            + x5147 * x5320
        )
        + x2217
        * x558
        * (
            x1035 * x5322
            - x1035 * x5324
            - x1035 * x5325
            + x1035 * x5327
            + x1035 * x5329
            + x1035 * x5330
            + x1035 * x5331
            + x5123 * x5294
            + x5123 * x5296
            + x5164 * x5320
        )
        + x2227
        * x4513
        * (
            x2219 * x5300
            + x5167 * x5300
            + x5322 * x919
            - x5324 * x919
            - x5325 * x919
            + x5327 * x919
            + x5329 * x919
            + x5330 * x919
            + x5331 * x919
            + x5332 * x919
        )
        + x2235
        * x4513
        * (
            x2229 * x5335
            + x5171 * x5335
            + x5322 * x828
            - x5324 * x828
            - x5325 * x828
            + x5327 * x828
            + x5329 * x828
            + x5330 * x828
            + x5332 * x828
            + x5334 * x828
        )
        + x2256
        * x558
        * (
            x137 * x5322
            - x137 * x5324
            + x2253 * x5170
            + x5115 * x5321
            + x5123 * x5245
            - x5177 * x5317
            + x5177 * x5319
            + x5179 * x5318
            + x5180 * x5320
            + x5180 * x5323
        )
        - x2658 * x5117 * x5124
        + x3016 * x5106
        + x3147 * x5113
        - x5199 * x858
        + x5227 * x831
        + x5228 * x831
        + x5229 * x831
        + x5231 * x5309
        + x5232 * x5309
        + x5236 * x831
        + x5272 * x831
        + x5275 * x5310
        + x5276 * x5311
        - x5313 * x896
    )
    K_block_list.append(
        sigma_kin_v_7_6
        * x2242
        * (
            x5303 * x5365
            + x5305 * x5365
            - x5353 * x744
            - x5354 * x744
            + x5356 * x744
            + x5358 * x744
            + x5359 * x744
            + x5361 * x744
            + x5362 * x744
            + x5364 * x744
        )
        + x1313 * x461 * x5339
        - x1403 * x5338
        - x158 * x5116 * x5340 * x783
        - x158 * x5198 * x777
        - 8 * x193 * x461 * x5343
        + x2047 * x2435 * x4365 * x5089
        + x2052 * x5195
        + x2053 * x5235
        + x2056 * x4745 * x5101
        + x2061
        * (
            dq_i6 * x400 * x5181
            + dq_i6 * x5183
            - dq_i6 * x5188
            - dq_i6 * x5226
            + x482 * x5185
            - x5182 * x5346
            + x5182 * x5348
            + x5182 * x5355
            + x5186 * x5349
            + x5186 * x5350
            + x520 * x5357
        )
        + x2208
        * x561
        * (
            -x1156 * x5353
            - x1156 * x5354
            + x1156 * x5356
            + x1156 * x5357
            + x5126 * x5255
            + x5126 * x5258
            + x5144 * x5348
            + x5144 * x5355
            + x5147 * x5349
            + x5147 * x5350
        )
        + x2217
        * x561
        * (
            -x1035 * x5353
            - x1035 * x5354
            + x1035 * x5356
            + x1035 * x5357
            + x1035 * x5358
            + x1035 * x5359
            + x5126 * x5294
            + x5126 * x5296
            + x5164 * x5349
            + x5164 * x5350
        )
        + x2227
        * x4880
        * (
            x2219 * x5360
            + x5167 * x5360
            - x5353 * x919
            - x5354 * x919
            + x5356 * x919
            + x5357 * x919
            + x5358 * x919
            + x5359 * x919
            + x5361 * x919
            + x5362 * x919
        )
        + x2235
        * x4880
        * (
            x2229 * x5363
            + x5171 * x5363
            - x5353 * x828
            - x5354 * x828
            + x5356 * x828
            + x5357 * x828
            + x5358 * x828
            + x5359 * x828
            + x5361 * x828
            + x5364 * x828
        )
        + x2256
        * x561
        * (
            dq_i6 * x2253 * x5158
            - x137 * x5353
            + x5115 * x5352
            + x5126 * x5245
            + x5152 * x5351
            - x5177 * x5346
            + x5177 * x5348
            + x5179 * x5347
            + x5180 * x5349
            + x5180 * x5350
        )
        + x3104 * x5107
        - x3841 * x5138 * x5345
        - x504 * x5123 * x5341
        - x5117 * x5342 * x790
        - x5120 * x5342 * x797
        + x5227 * x747
        + x5228 * x747
        + x5229 * x747
        + x5231 * x5337
        + x5232 * x5337
        + x5236 * x747
        + x5272 * x747
        + x5274 * x5311
        + x5277 * x5310
        - x5338 * x812
    )
    K_block_list.append(
        -dq_i1 * x158 * x2180
        - dq_i2 * x158 * x3186
        + sigma_kin_v_7_1 * x150 * x2153 * x5092
        - sigma_kin_v_7_2 * x3189 * x5201 * x5371
        - sigma_kin_v_7_3 * x3305 * x5243 * x5371
        - sigma_kin_v_7_4 * x3528 * x5280 * x5371
        - sigma_kin_v_7_5 * x3514 * x5316 * x5371
        + x102 * x2047 * x5025 * (sigma_pot_7_c * x108 + sigma_pot_7_s * x105)
        + x2158 * x2168 * x5108
        - x2168 * x400 * x5047 * (x2197 + x2199 + x2200 + x2202 + x2214 + x2246 + x3198 - x568)
        - x2188 * x3215 * x5136 * x5137
        + x2209
        * (
            -x1156 * x5376
            + x1156 * x5395
            + x1156 * x5396
            + x1156 * x5400
            - x158 * x2205 * x568
            + x193 * x5255
            + x193 * x5258
            + x2368 * x5257
            + x417 * x5394
            + x417 * x5399
            + x421 * x5394
            + x421 * x5399
            + x5254 * x566
            + x5373 * x5392
            - x5378 * x5391
            - x5379 * x5391
            - x5380 * x5391
            - x5381 * x5391
            - x5382 * x5391
            + x5392 * x5393
            + x5392 * x5397
            + x5392 * x5398
        )
        + x2218
        * (
            -x1035 * x5376
            + x1035 * x5395
            + x1035 * x5396
            + x1035 * x5400
            - x1035 * x5402
            + x193 * x5294
            + x193 * x5296
            + x2368 * x5295
            + x417 * x5405
            + x417 * x5407
            + x421 * x5405
            + x421 * x5407
            + x5293 * x566
            - x5378 * x5401
            - x5379 * x5401
            - x5380 * x5401
            - x5381 * x5401
            - x5382 * x5401
            + x5393 * x5403
            + x5397 * x5403
            + x5403 * x5404
            + x5403 * x5406
        )
        + x2228
        * (
            x4010 * x5411
            + x4010 * x5413
            + x5373 * x5412
            + x5374 * x5411
            + x5374 * x5413
            - x5376 * x919
            - x5378 * x5410
            - x5379 * x5410
            - x5380 * x5410
            - x5381 * x5410
            - x5382 * x5410
            + x5393 * x5412
            + x5395 * x919
            + x5396 * x919
            + x5397 * x5412
            + x5398 * x5412
            + x5400 * x919
            - x5402 * x919
            + x5404 * x5412
            + x5406 * x5412
            + x5408 * x5409
            + x5408 * x566
        )
        + x2236
        * (
            x4010 * x5416
            + x4010 * x5418
            + x5373 * x5417
            + x5374 * x5416
            + x5374 * x5418
            - x5376 * x828
            - x5378 * x5415
            - x5379 * x5415
            - x5380 * x5415
            - x5381 * x5415
            - x5382 * x5415
            + x5393 * x5417
            + x5395 * x828
            + x5396 * x828
            + x5397 * x5417
            + x5398 * x5417
            + x5400 * x828
            - x5402 * x828
            + x5404 * x5417
            + x5406 * x5417
            + x5409 * x5414
            + x5414 * x566
        )
        + x2243
        * (
            x193 * x5174 * x744
            + x2203 * x5175
            + x417 * x5421
            + x417 * x5423
            + x421 * x5421
            + x421 * x5423
            + x5088 * x5174
            + x517 * x5343
            - 2 * x5175 * x568
            + x526 * x5422
            + x529 * x5422
            + x5339 * x5373
            + x5339 * x5393
            + x5339 * x5404
            - x5376 * x744
            - x5378 * x5420
            - x5379 * x5420
            - x5380 * x5420
            - x5381 * x5420
            - x5382 * x5420
            + x5409 * x5419
            + x5419 * x566
        )
        + x2250
        * (
            x158 * x2247
            + x193 * x522
            + x193 * x536
            + x2534
            + x2542
            + x3205 * x5139
            + x3494
            + x417 * x5375
            + x4182
            + x421 * x5375
            + x4696
            + x5049
            + x521 * x5373
            + x523
            + x527 * x5372
            + x564
            - 2 * x570
        )
        + x2257
        * (
            -x137 * x5376
            + x193 * x2253 * x5157
            + x193 * x5245
            + x2203 * x5115
            - x2252 * x5383
            + x2255 * x5157
            + x417 * x5386
            + x417 * x5389
            + x421 * x5386
            + x421 * x5389
            + x5244 * x566
            + x526 * x5385
            + x526 * x5388
            + x529 * x5385
            + x529 * x5388
            + x5351 * x5387
            + x5351 * x5390
            - x5377 * x5378
            - x5377 * x5379
            - x5377 * x5380
            - x5377 * x5381
            - x5377 * x5382
        )
        - x3493 * x3841 * x5137 * x5345
        - x3927 * x5117
        - x4443 * x5120
        - x4822 * x5123
        - x5074 * x5126
        + x5228 * x5366
        + x5229 * x5366
        + x5231 * x5367
        + x5232 * x5367
        + x5272 * x5366
        + x5368 * x5369
        + x5368 * x543
        - x5369 * x5370
        - x5370 * x543
    )

    return K_block_list
