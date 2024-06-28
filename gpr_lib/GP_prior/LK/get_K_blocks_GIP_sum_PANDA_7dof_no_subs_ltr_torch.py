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
    x65 = sigma_kin_p_4_3_c * x45 + sigma_kin_p_4_3_off + sigma_kin_p_4_3_s * x48
    x66 = x65**2
    x67 = torch.cos(q_i4)
    x68 = torch.cos(q_j4)
    x69 = x67 * x68
    x70 = torch.sin(q_i4)
    x71 = torch.sin(q_j4)
    x72 = x70 * x71
    x73 = sigma_kin_p_4_4_c * x69 + sigma_kin_p_4_4_off + sigma_kin_p_4_4_s * x72
    x74 = x73**2
    x75 = x66 * x74
    x76 = sigma_kin_p_4_1_c * x2 + sigma_kin_p_4_1_s * x5
    x77 = sigma_kin_p_4_2_c * x22 + sigma_kin_p_4_2_off + sigma_kin_p_4_2_s * x25
    x78 = x77**2
    x79 = sigma_kin_p_4_1_c * x5 + sigma_kin_p_4_1_off + sigma_kin_p_4_1_s * x2
    x80 = x78 * x79
    x81 = x76 * x80
    x82 = x75 * x81
    x83 = sigma_kin_p_4_1_c * x14 - sigma_kin_p_4_1_s * x15
    x84 = sigma_kin_p_4_1_c * x15 - sigma_kin_p_4_1_s * x14
    x85 = x78 * x83 * x84
    x86 = x75 * x85
    x87 = sigma_pot_4_c * x69 + sigma_pot_4_off + sigma_pot_4_s * x72
    x88 = torch.cos(q_i5)
    x89 = torch.cos(q_j5)
    x90 = x88 * x89
    x91 = torch.sin(q_i5)
    x92 = torch.sin(q_j5)
    x93 = x91 * x92
    x94 = sigma_pot_5_c * x90 + sigma_pot_5_off + sigma_pot_5_s * x93
    x95 = torch.cos(q_i6)
    x96 = torch.cos(q_j6)
    x97 = x95 * x96
    x98 = torch.sin(q_i6)
    x99 = torch.sin(q_j6)
    x100 = x98 * x99
    x101 = sigma_pot_6_c * x97 + sigma_pot_6_off + sigma_pot_6_s * x100
    x102 = torch.cos(q_i7)
    x103 = torch.cos(q_j7)
    x104 = x102 * x103
    x105 = torch.sin(q_i7)
    x106 = torch.sin(q_j7)
    x107 = x105 * x106
    x108 = sigma_pot_7_c * x104 + sigma_pot_7_off + sigma_pot_7_s * x107
    x109 = 1.0 * x101 * x108 * x87 * x94
    x110 = sigma_pot_2_c * x22 + sigma_pot_2_off + sigma_pot_2_s * x25
    x111 = sigma_pot_3_c * x45 + sigma_pot_3_off + sigma_pot_3_s * x48
    x112 = x110 * x111
    x113 = dq_i5 * dq_j5
    x114 = (
        sigma_kin_v_5_1 * x31
        + sigma_kin_v_5_2 * x32
        + sigma_kin_v_5_3 * x54
        + sigma_kin_v_5_4 * x61
        + sigma_kin_v_5_5 * x113
    )
    x115 = x114**2
    x116 = 2 * x115
    x117 = sigma_kin_p_5_3_c * x45 + sigma_kin_p_5_3_off + sigma_kin_p_5_3_s * x48
    x118 = x117**2
    x119 = sigma_kin_p_5_4_c * x69 + sigma_kin_p_5_4_off + sigma_kin_p_5_4_s * x72
    x120 = x119**2
    x121 = sigma_kin_p_5_5_c * x90 + sigma_kin_p_5_5_off + sigma_kin_p_5_5_s * x93
    x122 = x121**2
    x123 = x118 * x120 * x122
    x124 = sigma_kin_p_5_1_c * x2 + sigma_kin_p_5_1_s * x5
    x125 = sigma_kin_p_5_2_c * x22 + sigma_kin_p_5_2_off + sigma_kin_p_5_2_s * x25
    x126 = x125**2
    x127 = sigma_kin_p_5_1_c * x5 + sigma_kin_p_5_1_off + sigma_kin_p_5_1_s * x2
    x128 = x126 * x127
    x129 = x124 * x128
    x130 = x123 * x129
    x131 = sigma_kin_p_5_1_c * x14 - sigma_kin_p_5_1_s * x15
    x132 = sigma_kin_p_5_1_c * x15 - sigma_kin_p_5_1_s * x14
    x133 = x126 * x131 * x132
    x134 = x123 * x133
    x135 = 4 * dq_j1
    x136 = sigma_kin_p_7_1_c * x14 - sigma_kin_p_7_1_s * x15
    x137 = sigma_kin_p_7_1_c * x5 + sigma_kin_p_7_1_off + sigma_kin_p_7_1_s * x2
    x138 = sigma_kin_p_7_2_c * x22 + sigma_kin_p_7_2_off + sigma_kin_p_7_2_s * x25
    x139 = x138**2
    x140 = x137 * x139
    x141 = x136 * x140
    x142 = sigma_kin_p_7_3_c * x45 + sigma_kin_p_7_3_off + sigma_kin_p_7_3_s * x48
    x143 = x142**2
    x144 = sigma_kin_p_7_4_c * x69 + sigma_kin_p_7_4_off + sigma_kin_p_7_4_s * x72
    x145 = x144**2
    x146 = sigma_kin_p_7_5_c * x90 + sigma_kin_p_7_5_off + sigma_kin_p_7_5_s * x93
    x147 = x146**2
    x148 = sigma_kin_p_7_6_c * x97 + sigma_kin_p_7_6_off + sigma_kin_p_7_6_s * x100
    x149 = x148**2
    x150 = sigma_kin_p_7_7_c * x103
    x151 = sigma_kin_p_7_7_s * x106
    x152 = sigma_kin_p_7_7_off + x102 * x150 + x105 * x151
    x153 = x152**2
    x154 = x143 * x145 * x147 * x149 * x153
    x155 = sigma_kin_v_7_1 * x154
    x156 = x141 * x155
    x157 = dq_j7 * sigma_kin_v_7_7
    x158 = ddq_i7 * x157
    x159 = x156 * x158
    x160 = dq_i6 * dq_j6
    x161 = (
        sigma_kin_v_6_1 * x31
        + sigma_kin_v_6_2 * x32
        + sigma_kin_v_6_3 * x54
        + sigma_kin_v_6_4 * x61
        + sigma_kin_v_6_5 * x113
        + sigma_kin_v_6_6 * x160
    )
    x162 = x161**2
    x163 = 2 * x162
    x164 = sigma_kin_p_6_3_c * x45 + sigma_kin_p_6_3_off + sigma_kin_p_6_3_s * x48
    x165 = x164**2
    x166 = sigma_kin_p_6_4_c * x69 + sigma_kin_p_6_4_off + sigma_kin_p_6_4_s * x72
    x167 = x166**2
    x168 = sigma_kin_p_6_5_c * x90 + sigma_kin_p_6_5_off + sigma_kin_p_6_5_s * x93
    x169 = x168**2
    x170 = sigma_kin_p_6_6_c * x97 + sigma_kin_p_6_6_off + sigma_kin_p_6_6_s * x100
    x171 = x170**2
    x172 = x165 * x167 * x169 * x171
    x173 = sigma_kin_p_6_1_c * x2 + sigma_kin_p_6_1_s * x5
    x174 = sigma_kin_p_6_2_c * x22 + sigma_kin_p_6_2_off + sigma_kin_p_6_2_s * x25
    x175 = x174**2
    x176 = sigma_kin_p_6_1_c * x5 + sigma_kin_p_6_1_off + sigma_kin_p_6_1_s * x2
    x177 = x175 * x176
    x178 = x173 * x177
    x179 = x172 * x178
    x180 = sigma_kin_p_6_1_c * x14 - sigma_kin_p_6_1_s * x15
    x181 = sigma_kin_p_6_1_c * x15 - sigma_kin_p_6_1_s * x14
    x182 = x175 * x180 * x181
    x183 = x172 * x182
    x184 = sigma_kin_v_7_1 * x31
    x185 = sigma_kin_v_7_2 * x32
    x186 = sigma_kin_v_7_3 * x54
    x187 = sigma_kin_v_7_4 * x61
    x188 = sigma_kin_v_7_5 * x113
    x189 = sigma_kin_v_7_6 * x160
    x190 = dq_i7 * x157
    x191 = x185 + x186 + x187 + x188 + x189 + x190
    x192 = x184 + x191
    x193 = x192**2
    x194 = 2 * x193
    x195 = sigma_kin_p_7_1_c * x2 + sigma_kin_p_7_1_s * x5
    x196 = x140 * x195
    x197 = x154 * x196
    x198 = sigma_kin_p_7_1_c * x15 - sigma_kin_p_7_1_s * x14
    x199 = x136 * x139
    x200 = x198 * x199
    x201 = x154 * x200
    x202 = 8 * dq_j1
    x203 = sigma_kin_v_7_1 * x192
    x204 = x149 * x203
    x205 = x143 * x147
    x206 = x141 * x145
    x207 = x205 * x206
    x208 = -x102 * x151 + x105 * x150
    x209 = dq_i7 * x152
    x210 = x208 * x209
    x211 = x207 * x210
    x212 = x177 * x180
    x213 = sigma_kin_v_6_1 * x172
    x214 = x212 * x213
    x215 = sigma_kin_v_6_6 * x214 + sigma_kin_v_7_6 * x156
    x216 = ddq_i6 * dq_j6
    x217 = x215 * x216
    x218 = x96 * x98
    x219 = x95 * x99
    x220 = sigma_kin_p_6_6_c * x218 - sigma_kin_p_6_6_s * x219
    x221 = x169 * x170
    x222 = x220 * x221
    x223 = x167 * x212
    x224 = sigma_kin_v_6_1 * x161
    x225 = x165 * x224
    x226 = x223 * x225
    x227 = x222 * x226
    x228 = x137 * x148
    x229 = x203 * x228
    x230 = sigma_kin_p_7_6_c * x218 - sigma_kin_p_7_6_s * x219
    x231 = x147 * x230
    x232 = x143 * x153
    x233 = x145 * x232
    x234 = x231 * x233
    x235 = x199 * x234
    x236 = x229 * x235
    x237 = x227 + x236
    x238 = x128 * x131
    x239 = sigma_kin_v_5_1 * x123
    x240 = x238 * x239
    x241 = sigma_kin_v_5_5 * x240 + sigma_kin_v_6_5 * x214 + sigma_kin_v_7_5 * x156
    x242 = ddq_i5 * dq_j5
    x243 = x241 * x242
    x244 = x80 * x83
    x245 = sigma_kin_v_4_1 * x75
    x246 = x244 * x245
    x247 = sigma_kin_v_4_4 * x246 + sigma_kin_v_5_4 * x240 + sigma_kin_v_6_4 * x214 + sigma_kin_v_7_4 * x156
    x248 = ddq_i4 * dq_j4
    x249 = x247 * x248
    x250 = x118 * x121
    x251 = x89 * x91
    x252 = x88 * x92
    x253 = sigma_kin_p_5_5_c * x251 - sigma_kin_p_5_5_s * x252
    x254 = sigma_kin_v_5_1 * x114
    x255 = x120 * x254
    x256 = x253 * x255
    x257 = x250 * x256
    x258 = x238 * x257
    x259 = sigma_kin_p_6_5_c * x251 - sigma_kin_p_6_5_s * x252
    x260 = x168 * x171
    x261 = x259 * x260
    x262 = x226 * x261
    x263 = sigma_kin_p_7_5_c * x251 - sigma_kin_p_7_5_s * x252
    x264 = x146 * x263
    x265 = x204 * x264
    x266 = x232 * x265
    x267 = x206 * x266
    x268 = x258 + x262 + x267
    x269 = x52 * x58
    x270 = sigma_kin_v_3_1 * sigma_kin_v_3_3
    x271 = (
        sigma_kin_v_4_3 * x246 + sigma_kin_v_5_3 * x240 + sigma_kin_v_6_3 * x214 + sigma_kin_v_7_3 * x156 + x269 * x270
    )
    x272 = ddq_i3 * dq_j3
    x273 = x271 * x272
    x274 = sigma_kin_v_2_1 * sigma_kin_v_2_2
    x275 = x274 * x29
    x276 = sigma_kin_v_3_1 * sigma_kin_v_3_2
    x277 = (
        sigma_kin_v_4_2 * x246
        + sigma_kin_v_5_2 * x240
        + sigma_kin_v_6_2 * x214
        + sigma_kin_v_7_2 * x156
        + x269 * x276
        + x275 * x36
    )
    x278 = ddq_i2 * dq_j2
    x279 = x277 * x278
    x280 = x68 * x70
    x281 = x67 * x71
    x282 = sigma_kin_p_4_4_c * x280 - sigma_kin_p_4_4_s * x281
    x283 = x66 * x73
    x284 = x282 * x283
    x285 = sigma_kin_v_4_1 * x62
    x286 = x284 * x285
    x287 = x244 * x286
    x288 = sigma_kin_p_5_4_c * x280 - sigma_kin_p_5_4_s * x281
    x289 = x119 * x288
    x290 = x254 * x289
    x291 = x122 * x238
    x292 = x118 * x291
    x293 = x290 * x292
    x294 = x165 * x166
    x295 = sigma_kin_p_6_4_c * x280 - sigma_kin_p_6_4_s * x281
    x296 = x169 * x171
    x297 = x224 * x296
    x298 = x295 * x297
    x299 = x294 * x298
    x300 = x212 * x299
    x301 = x143 * x144
    x302 = sigma_kin_p_7_4_c * x280 - sigma_kin_p_7_4_s * x281
    x303 = x147 * x149 * x153
    x304 = x203 * x303
    x305 = x302 * x304
    x306 = x301 * x305
    x307 = x141 * x306
    x308 = x287 + x293 + x300 + x307
    x309 = sigma_kin_v_2_1**2
    x310 = x29 * x309
    x311 = sigma_kin_v_3_1**2
    x312 = x311 * x52
    x313 = sigma_kin_v_4_1**2
    x314 = x313 * x75
    x315 = x314 * x80
    x316 = sigma_kin_v_5_1**2
    x317 = x123 * x316
    x318 = x128 * x317
    x319 = sigma_kin_v_6_1**2
    x320 = x172 * x319
    x321 = x177 * x320
    x322 = sigma_kin_v_7_1**2
    x323 = x154 * x322
    x324 = x140 * x323
    x325 = x131 * x318 + x136 * x324 + x16 * x8 + x180 * x321 + x310 * x36 + x312 * x58 + x315 * x83
    x326 = x44 * x46
    x327 = x43 * x47
    x328 = sigma_kin_p_3_3_c * x326 - sigma_kin_p_3_3_s * x327
    x329 = x42 * x49
    x330 = x328 * x329
    x331 = sigma_kin_v_3_1 * x55
    x332 = x331 * x40
    x333 = x332 * x58
    x334 = x330 * x333
    x335 = sigma_kin_p_4_3_c * x326 - sigma_kin_p_4_3_s * x327
    x336 = x65 * x74
    x337 = x335 * x336
    x338 = x244 * x337
    x339 = x285 * x338
    x340 = sigma_kin_p_5_3_c * x326 - sigma_kin_p_5_3_s * x327
    x341 = x117 * x340
    x342 = x255 * x341
    x343 = x291 * x342
    x344 = sigma_kin_p_6_3_c * x326 - sigma_kin_p_6_3_s * x327
    x345 = x164 * x344
    x346 = x297 * x345
    x347 = x223 * x346
    x348 = sigma_kin_p_7_3_c * x326 - sigma_kin_p_7_3_s * x327
    x349 = x142 * x348
    x350 = x304 * x349
    x351 = x206 * x350
    x352 = x334 + x339 + x343 + x347 + x351
    x353 = x21 * x23
    x354 = x20 * x24
    x355 = sigma_kin_p_2_2_c * x353 - sigma_kin_p_2_2_s * x354
    x356 = sigma_kin_v_2_1 * x33
    x357 = x355 * x356
    x358 = x26 * x28
    x359 = x358 * x36
    x360 = x357 * x359
    x361 = sigma_kin_p_3_2_c * x353 - sigma_kin_p_3_2_s * x354
    x362 = x41 * x50
    x363 = x361 * x362
    x364 = x333 * x363
    x365 = sigma_kin_p_4_2_c * x353 - sigma_kin_p_4_2_s * x354
    x366 = x245 * x62
    x367 = x365 * x366
    x368 = x77 * x79
    x369 = x368 * x83
    x370 = x367 * x369
    x371 = sigma_kin_p_5_2_c * x353 - sigma_kin_p_5_2_s * x354
    x372 = x114 * x239
    x373 = x371 * x372
    x374 = x125 * x127
    x375 = x131 * x374
    x376 = x373 * x375
    x377 = sigma_kin_p_6_2_c * x353 - sigma_kin_p_6_2_s * x354
    x378 = x161 * x213
    x379 = x377 * x378
    x380 = x174 * x176
    x381 = x180 * x380
    x382 = x379 * x381
    x383 = sigma_kin_p_7_2_c * x353 - sigma_kin_p_7_2_s * x354
    x384 = x155 * x192
    x385 = x383 * x384
    x386 = x137 * x138
    x387 = x136 * x386
    x388 = x385 * x387
    x389 = x360 + x364 + x370 + x376 + x382 + x388
    x390 = x137**2
    x391 = x138 * x142
    x392 = x144 * x146
    x393 = x391 * x392
    x394 = x148 * x152
    x395 = ddq_i7 * x394
    x396 = x393 * x395
    x397 = x142 * x392
    x398 = x383 * x397
    x399 = 2 * dq_i2
    x400 = dq_i7 * x394
    x401 = x399 * x400
    x402 = x398 * x401
    x403 = x138 * x392
    x404 = x348 * x403
    x405 = 2 * dq_i3
    x406 = x400 * x405
    x407 = x404 * x406
    x408 = x391 * x400
    x409 = 2 * dq_i4
    x410 = x146 * x302
    x411 = x409 * x410
    x412 = x408 * x411
    x413 = 2 * dq_i5
    x414 = x144 * x263
    x415 = x413 * x414
    x416 = x408 * x415
    x417 = 2 * dq_i6
    x418 = x230 * x417
    x419 = x209 * x393
    x420 = x148 * x393
    x421 = dq_i7**2
    x422 = x208 * x421
    x423 = 2 * x422
    x424 = sigma_kin_v_7_1 * x394
    x425 = ddq_j7 * sigma_kin_v_7_7
    x426 = 2 * x425
    x427 = x18 * x31
    x428 = x31 * x9
    x429 = x356 * x38
    x430 = x30 * x356
    x431 = x331 * x60
    x432 = x331 * x53
    x433 = x366 * x85
    x434 = x366 * x81
    x435 = x133 * x372
    x436 = x129 * x372
    x437 = x182 * x378
    x438 = x178 * x378
    x439 = x200 * x384
    x440 = x196 * x384
    x441 = x429 + x430 + x431 + x432 + x433 + x434 + x435 + x436 + x437 + x438 + x439 + x440
    x442 = sigma_kin_v_7_1 * sigma_kin_v_7_6
    x443 = x149 * x442
    x444 = x209 * x417
    x445 = x139 * x390
    x446 = x145 * x445
    x447 = x205 * x446
    x448 = x208 * x447
    x449 = x444 * x448
    x450 = x176**2
    x451 = x175 * x450
    x452 = x172 * x451
    x453 = sigma_kin_v_6_1 * x452
    x454 = x154 * x445
    x455 = sigma_kin_v_7_1 * x454
    x456 = sigma_kin_v_6_6 * x453 + sigma_kin_v_7_6 * x455
    x457 = x177 * x181
    x458 = sigma_kin_v_6_6 * x213
    x459 = x140 * x198
    x460 = sigma_kin_v_7_6 * x155
    x461 = x457 * x458 + x459 * x460
    x462 = dq_i1 * x417
    x463 = x174 * x450
    x464 = x377 * x463
    x465 = x138 * x390
    x466 = x383 * x465
    x467 = x458 * x464 + x460 * x466
    x468 = dq_i6 * x399
    x469 = x167 * x451
    x470 = x164 * x469
    x471 = x344 * x470
    x472 = x296 * x471
    x473 = sigma_kin_v_6_1 * sigma_kin_v_6_6
    x474 = x142 * x446
    x475 = x348 * x474
    x476 = x303 * x475
    x477 = x442 * x476 + x472 * x473
    x478 = dq_i6 * x405
    x479 = x294 * x451
    x480 = x295 * x479
    x481 = x296 * x480
    x482 = x301 * x445
    x483 = x302 * x482
    x484 = x303 * x483
    x485 = x442 * x484 + x473 * x481
    x486 = dq_i6 * x409
    x487 = x165 * x469
    x488 = x261 * x487
    x489 = x232 * x446
    x490 = x264 * x489
    x491 = x149 * x490
    x492 = x442 * x491 + x473 * x488
    x493 = dq_i6 * x413
    x494 = x222 * x487
    x495 = x231 * x489
    x496 = x148 * x495
    x497 = x442 * x496 + x473 * x494
    x498 = dq_i6**2
    x499 = 2 * x498
    x500 = 2 * ddq_j6
    x501 = x102 * x106
    x502 = x103 * x105
    x503 = sigma_kin_p_7_7_c * x501 - sigma_kin_p_7_7_s * x502
    x504 = x394 * x503
    x505 = x192 * x504
    x506 = 2 * dq_i1
    x507 = x198 * x506
    x508 = x393 * x507
    x509 = x505 * x508
    x510 = 2 * x184 + x191
    x511 = x504 * x510
    x512 = x398 * x399
    x513 = x137 * x511
    x514 = x404 * x405
    x515 = x391 * x513
    x516 = x137 * x393
    x517 = x230 * x516
    x518 = x152 * x517
    x519 = x503 * x518
    x520 = x417 * x519
    x521 = ddq_i1 * dq_j1
    x522 = 2 * x521
    x523 = x503 * x516
    x524 = x148 * x516
    x525 = x208 * x503
    x526 = x524 * x525
    x527 = dq_i7 * x526
    x528 = sigma_kin_p_7_7_c * x107 + sigma_kin_p_7_7_s * x104
    x529 = x516 * x528
    x530 = x400 * x529
    x531 = sigma_kin_v_7_2 * x394
    x532 = x523 * x531
    x533 = -x278 * x532
    x534 = sigma_kin_v_7_3 * x394
    x535 = x523 * x534
    x536 = -x272 * x535
    x537 = sigma_kin_v_7_4 * x394
    x538 = x523 * x537
    x539 = -x248 * x538
    x540 = sigma_kin_v_7_5 * x394
    x541 = x523 * x540
    x542 = -x242 * x541
    x543 = sigma_kin_v_7_6 * x394
    x544 = x523 * x543
    x545 = -x216 * x544
    x546 = x533 + x536 + x539 + x542 + x545
    x547 = x192 * x524
    x548 = x525 * x547
    x549 = x192 * x516
    x550 = x528 * x549
    x551 = x137 * x396
    x552 = x157 * x503
    x553 = x551 * x552
    x554 = dq_i7 * x548 + x400 * x550 - x553
    x555 = 4 * x524
    x556 = dq_j7 * x555
    x557 = sigma_kin_v_7_1 * x556
    x558 = sigma_kin_v_7_1 * sigma_kin_v_7_5
    x559 = x149 * x558
    x560 = x209 * x448
    x561 = x413 * x560
    x562 = sigma_kin_v_6_1 * sigma_kin_v_6_5
    x563 = x494 * x562 + x496 * x558
    x564 = x127**2
    x565 = x126 * x564
    x566 = x123 * x565
    x567 = sigma_kin_v_5_1 * x566
    x568 = sigma_kin_v_5_5 * x567 + sigma_kin_v_6_5 * x453 + sigma_kin_v_7_5 * x455
    x569 = x128 * x132
    x570 = sigma_kin_v_5_5 * x239
    x571 = sigma_kin_v_6_5 * x213
    x572 = sigma_kin_v_7_5 * x155
    x573 = x457 * x571 + x459 * x572 + x569 * x570
    x574 = dq_i1 * x413
    x575 = x125 * x564
    x576 = x371 * x575
    x577 = x464 * x571 + x466 * x572 + x570 * x576
    x578 = dq_i5 * x399
    x579 = x122 * x565
    x580 = x117 * x579
    x581 = x340 * x580
    x582 = x120 * x581
    x583 = sigma_kin_v_5_1 * sigma_kin_v_5_5
    x584 = x472 * x562 + x476 * x558 + x582 * x583
    x585 = dq_i5 * x405
    x586 = x118 * x579
    x587 = x289 * x586
    x588 = x481 * x562 + x484 * x558 + x583 * x587
    x589 = dq_i5 * x409
    x590 = x250 * x565
    x591 = x253 * x590
    x592 = x120 * x591
    x593 = x488 * x562 + x491 * x558 + x583 * x592
    x594 = dq_i5**2
    x595 = 2 * x594
    x596 = 2 * ddq_j5
    x597 = sigma_kin_v_7_1 * sigma_kin_v_7_4
    x598 = x149 * x597
    x599 = x409 * x560
    x600 = sigma_kin_v_6_1 * sigma_kin_v_6_4
    x601 = x487 * x600
    x602 = x222 * x601 + x496 * x597
    x603 = sigma_kin_v_5_1 * sigma_kin_v_5_4
    x604 = x261 * x601 + x491 * x597 + x592 * x603
    x605 = x79**2
    x606 = x605 * x78
    x607 = x606 * x75
    x608 = sigma_kin_v_4_1 * x607
    x609 = sigma_kin_v_4_4 * x608 + sigma_kin_v_5_4 * x567 + sigma_kin_v_6_4 * x453 + sigma_kin_v_7_4 * x455
    x610 = x80 * x84
    x611 = sigma_kin_v_4_4 * x245
    x612 = sigma_kin_v_5_4 * x239
    x613 = sigma_kin_v_6_4 * x213
    x614 = sigma_kin_v_7_4 * x155
    x615 = x457 * x613 + x459 * x614 + x569 * x612 + x610 * x611
    x616 = dq_i1 * x409
    x617 = x605 * x77
    x618 = x365 * x617
    x619 = x464 * x613 + x466 * x614 + x576 * x612 + x611 * x618
    x620 = dq_i4 * x399
    x621 = sigma_kin_v_4_1 * x606
    x622 = sigma_kin_v_4_4 * x621
    x623 = x337 * x622 + x472 * x600 + x476 * x597 + x582 * x603
    x624 = dq_i4 * x405
    x625 = x284 * x622 + x481 * x600 + x484 * x597 + x587 * x603
    x626 = dq_i4**2
    x627 = 2 * x626
    x628 = 2 * ddq_j4
    x629 = sigma_kin_v_7_1 * sigma_kin_v_7_3
    x630 = x149 * x629
    x631 = x405 * x560
    x632 = sigma_kin_v_6_1 * sigma_kin_v_6_3
    x633 = x487 * x632
    x634 = x222 * x633 + x496 * x629
    x635 = sigma_kin_v_5_1 * sigma_kin_v_5_3
    x636 = x261 * x633 + x491 * x629 + x592 * x635
    x637 = sigma_kin_v_4_3 * x621
    x638 = x284 * x637 + x481 * x632 + x484 * x629 + x587 * x635
    x639 = x40**2
    x640 = x51 * x639
    x641 = sigma_kin_v_3_1 * x640
    x642 = (
        sigma_kin_v_3_3 * x641
        + sigma_kin_v_4_3 * x608
        + sigma_kin_v_5_3 * x567
        + sigma_kin_v_6_3 * x453
        + sigma_kin_v_7_3 * x455
    )
    x643 = x52 * x59
    x644 = sigma_kin_v_4_3 * x245
    x645 = sigma_kin_v_5_3 * x239
    x646 = sigma_kin_v_6_3 * x213
    x647 = sigma_kin_v_7_3 * x155
    x648 = x270 * x643 + x457 * x646 + x459 * x647 + x569 * x645 + x610 * x644
    x649 = dq_i1 * x405
    x650 = x270 * x639
    x651 = x363 * x650 + x464 * x646 + x466 * x647 + x576 * x645 + x618 * x644
    x652 = dq_i3 * x399
    x653 = x330 * x650 + x337 * x637 + x472 * x632 + x476 * x629 + x582 * x635
    x654 = dq_i3**2
    x655 = 2 * x654
    x656 = 2 * ddq_j3
    x657 = dq_i1 * x399
    x658 = x275 * x37
    x659 = x276 * x643
    x660 = sigma_kin_v_4_2 * x245
    x661 = x610 * x660
    x662 = sigma_kin_v_5_2 * x239
    x663 = x569 * x662
    x664 = sigma_kin_v_6_2 * x213
    x665 = x457 * x664
    x666 = sigma_kin_v_7_2 * x155
    x667 = x459 * x666
    x668 = sigma_kin_v_7_1 * sigma_kin_v_7_2
    x669 = x149 * x668
    x670 = x399 * x560
    x671 = sigma_kin_v_6_1 * sigma_kin_v_6_2
    x672 = x487 * x671
    x673 = x222 * x672 + x496 * x668
    x674 = sigma_kin_v_5_1 * sigma_kin_v_5_2
    x675 = x120 * x674
    x676 = x261 * x672 + x491 * x668 + x591 * x675
    x677 = sigma_kin_v_4_2 * x621
    x678 = x296 * x671
    x679 = x303 * x668
    x680 = x284 * x677 + x480 * x678 + x483 * x679 + x587 * x674
    x681 = x28**2
    x682 = x27 * x681
    x683 = (
        sigma_kin_v_3_2 * x641
        + sigma_kin_v_4_2 * x608
        + sigma_kin_v_5_2 * x567
        + sigma_kin_v_6_2 * x453
        + sigma_kin_v_7_2 * x455
        + x274 * x682
    )
    x684 = x276 * x639
    x685 = sigma_kin_v_4_2 * x337
    x686 = x330 * x684 + x472 * x671 + x476 * x668 + x582 * x674 + x621 * x685
    x687 = x658 + x659 + x661 + x663 + x665 + x667
    x688 = x26 * x681
    x689 = x274 * x688
    x690 = x355 * x689
    x691 = x363 * x684
    x692 = x618 * x660
    x693 = x576 * x662
    x694 = x464 * x664
    x695 = x466 * x666
    x696 = x690 + x691 + x692 + x693 + x694 + x695
    x697 = dq_i2**2
    x698 = 2 * x697
    x699 = 2 * ddq_j2
    x700 = x17 * x8
    x701 = x310 * x37
    x702 = x312 * x59
    x703 = x315 * x84
    x704 = x132 * x318
    x705 = x181 * x321
    x706 = x198 * x324
    x707 = x149 * x322
    x708 = x506 * x560
    x709 = x319 * x487
    x710 = x221 * x709
    x711 = x148 * x322
    x712 = x120 * x316
    x713 = x590 * x712
    x714 = x313 * x606
    x715 = x283 * x714
    x716 = x289 * x316
    x717 = x296 * x319
    x718 = x479 * x717
    x719 = x303 * x322
    x720 = x482 * x719
    x721 = x311 * x639
    x722 = x329 * x721
    x723 = x309 * x688
    x724 = x362 * x721
    x725 = x314 * x617
    x726 = x317 * x575
    x727 = x320 * x463
    x728 = x323 * x465
    x729 = 4 * ddq_j1
    x730 = sigma_kin_p_7_6_c * x219 - sigma_kin_p_7_6_s * x218
    x731 = x147 * x489
    x732 = x730 * x731
    x733 = x148 * x732
    x734 = sigma_kin_v_7_1 * x158
    x735 = sigma_kin_p_6_6_c * x219 - sigma_kin_p_6_6_s * x218
    x736 = x221 * x735
    x737 = x167 * x225
    x738 = x736 * x737
    x739 = x457 * x738
    x740 = x147 * x730
    x741 = x233 * x740
    x742 = x668 * x733 + x672 * x736
    x743 = x629 * x733 + x633 * x736
    x744 = x597 * x733 + x601 * x736
    x745 = x487 * x736
    x746 = x558 * x733 + x562 * x745
    x747 = x442 * x733 + x473 * x745
    x748 = x711 * x730
    x749 = x165 * x167
    x750 = x457 * x749
    x751 = x31 * x319
    x752 = x736 * x751
    x753 = x31 * x322
    x754 = x139 * x198
    x755 = x741 * x754
    x756 = x228 * x755
    x757 = x464 * x749
    x758 = x31 * x748
    x759 = x233 * x466
    x760 = x147 * x759
    x761 = x464 * x738
    x762 = x148 * x203
    x763 = x466 * x741
    x764 = x762 * x763
    x765 = x153 * x475
    x766 = x147 * x758
    x767 = x224 * x736
    x768 = x471 * x767
    x769 = x740 * x762
    x770 = x765 * x769
    x771 = x153 * x483
    x772 = x480 * x767
    x773 = x769 * x771
    x774 = x31 * x709
    x775 = x168 * x170
    x776 = x259 * x735 * x775
    x777 = x224 * x487
    x778 = x776 * x777
    x779 = x490 * x730
    x780 = x762 * x779
    x781 = sigma_kin_p_6_6_c * x100 + sigma_kin_p_6_6_s * x97
    x782 = x169 * x735
    x783 = x220 * x782
    x784 = sigma_kin_p_7_6_c * x100 + sigma_kin_p_7_6_s * x97
    x785 = x731 * x784
    x786 = x31 * x711
    x787 = x495 * x730
    x788 = x221 * x781
    x789 = x777 * x788
    x790 = x220 * x777
    x791 = x782 * x790
    x792 = x762 * x785
    x793 = x203 * x787
    x794 = 4 * dq_j6
    x795 = sigma_kin_p_5_5_c * x252 - sigma_kin_p_5_5_s * x251
    x796 = x250 * x795
    x797 = x255 * x796
    x798 = x569 * x797
    x799 = sigma_kin_p_7_5_c * x252 - sigma_kin_p_7_5_s * x251
    x800 = x146 * x489
    x801 = x799 * x800
    x802 = x149 * x801
    x803 = sigma_kin_p_6_5_c * x252 - sigma_kin_p_6_5_s * x251
    x804 = x260 * x803
    x805 = x737 * x804
    x806 = x457 * x805
    x807 = x204 * x799
    x808 = x146 * x807
    x809 = x233 * x808
    x810 = x459 * x809
    x811 = x487 * x804
    x812 = x442 * x802 + x473 * x811
    x813 = x590 * x795
    x814 = x668 * x802 + x672 * x804 + x675 * x813
    x815 = x120 * x813
    x816 = x629 * x802 + x633 * x804 + x635 * x815
    x817 = x597 * x802 + x601 * x804 + x603 * x815
    x818 = x558 * x802 + x562 * x811 + x583 * x815
    x819 = x707 * x799
    x820 = x775 * x803
    x821 = x220 * x820
    x822 = x230 * x801
    x823 = x790 * x820
    x824 = x762 * x822
    x825 = x31 * x712
    x826 = x796 * x825
    x827 = x751 * x804
    x828 = x233 * x459
    x829 = x31 * x819
    x830 = x146 * x829
    x831 = x576 * x797
    x832 = x464 * x805
    x833 = x466 * x809
    x834 = x565 * x795
    x835 = x121 * x834
    x836 = x341 * x835
    x837 = x342 * x835
    x838 = x224 * x804
    x839 = x471 * x838
    x840 = x765 * x808
    x841 = x31 * x716
    x842 = x290 * x813
    x843 = x232 * x392
    x844 = x445 * x843
    x845 = x302 * x844
    x846 = x480 * x838
    x847 = x807 * x845
    x848 = sigma_kin_p_5_5_c * x93 + sigma_kin_p_5_5_s * x90
    x849 = x118 * x834
    x850 = x253 * x849
    x851 = x260 * (sigma_kin_p_6_5_c * x93 + sigma_kin_p_6_5_s * x90)
    x852 = x590 * x848
    x853 = x255 * x852
    x854 = x171 * x259 * x803
    x855 = x256 * x849
    x856 = sigma_kin_p_7_5_c * x93 + sigma_kin_p_7_5_s * x90
    x857 = x800 * x856
    x858 = x31 * x707
    x859 = x263 * x489
    x860 = x777 * x851
    x861 = x777 * x854
    x862 = x204 * x857
    x863 = x807 * x859
    x864 = 4 * dq_j5
    x865 = sigma_kin_p_4_4_c * x281 - sigma_kin_p_4_4_s * x280
    x866 = x283 * x865
    x867 = x285 * x866
    x868 = x610 * x867
    x869 = x122 * x569
    x870 = sigma_kin_p_5_4_c * x281 - sigma_kin_p_5_4_s * x280
    x871 = x254 * x870
    x872 = x119 * x871
    x873 = x118 * x872
    x874 = x869 * x873
    x875 = sigma_kin_p_7_4_c * x281 - sigma_kin_p_7_4_s * x280
    x876 = x482 * x875
    x877 = x303 * x876
    x878 = x297 * x457
    x879 = sigma_kin_p_6_4_c * x281 - sigma_kin_p_6_4_s * x280
    x880 = x294 * x879
    x881 = x878 * x880
    x882 = x301 * x875
    x883 = x304 * x882
    x884 = x459 * x883
    x885 = x479 * x879
    x886 = x296 * x885
    x887 = x442 * x877 + x473 * x886
    x888 = x119 * x586
    x889 = x870 * x888
    x890 = x558 * x877 + x562 * x886 + x583 * x889
    x891 = x674 * x889 + x677 * x866 + x678 * x885 + x679 * x876
    x892 = x629 * x877 + x632 * x886 + x635 * x889 + x637 * x866
    x893 = x597 * x877 + x600 * x886 + x603 * x889 + x622 * x866
    x894 = x222 * x885
    x895 = x153 * x876
    x896 = x231 * x895
    x897 = x224 * x885
    x898 = x222 * x897
    x899 = x231 * x762
    x900 = x895 * x899
    x901 = x31 * x316
    x902 = x119 * x870
    x903 = x591 * x902
    x904 = x261 * x885
    x905 = x591 * x872
    x906 = x445 * x875
    x907 = x843 * x906
    x908 = x263 * x907
    x909 = x261 * x897
    x910 = x204 * x908
    x911 = x31 * x313
    x912 = x866 * x911
    x913 = x901 * x902
    x914 = x118 * x913
    x915 = x31 * x717
    x916 = x880 * x915
    x917 = x31 * x719
    x918 = x882 * x917
    x919 = x618 * x867
    x920 = x122 * x576
    x921 = x873 * x920
    x922 = x297 * x464
    x923 = x880 * x922
    x924 = x466 * x883
    x925 = x31 * x714
    x926 = x65 * x73
    x927 = x335 * x865 * x926
    x928 = x62 * x621
    x929 = x927 * x928
    x930 = x451 * x879
    x931 = x166 * x930
    x932 = x345 * x931
    x933 = x581 * x872
    x934 = x144 * x906
    x935 = x349 * x934
    x936 = x346 * x931
    x937 = x350 * x934
    x938 = sigma_kin_p_4_4_c * x72 + sigma_kin_p_4_4_s * x69
    x939 = x66 * x865
    x940 = x282 * x939
    x941 = x283 * x938
    x942 = x928 * x941
    x943 = x282 * x928
    x944 = x939 * x943
    x945 = x888 * (sigma_kin_p_5_4_c * x72 + sigma_kin_p_5_4_s * x69)
    x946 = x288 * x586
    x947 = x870 * x946
    x948 = sigma_kin_p_6_4_c * x72 + sigma_kin_p_6_4_s * x69
    x949 = x254 * x945
    x950 = x165 * x930
    x951 = x295 * x950
    x952 = x871 * x946
    x953 = sigma_kin_p_7_4_c * x72 + sigma_kin_p_7_4_s * x69
    x954 = x143 * x906
    x955 = x302 * x954
    x956 = x479 * x948
    x957 = x297 * x956
    x958 = x298 * x950
    x959 = x482 * x953
    x960 = x304 * x959
    x961 = x305 * x954
    x962 = 4 * dq_j4
    x963 = sigma_kin_p_3_3_c * x327 - sigma_kin_p_3_3_s * x326
    x964 = x329 * x963
    x965 = x332 * x59
    x966 = x964 * x965
    x967 = sigma_kin_p_4_3_c * x327 - sigma_kin_p_4_3_s * x326
    x968 = x336 * x967
    x969 = x285 * x968
    x970 = x610 * x969
    x971 = sigma_kin_p_5_3_c * x327 - sigma_kin_p_5_3_s * x326
    x972 = x117 * x971
    x973 = x255 * x972
    x974 = x869 * x973
    x975 = sigma_kin_p_7_3_c * x327 - sigma_kin_p_7_3_s * x326
    x976 = x474 * x975
    x977 = x303 * x976
    x978 = sigma_kin_p_6_3_c * x327 - sigma_kin_p_6_3_s * x326
    x979 = x164 * x978
    x980 = x167 * x979
    x981 = x878 * x980
    x982 = x145 * x304
    x983 = x142 * x975
    x984 = x459 * x983
    x985 = x982 * x984
    x986 = x470 * x978
    x987 = x296 * x986
    x988 = x442 * x977 + x473 * x987
    x989 = x580 * x971
    x990 = x120 * x989
    x991 = x558 * x977 + x562 * x987 + x583 * x990
    x992 = x597 * x977 + x600 * x987 + x603 * x990 + x622 * x968
    x993 = x675 * x989 + x677 * x968 + x678 * x986 + x679 * x976 + x684 * x964
    x994 = x629 * x977 + x632 * x987 + x635 * x990 + x637 * x968 + x650 * x964
    x995 = x222 * x986
    x996 = x153 * x976
    x997 = x231 * x996
    x998 = x224 * x986
    x999 = x222 * x998
    x1000 = x899 * x996
    x1001 = x121 * x565
    x1002 = x1001 * x972
    x1003 = x1002 * x253
    x1004 = x261 * x986
    x1005 = x1002 * x256
    x1006 = x264 * x996
    x1007 = x261 * x998
    x1008 = x265 * x996
    x1009 = x926 * x967
    x1010 = x1009 * x282
    x1011 = x1009 * x943
    x1012 = x166 * x451
    x1013 = x1012 * x979
    x1014 = x1013 * x295
    x1015 = x290 * x989
    x1016 = x144 * x445
    x1017 = x1016 * x983
    x1018 = x1017 * x302
    x1019 = x1013 * x298
    x1020 = x1017 * x305
    x1021 = x59 * x964
    x1022 = x31 * x311
    x1023 = x1022 * x40
    x1024 = x911 * x968
    x1025 = x825 * x972
    x1026 = x915 * x980
    x1027 = x145 * x984
    x1028 = x31 * x721
    x1029 = x41 * x49
    x1030 = x1029 * x361 * x963
    x1031 = x331 * x639
    x1032 = x1030 * x1031
    x1033 = x618 * x969
    x1034 = x920 * x973
    x1035 = x383 * x390
    x1036 = x1035 * x975
    x1037 = x145 * x391
    x1038 = x1036 * x1037
    x1039 = x922 * x980
    x1040 = x391 * x982
    x1041 = x1036 * x1040
    x1042 = sigma_kin_p_3_3_c * x48 + sigma_kin_p_3_3_s * x45
    x1043 = x42 * x963
    x1044 = x1043 * x328
    x1045 = x1042 * x329
    x1046 = x1031 * x1045
    x1047 = x1031 * x328
    x1048 = x336 * (sigma_kin_p_4_3_c * x48 + sigma_kin_p_4_3_s * x45)
    x1049 = x335 * x74 * x967
    x1050 = x1048 * x928
    x1051 = x580 * (sigma_kin_p_5_3_c * x48 + sigma_kin_p_5_3_s * x45)
    x1052 = x340 * x971
    x1053 = x1052 * x579
    x1054 = x470 * (sigma_kin_p_6_3_c * x48 + sigma_kin_p_6_3_s * x45)
    x1055 = x1051 * x255
    x1056 = x344 * x978
    x1057 = x1056 * x469
    x1058 = x120 * x579
    x1059 = x1052 * x1058
    x1060 = sigma_kin_p_7_3_c * x48 + sigma_kin_p_7_3_s * x45
    x1061 = x1060 * x474
    x1062 = x348 * x975
    x1063 = x1062 * x446
    x1064 = x1054 * x297
    x1065 = x296 * x469
    x1066 = x1056 * x1065
    x1067 = x1061 * x304
    x1068 = x303 * x446
    x1069 = x1062 * x1068
    x1070 = 4 * dq_j3
    x1071 = sigma_kin_p_2_2_c * x354 - sigma_kin_p_2_2_s * x353
    x1072 = x1071 * x358 * x37
    x1073 = x1072 * x356
    x1074 = sigma_kin_p_3_2_c * x354 - sigma_kin_p_3_2_s * x353
    x1075 = x1074 * x362
    x1076 = x1075 * x965
    x1077 = sigma_kin_p_4_2_c * x354 - sigma_kin_p_4_2_s * x353
    x1078 = x1077 * x368 * x84
    x1079 = x1078 * x366
    x1080 = sigma_kin_p_5_2_c * x354 - sigma_kin_p_5_2_s * x353
    x1081 = x1080 * x132 * x374
    x1082 = x1081 * x372
    x1083 = sigma_kin_p_7_2_c * x354 - sigma_kin_p_7_2_s * x353
    x1084 = x1083 * x465
    x1085 = x1084 * x155
    x1086 = sigma_kin_p_6_2_c * x354 - sigma_kin_p_6_2_s * x353
    x1087 = x1086 * x181 * x380
    x1088 = x1087 * x378
    x1089 = x1083 * x137
    x1090 = x1086 * x463
    x1091 = x1084 * x460 + x1090 * x458
    x1092 = x1080 * x575
    x1093 = x1084 * x572 + x1090 * x571 + x1092 * x570
    x1094 = x1077 * x617
    x1095 = x1084 * x614 + x1090 * x613 + x1092 * x612 + x1094 * x611
    x1096 = x1075 * x650 + x1084 * x647 + x1090 * x646 + x1092 * x645 + x1094 * x644
    x1097 = x1071 * x689 + x1075 * x684 + x1084 * x666 + x1090 * x664 + x1092 * x662 + x1094 * x660
    x1098 = x1090 * x167
    x1099 = x1098 * x165
    x1100 = x1099 * x222
    x1101 = x1084 * x234
    x1102 = x1098 * x225
    x1103 = x1102 * x222
    x1104 = x1101 * x762
    x1105 = x1092 * x250
    x1106 = x1105 * x253
    x1107 = x1099 * x261
    x1108 = x1092 * x257
    x1109 = x1084 * x233
    x1110 = x1109 * x264
    x1111 = x1102 * x261
    x1112 = x1084 * x145
    x1113 = x1112 * x266
    x1114 = x1094 * x284
    x1115 = x1094 * x286
    x1116 = x1092 * x122
    x1117 = x1116 * x118
    x1118 = x1090 * x294
    x1119 = x1118 * x295
    x1120 = x1117 * x290
    x1121 = x1084 * x301
    x1122 = x1121 * x302
    x1123 = x1090 * x299
    x1124 = x1084 * x306
    x1125 = x1029 * x1074
    x1126 = x1125 * x328
    x1127 = x1047 * x1125
    x1128 = x1094 * x337
    x1129 = x1128 * x285
    x1130 = x1116 * x341
    x1131 = x1098 * x345
    x1132 = x1116 * x342
    x1133 = x1083 * x390
    x1134 = x1133 * x348
    x1135 = x1037 * x1134
    x1136 = x1098 * x346
    x1137 = x1040 * x1134
    x1138 = x309 * x31
    x1139 = x1075 * x59
    x1140 = x31 * x314
    x1141 = x31 * x317
    x1142 = x31 * x320
    x1143 = x31 * x323
    x1144 = x138 * x198
    x1145 = x1089 * x1144
    x1146 = sigma_kin_p_2_2_c * x25 + sigma_kin_p_2_2_s * x22
    x1147 = x1071 * x681
    x1148 = x1147 * x355
    x1149 = x1146 * x688
    x1150 = x1149 * x356
    x1151 = sigma_kin_p_3_2_c * x25 + sigma_kin_p_3_2_s * x22
    x1152 = x1074 * x361 * x50
    x1153 = x1151 * x362
    x1154 = x1031 * x1153
    x1155 = sigma_kin_p_4_2_c * x25 + sigma_kin_p_4_2_s * x22
    x1156 = x1077 * x605
    x1157 = x1156 * x365
    x1158 = x1155 * x617
    x1159 = x1158 * x366
    x1160 = sigma_kin_p_5_2_c * x25 + sigma_kin_p_5_2_s * x22
    x1161 = x1080 * x564
    x1162 = x1161 * x371
    x1163 = sigma_kin_p_6_2_c * x25 + sigma_kin_p_6_2_s * x22
    x1164 = x1160 * x575
    x1165 = x1164 * x372
    x1166 = x1086 * x450
    x1167 = x1166 * x377
    x1168 = sigma_kin_p_7_2_c * x25 + sigma_kin_p_7_2_s * x22
    x1169 = x1133 * x383
    x1170 = x1163 * x463
    x1171 = x1170 * x378
    x1172 = x1168 * x465
    x1173 = x1172 * x384
    x1174 = 4 * dq_j2
    x1175 = dq_j1 * x11
    x1176 = 2 * x210
    x1177 = x165 * x223
    x1178 = x1177 * x222
    x1179 = x228 * x235
    x1180 = x238 * x250
    x1181 = x1180 * x253
    x1182 = x1177 * x261
    x1183 = x206 * x232
    x1184 = x1183 * x264
    x1185 = x244 * x284
    x1186 = x212 * x294
    x1187 = x1186 * x295
    x1188 = x141 * x301
    x1189 = x1188 * x302
    x1190 = x40 * x58
    x1191 = x1190 * x330
    x1192 = x291 * x341
    x1193 = x223 * x345
    x1194 = x206 * x349
    x1195 = x355 * x359
    x1196 = x1190 * x363
    x1197 = x365 * x369
    x1198 = x371 * x375
    x1199 = x377 * x381
    x1200 = x383 * x387
    x1201 = 4 * x56
    x1202 = sigma_pot_1_c * x14 - sigma_pot_1_s * x15
    x1203 = x109 * x111
    x1204 = 4 * x63
    x1205 = 4 * x115
    x1206 = x141 * x154
    x1207 = sigma_kin_v_7_2 * x1206
    x1208 = x1207 * x158
    x1209 = 4 * dq_i7 * x425
    x1210 = dq_i1 * x1209
    x1211 = 4 * x162
    x1212 = 4 * x193
    x1213 = 8 * dq_j2
    x1214 = sigma_kin_v_7_2 * x192
    x1215 = x1214 * x149
    x1216 = x1213 * x1215
    x1217 = x172 * x212
    x1218 = sigma_kin_v_6_2 * x1217
    x1219 = sigma_kin_v_6_6 * x1218 + sigma_kin_v_7_6 * x1207
    x1220 = x1219 * x216
    x1221 = -x377
    x1222 = x1221 * x463
    x1223 = -x383
    x1224 = x1223 * x465
    x1225 = 4 * ddq_j6
    x1226 = dq_i6 * x1225
    x1227 = dq_i1 * x1226
    x1228 = sigma_kin_v_6_2 * x161
    x1229 = x1228 * x165
    x1230 = x1229 * x223
    x1231 = x1230 * x222
    x1232 = x1214 * x228
    x1233 = x1232 * x235
    x1234 = x1231 + x1233
    x1235 = dq_i6 * x1213
    x1236 = x123 * x238
    x1237 = sigma_kin_v_5_2 * x1236
    x1238 = sigma_kin_v_5_5 * x1237 + sigma_kin_v_6_5 * x1218 + sigma_kin_v_7_5 * x1207
    x1239 = x1238 * x242
    x1240 = -x371
    x1241 = x1240 * x575
    x1242 = 4 * ddq_j5
    x1243 = dq_i5 * x1242
    x1244 = dq_i1 * x1243
    x1245 = x244 * x75
    x1246 = sigma_kin_v_4_2 * sigma_kin_v_4_4
    x1247 = sigma_kin_v_5_4 * x1237 + sigma_kin_v_6_4 * x1218 + sigma_kin_v_7_4 * x1207 + x1245 * x1246
    x1248 = x1247 * x248
    x1249 = -x365
    x1250 = x1249 * x617
    x1251 = 4 * ddq_j4
    x1252 = dq_i4 * x1251
    x1253 = dq_i1 * x1252
    x1254 = sigma_kin_v_5_2 * x114
    x1255 = x120 * x1254
    x1256 = x1255 * x253
    x1257 = x1180 * x1256
    x1258 = x1230 * x261
    x1259 = x1215 * x264
    x1260 = x1183 * x1259
    x1261 = x1257 + x1258 + x1260
    x1262 = dq_i5 * x1213
    x1263 = sigma_kin_v_3_2 * sigma_kin_v_3_3
    x1264 = sigma_kin_v_4_2 * sigma_kin_v_4_3
    x1265 = sigma_kin_v_5_2 * sigma_kin_v_5_3
    x1266 = sigma_kin_v_6_2 * sigma_kin_v_6_3
    x1267 = sigma_kin_v_7_2 * sigma_kin_v_7_3
    x1268 = x1206 * x1267 + x1217 * x1266 + x1236 * x1265 + x1245 * x1264 + x1263 * x269
    x1269 = x1268 * x272
    x1270 = -x361
    x1271 = x1270 * x650
    x1272 = 4 * ddq_j3
    x1273 = dq_i3 * x1272
    x1274 = dq_i1 * x1273
    x1275 = ddq_i1 * dq_j2
    x1276 = x1275 * x277
    x1277 = sigma_kin_v_2_2**2
    x1278 = x1277 * x29
    x1279 = sigma_kin_v_3_2**2
    x1280 = x1279 * x52
    x1281 = sigma_kin_v_4_2**2
    x1282 = x1281 * x75
    x1283 = sigma_kin_v_5_2**2
    x1284 = x123 * x1283
    x1285 = sigma_kin_v_6_2**2
    x1286 = x1285 * x172
    x1287 = sigma_kin_v_7_2**2
    x1288 = x1287 * x154
    x1289 = x1278 * x36 + x1280 * x58 + x1282 * x244 + x1284 * x238 + x1286 * x212 + x1288 * x141
    x1290 = 4 * ddq_i2 * dq_j2**2
    x1291 = -x355
    x1292 = x11 * x729
    x1293 = sigma_kin_v_4_2 * x62
    x1294 = x1185 * x1293
    x1295 = x1254 * x289
    x1296 = x1295 * x292
    x1297 = x1228 * x296
    x1298 = x1297 * x295
    x1299 = x1186 * x1298
    x1300 = x1214 * x303
    x1301 = x1300 * x302
    x1302 = x1188 * x1301
    x1303 = x1294 + x1296 + x1299 + x1302
    x1304 = dq_i4 * x1213
    x1305 = sigma_kin_v_3_2 * x55
    x1306 = x1305 * x40
    x1307 = x1306 * x58
    x1308 = x1307 * x330
    x1309 = x244 * x685
    x1310 = x1309 * x62
    x1311 = x1255 * x341
    x1312 = x1311 * x291
    x1313 = x1297 * x345
    x1314 = x1313 * x223
    x1315 = x1300 * x349
    x1316 = x1315 * x206
    x1317 = x1308 + x1310 + x1312 + x1314 + x1316
    x1318 = dq_i3 * x1213
    x1319 = sigma_kin_v_2_2 * x33
    x1320 = x1195 * x1319
    x1321 = x1307 * x363
    x1322 = x1293 * x75
    x1323 = x1197 * x1322
    x1324 = x123 * x1254
    x1325 = x1198 * x1324
    x1326 = x1228 * x172
    x1327 = x1199 * x1326
    x1328 = x1214 * x154
    x1329 = x1200 * x1328
    x1330 = x1320 + x1321 + x1323 + x1325 + x1327 + x1329
    x1331 = (
        x1214 * x197
        + x1214 * x201
        + x1228 * x179
        + x1228 * x183
        + x1254 * x130
        + x1254 * x134
        + x1293 * x82
        + x1293 * x86
        + x1305 * x53
        + x1305 * x60
        + x1319 * x30
        + x1319 * x38
    )
    x1332 = dq_i1 * dq_j2
    x1333 = sigma_kin_v_7_2 * x1332
    x1334 = x137 * x398
    x1335 = x504 * x506
    x1336 = dq_j2 * x649
    x1337 = x137 * x531
    x1338 = x404 * x503
    x1339 = dq_j2 * x616
    x1340 = x391 * x410
    x1341 = x1340 * x503
    x1342 = dq_j2 * x574
    x1343 = x391 * x414
    x1344 = x1343 * x503
    x1345 = dq_j2 * x462
    x1346 = dq_j2 * x12
    x1347 = x198 * x393
    x1348 = x1347 * x503
    x1349 = x137 * x505
    x1350 = x1349 * x506
    x1351 = x448 * x730
    x1352 = x1351 * x400
    x1353 = dq_j2 * x506
    x1354 = x165 * x671
    x1355 = x1354 * x167
    x1356 = x1355 * x736
    x1357 = x148 * x668
    x1358 = x32 * x506
    x1359 = x671 * x736
    x1360 = x1357 * x740
    x1361 = x228 * x668
    x1362 = x148 * x785
    x1363 = x146 * x669
    x1364 = x1363 * x799
    x1365 = x143 * x446
    x1366 = x1365 * x210
    x1367 = x675 * x796
    x1368 = x1355 * x804
    x1369 = x341 * x675
    x1370 = x671 * x804
    x1371 = x289 * x674
    x1372 = x799 * x845
    x1373 = x799 * x859
    x1374 = (
        ddq_i1 * x683
        - x12 * x687
        - x462 * x673
        - x574 * x676
        - x616 * x680
        - x649 * x686
        + x657 * x690
        + x657 * x691
        + x657 * x692
        + x657 * x693
        + x657 * x694
        + x657 * x695
        - x657 * x696
        - x669 * x708
    )
    x1375 = x147 * x876
    x1376 = x1375 * x210
    x1377 = x1353 * x669
    x1378 = x671 * x885
    x1379 = x1357 * x231
    x1380 = x119 * x674
    x1381 = x1380 * x870
    x1382 = sigma_kin_v_4_1 * sigma_kin_v_4_2
    x1383 = x1382 * x866
    x1384 = x118 * x1380
    x1385 = x1384 * x870
    x1386 = x678 * x880
    x1387 = x679 * x882
    x1388 = x345 * x678
    x1389 = x349 * x679
    x1390 = x147 * x976
    x1391 = x1390 * x210
    x1392 = x671 * x986
    x1393 = x253 * x675
    x1394 = x264 * x669
    x1395 = x295 * x678
    x1396 = x302 * x679
    x1397 = x1382 * x968
    x1398 = x675 * x972
    x1399 = x678 * x980
    x1400 = x276 * x40
    x1401 = x1354 * x223
    x1402 = x1400 * x58
    x1403 = -x1071
    x1404 = x1291 * x681
    x1405 = -x1074
    x1406 = x1270 * x50
    x1407 = dq_i1 * x1031
    x1408 = -x1077
    x1409 = x1249 * x605
    x1410 = -x1080
    x1411 = x1240 * x564
    x1412 = -x1086
    x1413 = x1221 * x450
    x1414 = -x1083
    x1415 = x145 * x205
    x1416 = -x208 * x209
    x1417 = x1223 * x390
    x1418 = -x220
    x1419 = x1418 * x221
    x1420 = x1355 * x1412 * x463
    x1421 = x1414 * x233 * x465
    x1422 = -x230
    x1423 = x1422 * x147
    x1424 = -x253
    x1425 = x1410 * x575
    x1426 = x1425 * x675
    x1427 = -x259 * x260
    x1428 = -x263
    x1429 = -x282
    x1430 = x1429 * x283
    x1431 = x1382 * x1408 * x617
    x1432 = -x288
    x1433 = -x295
    x1434 = x1412 * x463 * x678
    x1435 = -x302
    x1436 = x1414 * x679
    x1437 = x1405 * x684
    x1438 = x1408 * x660
    x1439 = x1410 * x662
    x1440 = x1412 * x664
    x1441 = x1414 * x666
    x1442 = -x328
    x1443 = -x335
    x1444 = x1443 * x336
    x1445 = -x340
    x1446 = x117 * x122
    x1447 = -x344
    x1448 = x164 * x167
    x1449 = -x348
    x1450 = x1403 * x274
    x1451 = -x59
    x1452 = -x84
    x1453 = -x132
    x1454 = -x181
    x1455 = -x198
    x1456 = x57 * x639
    x1457 = x64 * x75
    x1458 = sigma_pot_1_c * x5 + sigma_pot_1_off + sigma_pot_1_s * x2
    x1459 = x116 * x123
    x1460 = x1084 * x154
    x1461 = sigma_kin_v_7_2 * x158
    x1462 = x1460 * x1461
    x1463 = x163 * x172
    x1464 = x154 * x194
    x1465 = x1112 * x205
    x1466 = x1465 * x210
    x1467 = sigma_kin_v_6_2 * sigma_kin_v_6_6
    x1468 = x1467 * x172
    x1469 = sigma_kin_v_7_2 * sigma_kin_v_7_6
    x1470 = x1469 * x154
    x1471 = x1084 * x1470 + x1090 * x1468
    x1472 = x1471 * x216
    x1473 = x1098 * x1229
    x1474 = x1473 * x222
    x1475 = x1214 * x148
    x1476 = x1101 * x1475
    x1477 = x1474 + x1476
    x1478 = sigma_kin_v_5_2 * sigma_kin_v_5_5
    x1479 = x123 * x1478
    x1480 = sigma_kin_v_6_2 * sigma_kin_v_6_5
    x1481 = x1480 * x172
    x1482 = sigma_kin_v_7_2 * sigma_kin_v_7_5
    x1483 = x1482 * x154
    x1484 = x1084 * x1483 + x1090 * x1481 + x1092 * x1479
    x1485 = x1484 * x242
    x1486 = x1246 * x75
    x1487 = sigma_kin_v_5_2 * sigma_kin_v_5_4
    x1488 = x123 * x1487
    x1489 = sigma_kin_v_6_2 * sigma_kin_v_6_4
    x1490 = x1489 * x172
    x1491 = sigma_kin_v_7_2 * sigma_kin_v_7_4
    x1492 = x1491 * x154
    x1493 = x1084 * x1492 + x1090 * x1490 + x1092 * x1488 + x1094 * x1486
    x1494 = x1493 * x248
    x1495 = x1105 * x1256
    x1496 = x1473 * x261
    x1497 = x1109 * x1259
    x1498 = x1495 + x1496 + x1497
    x1499 = x1263 * x639
    x1500 = x1264 * x75
    x1501 = x123 * x1265
    x1502 = x1266 * x172
    x1503 = x1267 * x154
    x1504 = x1075 * x1499 + x1084 * x1503 + x1090 * x1502 + x1092 * x1501 + x1094 * x1500
    x1505 = x1504 * x272
    x1506 = x1277 * x688
    x1507 = x1279 * x639
    x1508 = x1282 * x617
    x1509 = x1284 * x575
    x1510 = x1286 * x463
    x1511 = x1288 * x465
    x1512 = x1071 * x1506 + x1075 * x1507 + x1077 * x1508 + x1080 * x1509 + x1083 * x1511 + x1086 * x1510
    x1513 = x1114 * x1293
    x1514 = x1117 * x1295
    x1515 = x1118 * x1298
    x1516 = x1121 * x1301
    x1517 = x1513 + x1514 + x1515 + x1516
    x1518 = x1305 * x639
    x1519 = x1126 * x1518
    x1520 = x1094 * x685
    x1521 = x1520 * x62
    x1522 = x1116 * x1311
    x1523 = x1098 * x1313
    x1524 = x1135 * x1300
    x1525 = x1519 + x1521 + x1522 + x1523 + x1524
    x1526 = x1072 * x1319 + x1078 * x1322 + x1081 * x1324 + x1087 * x1326 + x1139 * x1306 + x1145 * x1328
    x1527 = x137 * x395
    x1528 = x400 * x507
    x1529 = x137 * x392
    x1530 = x348 * x406
    x1531 = x137 * x400
    x1532 = x142 * x1531
    x1533 = x137 * x209
    x1534 = x1533 * x418
    x1535 = x228 * x423
    x1536 = x1149 * x1319
    x1537 = x1148 * x1319
    x1538 = x1153 * x1518
    x1539 = x1152 * x1518
    x1540 = x1158 * x1322
    x1541 = x1157 * x1322
    x1542 = x1164 * x1324
    x1543 = x1162 * x1324
    x1544 = x1170 * x1326
    x1545 = x1167 * x1326
    x1546 = x1172 * x1328
    x1547 = x1169 * x1328
    x1548 = x1536 + x1537 + x1538 + x1539 + x1540 + x1541 + x1542 + x1543 + x1544 + x1545 + x1546 + x1547
    x1549 = x1469 * x149
    x1550 = sigma_kin_v_6_2 * x452
    x1551 = sigma_kin_v_7_2 * x454
    x1552 = sigma_kin_v_6_6 * x1550 + sigma_kin_v_7_6 * x1551
    x1553 = x1468 * x457 + x1470 * x459
    x1554 = x1468 * x464 + x1470 * x466
    x1555 = x1467 * x472 + x1469 * x476
    x1556 = x1467 * x481 + x1469 * x484
    x1557 = x1467 * x488 + x1469 * x491
    x1558 = x1467 * x494 + x1469 * x496
    x1559 = x1349 * x512
    x1560 = x184 + x187 + x188 + x189 + x190
    x1561 = x1560 + 2 * x185 + x186
    x1562 = x1561 * x504
    x1563 = x137 * x1562
    x1564 = x1563 * x391
    x1565 = 2 * x278
    x1566 = x424 * x521
    x1567 = -x1566 * x523
    x1568 = x1567 + x539 + x542 + x545 + x554
    x1569 = sigma_kin_v_7_2 * x556
    x1570 = x1482 * x149
    x1571 = x1480 * x494 + x1482 * x496
    x1572 = sigma_kin_v_5_2 * x566
    x1573 = sigma_kin_v_5_5 * x1572 + sigma_kin_v_6_5 * x1550 + sigma_kin_v_7_5 * x1551
    x1574 = x1479 * x569 + x1481 * x457 + x1483 * x459
    x1575 = x1479 * x576 + x1481 * x464 + x1483 * x466
    x1576 = x1478 * x582 + x1480 * x472 + x1482 * x476
    x1577 = x1478 * x587 + x1480 * x481 + x1482 * x484
    x1578 = x1478 * x592 + x1480 * x488 + x1482 * x491
    x1579 = x149 * x1491
    x1580 = x1489 * x494 + x1491 * x496
    x1581 = x1487 * x592 + x1489 * x488 + x1491 * x491
    x1582 = sigma_kin_v_4_2 * x607
    x1583 = sigma_kin_v_4_4 * x1582 + sigma_kin_v_5_4 * x1572 + sigma_kin_v_6_4 * x1550 + sigma_kin_v_7_4 * x1551
    x1584 = x1486 * x610 + x1488 * x569 + x1490 * x457 + x1492 * x459
    x1585 = x1486 * x618 + x1488 * x576 + x1490 * x464 + x1492 * x466
    x1586 = x606 * x685
    x1587 = sigma_kin_v_4_4 * x1586 + x1487 * x582 + x1489 * x472 + x1491 * x476
    x1588 = x284 * x606
    x1589 = x1246 * x1588 + x1487 * x587 + x1489 * x481 + x1491 * x484
    x1590 = x1267 * x149
    x1591 = x1266 * x494 + x1267 * x496
    x1592 = x1265 * x592 + x1266 * x488 + x1267 * x491
    x1593 = x1264 * x1588 + x1265 * x587 + x1266 * x481 + x1267 * x484
    x1594 = (
        sigma_kin_v_4_3 * x1582
        + sigma_kin_v_5_3 * x1572
        + sigma_kin_v_6_3 * x1550
        + sigma_kin_v_7_3 * x1551
        + x1263 * x640
    )
    x1595 = x1263 * x643 + x1500 * x610 + x1501 * x569 + x1502 * x457 + x1503 * x459
    x1596 = x1499 * x363 + x1500 * x618 + x1501 * x576 + x1502 * x464 + x1503 * x466
    x1597 = sigma_kin_v_4_3 * x1586 + x1265 * x582 + x1266 * x472 + x1267 * x476 + x1499 * x330
    x1598 = 2 * ddq_j1
    x1599 = x1506 * x355
    x1600 = x1507 * x363
    x1601 = x1508 * x365
    x1602 = x1509 * x371
    x1603 = x1510 * x377
    x1604 = x1511 * x383
    x1605 = x1287 * x149
    x1606 = x1285 * x487
    x1607 = x1606 * x221
    x1608 = x1287 * x148
    x1609 = x120 * x1283
    x1610 = x1609 * x590
    x1611 = x1281 * x606
    x1612 = x1611 * x283
    x1613 = x1285 * x296
    x1614 = x1613 * x479
    x1615 = x1287 * x303
    x1616 = x1615 * x482
    x1617 = x1507 * x329
    x1618 = 4 * ddq_j2
    x1619 = x1229 * x167
    x1620 = x1619 * x736
    x1621 = x1620 * x464
    x1622 = x1475 * x763
    x1623 = x1266 * x745 + x1267 * x733
    x1624 = x1489 * x745 + x1491 * x733
    x1625 = x1480 * x745 + x1482 * x733
    x1626 = x1467 * x745 + x1469 * x733
    x1627 = x1285 * x32
    x1628 = x1627 * x736
    x1629 = x1287 * x32
    x1630 = x1608 * x32
    x1631 = x1630 * x740
    x1632 = x1228 * x736
    x1633 = x1632 * x471
    x1634 = x1475 * x740
    x1635 = x1634 * x765
    x1636 = x1632 * x480
    x1637 = x1634 * x771
    x1638 = x1606 * x32
    x1639 = x1228 * x487
    x1640 = x1639 * x776
    x1641 = x1475 * x779
    x1642 = x1639 * x788
    x1643 = x1639 * x783
    x1644 = x1214 * x1362
    x1645 = x1214 * x787
    x1646 = x1255 * x796
    x1647 = x1646 * x576
    x1648 = x1619 * x804
    x1649 = x1648 * x464
    x1650 = x146 * x799
    x1651 = x1215 * x1650
    x1652 = x1651 * x759
    x1653 = x1467 * x811 + x1469 * x802
    x1654 = x1265 * x815 + x1266 * x811 + x1267 * x802
    x1655 = x1487 * x815 + x1489 * x811 + x1491 * x802
    x1656 = x1478 * x815 + x1480 * x811 + x1482 * x802
    x1657 = x1639 * x821
    x1658 = x1475 * x822
    x1659 = x1609 * x32
    x1660 = x1659 * x796
    x1661 = x1627 * x804
    x1662 = x1605 * x32
    x1663 = x1650 * x1662
    x1664 = x1311 * x835
    x1665 = x1228 * x804
    x1666 = x1665 * x471
    x1667 = x1651 * x765
    x1668 = x1283 * x32
    x1669 = x289 * x813
    x1670 = x1295 * x813
    x1671 = x1665 * x480
    x1672 = x1215 * x1372
    x1673 = x1255 * x852
    x1674 = x1255 * x850
    x1675 = x1639 * x851
    x1676 = x1639 * x854
    x1677 = x1215 * x857
    x1678 = x1215 * x1373
    x1679 = x1293 * x866
    x1680 = x1679 * x618
    x1681 = x1254 * x902
    x1682 = x118 * x1681
    x1683 = x1682 * x920
    x1684 = x1297 * x880
    x1685 = x1684 * x464
    x1686 = x1300 * x882
    x1687 = x1686 * x466
    x1688 = x1467 * x886 + x1469 * x877
    x1689 = x1478 * x889 + x1480 * x886 + x1482 * x877
    x1690 = x606 * x866
    x1691 = x1264 * x1690 + x1265 * x889 + x1266 * x886 + x1267 * x877
    x1692 = x1246 * x1690 + x1487 * x889 + x1489 * x886 + x1491 * x877
    x1693 = x1228 * x885
    x1694 = x1693 * x222
    x1695 = x1475 * x231
    x1696 = x1695 * x895
    x1697 = x1681 * x591
    x1698 = x1693 * x261
    x1699 = x1215 * x908
    x1700 = x1281 * x32
    x1701 = x1700 * x866
    x1702 = x1668 * x902
    x1703 = x118 * x1702
    x1704 = x1613 * x32
    x1705 = x1704 * x880
    x1706 = x1615 * x32
    x1707 = x1706 * x882
    x1708 = x1611 * x32
    x1709 = x1293 * x606
    x1710 = x1709 * x927
    x1711 = x1681 * x581
    x1712 = x1313 * x931
    x1713 = x1315 * x934
    x1714 = x1709 * x941
    x1715 = x1709 * x940
    x1716 = x1254 * x945
    x1717 = x1254 * x947
    x1718 = x1297 * x956
    x1719 = x1297 * x951
    x1720 = x1300 * x959
    x1721 = x1300 * x955
    x1722 = x1030 * x1518
    x1723 = x1293 * x968
    x1724 = x1723 * x618
    x1725 = x1255 * x972
    x1726 = x1725 * x920
    x1727 = x1297 * x980
    x1728 = x1727 * x464
    x1729 = x1038 * x1300
    x1730 = x1467 * x987 + x1469 * x977
    x1731 = x1478 * x990 + x1480 * x987 + x1482 * x977
    x1732 = x606 * x968
    x1733 = x1246 * x1732 + x1487 * x990 + x1489 * x987 + x1491 * x977
    x1734 = x1264 * x1732 + x1265 * x990 + x1266 * x987 + x1267 * x977 + x1499 * x964
    x1735 = x1228 * x986
    x1736 = x1735 * x222
    x1737 = x1695 * x996
    x1738 = x1002 * x1256
    x1739 = x1735 * x261
    x1740 = x1259 * x996
    x1741 = x1010 * x1709
    x1742 = x289 * x989
    x1743 = x1295 * x989
    x1744 = x1013 * x1298
    x1745 = x1017 * x1301
    x1746 = x1279 * x32
    x1747 = x1746 * x40
    x1748 = x1700 * x968
    x1749 = x1659 * x972
    x1750 = x1704 * x980
    x1751 = x1507 * x32
    x1752 = x1045 * x1518
    x1753 = x1048 * x1709
    x1754 = x1051 * x1255
    x1755 = x1054 * x1297
    x1756 = x1061 * x1300
    x1757 = x289 * x292
    x1758 = x1277 * x32
    x1759 = x1282 * x32
    x1760 = x1284 * x32
    x1761 = x1286 * x32
    x1762 = x1288 * x32
    x1763 = x1117 * x289
    x1764 = x109 * (sigma_pot_3_c * x326 - sigma_pot_3_s * x327)
    x1765 = x120 * x1205
    x1766 = sigma_kin_v_7_3 * x1206
    x1767 = x158 * x1766
    x1768 = sigma_kin_v_7_1 * x1210
    x1769 = x1211 * x296
    x1770 = x1212 * x303
    x1771 = 8 * dq_j3
    x1772 = sigma_kin_v_7_3 * x192
    x1773 = x149 * x1772
    x1774 = x1771 * x1773
    x1775 = sigma_kin_v_6_3 * x1217
    x1776 = sigma_kin_v_6_6 * x1775 + sigma_kin_v_7_6 * x1766
    x1777 = x1776 * x216
    x1778 = x1447 * x470
    x1779 = x1778 * x296
    x1780 = x1449 * x474
    x1781 = x1780 * x303
    x1782 = sigma_kin_v_6_3 * x161
    x1783 = x1178 * x1782
    x1784 = x1179 * x1772
    x1785 = x1783 + x1784
    x1786 = dq_i6 * x1771
    x1787 = sigma_kin_v_5_3 * x1236
    x1788 = sigma_kin_v_5_5 * x1787 + sigma_kin_v_6_5 * x1775 + sigma_kin_v_7_5 * x1766
    x1789 = x1788 * x242
    x1790 = x1445 * x580
    x1791 = x120 * x1790
    x1792 = sigma_kin_v_4_3 * sigma_kin_v_4_4
    x1793 = sigma_kin_v_5_4 * x1787 + sigma_kin_v_6_4 * x1775 + sigma_kin_v_7_4 * x1766 + x1245 * x1792
    x1794 = x1793 * x248
    x1795 = sigma_kin_v_5_3 * x114
    x1796 = x120 * x1795
    x1797 = x1181 * x1796
    x1798 = x1782 * x261
    x1799 = x1177 * x1798
    x1800 = x1184 * x1773
    x1801 = x1797 + x1799 + x1800
    x1802 = dq_i5 * x1771
    x1803 = ddq_i1 * dq_j3
    x1804 = x1803 * x271
    x1805 = x1268 * x278
    x1806 = dq_i1 * dq_i2
    x1807 = x1806 * (x1442 * x329 * x684 + x1444 * x677 + x1778 * x678 + x1780 * x679 + x1790 * x675)
    x1808 = sigma_kin_v_3_3**2
    x1809 = x1808 * x52
    x1810 = sigma_kin_v_4_3**2
    x1811 = x1810 * x75
    x1812 = sigma_kin_v_5_3**2
    x1813 = x123 * x1812
    x1814 = sigma_kin_v_6_3**2
    x1815 = x172 * x1814
    x1816 = sigma_kin_v_7_3**2
    x1817 = x154 * x1816
    x1818 = x141 * x1817 + x1809 * x58 + x1811 * x244 + x1813 * x238 + x1815 * x212
    x1819 = 4 * ddq_i3 * dq_j3**2
    x1820 = sigma_kin_v_4_3 * x62
    x1821 = x1185 * x1820
    x1822 = x1757 * x1795
    x1823 = x1782 * x296
    x1824 = x1187 * x1823
    x1825 = x1772 * x303
    x1826 = x1189 * x1825
    x1827 = x1821 + x1822 + x1824 + x1826
    x1828 = dq_i4 * x1771
    x1829 = sigma_kin_v_3_3 * x55
    x1830 = x1820 * x75
    x1831 = x123 * x1795
    x1832 = x172 * x1782
    x1833 = x154 * x1772
    x1834 = x1196 * x1829 + x1197 * x1830 + x1198 * x1831 + x1199 * x1832 + x1200 * x1833
    x1835 = dq_i2 * x1771
    x1836 = x1191 * x1829
    x1837 = x1820 * x338
    x1838 = x1192 * x1796
    x1839 = x1193 * x1823
    x1840 = x1194 * x1825
    x1841 = x1836 + x1837 + x1838 + x1839 + x1840
    x1842 = 8 * x54
    x1843 = (
        x130 * x1795
        + x134 * x1795
        + x1772 * x197
        + x1772 * x201
        + x1782 * x179
        + x1782 * x183
        + x1820 * x82
        + x1820 * x86
        + x1829 * x53
        + x1829 * x60
    )
    x1844 = dq_i1 * dq_j3
    x1845 = sigma_kin_v_7_3 * x1844
    x1846 = dq_j3 * x657
    x1847 = x137 * x534
    x1848 = x398 * x503
    x1849 = x1847 * x1848
    x1850 = x137 * x404
    x1851 = x1850 * x186
    x1852 = dq_j3 * x616
    x1853 = x1341 * x1847
    x1854 = dq_j3 * x574
    x1855 = x1344 * x1847
    x1856 = dq_j3 * x462
    x1857 = sigma_kin_v_7_3 * x519
    x1858 = dq_j3 * x12
    x1859 = x1348 * x534
    x1860 = x1850 * x505
    x1861 = dq_j3 * x506
    x1862 = x165 * x632
    x1863 = x167 * x1862
    x1864 = x1863 * x736
    x1865 = x148 * x629
    x1866 = x632 * x736
    x1867 = x1865 * x740
    x1868 = x506 * x54
    x1869 = x228 * x629
    x1870 = x1650 * x630
    x1871 = x120 * x635
    x1872 = x1871 * x796
    x1873 = x1863 * x804
    x1874 = x1871 * x341
    x1875 = x632 * x804
    x1876 = x289 * x635
    x1877 = (
        ddq_i1 * x642
        + 2 * dq_i1 * dq_i3 * sigma_kin_v_3_1 * sigma_kin_v_3_3 * x328 * x42 * x49 * x639
        + 2 * dq_i1 * dq_i3 * sigma_kin_v_4_1 * sigma_kin_v_4_3 * x335 * x605 * x65 * x74 * x78
        + 2 * dq_i1 * dq_i3 * sigma_kin_v_5_1 * sigma_kin_v_5_3 * x117 * x120 * x122 * x126 * x340 * x564
        + 2 * dq_i1 * dq_i3 * sigma_kin_v_6_1 * sigma_kin_v_6_3 * x164 * x167 * x169 * x171 * x175 * x344 * x450
        + 2 * dq_i1 * dq_i3 * sigma_kin_v_7_1 * sigma_kin_v_7_3 * x139 * x142 * x145 * x147 * x149 * x153 * x348 * x390
        - x12 * x648
        - x462 * x634
        - x574 * x636
        - x616 * x638
        - x630 * x708
        - x649 * x653
        - x651 * x657
    )
    x1878 = x1861 * x630
    x1879 = x632 * x885
    x1880 = x635 * x902
    x1881 = sigma_kin_v_4_1 * sigma_kin_v_4_3
    x1882 = x1881 * x866
    x1883 = x118 * x1880
    x1884 = x296 * x632
    x1885 = x1884 * x880
    x1886 = x303 * x629
    x1887 = x1886 * x882
    x1888 = x1884 * x345
    x1889 = x1886 * x349
    x1890 = x1862 * x223
    x1891 = x1871 * x253
    x1892 = x264 * x630
    x1893 = x1884 * x295
    x1894 = x1886 * x302
    x1895 = x1098 * x1862
    x1896 = x1096 * x1803
    x1897 = x270 * x40
    x1898 = dq_i2 * x1844
    x1899 = -x963
    x1900 = x1442 * x42
    x1901 = x1899 * x1900
    x1902 = -x967
    x1903 = x1443 * x74
    x1904 = x1902 * x1903
    x1905 = -x971
    x1906 = x1058 * x1445 * x1905
    x1907 = -x978
    x1908 = x1065 * x1447 * x1907
    x1909 = -x975
    x1910 = x1909 * x474
    x1911 = x1416 * x147 * x1910
    x1912 = x1068 * x1449 * x1909
    x1913 = x1907 * x470
    x1914 = x1913 * x632
    x1915 = x153 * x1910
    x1916 = x1423 * x1915
    x1917 = x1871 * x1905
    x1918 = x1001 * x117 * x1424
    x1919 = x1428 * x146 * x1915
    x1920 = x1902 * x637
    x1921 = x1429 * x926
    x1922 = x1905 * x580
    x1923 = x119 * x1432 * x1922
    x1924 = x1884 * x1907
    x1925 = x1012 * x1433 * x164
    x1926 = x1886 * x1909
    x1927 = x1016 * x142
    x1928 = x1435 * x1927
    x1929 = x1899 * x650
    x1930 = x1902 * x336
    x1931 = x1881 * x1930
    x1932 = x1446 * x1917
    x1933 = x1448 * x1924
    x1934 = x1037 * x1417
    x1935 = x1451 * x1899 * x329
    x1936 = x1452 * x80
    x1937 = x128 * x1453
    x1938 = x1454 * x177
    x1939 = x142 * x145
    x1940 = x140 * x1455 * x1939
    x1941 = x1458 * (sigma_pot_2_c * x354 - sigma_pot_2_s * x353)
    x1942 = sigma_kin_v_7_3 * x1460
    x1943 = x158 * x1942
    x1944 = dq_i2 * sigma_kin_v_7_2 * x1209
    x1945 = x1090 * x172
    x1946 = sigma_kin_v_6_3 * x1945
    x1947 = sigma_kin_v_6_6 * x1946 + sigma_kin_v_7_6 * x1942
    x1948 = x1947 * x216
    x1949 = dq_i2 * x1226
    x1950 = x1782 * x222
    x1951 = x1099 * x1950
    x1952 = x148 * x1772
    x1953 = x1101 * x1952
    x1954 = x1951 + x1953
    x1955 = x1092 * x123
    x1956 = sigma_kin_v_5_3 * sigma_kin_v_5_5
    x1957 = sigma_kin_v_6_5 * x1946 + sigma_kin_v_7_5 * x1942 + x1955 * x1956
    x1958 = x1957 * x242
    x1959 = dq_i2 * x1243
    x1960 = x1792 * x75
    x1961 = sigma_kin_v_5_3 * sigma_kin_v_5_4
    x1962 = sigma_kin_v_6_3 * sigma_kin_v_6_4
    x1963 = sigma_kin_v_7_3 * sigma_kin_v_7_4
    x1964 = x1094 * x1960 + x1460 * x1963 + x1945 * x1962 + x1955 * x1961
    x1965 = x1964 * x248
    x1966 = x1246 * x606
    x1967 = dq_i2 * x1252
    x1968 = x1106 * x1796
    x1969 = x1099 * x1798
    x1970 = x1110 * x1773
    x1971 = x1968 + x1969 + x1970
    x1972 = x1504 * x278
    x1973 = x1808 * x639
    x1974 = x1811 * x617
    x1975 = x1813 * x575
    x1976 = x1815 * x463
    x1977 = x1817 * x465
    x1978 = x1075 * x1973 + x1077 * x1974 + x1080 * x1975 + x1083 * x1977 + x1086 * x1976
    x1979 = x1618 * x697
    x1980 = x1114 * x1820
    x1981 = x1795 * x289
    x1982 = x1117 * x1981
    x1983 = x1119 * x1823
    x1984 = x1122 * x1825
    x1985 = x1980 + x1982 + x1983 + x1984
    x1986 = x1829 * x40
    x1987 = x1078 * x1830 + x1081 * x1831 + x1087 * x1832 + x1139 * x1986 + x1145 * x1833
    x1988 = 8 * x1844
    x1989 = x1829 * x639
    x1990 = x1126 * x1989
    x1991 = x1128 * x1820
    x1992 = x1130 * x1796
    x1993 = x1131 * x1823
    x1994 = x1135 * x1825
    x1995 = x1990 + x1991 + x1992 + x1993 + x1994
    x1996 = (
        x1152 * x1989
        + x1153 * x1989
        + x1157 * x1830
        + x1158 * x1830
        + x1162 * x1831
        + x1164 * x1831
        + x1167 * x1832
        + x1169 * x1833
        + x1170 * x1832
        + x1172 * x1833
    )
    x1997 = dq_i2 * dq_j3
    x1998 = sigma_kin_v_7_3 * x1997
    x1999 = ddq_i2 * dq_j3
    x2000 = x399 * x504
    x2001 = dq_j3 * x620
    x2002 = dq_j3 * x578
    x2003 = dq_j3 * x468
    x2004 = dq_j3 * x698
    x2005 = x1351 * x401
    x2006 = x1266 * x736
    x2007 = x1267 * x148
    x2008 = x2007 * x740
    x2009 = x399 * x54
    x2010 = x1266 * x487
    x2011 = x1590 * x1650
    x2012 = dq_j3 * x399
    x2013 = x120 * x1265
    x2014 = x2013 * x796
    x2015 = x1266 * x804
    x2016 = x2013 * x341
    x2017 = x1265 * x289
    x2018 = (
        ddq_i2 * x1594
        + 2 * dq_i2 * dq_i3 * sigma_kin_v_3_2 * sigma_kin_v_3_3 * x328 * x42 * x49 * x639
        + 2 * dq_i2 * dq_i3 * sigma_kin_v_4_2 * sigma_kin_v_4_3 * x335 * x605 * x65 * x74 * x78
        + 2 * dq_i2 * dq_i3 * sigma_kin_v_5_2 * sigma_kin_v_5_3 * x117 * x120 * x122 * x126 * x340 * x564
        + 2 * dq_i2 * dq_i3 * sigma_kin_v_6_2 * sigma_kin_v_6_3 * x164 * x167 * x169 * x171 * x175 * x344 * x450
        + 2 * dq_i2 * dq_i3 * sigma_kin_v_7_2 * sigma_kin_v_7_3 * x139 * x142 * x145 * x147 * x149 * x153 * x348 * x390
        - x1590 * x670
        - x1591 * x468
        - x1592 * x578
        - x1593 * x620
        - x1595 * x657
        - x1596 * x698
        - x1597 * x652
    )
    x2019 = x1590 * x2012
    x2020 = x1266 * x222
    x2021 = x1265 * x902
    x2022 = x1266 * x261
    x2023 = x1264 * x866
    x2024 = x118 * x2021
    x2025 = x1266 * x296
    x2026 = x2025 * x880
    x2027 = x1267 * x303
    x2028 = x2027 * x882
    x2029 = x1264 * x606
    x2030 = x2025 * x345
    x2031 = x1263 * x40
    x2032 = x1266 * x1913
    x2033 = x1905 * x2013
    x2034 = x1907 * x2025
    x2035 = x1909 * x2027
    x2036 = x1499 * x1899
    x2037 = x1264 * x1930
    x2038 = x1446 * x2033
    x2039 = x1448 * x2034
    x2040 = x606 * x64
    x2041 = x110 * x1458
    x2042 = x1051 * x120
    x2043 = sigma_kin_v_7_3 * x158
    x2044 = x2043 * x977
    x2045 = x1054 * x296
    x2046 = x1061 * x303
    x2047 = sigma_kin_v_6_3 * sigma_kin_v_6_6
    x2048 = sigma_kin_v_7_3 * sigma_kin_v_7_6
    x2049 = x2047 * x987 + x2048 * x977
    x2050 = x2049 * x216
    x2051 = x1950 * x986
    x2052 = x1952 * x997
    x2053 = x2051 + x2052
    x2054 = sigma_kin_v_6_3 * sigma_kin_v_6_5
    x2055 = sigma_kin_v_7_3 * sigma_kin_v_7_5
    x2056 = x1956 * x990 + x2054 * x987 + x2055 * x977
    x2057 = x2056 * x242
    x2058 = x1732 * x1792 + x1961 * x990 + x1962 * x987 + x1963 * x977
    x2059 = x2058 * x248
    x2060 = x1003 * x1796
    x2061 = x1798 * x986
    x2062 = x1006 * x1773
    x2063 = x2060 + x2061 + x2062
    x2064 = x1734 * x278
    x2065 = x1973 * x329
    x2066 = x1810 * x606
    x2067 = x120 * x1812
    x2068 = x1814 * x296
    x2069 = x1816 * x303
    x2070 = x2065 * x963 + x2066 * x968 + x2067 * x989 + x2068 * x986 + x2069 * x976
    x2071 = x1820 * x606
    x2072 = x1010 * x2071
    x2073 = x1981 * x989
    x2074 = x1014 * x1823
    x2075 = x1018 * x1825
    x2076 = x2072 + x2073 + x2074 + x2075
    x2077 = x1820 * x968
    x2078 = x1796 * x972
    x2079 = x1823 * x980
    x2080 = x1021 * x1986 + x1027 * x1825 + x2077 * x610 + x2078 * x869 + x2079 * x457
    x2081 = x1030 * x1989 + x1038 * x1825 + x2077 * x618 + x2078 * x920 + x2079 * x464
    x2082 = x383 * x401
    x2083 = x138 * x1531
    x2084 = x1045 * x1989
    x2085 = x1044 * x1989
    x2086 = x1048 * x2071
    x2087 = x1049 * x2071
    x2088 = x1051 * x1796
    x2089 = x1059 * x1795
    x2090 = x1054 * x1823
    x2091 = x1066 * x1782
    x2092 = x1061 * x1825
    x2093 = x1069 * x1772
    x2094 = x2084 + x2085 + x2086 + x2087 + x2088 + x2089 + x2090 + x2091 + x2092 + x2093
    x2095 = x149 * x2048
    x2096 = sigma_kin_v_6_3 * x452
    x2097 = sigma_kin_v_7_3 * x454
    x2098 = sigma_kin_v_6_6 * x2096 + sigma_kin_v_7_6 * x2097
    x2099 = x172 * x2047
    x2100 = x154 * x2048
    x2101 = x2099 * x457 + x2100 * x459
    x2102 = x2099 * x464 + x2100 * x466
    x2103 = x2047 * x472 + x2048 * x476
    x2104 = x2047 * x481 + x2048 * x484
    x2105 = x2047 * x488 + x2048 * x491
    x2106 = x2047 * x494 + x2048 * x496
    x2107 = x1349 * x514
    x2108 = x1560 + x185 + 2 * x186
    x2109 = x2108 * x504
    x2110 = x137 * x2109
    x2111 = x2110 * x391
    x2112 = 2 * x272
    x2113 = sigma_kin_v_7_3 * x556
    x2114 = x149 * x2055
    x2115 = x2054 * x494 + x2055 * x496
    x2116 = sigma_kin_v_5_3 * x566
    x2117 = sigma_kin_v_5_5 * x2116 + sigma_kin_v_6_5 * x2096 + sigma_kin_v_7_5 * x2097
    x2118 = x123 * x1956
    x2119 = x172 * x2054
    x2120 = x154 * x2055
    x2121 = x2118 * x569 + x2119 * x457 + x2120 * x459
    x2122 = x2118 * x576 + x2119 * x464 + x2120 * x466
    x2123 = x1956 * x582 + x2054 * x472 + x2055 * x476
    x2124 = x1956 * x587 + x2054 * x481 + x2055 * x484
    x2125 = x1956 * x592 + x2054 * x488 + x2055 * x491
    x2126 = x149 * x1963
    x2127 = x1962 * x494 + x1963 * x496
    x2128 = x1961 * x592 + x1962 * x488 + x1963 * x491
    x2129 = sigma_kin_v_5_4 * x2116 + sigma_kin_v_6_4 * x2096 + sigma_kin_v_7_4 * x2097 + x1792 * x607
    x2130 = x123 * x1961
    x2131 = x172 * x1962
    x2132 = x154 * x1963
    x2133 = x1960 * x610 + x2130 * x569 + x2131 * x457 + x2132 * x459
    x2134 = x1960 * x618 + x2130 * x576 + x2131 * x464 + x2132 * x466
    x2135 = x1792 * x606
    x2136 = x1961 * x582 + x1962 * x472 + x1963 * x476 + x2135 * x337
    x2137 = x1588 * x1792 + x1961 * x587 + x1962 * x481 + x1963 * x484
    x2138 = x149 * x1816
    x2139 = x1814 * x487
    x2140 = x2139 * x221
    x2141 = x148 * x1816
    x2142 = x2067 * x590
    x2143 = x2066 * x283
    x2144 = x2068 * x479
    x2145 = x2069 * x482
    x2146 = x1782 * x736
    x2147 = x2146 * x471
    x2148 = x1952 * x740
    x2149 = x2148 * x765
    x2150 = x1962 * x745 + x1963 * x733
    x2151 = x2054 * x745 + x2055 * x733
    x2152 = x2047 * x745 + x2048 * x733
    x2153 = x1814 * x54
    x2154 = x2153 * x736
    x2155 = x1816 * x54
    x2156 = x2141 * x54
    x2157 = x2156 * x740
    x2158 = x2146 * x480
    x2159 = x2148 * x771
    x2160 = x2139 * x54
    x2161 = x1782 * x487
    x2162 = x2161 * x776
    x2163 = x1952 * x779
    x2164 = x2161 * x788
    x2165 = x2161 * x783
    x2166 = x1362 * x1772
    x2167 = x1772 * x787
    x2168 = x1796 * x836
    x2169 = x1782 * x804
    x2170 = x2169 * x471
    x2171 = x1650 * x1773
    x2172 = x2171 * x765
    x2173 = x2047 * x811 + x2048 * x802
    x2174 = x1961 * x815 + x1962 * x811 + x1963 * x802
    x2175 = x1956 * x815 + x2054 * x811 + x2055 * x802
    x2176 = x2161 * x821
    x2177 = x1952 * x822
    x2178 = x2067 * x54
    x2179 = x2178 * x796
    x2180 = x2153 * x804
    x2181 = x1796 * x796
    x2182 = x2138 * x54
    x2183 = x1650 * x2182
    x2184 = x1812 * x54
    x2185 = x1981 * x813
    x2186 = x2169 * x480
    x2187 = x1372 * x1773
    x2188 = x1796 * x852
    x2189 = x1796 * x850
    x2190 = x2161 * x851
    x2191 = x2161 * x854
    x2192 = x1773 * x857
    x2193 = x1373 * x1773
    x2194 = x2071 * x927
    x2195 = x1795 * x902
    x2196 = x2195 * x581
    x2197 = x1823 * x932
    x2198 = x1825 * x935
    x2199 = x2047 * x886 + x2048 * x877
    x2200 = x1956 * x889 + x2054 * x886 + x2055 * x877
    x2201 = x1690 * x1792 + x1961 * x889 + x1962 * x886 + x1963 * x877
    x2202 = x1950 * x885
    x2203 = x1952 * x896
    x2204 = x2195 * x591
    x2205 = x1798 * x885
    x2206 = x1773 * x908
    x2207 = x1810 * x54
    x2208 = x2207 * x866
    x2209 = x1820 * x866
    x2210 = x2184 * x902
    x2211 = x118 * x2210
    x2212 = x2068 * x54
    x2213 = x2212 * x880
    x2214 = x118 * x2195
    x2215 = x2069 * x54
    x2216 = x2215 * x882
    x2217 = x1823 * x880
    x2218 = x1825 * x882
    x2219 = x2066 * x54
    x2220 = x2071 * x941
    x2221 = x2071 * x940
    x2222 = x1795 * x945
    x2223 = x1795 * x947
    x2224 = x1823 * x956
    x2225 = x1823 * x951
    x2226 = x1825 * x959
    x2227 = x1825 * x955
    x2228 = x1808 * x54
    x2229 = x1811 * x54
    x2230 = x1813 * x54
    x2231 = x1815 * x54
    x2232 = x1817 * x54
    x2233 = x2228 * x40
    x2234 = x1973 * x54
    x2235 = x2207 * x968
    x2236 = x2178 * x972
    x2237 = x2212 * x980
    x2238 = 1.0 * x101 * x108
    x2239 = x2238 * (sigma_pot_4_c * x280 - sigma_pot_4_s * x281)
    x2240 = x112 * x94
    x2241 = x1202 * x2240
    x2242 = sigma_kin_v_7_4 * x1206
    x2243 = x158 * x2242
    x2244 = 8 * dq_j4
    x2245 = sigma_kin_v_7_4 * x192
    x2246 = x149 * x2245
    x2247 = x2244 * x2246
    x2248 = sigma_kin_v_6_4 * x1217
    x2249 = sigma_kin_v_6_6 * x2248 + sigma_kin_v_7_6 * x2242
    x2250 = x216 * x2249
    x2251 = x1433 * x479
    x2252 = x2251 * x296
    x2253 = x1435 * x482
    x2254 = x2253 * x303
    x2255 = sigma_kin_v_6_4 * x161
    x2256 = x1178 * x2255
    x2257 = x1179 * x2245
    x2258 = x2256 + x2257
    x2259 = dq_i6 * x2244
    x2260 = sigma_kin_v_5_4 * sigma_kin_v_5_5
    x2261 = sigma_kin_v_6_5 * x2248 + sigma_kin_v_7_5 * x2242 + x1236 * x2260
    x2262 = x2261 * x242
    x2263 = x1432 * x888
    x2264 = ddq_i1 * dq_j4
    x2265 = x2264 * x247
    x2266 = x1247 * x278
    x2267 = x1793 * x272
    x2268 = x1806 * (x1430 * x677 + x2251 * x678 + x2253 * x679 + x2263 * x674)
    x2269 = x1430 * x637 + x1884 * x2251 + x1886 * x2253 + x2263 * x635
    x2270 = sigma_kin_v_4_4**2
    x2271 = x2270 * x75
    x2272 = sigma_kin_v_5_4**2
    x2273 = x123 * x2272
    x2274 = sigma_kin_v_6_4**2
    x2275 = x172 * x2274
    x2276 = sigma_kin_v_7_4**2
    x2277 = x154 * x2276
    x2278 = x141 * x2277 + x212 * x2275 + x2271 * x244 + x2273 * x238
    x2279 = 4 * ddq_i4 * dq_j4**2
    x2280 = sigma_kin_v_5_4 * x114
    x2281 = x120 * x2280
    x2282 = x1181 * x2281
    x2283 = x1182 * x2255
    x2284 = x1184 * x2246
    x2285 = x2282 + x2283 + x2284
    x2286 = dq_i5 * x2244
    x2287 = sigma_kin_v_4_4 * x62
    x2288 = x2287 * x75
    x2289 = x123 * x2280
    x2290 = x172 * x2255
    x2291 = x154 * x2245
    x2292 = x1197 * x2288 + x1198 * x2289 + x1199 * x2290 + x1200 * x2291
    x2293 = dq_i2 * x2244
    x2294 = x2255 * x296
    x2295 = x2245 * x303
    x2296 = x1192 * x2281 + x1193 * x2294 + x1194 * x2295 + x2287 * x338
    x2297 = dq_i3 * x2244
    x2298 = x1185 * x2287
    x2299 = x1757 * x2280
    x2300 = x1187 * x2294
    x2301 = x1189 * x2295 + x2298 + x2299 + x2300
    x2302 = 8 * x61
    x2303 = (
        x130 * x2280
        + x134 * x2280
        + x179 * x2255
        + x183 * x2255
        + x197 * x2245
        + x201 * x2245
        + x2287 * x82
        + x2287 * x86
    )
    x2304 = dq_i1 * dq_j4
    x2305 = sigma_kin_v_7_4 * x2304
    x2306 = dq_j4 * x657
    x2307 = x137 * x537
    x2308 = x1848 * x2307
    x2309 = dq_j4 * x649
    x2310 = x1338 * x2307
    x2311 = x1340 * x137
    x2312 = x187 * x2311
    x2313 = dq_j4 * x574
    x2314 = x1344 * x2307
    x2315 = dq_j4 * x462
    x2316 = sigma_kin_v_7_4 * x519
    x2317 = dq_j4 * x12
    x2318 = x1348 * x537
    x2319 = dq_j4 * x506
    x2320 = x600 * x736
    x2321 = x148 * x597
    x2322 = x2321 * x740
    x2323 = x506 * x61
    x2324 = (
        ddq_i1 * x609
        + 2 * dq_i1 * dq_i4 * sigma_kin_v_4_1 * sigma_kin_v_4_4 * x282 * x605 * x66 * x73 * x78
        + 2 * dq_i1 * dq_i4 * sigma_kin_v_5_1 * sigma_kin_v_5_4 * x118 * x119 * x122 * x126 * x288 * x564
        + 2 * dq_i1 * dq_i4 * sigma_kin_v_6_1 * sigma_kin_v_6_4 * x165 * x166 * x169 * x171 * x175 * x295 * x450
        + 2 * dq_i1 * dq_i4 * sigma_kin_v_7_1 * sigma_kin_v_7_4 * x139 * x143 * x144 * x147 * x149 * x153 * x302 * x390
        - x12 * x615
        - x462 * x602
        - x574 * x604
        - x598 * x708
        - x616 * x625
        - x619 * x657
        - x623 * x649
    )
    x2325 = x1650 * x598
    x2326 = x120 * x603
    x2327 = x2326 * x796
    x2328 = x600 * x804
    x2329 = x289 * x603
    x2330 = x2319 * x598
    x2331 = x261 * x600
    x2332 = sigma_kin_v_4_1 * sigma_kin_v_4_4
    x2333 = x296 * x600
    x2334 = x303 * x597
    x2335 = x222 * x600
    x2336 = x1095 * x2264
    x2337 = dq_i2 * x2304
    x2338 = x2264 * x992
    x2339 = x2332 * x968
    x2340 = x2326 * x972
    x2341 = x2333 * x980
    x2342 = dq_i3 * x2304
    x2343 = x603 * x902
    x2344 = x2264 * x893
    x2345 = x2332 * x866
    x2346 = x118 * x2343
    x2347 = x2333 * x880
    x2348 = x2334 * x882
    x2349 = x2239 * x94
    x2350 = x111 * x1941
    x2351 = sigma_kin_v_7_4 * x1460
    x2352 = x158 * x2351
    x2353 = sigma_kin_v_6_4 * x1945
    x2354 = sigma_kin_v_6_6 * x2353 + sigma_kin_v_7_6 * x2351
    x2355 = x216 * x2354
    x2356 = x1099 * x2255
    x2357 = x222 * x2356
    x2358 = x1101 * x148
    x2359 = x2245 * x2358
    x2360 = x2357 + x2359
    x2361 = sigma_kin_v_6_5 * x2353 + sigma_kin_v_7_5 * x2351 + x1955 * x2260
    x2362 = x2361 * x242
    x2363 = x1493 * x278
    x2364 = x1964 * x272
    x2365 = dq_i2 * (x1265 * x2263 + x1430 * x2029 + x2025 * x2251 + x2027 * x2253)
    x2366 = x2271 * x617
    x2367 = x2273 * x575
    x2368 = x2275 * x463
    x2369 = x2277 * x465
    x2370 = x1077 * x2366 + x1080 * x2367 + x1083 * x2369 + x1086 * x2368
    x2371 = x1106 * x2281
    x2372 = x2356 * x261
    x2373 = x1110 * x2246
    x2374 = x2371 + x2372 + x2373
    x2375 = x1078 * x2288 + x1081 * x2289 + x1087 * x2290 + x1145 * x2291
    x2376 = 8 * x2304
    x2377 = x1128 * x2287 + x1130 * x2281 + x1131 * x2294 + x1135 * x2295
    x2378 = x1114 * x2287
    x2379 = x1763 * x2280
    x2380 = x1119 * x2294
    x2381 = x1122 * x2295 + x2378 + x2379 + x2380
    x2382 = (
        x1157 * x2288
        + x1158 * x2288
        + x1162 * x2289
        + x1164 * x2289
        + x1167 * x2290
        + x1169 * x2291
        + x1170 * x2290
        + x1172 * x2291
    )
    x2383 = dq_i2 * dq_j4
    x2384 = sigma_kin_v_7_4 * x2383
    x2385 = ddq_i2 * dq_j4
    x2386 = dq_j4 * x652
    x2387 = dq_j4 * x578
    x2388 = dq_j4 * x468
    x2389 = dq_j4 * x698
    x2390 = x1340 * x1349
    x2391 = x1489 * x736
    x2392 = x148 * x1491
    x2393 = x2392 * x740
    x2394 = x399 * x61
    x2395 = x1489 * x487
    x2396 = (
        ddq_i2 * x1583
        + 2 * dq_i2 * dq_i4 * sigma_kin_v_4_2 * sigma_kin_v_4_4 * x282 * x605 * x66 * x73 * x78
        + 2 * dq_i2 * dq_i4 * sigma_kin_v_5_2 * sigma_kin_v_5_4 * x118 * x119 * x122 * x126 * x288 * x564
        + 2 * dq_i2 * dq_i4 * sigma_kin_v_6_2 * sigma_kin_v_6_4 * x165 * x166 * x169 * x171 * x175 * x295 * x450
        + 2 * dq_i2 * dq_i4 * sigma_kin_v_7_2 * sigma_kin_v_7_4 * x139 * x143 * x144 * x147 * x149 * x153 * x302 * x390
        - x1579 * x670
        - x1580 * x468
        - x1581 * x578
        - x1584 * x657
        - x1585 * x698
        - x1587 * x652
        - x1589 * x620
    )
    x2397 = x1579 * x1650
    x2398 = dq_j4 * x399
    x2399 = x120 * x1487
    x2400 = x2399 * x796
    x2401 = x1489 * x804
    x2402 = x1487 * x289
    x2403 = x1579 * x2398
    x2404 = x1489 * x296
    x2405 = x1491 * x303
    x2406 = x1099 * x1489
    x2407 = x1489 * x986
    x2408 = x1246 * x968
    x2409 = x2399 * x972
    x2410 = x2404 * x980
    x2411 = dq_i3 * x2383
    x2412 = x1489 * x885
    x2413 = x1487 * x902
    x2414 = x1246 * x866
    x2415 = x118 * x2413
    x2416 = x2404 * x880
    x2417 = x2405 * x882
    x2418 = x2041 * (sigma_pot_3_c * x327 - sigma_pot_3_s * x326)
    x2419 = sigma_kin_v_7_4 * x977
    x2420 = x158 * x2419
    x2421 = dq_i3 * sigma_kin_v_7_3 * x1209
    x2422 = sigma_kin_v_6_4 * x987
    x2423 = sigma_kin_v_6_6 * x2422 + sigma_kin_v_7_6 * x2419
    x2424 = x216 * x2423
    x2425 = dq_i3 * x1226
    x2426 = x2255 * x986
    x2427 = x222 * x2426
    x2428 = x148 * x2245
    x2429 = x2428 * x997
    x2430 = x2427 + x2429
    x2431 = sigma_kin_v_6_5 * x2422 + sigma_kin_v_7_5 * x2419 + x2260 * x990
    x2432 = x242 * x2431
    x2433 = dq_i3 * x1243
    x2434 = x1733 * x278
    x2435 = x2058 * x272
    x2436 = dq_i1 * x729
    x2437 = dq_i3 * x2436
    x2438 = dq_i3 * x1618
    x2439 = x2270 * x606
    x2440 = x120 * x2272
    x2441 = x2274 * x296
    x2442 = x2276 * x303
    x2443 = x2439 * x968 + x2440 * x989 + x2441 * x986 + x2442 * x976
    x2444 = x1272 * x654
    x2445 = x1003 * x2281
    x2446 = x2426 * x261
    x2447 = x1006 * x2246
    x2448 = x2445 + x2446 + x2447
    x2449 = x2287 * x968
    x2450 = x2281 * x972
    x2451 = x2294 * x980
    x2452 = x1027 * x2295 + x2449 * x610 + x2450 * x869 + x2451 * x457
    x2453 = x1038 * x2295 + x2449 * x618 + x2450 * x920 + x2451 * x464
    x2454 = x2287 * x606
    x2455 = x1010 * x2454
    x2456 = x1742 * x2280
    x2457 = x1014 * x2294
    x2458 = x1018 * x2295 + x2455 + x2456 + x2457
    x2459 = (
        x1048 * x2454
        + x1049 * x2454
        + x1059 * x2280
        + x1066 * x2255
        + x1069 * x2245
        + x2042 * x2280
        + x2045 * x2255
        + x2046 * x2245
    )
    x2460 = dq_i3 * dq_j4
    x2461 = sigma_kin_v_7_4 * x2460
    x2462 = ddq_i3 * dq_j4
    x2463 = x405 * x504
    x2464 = dq_j4 * x585
    x2465 = dq_j4 * x478
    x2466 = dq_j4 * x655
    x2467 = x1351 * x406
    x2468 = x1962 * x736
    x2469 = x148 * x1963
    x2470 = x2469 * x740
    x2471 = x405 * x61
    x2472 = x1962 * x487
    x2473 = (
        ddq_i3 * x2129
        + 2 * dq_i3 * dq_i4 * sigma_kin_v_4_3 * sigma_kin_v_4_4 * x282 * x605 * x66 * x73 * x78
        + 2 * dq_i3 * dq_i4 * sigma_kin_v_5_3 * sigma_kin_v_5_4 * x118 * x119 * x122 * x126 * x288 * x564
        + 2 * dq_i3 * dq_i4 * sigma_kin_v_6_3 * sigma_kin_v_6_4 * x165 * x166 * x169 * x171 * x175 * x295 * x450
        + 2 * dq_i3 * dq_i4 * sigma_kin_v_7_3 * sigma_kin_v_7_4 * x139 * x143 * x144 * x147 * x149 * x153 * x302 * x390
        - x2126 * x631
        - x2127 * x478
        - x2128 * x585
        - x2133 * x649
        - x2134 * x652
        - x2136 * x655
        - x2137 * x624
    )
    x2474 = x1650 * x2126
    x2475 = dq_j4 * x405
    x2476 = x120 * x1961
    x2477 = x2476 * x796
    x2478 = x1962 * x804
    x2479 = x1961 * x289
    x2480 = x2126 * x2475
    x2481 = x1962 * x296
    x2482 = x1963 * x303
    x2483 = x1099 * x1962
    x2484 = x1962 * x986
    x2485 = x1792 * x968
    x2486 = x2476 * x972
    x2487 = x2481 * x980
    x2488 = x1962 * x885
    x2489 = x1961 * x902
    x2490 = x1792 * x866
    x2491 = x118 * x2489
    x2492 = x2481 * x880
    x2493 = x2482 * x882
    x2494 = x1458 * x2238
    x2495 = sigma_kin_v_7_4 * x158
    x2496 = x2495 * x877
    x2497 = x163 * x296
    x2498 = x194 * x303
    x2499 = sigma_kin_v_6_4 * sigma_kin_v_6_6
    x2500 = sigma_kin_v_7_4 * sigma_kin_v_7_6
    x2501 = x2499 * x886 + x2500 * x877
    x2502 = x216 * x2501
    x2503 = x2255 * x885
    x2504 = x222 * x2503
    x2505 = x2428 * x896
    x2506 = x2504 + x2505
    x2507 = sigma_kin_v_6_4 * sigma_kin_v_6_5
    x2508 = sigma_kin_v_7_4 * sigma_kin_v_7_5
    x2509 = x2260 * x889 + x2507 * x886 + x2508 * x877
    x2510 = x242 * x2509
    x2511 = x1692 * x278
    x2512 = x2201 * x272
    x2513 = x2439 * x283
    x2514 = x2441 * x479
    x2515 = x2442 * x482
    x2516 = x2272 * x889 + x2513 * x865 + x2514 * x879 + x2515 * x875
    x2517 = x2280 * x902
    x2518 = x2517 * x591
    x2519 = x2503 * x261
    x2520 = x2246 * x908
    x2521 = x2518 + x2519 + x2520
    x2522 = x2287 * x866
    x2523 = x118 * x2517
    x2524 = x2294 * x880
    x2525 = x2295 * x882
    x2526 = x2522 * x610 + x2523 * x869 + x2524 * x457 + x2525 * x459
    x2527 = x2522 * x618 + x2523 * x920 + x2524 * x464 + x2525 * x466
    x2528 = x2294 * x932 + x2295 * x935 + x2454 * x927 + x2517 * x581
    x2529 = x1527 * x391
    x2530 = x408 * x507
    x2531 = x137 * x146
    x2532 = x142 * x2082
    x2533 = x138 * x1530
    x2534 = x137 * x408
    x2535 = x263 * x413
    x2536 = x1534 * x391
    x2537 = x1535 * x391
    x2538 = x391 * x537
    x2539 = x2454 * x941
    x2540 = x2454 * x940
    x2541 = x2280 * x945
    x2542 = x2280 * x947
    x2543 = x2294 * x956
    x2544 = x2294 * x951
    x2545 = x2295 * x959
    x2546 = x2295 * x955
    x2547 = x2539 + x2540 + x2541 + x2542 + x2543 + x2544 + x2545 + x2546
    x2548 = x149 * x2500
    x2549 = sigma_kin_v_6_4 * x452
    x2550 = sigma_kin_v_7_4 * x454
    x2551 = sigma_kin_v_6_6 * x2549 + sigma_kin_v_7_6 * x2550
    x2552 = x172 * x2499
    x2553 = x154 * x2500
    x2554 = x2552 * x457 + x2553 * x459
    x2555 = x2552 * x464 + x2553 * x466
    x2556 = x2499 * x472 + x2500 * x476
    x2557 = x2499 * x481 + x2500 * x484
    x2558 = x2499 * x488 + x2500 * x491
    x2559 = x2499 * x494 + x2500 * x496
    x2560 = x391 * x411
    x2561 = x1349 * x2560
    x2562 = x184 + x185 + x186 + x189 + x190
    x2563 = 2 * x187 + x188 + x2562
    x2564 = x2563 * x504
    x2565 = x137 * x2564
    x2566 = x391 * x415
    x2567 = 2 * x248
    x2568 = x1567 + x533 + x536 + x545 + x554
    x2569 = sigma_kin_v_7_4 * x556
    x2570 = x149 * x2508
    x2571 = x2507 * x494 + x2508 * x496
    x2572 = sigma_kin_v_6_5 * x2549 + sigma_kin_v_7_5 * x2550 + x2260 * x566
    x2573 = x123 * x2260
    x2574 = x172 * x2507
    x2575 = x154 * x2508
    x2576 = x2573 * x569 + x2574 * x457 + x2575 * x459
    x2577 = x2573 * x576 + x2574 * x464 + x2575 * x466
    x2578 = x2260 * x582 + x2507 * x472 + x2508 * x476
    x2579 = x2260 * x587 + x2507 * x481 + x2508 * x484
    x2580 = x2260 * x592 + x2507 * x488 + x2508 * x491
    x2581 = x149 * x2276
    x2582 = x2274 * x487
    x2583 = x221 * x2582
    x2584 = x148 * x2276
    x2585 = x2440 * x590
    x2586 = x2255 * x736
    x2587 = x2586 * x480
    x2588 = x302 * x409
    x2589 = x2428 * x740
    x2590 = x2507 * x745 + x2508 * x733
    x2591 = x2499 * x745 + x2500 * x733
    x2592 = x2274 * x61
    x2593 = x2592 * x736
    x2594 = x2276 * x61
    x2595 = x2584 * x61
    x2596 = x2595 * x740
    x2597 = x2582 * x61
    x2598 = x2255 * x487
    x2599 = x2598 * x776
    x2600 = x2428 * x779
    x2601 = x2598 * x788
    x2602 = x2598 * x783
    x2603 = x1362 * x2245
    x2604 = x2245 * x787
    x2605 = x1669 * x2280
    x2606 = x2255 * x804
    x2607 = x2606 * x480
    x2608 = x2499 * x811 + x2500 * x802
    x2609 = x2260 * x815 + x2507 * x811 + x2508 * x802
    x2610 = x2598 * x821
    x2611 = x2428 * x822
    x2612 = x2440 * x61
    x2613 = x2612 * x796
    x2614 = x2592 * x804
    x2615 = x2281 * x796
    x2616 = x2581 * x61
    x2617 = x1650 * x2616
    x2618 = x1650 * x2246
    x2619 = x2272 * x61
    x2620 = x2281 * x852
    x2621 = x2281 * x850
    x2622 = x2598 * x851
    x2623 = x2598 * x854
    x2624 = x2246 * x857
    x2625 = x1373 * x2246
    x2626 = x2295 * x2588
    x2627 = x2271 * x61
    x2628 = x2273 * x61
    x2629 = x2275 * x61
    x2630 = x2277 * x61
    x2631 = x2270 * x61
    x2632 = x2441 * x61
    x2633 = x2442 * x61
    x2634 = x2631 * x968
    x2635 = x2612 * x972
    x2636 = x2632 * x980
    x2637 = x2439 * x61
    x2638 = x2631 * x866
    x2639 = x2619 * x902
    x2640 = x118 * x2639
    x2641 = x2632 * x880
    x2642 = x2633 * x882
    x2643 = sigma_pot_5_c * x251 - sigma_pot_5_s * x252
    x2644 = x112 * x2643
    x2645 = x2238 * x87
    x2646 = sigma_kin_v_7_5 * x158
    x2647 = x1206 * x2646
    x2648 = x1212 * x149
    x2649 = 8 * dq_j5
    x2650 = sigma_kin_v_7_5 * x192
    x2651 = x149 * x2650
    x2652 = x2649 * x2651
    x2653 = sigma_kin_v_6_5 * sigma_kin_v_6_6
    x2654 = sigma_kin_v_7_5 * sigma_kin_v_7_6
    x2655 = x1206 * x2654 + x1217 * x2653
    x2656 = x216 * x2655
    x2657 = x1427 * x487
    x2658 = x1428 * x800
    x2659 = x149 * x2658
    x2660 = sigma_kin_v_6_5 * x161
    x2661 = x1178 * x2660
    x2662 = x1179 * x2650
    x2663 = x2661 + x2662
    x2664 = dq_i6 * x2649
    x2665 = ddq_i1 * dq_j5
    x2666 = x241 * x2665
    x2667 = x1238 * x278
    x2668 = x1788 * x272
    x2669 = x2261 * x248
    x2670 = x1424 * x590
    x2671 = x1806 * (x1427 * x672 + x2658 * x669 + x2670 * x675)
    x2672 = x1427 * x633 + x1871 * x2670 + x2658 * x630
    x2673 = x1427 * x601 + x2326 * x2670 + x2658 * x598
    x2674 = sigma_kin_v_5_5**2
    x2675 = x123 * x2674
    x2676 = sigma_kin_v_6_5**2
    x2677 = x172 * x2676
    x2678 = sigma_kin_v_7_5**2
    x2679 = x154 * x2678
    x2680 = x141 * x2679 + x212 * x2677 + x238 * x2675
    x2681 = 4 * ddq_i5 * dq_j5**2
    x2682 = sigma_kin_v_5_5 * x114
    x2683 = x123 * x2682
    x2684 = x172 * x2660
    x2685 = x154 * x2650
    x2686 = x1198 * x2683 + x1199 * x2684 + x1200 * x2685
    x2687 = dq_i2 * x2649
    x2688 = x120 * x2682
    x2689 = x2660 * x296
    x2690 = x2650 * x303
    x2691 = x1192 * x2688 + x1193 * x2689 + x1194 * x2690
    x2692 = dq_i3 * x2649
    x2693 = x1187 * x2689 + x1189 * x2690 + x1757 * x2682
    x2694 = dq_i4 * x2649
    x2695 = x1181 * x2688
    x2696 = x1182 * x2660
    x2697 = x1184 * x2651
    x2698 = x2695 + x2696 + x2697
    x2699 = 8 * x113
    x2700 = x130 * x2682 + x134 * x2682 + x179 * x2660 + x183 * x2660 + x197 * x2650 + x201 * x2650
    x2701 = dq_i1 * dq_j5
    x2702 = sigma_kin_v_7_5 * x2701
    x2703 = dq_j5 * x657
    x2704 = x137 * x540
    x2705 = x1848 * x2704
    x2706 = dq_j5 * x649
    x2707 = x1338 * x2704
    x2708 = dq_j5 * x616
    x2709 = x1341 * x2704
    x2710 = x1343 * x137
    x2711 = x188 * x2710
    x2712 = dq_j5 * x462
    x2713 = sigma_kin_v_7_5 * x519
    x2714 = dq_j5 * x12
    x2715 = x1348 * x540
    x2716 = dq_j5 * x506
    x2717 = x562 * x736
    x2718 = x148 * x558
    x2719 = x2718 * x740
    x2720 = x487 * x562
    x2721 = x113 * x506
    x2722 = (
        ddq_i1 * x568
        + 2 * dq_i1 * dq_i5 * sigma_kin_v_5_1 * sigma_kin_v_5_5 * x118 * x120 * x121 * x126 * x253 * x564
        + 2 * dq_i1 * dq_i5 * sigma_kin_v_6_1 * sigma_kin_v_6_5 * x165 * x167 * x168 * x171 * x175 * x259 * x450
        + 2 * dq_i1 * dq_i5 * sigma_kin_v_7_1 * sigma_kin_v_7_5 * x139 * x143 * x145 * x146 * x149 * x153 * x263 * x390
        - x12 * x573
        - x462 * x563
        - x559 * x708
        - x574 * x593
        - x577 * x657
        - x584 * x649
        - x588 * x616
    )
    x2723 = x2716 * x559
    x2724 = x120 * x583
    x2725 = x296 * x562
    x2726 = x303 * x558
    x2727 = x1099 * x562
    x2728 = x1093 * x2665
    x2729 = dq_i2 * x2701
    x2730 = x562 * x986
    x2731 = x2665 * x991
    x2732 = x2724 * x972
    x2733 = x2725 * x980
    x2734 = dq_i3 * x2701
    x2735 = x562 * x885
    x2736 = x2665 * x890
    x2737 = x583 * x902
    x2738 = x118 * x2737
    x2739 = x2725 * x880
    x2740 = x2726 * x882
    x2741 = dq_i4 * x2701
    x2742 = x2665 * x818
    x2743 = x2724 * x796
    x2744 = x562 * x804
    x2745 = x1650 * x559
    x2746 = x2643 * x2645
    x2747 = x1460 * x2646
    x2748 = x1460 * x2654 + x1945 * x2653
    x2749 = x216 * x2748
    x2750 = x1099 * x2660
    x2751 = x222 * x2750
    x2752 = x2358 * x2650
    x2753 = x2751 + x2752
    x2754 = x1484 * x278
    x2755 = x1957 * x272
    x2756 = x2361 * x248
    x2757 = dq_i2 * (x1427 * x2010 + x1590 * x2658 + x2013 * x2670)
    x2758 = x1427 * x2395 + x1579 * x2658 + x2399 * x2670
    x2759 = x2675 * x575
    x2760 = x2677 * x463
    x2761 = x2679 * x465
    x2762 = x1080 * x2759 + x1083 * x2761 + x1086 * x2760
    x2763 = x1081 * x2683 + x1087 * x2684 + x1145 * x2685
    x2764 = 8 * x2701
    x2765 = x1130 * x2688 + x1131 * x2689 + x1135 * x2690
    x2766 = x1119 * x2689 + x1122 * x2690 + x1763 * x2682
    x2767 = x1106 * x2688
    x2768 = x261 * x2750
    x2769 = x1110 * x2651
    x2770 = x2767 + x2768 + x2769
    x2771 = x1162 * x2683 + x1164 * x2683 + x1167 * x2684 + x1169 * x2685 + x1170 * x2684 + x1172 * x2685
    x2772 = dq_i2 * dq_j5
    x2773 = sigma_kin_v_7_5 * x2772
    x2774 = ddq_i2 * dq_j5
    x2775 = dq_j5 * x652
    x2776 = dq_j5 * x620
    x2777 = dq_j5 * x468
    x2778 = dq_j5 * x698
    x2779 = x1343 * x1349
    x2780 = x1480 * x736
    x2781 = x148 * x1482
    x2782 = x2781 * x740
    x2783 = x1480 * x487
    x2784 = x113 * x399
    x2785 = (
        ddq_i2 * x1573
        + 2 * dq_i2 * dq_i5 * sigma_kin_v_5_2 * sigma_kin_v_5_5 * x118 * x120 * x121 * x126 * x253 * x564
        + 2 * dq_i2 * dq_i5 * sigma_kin_v_6_2 * sigma_kin_v_6_5 * x165 * x167 * x168 * x171 * x175 * x259 * x450
        + 2 * dq_i2 * dq_i5 * sigma_kin_v_7_2 * sigma_kin_v_7_5 * x139 * x143 * x145 * x146 * x149 * x153 * x263 * x390
        - x1570 * x670
        - x1571 * x468
        - x1574 * x657
        - x1575 * x698
        - x1576 * x652
        - x1577 * x620
        - x1578 * x578
    )
    x2786 = dq_j5 * x1570 * x399
    x2787 = x120 * x1478
    x2788 = x1480 * x296
    x2789 = x1482 * x303
    x2790 = x1099 * x1480
    x2791 = x1480 * x986
    x2792 = x2787 * x972
    x2793 = x2788 * x980
    x2794 = dq_i3 * x2772
    x2795 = x1480 * x885
    x2796 = x1478 * x902
    x2797 = x118 * x2796
    x2798 = x2788 * x880
    x2799 = x2789 * x882
    x2800 = dq_i4 * x2772
    x2801 = x2787 * x796
    x2802 = x1480 * x804
    x2803 = x1570 * x1650
    x2804 = x2646 * x977
    x2805 = x2653 * x987 + x2654 * x977
    x2806 = x216 * x2805
    x2807 = x2660 * x986
    x2808 = x222 * x2807
    x2809 = x148 * x997
    x2810 = x2650 * x2809
    x2811 = x2808 + x2810
    x2812 = x1731 * x278
    x2813 = x2056 * x272
    x2814 = x2431 * x248
    x2815 = x1427 * x2472 + x2126 * x2658 + x2476 * x2670
    x2816 = dq_i3 * x1252
    x2817 = x120 * x2674
    x2818 = x2676 * x296
    x2819 = x2678 * x303
    x2820 = x2817 * x989 + x2818 * x986 + x2819 * x976
    x2821 = x2688 * x972
    x2822 = x2689 * x980
    x2823 = x1027 * x2690 + x2821 * x869 + x2822 * x457
    x2824 = x1038 * x2690 + x2821 * x920 + x2822 * x464
    x2825 = x1014 * x2689 + x1018 * x2690 + x1742 * x2682
    x2826 = x1003 * x2688
    x2827 = x261 * x2807
    x2828 = x1006 * x2651
    x2829 = x2826 + x2827 + x2828
    x2830 = x1059 * x2682 + x1066 * x2660 + x1069 * x2650 + x2042 * x2682 + x2045 * x2660 + x2046 * x2650
    x2831 = dq_i3 * dq_j5
    x2832 = sigma_kin_v_7_5 * x2831
    x2833 = ddq_i3 * dq_j5
    x2834 = dq_j5 * x624
    x2835 = dq_j5 * x478
    x2836 = dq_j5 * x655
    x2837 = x2054 * x736
    x2838 = x148 * x2055
    x2839 = x2838 * x740
    x2840 = x2054 * x487
    x2841 = x113 * x405
    x2842 = (
        ddq_i3 * x2117
        + 2 * dq_i3 * dq_i5 * sigma_kin_v_5_3 * sigma_kin_v_5_5 * x118 * x120 * x121 * x126 * x253 * x564
        + 2 * dq_i3 * dq_i5 * sigma_kin_v_6_3 * sigma_kin_v_6_5 * x165 * x167 * x168 * x171 * x175 * x259 * x450
        + 2 * dq_i3 * dq_i5 * sigma_kin_v_7_3 * sigma_kin_v_7_5 * x139 * x143 * x145 * x146 * x149 * x153 * x263 * x390
        - x2114 * x631
        - x2115 * x478
        - x2121 * x649
        - x2122 * x652
        - x2123 * x655
        - x2124 * x624
        - x2125 * x585
    )
    x2843 = dq_j5 * x2114 * x405
    x2844 = x2054 * x296
    x2845 = x2055 * x303
    x2846 = x120 * x1956
    x2847 = x2054 * x222
    x2848 = x2846 * x972
    x2849 = x2844 * x980
    x2850 = x2054 * x261
    x2851 = x1956 * x902
    x2852 = x118 * x2851
    x2853 = x2844 * x880
    x2854 = x2845 * x882
    x2855 = dq_i4 * x2831
    x2856 = x2846 * x796
    x2857 = x2054 * x804
    x2858 = x1650 * x2114
    x2859 = sigma_pot_4_c * x281 - sigma_pot_4_s * x280
    x2860 = x2646 * x877
    x2861 = dq_i4 * sigma_kin_v_7_4 * x1209
    x2862 = x2653 * x886 + x2654 * x877
    x2863 = x216 * x2862
    x2864 = dq_i4 * dq_i6
    x2865 = x1225 * x2864
    x2866 = x2660 * x894
    x2867 = x148 * x2650
    x2868 = x2867 * x896
    x2869 = x2866 + x2868
    x2870 = x1689 * x278
    x2871 = x2200 * x272
    x2872 = x248 * x2509
    x2873 = dq_i4 * x2436
    x2874 = dq_i2 * x1618
    x2875 = dq_i4 * x2874
    x2876 = dq_i4 * x1273
    x2877 = x2818 * x479
    x2878 = x2819 * x482
    x2879 = x2674 * x889 + x2877 * x879 + x2878 * x875
    x2880 = x1251 * x626
    x2881 = x2682 * x902
    x2882 = x118 * x2881
    x2883 = x2689 * x880
    x2884 = x2690 * x882
    x2885 = x2882 * x869 + x2883 * x457 + x2884 * x459
    x2886 = x2882 * x920 + x2883 * x464 + x2884 * x466
    x2887 = x2689 * x932 + x2690 * x935 + x2881 * x581
    x2888 = x2881 * x591
    x2889 = x2660 * x904
    x2890 = x2651 * x908 + x2888 + x2889
    x2891 = x2682 * x945 + x2682 * x947 + x2689 * x951 + x2689 * x956 + x2690 * x955 + x2690 * x959
    x2892 = dq_j5 * sigma_kin_v_7_5
    x2893 = dq_i4 * x530
    x2894 = ddq_i4 * dq_j5
    x2895 = dq_i4 * x527
    x2896 = dq_j5 * x486
    x2897 = dq_j5 * x627
    x2898 = dq_j5 * x409
    x2899 = x2507 * x736
    x2900 = x148 * x2508
    x2901 = x2900 * x740
    x2902 = x2507 * x487
    x2903 = x113 * x409
    x2904 = (
        ddq_i4 * x2572
        + 2 * dq_i4 * dq_i5 * sigma_kin_v_5_4 * sigma_kin_v_5_5 * x118 * x120 * x121 * x126 * x253 * x564
        + 2 * dq_i4 * dq_i5 * sigma_kin_v_6_4 * sigma_kin_v_6_5 * x165 * x167 * x168 * x171 * x175 * x259 * x450
        + 2 * dq_i4 * dq_i5 * sigma_kin_v_7_4 * sigma_kin_v_7_5 * x139 * x143 * x145 * x146 * x149 * x153 * x263 * x390
        - x2570 * x599
        - x2571 * x486
        - x2576 * x616
        - x2577 * x620
        - x2578 * x624
        - x2579 * x627
        - x2580 * x589
    )
    x2905 = x2570 * x2898
    x2906 = x120 * x2260
    x2907 = x2507 * x296
    x2908 = x2508 * x303
    x2909 = x222 * x2507
    x2910 = x2906 * x972
    x2911 = x2907 * x980
    x2912 = x2260 * x902
    x2913 = x118 * x2912
    x2914 = x2907 * x880
    x2915 = x2908 * x882
    x2916 = x2906 * x796
    x2917 = x2507 * x804
    x2918 = x1650 * x2570
    x2919 = x112 * x87
    x2920 = x116 * x120
    x2921 = x2646 * x802
    x2922 = x163 * x487
    x2923 = x149 * x194
    x2924 = x1650 * x2651
    x2925 = x2653 * x811 + x2654 * x802
    x2926 = x216 * x2925
    x2927 = x2660 * x487
    x2928 = x2927 * x821
    x2929 = x2867 * x822
    x2930 = x2928 + x2929
    x2931 = x1656 * x278
    x2932 = x2175 * x272
    x2933 = x248 * x2609
    x2934 = x2817 * x590
    x2935 = x2676 * x487
    x2936 = x149 * x2678
    x2937 = x2934 * x795 + x2935 * x804 + x2936 * x801
    x2938 = x2688 * x796
    x2939 = x2660 * x804
    x2940 = x2924 * x828 + x2938 * x569 + x2939 * x750
    x2941 = x2924 * x759 + x2938 * x576 + x2939 * x757
    x2942 = x2688 * x836 + x2924 * x765 + x2939 * x471
    x2943 = x1372 * x2651 + x1669 * x2682 + x2939 * x480
    x2944 = x137 * x144
    x2945 = x391 * x540
    x2946 = x2688 * x852
    x2947 = x2688 * x850
    x2948 = x2927 * x851
    x2949 = x2927 * x854
    x2950 = x2651 * x857
    x2951 = x1373 * x2651
    x2952 = x2946 + x2947 + x2948 + x2949 + x2950 + x2951
    x2953 = x149 * x2654
    x2954 = x2653 * x452 + x2654 * x454
    x2955 = x172 * x2653
    x2956 = x154 * x2654
    x2957 = x2955 * x457 + x2956 * x459
    x2958 = x2955 * x464 + x2956 * x466
    x2959 = x2653 * x472 + x2654 * x476
    x2960 = x2653 * x481 + x2654 * x484
    x2961 = x2653 * x488 + x2654 * x491
    x2962 = x2653 * x494 + x2654 * x496
    x2963 = x1349 * x2566
    x2964 = x187 + 2 * x188 + x2562
    x2965 = x2964 * x504
    x2966 = x137 * x2965
    x2967 = 2 * x242
    x2968 = sigma_kin_v_7_5 * x556
    x2969 = x221 * x2935
    x2970 = x148 * x2678
    x2971 = x2927 * x776
    x2972 = x2867 * x779
    x2973 = x2653 * x745 + x2654 * x733
    x2974 = x113 * x2676
    x2975 = x2974 * x736
    x2976 = x113 * x2678
    x2977 = x2660 * x736
    x2978 = x113 * x2970
    x2979 = x2978 * x740
    x2980 = x2867 * x740
    x2981 = x113 * x2935
    x2982 = x2927 * x788
    x2983 = x2927 * x783
    x2984 = x1362 * x2650
    x2985 = x2650 * x787
    x2986 = x113 * x2675
    x2987 = x113 * x2677
    x2988 = x113 * x2679
    x2989 = x113 * x2817
    x2990 = x113 * x2818
    x2991 = x113 * x2819
    x2992 = x113 * x2674
    x2993 = x113 * x2936
    x2994 = x2989 * x972
    x2995 = x2990 * x980
    x2996 = x2992 * x902
    x2997 = x118 * x2996
    x2998 = x2990 * x880
    x2999 = x2991 * x882
    x3000 = x2989 * x796
    x3001 = x2974 * x804
    x3002 = x1650 * x2993
    x3003 = x108 * (sigma_pot_6_c * x218 - sigma_pot_6_s * x219)
    x3004 = 1.0 * x87
    x3005 = x2241 * x3004
    x3006 = sigma_kin_v_7_6 * x158
    x3007 = x1206 * x3006
    x3008 = 8 * dq_j6
    x3009 = sigma_kin_v_7_6 * x192
    x3010 = x149 * x3009
    x3011 = x3008 * x3010
    x3012 = ddq_i1 * dq_j6
    x3013 = x215 * x3012
    x3014 = x1219 * x278
    x3015 = x1776 * x272
    x3016 = x2249 * x248
    x3017 = x242 * x2655
    x3018 = x1422 * x731
    x3019 = x148 * x3018
    x3020 = x1806 * (x1419 * x672 + x3019 * x668)
    x3021 = x1419 * x633 + x3019 * x629
    x3022 = x1419 * x601 + x3019 * x597
    x3023 = x1419 * x2720 + x3019 * x558
    x3024 = sigma_kin_v_6_6**2
    x3025 = x172 * x3024
    x3026 = sigma_kin_v_7_6**2
    x3027 = x154 * x3026
    x3028 = x141 * x3027 + x212 * x3025
    x3029 = 4 * ddq_i6 * dq_j6**2
    x3030 = sigma_kin_v_6_6 * x161
    x3031 = x172 * x3030
    x3032 = x154 * x3009
    x3033 = x1199 * x3031 + x1200 * x3032
    x3034 = dq_i2 * x3008
    x3035 = x296 * x3030
    x3036 = x3009 * x303
    x3037 = x1193 * x3035 + x1194 * x3036
    x3038 = dq_i3 * x3008
    x3039 = x1187 * x3035 + x1189 * x3036
    x3040 = dq_i4 * x3008
    x3041 = x1182 * x3030 + x1184 * x3010
    x3042 = dq_i5 * x3008
    x3043 = x1178 * x3030
    x3044 = x1179 * x3009
    x3045 = x3043 + x3044
    x3046 = 8 * x160
    x3047 = x179 * x3030 + x183 * x3030 + x197 * x3009 + x201 * x3009
    x3048 = dq_i1 * dq_j6
    x3049 = sigma_kin_v_7_6 * x3048
    x3050 = dq_j6 * x657
    x3051 = x137 * x543
    x3052 = x1848 * x3051
    x3053 = dq_j6 * x649
    x3054 = x1338 * x3051
    x3055 = dq_j6 * x616
    x3056 = x1341 * x3051
    x3057 = dq_j6 * x574
    x3058 = x1344 * x3051
    x3059 = x189 * x519
    x3060 = dq_j6 * x12
    x3061 = x1348 * x543
    x3062 = x192 * x519
    x3063 = (
        ddq_i1 * x456
        + 2 * dq_i1 * dq_i6 * sigma_kin_v_6_1 * sigma_kin_v_6_6 * x165 * x167 * x169 * x170 * x175 * x220 * x450
        + 2 * dq_i1 * dq_i6 * sigma_kin_v_7_1 * sigma_kin_v_7_6 * x139 * x143 * x145 * x147 * x148 * x153 * x230 * x390
        - x12 * x461
        - x443 * x708
        - x462 * x497
        - x467 * x657
        - x477 * x649
        - x485 * x616
        - x492 * x574
    )
    x3064 = dq_j6 * x506
    x3065 = x3064 * x443
    x3066 = x296 * x473
    x3067 = x303 * x442
    x3068 = x160 * x506
    x3069 = x1091 * x3012
    x3070 = x222 * x473
    x3071 = dq_i2 * x3048
    x3072 = x3012 * x988
    x3073 = x3066 * x980
    x3074 = dq_i3 * x3048
    x3075 = x3012 * x887
    x3076 = x3066 * x880
    x3077 = x3067 * x882
    x3078 = x148 * x442
    x3079 = dq_i4 * x3048
    x3080 = x1650 * x443
    x3081 = x3012 * x812
    x3082 = x473 * x804
    x3083 = x473 * x487
    x3084 = dq_i5 * x3048
    x3085 = x3012 * x747
    x3086 = x473 * x736
    x3087 = x3078 * x740
    x3088 = x3004 * x94
    x3089 = x3003 * x3088
    x3090 = x1460 * x3006
    x3091 = x1471 * x278
    x3092 = x1947 * x272
    x3093 = x2354 * x248
    x3094 = x242 * x2748
    x3095 = dq_i2 * (x1267 * x3019 + x1419 * x2010)
    x3096 = x1419 * x2395 + x1491 * x3019
    x3097 = x1419 * x2783 + x1482 * x3019
    x3098 = x3025 * x463
    x3099 = x3027 * x465
    x3100 = x1083 * x3099 + x1086 * x3098
    x3101 = x1087 * x3031 + x1145 * x3032
    x3102 = 8 * x3048
    x3103 = x1131 * x3035 + x1135 * x3036
    x3104 = x1119 * x3035 + x1122 * x3036
    x3105 = x1107 * x3030 + x1110 * x3010
    x3106 = x1100 * x3030
    x3107 = x2358 * x3009
    x3108 = x3106 + x3107
    x3109 = x1167 * x3031 + x1169 * x3032 + x1170 * x3031 + x1172 * x3032
    x3110 = dq_i2 * dq_j6
    x3111 = sigma_kin_v_7_6 * x3110
    x3112 = ddq_i2 * dq_j6
    x3113 = dq_j6 * x652
    x3114 = dq_j6 * x620
    x3115 = dq_j6 * x578
    x3116 = dq_j6 * x698
    x3117 = (
        ddq_i2 * x1552
        + 2 * dq_i2 * dq_i6 * sigma_kin_v_6_2 * sigma_kin_v_6_6 * x165 * x167 * x169 * x170 * x175 * x220 * x450
        + 2 * dq_i2 * dq_i6 * sigma_kin_v_7_2 * sigma_kin_v_7_6 * x139 * x143 * x145 * x147 * x148 * x153 * x230 * x390
        - x1549 * x670
        - x1553 * x657
        - x1554 * x698
        - x1555 * x652
        - x1556 * x620
        - x1557 * x578
        - x1558 * x468
    )
    x3118 = dq_j6 * x399
    x3119 = x1549 * x3118
    x3120 = x1467 * x296
    x3121 = x1469 * x303
    x3122 = x160 * x399
    x3123 = x1467 * x222
    x3124 = x3120 * x980
    x3125 = dq_i3 * x3110
    x3126 = x3120 * x880
    x3127 = x3121 * x882
    x3128 = x1469 * x148
    x3129 = dq_i4 * x3110
    x3130 = x1549 * x1650
    x3131 = x1467 * x804
    x3132 = x1467 * x487
    x3133 = dq_i5 * x3110
    x3134 = x1467 * x736
    x3135 = x3128 * x740
    x3136 = x3006 * x977
    x3137 = x1730 * x278
    x3138 = x2049 * x272
    x3139 = x2423 * x248
    x3140 = x242 * x2805
    x3141 = x1419 * x2472 + x1963 * x3019
    x3142 = x1419 * x2840 + x2055 * x3019
    x3143 = x296 * x3024
    x3144 = x3026 * x303
    x3145 = x3143 * x986 + x3144 * x976
    x3146 = x3035 * x980
    x3147 = x1027 * x3036 + x3146 * x457
    x3148 = x1038 * x3036 + x3146 * x464
    x3149 = x1014 * x3035 + x1018 * x3036
    x3150 = x1004 * x3030 + x1006 * x3010
    x3151 = x3030 * x995
    x3152 = x2809 * x3009
    x3153 = x3151 + x3152
    x3154 = x1066 * x3030 + x1069 * x3009 + x2045 * x3030 + x2046 * x3009
    x3155 = dq_i3 * dq_j6
    x3156 = sigma_kin_v_7_6 * x3155
    x3157 = ddq_i3 * dq_j6
    x3158 = dq_j6 * x624
    x3159 = dq_j6 * x585
    x3160 = dq_j6 * x655
    x3161 = (
        ddq_i3 * x2098
        + 2 * dq_i3 * dq_i6 * sigma_kin_v_6_3 * sigma_kin_v_6_6 * x165 * x167 * x169 * x170 * x175 * x220 * x450
        + 2 * dq_i3 * dq_i6 * sigma_kin_v_7_3 * sigma_kin_v_7_6 * x139 * x143 * x145 * x147 * x148 * x153 * x230 * x390
        - x2095 * x631
        - x2101 * x649
        - x2102 * x652
        - x2103 * x655
        - x2104 * x624
        - x2105 * x585
        - x2106 * x478
    )
    x3162 = dq_j6 * x405
    x3163 = x2095 * x3162
    x3164 = x2047 * x296
    x3165 = x2048 * x303
    x3166 = x160 * x405
    x3167 = x3164 * x980
    x3168 = x3164 * x880
    x3169 = x3165 * x882
    x3170 = x148 * x2048
    x3171 = dq_i4 * x3155
    x3172 = x1650 * x2095
    x3173 = x2047 * x804
    x3174 = x2047 * x487
    x3175 = dq_i5 * x3155
    x3176 = x2047 * x736
    x3177 = x3170 * x740
    x3178 = 1.0 * x3003
    x3179 = x1458 * x2240
    x3180 = x2859 * x3179
    x3181 = x3006 * x877
    x3182 = x148 * x896
    x3183 = x1688 * x278
    x3184 = x2199 * x272
    x3185 = x248 * x2501
    x3186 = x242 * x2862
    x3187 = dq_i4 * dq_i5
    x3188 = x3187 * (x1419 * x2902 + x2508 * x3019)
    x3189 = x3143 * x479
    x3190 = x3144 * x482
    x3191 = x3189 * x879 + x3190 * x875
    x3192 = x3035 * x880
    x3193 = x3036 * x882
    x3194 = x3192 * x457 + x3193 * x459
    x3195 = x3192 * x464 + x3193 * x466
    x3196 = x3035 * x932 + x3036 * x935
    x3197 = x3010 * x908 + x3030 * x904
    x3198 = x3030 * x894
    x3199 = x3009 * x3182
    x3200 = x3198 + x3199
    x3201 = x3035 * x951 + x3035 * x956 + x3036 * x955 + x3036 * x959
    x3202 = dq_j6 * sigma_kin_v_7_6
    x3203 = ddq_i4 * dq_j6
    x3204 = dq_j6 * x589
    x3205 = dq_j6 * x627
    x3206 = (
        ddq_i4 * x2551
        + 2 * dq_i4 * dq_i6 * sigma_kin_v_6_4 * sigma_kin_v_6_6 * x165 * x167 * x169 * x170 * x175 * x220 * x450
        + 2 * dq_i4 * dq_i6 * sigma_kin_v_7_4 * sigma_kin_v_7_6 * x139 * x143 * x145 * x147 * x148 * x153 * x230 * x390
        - x2548 * x599
        - x2554 * x616
        - x2555 * x620
        - x2556 * x624
        - x2557 * x627
        - x2558 * x589
        - x2559 * x486
    )
    x3207 = dq_j6 * x409
    x3208 = x2548 * x3207
    x3209 = x2499 * x296
    x3210 = x2500 * x303
    x3211 = x160 * x409
    x3212 = x3209 * x980
    x3213 = x3209 * x880
    x3214 = x3210 * x882
    x3215 = x148 * x2500
    x3216 = x1650 * x2548
    x3217 = x2499 * x804
    x3218 = x2499 * x487
    x3219 = dq_j6 * x3187
    x3220 = x2499 * x736
    x3221 = x3215 * x740
    x3222 = x1458 * x2919 * (sigma_pot_5_c * x252 - sigma_pot_5_s * x251)
    x3223 = x3006 * x802
    x3224 = x148 * x822
    x3225 = x1650 * x3010
    x3226 = x1653 * x278
    x3227 = x2173 * x272
    x3228 = x248 * x2608
    x3229 = x242 * x2925
    x3230 = dq_i5 * x2436
    x3231 = dq_i5 * x2874
    x3232 = dq_i5 * x1273
    x3233 = x3024 * x487
    x3234 = x149 * x3026
    x3235 = x3233 * x804 + x3234 * x801
    x3236 = x1242 * x594
    x3237 = x3030 * x804
    x3238 = x3225 * x828 + x3237 * x750
    x3239 = x3225 * x759 + x3237 * x757
    x3240 = x3225 * x765 + x3237 * x471
    x3241 = x1372 * x3010 + x3237 * x480
    x3242 = x3030 * x487
    x3243 = x3242 * x821
    x3244 = x3009 * x3224 + x3243
    x3245 = x1373 * x3010 + x3010 * x857 + x3242 * x851 + x3242 * x854
    x3246 = dq_i5 * x3202
    x3247 = ddq_i5 * dq_j6
    x3248 = dq_j6 * x595
    x3249 = (
        ddq_i5 * x2954
        + 2 * dq_i5 * dq_i6 * sigma_kin_v_6_5 * sigma_kin_v_6_6 * x165 * x167 * x169 * x170 * x175 * x220 * x450
        + 2 * dq_i5 * dq_i6 * sigma_kin_v_7_5 * sigma_kin_v_7_6 * x139 * x143 * x145 * x147 * x148 * x153 * x230 * x390
        - x2953 * x561
        - x2957 * x574
        - x2958 * x578
        - x2959 * x585
        - x2960 * x589
        - x2961 * x595
        - x2962 * x493
    )
    x3250 = dq_j6 * x413
    x3251 = x2953 * x3250
    x3252 = x2653 * x296
    x3253 = x2654 * x303
    x3254 = x160 * x413
    x3255 = x3252 * x980
    x3256 = x3252 * x880
    x3257 = x3253 * x882
    x3258 = x1650 * x2953
    x3259 = x2653 * x804
    x3260 = x2653 * x487
    x3261 = x148 * x2654
    x3262 = x2653 * x736
    x3263 = x3261 * x740
    x3264 = x3004 * x3179
    x3265 = x3006 * x733
    x3266 = x1626 * x278
    x3267 = x2152 * x272
    x3268 = x248 * x2591
    x3269 = x242 * x2973
    x3270 = x221 * x3233
    x3271 = x148 * x3026
    x3272 = x3270 * x735 + x3271 * x732
    x3273 = x3030 * x736
    x3274 = x3009 * x756 + x3273 * x750
    x3275 = x148 * x3009
    x3276 = x3273 * x757 + x3275 * x763
    x3277 = x3275 * x740
    x3278 = x3273 * x471 + x3277 * x765
    x3279 = x3273 * x480 + x3277 * x771
    x3280 = x3242 * x776 + x3275 * x779
    x3281 = x3242 * x788
    x3282 = x3242 * x783
    x3283 = x1362 * x3009
    x3284 = x3009 * x787
    x3285 = x3281 + x3282 + x3283 + x3284
    x3286 = ddq_i7 * x152
    x3287 = x1533 * x391
    x3288 = x152 * x516
    x3289 = sigma_kin_v_7_6 * x426
    x3290 = x192 * x520
    x3291 = x184 + x185 + x186 + x187 + x188 + 2 * x189 + x190
    x3292 = x3291 * x504
    x3293 = x137 * x3292
    x3294 = 2 * x216
    x3295 = sigma_kin_v_7_6 * x556
    x3296 = x160 * x3025
    x3297 = x160 * x3027
    x3298 = x160 * x3143
    x3299 = x160 * x3144
    x3300 = x160 * x3024
    x3301 = x160 * x3234
    x3302 = x160 * x3026
    x3303 = x160 * x3271
    x3304 = x3298 * x980
    x3305 = x3298 * x880
    x3306 = x3299 * x882
    x3307 = x3300 * x804
    x3308 = x1650 * x3301
    x3309 = x160 * x3233
    x3310 = x3300 * x736
    x3311 = x3303 * x740
    x3312 = sigma_pot_7_c * x502 - sigma_pot_7_s * x501
    x3313 = x101 * x3312
    x3314 = ddq_i1 * x135
    x3315 = x157 * x3314
    x3316 = 4 * x157
    x3317 = x278 * x3316
    x3318 = x272 * x3316
    x3319 = x248 * x3316
    x3320 = x1206 * x3316
    x3321 = sigma_kin_v_7_5 * x242
    x3322 = sigma_kin_v_7_6 * x216
    x3323 = x152 * x208
    x3324 = x3323 * x447
    x3325 = x1806 * x3324 * x669
    x3326 = x3324 * x630
    x3327 = x3324 * x598
    x3328 = x3324 * x559
    x3329 = x3324 * x443
    x3330 = 4 * sigma_kin_v_7_7**2
    x3331 = dq_j7**2 * x3330
    x3332 = ddq_i7 * x3331
    x3333 = 4 * x192
    x3334 = dq_i1 * x157
    x3335 = x3333 * x3334
    x3336 = x2648 * x3323
    x3337 = x136 * x192
    x3338 = dq_i2 * x157
    x3339 = x154 * x3338
    x3340 = dq_i3 * x157
    x3341 = 8 * x192
    x3342 = x303 * x3340 * x3341
    x3343 = dq_i4 * x157
    x3344 = x303 * x3343
    x3345 = x3341 * x3344
    x3346 = dq_i5 * x157
    x3347 = x149 * x3346
    x3348 = x3341 * x3347
    x3349 = dq_i6 * x157
    x3350 = x3341 * x3349
    x3351 = x149 * x3341
    x3352 = x190 * x3323
    x3353 = x3351 * x3352
    x3354 = x228 * x398
    x3355 = x228 * x404
    x3356 = x228 * x391
    x3357 = x3356 * x410
    x3358 = x3356 * x414
    x3359 = x462 * x517
    x3360 = x198 * x420
    x3361 = (
        -ddq_i1 * x137 * x138 * x142 * x144 * x146 * x148
        + x12 * x3360
        + x3354 * x657
        + x3355 * x649
        + x3357 * x616
        + x3358 * x574
        + x3359
    )
    x3362 = x153 * x524
    x3363 = x3362 * x426
    x3364 = x1168 * x516
    x3365 = x157 * x394
    x3366 = x1806 * x3365
    x3367 = ddq_i1 * x516
    x3368 = x3365 * x3367
    x3369 = x1089 * x398
    x3370 = x1089 * x3365
    x3371 = x3370 * x404
    x3372 = x1340 * x3370
    x3373 = x1343 * x3370
    x3374 = x1083 * x157
    x3375 = x152 * x3359
    x3376 = x208 * x524
    x3377 = x3376 * x506
    x3378 = x190 * x3377
    x3379 = x12 * x1347
    x3380 = x3365 * x3379
    x3381 = x208 * x547
    x3382 = x3381 * x506
    x3383 = x137 * x424
    x3384 = x1174 * x397
    x3385 = x1060 * x516
    x3386 = dq_i1 * x394
    x3387 = x157 * x3386
    x3388 = dq_i3 * x3387
    x3389 = x1850 * x975
    x3390 = x1334 * x657
    x3391 = x3365 * x975
    x3392 = x2311 * x616
    x3393 = x2710 * x574
    x3394 = x157 * x975
    x3395 = x1070 * x403
    x3396 = x516 * x953
    x3397 = dq_i4 * x3387
    x3398 = x2311 * x875
    x3399 = x3365 * x875
    x3400 = x1850 * x649
    x3401 = x157 * x875
    x3402 = x190 * x875
    x3403 = x391 * x424
    x3404 = x2531 * x962
    x3405 = x516 * x856
    x3406 = dq_i5 * x3387
    x3407 = x2710 * x799
    x3408 = x3365 * x799
    x3409 = x157 * x799
    x3410 = x2944 * x864
    x3411 = dq_i6 * x516
    x3412 = x3411 * x784
    x3413 = dq_i6 * x518
    x3414 = x157 * x730
    x3415 = x3413 * x3414
    x3416 = x3365 * x730
    x3417 = x190 * x730
    x3418 = x3288 * x794
    x3419 = x136 * x3365
    x3420 = x11 * x3365
    x3421 = x195 * x516
    x3422 = x1347 * x136
    x3423 = x136 * x157
    x3424 = x136 * x190
    x3425 = x135 * x393
    x3426 = x190 * x529
    x3427 = x157 * x504
    x3428 = x190 * x526
    x3429 = x394 * x550
    x3430 = x3088 * x3313
    x3431 = sigma_kin_v_7_2 * x3317
    x3432 = x1460 * x3316
    x3433 = x1273 * x3324
    x3434 = dq_i2 * x1590
    x3435 = x1579 * x3324
    x3436 = x1570 * x3324
    x3437 = x1549 * x3324
    x3438 = x3333 * x3339
    x3439 = x1089 * x192
    x3440 = 8 * x3334
    x3441 = x468 * x517
    x3442 = (
        -ddq_i2 * x137 * x138 * x142 * x144 * x146 * x148
        + x3354 * x698
        + x3355 * x652
        + x3357 * x620
        + x3358 * x578
        + x3360 * x657
        + x3441
    )
    x3443 = ddq_i2 * x516
    x3444 = x1850 * x652
    x3445 = x2311 * x620
    x3446 = x2710 * x578
    x3447 = x152 * x3441
    x3448 = x3376 * x399
    x3449 = x1334 * x698
    x3450 = x3381 * x399
    x3451 = dq_i2 * x394
    x3452 = x157 * x3451
    x3453 = dq_i3 * x3452
    x3454 = x3365 * x3443
    x3455 = x1347 * x657
    x3456 = x190 * x3448
    x3457 = x3365 * x3449
    x3458 = dq_i4 * x3452
    x3459 = x391 * x531
    x3460 = dq_i5 * x3452
    x3461 = x3365 * x697
    x3462 = x1083 * x3365
    x3463 = sigma_kin_v_7_1 * x3315
    x3464 = sigma_kin_v_7_3 * x3318
    x3465 = x3316 * x977
    x3466 = x2126 * x3324
    x3467 = x2114 * x3324
    x3468 = x3333 * x3340
    x3469 = x147 * x3336
    x3470 = x192 * x975
    x3471 = x303 * x3470
    x3472 = x3440 * x459
    x3473 = 8 * x3338
    x3474 = x478 * x517
    x3475 = (
        -ddq_i3 * x137 * x138 * x142 * x144 * x146 * x148
        + x3354 * x652
        + x3355 * x655
        + x3357 * x624
        + x3358 * x585
        + x3360 * x649
        + x3474
    )
    x3476 = x195 * x3386
    x3477 = x3340 * x516
    x3478 = ddq_i3 * x516
    x3479 = x1347 * x3386
    x3480 = x136 * x3479
    x3481 = x1334 * x652
    x3482 = x2311 * x624
    x3483 = x2710 * x585
    x3484 = x152 * x3474
    x3485 = x3376 * x405
    x3486 = x1850 * x655
    x3487 = x3381 * x405
    x3488 = x1168 * x3451
    x3489 = x3365 * x3478
    x3490 = x3451 * x398
    x3491 = x1089 * x3490
    x3492 = x1347 * x649
    x3493 = x190 * x3485
    x3494 = dq_i3 * x394
    x3495 = x157 * x3494
    x3496 = dq_i4 * x3495
    x3497 = x3365 * x3486
    x3498 = x391 * x534
    x3499 = dq_i5 * x3495
    x3500 = x3365 * x654
    x3501 = 1.0 * x3313
    x3502 = sigma_kin_v_7_4 * x3319
    x3503 = x3316 * x877
    x3504 = x2570 * x3187 * x3324
    x3505 = x2548 * x3324
    x3506 = x3333 * x3344
    x3507 = x192 * x875
    x3508 = x301 * x303 * x3507
    x3509 = x486 * x517
    x3510 = (
        -ddq_i4 * x137 * x138 * x142 * x144 * x146 * x148
        + x3354 * x620
        + x3355 * x624
        + x3357 * x627
        + x3358 * x589
        + x3360 * x616
        + x3509
    )
    x3511 = x3343 * x516
    x3512 = ddq_i4 * x516
    x3513 = x1334 * x620
    x3514 = x1850 * x624
    x3515 = x2710 * x589
    x3516 = x152 * x3509
    x3517 = x3376 * x409
    x3518 = x2311 * x627
    x3519 = x3381 * x409
    x3520 = x3365 * x3512
    x3521 = x1347 * x616
    x3522 = x190 * x3517
    x3523 = x1060 * x3494
    x3524 = x1850 * x3494
    x3525 = x3524 * x975
    x3526 = x3365 * x3518
    x3527 = x3187 * x3365
    x3528 = x394 * x784
    x3529 = x3528 * x516
    x3530 = x157 * x2864
    x3531 = x518 * x730
    x3532 = x3365 * x626
    x3533 = dq_i4 * x394
    x3534 = x3316 * x802
    x3535 = x2953 * x3324
    x3536 = x3333 * x3347
    x3537 = x192 * x799
    x3538 = x149 * x3537
    x3539 = x146 * x3538
    x3540 = 8 * x3539
    x3541 = x3340 * x765
    x3542 = 8 * x3343
    x3543 = x493 * x517
    x3544 = (
        -ddq_i5 * x137 * x138 * x142 * x144 * x146 * x148
        + x3354 * x578
        + x3355 * x585
        + x3357 * x589
        + x3358 * x595
        + x3360 * x574
        + x3543
    )
    x3545 = x3346 * x516
    x3546 = ddq_i5 * x516
    x3547 = x1334 * x578
    x3548 = x1850 * x585
    x3549 = x2311 * x589
    x3550 = x152 * x3543
    x3551 = x3376 * x413
    x3552 = x2710 * x595
    x3553 = x3381 * x413
    x3554 = x3365 * x3546
    x3555 = x1347 * x574
    x3556 = x190 * x3551
    x3557 = x3365 * x3552
    x3558 = dq_i5 * x394
    x3559 = x3365 * x594
    x3560 = x3316 * x733
    x3561 = x3333 * x3349
    x3562 = x394 * x730
    x3563 = x192 * x730
    x3564 = x147 * x3563
    x3565 = x148 * x3563
    x3566 = x148 * x3564
    x3567 = x499 * x517
    x3568 = (
        -ddq_i6 * x137 * x138 * x142 * x144 * x146 * x148
        + x3354 * x468
        + x3355 * x478
        + x3357 * x486
        + x3358 * x493
        + x3360 * x462
        + x3567
    )
    x3569 = x157 * x3411
    x3570 = ddq_i6 * x516
    x3571 = x1334 * x468
    x3572 = x1850 * x478
    x3573 = x2311 * x486
    x3574 = x2710 * x493
    x3575 = x3376 * x417
    x3576 = x152 * x3567
    x3577 = x3381 * x417
    x3578 = x3365 * x3570
    x3579 = x1347 * x462
    x3580 = x190 * x3575
    x3581 = x157 * x3576
    x3582 = x2864 * x3365
    x3583 = x391 * x543
    x3584 = x3558 * x856
    x3585 = x1343 * x3558
    x3586 = x137 * x3585 * x799
    x3587 = x157 * x498
    x3588 = x149 * x447
    x3589 = x152 * x552
    x3590 = x3588 * x3589
    x3591 = 4 * x3590
    x3592 = x2923 * x447
    x3593 = x152 * x528
    x3594 = x190 * x3333 * x3588
    x3595 = x3351 * x3589
    x3596 = x1415 * x3595
    x3597 = x147 * x3595
    x3598 = sigma_kin_v_7_7 * x3362
    x3599 = x422 * x524
    x3600 = x190 * x504
    x3601 = x137 * x3600
    x3602 = x190 * x514
    x3603 = x137 * x3602
    x3604 = x190 * x516
    x3605 = x136 * x516
    x3606 = x278 * x531
    x3607 = x272 * x534
    x3608 = x248 * x537
    x3609 = x242 * x540
    x3610 = x216 * x543
    x3611 = 2 * x551
    x3612 = x137 * x394
    x3613 = x3424 * x3612
    x3614 = x417 * x518
    x3615 = x423 * x524
    x3616 = x3337 * x3612
    x3617 = sigma_kin_v_7_7 * x394
    x3618 = x1083 * x516
    x3619 = x394 * x508
    x3620 = x1083 * x190
    x3621 = x1089 * x394
    x3622 = x190 * x3621
    x3623 = x1083 * x192
    x3624 = x3439 * x394
    x3625 = x137 * x3617
    x3626 = x516 * x975
    x3627 = x190 * x975
    x3628 = x3612 * x3627
    x3629 = x3470 * x3612
    x3630 = x3533 * x953
    x3631 = x516 * x875
    x3632 = x1340 * x3533
    x3633 = x3402 * x3612
    x3634 = x137 * x3507
    x3635 = x3634 * x394
    x3636 = x3617 * x391
    x3637 = x516 * x799
    x3638 = x190 * x799
    x3639 = x3612 * x3638
    x3640 = x137 * x3537
    x3641 = x3640 * x394
    x3642 = dq_i6 * x3528
    x3643 = x516 * x730
    x3644 = x190 * x3562
    x3645 = x137 * x3644
    x3646 = x3563 * x3612

    K_block_list = []
    K_block_list.append(
        4 * ddq_i1 * x10 * x325
        - dq_i2 * x202 * x389
        - dq_i3 * x202 * x352
        - dq_i4 * x202 * x308
        - dq_i5 * x202 * x268
        - dq_i6 * x202 * x237
        + x1070
        * (
            2 * dq_i1 * dq_i7 * dq_j1 * x139 * x142 * x145 * x147 * x149 * x152 * x208 * x322 * x390 * x975
            + 2
            * dq_i1
            * (
                x1021 * x1023
                + x1024 * x610
                + x1025 * x869
                + x1026 * x457
                + x1027 * x917
                + x966
                + x970
                + x974
                + x981
                + x985
            )
            + 2
            * dq_i2
            * (
                x1024 * x618
                + x1025 * x920
                + x1026 * x464
                + x1028 * x1030
                + x1032
                + x1033
                + x1034
                + x1038 * x917
                + x1039
                + x1041
            )
            + dq_i3
            * (
                x1028 * x1044
                + x1042 * x31 * x722
                + x1043 * x1047
                + x1046
                + x1048 * x925
                + x1049 * x925
                + x1049 * x928
                + x1050
                + x1051 * x825
                + x1053 * x825
                + x1054 * x915
                + x1055
                + x1057 * x915
                + x1059 * x254
                + x1061 * x917
                + x1063 * x917
                + x1064
                + x1066 * x224
                + x1067
                + x1069 * x203
            )
            + 2 * dq_i4 * (x1010 * x925 + x1011 + x1014 * x915 + x1015 + x1018 * x917 + x1019 + x1020 + x841 * x989)
            + 2 * dq_i5 * (x1003 * x825 + x1004 * x751 + x1005 + x1006 * x858 + x1007 + x1008)
            + 2 * dq_i6 * (x1000 + x751 * x995 + x786 * x997 + x999)
            + 2 * dq_i7 * sigma_kin_v_7_1 * x139 * x142 * x145 * x147 * x149 * x152 * x192 * x208 * x390 * x975
            - x216 * x988
            - x242 * x991
            - x248 * x992
            - x272 * x994
            - x278 * x993
            - x506 * x966
            - x506 * x970
            - x506 * x974
            - x506 * x981
            - x506 * x985
            - x522 * (x712 * x989 + x714 * x968 + x717 * x986 + x719 * x976 + x722 * x963)
            - x734 * x977
        )
        + x109 * x112 * (sigma_pot_1_c * x2 + sigma_pot_1_s * x5)
        + x116 * x130
        + x116 * x134
        + x1174
        * (
            2 * dq_i1 * dq_i7 * dq_j1 * x1083 * x138 * x143 * x145 * x147 * x149 * x152 * x208 * x322 * x390
            + 2
            * dq_i1
            * (
                x1023 * x1139
                + x1072 * x1138
                + x1073
                + x1076
                + x1078 * x1140
                + x1079
                + x1081 * x1141
                + x1082
                + x1087 * x1142
                + x1088
                + x1143 * x1145
                + x1145 * x384
            )
            + dq_i2
            * (
                x1028 * x1152
                + x1031 * x1152
                + x1133 * x385
                + x1138 * x1148
                + x1140 * x1157
                + x1141 * x1162
                + x1142 * x1167
                + x1143 * x1169
                + x1146 * x31 * x723
                + x1147 * x357
                + x1150
                + x1151 * x31 * x724
                + x1154
                + x1155 * x31 * x725
                + x1156 * x367
                + x1159
                + x1160 * x31 * x726
                + x1161 * x373
                + x1163 * x31 * x727
                + x1165
                + x1166 * x379
                + x1168 * x31 * x728
                + x1171
                + x1173
            )
            + 2
            * dq_i3
            * (
                x1028 * x1126
                + x1127
                + x1128 * x911
                + x1129
                + x1130 * x825
                + x1131 * x915
                + x1132
                + x1135 * x917
                + x1136
                + x1137
            )
            + 2 * dq_i4 * (x1114 * x911 + x1115 + x1117 * x841 + x1119 * x915 + x1120 + x1122 * x917 + x1123 + x1124)
            + 2 * dq_i5 * (x1106 * x825 + x1107 * x751 + x1108 + x1110 * x858 + x1111 + x1113)
            + 2 * dq_i6 * (x1100 * x751 + x1101 * x786 + x1103 + x1104)
            + 2 * dq_i7 * sigma_kin_v_7_1 * x1083 * x138 * x143 * x145 * x147 * x149 * x152 * x192 * x208 * x390
            - x1073 * x506
            - x1076 * x506
            - x1079 * x506
            - x1082 * x506
            - x1085 * x158
            - x1088 * x506
            - x1089 * x138 * x384 * x507
            - x1091 * x216
            - x1093 * x242
            - x1095 * x248
            - x1096 * x272
            - x1097 * x278
            - x522 * (x1071 * x723 + x1074 * x724 + x1077 * x725 + x1080 * x726 + x1083 * x728 + x1086 * x727)
        )
        + x13 * x18
        + x13 * x9
        + x135 * x159
        + x135 * x217
        + x135 * x243
        + x135 * x249
        + x135 * x273
        + x135 * x279
        - x135
        * (
            dq_i1 * x429
            + dq_i1 * x430
            + dq_i1 * x431
            + dq_i1 * x432
            + dq_i1 * x433
            + dq_i1 * x434
            + dq_i1 * x435
            + dq_i1 * x436
            + dq_i1 * x437
            + dq_i1 * x438
            + dq_i1 * x439
            + dq_i1 * x440
            - dq_i1
            * (
                x1022 * x60
                + x1138 * x38
                + x1140 * x85
                + x1141 * x133
                + x1142 * x182
                + x1143 * x200
                + x124 * x31 * x318
                + x173 * x31 * x321
                + x19 * x31 * x310
                + x195 * x31 * x324
                + x31 * x312 * x39
                + x31 * x315 * x76
                + 2 * x427
                + 2 * x428
                + x441
            )
            + x1175 * x18
            + x1175 * x9
            - x1176 * x204 * x207
            - x1176 * x207 * x858
            + x159
            + x217
            + x243
            + x249
            + x273
            + x279
            + x325 * x522
            - x399
            * (x1022 * x1196 + x1138 * x1195 + x1140 * x1197 + x1141 * x1198 + x1142 * x1199 + x1143 * x1200 + x389)
            - x405 * (x1022 * x1191 + x1192 * x825 + x1193 * x915 + x1194 * x917 + x338 * x911 + x352)
            - x409 * (x1185 * x911 + x1187 * x915 + x1189 * x917 + x292 * x841 + x308)
            - x413 * (x1181 * x825 + x1182 * x751 + x1184 * x858 + x268)
            - x417 * (x1178 * x751 + x1179 * x753 + x237)
        )
        + x163 * x179
        + x163 * x183
        + x194 * x197
        + x194 * x201
        - x202 * x204 * x211
        + x30 * x35
        - 4 * x31 * (x427 + x428 + x441)
        + x35 * x38
        - x390 * x393 * x424 * x426 * (-x396 + x402 + x407 + x412 + x416 + x418 * x419 + x420 * x423)
        + x500
        * (
            ddq_i6 * x456
            + 2 * dq_i1 * dq_i6 * sigma_kin_v_6_1 * sigma_kin_v_6_6 * x165 * x167 * x169 * x171 * x175 * x176 * x181
            + 2
            * dq_i1
            * dq_i6
            * sigma_kin_v_7_1
            * sigma_kin_v_7_6
            * x137
            * x139
            * x143
            * x145
            * x147
            * x149
            * x153
            * x198
            - x443 * x449
            - x461 * x462
            - x467 * x468
            - x477 * x478
            - x485 * x486
            - x492 * x493
            - x497 * x499
        )
        + x53 * x57
        + x557
        * (
            x184 * x527
            + x184 * x530
            + x411 * x515
            + x415 * x515
            - x424 * x522 * x523
            + x508 * x511
            - x509
            + x510 * x520
            + x512 * x513
            + x513 * x514
            + x546
            + x554
        )
        + x57 * x60
        + x596
        * (
            ddq_i5 * x568
            + 2 * dq_i1 * dq_i5 * sigma_kin_v_5_1 * sigma_kin_v_5_5 * x118 * x120 * x122 * x126 * x127 * x132
            + 2 * dq_i1 * dq_i5 * sigma_kin_v_6_1 * sigma_kin_v_6_5 * x165 * x167 * x169 * x171 * x175 * x176 * x181
            + 2
            * dq_i1
            * dq_i5
            * sigma_kin_v_7_1
            * sigma_kin_v_7_5
            * x137
            * x139
            * x143
            * x145
            * x147
            * x149
            * x153
            * x198
            - x493 * x563
            - x559 * x561
            - x573 * x574
            - x577 * x578
            - x584 * x585
            - x588 * x589
            - x593 * x595
        )
        + x628
        * (
            ddq_i4 * x609
            + 2 * dq_i1 * dq_i4 * sigma_kin_v_4_1 * sigma_kin_v_4_4 * x66 * x74 * x78 * x79 * x84
            + 2 * dq_i1 * dq_i4 * sigma_kin_v_5_1 * sigma_kin_v_5_4 * x118 * x120 * x122 * x126 * x127 * x132
            + 2 * dq_i1 * dq_i4 * sigma_kin_v_6_1 * sigma_kin_v_6_4 * x165 * x167 * x169 * x171 * x175 * x176 * x181
            + 2
            * dq_i1
            * dq_i4
            * sigma_kin_v_7_1
            * sigma_kin_v_7_4
            * x137
            * x139
            * x143
            * x145
            * x147
            * x149
            * x153
            * x198
            - x486 * x602
            - x589 * x604
            - x598 * x599
            - x615 * x616
            - x619 * x620
            - x623 * x624
            - x625 * x627
        )
        + x64 * x82
        + x64 * x86
        + x656
        * (
            ddq_i3 * x642
            + 2 * dq_i1 * dq_i3 * sigma_kin_v_3_1 * sigma_kin_v_3_3 * x40 * x42 * x50 * x59
            + 2 * dq_i1 * dq_i3 * sigma_kin_v_4_1 * sigma_kin_v_4_3 * x66 * x74 * x78 * x79 * x84
            + 2 * dq_i1 * dq_i3 * sigma_kin_v_5_1 * sigma_kin_v_5_3 * x118 * x120 * x122 * x126 * x127 * x132
            + 2 * dq_i1 * dq_i3 * sigma_kin_v_6_1 * sigma_kin_v_6_3 * x165 * x167 * x169 * x171 * x175 * x176 * x181
            + 2
            * dq_i1
            * dq_i3
            * sigma_kin_v_7_1
            * sigma_kin_v_7_3
            * x137
            * x139
            * x143
            * x145
            * x147
            * x149
            * x153
            * x198
            - x478 * x634
            - x585 * x636
            - x624 * x638
            - x630 * x631
            - x648 * x649
            - x651 * x652
            - x653 * x655
        )
        + x699
        * (
            ddq_i2 * x683
            - x468 * x673
            - x578 * x676
            - x620 * x680
            - x652 * x686
            + x657 * x658
            + x657 * x659
            + x657 * x661
            + x657 * x663
            + x657 * x665
            + x657 * x667
            - x657 * x687
            - x669 * x670
            - x696 * x698
        )
        + x729
        * (
            ddq_i1 * (x309 * x682 + x314 * x606 + x317 * x565 + x320 * x451 + x323 * x445 + x51 * x721 + x6 * x7**2)
            + x11 * x700
            + x11 * x701
            + x11 * x702
            + x11 * x703
            + x11 * x704
            + x11 * x705
            + x11 * x706
            - x12 * (x700 + x701 + x702 + x703 + x704 + x705 + x706)
            - x462 * (x220 * x710 + x495 * x711)
            - x574 * (x253 * x713 + x261 * x709 + x490 * x707)
            - x616 * (x282 * x715 + x295 * x718 + x302 * x720 + x586 * x716)
            - x649 * (x328 * x722 + x337 * x714 + x471 * x717 + x475 * x719 + x581 * x712)
            - x657 * (x355 * x723 + x361 * x724 + x365 * x725 + x371 * x726 + x377 * x727 + x383 * x728)
            - x707 * x708
        )
        + x794
        * (
            2 * dq_i1 * dq_i7 * dq_j1 * x139 * x143 * x145 * x147 * x148 * x152 * x208 * x322 * x390 * x730
            + 2 * dq_i1 * (x229 * x755 + x739 + x750 * x752 + x753 * x756)
            + 2 * dq_i2 * (x752 * x757 + x758 * x760 + x761 + x764)
            + 2 * dq_i3 * (x471 * x752 + x765 * x766 + x768 + x770)
            + 2 * dq_i4 * (x480 * x752 + x766 * x771 + x772 + x773)
            + 2 * dq_i5 * (x490 * x758 + x774 * x776 + x778 + x780)
            + dq_i6 * (x31 * x710 * x781 + x753 * x787 + x774 * x783 + x785 * x786 + x789 + x791 + x792 + x793)
            + 2 * dq_i7 * sigma_kin_v_7_1 * x139 * x143 * x145 * x147 * x148 * x152 * x192 * x208 * x390 * x730
            - x139 * x229 * x507 * x741
            - x216 * x747
            - x242 * x746
            - x248 * x744
            - x272 * x743
            - x278 * x742
            - x506 * x739
            - x522 * (x710 * x735 + x731 * x748)
            - x733 * x734
        )
        + x864
        * (
            2 * dq_i1 * dq_i7 * dq_j1 * x139 * x143 * x145 * x146 * x149 * x152 * x208 * x322 * x390 * x799
            + 2 * dq_i1 * (x569 * x826 + x750 * x827 + x798 + x806 + x810 + x828 * x830)
            + 2 * dq_i2 * (x576 * x826 + x757 * x827 + x759 * x830 + x831 + x832 + x833)
            + 2 * dq_i3 * (x471 * x827 + x765 * x830 + x825 * x836 + x837 + x839 + x840)
            + 2 * dq_i4 * (x480 * x827 + x813 * x841 + x829 * x845 + x842 + x846 + x847)
            + dq_i5
            * (
                x31 * x713 * x848
                + x774 * x851
                + x774 * x854
                + x825 * x850
                + x829 * x859
                + x853
                + x855
                + x857 * x858
                + x860
                + x861
                + x862
                + x863
            )
            + 2 * dq_i6 * (x774 * x821 + x786 * x822 + x823 + x824)
            + 2 * dq_i7 * sigma_kin_v_7_1 * x139 * x143 * x145 * x146 * x149 * x152 * x192 * x208 * x390 * x799
            - x216 * x812
            - x242 * x818
            - x248 * x817
            - x272 * x816
            - x278 * x814
            - x506 * x798
            - x506 * x806
            - x506 * x810
            - x522 * (x709 * x804 + x713 * x795 + x800 * x819)
            - x734 * x802
        )
        + x962
        * (
            2 * dq_i1 * dq_i7 * dq_j1 * x139 * x143 * x144 * x147 * x149 * x152 * x208 * x322 * x390 * x875
            + 2 * dq_i1 * (x457 * x916 + x459 * x918 + x610 * x912 + x868 + x869 * x914 + x874 + x881 + x884)
            + 2 * dq_i2 * (x464 * x916 + x466 * x918 + x618 * x912 + x914 * x920 + x919 + x921 + x923 + x924)
            + 2 * dq_i3 * (x581 * x913 + x915 * x932 + x917 * x935 + x925 * x927 + x929 + x933 + x936 + x937)
            + dq_i4
            * (
                x31 * x715 * x938
                + x31 * x718 * x948
                + x31 * x720 * x953
                + x901 * x945
                + x901 * x947
                + x915 * x951
                + x917 * x955
                + x925 * x940
                + x942
                + x944
                + x949
                + x952
                + x957
                + x958
                + x960
                + x961
            )
            + 2 * dq_i5 * (x751 * x904 + x858 * x908 + x901 * x903 + x905 + x909 + x910)
            + 2 * dq_i6 * (x751 * x894 + x786 * x896 + x898 + x900)
            + 2 * dq_i7 * sigma_kin_v_7_1 * x139 * x143 * x144 * x147 * x149 * x152 * x192 * x208 * x390 * x875
            - x216 * x887
            - x242 * x890
            - x248 * x893
            - x272 * x892
            - x278 * x891
            - x506 * x868
            - x506 * x874
            - x506 * x881
            - x506 * x884
            - x522 * (x316 * x889 + x715 * x865 + x718 * x879 + x720 * x875)
            - x734 * x877
        )
    )
    K_block_list.append(
        -dq_i1 * x1174 * x1331
        + x1070
        * (
            dq_i3
            * x1332
            * (
                x1044 * x684
                + x1045 * x684
                + x1048 * x677
                + x1049 * x677
                + x1051 * x675
                + x1053 * x675
                + x1054 * x678
                + x1057 * x678
                + x1061 * x679
                + x1063 * x679
            )
            - x1032 * x506
            - x1033 * x506
            - x1034 * x506
            - x1039 * x506
            - x1041 * x506
            - x1275 * x993
            + x1339 * (x1010 * x677 + x1013 * x1395 + x1017 * x1396 + x1371 * x989)
            + x1342 * (x1002 * x1393 + x1392 * x261 + x1394 * x996)
            + x1345 * (x1379 * x996 + x1392 * x222)
            + x1346 * (x1021 * x1400 + x1027 * x679 + x1397 * x610 + x1398 * x869 + x1399 * x457)
            + x1358 * (x1030 * x684 + x1038 * x679 + x1397 * x618 + x1398 * x920 + x1399 * x464)
            + x1377 * x1391
        )
        + x1174 * x1208
        + x1174 * x1220
        + x1174 * x1239
        + x1174 * x1248
        + x1174 * x1269
        - x1174
        * (
            dq_i1 * x1150
            + dq_i1 * x1154
            + dq_i1 * x1159
            + dq_i1 * x1165
            + dq_i1 * x1171
            + dq_i1 * x1173
            + dq_i1 * x1403 * x1404 * x356
            + dq_i1 * x1408 * x1409 * x366
            + dq_i1 * x1410 * x1411 * x372
            + dq_i1 * x1412 * x1413 * x378
            + dq_i1 * x1414 * x1417 * x384
            - dq_i1
            * x32
            * (
                x1146 * x689
                + x1153 * x684
                + x1158 * x660
                + x1164 * x662
                + x1170 * x664
                + x1172 * x666
                + x1404 * x1450
                + x1406 * x1437
                + x1409 * x1438
                + x1411 * x1439
                + x1413 * x1440
                + x1417 * x1441
            )
            - x1275 * (x1403 * x689 + x1437 * x362 + x1438 * x617 + x1439 * x575 + x1440 * x463 + x1441 * x465)
            - x1336
            * (
                x1029 * x1437 * x1442
                + x1037 * x1436 * x1449 * x390
                + x1426 * x1445 * x1446
                + x1431 * x1444
                + x1434 * x1447 * x1448
            )
            - x1339
            * (x122 * x1384 * x1425 * x1432 + x1430 * x1431 + x1433 * x1434 * x294 + x1435 * x1436 * x301 * x465)
            - x1342 * (x1363 * x1421 * x1428 + x1420 * x1427 + x1424 * x1426 * x250)
            - x1345 * (x1357 * x1421 * x1423 + x1419 * x1420)
            - x1346
            * (
                x1400 * x1405 * x1451 * x362
                + x1438 * x1452 * x368
                + x1439 * x1453 * x374
                + x1440 * x1454 * x380
                + x1441 * x1455 * x386
                - x1450 * x358 * x37
            )
            - x1377 * x1414 * x1415 * x1416 * x465
            + x1405 * x1406 * x1407
        )
        + 4 * x1195 * x34
        + x1196 * x1201
        + x1197 * x1204 * x75
        + x1198 * x1205 * x123
        + x1199 * x1211 * x172
        + x1200 * x1212 * x154
        + x1202 * x1203 * (sigma_pot_2_c * x353 - sigma_pot_2_s * x354)
        + x1210 * x155 * x466
        - x1216 * x211
        - x1227 * (x1222 * x458 + x1224 * x460)
        - x1234 * x1235
        - x1244 * (x1222 * x571 + x1224 * x572 + x1241 * x570)
        - x1253 * (x1222 * x613 + x1224 * x614 + x1241 * x612 + x1250 * x611)
        - x1261 * x1262
        - x1274 * (x1222 * x646 + x1224 * x647 + x1241 * x645 + x1250 * x644 + x1271 * x362)
        + x1276 * x135
        + x1289 * x1290
        - x1292 * (x1221 * x727 + x1223 * x728 + x1240 * x726 + x1249 * x725 + x1270 * x724 + x1291 * x723)
        - x1303 * x1304
        - x1317 * x1318
        - 8 * x1330 * x32
        + x135
        * (
            2
            * dq_i1
            * dq_i2
            * dq_j2
            * (x1195 * x274 + x1197 * x660 + x1198 * x662 + x1199 * x664 + x1200 * x666 + x1402 * x363)
            + 2
            * dq_i1
            * dq_i3
            * dq_j2
            * (sigma_kin_v_4_1 * x1309 + x1369 * x291 + x1388 * x223 + x1389 * x206 + x1402 * x330)
            + 2 * dq_i1 * dq_i4 * dq_j2 * (x1185 * x1382 + x1186 * x1395 + x1188 * x1396 + x1371 * x292)
            + 2 * dq_i1 * dq_i5 * dq_j2 * (x1180 * x1393 + x1183 * x1394 + x1401 * x261)
            + 2 * dq_i1 * dq_i6 * dq_j2 * (x1361 * x235 + x1401 * x222)
            + 2
            * dq_i1
            * dq_i7
            * dq_j2
            * sigma_kin_v_7_1
            * sigma_kin_v_7_2
            * x136
            * x137
            * x139
            * x143
            * x145
            * x147
            * x149
            * x152
            * x208
            + dq_j2
            * x11
            * (
                x129 * x662
                + x133 * x662
                + x178 * x664
                + x182 * x664
                + x196 * x666
                + x200 * x666
                + x274 * x30
                + x274 * x38
                + x276 * x53
                + x276 * x60
                + x660 * x81
                + x660 * x85
            )
            - x1276
            - x360 * x506
            - x364 * x506
            - x370 * x506
            - x376 * x506
            - x382 * x506
            - x388 * x506
        )
        + x1374 * x699
        + x557
        * (
            sigma_kin_v_7_2 * x1345 * x519
            - x1275 * x532
            + x1333 * x527
            + x1333 * x530
            + x1334 * x1335 * x185
            + x1336 * x1337 * x1338
            + x1337 * x1339 * x1341
            + x1337 * x1342 * x1344
            + x1346 * x1348 * x531
            - x1350 * x398
        )
        + x794
        * (
            dq_i6 * x1332 * (x1362 * x668 + x668 * x787 + x672 * x783 + x672 * x788)
            - x1275 * x742
            + x1336 * (x1359 * x471 + x1360 * x765)
            + x1339 * (x1359 * x480 + x1360 * x771)
            + x1342 * (x1357 * x779 + x672 * x776)
            + x1346 * (x1356 * x457 + x1361 * x755)
            + x1352 * x1353 * x668
            + x1358 * (x1356 * x464 + x1357 * x763)
            - x506 * x761
            - x506 * x764
        )
        + x864
        * (
            dq_i5 * x1332 * (x1373 * x669 + x669 * x857 + x672 * x851 + x672 * x854 + x675 * x850 + x675 * x852)
            - x1275 * x814
            + x1336 * (x1364 * x765 + x1369 * x835 + x1370 * x471)
            + x1339 * (x1370 * x480 + x1371 * x813 + x1372 * x669)
            + x1345 * (x1357 * x822 + x672 * x821)
            + x1346 * (x1364 * x828 + x1367 * x569 + x1368 * x457)
            + x1353 * x1364 * x1366
            + x1358 * (x1364 * x759 + x1367 * x576 + x1368 * x464)
            - x506 * x831
            - x506 * x832
            - x506 * x833
        )
        + x962
        * (
            dq_i4
            * x1332
            * (
                x674 * x945
                + x674 * x947
                + x677 * x940
                + x677 * x941
                + x678 * x951
                + x678 * x956
                + x679 * x955
                + x679 * x959
            )
            - x1275 * x891
            + x1336 * (x1381 * x581 + x1388 * x931 + x1389 * x934 + x677 * x927)
            + x1342 * (x1378 * x261 + x1381 * x591 + x669 * x908)
            + x1345 * (x1378 * x222 + x1379 * x895)
            + x1346 * (x1383 * x610 + x1385 * x869 + x1386 * x457 + x1387 * x459)
            + x1358 * (x1383 * x618 + x1385 * x920 + x1386 * x464 + x1387 * x466)
            + x1376 * x1377
            - x506 * x919
            - x506 * x921
            - x506 * x923
            - x506 * x924
        )
    )
    K_block_list.append(
        x1070
        * (
            2
            * dq_i1
            * (
                x1021 * x1306
                + x1021 * x1747
                + x1027 * x1300
                + x1027 * x1706
                + x1723 * x610
                + x1725 * x869
                + x1727 * x457
                + x1748 * x610
                + x1749 * x869
                + x1750 * x457
            )
            + 2 * dq_i2 * dq_i7 * dq_j2 * x1287 * x139 * x142 * x145 * x147 * x149 * x152 * x208 * x390 * x975
            + 2
            * dq_i2
            * (
                x1030 * x1751
                + x1038 * x1706
                + x1722
                + x1724
                + x1726
                + x1728
                + x1729
                + x1748 * x618
                + x1749 * x920
                + x1750 * x464
            )
            + dq_i3
            * (
                x1042 * x1617 * x32
                + x1044 * x1518
                + x1044 * x1751
                + x1048 * x1708
                + x1049 * x1708
                + x1049 * x1709
                + x1051 * x1659
                + x1053 * x1659
                + x1054 * x1704
                + x1057 * x1704
                + x1059 * x1254
                + x1061 * x1706
                + x1063 * x1706
                + x1066 * x1228
                + x1069 * x1214
                + x1752
                + x1753
                + x1754
                + x1755
                + x1756
            )
            + 2
            * dq_i4
            * (x1010 * x1708 + x1014 * x1704 + x1018 * x1706 + x1668 * x1742 + x1741 + x1743 + x1744 + x1745)
            + 2 * dq_i5 * (x1003 * x1659 + x1004 * x1627 + x1006 * x1662 + x1738 + x1739 + x1740)
            + 2 * dq_i6 * (x1627 * x995 + x1630 * x997 + x1736 + x1737)
            + 2 * dq_i7 * sigma_kin_v_7_2 * x139 * x142 * x145 * x147 * x149 * x152 * x192 * x208 * x390 * x975
            - x1461 * x977
            - x1565 * (x1609 * x989 + x1611 * x968 + x1613 * x986 + x1615 * x976 + x1617 * x963)
            - x1722 * x399
            - x1724 * x399
            - x1726 * x399
            - x1728 * x399
            - x1729 * x399
            - x1730 * x216
            - x1731 * x242
            - x1733 * x248
            - x1734 * x272
            - x521 * x993
        )
        + x1097 * x1275 * x135
        + x1148 * x35
        + x1149 * x35
        + x1152 * x1456
        + x1153 * x1456
        + x1157 * x1457
        + x1158 * x1457
        + x1162 * x1459
        + x1164 * x1459
        + x1167 * x1463
        + x1169 * x1464
        + x1170 * x1463
        + x1172 * x1464
        + x1174 * x1462
        + x1174 * x1472
        + x1174 * x1485
        + x1174 * x1494
        + x1174 * x1505
        + x1174
        * (
            2
            * dq_i1
            * (x1072 * x1758 + x1078 * x1759 + x1081 * x1760 + x1087 * x1761 + x1139 * x1747 + x1145 * x1762 + x1526)
            + 2 * dq_i2 * dq_i7 * dq_j2 * x1083 * x1287 * x138 * x143 * x145 * x147 * x149 * x152 * x208 * x390
            - dq_i2 * x1536
            - dq_i2 * x1537
            - dq_i2 * x1538
            - dq_i2 * x1539
            - dq_i2 * x1540
            - dq_i2 * x1541
            - dq_i2 * x1542
            - dq_i2 * x1543
            - dq_i2 * x1544
            - dq_i2 * x1545
            - dq_i2 * x1546
            - dq_i2 * x1547
            + dq_i2
            * (
                x1146 * x1506 * x32
                + x1148 * x1758
                + x1152 * x1751
                + x1153 * x1751
                + x1155 * x1508 * x32
                + x1157 * x1759
                + x1160 * x1509 * x32
                + x1162 * x1760
                + x1163 * x1510 * x32
                + x1167 * x1761
                + x1168 * x1511 * x32
                + x1169 * x1762
                + x1548
            )
            + 2 * dq_i3 * (x1126 * x1751 + x1128 * x1700 + x1130 * x1659 + x1131 * x1704 + x1135 * x1706 + x1525)
            + 2 * dq_i4 * (x1114 * x1700 + x1119 * x1704 + x1122 * x1706 + x1517 + x1668 * x1763)
            + 2 * dq_i5 * (x1106 * x1659 + x1107 * x1627 + x1110 * x1662 + x1498)
            + 2 * dq_i6 * (x1100 * x1627 + x1101 * x1630 + x1477)
            + 2 * dq_i7 * sigma_kin_v_7_2 * x1083 * x138 * x143 * x145 * x147 * x149 * x152 * x192 * x208 * x390
            - x1097 * x521
            - x1462
            - x1472
            - x1485
            - x1494
            - x1505
            - x1512 * x1565
        )
        + x1203 * x1458 * (sigma_pot_2_c * x25 + sigma_pot_2_s * x22)
        - x1216 * x1466
        - x1235 * x1477
        - x1262 * x1498
        + x1290 * x1512
        - x1304 * x1517
        - x1318 * x1525
        - 8 * x1332 * x1526
        + x135
        * (
            dq_i1
            * (
                x1278 * x19 * x32
                + x1280 * x32 * x39
                + x129 * x1760
                + x133 * x1760
                + x1331
                + x1746 * x60
                + x1758 * x38
                + x1759 * x81
                + x1759 * x85
                + x1761 * x178
                + x1761 * x182
                + x1762 * x196
                + x1762 * x200
            )
            + 2 * dq_i2 * dq_i7 * dq_j2 * x1287 * x136 * x137 * x139 * x143 * x145 * x147 * x149 * x152 * x208
            + 2
            * dq_i2
            * (x1195 * x1758 + x1196 * x1746 + x1197 * x1759 + x1198 * x1760 + x1199 * x1761 + x1200 * x1762 + x1330)
            + 2 * dq_i3 * (x1191 * x1746 + x1192 * x1659 + x1193 * x1704 + x1194 * x1706 + x1317 + x1700 * x338)
            + 2 * dq_i4 * (x1185 * x1700 + x1187 * x1704 + x1189 * x1706 + x1303 + x1668 * x1757)
            + 2 * dq_i5 * (x1181 * x1659 + x1182 * x1627 + x1184 * x1662 + x1261)
            + 2 * dq_i6 * (x1178 * x1627 + x1179 * x1629 + x1234)
            + 2 * dq_i7 * sigma_kin_v_7_2 * x136 * x137 * x139 * x143 * x145 * x147 * x149 * x152 * x192 * x208
            - x1208
            - x1220
            - x1239
            - x1248
            - x1269
            - x1289 * x1565
            - x1320 * x399
            - x1321 * x399
            - x1323 * x399
            - x1325 * x399
            - x1327 * x399
            - x1329 * x399
            - x277 * x521
        )
        + x1374 * x1598
        - x140
        * x397
        * x426
        * x531
        * (-x1527 * x397 + x1528 * x397 + x1529 * x1530 + x1532 * x411 + x1532 * x415 + x1534 * x397 + x1535 * x397)
        - 4 * x1548 * x32
        + x1569
        * (
            -x1559
            + x1561 * x520
            + x1562 * x508
            + x1563 * x512
            + x1563 * x514
            + x1564 * x411
            + x1564 * x415
            - x1565 * x532
            + x1568
            + x185 * x527
            + x185 * x530
            + x536
        )
        + x1618
        * (
            ddq_i2 * (x1277 * x682 + x1279 * x640 + x1281 * x607 + x1283 * x566 + x1285 * x452 + x1287 * x454)
            + x1599 * x697
            + x1600 * x697
            + x1601 * x697
            + x1602 * x697
            + x1603 * x697
            + x1604 * x697
            - x1605 * x670
            - x468 * (x1607 * x220 + x1608 * x495)
            - x578 * (x1605 * x490 + x1606 * x261 + x1610 * x253)
            - x620 * (x1283 * x587 + x1612 * x282 + x1614 * x295 + x1616 * x302)
            - x652 * (x1609 * x581 + x1611 * x337 + x1613 * x471 + x1615 * x475 + x1617 * x328)
            - x657 * (x1278 * x37 + x1280 * x59 + x1282 * x610 + x1284 * x569 + x1286 * x457 + x1288 * x459)
            - x698 * (x1599 + x1600 + x1601 + x1602 + x1603 + x1604)
        )
        + x500
        * (
            ddq_i6 * x1552
            + 2 * dq_i2 * dq_i6 * sigma_kin_v_6_2 * sigma_kin_v_6_6 * x165 * x167 * x169 * x171 * x174 * x377 * x450
            + 2
            * dq_i2
            * dq_i6
            * sigma_kin_v_7_2
            * sigma_kin_v_7_6
            * x138
            * x143
            * x145
            * x147
            * x149
            * x153
            * x383
            * x390
            - x1549 * x449
            - x1553 * x462
            - x1554 * x468
            - x1555 * x478
            - x1556 * x486
            - x1557 * x493
            - x1558 * x499
        )
        + x596
        * (
            ddq_i5 * x1573
            + 2 * dq_i2 * dq_i5 * sigma_kin_v_5_2 * sigma_kin_v_5_5 * x118 * x120 * x122 * x125 * x371 * x564
            + 2 * dq_i2 * dq_i5 * sigma_kin_v_6_2 * sigma_kin_v_6_5 * x165 * x167 * x169 * x171 * x174 * x377 * x450
            + 2
            * dq_i2
            * dq_i5
            * sigma_kin_v_7_2
            * sigma_kin_v_7_5
            * x138
            * x143
            * x145
            * x147
            * x149
            * x153
            * x383
            * x390
            - x1570 * x561
            - x1571 * x493
            - x1574 * x574
            - x1575 * x578
            - x1576 * x585
            - x1577 * x589
            - x1578 * x595
        )
        + x628
        * (
            ddq_i4 * x1583
            + 2 * dq_i2 * dq_i4 * sigma_kin_v_4_2 * sigma_kin_v_4_4 * x365 * x605 * x66 * x74 * x77
            + 2 * dq_i2 * dq_i4 * sigma_kin_v_5_2 * sigma_kin_v_5_4 * x118 * x120 * x122 * x125 * x371 * x564
            + 2 * dq_i2 * dq_i4 * sigma_kin_v_6_2 * sigma_kin_v_6_4 * x165 * x167 * x169 * x171 * x174 * x377 * x450
            + 2
            * dq_i2
            * dq_i4
            * sigma_kin_v_7_2
            * sigma_kin_v_7_4
            * x138
            * x143
            * x145
            * x147
            * x149
            * x153
            * x383
            * x390
            - x1579 * x599
            - x1580 * x486
            - x1581 * x589
            - x1584 * x616
            - x1585 * x620
            - x1587 * x624
            - x1589 * x627
        )
        + x656
        * (
            ddq_i3 * x1594
            + 2 * dq_i2 * dq_i3 * sigma_kin_v_3_2 * sigma_kin_v_3_3 * x361 * x41 * x50 * x639
            + 2 * dq_i2 * dq_i3 * sigma_kin_v_4_2 * sigma_kin_v_4_3 * x365 * x605 * x66 * x74 * x77
            + 2 * dq_i2 * dq_i3 * sigma_kin_v_5_2 * sigma_kin_v_5_3 * x118 * x120 * x122 * x125 * x371 * x564
            + 2 * dq_i2 * dq_i3 * sigma_kin_v_6_2 * sigma_kin_v_6_3 * x165 * x167 * x169 * x171 * x174 * x377 * x450
            + 2
            * dq_i2
            * dq_i3
            * sigma_kin_v_7_2
            * sigma_kin_v_7_3
            * x138
            * x143
            * x145
            * x147
            * x149
            * x153
            * x383
            * x390
            - x1590 * x631
            - x1591 * x478
            - x1592 * x585
            - x1593 * x624
            - x1595 * x649
            - x1596 * x652
            - x1597 * x655
        )
        + x794
        * (
            2 * dq_i1 * (x1232 * x755 + x1620 * x457 + x1628 * x750 + x1629 * x756)
            + 2 * dq_i2 * dq_i7 * dq_j2 * x1287 * x139 * x143 * x145 * x147 * x148 * x152 * x208 * x390 * x730
            + 2 * dq_i2 * (x1621 + x1622 + x1628 * x757 + x1630 * x763)
            + 2 * dq_i3 * (x1628 * x471 + x1631 * x765 + x1633 + x1635)
            + 2 * dq_i4 * (x1628 * x480 + x1631 * x771 + x1636 + x1637)
            + 2 * dq_i5 * (x1630 * x779 + x1638 * x776 + x1640 + x1641)
            + dq_i6 * (x1607 * x32 * x781 + x1629 * x787 + x1630 * x785 + x1638 * x783 + x1642 + x1643 + x1644 + x1645)
            + 2 * dq_i7 * sigma_kin_v_7_2 * x139 * x143 * x145 * x147 * x148 * x152 * x192 * x208 * x390 * x730
            - x1461 * x733
            - x1565 * (x1607 * x735 + x1608 * x732)
            - x1621 * x399
            - x1622 * x399
            - x1623 * x272
            - x1624 * x248
            - x1625 * x242
            - x1626 * x216
            - x521 * x742
        )
        + x864
        * (
            2 * dq_i1 * (x1646 * x569 + x1648 * x457 + x1651 * x828 + x1660 * x569 + x1661 * x750 + x1663 * x828)
            + 2 * dq_i2 * dq_i7 * dq_j2 * x1287 * x139 * x143 * x145 * x146 * x149 * x152 * x208 * x390 * x799
            + 2 * dq_i2 * (x1647 + x1649 + x1652 + x1660 * x576 + x1661 * x757 + x1663 * x759)
            + 2 * dq_i3 * (x1659 * x836 + x1661 * x471 + x1663 * x765 + x1664 + x1666 + x1667)
            + 2 * dq_i4 * (x1372 * x1662 + x1661 * x480 + x1668 * x1669 + x1670 + x1671 + x1672)
            + dq_i5
            * (
                x1373 * x1662
                + x1610 * x32 * x848
                + x1638 * x851
                + x1638 * x854
                + x1659 * x850
                + x1662 * x857
                + x1673
                + x1674
                + x1675
                + x1676
                + x1677
                + x1678
            )
            + 2 * dq_i6 * (x1630 * x822 + x1638 * x821 + x1657 + x1658)
            + 2 * dq_i7 * sigma_kin_v_7_2 * x139 * x143 * x145 * x146 * x149 * x152 * x192 * x208 * x390 * x799
            - x1461 * x802
            - x1565 * (x1605 * x801 + x1606 * x804 + x1610 * x795)
            - x1647 * x399
            - x1649 * x399
            - x1652 * x399
            - x1653 * x216
            - x1654 * x272
            - x1655 * x248
            - x1656 * x242
            - x521 * x814
        )
        + x962
        * (
            2
            * dq_i1
            * (
                x1679 * x610
                + x1682 * x869
                + x1684 * x457
                + x1686 * x459
                + x1701 * x610
                + x1703 * x869
                + x1705 * x457
                + x1707 * x459
            )
            + 2 * dq_i2 * dq_i7 * dq_j2 * x1287 * x139 * x143 * x144 * x147 * x149 * x152 * x208 * x390 * x875
            + 2 * dq_i2 * (x1680 + x1683 + x1685 + x1687 + x1701 * x618 + x1703 * x920 + x1705 * x464 + x1707 * x466)
            + 2 * dq_i3 * (x1702 * x581 + x1704 * x932 + x1706 * x935 + x1708 * x927 + x1710 + x1711 + x1712 + x1713)
            + dq_i4
            * (
                x1612 * x32 * x938
                + x1614 * x32 * x948
                + x1616 * x32 * x953
                + x1668 * x945
                + x1668 * x947
                + x1704 * x951
                + x1706 * x955
                + x1708 * x940
                + x1714
                + x1715
                + x1716
                + x1717
                + x1718
                + x1719
                + x1720
                + x1721
            )
            + 2 * dq_i5 * (x1627 * x904 + x1662 * x908 + x1668 * x903 + x1697 + x1698 + x1699)
            + 2 * dq_i6 * (x1627 * x894 + x1630 * x896 + x1694 + x1696)
            + 2 * dq_i7 * sigma_kin_v_7_2 * x139 * x143 * x144 * x147 * x149 * x152 * x192 * x208 * x390 * x875
            - x1461 * x877
            - x1565 * (x1283 * x889 + x1612 * x865 + x1614 * x879 + x1616 * x875)
            - x1680 * x399
            - x1683 * x399
            - x1685 * x399
            - x1687 * x399
            - x1688 * x216
            - x1689 * x242
            - x1691 * x272
            - x1692 * x248
            - x521 * x891
        )
    )
    K_block_list.append(
        -dq_i1 * x1070 * x1843
        + x1070 * x1767
        + x1070 * x1777
        + x1070 * x1789
        + x1070 * x1794
        + x1070 * x1805
        - x1070
        * (
            dq_i1 * x1046
            + dq_i1 * x1050
            + dq_i1 * x1055
            + dq_i1 * x1064
            + dq_i1 * x1067
            + dq_i1 * x1904 * x928
            + dq_i1 * x1906 * x254
            + dq_i1 * x1908 * x224
            + dq_i1 * x1912 * x203
            - dq_i1
            * x54
            * (
                x1045 * x650
                + x1048 * x637
                + x1051 * x1871
                + x1054 * x1884
                + x1061 * x1886
                + x1900 * x1929
                + x1903 * x1920
                + x1906 * x635
                + x1908 * x632
                + x1912 * x629
            )
            + x1407 * x1901
            - x1803 * (x1871 * x1922 + x1884 * x1913 + x1886 * x1910 + x1920 * x336 + x1929 * x329)
            - x1846 * (x1029 * x1271 * x1899 + x1222 * x1933 + x1241 * x1932 + x1250 * x1931 + x1926 * x1934)
            - x1852 * (x1920 * x1921 + x1923 * x635 + x1924 * x1925 + x1926 * x1928)
            - x1854 * (x1427 * x1914 + x1917 * x1918 + x1919 * x630)
            - x1856 * (x1419 * x1914 + x1865 * x1916)
            - x1858 * (x1897 * x1935 + x1926 * x1940 + x1931 * x1936 + x1932 * x1937 + x1933 * x1938)
            - x1878 * x1911
        )
        + x110 * x1202 * x1764
        + x1174
        * (
            -x1127 * x506
            - x1129 * x506
            - x1132 * x506
            - x1136 * x506
            - x1137 * x506
            + x1466 * x1878
            + x1852 * (x1114 * x1881 + x1117 * x1876 + x1118 * x1893 + x1121 * x1894)
            + x1854 * (x1105 * x1891 + x1109 * x1892 + x1895 * x261)
            + x1856 * (x1101 * x1865 + x1895 * x222)
            + x1858 * (x1078 * x644 + x1081 * x645 + x1087 * x646 + x1139 * x1897 + x1145 * x647)
            + x1868 * (x1098 * x1888 + x1116 * x1874 + x1126 * x650 + x1128 * x1881 + x1135 * x1886)
            - x1896
            + x1898
            * (
                x1152 * x650
                + x1153 * x650
                + x1157 * x644
                + x1158 * x644
                + x1162 * x645
                + x1164 * x645
                + x1167 * x646
                + x1169 * x647
                + x1170 * x646
                + x1172 * x647
            )
        )
        + x1191 * x1201
        + x1192 * x1765
        + x1193 * x1769
        + x1194 * x1770
        + x1204 * x338
        - x1227 * (x1779 * x473 + x1781 * x442)
        - x1244 * (x1779 * x562 + x1781 * x558 + x1791 * x583)
        - x1253 * (x1444 * x622 + x1779 * x600 + x1781 * x597 + x1791 * x603)
        - x1292 * (x1442 * x722 + x1444 * x714 + x1778 * x717 + x1780 * x719 + x1790 * x712)
        + x135 * x1804
        + x135
        * (
            dq_j3
            * x11
            * (
                x129 * x645
                + x133 * x645
                + x178 * x646
                + x182 * x646
                + x196 * x647
                + x200 * x647
                + x270 * x53
                + x270 * x60
                + x644 * x81
                + x644 * x85
            )
            - x1804
            + x1846 * (x1196 * x270 + x1197 * x644 + x1198 * x645 + x1199 * x646 + x1200 * x647)
            + x1852 * (x1185 * x1881 + x1186 * x1893 + x1188 * x1894 + x1876 * x292)
            + x1854 * (x1180 * x1891 + x1183 * x1892 + x1890 * x261)
            + x1856 * (x1869 * x235 + x1890 * x222)
            + x1868 * (x1191 * x270 + x1874 * x291 + x1881 * x338 + x1888 * x223 + x1889 * x206)
            + x1878 * x211
            - x334 * x506
            - x339 * x506
            - x343 * x506
            - x347 * x506
            - x351 * x506
        )
        - x1618 * x1807
        + x1768 * x476
        - x1774 * x211
        - x1785 * x1786
        - x1801 * x1802
        + x1818 * x1819
        - x1827 * x1828
        - x1834 * x1835
        - x1841 * x1842
        + x1877 * x656
        + x557
        * (
            x1335 * x1851
            - x1803 * x535
            + x1845 * x527
            + x1845 * x530
            + x1846 * x1849
            + x1852 * x1853
            + x1854 * x1855
            + x1856 * x1857
            + x1858 * x1859
            - x1860 * x506
        )
        + x794
        * (
            dq_i6 * x1844 * (x1362 * x629 + x629 * x787 + x633 * x783 + x633 * x788)
            + x1352 * x1861 * x629
            - x1803 * x743
            + x1846 * (x1864 * x464 + x1865 * x763)
            + x1852 * (x1866 * x480 + x1867 * x771)
            + x1854 * (x1865 * x779 + x633 * x776)
            + x1858 * (x1864 * x457 + x1869 * x755)
            + x1868 * (x1866 * x471 + x1867 * x765)
            - x506 * x768
            - x506 * x770
        )
        + x864
        * (
            dq_i5 * x1844 * (x1373 * x630 + x1871 * x850 + x1871 * x852 + x630 * x857 + x633 * x851 + x633 * x854)
            + x1366 * x1861 * x1870
            - x1803 * x816
            + x1846 * (x1870 * x759 + x1872 * x576 + x1873 * x464)
            + x1852 * (x1372 * x630 + x1875 * x480 + x1876 * x813)
            + x1856 * (x1865 * x822 + x633 * x821)
            + x1858 * (x1870 * x828 + x1872 * x569 + x1873 * x457)
            + x1868 * (x1870 * x765 + x1874 * x835 + x1875 * x471)
            - x506 * x837
            - x506 * x839
            - x506 * x840
        )
        + x962
        * (
            dq_i4
            * x1844
            * (
                x1884 * x951
                + x1884 * x956
                + x1886 * x955
                + x1886 * x959
                + x635 * x945
                + x635 * x947
                + x637 * x940
                + x637 * x941
            )
            + x1376 * x1878
            - x1803 * x892
            + x1846 * (x1882 * x618 + x1883 * x920 + x1885 * x464 + x1887 * x466)
            + x1854 * (x1879 * x261 + x1880 * x591 + x630 * x908)
            + x1856 * (x1865 * x896 + x1879 * x222)
            + x1858 * (x1882 * x610 + x1883 * x869 + x1885 * x457 + x1887 * x459)
            + x1868 * (x1880 * x581 + x1888 * x931 + x1889 * x934 + x637 * x927)
            - x506 * x929
            - x506 * x933
            - x506 * x936
            - x506 * x937
        )
    )
    K_block_list.append(
        -dq_i2 * x1070 * x1996
        + x1070 * x1943
        + x1070 * x1948
        + x1070 * x1958
        + x1070 * x1965
        + x1070 * x1972
        - x1070
        * (
            dq_i2 * x1214 * x1912
            + dq_i2 * x1228 * x1908
            + dq_i2 * x1254 * x1906
            + dq_i2 * x1518 * x1901
            + dq_i2 * x1709 * x1904
            + dq_i2 * x1752
            + dq_i2 * x1753
            + dq_i2 * x1754
            + dq_i2 * x1755
            + dq_i2 * x1756
            - dq_i2
            * x54
            * (
                x1045 * x1499
                + x1048 * x2029
                + x1051 * x2013
                + x1054 * x2025
                + x1061 * x2027
                + x1265 * x1906
                + x1266 * x1908
                + x1267 * x1912
                + x1900 * x2036
                + x1904 * x2029
            )
            - x1846 * (x1935 * x2031 + x1936 * x2037 + x1937 * x2038 + x1938 * x2039 + x1940 * x2035)
            - x1911 * x2019
            - x1999 * (x1910 * x2027 + x1913 * x2025 + x1922 * x2013 + x1930 * x2029 + x2036 * x329)
            - x2001 * (x1265 * x1923 + x1902 * x1921 * x2029 + x1925 * x2034 + x1928 * x2035)
            - x2002 * (x1427 * x2032 + x1590 * x1919 + x1918 * x2033)
            - x2003 * (x1419 * x2032 + x1916 * x2007)
            - x2004 * (x1029 * x1270 * x2036 + x1222 * x2039 + x1241 * x2038 + x1250 * x2037 + x1934 * x2035)
        )
        + x1126 * x1201 * x639
        + x1128 * x1204
        + x1130 * x1765
        + x1131 * x1769
        + x1135 * x1770
        + x1174
        * (
            dq_j3
            * x697
            * (
                x1152 * x1499
                + x1153 * x1499
                + x1157 * x1500
                + x1158 * x1500
                + x1162 * x1501
                + x1164 * x1501
                + x1167 * x1502
                + x1169 * x1503
                + x1170 * x1502
                + x1172 * x1503
            )
            + x1466 * x2019
            - x1504 * x1999
            - x1519 * x399
            - x1521 * x399
            - x1522 * x399
            - x1523 * x399
            - x1524 * x399
            + x1846 * (x1078 * x1500 + x1081 * x1501 + x1087 * x1502 + x1139 * x2031 + x1145 * x1503)
            + x2001 * (x1114 * x1264 + x1117 * x2017 + x1119 * x2025 + x1122 * x2027)
            + x2002 * (x1099 * x2022 + x1106 * x2013 + x1110 * x1590)
            + x2003 * (x1099 * x2020 + x1101 * x2007)
            + x2009 * (sigma_kin_v_4_3 * x1520 + x1098 * x2030 + x1116 * x2016 + x1126 * x1499 + x1135 * x2027)
        )
        + x135 * x1896
        + x135
        * (
            -x1268 * x1999
            - x1308 * x399
            - x1310 * x399
            - x1312 * x399
            - x1314 * x399
            - x1316 * x399
            + x1898
            * (
                x1263 * x53
                + x1263 * x60
                + x1264 * x82
                + x1264 * x86
                + x1265 * x130
                + x1265 * x134
                + x1266 * x179
                + x1266 * x183
                + x1267 * x197
                + x1267 * x201
            )
            + x2001 * (x1185 * x1264 + x1187 * x2025 + x1189 * x2027 + x1265 * x1757)
            + x2002 * (x1177 * x2022 + x1181 * x2013 + x1184 * x1590)
            + x2003 * (x1178 * x1266 + x1179 * x1267)
            + x2004 * (x1196 * x1263 + x1197 * x1500 + x1198 * x1501 + x1199 * x1502 + x1200 * x1503)
            + x2009 * (sigma_kin_v_4_3 * x1309 + x1191 * x1263 + x1192 * x2013 + x1193 * x2025 + x1194 * x2027)
            + x2019 * x211
        )
        - x1466 * x1774
        + x1569
        * (
            x1846 * x1859
            + x1849 * x2004
            + x1851 * x2000
            + x1853 * x2001
            + x1855 * x2002
            + x1857 * x2003
            - x1860 * x399
            + x1998 * x527
            + x1998 * x530
            - x1999 * x535
        )
        + x1764 * x1941
        - x1786 * x1954
        - x1802 * x1971
        - x1807 * x729
        + x1819 * x1978
        - x1828 * x1985
        - x1842 * x1995
        + x1944 * x476
        - x1949 * (x1467 * x1779 + x1469 * x1781)
        - x1959 * (x1478 * x1791 + x1480 * x1779 + x1482 * x1781)
        - x1967 * (x1444 * x1966 + x1487 * x1791 + x1489 * x1779 + x1491 * x1781)
        - x1979 * (x1442 * x1617 + x1444 * x1611 + x1609 * x1790 + x1613 * x1778 + x1615 * x1780)
        - x1987 * x1988
        + x2018 * x656
        + x794
        * (
            dq_i6 * x1997 * (x1267 * x1362 + x1267 * x787 + x2010 * x783 + x2010 * x788)
            + dq_j3 * x1267 * x2005
            - x1623 * x1999
            - x1633 * x399
            - x1635 * x399
            + x1846 * (x1267 * x756 + x2006 * x750)
            + x2001 * (x2006 * x480 + x2008 * x771)
            + x2002 * (x2007 * x779 + x2010 * x776)
            + x2004 * (x2006 * x757 + x2007 * x763)
            + x2009 * (x2006 * x471 + x2008 * x765)
        )
        + x864
        * (
            dq_i5 * x1997 * (x1373 * x1590 + x1590 * x857 + x2010 * x851 + x2010 * x854 + x2013 * x850 + x2013 * x852)
            + x1366 * x2011 * x2012
            - x1654 * x1999
            - x1664 * x399
            - x1666 * x399
            - x1667 * x399
            + x1846 * (x2011 * x828 + x2014 * x569 + x2015 * x750)
            + x2001 * (x1372 * x1590 + x2015 * x480 + x2017 * x813)
            + x2003 * (x2007 * x822 + x2010 * x821)
            + x2004 * (x2011 * x759 + x2014 * x576 + x2015 * x757)
            + x2009 * (x2011 * x765 + x2015 * x471 + x2016 * x835)
        )
        + x962
        * (
            dq_i4
            * x1997
            * (
                x1265 * x945
                + x1265 * x947
                + x2025 * x951
                + x2025 * x956
                + x2027 * x955
                + x2027 * x959
                + x2029 * x940
                + x2029 * x941
            )
            + x1376 * x2019
            - x1691 * x1999
            - x1710 * x399
            - x1711 * x399
            - x1712 * x399
            - x1713 * x399
            + x1846 * (x2023 * x610 + x2024 * x869 + x2026 * x457 + x2028 * x459)
            + x2002 * (x1590 * x908 + x2021 * x591 + x2022 * x885)
            + x2003 * (x2007 * x896 + x2020 * x885)
            + x2004 * (x2023 * x618 + x2024 * x920 + x2026 * x464 + x2028 * x466)
            + x2009 * (x2021 * x581 + x2027 * x935 + x2029 * x927 + x2030 * x931)
        )
    )
    K_block_list.append(
        x1044 * x1456
        + x1045 * x1456
        + x1048 * x2040
        + x1049 * x2040
        + x1059 * x116
        + x1066 * x163
        + x1069 * x194
        + x1070 * x2044
        + x1070 * x2050
        + x1070 * x2057
        + x1070 * x2059
        + x1070 * x2064
        + x1070
        * (
            2 * dq_i1 * (x1021 * x2233 + x1027 * x2215 + x2080 + x2235 * x610 + x2236 * x869 + x2237 * x457)
            + 2 * dq_i2 * (x1030 * x2234 + x1038 * x2215 + x2081 + x2235 * x618 + x2236 * x920 + x2237 * x464)
            + 2 * dq_i3 * dq_i7 * dq_j3 * x139 * x142 * x145 * x147 * x149 * x152 * x1816 * x208 * x390 * x975
            - dq_i3 * x2084
            - dq_i3 * x2085
            - dq_i3 * x2086
            - dq_i3 * x2087
            - dq_i3 * x2088
            - dq_i3 * x2089
            - dq_i3 * x2090
            - dq_i3 * x2091
            - dq_i3 * x2092
            - dq_i3 * x2093
            + dq_i3
            * (
                x1042 * x2065 * x54
                + x1044 * x2234
                + x1048 * x2219
                + x1049 * x2219
                + x1051 * x2178
                + x1053 * x2178
                + x1054 * x2212
                + x1057 * x2212
                + x1061 * x2215
                + x1063 * x2215
                + x2094
            )
            + 2 * dq_i4 * (x1010 * x2219 + x1014 * x2212 + x1018 * x2215 + x1742 * x2184 + x2076)
            + 2 * dq_i5 * (x1003 * x2178 + x1004 * x2153 + x1006 * x2182 + x2063)
            + 2 * dq_i6 * (x2053 + x2153 * x995 + x2156 * x997)
            + 2 * dq_i7 * sigma_kin_v_7_3 * x139 * x142 * x145 * x147 * x149 * x152 * x192 * x208 * x390 * x975
            - x2044
            - x2050
            - x2057
            - x2059
            - x2064
            - x2070 * x2112
            - x521 * x994
        )
        + x109 * x2041 * (sigma_pot_3_c * x48 + sigma_pot_3_s * x45)
        + x116 * x2042
        + x1174
        * (
            2 * dq_i1 * (x1078 * x2229 + x1081 * x2230 + x1087 * x2231 + x1139 * x2233 + x1145 * x2232 + x1987)
            + dq_i2
            * (
                x1152 * x2234
                + x1153 * x2234
                + x1155 * x1974 * x54
                + x1157 * x2229
                + x1160 * x1975 * x54
                + x1162 * x2230
                + x1163 * x1976 * x54
                + x1167 * x2231
                + x1168 * x1977 * x54
                + x1169 * x2232
                + x1996
            )
            + 2 * dq_i3 * dq_i7 * dq_j3 * x1083 * x138 * x143 * x145 * x147 * x149 * x152 * x1816 * x208 * x390
            + 2 * dq_i3 * (x1126 * x2234 + x1128 * x2207 + x1130 * x2178 + x1131 * x2212 + x1135 * x2215 + x1995)
            + 2 * dq_i4 * (x1114 * x2207 + x1119 * x2212 + x1122 * x2215 + x1763 * x2184 + x1985)
            + 2 * dq_i5 * (x1106 * x2178 + x1107 * x2153 + x1110 * x2182 + x1971)
            + 2 * dq_i6 * (x1100 * x2153 + x1101 * x2156 + x1954)
            + 2 * dq_i7 * sigma_kin_v_7_3 * x1083 * x138 * x143 * x145 * x147 * x149 * x152 * x192 * x208 * x390
            - x1096 * x521
            - x1943
            - x1948
            - x1958
            - x1965
            - x1972
            - x1978 * x2112
            - x1990 * x405
            - x1991 * x405
            - x1992 * x405
            - x1993 * x405
            - x1994 * x405
        )
        + x1272
        * (
            ddq_i3 * (x1808 * x640 + x1810 * x607 + x1812 * x566 + x1814 * x452 + x1816 * x454)
            + x117 * x120 * x122 * x126 * x1812 * x340 * x564 * x654
            + x139 * x142 * x145 * x147 * x149 * x153 * x1816 * x348 * x390 * x654
            + x164 * x167 * x169 * x171 * x175 * x1814 * x344 * x450 * x654
            + x1808 * x328 * x42 * x49 * x639 * x654
            + x1810 * x335 * x605 * x65 * x654 * x74 * x78
            - x2138 * x631
            - x478 * (x2140 * x220 + x2141 * x495)
            - x585 * (x2138 * x490 + x2139 * x261 + x2142 * x253)
            - x624 * (x1812 * x587 + x2143 * x282 + x2144 * x295 + x2145 * x302)
            - x649 * (x1809 * x59 + x1811 * x610 + x1813 * x569 + x1815 * x457 + x1817 * x459)
            - x652 * (x1973 * x363 + x1974 * x365 + x1975 * x371 + x1976 * x377 + x1977 * x383)
            - x655 * (x2065 * x328 + x2066 * x337 + x2067 * x581 + x2068 * x471 + x2069 * x475)
        )
        + x135 * x1803 * x994
        + x135
        * (
            dq_i1
            * (
                x129 * x2230
                + x133 * x2230
                + x178 * x2231
                + x1809 * x39 * x54
                + x182 * x2231
                + x1843
                + x196 * x2232
                + x200 * x2232
                + x2228 * x60
                + x2229 * x81
                + x2229 * x85
            )
            + 2 * dq_i2 * (x1196 * x2228 + x1197 * x2229 + x1198 * x2230 + x1199 * x2231 + x1200 * x2232 + x1834)
            + 2 * dq_i3 * dq_i7 * dq_j3 * x136 * x137 * x139 * x143 * x145 * x147 * x149 * x152 * x1816 * x208
            + 2 * dq_i3 * (x1191 * x2228 + x1192 * x2178 + x1193 * x2212 + x1194 * x2215 + x1841 + x2207 * x338)
            + 2 * dq_i4 * (x1185 * x2207 + x1187 * x2212 + x1189 * x2215 + x1757 * x2184 + x1827)
            + 2 * dq_i5 * (x1181 * x2178 + x1182 * x2153 + x1184 * x2182 + x1801)
            + 2 * dq_i6 * (x1178 * x2153 + x1179 * x2155 + x1785)
            + 2 * dq_i7 * sigma_kin_v_7_3 * x136 * x137 * x139 * x143 * x145 * x147 * x149 * x152 * x192 * x208
            - x1767
            - x1777
            - x1789
            - x1794
            - x1805
            - x1818 * x2112
            - x1836 * x405
            - x1837 * x405
            - x1838 * x405
            - x1839 * x405
            - x1840 * x405
            - x271 * x521
        )
        - x1391 * x1774
        - x143
        * x1847
        * x403
        * x426
        * (-x1527 * x403 + x1528 * x403 + x1529 * x2082 + x1534 * x403 + x1535 * x403 + x2083 * x411 + x2083 * x415)
        + x1598 * x1877
        + x163 * x2045
        - x1786 * x2053
        - x1802 * x2063
        + x1819 * x2070
        - x1828 * x2076
        - x1835 * x2081
        + x194 * x2046
        - x1988 * x2080
        + x2018 * x699
        - 4 * x2094 * x54
        + x2113
        * (
            x1568
            + x186 * x527
            + x186 * x530
            - x2107
            + x2108 * x520
            + x2109 * x508
            + x2110 * x512
            + x2110 * x514
            + x2111 * x411
            + x2111 * x415
            - x2112 * x535
            + x533
        )
        + x500
        * (
            ddq_i6 * x2098
            + 2 * dq_i3 * dq_i6 * sigma_kin_v_6_3 * sigma_kin_v_6_6 * x164 * x167 * x169 * x171 * x175 * x344 * x450
            + 2
            * dq_i3
            * dq_i6
            * sigma_kin_v_7_3
            * sigma_kin_v_7_6
            * x139
            * x142
            * x145
            * x147
            * x149
            * x153
            * x348
            * x390
            - x2095 * x449
            - x2101 * x462
            - x2102 * x468
            - x2103 * x478
            - x2104 * x486
            - x2105 * x493
            - x2106 * x499
        )
        + x596
        * (
            ddq_i5 * x2117
            + 2 * dq_i3 * dq_i5 * sigma_kin_v_5_3 * sigma_kin_v_5_5 * x117 * x120 * x122 * x126 * x340 * x564
            + 2 * dq_i3 * dq_i5 * sigma_kin_v_6_3 * sigma_kin_v_6_5 * x164 * x167 * x169 * x171 * x175 * x344 * x450
            + 2
            * dq_i3
            * dq_i5
            * sigma_kin_v_7_3
            * sigma_kin_v_7_5
            * x139
            * x142
            * x145
            * x147
            * x149
            * x153
            * x348
            * x390
            - x2114 * x561
            - x2115 * x493
            - x2121 * x574
            - x2122 * x578
            - x2123 * x585
            - x2124 * x589
            - x2125 * x595
        )
        + x628
        * (
            ddq_i4 * x2129
            + 2 * dq_i3 * dq_i4 * sigma_kin_v_4_3 * sigma_kin_v_4_4 * x335 * x605 * x65 * x74 * x78
            + 2 * dq_i3 * dq_i4 * sigma_kin_v_5_3 * sigma_kin_v_5_4 * x117 * x120 * x122 * x126 * x340 * x564
            + 2 * dq_i3 * dq_i4 * sigma_kin_v_6_3 * sigma_kin_v_6_4 * x164 * x167 * x169 * x171 * x175 * x344 * x450
            + 2
            * dq_i3
            * dq_i4
            * sigma_kin_v_7_3
            * sigma_kin_v_7_4
            * x139
            * x142
            * x145
            * x147
            * x149
            * x153
            * x348
            * x390
            - x2126 * x599
            - x2127 * x486
            - x2128 * x589
            - x2133 * x616
            - x2134 * x620
            - x2136 * x624
            - x2137 * x627
        )
        + x794
        * (
            2 * dq_i1 * (x1772 * x756 + x2146 * x750 + x2154 * x750 + x2155 * x756)
            + 2 * dq_i2 * (x1952 * x763 + x2146 * x757 + x2154 * x757 + x2156 * x763)
            + 2 * dq_i3 * dq_i7 * dq_j3 * x139 * x143 * x145 * x147 * x148 * x152 * x1816 * x208 * x390 * x730
            + 2 * dq_i3 * (x2147 + x2149 + x2154 * x471 + x2157 * x765)
            + 2 * dq_i4 * (x2154 * x480 + x2157 * x771 + x2158 + x2159)
            + 2 * dq_i5 * (x2156 * x779 + x2160 * x776 + x2162 + x2163)
            + dq_i6 * (x2140 * x54 * x781 + x2155 * x787 + x2156 * x785 + x2160 * x783 + x2164 + x2165 + x2166 + x2167)
            + 2 * dq_i7 * sigma_kin_v_7_3 * x139 * x143 * x145 * x147 * x148 * x152 * x192 * x208 * x390 * x730
            - x1623 * x278
            - x2043 * x733
            - x2112 * (x2140 * x735 + x2141 * x732)
            - x2147 * x405
            - x2149 * x405
            - x2150 * x248
            - x2151 * x242
            - x2152 * x216
            - x521 * x743
        )
        + x864
        * (
            2 * dq_i1 * (x2169 * x750 + x2171 * x828 + x2179 * x569 + x2180 * x750 + x2181 * x569 + x2183 * x828)
            + 2 * dq_i2 * (x2169 * x757 + x2171 * x759 + x2179 * x576 + x2180 * x757 + x2181 * x576 + x2183 * x759)
            + 2 * dq_i3 * dq_i7 * dq_j3 * x139 * x143 * x145 * x146 * x149 * x152 * x1816 * x208 * x390 * x799
            + 2 * dq_i3 * (x2168 + x2170 + x2172 + x2178 * x836 + x2180 * x471 + x2183 * x765)
            + 2 * dq_i4 * (x1372 * x2182 + x1669 * x2184 + x2180 * x480 + x2185 + x2186 + x2187)
            + dq_i5
            * (
                x1373 * x2182
                + x2142 * x54 * x848
                + x2160 * x851
                + x2160 * x854
                + x2178 * x850
                + x2182 * x857
                + x2188
                + x2189
                + x2190
                + x2191
                + x2192
                + x2193
            )
            + 2 * dq_i6 * (x2156 * x822 + x2160 * x821 + x2176 + x2177)
            + 2 * dq_i7 * sigma_kin_v_7_3 * x139 * x143 * x145 * x146 * x149 * x152 * x192 * x208 * x390 * x799
            - x1654 * x278
            - x2043 * x802
            - x2112 * (x2138 * x801 + x2139 * x804 + x2142 * x795)
            - x216 * x2173
            - x2168 * x405
            - x2170 * x405
            - x2172 * x405
            - x2174 * x248
            - x2175 * x242
            - x521 * x816
        )
        + x962
        * (
            2
            * dq_i1
            * (
                x2208 * x610
                + x2209 * x610
                + x2211 * x869
                + x2213 * x457
                + x2214 * x869
                + x2216 * x459
                + x2217 * x457
                + x2218 * x459
            )
            + 2
            * dq_i2
            * (
                x2208 * x618
                + x2209 * x618
                + x2211 * x920
                + x2213 * x464
                + x2214 * x920
                + x2216 * x466
                + x2217 * x464
                + x2218 * x466
            )
            + 2 * dq_i3 * dq_i7 * dq_j3 * x139 * x143 * x144 * x147 * x149 * x152 * x1816 * x208 * x390 * x875
            + 2 * dq_i3 * (x2194 + x2196 + x2197 + x2198 + x2210 * x581 + x2212 * x932 + x2215 * x935 + x2219 * x927)
            + dq_i4
            * (
                x2143 * x54 * x938
                + x2144 * x54 * x948
                + x2145 * x54 * x953
                + x2184 * x945
                + x2184 * x947
                + x2212 * x951
                + x2215 * x955
                + x2219 * x940
                + x2220
                + x2221
                + x2222
                + x2223
                + x2224
                + x2225
                + x2226
                + x2227
            )
            + 2 * dq_i5 * (x2153 * x904 + x2182 * x908 + x2184 * x903 + x2204 + x2205 + x2206)
            + 2 * dq_i6 * (x2153 * x894 + x2156 * x896 + x2202 + x2203)
            + 2 * dq_i7 * sigma_kin_v_7_3 * x139 * x143 * x144 * x147 * x149 * x152 * x192 * x208 * x390 * x875
            - x1691 * x278
            - x2043 * x877
            - x2112 * (x1812 * x889 + x2143 * x865 + x2144 * x879 + x2145 * x875)
            - x216 * x2199
            - x2194 * x405
            - x2196 * x405
            - x2197 * x405
            - x2198 * x405
            - x2200 * x242
            - x2201 * x248
            - x521 * x892
        )
    )
    K_block_list.append(
        -dq_i1 * x2303 * x962
        + x1070
        * (
            -x1011 * x506
            - x1015 * x506
            - x1019 * x506
            - x1020 * x506
            + x1391 * x2330
            + x2306 * (x1038 * x2334 + x2339 * x618 + x2340 * x920 + x2341 * x464)
            + x2313 * (x1003 * x2326 + x1006 * x598 + x2331 * x986)
            + x2315 * (x2321 * x997 + x2335 * x986)
            + x2317 * (x1027 * x2334 + x2339 * x610 + x2340 * x869 + x2341 * x457)
            + x2323 * (x1010 * x622 + x1014 * x2333 + x1018 * x2334 + x2329 * x989)
            - x2338
            + x2342
            * (
                x1048 * x622
                + x1049 * x622
                + x1059 * x603
                + x1066 * x600
                + x1069 * x597
                + x2042 * x603
                + x2045 * x600
                + x2046 * x597
            )
        )
        + x1174
        * (
            -x1115 * x506
            - x1120 * x506
            - x1123 * x506
            - x1124 * x506
            + x1466 * x2330
            + x2309 * (x1128 * x2332 + x1130 * x2326 + x1131 * x2333 + x1135 * x2334)
            + x2313 * (x1099 * x2331 + x1106 * x2326 + x1110 * x598)
            + x2315 * (x1099 * x2335 + x1101 * x2321)
            + x2317 * (x1078 * x611 + x1081 * x612 + x1087 * x613 + x1145 * x614)
            + x2323 * (x1114 * x2332 + x1117 * x2329 + x1119 * x2333 + x1122 * x2334)
            - x2336
            + x2337
            * (
                x1157 * x611
                + x1158 * x611
                + x1162 * x612
                + x1164 * x612
                + x1167 * x613
                + x1169 * x614
                + x1170 * x613
                + x1172 * x614
            )
        )
        + x1185 * x1204
        + x1187 * x1769
        + x1189 * x1770
        + x1205 * x1757
        - x1227 * (x2252 * x473 + x2254 * x442)
        - x1244 * (x2252 * x562 + x2254 * x558 + x2263 * x583)
        - x1274 * x2269
        - x1292 * (x1429 * x715 + x1433 * x718 + x1435 * x720 + x2263 * x316)
        + x135 * x2265
        + x135
        * (
            dq_j4
            * x11
            * (
                x129 * x612
                + x133 * x612
                + x178 * x613
                + x182 * x613
                + x196 * x614
                + x200 * x614
                + x611 * x81
                + x611 * x85
            )
            + x211 * x2330
            - x2265
            + x2306 * (x1197 * x611 + x1198 * x612 + x1199 * x613 + x1200 * x614)
            + x2309 * (x1192 * x2326 + x1193 * x2333 + x1194 * x2334 + x2332 * x338)
            + x2313 * (x1177 * x2331 + x1181 * x2326 + x1184 * x598)
            + x2315 * (x1178 * x600 + x1179 * x597)
            + x2323 * (x1185 * x2332 + x1187 * x2333 + x1189 * x2334 + x1757 * x603)
            - x287 * x506
            - x293 * x506
            - x300 * x506
            - x307 * x506
        )
        - x1618 * x2268
        + x1768 * x484
        - x211 * x2247
        + x2239 * x2241
        + x2243 * x962
        + x2250 * x962
        - x2258 * x2259
        + x2262 * x962
        + x2266 * x962
        + x2267 * x962
        + x2278 * x2279
        - x2285 * x2286
        - x2292 * x2293
        - x2296 * x2297
        - x2301 * x2302
        + x2324 * x628
        + x557
        * (
            x1335 * x2312
            - x1340 * x1350
            - x2264 * x538
            + x2305 * x527
            + x2305 * x530
            + x2306 * x2308
            + x2309 * x2310
            + x2313 * x2314
            + x2315 * x2316
            + x2317 * x2318
        )
        + x794
        * (
            dq_i6 * x2304 * (x1362 * x597 + x597 * x787 + x601 * x783 + x601 * x788)
            + x1352 * x2319 * x597
            - x2264 * x744
            + x2306 * (x2320 * x757 + x2321 * x763)
            + x2309 * (x2320 * x471 + x2322 * x765)
            + x2313 * (x2321 * x779 + x601 * x776)
            + x2317 * (x2320 * x750 + x597 * x756)
            + x2323 * (x2320 * x480 + x2322 * x771)
            - x506 * x772
            - x506 * x773
        )
        + x864
        * (
            dq_i5 * x2304 * (x1373 * x598 + x2326 * x850 + x2326 * x852 + x598 * x857 + x601 * x851 + x601 * x854)
            + x1366 * x2319 * x2325
            - x2264 * x817
            + x2306 * (x2325 * x759 + x2327 * x576 + x2328 * x757)
            + x2309 * (x2325 * x765 + x2326 * x836 + x2328 * x471)
            + x2315 * (x2321 * x822 + x601 * x821)
            + x2317 * (x2325 * x828 + x2327 * x569 + x2328 * x750)
            + x2323 * (x1372 * x598 + x2328 * x480 + x2329 * x813)
            - x506 * x842
            - x506 * x846
            - x506 * x847
        )
        + x962
        * (
            2 * dq_i1 * dq_i2 * dq_j4 * (x2345 * x618 + x2346 * x920 + x2347 * x464 + x2348 * x466)
            + 2 * dq_i1 * dq_i3 * dq_j4 * (x2333 * x932 + x2334 * x935 + x2343 * x581 + x622 * x927)
            + dq_i1
            * dq_i4
            * dq_j4
            * (
                x2333 * x951
                + x2333 * x956
                + x2334 * x955
                + x2334 * x959
                + x603 * x945
                + x603 * x947
                + x622 * x940
                + x622 * x941
            )
            + 2 * dq_i1 * dq_i5 * dq_j4 * (x2331 * x885 + x2343 * x591 + x598 * x908)
            + 2 * dq_i1 * dq_i6 * dq_j4 * (x2321 * x896 + x2335 * x885)
            + 2
            * dq_i1
            * dq_i7
            * dq_j4
            * sigma_kin_v_7_1
            * sigma_kin_v_7_4
            * x139
            * x143
            * x144
            * x147
            * x149
            * x152
            * x208
            * x390
            * x875
            - dq_i1 * x942
            - dq_i1 * x944
            - dq_i1 * x949
            - dq_i1 * x952
            - dq_i1 * x957
            - dq_i1 * x958
            - dq_i1 * x960
            - dq_i1 * x961
            + 2 * dq_j4 * x11 * (x2345 * x610 + x2346 * x869 + x2347 * x457 + x2348 * x459)
            - x2344
        )
    )
    K_block_list.append(
        -dq_i2 * x2382 * x962
        + x1070
        * (
            x1391 * x2403
            - x1733 * x2385
            - x1741 * x399
            - x1743 * x399
            - x1744 * x399
            - x1745 * x399
            + x2306 * (x1027 * x2405 + x2408 * x610 + x2409 * x869 + x2410 * x457)
            + x2387 * (x1003 * x2399 + x1006 * x1579 + x2407 * x261)
            + x2388 * (x222 * x2407 + x2392 * x997)
            + x2389 * (x1038 * x2405 + x2408 * x618 + x2409 * x920 + x2410 * x464)
            + x2394 * (x1010 * x1966 + x1014 * x2404 + x1018 * x2405 + x2402 * x989)
            + x2411
            * (
                x1048 * x1966
                + x1049 * x1966
                + x1059 * x1487
                + x1066 * x1489
                + x1069 * x1491
                + x1487 * x2042
                + x1489 * x2045
                + x1491 * x2046
            )
        )
        + x1114 * x1204
        + x1119 * x1769
        + x1122 * x1770
        + x1174
        * (
            dq_j4
            * x697
            * (
                x1157 * x1486
                + x1158 * x1486
                + x1162 * x1488
                + x1164 * x1488
                + x1167 * x1490
                + x1169 * x1492
                + x1170 * x1490
                + x1172 * x1492
            )
            + x1466 * x2403
            - x1493 * x2385
            - x1513 * x399
            - x1514 * x399
            - x1515 * x399
            - x1516 * x399
            + x2306 * (x1078 * x1486 + x1081 * x1488 + x1087 * x1490 + x1145 * x1492)
            + x2386 * (sigma_kin_v_4_4 * x1520 + x1130 * x2399 + x1131 * x2404 + x1135 * x2405)
            + x2387 * (x1106 * x2399 + x1110 * x1579 + x2406 * x261)
            + x2388 * (x1101 * x2392 + x222 * x2406)
            + x2394 * (x1114 * x1246 + x1117 * x2402 + x1119 * x2404 + x1122 * x2405)
        )
        + x1205 * x1763
        - x1273 * x2365
        + x135 * x2336
        + x135
        * (
            -x1247 * x2385
            - x1294 * x399
            - x1296 * x399
            - x1299 * x399
            - x1302 * x399
            + x211 * x2403
            + x2337
            * (
                x1246 * x82
                + x1246 * x86
                + x130 * x1487
                + x134 * x1487
                + x1489 * x179
                + x1489 * x183
                + x1491 * x197
                + x1491 * x201
            )
            + x2386 * (sigma_kin_v_4_4 * x1309 + x1192 * x2399 + x1193 * x2404 + x1194 * x2405)
            + x2387 * (x1181 * x2399 + x1182 * x1489 + x1184 * x1579)
            + x2388 * (x1178 * x1489 + x1179 * x1491)
            + x2389 * (x1197 * x1486 + x1198 * x1488 + x1199 * x1490 + x1200 * x1492)
            + x2394 * (x1185 * x1246 + x1187 * x2404 + x1189 * x2405 + x1487 * x1757)
        )
        - x1466 * x2247
        + x1569
        * (
            x2000 * x2312
            + x2306 * x2318
            + x2308 * x2389
            + x2310 * x2386
            + x2314 * x2387
            + x2316 * x2388
            + x2384 * x527
            + x2384 * x530
            - x2385 * x538
            - x2390 * x399
        )
        + x1944 * x484
        - x1949 * (x1467 * x2252 + x1469 * x2254)
        - x1959 * (x1478 * x2263 + x1480 * x2252 + x1482 * x2254)
        - x1979 * (x1283 * x2263 + x1429 * x1612 + x1433 * x1614 + x1435 * x1616)
        - x2259 * x2360
        - x2268 * x729
        + x2279 * x2370
        - x2286 * x2374
        - x2297 * x2377
        - x2302 * x2381
        + x2349 * x2350
        + x2352 * x962
        + x2355 * x962
        + x2362 * x962
        + x2363 * x962
        + x2364 * x962
        - x2375 * x2376
        + x2396 * x628
        + x794
        * (
            dq_i6 * x2383 * (x1362 * x1491 + x1491 * x787 + x2395 * x783 + x2395 * x788)
            + dq_j4 * x1491 * x2005
            - x1624 * x2385
            - x1636 * x399
            - x1637 * x399
            + x2306 * (x1491 * x756 + x2391 * x750)
            + x2386 * (x2391 * x471 + x2393 * x765)
            + x2387 * (x2392 * x779 + x2395 * x776)
            + x2389 * (x2391 * x757 + x2392 * x763)
            + x2394 * (x2391 * x480 + x2393 * x771)
        )
        + x864
        * (
            dq_i5 * x2383 * (x1373 * x1579 + x1579 * x857 + x2395 * x851 + x2395 * x854 + x2399 * x850 + x2399 * x852)
            + x1366 * x2397 * x2398
            - x1655 * x2385
            - x1670 * x399
            - x1671 * x399
            - x1672 * x399
            + x2306 * (x2397 * x828 + x2400 * x569 + x2401 * x750)
            + x2386 * (x2397 * x765 + x2399 * x836 + x2401 * x471)
            + x2388 * (x2392 * x822 + x2395 * x821)
            + x2389 * (x2397 * x759 + x2400 * x576 + x2401 * x757)
            + x2394 * (x1372 * x1579 + x2401 * x480 + x2402 * x813)
        )
        + x962
        * (
            2 * dq_i1 * dq_i2 * dq_j4 * (x2414 * x610 + x2415 * x869 + x2416 * x457 + x2417 * x459)
            + 2 * dq_i2 * dq_i3 * dq_j4 * (x1966 * x927 + x2404 * x932 + x2405 * x935 + x2413 * x581)
            + dq_i2
            * dq_i4
            * dq_j4
            * (
                x1487 * x945
                + x1487 * x947
                + x1966 * x940
                + x1966 * x941
                + x2404 * x951
                + x2404 * x956
                + x2405 * x955
                + x2405 * x959
            )
            + 2 * dq_i2 * dq_i5 * dq_j4 * (x1579 * x908 + x2412 * x261 + x2413 * x591)
            + 2 * dq_i2 * dq_i6 * dq_j4 * (x222 * x2412 + x2392 * x896)
            + 2
            * dq_i2
            * dq_i7
            * dq_j4
            * sigma_kin_v_7_2
            * sigma_kin_v_7_4
            * x139
            * x143
            * x144
            * x147
            * x149
            * x152
            * x208
            * x390
            * x875
            - dq_i2 * x1714
            - dq_i2 * x1715
            - dq_i2 * x1716
            - dq_i2 * x1717
            - dq_i2 * x1718
            - dq_i2 * x1719
            - dq_i2 * x1720
            - dq_i2 * x1721
            + 2 * dq_j4 * x697 * (x2414 * x618 + x2415 * x920 + x2416 * x464 + x2417 * x466)
            - x1692 * x2385
        )
    )
    K_block_list.append(
        -dq_i3 * x2459 * x962
        + x1010 * x1204 * x606
        + x1014 * x1769
        + x1018 * x1770
        + x1070
        * (
            dq_j4
            * x654
            * (
                x1048 * x2135
                + x1049 * x2135
                + x1059 * x1961
                + x1066 * x1962
                + x1069 * x1963
                + x1961 * x2042
                + x1962 * x2045
                + x1963 * x2046
            )
            + x1391 * x2480
            - x2058 * x2462
            - x2072 * x405
            - x2073 * x405
            - x2074 * x405
            - x2075 * x405
            + x2309 * (x1027 * x2482 + x2485 * x610 + x2486 * x869 + x2487 * x457)
            + x2386 * (x1038 * x2482 + x2485 * x618 + x2486 * x920 + x2487 * x464)
            + x2464 * (x1003 * x2476 + x1006 * x2126 + x2484 * x261)
            + x2465 * (x222 * x2484 + x2469 * x997)
            + x2471 * (x1010 * x2135 + x1014 * x2481 + x1018 * x2482 + x2479 * x989)
        )
        + x1174
        * (
            x1466 * x2480
            - x1964 * x2462
            - x1980 * x405
            - x1982 * x405
            - x1983 * x405
            - x1984 * x405
            + x2309 * (x1078 * x1960 + x1081 * x2130 + x1087 * x2131 + x1145 * x2132)
            + x2411
            * (
                x1157 * x1960
                + x1158 * x1960
                + x1162 * x2130
                + x1164 * x2130
                + x1167 * x2131
                + x1169 * x2132
                + x1170 * x2131
                + x1172 * x2132
            )
            + x2464 * (x1106 * x2476 + x1110 * x2126 + x2483 * x261)
            + x2465 * (x1963 * x2358 + x222 * x2483)
            + x2466 * (x1128 * x1792 + x1130 * x2476 + x1131 * x2481 + x1135 * x2482)
            + x2471 * (x1114 * x1792 + x1119 * x2481 + x1122 * x2482 + x1763 * x1961)
        )
        + x1205 * x1742
        + x135 * x2338
        + x135
        * (
            -x1793 * x2462
            - x1821 * x405
            - x1822 * x405
            - x1824 * x405
            - x1826 * x405
            + x211 * x2480
            + x2342
            * (
                x130 * x1961
                + x134 * x1961
                + x179 * x1962
                + x1792 * x82
                + x1792 * x86
                + x183 * x1962
                + x1963 * x197
                + x1963 * x201
            )
            + x2386 * (x1197 * x1960 + x1198 * x2130 + x1199 * x2131 + x1200 * x2132)
            + x2464 * (x1181 * x2476 + x1182 * x1962 + x1184 * x2126)
            + x2465 * (x1178 * x1962 + x1179 * x1963)
            + x2466 * (x1192 * x2476 + x1193 * x2481 + x1194 * x2482 + x1792 * x338)
            + x2471 * (x1185 * x1792 + x1187 * x2481 + x1189 * x2482 + x1757 * x1961)
        )
        - x1391 * x2247
        + x2113
        * (
            x2308 * x2386
            + x2309 * x2318
            + x2310 * x2466
            + x2312 * x2463
            + x2314 * x2464
            + x2316 * x2465
            - x2390 * x405
            + x2461 * x527
            + x2461 * x530
            - x2462 * x538
        )
        - x2259 * x2430
        - x2269 * x2437
        + x2279 * x2443
        - x2286 * x2448
        - x2293 * x2453
        - x2302 * x2458
        + x2349 * x2418
        - x2365 * x2438
        - x2376 * x2452
        + x2420 * x962
        + x2421 * x484
        + x2424 * x962
        - x2425 * (x2047 * x2252 + x2048 * x2254)
        + x2432 * x962
        - x2433 * (x1956 * x2263 + x2054 * x2252 + x2055 * x2254)
        + x2434 * x962
        + x2435 * x962
        - x2444 * (x1429 * x2143 + x1433 * x2144 + x1435 * x2145 + x1812 * x2263)
        + x2473 * x628
        + x794
        * (
            dq_i6 * x2460 * (x1362 * x1963 + x1963 * x787 + x2472 * x783 + x2472 * x788)
            + dq_j4 * x1963 * x2467
            - x2150 * x2462
            - x2158 * x405
            - x2159 * x405
            + x2309 * (x1963 * x756 + x2468 * x750)
            + x2386 * (x2468 * x757 + x2469 * x763)
            + x2464 * (x2469 * x779 + x2472 * x776)
            + x2466 * (x2468 * x471 + x2470 * x765)
            + x2471 * (x2468 * x480 + x2470 * x771)
        )
        + x864
        * (
            dq_i5 * x2460 * (x1373 * x2126 + x2126 * x857 + x2472 * x851 + x2472 * x854 + x2476 * x850 + x2476 * x852)
            + x1366 * x2474 * x2475
            - x2174 * x2462
            - x2185 * x405
            - x2186 * x405
            - x2187 * x405
            + x2309 * (x2474 * x828 + x2477 * x569 + x2478 * x750)
            + x2386 * (x2474 * x759 + x2477 * x576 + x2478 * x757)
            + x2465 * (x2469 * x822 + x2472 * x821)
            + x2466 * (x2474 * x765 + x2476 * x836 + x2478 * x471)
            + x2471 * (x1372 * x2126 + x2478 * x480 + x2479 * x813)
        )
        + x962
        * (
            2 * dq_i1 * dq_i3 * dq_j4 * (x2490 * x610 + x2491 * x869 + x2492 * x457 + x2493 * x459)
            + 2 * dq_i2 * dq_i3 * dq_j4 * (x2490 * x618 + x2491 * x920 + x2492 * x464 + x2493 * x466)
            + dq_i3
            * dq_i4
            * dq_j4
            * (
                x1961 * x945
                + x1961 * x947
                + x2135 * x940
                + x2135 * x941
                + x2481 * x951
                + x2481 * x956
                + x2482 * x955
                + x2482 * x959
            )
            + 2 * dq_i3 * dq_i5 * dq_j4 * (x2126 * x908 + x2488 * x261 + x2489 * x591)
            + 2 * dq_i3 * dq_i6 * dq_j4 * (x222 * x2488 + x2469 * x896)
            + 2
            * dq_i3
            * dq_i7
            * dq_j4
            * sigma_kin_v_7_3
            * sigma_kin_v_7_4
            * x139
            * x143
            * x144
            * x147
            * x149
            * x152
            * x208
            * x390
            * x875
            - dq_i3 * x2220
            - dq_i3 * x2221
            - dq_i3 * x2222
            - dq_i3 * x2223
            - dq_i3 * x2224
            - dq_i3 * x2225
            - dq_i3 * x2226
            - dq_i3 * x2227
            + 2 * dq_j4 * x654 * (x2135 * x927 + x2481 * x932 + x2482 * x935 + x2489 * x581)
            - x2201 * x2462
        )
    )
    K_block_list.append(
        x1070
        * (
            2 * dq_i1 * (x1027 * x2633 + x2452 + x2634 * x610 + x2635 * x869 + x2636 * x457)
            + 2 * dq_i2 * (x1038 * x2633 + x2453 + x2634 * x618 + x2635 * x920 + x2636 * x464)
            + dq_i3
            * (
                x1048 * x2637
                + x1049 * x2637
                + x1051 * x2612
                + x1053 * x2612
                + x1054 * x2632
                + x1057 * x2632
                + x1061 * x2633
                + x1063 * x2633
                + x2459
            )
            + 2 * dq_i4 * dq_i7 * dq_j4 * x139 * x142 * x145 * x147 * x149 * x152 * x208 * x2276 * x390 * x975
            + 2 * dq_i4 * (x1010 * x2637 + x1014 * x2632 + x1018 * x2633 + x1742 * x2619 + x2458)
            + 2 * dq_i5 * (x1003 * x2612 + x1004 * x2592 + x1006 * x2616 + x2448)
            + 2 * dq_i6 * (x2430 + x2592 * x995 + x2595 * x997)
            + 2 * dq_i7 * sigma_kin_v_7_4 * x139 * x142 * x145 * x147 * x149 * x152 * x192 * x208 * x390 * x975
            - x1017 * x2626
            - x2420
            - x2424
            - x2432
            - x2434
            - x2435
            - x2443 * x2567
            - x2455 * x409
            - x2456 * x409
            - x2457 * x409
            - x521 * x992
        )
        + x116 * x945
        + x116 * x947
        + x1174
        * (
            2 * dq_i1 * (x1078 * x2627 + x1081 * x2628 + x1087 * x2629 + x1145 * x2630 + x2375)
            + dq_i2
            * (
                x1155 * x2366 * x61
                + x1157 * x2627
                + x1160 * x2367 * x61
                + x1162 * x2628
                + x1163 * x2368 * x61
                + x1167 * x2629
                + x1168 * x2369 * x61
                + x1169 * x2630
                + x2382
            )
            + 2 * dq_i3 * (x1128 * x2631 + x1130 * x2612 + x1131 * x2632 + x1135 * x2633 + x2377)
            + 2 * dq_i4 * dq_i7 * dq_j4 * x1083 * x138 * x143 * x145 * x147 * x149 * x152 * x208 * x2276 * x390
            + 2 * dq_i4 * (x1114 * x2631 + x1119 * x2632 + x1122 * x2633 + x1763 * x2619 + x2381)
            + 2 * dq_i5 * (x1106 * x2612 + x1107 * x2592 + x1110 * x2616 + x2374)
            + 2 * dq_i6 * (x1100 * x2592 + x1101 * x2595 + x2360)
            + 2 * dq_i7 * sigma_kin_v_7_4 * x1083 * x138 * x143 * x145 * x147 * x149 * x152 * x192 * x208 * x390
            - x1095 * x521
            - x1121 * x2626
            - x2352
            - x2355
            - x2362
            - x2363
            - x2364
            - x2370 * x2567
            - x2378 * x409
            - x2379 * x409
            - x2380 * x409
        )
        + x1251
        * (
            ddq_i4 * (x2270 * x607 + x2272 * x566 + x2274 * x452 + x2276 * x454)
            + x118 * x119 * x122 * x126 * x2272 * x288 * x564 * x626
            + x139 * x143 * x144 * x147 * x149 * x153 * x2276 * x302 * x390 * x626
            + x165 * x166 * x169 * x171 * x175 * x2274 * x295 * x450 * x626
            + x2270 * x282 * x605 * x626 * x66 * x73 * x78
            - x2581 * x599
            - x486 * (x220 * x2583 + x2584 * x495)
            - x589 * (x253 * x2585 + x2581 * x490 + x2582 * x261)
            - x616 * (x2271 * x610 + x2273 * x569 + x2275 * x457 + x2277 * x459)
            - x620 * (x2366 * x365 + x2367 * x371 + x2368 * x377 + x2369 * x383)
            - x624 * (x2439 * x337 + x2440 * x581 + x2441 * x471 + x2442 * x475)
            - x627 * (x2272 * x587 + x2513 * x282 + x2514 * x295 + x2515 * x302)
        )
        + x135 * x2344
        + x135
        * (
            dq_i1
            * (
                x129 * x2628
                + x133 * x2628
                + x178 * x2629
                + x182 * x2629
                + x196 * x2630
                + x200 * x2630
                + x2303
                + x2627 * x81
                + x2627 * x85
            )
            + 2 * dq_i2 * (x1197 * x2627 + x1198 * x2628 + x1199 * x2629 + x1200 * x2630 + x2292)
            + 2 * dq_i3 * (x1192 * x2612 + x1193 * x2632 + x1194 * x2633 + x2296 + x2631 * x338)
            + 2 * dq_i4 * dq_i7 * dq_j4 * x136 * x137 * x139 * x143 * x145 * x147 * x149 * x152 * x208 * x2276
            + 2 * dq_i4 * (x1185 * x2631 + x1187 * x2632 + x1189 * x2633 + x1757 * x2619 + x2301)
            + 2 * dq_i5 * (x1181 * x2612 + x1182 * x2592 + x1184 * x2616 + x2285)
            + 2 * dq_i6 * (x1178 * x2592 + x1179 * x2594 + x2258)
            + 2 * dq_i7 * sigma_kin_v_7_4 * x136 * x137 * x139 * x143 * x145 * x147 * x149 * x152 * x192 * x208
            - x1188 * x2626
            - x2243
            - x2250
            - x2262
            - x2266
            - x2267
            - x2278 * x2567
            - x2298 * x409
            - x2299 * x409
            - x2300 * x409
            - x247 * x521
        )
        - x1376 * x2247
        - x145
        * x2531
        * x2538
        * x426
        * (-x146 * x2529 + x146 * x2530 + x146 * x2536 + x146 * x2537 + x2531 * x2532 + x2531 * x2533 + x2534 * x2535)
        + x1598 * x2324
        + x2040 * x940
        + x2040 * x941
        + x2240 * x2494 * (sigma_pot_4_c * x72 + sigma_pot_4_s * x69)
        - x2259 * x2506
        + x2279 * x2516
        - x2286 * x2521
        - x2293 * x2527
        - x2297 * x2528
        - x2376 * x2526
        + x2396 * x699
        + x2473 * x656
        + x2496 * x962
        + x2497 * x951
        + x2497 * x956
        + x2498 * x955
        + x2498 * x959
        + x2502 * x962
        + x2510 * x962
        + x2511 * x962
        + x2512 * x962
        - 4 * x2547 * x61
        + x2569
        * (
            x187 * x527
            + x187 * x530
            + x2560 * x2565
            - x2561
            + x2563 * x520
            + x2564 * x508
            + x2565 * x2566
            + x2565 * x512
            + x2565 * x514
            - x2567 * x538
            + x2568
            + x542
        )
        + x500
        * (
            ddq_i6 * x2551
            + 2 * dq_i4 * dq_i6 * sigma_kin_v_6_4 * sigma_kin_v_6_6 * x165 * x166 * x169 * x171 * x175 * x295 * x450
            + 2
            * dq_i4
            * dq_i6
            * sigma_kin_v_7_4
            * sigma_kin_v_7_6
            * x139
            * x143
            * x144
            * x147
            * x149
            * x153
            * x302
            * x390
            - x2548 * x449
            - x2554 * x462
            - x2555 * x468
            - x2556 * x478
            - x2557 * x486
            - x2558 * x493
            - x2559 * x499
        )
        + x596
        * (
            ddq_i5 * x2572
            + 2 * dq_i4 * dq_i5 * sigma_kin_v_5_4 * sigma_kin_v_5_5 * x118 * x119 * x122 * x126 * x288 * x564
            + 2 * dq_i4 * dq_i5 * sigma_kin_v_6_4 * sigma_kin_v_6_5 * x165 * x166 * x169 * x171 * x175 * x295 * x450
            + 2
            * dq_i4
            * dq_i5
            * sigma_kin_v_7_4
            * sigma_kin_v_7_5
            * x139
            * x143
            * x144
            * x147
            * x149
            * x153
            * x302
            * x390
            - x2570 * x561
            - x2571 * x493
            - x2576 * x574
            - x2577 * x578
            - x2578 * x585
            - x2579 * x589
            - x2580 * x595
        )
        + x794
        * (
            2 * dq_i1 * (x2245 * x756 + x2586 * x750 + x2593 * x750 + x2594 * x756)
            + 2 * dq_i2 * (x2428 * x763 + x2586 * x757 + x2593 * x757 + x2595 * x763)
            + 2 * dq_i3 * (x2586 * x471 + x2589 * x765 + x2593 * x471 + x2596 * x765)
            + 2 * dq_i4 * dq_i7 * dq_j4 * x139 * x143 * x145 * x147 * x148 * x152 * x208 * x2276 * x390 * x730
            + 2 * dq_i4 * (x2587 + x2589 * x771 + x2593 * x480 + x2596 * x771)
            + 2 * dq_i5 * (x2595 * x779 + x2597 * x776 + x2599 + x2600)
            + dq_i6 * (x2583 * x61 * x781 + x2594 * x787 + x2595 * x785 + x2597 * x783 + x2601 + x2602 + x2603 + x2604)
            + 2 * dq_i7 * sigma_kin_v_7_4 * x139 * x143 * x145 * x147 * x148 * x152 * x192 * x208 * x390 * x730
            - x153 * x2588 * x2589 * x482
            - x1624 * x278
            - x2150 * x272
            - x216 * x2591
            - x242 * x2590
            - x2495 * x733
            - x2567 * (x2583 * x735 + x2584 * x732)
            - x2587 * x409
            - x521 * x744
        )
        + x864
        * (
            2 * dq_i1 * (x2606 * x750 + x2613 * x569 + x2614 * x750 + x2615 * x569 + x2617 * x828 + x2618 * x828)
            + 2 * dq_i2 * (x2606 * x757 + x2613 * x576 + x2614 * x757 + x2615 * x576 + x2617 * x759 + x2618 * x759)
            + 2 * dq_i3 * (x2281 * x836 + x2606 * x471 + x2612 * x836 + x2614 * x471 + x2617 * x765 + x2618 * x765)
            + 2 * dq_i4 * dq_i7 * dq_j4 * x139 * x143 * x145 * x146 * x149 * x152 * x208 * x2276 * x390 * x799
            + 2 * dq_i4 * (x1372 * x2246 + x1372 * x2616 + x1669 * x2619 + x2605 + x2607 + x2614 * x480)
            + dq_i5
            * (
                x1373 * x2616
                + x2585 * x61 * x848
                + x2597 * x851
                + x2597 * x854
                + x2612 * x850
                + x2616 * x857
                + x2620
                + x2621
                + x2622
                + x2623
                + x2624
                + x2625
            )
            + 2 * dq_i6 * (x2595 * x822 + x2597 * x821 + x2610 + x2611)
            + 2 * dq_i7 * sigma_kin_v_7_4 * x139 * x143 * x145 * x146 * x149 * x152 * x192 * x208 * x390 * x799
            - x1655 * x278
            - x216 * x2608
            - x2174 * x272
            - x2246 * x2588 * x799 * x844
            - x242 * x2609
            - x2495 * x802
            - x2567 * (x2581 * x801 + x2582 * x804 + x2585 * x795)
            - x2605 * x409
            - x2607 * x409
            - x521 * x817
        )
        + x962
        * (
            2 * dq_i1 * (x2526 + x2638 * x610 + x2640 * x869 + x2641 * x457 + x2642 * x459)
            + 2 * dq_i2 * (x2527 + x2638 * x618 + x2640 * x920 + x2641 * x464 + x2642 * x466)
            + 2 * dq_i3 * (x2528 + x2632 * x932 + x2633 * x935 + x2637 * x927 + x2639 * x581)
            + 2 * dq_i4 * dq_i7 * dq_j4 * x139 * x143 * x144 * x147 * x149 * x152 * x208 * x2276 * x390 * x875
            - dq_i4 * x2539
            - dq_i4 * x2540
            - dq_i4 * x2541
            - dq_i4 * x2542
            - dq_i4 * x2543
            - dq_i4 * x2544
            - dq_i4 * x2545
            - dq_i4 * x2546
            + dq_i4
            * (
                x2513 * x61 * x938
                + x2514 * x61 * x948
                + x2515 * x61 * x953
                + x2547
                + x2619 * x945
                + x2619 * x947
                + x2632 * x951
                + x2633 * x955
                + x2637 * x940
            )
            + 2 * dq_i5 * (x2521 + x2592 * x904 + x2616 * x908 + x2619 * x903)
            + 2 * dq_i6 * (x2506 + x2592 * x894 + x2595 * x896)
            + 2 * dq_i7 * sigma_kin_v_7_4 * x139 * x143 * x144 * x147 * x149 * x152 * x192 * x208 * x390 * x875
            - x2496
            - x2502
            - x2510
            - x2511
            - x2512
            - x2516 * x2567
            - x521 * x893
        )
    )
    K_block_list.append(
        -dq_i1 * x2700 * x864
        + x1070
        * (
            -x1005 * x506
            - x1007 * x506
            - x1008 * x506
            + x1391 * x2723
            + x2703 * (x1038 * x2726 + x2732 * x920 + x2733 * x464)
            + x2708 * (x1014 * x2725 + x1018 * x2726 + x1742 * x583)
            + x2712 * (x222 * x2730 + x2718 * x997)
            + x2714 * (x1027 * x2726 + x2732 * x869 + x2733 * x457)
            + x2721 * (x1003 * x2724 + x1006 * x559 + x261 * x2730)
            - x2731
            + x2734 * (x1059 * x583 + x1066 * x562 + x1069 * x558 + x2042 * x583 + x2045 * x562 + x2046 * x558)
        )
        + x1174
        * (
            -x1108 * x506
            - x1111 * x506
            - x1113 * x506
            + x1466 * x2723
            + x2706 * (x1130 * x2724 + x1131 * x2725 + x1135 * x2726)
            + x2708 * (x1119 * x2725 + x1122 * x2726 + x1763 * x583)
            + x2712 * (x222 * x2727 + x2358 * x558)
            + x2714 * (x1081 * x570 + x1087 * x571 + x1145 * x572)
            + x2721 * (x1106 * x2724 + x1110 * x559 + x261 * x2727)
            - x2728
            + x2729 * (x1162 * x570 + x1164 * x570 + x1167 * x571 + x1169 * x572 + x1170 * x571 + x1172 * x572)
        )
        + x1181 * x1765
        + x1182 * x1211
        + x1184 * x2648
        + x1202 * x2644 * x2645
        - x1227 * (x2657 * x473 + x2659 * x442)
        - x1253 * x2673
        - x1274 * x2672
        - x1292 * (x1424 * x713 + x1427 * x709 + x2658 * x707)
        + x135 * x2666
        + x135
        * (
            dq_j5 * x11 * (x129 * x570 + x133 * x570 + x178 * x571 + x182 * x571 + x196 * x572 + x200 * x572)
            + x211 * x2723
            - x258 * x506
            - x262 * x506
            - x2666
            - x267 * x506
            + x2703 * (x1198 * x570 + x1199 * x571 + x1200 * x572)
            + x2706 * (x1192 * x2724 + x1193 * x2725 + x1194 * x2726)
            + x2708 * (x1187 * x2725 + x1189 * x2726 + x1757 * x583)
            + x2712 * (x1178 * x562 + x1179 * x558)
            + x2721 * (x1181 * x2724 + x1182 * x562 + x1184 * x559)
        )
        - x1618 * x2671
        + x1768 * x491
        - x211 * x2652
        + x2647 * x864
        + x2656 * x864
        - x2663 * x2664
        + x2667 * x864
        + x2668 * x864
        + x2669 * x864
        + x2680 * x2681
        - x2686 * x2687
        - x2691 * x2692
        - x2693 * x2694
        - x2698 * x2699
        + x2722 * x596
        + x557
        * (
            x1335 * x2711
            - x1343 * x1350
            - x2665 * x541
            + x2702 * x527
            + x2702 * x530
            + x2703 * x2705
            + x2706 * x2707
            + x2708 * x2709
            + x2712 * x2713
            + x2714 * x2715
        )
        + x794
        * (
            dq_i6 * x2701 * (x1362 * x558 + x2720 * x783 + x2720 * x788 + x558 * x787)
            + x1352 * x2716 * x558
            - x2665 * x746
            + x2703 * (x2717 * x757 + x2718 * x763)
            + x2706 * (x2717 * x471 + x2719 * x765)
            + x2708 * (x2717 * x480 + x2719 * x771)
            + x2714 * (x2717 * x750 + x558 * x756)
            + x2721 * (x2718 * x779 + x2720 * x776)
            - x506 * x778
            - x506 * x780
        )
        + x864
        * (
            2 * dq_i1 * dq_i2 * dq_j5 * (x2743 * x576 + x2744 * x757 + x2745 * x759)
            + 2 * dq_i1 * dq_i3 * dq_j5 * (x2724 * x836 + x2744 * x471 + x2745 * x765)
            + 2 * dq_i1 * dq_i4 * dq_j5 * (x1372 * x559 + x1669 * x583 + x2744 * x480)
            + dq_i1
            * dq_i5
            * dq_j5
            * (x1373 * x559 + x2720 * x851 + x2720 * x854 + x2724 * x850 + x2724 * x852 + x559 * x857)
            + 2 * dq_i1 * dq_i6 * dq_j5 * (x2718 * x822 + x2720 * x821)
            + 2
            * dq_i1
            * dq_i7
            * dq_j5
            * sigma_kin_v_7_1
            * sigma_kin_v_7_5
            * x139
            * x143
            * x145
            * x146
            * x149
            * x152
            * x208
            * x390
            * x799
            - dq_i1 * x853
            - dq_i1 * x855
            - dq_i1 * x860
            - dq_i1 * x861
            - dq_i1 * x862
            - dq_i1 * x863
            + 2 * dq_j5 * x11 * (x2743 * x569 + x2744 * x750 + x2745 * x828)
            - x2742
        )
        + x962
        * (
            x1376 * x2723
            + x2703 * (x2738 * x920 + x2739 * x464 + x2740 * x466)
            + x2706 * (x2725 * x932 + x2726 * x935 + x2737 * x581)
            + x2712 * (x222 * x2735 + x2718 * x896)
            + x2714 * (x2738 * x869 + x2739 * x457 + x2740 * x459)
            + x2721 * (x261 * x2735 + x2737 * x591 + x559 * x908)
            - x2736
            + x2741 * (x2725 * x951 + x2725 * x956 + x2726 * x955 + x2726 * x959 + x583 * x945 + x583 * x947)
            - x506 * x905
            - x506 * x909
            - x506 * x910
        )
    )
    K_block_list.append(
        -dq_i2 * x2771 * x864
        + x1070
        * (
            x1391 * x2786
            - x1731 * x2774
            - x1738 * x399
            - x1739 * x399
            - x1740 * x399
            + x2703 * (x1027 * x2789 + x2792 * x869 + x2793 * x457)
            + x2776 * (x1014 * x2788 + x1018 * x2789 + x1478 * x1742)
            + x2777 * (x222 * x2791 + x2781 * x997)
            + x2778 * (x1038 * x2789 + x2792 * x920 + x2793 * x464)
            + x2784 * (x1003 * x2787 + x1006 * x1570 + x261 * x2791)
            + x2794 * (x1059 * x1478 + x1066 * x1480 + x1069 * x1482 + x1478 * x2042 + x1480 * x2045 + x1482 * x2046)
        )
        + x1106 * x1765
        + x1107 * x1211
        + x1110 * x2648
        + x1174
        * (
            dq_j5
            * x697
            * (x1162 * x1479 + x1164 * x1479 + x1167 * x1481 + x1169 * x1483 + x1170 * x1481 + x1172 * x1483)
            + x1466 * x2786
            - x1484 * x2774
            - x1495 * x399
            - x1496 * x399
            - x1497 * x399
            + x2703 * (x1081 * x1479 + x1087 * x1481 + x1145 * x1483)
            + x2775 * (x1130 * x2787 + x1131 * x2788 + x1135 * x2789)
            + x2776 * (x1119 * x2788 + x1122 * x2789 + x1478 * x1763)
            + x2777 * (x1482 * x2358 + x222 * x2790)
            + x2784 * (x1106 * x2787 + x1110 * x1570 + x261 * x2790)
        )
        - x1273 * x2757
        + x135 * x2728
        + x135
        * (
            -x1238 * x2774
            - x1257 * x399
            - x1258 * x399
            - x1260 * x399
            + x211 * x2786
            + x2729 * (x130 * x1478 + x134 * x1478 + x1480 * x179 + x1480 * x183 + x1482 * x197 + x1482 * x201)
            + x2775 * (x1192 * x2787 + x1193 * x2788 + x1194 * x2789)
            + x2776 * (x1187 * x2788 + x1189 * x2789 + x1478 * x1757)
            + x2777 * (x1178 * x1480 + x1179 * x1482)
            + x2778 * (x1198 * x1479 + x1199 * x1481 + x1200 * x1483)
            + x2784 * (x1181 * x2787 + x1182 * x1480 + x1184 * x1570)
        )
        - x1466 * x2652
        + x1569
        * (
            x2000 * x2711
            + x2703 * x2715
            + x2705 * x2778
            + x2707 * x2775
            + x2709 * x2776
            + x2713 * x2777
            + x2773 * x527
            + x2773 * x530
            - x2774 * x541
            - x2779 * x399
        )
        + x1944 * x491
        - x1949 * (x1467 * x2657 + x1469 * x2659)
        - x1967 * x2758
        - x1979 * (x1424 * x1610 + x1427 * x1606 + x1605 * x2658)
        + x2350 * x2746
        - x2664 * x2753
        - x2671 * x729
        + x2681 * x2762
        - x2692 * x2765
        - x2694 * x2766
        - x2699 * x2770
        + x2747 * x864
        + x2749 * x864
        + x2754 * x864
        + x2755 * x864
        + x2756 * x864
        - x2763 * x2764
        + x2785 * x596
        + x794
        * (
            dq_i6 * x2772 * (x1362 * x1482 + x1482 * x787 + x2783 * x783 + x2783 * x788)
            + dq_j5 * x1482 * x2005
            - x1625 * x2774
            - x1640 * x399
            - x1641 * x399
            + x2703 * (x1482 * x756 + x2780 * x750)
            + x2775 * (x2780 * x471 + x2782 * x765)
            + x2776 * (x2780 * x480 + x2782 * x771)
            + x2778 * (x2780 * x757 + x2781 * x763)
            + x2784 * (x2781 * x779 + x2783 * x776)
        )
        + x864
        * (
            2 * dq_i1 * dq_i2 * dq_j5 * (x2801 * x569 + x2802 * x750 + x2803 * x828)
            + 2 * dq_i2 * dq_i3 * dq_j5 * (x2787 * x836 + x2802 * x471 + x2803 * x765)
            + 2 * dq_i2 * dq_i4 * dq_j5 * (x1372 * x1570 + x1478 * x1669 + x2802 * x480)
            + dq_i2
            * dq_i5
            * dq_j5
            * (x1373 * x1570 + x1570 * x857 + x2783 * x851 + x2783 * x854 + x2787 * x850 + x2787 * x852)
            + 2 * dq_i2 * dq_i6 * dq_j5 * (x2781 * x822 + x2783 * x821)
            + 2
            * dq_i2
            * dq_i7
            * dq_j5
            * sigma_kin_v_7_2
            * sigma_kin_v_7_5
            * x139
            * x143
            * x145
            * x146
            * x149
            * x152
            * x208
            * x390
            * x799
            - dq_i2 * x1673
            - dq_i2 * x1674
            - dq_i2 * x1675
            - dq_i2 * x1676
            - dq_i2 * x1677
            - dq_i2 * x1678
            + 2 * dq_j5 * x697 * (x2801 * x576 + x2802 * x757 + x2803 * x759)
            - x1656 * x2774
        )
        + x962
        * (
            x1376 * x2786
            - x1689 * x2774
            - x1697 * x399
            - x1698 * x399
            - x1699 * x399
            + x2703 * (x2797 * x869 + x2798 * x457 + x2799 * x459)
            + x2775 * (x2788 * x932 + x2789 * x935 + x2796 * x581)
            + x2777 * (x222 * x2795 + x2781 * x896)
            + x2778 * (x2797 * x920 + x2798 * x464 + x2799 * x466)
            + x2784 * (x1570 * x908 + x261 * x2795 + x2796 * x591)
            + x2800 * (x1478 * x945 + x1478 * x947 + x2788 * x951 + x2788 * x956 + x2789 * x955 + x2789 * x959)
        )
    )
    K_block_list.append(
        -dq_i3 * x2830 * x864
        + x1003 * x1765
        + x1004 * x1211
        + x1006 * x2648
        + x1070
        * (
            dq_j5
            * x654
            * (x1059 * x1956 + x1066 * x2054 + x1069 * x2055 + x1956 * x2042 + x2045 * x2054 + x2046 * x2055)
            + x1391 * x2843
            - x2056 * x2833
            - x2060 * x405
            - x2061 * x405
            - x2062 * x405
            + x2706 * (x1027 * x2845 + x2848 * x869 + x2849 * x457)
            + x2775 * (x1038 * x2845 + x2848 * x920 + x2849 * x464)
            + x2834 * (x1014 * x2844 + x1018 * x2845 + x1742 * x1956)
            + x2835 * (x2838 * x997 + x2847 * x986)
            + x2841 * (x1003 * x2846 + x1006 * x2114 + x2850 * x986)
        )
        + x1174
        * (
            x1466 * x2843
            - x1957 * x2833
            - x1968 * x405
            - x1969 * x405
            - x1970 * x405
            + x2706 * (x1081 * x2118 + x1087 * x2119 + x1145 * x2120)
            + x2794 * (x1162 * x2118 + x1164 * x2118 + x1167 * x2119 + x1169 * x2120 + x1170 * x2119 + x1172 * x2120)
            + x2834 * (x1119 * x2844 + x1122 * x2845 + x1763 * x1956)
            + x2835 * (x1099 * x2847 + x2055 * x2358)
            + x2836 * (x1130 * x2846 + x1131 * x2844 + x1135 * x2845)
            + x2841 * (x1106 * x2846 + x1107 * x2054 + x1110 * x2114)
        )
        + x135 * x2731
        + x135
        * (
            -x1788 * x2833
            - x1797 * x405
            - x1799 * x405
            - x1800 * x405
            + x211 * x2843
            + x2734 * (x130 * x1956 + x134 * x1956 + x179 * x2054 + x183 * x2054 + x197 * x2055 + x201 * x2055)
            + x2775 * (x1198 * x2118 + x1199 * x2119 + x1200 * x2120)
            + x2834 * (x1187 * x2844 + x1189 * x2845 + x1757 * x1956)
            + x2835 * (x1178 * x2054 + x1179 * x2055)
            + x2836 * (x1192 * x2846 + x1193 * x2844 + x1194 * x2845)
            + x2841 * (x1181 * x2846 + x1182 * x2054 + x1184 * x2114)
        )
        - x1391 * x2652
        + x2113
        * (
            x2463 * x2711
            + x2705 * x2775
            + x2706 * x2715
            + x2707 * x2836
            + x2709 * x2834
            + x2713 * x2835
            - x2779 * x405
            + x2832 * x527
            + x2832 * x530
            - x2833 * x541
        )
        + x2418 * x2746
        + x2421 * x491
        - x2425 * (x2047 * x2657 + x2048 * x2659)
        - x2437 * x2672
        - x2438 * x2757
        - x2444 * (x1424 * x2142 + x1427 * x2139 + x2138 * x2658)
        - x2664 * x2811
        + x2681 * x2820
        - x2687 * x2824
        - x2694 * x2825
        - x2699 * x2829
        - x2764 * x2823
        + x2804 * x864
        + x2806 * x864
        + x2812 * x864
        + x2813 * x864
        + x2814 * x864
        - x2815 * x2816
        + x2842 * x596
        + x794
        * (
            dq_i6 * x2831 * (x1362 * x2055 + x2055 * x787 + x2840 * x783 + x2840 * x788)
            + dq_j5 * x2055 * x2467
            - x2151 * x2833
            - x2162 * x405
            - x2163 * x405
            + x2706 * (x2055 * x756 + x2837 * x750)
            + x2775 * (x2837 * x757 + x2838 * x763)
            + x2834 * (x2837 * x480 + x2839 * x771)
            + x2836 * (x2837 * x471 + x2839 * x765)
            + x2841 * (x2838 * x779 + x2840 * x776)
        )
        + x864
        * (
            2 * dq_i1 * dq_i3 * dq_j5 * (x2856 * x569 + x2857 * x750 + x2858 * x828)
            + 2 * dq_i2 * dq_i3 * dq_j5 * (x2856 * x576 + x2857 * x757 + x2858 * x759)
            + 2 * dq_i3 * dq_i4 * dq_j5 * (x1372 * x2114 + x1669 * x1956 + x2857 * x480)
            + dq_i3
            * dq_i5
            * dq_j5
            * (x1373 * x2114 + x2114 * x857 + x2840 * x851 + x2840 * x854 + x2846 * x850 + x2846 * x852)
            + 2 * dq_i3 * dq_i6 * dq_j5 * (x2838 * x822 + x2840 * x821)
            + 2
            * dq_i3
            * dq_i7
            * dq_j5
            * sigma_kin_v_7_3
            * sigma_kin_v_7_5
            * x139
            * x143
            * x145
            * x146
            * x149
            * x152
            * x208
            * x390
            * x799
            - dq_i3 * x2188
            - dq_i3 * x2189
            - dq_i3 * x2190
            - dq_i3 * x2191
            - dq_i3 * x2192
            - dq_i3 * x2193
            + 2 * dq_j5 * x654 * (x2846 * x836 + x2857 * x471 + x2858 * x765)
            - x2175 * x2833
        )
        + x962
        * (
            x1376 * x2843
            - x2200 * x2833
            - x2204 * x405
            - x2205 * x405
            - x2206 * x405
            + x2706 * (x2852 * x869 + x2853 * x457 + x2854 * x459)
            + x2775 * (x2852 * x920 + x2853 * x464 + x2854 * x466)
            + x2835 * (x2838 * x896 + x2847 * x885)
            + x2836 * (x2844 * x932 + x2845 * x935 + x2851 * x581)
            + x2841 * (x2114 * x908 + x2850 * x885 + x2851 * x591)
            + x2855 * (x1956 * x945 + x1956 * x947 + x2844 * x951 + x2844 * x956 + x2845 * x955 + x2845 * x959)
        )
    )
    K_block_list.append(
        -dq_i4 * x2891 * x864
        + x1070
        * (
            x1391 * x2905
            - x2431 * x2894
            - x2445 * x409
            - x2446 * x409
            - x2447 * x409
            + x2708 * (x1027 * x2908 + x2910 * x869 + x2911 * x457)
            + x2776 * (x1038 * x2908 + x2910 * x920 + x2911 * x464)
            + x2855 * (x1059 * x2260 + x1066 * x2507 + x1069 * x2508 + x2042 * x2260 + x2045 * x2507 + x2046 * x2508)
            + x2896 * (x2508 * x2809 + x2909 * x986)
            + x2897 * (x1014 * x2907 + x1018 * x2908 + x1742 * x2260)
            + x2903 * (x1003 * x2906 + x1004 * x2507 + x1006 * x2570)
        )
        + x1174
        * (
            x1466 * x2905
            - x2361 * x2894
            - x2371 * x409
            - x2372 * x409
            - x2373 * x409
            + x2708 * (x1081 * x2573 + x1087 * x2574 + x1145 * x2575)
            + x2800 * (x1162 * x2573 + x1164 * x2573 + x1167 * x2574 + x1169 * x2575 + x1170 * x2574 + x1172 * x2575)
            + x2834 * (x1130 * x2906 + x1131 * x2907 + x1135 * x2908)
            + x2896 * (x1099 * x2909 + x2358 * x2508)
            + x2897 * (x1119 * x2907 + x1122 * x2908 + x1763 * x2260)
            + x2903 * (x1106 * x2906 + x1107 * x2507 + x1110 * x2570)
        )
        + x1205 * x903
        + x1211 * x904
        + x135 * x2736
        + x135
        * (
            x211 * x2905
            - x2261 * x2894
            - x2282 * x409
            - x2283 * x409
            - x2284 * x409
            + x2741 * (x130 * x2260 + x134 * x2260 + x179 * x2507 + x183 * x2507 + x197 * x2508 + x201 * x2508)
            + x2776 * (x1198 * x2573 + x1199 * x2574 + x1200 * x2575)
            + x2834 * (x1192 * x2906 + x1193 * x2907 + x1194 * x2908)
            + x2896 * (x1178 * x2507 + x1179 * x2508)
            + x2897 * (x1187 * x2907 + x1189 * x2908 + x1757 * x2260)
            + x2903 * (x1181 * x2906 + x1182 * x2507 + x1184 * x2570)
        )
        - x1376 * x2652
        + x2494 * x2644 * x2859
        + x2569
        * (
            x2705 * x2776
            + x2707 * x2834
            + x2708 * x2715
            + x2709 * x2897
            + x2711 * x409 * x504
            + x2713 * x2896
            - x2779 * x409
            + x2892 * x2893
            + x2892 * x2895
            - x2894 * x541
        )
        + x2648 * x908
        - x2664 * x2869
        - x2673 * x2873
        + x2681 * x2879
        - x2687 * x2886
        - x2692 * x2887
        - x2699 * x2890
        - x2758 * x2875
        - x2764 * x2885
        - x2815 * x2876
        + x2860 * x864
        + x2861 * x491
        + x2863 * x864
        - x2865 * (x2499 * x2657 + x2500 * x2659)
        + x2870 * x864
        + x2871 * x864
        + x2872 * x864
        - x2880 * (x1424 * x2585 + x1427 * x2582 + x2581 * x2658)
        + x2904 * x596
        + x794
        * (
            dq_j5 * x2864 * (x1362 * x2508 + x2508 * x787 + x2902 * x783 + x2902 * x788)
            + x1352 * x2508 * x2898
            - x2590 * x2894
            - x2599 * x409
            - x2600 * x409
            + x2708 * (x2508 * x756 + x2899 * x750)
            + x2776 * (x2899 * x757 + x2900 * x763)
            + x2834 * (x2899 * x471 + x2901 * x765)
            + x2897 * (x2899 * x480 + x2901 * x771)
            + x2903 * (x2900 * x779 + x2902 * x776)
        )
        + x864
        * (
            2 * dq_i1 * dq_i4 * dq_j5 * (x2916 * x569 + x2917 * x750 + x2918 * x828)
            + 2 * dq_i2 * dq_i4 * dq_j5 * (x2916 * x576 + x2917 * x757 + x2918 * x759)
            + 2 * dq_i3 * dq_i4 * dq_j5 * (x2906 * x836 + x2917 * x471 + x2918 * x765)
            + dq_i4
            * dq_i5
            * dq_j5
            * (x1373 * x2570 + x2570 * x857 + x2902 * x851 + x2902 * x854 + x2906 * x850 + x2906 * x852)
            + 2 * dq_i4 * dq_i6 * dq_j5 * (x2900 * x822 + x2902 * x821)
            + 2
            * dq_i4
            * dq_i7
            * dq_j5
            * sigma_kin_v_7_4
            * sigma_kin_v_7_5
            * x139
            * x143
            * x145
            * x146
            * x149
            * x152
            * x208
            * x390
            * x799
            - dq_i4 * x2620
            - dq_i4 * x2621
            - dq_i4 * x2622
            - dq_i4 * x2623
            - dq_i4 * x2624
            - dq_i4 * x2625
            + 2 * dq_j5 * x626 * (x1372 * x2570 + x1669 * x2260 + x2917 * x480)
            - x2609 * x2894
        )
        + x962
        * (
            dq_j5 * x626 * (x2260 * x945 + x2260 * x947 + x2907 * x951 + x2907 * x956 + x2908 * x955 + x2908 * x959)
            + x1376 * x2905
            - x2509 * x2894
            - x2518 * x409
            - x2519 * x409
            - x2520 * x409
            + x2708 * (x2913 * x869 + x2914 * x457 + x2915 * x459)
            + x2776 * (x2913 * x920 + x2914 * x464 + x2915 * x466)
            + x2834 * (x2907 * x932 + x2908 * x935 + x2912 * x581)
            + x2896 * (x2900 * x896 + x2909 * x885)
            + x2903 * (x2507 * x904 + x2570 * x908 + x2912 * x591)
        )
    )
    K_block_list.append(
        x1070
        * (
            2 * dq_i1 * (x1027 * x2991 + x2823 + x2994 * x869 + x2995 * x457)
            + 2 * dq_i2 * (x1038 * x2991 + x2824 + x2994 * x920 + x2995 * x464)
            + dq_i3
            * (x1051 * x2989 + x1053 * x2989 + x1054 * x2990 + x1057 * x2990 + x1061 * x2991 + x1063 * x2991 + x2830)
            + 2 * dq_i4 * (x1014 * x2990 + x1018 * x2991 + x1742 * x2992 + x2825)
            + 2 * dq_i5 * dq_i7 * dq_j5 * x139 * x142 * x145 * x147 * x149 * x152 * x208 * x2678 * x390 * x975
            + 2 * dq_i5 * (x1003 * x2989 + x1004 * x2974 + x1006 * x2993 + x2829)
            + 2 * dq_i6 * (x2811 + x2974 * x995 + x2978 * x997)
            + 2 * dq_i7 * sigma_kin_v_7_5 * x139 * x142 * x145 * x147 * x149 * x152 * x192 * x208 * x390 * x975
            - x2804
            - x2806
            - x2812
            - x2813
            - x2814
            - x2820 * x2967
            - x2826 * x413
            - x2827 * x413
            - x2828 * x413
            - x521 * x991
        )
        - 4 * x113 * x2952
        + x1174
        * (
            2 * dq_i1 * (x1081 * x2986 + x1087 * x2987 + x1145 * x2988 + x2763)
            + dq_i2
            * (
                x113 * x1160 * x2759
                + x113 * x1163 * x2760
                + x113 * x1168 * x2761
                + x1162 * x2986
                + x1167 * x2987
                + x1169 * x2988
                + x2771
            )
            + 2 * dq_i3 * (x1130 * x2989 + x1131 * x2990 + x1135 * x2991 + x2765)
            + 2 * dq_i4 * (x1119 * x2990 + x1122 * x2991 + x1763 * x2992 + x2766)
            + 2 * dq_i5 * dq_i7 * dq_j5 * x1083 * x138 * x143 * x145 * x147 * x149 * x152 * x208 * x2678 * x390
            + 2 * dq_i5 * (x1106 * x2989 + x1107 * x2974 + x1110 * x2993 + x2770)
            + 2 * dq_i6 * (x1100 * x2974 + x1101 * x2978 + x2753)
            + 2 * dq_i7 * sigma_kin_v_7_5 * x1083 * x138 * x143 * x145 * x147 * x149 * x152 * x192 * x208 * x390
            - x1093 * x521
            - x2747
            - x2749
            - x2754
            - x2755
            - x2756
            - x2762 * x2967
            - x2767 * x413
            - x2768 * x413
            - x2769 * x413
        )
        + x1242
        * (
            ddq_i5 * (x2674 * x566 + x2676 * x452 + x2678 * x454)
            + x118 * x120 * x121 * x126 * x253 * x2674 * x564 * x594
            + x139 * x143 * x145 * x146 * x149 * x153 * x263 * x2678 * x390 * x594
            + x165 * x167 * x168 * x171 * x175 * x259 * x2676 * x450 * x594
            - x2936 * x561
            - x493 * (x220 * x2969 + x2970 * x495)
            - x574 * (x2675 * x569 + x2677 * x457 + x2679 * x459)
            - x578 * (x2759 * x371 + x2760 * x377 + x2761 * x383)
            - x585 * (x2817 * x581 + x2818 * x471 + x2819 * x475)
            - x589 * (x2674 * x587 + x2877 * x295 + x2878 * x302)
            - x595 * (x253 * x2934 + x261 * x2935 + x2936 * x490)
        )
        + x135 * x2742
        + x135
        * (
            dq_i1 * (x129 * x2986 + x133 * x2986 + x178 * x2987 + x182 * x2987 + x196 * x2988 + x200 * x2988 + x2700)
            + 2 * dq_i2 * (x1198 * x2986 + x1199 * x2987 + x1200 * x2988 + x2686)
            + 2 * dq_i3 * (x1192 * x2989 + x1193 * x2990 + x1194 * x2991 + x2691)
            + 2 * dq_i4 * (x1187 * x2990 + x1189 * x2991 + x1757 * x2992 + x2693)
            + 2 * dq_i5 * dq_i7 * dq_j5 * x136 * x137 * x139 * x143 * x145 * x147 * x149 * x152 * x208 * x2678
            + 2 * dq_i5 * (x1181 * x2989 + x1182 * x2974 + x1184 * x2993 + x2698)
            + 2 * dq_i6 * (x1178 * x2974 + x1179 * x2976 + x2663)
            + 2 * dq_i7 * sigma_kin_v_7_5 * x136 * x137 * x139 * x143 * x145 * x147 * x149 * x152 * x192 * x208
            - x241 * x521
            - x2647
            - x2656
            - x2667
            - x2668
            - x2669
            - x2680 * x2967
            - x2695 * x413
            - x2696 * x413
            - x2697 * x413
        )
        - x1366 * x2649 * x2924
        + x1373 * x2923
        - x147
        * x2944
        * x2945
        * x426
        * (-x144 * x2529 + x144 * x2530 + x144 * x2536 + x144 * x2537 + x2532 * x2944 + x2533 * x2944 + x2534 * x2588)
        + x1598 * x2722
        + x2494 * x2919 * (sigma_pot_5_c * x93 + sigma_pot_5_s * x90)
        - x2664 * x2930
        + x2681 * x2937
        - x2687 * x2941
        - x2692 * x2942
        - x2694 * x2943
        - x2764 * x2940
        + x2785 * x699
        + x2842 * x656
        + x2904 * x628
        + x2920 * x850
        + x2920 * x852
        + x2921 * x864
        + x2922 * x851
        + x2922 * x854
        + x2923 * x857
        + x2926 * x864
        + x2931 * x864
        + x2932 * x864
        + x2933 * x864
        + x2968
        * (
            x188 * x527
            + x188 * x530
            + x2560 * x2966
            + x2566 * x2966
            + x2568
            - x2963
            + x2964 * x520
            + x2965 * x508
            + x2966 * x512
            + x2966 * x514
            - x2967 * x541
            + x539
        )
        + x500
        * (
            ddq_i6 * x2954
            + 2 * dq_i5 * dq_i6 * sigma_kin_v_6_5 * sigma_kin_v_6_6 * x165 * x167 * x168 * x171 * x175 * x259 * x450
            + 2
            * dq_i5
            * dq_i6
            * sigma_kin_v_7_5
            * sigma_kin_v_7_6
            * x139
            * x143
            * x145
            * x146
            * x149
            * x153
            * x263
            * x390
            - x2953 * x449
            - x2957 * x462
            - x2958 * x468
            - x2959 * x478
            - x2960 * x486
            - x2961 * x493
            - x2962 * x499
        )
        + x794
        * (
            2 * dq_i1 * (x2650 * x756 + x2975 * x750 + x2976 * x756 + x2977 * x750)
            + 2 * dq_i2 * (x2867 * x763 + x2975 * x757 + x2977 * x757 + x2978 * x763)
            + 2 * dq_i3 * (x2975 * x471 + x2977 * x471 + x2979 * x765 + x2980 * x765)
            + 2 * dq_i4 * (x2975 * x480 + x2977 * x480 + x2979 * x771 + x2980 * x771)
            + 2 * dq_i5 * dq_i7 * dq_j5 * x139 * x143 * x145 * x147 * x148 * x152 * x208 * x2678 * x390 * x730
            + 2 * dq_i5 * (x2971 + x2972 + x2978 * x779 + x2981 * x776)
            + dq_i6 * (x113 * x2969 * x781 + x2976 * x787 + x2978 * x785 + x2981 * x783 + x2982 + x2983 + x2984 + x2985)
            + 2 * dq_i7 * sigma_kin_v_7_5 * x139 * x143 * x145 * x147 * x148 * x152 * x192 * x208 * x390 * x730
            - x1625 * x278
            - x2151 * x272
            - x216 * x2973
            - x248 * x2590
            - x2646 * x733
            - x2967 * (x2969 * x735 + x2970 * x732)
            - x2971 * x413
            - x2972 * x413
            - x521 * x746
        )
        + x864
        * (
            2 * dq_i1 * (x2940 + x3000 * x569 + x3001 * x750 + x3002 * x828)
            + 2 * dq_i2 * (x2941 + x3000 * x576 + x3001 * x757 + x3002 * x759)
            + 2 * dq_i3 * (x2942 + x2989 * x836 + x3001 * x471 + x3002 * x765)
            + 2 * dq_i4 * (x1372 * x2993 + x1669 * x2992 + x2943 + x3001 * x480)
            + 2 * dq_i5 * dq_i7 * dq_j5 * x139 * x143 * x145 * x146 * x149 * x152 * x208 * x2678 * x390 * x799
            - dq_i5 * x2946
            - dq_i5 * x2947
            - dq_i5 * x2948
            - dq_i5 * x2949
            - dq_i5 * x2950
            - dq_i5 * x2951
            + dq_i5
            * (x113 * x2934 * x848 + x1373 * x2993 + x2952 + x2981 * x851 + x2981 * x854 + x2989 * x850 + x2993 * x857)
            + 2 * dq_i6 * (x2930 + x2978 * x822 + x2981 * x821)
            + 2 * dq_i7 * sigma_kin_v_7_5 * x139 * x143 * x145 * x146 * x149 * x152 * x192 * x208 * x390 * x799
            - x2921
            - x2926
            - x2931
            - x2932
            - x2933
            - x2937 * x2967
            - x521 * x818
        )
        + x962
        * (
            2 * dq_i1 * (x2885 + x2997 * x869 + x2998 * x457 + x2999 * x459)
            + 2 * dq_i2 * (x2886 + x2997 * x920 + x2998 * x464 + x2999 * x466)
            + 2 * dq_i3 * (x2887 + x2990 * x932 + x2991 * x935 + x2996 * x581)
            + dq_i4
            * (
                x113 * x2877 * x948
                + x113 * x2878 * x953
                + x2891
                + x2990 * x951
                + x2991 * x955
                + x2992 * x945
                + x2992 * x947
            )
            + 2 * dq_i5 * dq_i7 * dq_j5 * x139 * x143 * x144 * x147 * x149 * x152 * x208 * x2678 * x390 * x875
            + 2 * dq_i5 * (x2890 + x2974 * x904 + x2992 * x903 + x2993 * x908)
            + 2 * dq_i6 * (x2869 + x2974 * x894 + x2978 * x896)
            + 2 * dq_i7 * sigma_kin_v_7_5 * x139 * x143 * x144 * x147 * x149 * x152 * x192 * x208 * x390 * x875
            - x2535 * x2651 * x907
            - x2860
            - x2863
            - x2870
            - x2871
            - x2872
            - x2879 * x2967
            - x2888 * x413
            - x2889 * x413
            - x521 * x890
        )
    )
    K_block_list.append(
        -dq_i1 * x3047 * x794
        + x1070
        * (
            -x1000 * x506
            + x1391 * x3065
            + x3050 * (x1038 * x3067 + x3073 * x464)
            + x3055 * (x1014 * x3066 + x1018 * x3067)
            + x3057 * (x1004 * x473 + x1006 * x443)
            + x3060 * (x1027 * x3067 + x3073 * x457)
            + x3068 * (x2809 * x442 + x3070 * x986)
            - x3072
            + x3074 * (x1066 * x473 + x1069 * x442 + x2045 * x473 + x2046 * x442)
            - x506 * x999
        )
        + x1174
        * (
            -x1103 * x506
            - x1104 * x506
            + x1466 * x3065
            + x3053 * (x1131 * x3066 + x1135 * x3067)
            + x3055 * (x1119 * x3066 + x1122 * x3067)
            + x3057 * (x1107 * x473 + x1110 * x443)
            + x3060 * (x1087 * x458 + x1145 * x460)
            + x3068 * (x1099 * x3070 + x2358 * x442)
            - x3069
            + x3071 * (x1167 * x458 + x1169 * x460 + x1170 * x458 + x1172 * x460)
        )
        + x1178 * x1211
        + x1179 * x1212
        - x1244 * x3023
        - x1253 * x3022
        - x1274 * x3021
        - x1292 * (x1418 * x710 + x3018 * x711)
        + x135 * x3013
        + x135
        * (
            dq_j6 * x11 * (x178 * x458 + x182 * x458 + x196 * x460 + x200 * x460)
            + x211 * x3065
            - x227 * x506
            - x236 * x506
            - x3013
            + x3050 * (x1199 * x458 + x1200 * x460)
            + x3053 * (x1193 * x3066 + x1194 * x3067)
            + x3055 * (x1187 * x3066 + x1189 * x3067)
            + x3057 * (x1182 * x473 + x1184 * x443)
            + x3068 * (x1178 * x473 + x1179 * x442)
        )
        - x1618 * x3020
        + x1768 * x496
        - x211 * x3011
        + x3003 * x3005
        + x3007 * x794
        + x3014 * x794
        + x3015 * x794
        + x3016 * x794
        + x3017 * x794
        + x3028 * x3029
        - x3033 * x3034
        - x3037 * x3038
        - x3039 * x3040
        - x3041 * x3042
        - x3045 * x3046
        + x3063 * x500
        + x557
        * (
            -x3012 * x544
            + x3049 * x527
            + x3049 * x530
            + x3050 * x3052
            + x3053 * x3054
            + x3055 * x3056
            + x3057 * x3058
            + x3059 * x506
            + x3060 * x3061
            - x3062 * x506
        )
        + x794
        * (
            dq_i1 * x160 * (x1362 * x442 + x3083 * x783 + x3083 * x788 + x442 * x787)
            - dq_i1 * x789
            - dq_i1 * x791
            - dq_i1 * x792
            - dq_i1 * x793
            + x1352 * x3064 * x442
            + x3050 * (x3078 * x763 + x3086 * x757)
            + x3053 * (x3086 * x471 + x3087 * x765)
            + x3055 * (x3086 * x480 + x3087 * x771)
            + x3057 * (x3078 * x779 + x3083 * x776)
            + x3060 * (x3086 * x750 + x442 * x756)
            - x3085
        )
        + x864
        * (
            x1366 * x3064 * x3080
            + x3050 * (x3080 * x759 + x3082 * x757)
            + x3053 * (x3080 * x765 + x3082 * x471)
            + x3055 * (x1372 * x443 + x3082 * x480)
            + x3060 * (x3080 * x828 + x3082 * x750)
            + x3068 * (x3078 * x822 + x3083 * x821)
            - x3081
            + x3084 * (x1373 * x443 + x3083 * x851 + x3083 * x854 + x443 * x857)
            - x506 * x823
            - x506 * x824
        )
        + x962
        * (
            x1376 * x3065
            + x3050 * (x3076 * x464 + x3077 * x466)
            + x3053 * (x3066 * x932 + x3067 * x935)
            + x3057 * (x443 * x908 + x473 * x904)
            + x3060 * (x3076 * x457 + x3077 * x459)
            + x3068 * (x3078 * x896 + x473 * x894)
            - x3075
            + x3079 * (x3066 * x951 + x3066 * x956 + x3067 * x955 + x3067 * x959)
            - x506 * x898
            - x506 * x900
        )
    )
    K_block_list.append(
        -dq_i2 * x3109 * x794
        + x1070
        * (
            x1391 * x3119
            - x1730 * x3112
            - x1736 * x399
            - x1737 * x399
            + x3050 * (x1027 * x3121 + x3124 * x457)
            + x3114 * (x1014 * x3120 + x1018 * x3121)
            + x3115 * (x1004 * x1467 + x1006 * x1549)
            + x3116 * (x1038 * x3121 + x3124 * x464)
            + x3122 * (x1469 * x2809 + x3123 * x986)
            + x3125 * (x1066 * x1467 + x1069 * x1469 + x1467 * x2045 + x1469 * x2046)
        )
        + x1100 * x1211
        + x1174
        * (
            dq_j6 * x697 * (x1167 * x1468 + x1169 * x1470 + x1170 * x1468 + x1172 * x1470)
            + x1466 * x3119
            - x1471 * x3112
            - x1474 * x399
            - x1476 * x399
            + x3050 * (x1087 * x1468 + x1145 * x1470)
            + x3113 * (x1131 * x3120 + x1135 * x3121)
            + x3114 * (x1119 * x3120 + x1122 * x3121)
            + x3115 * (x1107 * x1467 + x1110 * x1549)
            + x3122 * (x1099 * x3123 + x1469 * x2358)
        )
        + x1212 * x2358
        - x1273 * x3095
        + x135 * x3069
        + x135
        * (
            -x1219 * x3112
            - x1231 * x399
            - x1233 * x399
            + x211 * x3119
            + x3071 * (x1467 * x179 + x1467 * x183 + x1469 * x197 + x1469 * x201)
            + x3113 * (x1193 * x3120 + x1194 * x3121)
            + x3114 * (x1187 * x3120 + x1189 * x3121)
            + x3115 * (x1182 * x1467 + x1184 * x1549)
            + x3116 * (x1199 * x1468 + x1200 * x1470)
            + x3122 * (x1178 * x1467 + x1179 * x1469)
        )
        - x1466 * x3011
        + x1569
        * (
            x3050 * x3061
            + x3052 * x3116
            + x3054 * x3113
            + x3056 * x3114
            + x3058 * x3115
            + x3059 * x399
            - x3062 * x399
            + x3111 * x527
            + x3111 * x530
            - x3112 * x544
        )
        + x1944 * x496
        - x1959 * x3097
        - x1967 * x3096
        - x1979 * (x1418 * x1607 + x1608 * x3018)
        + x2350 * x3089
        - x3020 * x729
        + x3029 * x3100
        - x3038 * x3103
        - x3040 * x3104
        - x3042 * x3105
        - x3046 * x3108
        + x3090 * x794
        + x3091 * x794
        + x3092 * x794
        + x3093 * x794
        + x3094 * x794
        - x3101 * x3102
        + x3117 * x500
        + x794
        * (
            dq_i2 * x160 * (x1362 * x1469 + x1469 * x787 + x3132 * x783 + x3132 * x788)
            - dq_i2 * x1642
            - dq_i2 * x1643
            - dq_i2 * x1644
            - dq_i2 * x1645
            + dq_j6 * x1469 * x2005
            - x1626 * x3112
            + x3050 * (x1469 * x756 + x3134 * x750)
            + x3113 * (x3134 * x471 + x3135 * x765)
            + x3114 * (x3134 * x480 + x3135 * x771)
            + x3115 * (x3128 * x779 + x3132 * x776)
            + x3116 * (x3128 * x763 + x3134 * x757)
        )
        + x864
        * (
            x1366 * x3118 * x3130
            - x1653 * x3112
            - x1657 * x399
            - x1658 * x399
            + x3050 * (x3130 * x828 + x3131 * x750)
            + x3113 * (x3130 * x765 + x3131 * x471)
            + x3114 * (x1372 * x1549 + x3131 * x480)
            + x3116 * (x3130 * x759 + x3131 * x757)
            + x3122 * (x3128 * x822 + x3132 * x821)
            + x3133 * (x1373 * x1549 + x1549 * x857 + x3132 * x851 + x3132 * x854)
        )
        + x962
        * (
            x1376 * x3119
            - x1688 * x3112
            - x1694 * x399
            - x1696 * x399
            + x3050 * (x3126 * x457 + x3127 * x459)
            + x3113 * (x3120 * x932 + x3121 * x935)
            + x3115 * (x1467 * x904 + x1549 * x908)
            + x3116 * (x3126 * x464 + x3127 * x466)
            + x3122 * (x1467 * x894 + x3128 * x896)
            + x3129 * (x3120 * x951 + x3120 * x956 + x3121 * x955 + x3121 * x959)
        )
    )
    K_block_list.append(
        -dq_i3 * x3154 * x794
        + x1070
        * (
            dq_j6 * x654 * (x1066 * x2047 + x1069 * x2048 + x2045 * x2047 + x2046 * x2048)
            + x1391 * x3163
            - x2049 * x3157
            - x2051 * x405
            - x2052 * x405
            + x3053 * (x1027 * x3165 + x3167 * x457)
            + x3113 * (x1038 * x3165 + x3167 * x464)
            + x3158 * (x1014 * x3164 + x1018 * x3165)
            + x3159 * (x1004 * x2047 + x1006 * x2095)
            + x3166 * (x2047 * x995 + x2048 * x2809)
        )
        + x1174
        * (
            x1466 * x3163
            - x1947 * x3157
            - x1951 * x405
            - x1953 * x405
            + x3053 * (x1087 * x2099 + x1145 * x2100)
            + x3125 * (x1167 * x2099 + x1169 * x2100 + x1170 * x2099 + x1172 * x2100)
            + x3158 * (x1119 * x3164 + x1122 * x3165)
            + x3159 * (x1107 * x2047 + x1110 * x2095)
            + x3160 * (x1131 * x3164 + x1135 * x3165)
            + x3166 * (x1100 * x2047 + x2048 * x2358)
        )
        + x1211 * x995
        + x1212 * x2809
        + x135 * x3072
        + x135
        * (
            -x1776 * x3157
            - x1783 * x405
            - x1784 * x405
            + x211 * x3163
            + x3074 * (x179 * x2047 + x183 * x2047 + x197 * x2048 + x201 * x2048)
            + x3113 * (x1199 * x2099 + x1200 * x2100)
            + x3158 * (x1187 * x3164 + x1189 * x3165)
            + x3159 * (x1182 * x2047 + x1184 * x2095)
            + x3160 * (x1193 * x3164 + x1194 * x3165)
            + x3166 * (x1178 * x2047 + x1179 * x2048)
        )
        - x1391 * x3011
        + x2113
        * (
            x3052 * x3113
            + x3053 * x3061
            + x3054 * x3160
            + x3056 * x3158
            + x3058 * x3159
            + x3059 * x405
            - x3062 * x405
            + x3156 * x527
            + x3156 * x530
            - x3157 * x544
        )
        + x2418 * x3089
        + x2421 * x496
        - x2433 * x3142
        - x2437 * x3021
        - x2438 * x3095
        - x2444 * (x1418 * x2140 + x2141 * x3018)
        - x2816 * x3141
        + x3029 * x3145
        - x3034 * x3148
        - x3040 * x3149
        - x3042 * x3150
        - x3046 * x3153
        - x3102 * x3147
        + x3136 * x794
        + x3137 * x794
        + x3138 * x794
        + x3139 * x794
        + x3140 * x794
        + x3161 * x500
        + x794
        * (
            dq_i3 * x160 * (x1362 * x2048 + x2048 * x787 + x3174 * x783 + x3174 * x788)
            - dq_i3 * x2164
            - dq_i3 * x2165
            - dq_i3 * x2166
            - dq_i3 * x2167
            + dq_j6 * x2048 * x2467
            - x2152 * x3157
            + x3053 * (x2048 * x756 + x3176 * x750)
            + x3113 * (x3170 * x763 + x3176 * x757)
            + x3158 * (x3176 * x480 + x3177 * x771)
            + x3159 * (x3170 * x779 + x3174 * x776)
            + x3160 * (x3176 * x471 + x3177 * x765)
        )
        + x864
        * (
            x1366 * x3162 * x3172
            - x2173 * x3157
            - x2176 * x405
            - x2177 * x405
            + x3053 * (x3172 * x828 + x3173 * x750)
            + x3113 * (x3172 * x759 + x3173 * x757)
            + x3158 * (x1372 * x2095 + x3173 * x480)
            + x3160 * (x3172 * x765 + x3173 * x471)
            + x3166 * (x3170 * x822 + x3174 * x821)
            + x3175 * (x1373 * x2095 + x2095 * x857 + x3174 * x851 + x3174 * x854)
        )
        + x962
        * (
            x1376 * x3163
            - x2199 * x3157
            - x2202 * x405
            - x2203 * x405
            + x3053 * (x3168 * x457 + x3169 * x459)
            + x3113 * (x3168 * x464 + x3169 * x466)
            + x3159 * (x2047 * x904 + x2095 * x908)
            + x3160 * (x3164 * x932 + x3165 * x935)
            + x3166 * (x2047 * x894 + x3170 * x896)
            + x3171 * (x3164 * x951 + x3164 * x956 + x3165 * x955 + x3165 * x959)
        )
    )
    K_block_list.append(
        -dq_i4 * x3201 * x794
        + x1070
        * (
            x1391 * x3208
            - x2423 * x3203
            - x2427 * x409
            - x2429 * x409
            + x3055 * (x1027 * x3210 + x3212 * x457)
            + x3114 * (x1038 * x3210 + x3212 * x464)
            + x3171 * (x1066 * x2499 + x1069 * x2500 + x2045 * x2499 + x2046 * x2500)
            + x3204 * (x1004 * x2499 + x1006 * x2548)
            + x3205 * (x1014 * x3209 + x1018 * x3210)
            + x3211 * (x2499 * x995 + x2500 * x2809)
        )
        + x1174
        * (
            x1466 * x3208
            - x2354 * x3203
            - x2357 * x409
            - x2359 * x409
            + x3055 * (x1087 * x2552 + x1145 * x2553)
            + x3129 * (x1167 * x2552 + x1169 * x2553 + x1170 * x2552 + x1172 * x2553)
            + x3158 * (x1131 * x3209 + x1135 * x3210)
            + x3204 * (x1107 * x2499 + x1110 * x2548)
            + x3205 * (x1119 * x3209 + x1122 * x3210)
            + x3211 * (x1100 * x2499 + x2358 * x2500)
        )
        + x1211 * x894
        + x1212 * x3182
        - x1242 * x3188
        + x135 * x3075
        + x135
        * (
            x211 * x3208
            - x2249 * x3203
            - x2256 * x409
            - x2257 * x409
            + x3079 * (x179 * x2499 + x183 * x2499 + x197 * x2500 + x201 * x2500)
            + x3114 * (x1199 * x2552 + x1200 * x2553)
            + x3158 * (x1193 * x3209 + x1194 * x3210)
            + x3204 * (x1182 * x2499 + x1184 * x2548)
            + x3205 * (x1187 * x3209 + x1189 * x3210)
            + x3211 * (x1178 * x2499 + x1179 * x2500)
        )
        - x1376 * x3011
        + x2569
        * (
            x2893 * x3202
            + x2895 * x3202
            + x3052 * x3114
            + x3054 * x3158
            + x3055 * x3061
            + x3056 * x3205
            + x3058 * x3204
            + x3059 * x409
            - x3062 * x409
            - x3203 * x544
        )
        + x2861 * x496
        - x2873 * x3022
        - x2875 * x3096
        - x2876 * x3141
        - x2880 * (x1418 * x2583 + x2584 * x3018)
        + x3029 * x3191
        - x3034 * x3195
        - x3038 * x3196
        - x3042 * x3197
        - x3046 * x3200
        - x3102 * x3194
        + x3178 * x3180
        + x3181 * x794
        + x3183 * x794
        + x3184 * x794
        + x3185 * x794
        + x3186 * x794
        + x3206 * x500
        + x794
        * (
            dq_i4 * x160 * (x1362 * x2500 + x2500 * x787 + x3218 * x783 + x3218 * x788)
            - dq_i4 * x2601
            - dq_i4 * x2602
            - dq_i4 * x2603
            - dq_i4 * x2604
            + x1352 * x2500 * x3207
            - x2591 * x3203
            + x3055 * (x2500 * x756 + x3220 * x750)
            + x3114 * (x3215 * x763 + x3220 * x757)
            + x3158 * (x3220 * x471 + x3221 * x765)
            + x3204 * (x3215 * x779 + x3218 * x776)
            + x3205 * (x3220 * x480 + x3221 * x771)
        )
        + x864
        * (
            x1366 * x3207 * x3216
            - x2608 * x3203
            - x2610 * x409
            - x2611 * x409
            + x3055 * (x3216 * x828 + x3217 * x750)
            + x3114 * (x3216 * x759 + x3217 * x757)
            + x3158 * (x3216 * x765 + x3217 * x471)
            + x3205 * (x1372 * x2548 + x3217 * x480)
            + x3211 * (x3215 * x822 + x3218 * x821)
            + x3219 * (x1373 * x2548 + x2548 * x857 + x3218 * x851 + x3218 * x854)
        )
        + x962
        * (
            dq_j6 * x626 * (x3209 * x951 + x3209 * x956 + x3210 * x955 + x3210 * x959)
            + x1376 * x3208
            - x2501 * x3203
            - x2504 * x409
            - x2505 * x409
            + x3055 * (x3213 * x457 + x3214 * x459)
            + x3114 * (x3213 * x464 + x3214 * x466)
            + x3158 * (x3209 * x932 + x3210 * x935)
            + x3204 * (x2499 * x904 + x2548 * x908)
            + x3211 * (x2499 * x894 + x3215 * x896)
        )
    )
    K_block_list.append(
        dq_i5 * sigma_kin_v_7_5 * x1209 * x496
        - dq_i5 * x3245 * x794
        + x1070
        * (
            x1391 * x3251
            - x2805 * x3247
            - x2808 * x413
            - x2810 * x413
            + x3057 * (x1027 * x3253 + x3255 * x457)
            + x3115 * (x1038 * x3253 + x3255 * x464)
            + x3175 * (x1066 * x2653 + x1069 * x2654 + x2045 * x2653 + x2046 * x2654)
            + x3204 * (x1014 * x3252 + x1018 * x3253)
            + x3248 * (x1004 * x2653 + x1006 * x2953)
            + x3254 * (x2653 * x995 + x2654 * x2809)
        )
        + x1174
        * (
            x1466 * x3251
            - x2748 * x3247
            - x2751 * x413
            - x2752 * x413
            + x3057 * (x1087 * x2955 + x1145 * x2956)
            + x3133 * (x1167 * x2955 + x1169 * x2956 + x1170 * x2955 + x1172 * x2956)
            + x3159 * (x1131 * x3252 + x1135 * x3253)
            + x3204 * (x1119 * x3252 + x1122 * x3253)
            + x3248 * (x1107 * x2653 + x1110 * x2953)
            + x3254 * (x1100 * x2653 + x2358 * x2654)
        )
        + x1211 * x487 * x821
        + x1212 * x3224
        - x1251 * x3188
        + x135 * x3081
        + x135
        * (
            x211 * x3251
            - x2655 * x3247
            - x2661 * x413
            - x2662 * x413
            + x3084 * (x179 * x2653 + x183 * x2653 + x197 * x2654 + x201 * x2654)
            + x3115 * (x1199 * x2955 + x1200 * x2956)
            + x3159 * (x1193 * x3252 + x1194 * x3253)
            + x3204 * (x1187 * x3252 + x1189 * x3253)
            + x3248 * (x1182 * x2653 + x1184 * x2953)
            + x3254 * (x1178 * x2653 + x1179 * x2654)
        )
        - x1366 * x3008 * x3225
        + x2968
        * (
            x3052 * x3115
            + x3054 * x3159
            + x3056 * x3204
            + x3057 * x3061
            + x3058 * x3248
            + x3059 * x413
            - x3062 * x413
            + x3246 * x527
            + x3246 * x530
            - x3247 * x544
        )
        - x3023 * x3230
        + x3029 * x3235
        - x3034 * x3239
        - x3038 * x3240
        - x3040 * x3241
        - x3046 * x3244
        - x3097 * x3231
        - x3102 * x3238
        - x3142 * x3232
        + x3178 * x3222
        + x3223 * x794
        + x3226 * x794
        + x3227 * x794
        + x3228 * x794
        + x3229 * x794
        - x3236 * (x1418 * x2969 + x2970 * x3018)
        + x3249 * x500
        + x794
        * (
            dq_i5 * x160 * (x1362 * x2654 + x2654 * x787 + x3260 * x783 + x3260 * x788)
            - dq_i5 * x2982
            - dq_i5 * x2983
            - dq_i5 * x2984
            - dq_i5 * x2985
            + x1352 * x2654 * x3250
            - x2973 * x3247
            + x3057 * (x2654 * x756 + x3262 * x750)
            + x3115 * (x3261 * x763 + x3262 * x757)
            + x3159 * (x3262 * x471 + x3263 * x765)
            + x3204 * (x3262 * x480 + x3263 * x771)
            + x3248 * (x3260 * x776 + x3261 * x779)
        )
        + x864
        * (
            dq_j6 * x594 * (x1373 * x2953 + x2953 * x857 + x3260 * x851 + x3260 * x854)
            + x1366 * x3250 * x3258
            - x2925 * x3247
            - x2928 * x413
            - x2929 * x413
            + x3057 * (x3258 * x828 + x3259 * x750)
            + x3115 * (x3258 * x759 + x3259 * x757)
            + x3159 * (x3258 * x765 + x3259 * x471)
            + x3204 * (x1372 * x2953 + x3259 * x480)
            + x3254 * (x3260 * x821 + x3261 * x822)
        )
        + x962
        * (
            x1376 * x3251
            - x2862 * x3247
            - x2866 * x413
            - x2868 * x413
            + x3057 * (x3256 * x457 + x3257 * x459)
            + x3115 * (x3256 * x464 + x3257 * x466)
            + x3159 * (x3252 * x932 + x3253 * x935)
            + x3219 * (x3252 * x951 + x3252 * x956 + x3253 * x955 + x3253 * x959)
            + x3248 * (x2653 * x904 + x2953 * x908)
            + x3254 * (x2653 * x894 + x2654 * x3182)
        )
    )
    K_block_list.append(
        x1070
        * (
            2 * dq_i1 * (x1027 * x3299 + x3147 + x3304 * x457)
            + 2 * dq_i2 * (x1038 * x3299 + x3148 + x3304 * x464)
            + dq_i3 * (x1054 * x3298 + x1057 * x3298 + x1061 * x3299 + x1063 * x3299 + x3154)
            + 2 * dq_i4 * (x1014 * x3298 + x1018 * x3299 + x3149)
            + 2 * dq_i5 * (x1004 * x3300 + x1006 * x3301 + x3150)
            + 2 * dq_i6 * dq_i7 * dq_j6 * x139 * x142 * x145 * x147 * x149 * x152 * x208 * x3026 * x390 * x975
            + 2 * dq_i6 * (x3153 + x3300 * x995 + x3303 * x997)
            + 2 * dq_i7 * sigma_kin_v_7_6 * x139 * x142 * x145 * x147 * x149 * x152 * x192 * x208 * x390 * x975
            - x3136
            - x3137
            - x3138
            - x3139
            - x3140
            - x3145 * x3294
            - x3151 * x417
            - x3152 * x417
            - x521 * x988
        )
        + x108 * x3264 * (sigma_pot_6_c * x100 + sigma_pot_6_s * x97)
        + x1174
        * (
            2 * dq_i1 * (x1087 * x3296 + x1145 * x3297 + x3101)
            + dq_i2 * (x1163 * x160 * x3098 + x1167 * x3296 + x1168 * x160 * x3099 + x1169 * x3297 + x3109)
            + 2 * dq_i3 * (x1131 * x3298 + x1135 * x3299 + x3103)
            + 2 * dq_i4 * (x1119 * x3298 + x1122 * x3299 + x3104)
            + 2 * dq_i5 * (x1107 * x3300 + x1110 * x3301 + x3105)
            + 2 * dq_i6 * dq_i7 * dq_j6 * x1083 * x138 * x143 * x145 * x147 * x149 * x152 * x208 * x3026 * x390
            + 2 * dq_i6 * (x1100 * x3300 + x1101 * x3303 + x3108)
            + 2 * dq_i7 * sigma_kin_v_7_6 * x1083 * x138 * x143 * x145 * x147 * x149 * x152 * x192 * x208 * x390
            - x1091 * x521
            - x3090
            - x3091
            - x3092
            - x3093
            - x3094
            - x3100 * x3294
            - x3106 * x417
            - x3107 * x417
        )
        + x1225
        * (
            ddq_i6 * (x3024 * x452 + x3026 * x454)
            + x139 * x143 * x145 * x147 * x148 * x153 * x230 * x3026 * x390 * x498
            + x165 * x167 * x169 * x170 * x175 * x220 * x3024 * x450 * x498
            - x3234 * x449
            - x462 * (x3025 * x457 + x3027 * x459)
            - x468 * (x3098 * x377 + x3099 * x383)
            - x478 * (x3143 * x471 + x3144 * x475)
            - x486 * (x295 * x3189 + x302 * x3190)
            - x493 * (x261 * x3233 + x3234 * x490)
            - x499 * (x220 * x3270 + x3271 * x495)
        )
        + x135 * x3085
        + x135
        * (
            dq_i1 * (x178 * x3296 + x182 * x3296 + x196 * x3297 + x200 * x3297 + x3047)
            + 2 * dq_i2 * (x1199 * x3296 + x1200 * x3297 + x3033)
            + 2 * dq_i3 * (x1193 * x3298 + x1194 * x3299 + x3037)
            + 2 * dq_i4 * (x1187 * x3298 + x1189 * x3299 + x3039)
            + 2 * dq_i5 * (x1182 * x3300 + x1184 * x3301 + x3041)
            + 2 * dq_i6 * dq_i7 * dq_j6 * x136 * x137 * x139 * x143 * x145 * x147 * x149 * x152 * x208 * x3026
            + 2 * dq_i6 * (x1178 * x3300 + x1179 * x3302 + x3045)
            + 2 * dq_i7 * sigma_kin_v_7_6 * x136 * x137 * x139 * x143 * x145 * x147 * x149 * x152 * x192 * x208
            - x215 * x521
            - x3007
            - x3014
            - x3015
            - x3016
            - x3017
            - x3028 * x3294
            - x3043 * x417
            - x3044 * x417
        )
        - x1352 * x3008 * x3009
        + x1362 * x194
        - x149
        * x3288
        * x3289
        * (x1533 * x512 + x1533 * x514 - x3286 * x516 + x3287 * x411 + x3287 * x415 + x419 * x507 + x423 * x516)
        + x1598 * x3063
        - 4 * x160 * x3285
        + x194 * x787
        + x2922 * x783
        + x2922 * x788
        + x3029 * x3272
        - x3034 * x3276
        - x3038 * x3278
        - x3040 * x3279
        - x3042 * x3280
        - x3102 * x3274
        + x3117 * x699
        + x3161 * x656
        + x3206 * x628
        + x3249 * x596
        + x3265 * x794
        + x3266 * x794
        + x3267 * x794
        + x3268 * x794
        + x3269 * x794
        + x3295
        * (
            x1567
            + x189 * x527
            + x189 * x530
            + x2560 * x3293
            + x2566 * x3293
            - x3290
            + x3291 * x520
            + x3292 * x508
            + x3293 * x512
            + x3293 * x514
            - x3294 * x544
            + x533
            + x536
            + x539
            + x542
            + x554
        )
        + x794
        * (
            2 * dq_i1 * (x3274 + x3302 * x756 + x3310 * x750)
            + 2 * dq_i2 * (x3276 + x3303 * x763 + x3310 * x757)
            + 2 * dq_i3 * (x3278 + x3310 * x471 + x3311 * x765)
            + 2 * dq_i4 * (x3279 + x3310 * x480 + x3311 * x771)
            + 2 * dq_i5 * (x3280 + x3303 * x779 + x3309 * x776)
            + 2 * dq_i6 * dq_i7 * dq_j6 * x139 * x143 * x145 * x147 * x148 * x152 * x208 * x3026 * x390 * x730
            - dq_i6 * x3281
            - dq_i6 * x3282
            - dq_i6 * x3283
            - dq_i6 * x3284
            + dq_i6 * (x160 * x3270 * x781 + x3285 + x3302 * x787 + x3303 * x785 + x3309 * x783)
            + 2 * dq_i7 * sigma_kin_v_7_6 * x139 * x143 * x145 * x147 * x148 * x152 * x192 * x208 * x390 * x730
            - x3265
            - x3266
            - x3267
            - x3268
            - x3269
            - x3272 * x3294
            - x521 * x747
        )
        + x864
        * (
            2 * dq_i1 * (x3238 + x3307 * x750 + x3308 * x828)
            + 2 * dq_i2 * (x3239 + x3307 * x757 + x3308 * x759)
            + 2 * dq_i3 * (x3240 + x3307 * x471 + x3308 * x765)
            + 2 * dq_i4 * (x1372 * x3301 + x3241 + x3307 * x480)
            + dq_i5 * (x1373 * x3301 + x3245 + x3301 * x857 + x3309 * x851 + x3309 * x854)
            + 2 * dq_i6 * dq_i7 * dq_j6 * x139 * x143 * x145 * x146 * x149 * x152 * x208 * x3026 * x390 * x799
            + 2 * dq_i6 * (x3244 + x3303 * x822 + x3309 * x821)
            + 2 * dq_i7 * sigma_kin_v_7_6 * x139 * x143 * x145 * x146 * x149 * x152 * x192 * x208 * x390 * x799
            - x3223
            - x3226
            - x3227
            - x3228
            - x3229
            - x3235 * x3294
            - x3243 * x417
            - x3275 * x418 * x801
            - x521 * x812
        )
        + x962
        * (
            2 * dq_i1 * (x3194 + x3305 * x457 + x3306 * x459)
            + 2 * dq_i2 * (x3195 + x3305 * x464 + x3306 * x466)
            + 2 * dq_i3 * (x3196 + x3298 * x932 + x3299 * x935)
            + dq_i4 * (x160 * x3189 * x948 + x160 * x3190 * x953 + x3201 + x3298 * x951 + x3299 * x955)
            + 2 * dq_i5 * (x3197 + x3300 * x904 + x3301 * x908)
            + 2 * dq_i6 * dq_i7 * dq_j6 * x139 * x143 * x144 * x147 * x149 * x152 * x208 * x3026 * x390 * x875
            + 2 * dq_i6 * (x3200 + x3300 * x894 + x3303 * x896)
            + 2 * dq_i7 * sigma_kin_v_7_6 * x139 * x143 * x144 * x147 * x149 * x152 * x192 * x208 * x390 * x875
            - x3181
            - x3183
            - x3184
            - x3185
            - x3186
            - x3191 * x3294
            - x3198 * x417
            - x3199 * x417
            - x521 * x887
        )
    )
    K_block_list.append(
        -sigma_kin_v_7_1 * x3361 * x3363
        + sigma_kin_v_7_1
        * x3418
        * (
            dq_i1 * x3415
            - x3368 * x730
            + x3377 * x3417
            + x3380 * x730
            - x3382 * x730
            + x3387 * x3412
            + x3390 * x3416
            + x3392 * x3416
            + x3393 * x3416
            + x3400 * x3416
        )
        - x1179 * x3350
        - x1184 * x3348
        - x1189 * x3345
        - x1194 * x3342
        + x1206 * x3332
        + x1207 * x3317
        + x1227 * x3329
        + x1244 * x3328
        + x1253 * x3327
        + x1274 * x3326
        + x1292 * x3324 * x707
        + x156 * x3315
        + x1618 * x3325
        + x1766 * x3318
        - x197 * x3335
        - x201 * x3335
        + x207 * x3336
        - x207 * x3353
        + x2242 * x3319
        + x3005 * x3313
        + x3320 * x3321
        + x3320 * x3322
        - 8 * x3337 * x3339 * x383 * x386
        + x3383
        * x3384
        * (
            -x1083 * x3368
            + x1083 * x3378
            + x1083 * x3380
            - x1083 * x3382
            + x3364 * x3366
            + x3366 * x3369
            + x3371 * x649
            + x3372 * x616
            + x3373 * x574
            + x3374 * x3375
        )
        + x3383
        * x3395
        * (
            -x3368 * x975
            + x3375 * x3394
            + x3378 * x975
            + x3380 * x975
            - x3382 * x975
            + x3385 * x3388
            + x3388 * x3389
            + x3390 * x3391
            + x3391 * x3392
            + x3391 * x3393
        )
        + x3403
        * x3404
        * (
            -x3368 * x875
            + x3375 * x3401
            + x3377 * x3402
            + x3380 * x875
            - x3382 * x875
            + x3390 * x3399
            + x3393 * x3399
            + x3396 * x3397
            + x3397 * x3398
            + x3399 * x3400
        )
        + x3403
        * x3410
        * (
            -x3368 * x799
            + x3375 * x3409
            + x3378 * x799
            + x3380 * x799
            - x3382 * x799
            + x3390 * x3408
            + x3392 * x3408
            + x3400 * x3408
            + x3405 * x3406
            + x3406 * x3407
        )
        + x3425
        * x424
        * (
            -x136 * x3382
            - x3367 * x3419
            + x3375 * x3423
            + x3377 * x3424
            + x3390 * x3419
            + x3392 * x3419
            + x3393 * x3419
            + x3400 * x3419
            + x3420 * x3421
            + x3420 * x3422
        )
        + x557
        * (
            dq_i1 * x3428
            - dq_i1 * x3429
            - dq_i1 * x548
            - x3367 * x3427
            + x3375 * x552
            + x3379 * x3427
            + x3386 * x3426
            + x3390 * x3427
            + x3392 * x3427
            + x3393 * x3427
            + x3400 * x3427
        )
    )
    K_block_list.append(
        -sigma_kin_v_7_2 * x3363 * x3442
        + sigma_kin_v_7_2
        * x3418
        * (
            dq_i2 * x3415
            + x3412 * x3452
            + x3416 * x3444
            + x3416 * x3445
            + x3416 * x3446
            + x3416 * x3455
            + x3417 * x3448
            - x3450 * x730
            - x3454 * x730
            + x3457 * x730
        )
        + x1085 * x3315
        - x1110 * x3348
        - x1122 * x3345
        - x1135 * x3342
        - x1144 * x154 * x3439 * x3440
        - x1169 * x3438
        - x1172 * x3438
        + x1337
        * x3384
        * (
            -x1083 * x3450
            - x1083 * x3454
            + x1083 * x3456
            + x3364 * x3461
            + x3369 * x3461
            + x3371 * x652
            + x3372 * x620
            + x3373 * x578
            + x3374 * x3447
            + x3455 * x3462
        )
        + x1337
        * x3395
        * (
            x3385 * x3453
            + x3389 * x3453
            + x3391 * x3445
            + x3391 * x3446
            + x3391 * x3455
            + x3394 * x3447
            - x3450 * x975
            - x3454 * x975
            + x3456 * x975
            + x3457 * x975
        )
        + x1460 * x3332
        + x1460 * x3431
        + x1465 * x3336
        - x1465 * x3353
        + x1569
        * (
            dq_i2 * x3428
            - dq_i2 * x3429
            - dq_i2 * x548
            + x3426 * x3451
            - x3427 * x3443
            + x3427 * x3444
            + x3427 * x3445
            + x3427 * x3446
            + x3427 * x3449
            + x3427 * x3455
            + x3447 * x552
        )
        + x1605 * x1979 * x3324
        + x1942 * x3318
        + x1949 * x3437
        + x1959 * x3436
        + x1967 * x3435
        + x2350 * x3430
        + x2351 * x3319
        - x2358 * x3350
        + x3321 * x3432
        + x3322 * x3432
        + x3325 * x729
        + x3404
        * x3459
        * (
            x3396 * x3458
            + x3398 * x3458
            + x3399 * x3444
            + x3399 * x3446
            + x3399 * x3455
            + x3401 * x3447
            + x3402 * x3448
            - x3450 * x875
            - x3454 * x875
            + x3457 * x875
        )
        + x3410
        * x3459
        * (
            x3405 * x3460
            + x3407 * x3460
            + x3408 * x3444
            + x3408 * x3445
            + x3408 * x3455
            + x3409 * x3447
            - x3450 * x799
            - x3454 * x799
            + x3456 * x799
            + x3457 * x799
        )
        + x3425
        * x531
        * (
            -x136 * x3450
            + x3366 * x3421
            + x3366 * x3422
            - x3419 * x3443
            + x3419 * x3444
            + x3419 * x3445
            + x3419 * x3446
            + x3419 * x3449
            + x3423 * x3447
            + x3424 * x3448
        )
        + x3433 * x3434
    )
    K_block_list.append(
        -sigma_kin_v_7_3 * x3363 * x3475
        + sigma_kin_v_7_3
        * x3418
        * (
            dq_i3 * x3415
            + x3412 * x3495
            + x3416 * x3481
            + x3416 * x3482
            + x3416 * x3483
            + x3416 * x3492
            + x3417 * x3485
            - x3487 * x730
            - x3489 * x730
            + x3497 * x730
        )
        - x1006 * x3348
        - x1035 * x1037 * x3471 * x3473
        - x1069 * x3468
        - x1390 * x3353
        + x1847
        * x3384
        * (
            -x1083 * x3487
            - x1083 * x3489
            + x1083 * x3493
            + x3340 * x3491
            + x3371 * x655
            + x3372 * x624
            + x3373 * x585
            + x3374 * x3484
            + x3462 * x3492
            + x3477 * x3488
        )
        + x1847
        * x3395
        * (
            x3385 * x3500
            + x3389 * x3500
            + x3391 * x3481
            + x3391 * x3482
            + x3391 * x3483
            + x3391 * x3492
            + x3394 * x3484
            - x3487 * x975
            - x3489 * x975
            + x3493 * x975
        )
        - 8 * x1927 * x302 * x3344 * x3470
        - x1939 * x3471 * x3472
        - x2046 * x3468
        + x2095 * x2425 * x3324
        + x2113
        * (
            dq_i3 * x3428
            - dq_i3 * x3429
            - dq_i3 * x548
            + x3426 * x3494
            - x3427 * x3478
            + x3427 * x3481
            + x3427 * x3482
            + x3427 * x3483
            + x3427 * x3486
            + x3427 * x3492
            + x3484 * x552
        )
        + x2138 * x2444 * x3324
        + x2418 * x3430
        + x2419 * x3319
        + x2433 * x3467
        + x2437 * x3326
        + x2438 * x3324 * x3434
        - x2809 * x3350
        + x2816 * x3466
        + x3321 * x3465
        + x3322 * x3465
        + x3332 * x977
        + x3404
        * x3498
        * (
            x3396 * x3496
            + x3398 * x3496
            + x3399 * x3481
            + x3399 * x3483
            + x3399 * x3492
            + x3401 * x3484
            + x3402 * x3485
            - x3487 * x875
            - x3489 * x875
            + x3497 * x875
        )
        + x3410
        * x3498
        * (
            x3405 * x3499
            + x3407 * x3499
            + x3408 * x3481
            + x3408 * x3482
            + x3408 * x3492
            + x3409 * x3484
            - x3487 * x799
            - x3489 * x799
            + x3493 * x799
            + x3497 * x799
        )
        + x3425
        * x534
        * (
            -x136 * x3487
            + x3340 * x3480
            - x3419 * x3478
            + x3419 * x3481
            + x3419 * x3482
            + x3419 * x3483
            + x3419 * x3486
            + x3423 * x3484
            + x3424 * x3485
            + x3476 * x3477
        )
        + x3431 * x977
        + x3463 * x977
        + x3464 * x977
        + x3469 * x976
    )
    K_block_list.append(
        -sigma_kin_v_7_4 * x3363 * x3510
        + sigma_kin_v_7_4
        * x3418
        * (
            x3416 * x3513
            + x3416 * x3514
            + x3416 * x3515
            + x3416 * x3521
            + x3417 * x3517
            - x3519 * x730
            - x3520 * x730
            + x3526 * x730
            + x3529 * x3530
            + x3530 * x3531
        )
        + x1242 * x3504
        - x1375 * x3353
        + x2307
        * x3384
        * (
            -x1083 * x3519
            - x1083 * x3520
            + x1083 * x3522
            + x3343 * x3491
            + x3371 * x624
            + x3372 * x627
            + x3373 * x589
            + x3374 * x3516
            + x3462 * x3521
            + x3488 * x3511
        )
        + x2307
        * x3395
        * (
            x3343 * x3525
            + x3391 * x3513
            + x3391 * x3515
            + x3391 * x3521
            + x3394 * x3516
            + x3511 * x3523
            - x3519 * x975
            - x3520 * x975
            + x3522 * x975
            + x3526 * x975
        )
        + x2538
        * x3404
        * (
            x3396 * x3532
            + x3398 * x3532
            + x3399 * x3513
            + x3399 * x3514
            + x3399 * x3515
            + x3399 * x3521
            + x3401 * x3516
            + x3402 * x3517
            - x3519 * x875
            - x3520 * x875
        )
        + x2538
        * x3410
        * (
            x3405 * x3527
            + x3407 * x3527
            + x3408 * x3513
            + x3408 * x3514
            + x3408 * x3521
            + x3409 * x3516
            - x3519 * x799
            - x3520 * x799
            + x3522 * x799
            + x3526 * x799
        )
        + x2569
        * (
            dq_i4 * x3428
            - dq_i4 * x3429
            - dq_i4 * x548
            + x3426 * x3533
            - x3427 * x3512
            + x3427 * x3513
            + x3427 * x3514
            + x3427 * x3515
            + x3427 * x3518
            + x3427 * x3521
            + x3516 * x552
        )
        + x2581 * x2880 * x3324
        + x2865 * x3505
        + x2873 * x3327
        + x2875 * x3435
        + x2876 * x3466
        + x3180 * x3501
        - x3182 * x3350
        + x3321 * x3503
        + x3322 * x3503
        + x3332 * x877
        - x3342 * x935
        - x3348 * x908
        + x3425
        * x537
        * (
            -x136 * x3519
            + x3343 * x3480
            - x3419 * x3512
            + x3419 * x3513
            + x3419 * x3514
            + x3419 * x3515
            + x3419 * x3518
            + x3423 * x3516
            + x3424 * x3517
            + x3476 * x3511
        )
        + x3431 * x877
        + x3463 * x877
        + x3464 * x877
        + x3469 * x876
        - x3472 * x3508
        - x3473 * x3508 * x466
        + x3502 * x877
        - x3506 * x955
        - x3506 * x959
    )
    K_block_list.append(
        dq_i5 * x1226 * x3535
        - sigma_kin_v_7_5 * x3363 * x3544
        + sigma_kin_v_7_5
        * x3418
        * (
            dq_i5 * x3415
            + x157 * x3412 * x3558
            + x3416 * x3547
            + x3416 * x3548
            + x3416 * x3549
            + x3416 * x3555
            + x3417 * x3551
            - x3553 * x730
            - x3554 * x730
            + x3557 * x730
        )
        + x1251 * x3504
        + x1365 * x1650 * x3336
        - x1365 * x3352 * x3540
        - x1373 * x3536
        + x2704
        * x3384
        * (
            -x1083 * x3553
            - x1083 * x3554
            + x1083 * x3556
            + x3346 * x3491
            + x3371 * x585
            + x3372 * x589
            + x3373 * x595
            + x3374 * x3550
            + x3462 * x3555
            + x3488 * x3545
        )
        + x2704
        * x3395
        * (
            x3346 * x3525
            + x3391 * x3547
            + x3391 * x3549
            + x3391 * x3555
            + x3394 * x3550
            + x3523 * x3545
            - x3553 * x975
            - x3554 * x975
            + x3556 * x975
            + x3557 * x975
        )
        + x2936 * x3236 * x3324
        + x2945
        * x3404
        * (
            x3396 * x3527
            + x3398 * x3527
            + x3399 * x3547
            + x3399 * x3548
            + x3399 * x3555
            + x3401 * x3550
            + x3402 * x3551
            - x3553 * x875
            - x3554 * x875
            + x3557 * x875
        )
        + x2945
        * x3410
        * (
            x3405 * x3559
            + x3407 * x3559
            + x3408 * x3547
            + x3408 * x3548
            + x3408 * x3549
            + x3408 * x3555
            + x3409 * x3550
            - x3553 * x799
            - x3554 * x799
            + x3556 * x799
        )
        + x2968
        * (
            dq_i5 * x3428
            - dq_i5 * x3429
            - dq_i5 * x548
            + x3426 * x3558
            - x3427 * x3546
            + x3427 * x3547
            + x3427 * x3548
            + x3427 * x3549
            + x3427 * x3552
            + x3427 * x3555
            + x3550 * x552
        )
        + x3222 * x3501
        - x3224 * x3350
        + x3230 * x3328
        + x3231 * x3436
        + x3232 * x3467
        + x3321 * x3534
        + x3322 * x3534
        + x3332 * x802
        + x3425
        * x540
        * (
            -x136 * x3553
            + x3346 * x3480
            - x3419 * x3546
            + x3419 * x3547
            + x3419 * x3548
            + x3419 * x3549
            + x3419 * x3552
            + x3423 * x3550
            + x3424 * x3551
            + x3476 * x3545
        )
        + x3431 * x802
        - x3440 * x3539 * x828
        + x3463 * x802
        + x3464 * x802
        - x3473 * x3539 * x759
        + x3502 * x802
        - x3536 * x857
        - x3538 * x3542 * x845
        - x3540 * x3541
    )
    K_block_list.append(
        dq_i6 * x1243 * x3535
        + dq_i6 * x2095 * x3433
        + dq_i6 * x2436 * x3329
        + dq_i6 * x2874 * x3437
        + sigma_kin_v_7_6
        * x3418
        * (
            x3416 * x3571
            + x3416 * x3572
            + x3416 * x3573
            + x3416 * x3574
            + x3416 * x3579
            + x3417 * x3575
            + x3529 * x3587
            + x3531 * x3587
            - x3577 * x730
            - x3578 * x730
        )
        + x1212 * x3562 * x448
        + x1225 * x3234 * x3324 * x498
        + x1251 * x2864 * x3505
        - x1362 * x3561
        - 8 * x190 * x3563 * x394 * x448
        - x228 * x233 * x3440 * x3564 * x754
        + x3051
        * x3384
        * (
            -x1083 * x3577
            - x1083 * x3578
            + x1083 * x3580
            + x1083 * x3581
            + x3349 * x3491
            + x3371 * x478
            + x3372 * x486
            + x3373 * x493
            + x3462 * x3579
            + x3488 * x3569
        )
        + x3051
        * x3395
        * (
            x3349 * x3525
            + x3391 * x3571
            + x3391 * x3573
            + x3391 * x3574
            + x3391 * x3579
            + x3523 * x3569
            - x3577 * x975
            - x3578 * x975
            + x3580 * x975
            + x3581 * x975
        )
        + x3264 * x3312 * (sigma_pot_6_c * x219 - sigma_pot_6_s * x218)
        - x3289 * x3362 * x3568
        + x3295
        * (
            dq_i6 * x3426 * x394
            + dq_i6 * x3428
            - dq_i6 * x3429
            - dq_i6 * x548
            - x3427 * x3570
            + x3427 * x3571
            + x3427 * x3572
            + x3427 * x3573
            + x3427 * x3574
            + x3427 * x3579
            + x3576 * x552
        )
        + x3321 * x3560
        + x3322 * x3560
        + x3332 * x733
        - 8 * x3346 * x3565 * x490
        + x3404
        * x3583
        * (
            x3396 * x3582
            + x3398 * x3582
            + x3399 * x3571
            + x3399 * x3572
            + x3399 * x3574
            + x3399 * x3579
            + x3402 * x3575
            - x3577 * x875
            - x3578 * x875
            + x3581 * x875
        )
        + x3410
        * x3583
        * (
            x3349 * x3586
            + x3408 * x3571
            + x3408 * x3572
            + x3408 * x3573
            + x3408 * x3579
            + x3569 * x3584
            - x3577 * x799
            - x3578 * x799
            + x3580 * x799
            + x3581 * x799
        )
        + x3425
        * x543
        * (
            -x136 * x3577
            + x3349 * x3480
            - x3419 * x3570
            + x3419 * x3571
            + x3419 * x3572
            + x3419 * x3573
            + x3419 * x3574
            + x3423 * x3576
            + x3424 * x3575
            + x3476 * x3569
        )
        + x3431 * x733
        + x3463 * x733
        + x3464 * x733
        - x3473 * x3565 * x760
        + x3502 * x733
        - 8 * x3541 * x3566
        - x3542 * x3566 * x771
        - x3561 * x787
    )
    K_block_list.append(
        -ddq_j7
        * x3330
        * x394
        * x516
        * (x137 * x402 + x137 * x407 + x137 * x412 + x137 * x416 + x1528 * x393 + x3599 + x444 * x517 - x551)
        - dq_i1 * x3596 * x459
        - dq_i2 * x3596 * x466
        - dq_i3 * x3597 * x475
        - dq_i4 * x3597 * x483
        - dq_i5 * x1365 * x264 * x3595
        - sigma_kin_v_7_1 * x1598 * x3361 * x3598
        + sigma_kin_v_7_1 * x3314 * x3590
        + sigma_kin_v_7_2 * x278 * x3591
        - sigma_kin_v_7_2 * x3442 * x3598 * x699
        + sigma_kin_v_7_3 * x272 * x3591
        - sigma_kin_v_7_3 * x3475 * x3598 * x656
        + sigma_kin_v_7_4 * x248 * x3591
        - sigma_kin_v_7_4 * x3510 * x3598 * x628
        - sigma_kin_v_7_5 * x3544 * x3598 * x596
        - sigma_kin_v_7_6 * x3568 * x3598 * x500
        + sigma_kin_v_7_7
        * x3418
        * (
            -x1566 * x3643
            + x2560 * x3645
            + x2560 * x3646
            + x2566 * x3645
            + x2566 * x3646
            + x3413 * x3417
            + x3413 * x3563
            - x3414 * x3611
            + x3414 * x3615
            + x3562 * x3603
            + x3563 * x3619
            + x3604 * x3642
            - x3606 * x3643
            - x3607 * x3643
            - x3608 * x3643
            - x3609 * x3643
            - x3610 * x3643
            + x3642 * x549
            + x3644 * x508
            + x3645 * x512
            + x3646 * x512
            + x3646 * x514
        )
        + x101 * x3264 * (sigma_pot_7_c * x107 + sigma_pot_7_s * x104)
        - 8 * x1365 * x231 * x3349 * x505
        + x157
        * x555
        * (
            x1559
            + x1567
            + x190 * x520
            + x2107
            + x2560 * x3601
            + x2561
            + x2566 * x3601
            + x2963
            + x3290
            + x3365 * x421 * x529
            + x3599 * x552
            + x3600 * x508
            + x3601 * x512
            + x3603 * x504
            + x509
            + x546
            - 2 * x553
        )
        + x3286 * x3331 * x3588 * x503
        + x3321 * x3591
        + x3322 * x3591
        + x3384
        * x3625
        * (
            -x1566 * x3618
            + x190 * x3491
            + x2560 * x3622
            + x2560 * x3624
            + x2566 * x3622
            + x2566 * x3624
            - x3374 * x3611
            + x3374 * x3615
            + x3439 * x3490
            + x3488 * x3604
            + x3488 * x549
            + x3602 * x3621
            - x3606 * x3618
            - x3607 * x3618
            - x3608 * x3618
            - x3609 * x3618
            - x3610 * x3618
            + x3614 * x3620
            + x3614 * x3623
            + x3619 * x3620
            + x3619 * x3623
            + x3624 * x514
        )
        + x3395
        * x3625
        * (
            -x1566 * x3626
            + x190 * x3525
            + x2560 * x3628
            + x2560 * x3629
            + x2566 * x3628
            + x2566 * x3629
            - x3394 * x3611
            + x3394 * x3615
            + x3470 * x3524
            + x3470 * x3614
            + x3470 * x3619
            + x3523 * x3604
            + x3523 * x549
            - x3606 * x3626
            - x3607 * x3626
            - x3608 * x3626
            - x3609 * x3626
            - x3610 * x3626
            + x3614 * x3627
            + x3619 * x3627
            + x3628 * x512
            + x3629 * x512
        )
        + x3404
        * x3636
        * (
            x137 * x3402 * x3632
            - x1566 * x3631
            + x2566 * x3633
            + x2566 * x3635
            - x3401 * x3611
            + x3401 * x3615
            + x3402 * x3614
            + x3402 * x3619
            + x3507 * x3614
            + x3507 * x3619
            + x3604 * x3630
            - x3606 * x3631
            - x3607 * x3631
            - x3608 * x3631
            - x3609 * x3631
            - x3610 * x3631
            + x3630 * x549
            + x3632 * x3634
            + x3633 * x512
            + x3633 * x514
            + x3635 * x512
            + x3635 * x514
        )
        + x3410
        * x3636
        * (
            -x1566 * x3637
            + x190 * x3586
            + x2560 * x3639
            + x2560 * x3641
            - x3409 * x3611
            + x3409 * x3615
            + x3537 * x3614
            + x3537 * x3619
            + x3584 * x3604
            + x3584 * x549
            + x3585 * x3640
            + x3602 * x3612 * x799
            - x3606 * x3637
            - x3607 * x3637
            - x3608 * x3637
            - x3609 * x3637
            - x3610 * x3637
            + x3614 * x3638
            + x3619 * x3638
            + x3639 * x512
            + x3641 * x512
            + x3641 * x514
        )
        + x3425
        * x3617
        * (
            -x1566 * x3605
            + x190 * x3480
            + x2560 * x3613
            + x2560 * x3616
            + x2566 * x3613
            + x2566 * x3616
            + x3337 * x3479
            + x3337 * x3614
            - x3423 * x3611
            + x3423 * x3615
            + x3424 * x3614
            + x3476 * x3604
            + x3476 * x549
            - x3605 * x3606
            - x3605 * x3607
            - x3605 * x3608
            - x3605 * x3609
            - x3605 * x3610
            + x3613 * x512
            + x3613 * x514
            + x3616 * x512
            + x3616 * x514
        )
        + x3592 * x3593
        + x3592 * x525
        - x3593 * x3594
        - x3594 * x525
    )

    return K_block_list
