import torch


def get_K_blocks_GIP_sum_PANDA_7dof_no_subs(
    X1, X2, pos_indices, vel_indices, acc_indices, sigma_kin_vel_list, sigma_kin_pos_list, sigma_pot_list
):

    q_i1 = X1[:, pos_indices[0] : pos_indices[0] + 1]
    dq_i1 = X1[:, vel_indices[0] : vel_indices[0] + 1]
    ddq_i1 = X1[:, acc_indices[0] : acc_indices[0] + 1]
    q_j1 = X2[:, pos_indices[0] : pos_indices[0] + 1]
    dq_j1 = X2[:, vel_indices[0] : vel_indices[0] + 1]
    ddq_j1 = X2[:, acc_indices[0] : acc_indices[0] + 1]
    q_i2 = X1[:, pos_indices[1] : pos_indices[1] + 1]
    dq_i2 = X1[:, vel_indices[1] : vel_indices[1] + 1]
    ddq_i2 = X1[:, acc_indices[1] : acc_indices[1] + 1]
    q_j2 = X2[:, pos_indices[1] : pos_indices[1] + 1]
    dq_j2 = X2[:, vel_indices[1] : vel_indices[1] + 1]
    ddq_j2 = X2[:, acc_indices[1] : acc_indices[1] + 1]
    q_i3 = X1[:, pos_indices[2] : pos_indices[2] + 1]
    dq_i3 = X1[:, vel_indices[2] : vel_indices[2] + 1]
    ddq_i3 = X1[:, acc_indices[2] : acc_indices[2] + 1]
    q_j3 = X2[:, pos_indices[2] : pos_indices[2] + 1]
    dq_j3 = X2[:, vel_indices[2] : vel_indices[2] + 1]
    ddq_j3 = X2[:, acc_indices[2] : acc_indices[2] + 1]
    q_i4 = X1[:, pos_indices[3] : pos_indices[3] + 1]
    dq_i4 = X1[:, vel_indices[3] : vel_indices[3] + 1]
    ddq_i4 = X1[:, acc_indices[3] : acc_indices[3] + 1]
    q_j4 = X2[:, pos_indices[3] : pos_indices[3] + 1]
    dq_j4 = X2[:, vel_indices[3] : vel_indices[3] + 1]
    ddq_j4 = X2[:, acc_indices[3] : acc_indices[3] + 1]
    q_i5 = X1[:, pos_indices[4] : pos_indices[4] + 1]
    dq_i5 = X1[:, vel_indices[4] : vel_indices[4] + 1]
    ddq_i5 = X1[:, acc_indices[4] : acc_indices[4] + 1]
    q_j5 = X2[:, pos_indices[4] : pos_indices[4] + 1]
    dq_j5 = X2[:, vel_indices[4] : vel_indices[4] + 1]
    ddq_j5 = X2[:, acc_indices[4] : acc_indices[4] + 1]
    q_i6 = X1[:, pos_indices[5] : pos_indices[5] + 1]
    dq_i6 = X1[:, vel_indices[5] : vel_indices[5] + 1]
    ddq_i6 = X1[:, acc_indices[5] : acc_indices[5] + 1]
    q_j6 = X2[:, pos_indices[5] : pos_indices[5] + 1]
    dq_j6 = X2[:, vel_indices[5] : vel_indices[5] + 1]
    ddq_j6 = X2[:, acc_indices[5] : acc_indices[5] + 1]
    q_i7 = X1[:, pos_indices[6] : pos_indices[6] + 1]
    dq_i7 = X1[:, vel_indices[6] : vel_indices[6] + 1]
    ddq_i7 = X1[:, acc_indices[6] : acc_indices[6] + 1]
    q_j7 = X2[:, pos_indices[6] : pos_indices[6] + 1]
    dq_j7 = X2[:, vel_indices[6] : vel_indices[6] + 1]
    ddq_j7 = X2[:, acc_indices[6] : acc_indices[6] + 1]

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
    x34 = 2 * x33**2
    x35 = sigma_kin_p_2_1_c * x14 - sigma_kin_p_2_1_s * x15
    x36 = sigma_kin_p_2_1_c * x15 - sigma_kin_p_2_1_s * x14
    x37 = x27 * x35 * x36
    x38 = dq_i3 * dq_j3
    x39 = sigma_kin_v_3_1 * x31 + sigma_kin_v_3_2 * x32 + sigma_kin_v_3_3 * x38
    x40 = 2 * x39**2
    x41 = sigma_kin_p_3_2_c * x22 + sigma_kin_p_3_2_off + sigma_kin_p_3_2_s * x25
    x42 = x41**2
    x43 = sigma_kin_p_3_1_c * x2 + sigma_kin_p_3_1_s * x5
    x44 = torch.cos(q_i3)
    x45 = torch.cos(q_j3)
    x46 = x44 * x45
    x47 = torch.sin(q_i3)
    x48 = torch.sin(q_j3)
    x49 = x47 * x48
    x50 = sigma_kin_p_3_3_c * x46 + sigma_kin_p_3_3_off + sigma_kin_p_3_3_s * x49
    x51 = x50**2
    x52 = sigma_kin_p_3_1_c * x5 + sigma_kin_p_3_1_off + sigma_kin_p_3_1_s * x2
    x53 = x51 * x52
    x54 = x43 * x53
    x55 = x42 * x54
    x56 = sigma_kin_p_3_1_c * x14 - sigma_kin_p_3_1_s * x15
    x57 = sigma_kin_p_3_1_c * x15 - sigma_kin_p_3_1_s * x14
    x58 = x51 * x56 * x57
    x59 = x42 * x58
    x60 = dq_i4 * dq_j4
    x61 = sigma_kin_v_4_1 * x31 + sigma_kin_v_4_2 * x32 + sigma_kin_v_4_3 * x38 + sigma_kin_v_4_4 * x60
    x62 = 2 * x61**2
    x63 = sigma_kin_p_4_3_c * x46 + sigma_kin_p_4_3_off + sigma_kin_p_4_3_s * x49
    x64 = x63**2
    x65 = torch.cos(q_i4)
    x66 = torch.cos(q_j4)
    x67 = x65 * x66
    x68 = torch.sin(q_i4)
    x69 = torch.sin(q_j4)
    x70 = x68 * x69
    x71 = sigma_kin_p_4_4_c * x67 + sigma_kin_p_4_4_off + sigma_kin_p_4_4_s * x70
    x72 = x71**2
    x73 = x64 * x72
    x74 = sigma_kin_p_4_1_c * x2 + sigma_kin_p_4_1_s * x5
    x75 = sigma_kin_p_4_2_c * x22 + sigma_kin_p_4_2_off + sigma_kin_p_4_2_s * x25
    x76 = x75**2
    x77 = sigma_kin_p_4_1_c * x5 + sigma_kin_p_4_1_off + sigma_kin_p_4_1_s * x2
    x78 = x76 * x77
    x79 = x74 * x78
    x80 = x73 * x79
    x81 = sigma_kin_p_4_1_c * x14 - sigma_kin_p_4_1_s * x15
    x82 = sigma_kin_p_4_1_c * x15 - sigma_kin_p_4_1_s * x14
    x83 = x76 * x81 * x82
    x84 = x73 * x83
    x85 = sigma_pot_2_c * x22 + sigma_pot_2_off + sigma_pot_2_s * x25
    x86 = sigma_pot_3_c * x46 + sigma_pot_3_off + sigma_pot_3_s * x49
    x87 = sigma_pot_4_c * x67 + sigma_pot_4_off + sigma_pot_4_s * x70
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
    x109 = 1.0 * x101 * x108 * x86 * x87 * x94
    x110 = dq_i5 * dq_j5
    x111 = (
        sigma_kin_v_5_1 * x31
        + sigma_kin_v_5_2 * x32
        + sigma_kin_v_5_3 * x38
        + sigma_kin_v_5_4 * x60
        + sigma_kin_v_5_5 * x110
    )
    x112 = 2 * x111**2
    x113 = sigma_kin_p_5_3_c * x46 + sigma_kin_p_5_3_off + sigma_kin_p_5_3_s * x49
    x114 = x113**2
    x115 = sigma_kin_p_5_4_c * x67 + sigma_kin_p_5_4_off + sigma_kin_p_5_4_s * x70
    x116 = x115**2
    x117 = sigma_kin_p_5_5_c * x90 + sigma_kin_p_5_5_off + sigma_kin_p_5_5_s * x93
    x118 = x117**2
    x119 = x114 * x116 * x118
    x120 = sigma_kin_p_5_1_c * x2 + sigma_kin_p_5_1_s * x5
    x121 = sigma_kin_p_5_2_c * x22 + sigma_kin_p_5_2_off + sigma_kin_p_5_2_s * x25
    x122 = x121**2
    x123 = sigma_kin_p_5_1_c * x5 + sigma_kin_p_5_1_off + sigma_kin_p_5_1_s * x2
    x124 = x122 * x123
    x125 = x120 * x124
    x126 = x119 * x125
    x127 = sigma_kin_p_5_1_c * x14 - sigma_kin_p_5_1_s * x15
    x128 = sigma_kin_p_5_1_c * x15 - sigma_kin_p_5_1_s * x14
    x129 = x122 * x127 * x128
    x130 = x119 * x129
    x131 = 4 * dq_j1
    x132 = sigma_kin_p_7_1_c * x14 - sigma_kin_p_7_1_s * x15
    x133 = sigma_kin_p_7_1_c * x5 + sigma_kin_p_7_1_off + sigma_kin_p_7_1_s * x2
    x134 = sigma_kin_p_7_2_c * x22 + sigma_kin_p_7_2_off + sigma_kin_p_7_2_s * x25
    x135 = x134**2
    x136 = x133 * x135
    x137 = x132 * x136
    x138 = sigma_kin_p_7_3_c * x46 + sigma_kin_p_7_3_off + sigma_kin_p_7_3_s * x49
    x139 = x138**2
    x140 = sigma_kin_p_7_4_c * x67 + sigma_kin_p_7_4_off + sigma_kin_p_7_4_s * x70
    x141 = x140**2
    x142 = sigma_kin_p_7_5_c * x90 + sigma_kin_p_7_5_off + sigma_kin_p_7_5_s * x93
    x143 = x142**2
    x144 = sigma_kin_p_7_6_c * x97 + sigma_kin_p_7_6_off + sigma_kin_p_7_6_s * x100
    x145 = x144**2
    x146 = sigma_kin_p_7_7_c * x103
    x147 = sigma_kin_p_7_7_s * x106
    x148 = sigma_kin_p_7_7_off + x102 * x146 + x105 * x147
    x149 = x148**2
    x150 = x139 * x141 * x143 * x145 * x149
    x151 = sigma_kin_v_7_1 * x150
    x152 = x137 * x151
    x153 = dq_j7 * sigma_kin_v_7_7
    x154 = ddq_i7 * x153
    x155 = x152 * x154
    x156 = dq_i6 * dq_j6
    x157 = (
        sigma_kin_v_6_1 * x31
        + sigma_kin_v_6_2 * x32
        + sigma_kin_v_6_3 * x38
        + sigma_kin_v_6_4 * x60
        + sigma_kin_v_6_5 * x110
        + sigma_kin_v_6_6 * x156
    )
    x158 = 2 * x157**2
    x159 = sigma_kin_p_6_3_c * x46 + sigma_kin_p_6_3_off + sigma_kin_p_6_3_s * x49
    x160 = x159**2
    x161 = sigma_kin_p_6_4_c * x67 + sigma_kin_p_6_4_off + sigma_kin_p_6_4_s * x70
    x162 = x161**2
    x163 = sigma_kin_p_6_5_c * x90 + sigma_kin_p_6_5_off + sigma_kin_p_6_5_s * x93
    x164 = x163**2
    x165 = sigma_kin_p_6_6_c * x97 + sigma_kin_p_6_6_off + sigma_kin_p_6_6_s * x100
    x166 = x165**2
    x167 = x160 * x162 * x164 * x166
    x168 = sigma_kin_p_6_1_c * x2 + sigma_kin_p_6_1_s * x5
    x169 = sigma_kin_p_6_2_c * x22 + sigma_kin_p_6_2_off + sigma_kin_p_6_2_s * x25
    x170 = x169**2
    x171 = sigma_kin_p_6_1_c * x5 + sigma_kin_p_6_1_off + sigma_kin_p_6_1_s * x2
    x172 = x170 * x171
    x173 = x168 * x172
    x174 = x167 * x173
    x175 = sigma_kin_p_6_1_c * x14 - sigma_kin_p_6_1_s * x15
    x176 = sigma_kin_p_6_1_c * x15 - sigma_kin_p_6_1_s * x14
    x177 = x170 * x175 * x176
    x178 = x167 * x177
    x179 = sigma_kin_v_7_1 * x31
    x180 = sigma_kin_v_7_2 * x32
    x181 = sigma_kin_v_7_3 * x38
    x182 = sigma_kin_v_7_4 * x60
    x183 = sigma_kin_v_7_5 * x110
    x184 = sigma_kin_v_7_6 * x156
    x185 = dq_i7 * x153
    x186 = x180 + x181 + x182 + x183 + x184 + x185
    x187 = x179 + x186
    x188 = 2 * x187**2
    x189 = sigma_kin_p_7_1_c * x2 + sigma_kin_p_7_1_s * x5
    x190 = x136 * x189
    x191 = x150 * x190
    x192 = sigma_kin_p_7_1_c * x15 - sigma_kin_p_7_1_s * x14
    x193 = x150 * x192
    x194 = x132 * x135
    x195 = 8 * dq_j1
    x196 = sigma_kin_v_7_1 * x187
    x197 = x139 * x143
    x198 = x145 * x197
    x199 = x141 * x198
    x200 = x137 * x199
    x201 = -x102 * x147 + x105 * x146
    x202 = dq_i7 * x148
    x203 = x201 * x202
    x204 = x172 * x175
    x205 = sigma_kin_v_6_1 * x167
    x206 = x204 * x205
    x207 = sigma_kin_v_6_6 * x206 + sigma_kin_v_7_6 * x152
    x208 = ddq_i6 * dq_j6
    x209 = x207 * x208
    x210 = x96 * x98
    x211 = x95 * x99
    x212 = sigma_kin_p_6_6_c * x210 - sigma_kin_p_6_6_s * x211
    x213 = x164 * x165
    x214 = x212 * x213
    x215 = x162 * x204
    x216 = sigma_kin_v_6_1 * x157
    x217 = x160 * x216
    x218 = x215 * x217
    x219 = x139 * x149
    x220 = x196 * x219
    x221 = x141 * x220
    x222 = x137 * x221
    x223 = sigma_kin_p_7_6_c * x210 - sigma_kin_p_7_6_s * x211
    x224 = x143 * x144
    x225 = x223 * x224
    x226 = x214 * x218 + x222 * x225
    x227 = x124 * x127
    x228 = sigma_kin_v_5_1 * x119
    x229 = x227 * x228
    x230 = sigma_kin_v_5_5 * x229 + sigma_kin_v_6_5 * x206 + sigma_kin_v_7_5 * x152
    x231 = ddq_i5 * dq_j5
    x232 = x230 * x231
    x233 = x78 * x81
    x234 = sigma_kin_v_4_1 * x73
    x235 = x233 * x234
    x236 = sigma_kin_v_4_4 * x235 + sigma_kin_v_5_4 * x229 + sigma_kin_v_6_4 * x206 + sigma_kin_v_7_4 * x152
    x237 = ddq_i4 * dq_j4
    x238 = x236 * x237
    x239 = x114 * x117
    x240 = x89 * x91
    x241 = x88 * x92
    x242 = sigma_kin_p_5_5_c * x240 - sigma_kin_p_5_5_s * x241
    x243 = sigma_kin_v_5_1 * x111
    x244 = x116 * x243
    x245 = x242 * x244
    x246 = x239 * x245
    x247 = sigma_kin_p_6_5_c * x240 - sigma_kin_p_6_5_s * x241
    x248 = x163 * x166
    x249 = x247 * x248
    x250 = sigma_kin_p_7_5_c * x240 - sigma_kin_p_7_5_s * x241
    x251 = x142 * x250
    x252 = x145 * x251
    x253 = x218 * x249 + x222 * x252 + x227 * x246
    x254 = x53 * x56
    x255 = sigma_kin_v_3_1 * x42
    x256 = x254 * x255
    x257 = (
        sigma_kin_v_3_3 * x256
        + sigma_kin_v_4_3 * x235
        + sigma_kin_v_5_3 * x229
        + sigma_kin_v_6_3 * x206
        + sigma_kin_v_7_3 * x152
    )
    x258 = ddq_i3 * dq_j3
    x259 = x257 * x258
    x260 = sigma_kin_v_2_1 * sigma_kin_v_2_2
    x261 = x260 * x29
    x262 = (
        sigma_kin_v_3_2 * x256
        + sigma_kin_v_4_2 * x235
        + sigma_kin_v_5_2 * x229
        + sigma_kin_v_6_2 * x206
        + sigma_kin_v_7_2 * x152
        + x261 * x35
    )
    x263 = ddq_i2 * dq_j2
    x264 = x262 * x263
    x265 = x66 * x68
    x266 = x65 * x69
    x267 = sigma_kin_p_4_4_c * x265 - sigma_kin_p_4_4_s * x266
    x268 = x64 * x71
    x269 = x267 * x268
    x270 = sigma_kin_v_4_1 * x61
    x271 = x269 * x270
    x272 = sigma_kin_p_5_4_c * x265 - sigma_kin_p_5_4_s * x266
    x273 = x115 * x272
    x274 = x243 * x273
    x275 = x118 * x227
    x276 = x114 * x275
    x277 = x160 * x161
    x278 = sigma_kin_p_6_4_c * x265 - sigma_kin_p_6_4_s * x266
    x279 = x164 * x166
    x280 = x216 * x279
    x281 = x278 * x280
    x282 = x277 * x281
    x283 = x143 * x145 * x149
    x284 = x196 * x283
    x285 = x137 * x284
    x286 = sigma_kin_p_7_4_c * x265 - sigma_kin_p_7_4_s * x266
    x287 = x139 * x140
    x288 = x286 * x287
    x289 = x204 * x282 + x233 * x271 + x274 * x276 + x285 * x288
    x290 = sigma_kin_v_2_1**2
    x291 = x29 * x290
    x292 = sigma_kin_v_3_1**2
    x293 = x292 * x42
    x294 = x293 * x53
    x295 = sigma_kin_v_4_1**2
    x296 = x295 * x73
    x297 = x296 * x78
    x298 = sigma_kin_v_5_1**2
    x299 = x119 * x298
    x300 = x124 * x299
    x301 = sigma_kin_v_6_1**2
    x302 = x167 * x301
    x303 = x172 * x302
    x304 = sigma_kin_v_7_1**2
    x305 = x150 * x304
    x306 = x136 * x305
    x307 = x127 * x300 + x132 * x306 + x16 * x8 + x175 * x303 + x291 * x35 + x294 * x56 + x297 * x81
    x308 = x45 * x47
    x309 = x44 * x48
    x310 = sigma_kin_p_3_3_c * x308 - sigma_kin_p_3_3_s * x309
    x311 = x255 * x39
    x312 = x310 * x311
    x313 = x50 * x52
    x314 = x313 * x56
    x315 = sigma_kin_p_4_3_c * x308 - sigma_kin_p_4_3_s * x309
    x316 = x63 * x72
    x317 = x315 * x316
    x318 = x233 * x317
    x319 = sigma_kin_p_5_3_c * x308 - sigma_kin_p_5_3_s * x309
    x320 = x113 * x319
    x321 = x244 * x320
    x322 = sigma_kin_p_6_3_c * x308 - sigma_kin_p_6_3_s * x309
    x323 = x159 * x322
    x324 = x280 * x323
    x325 = sigma_kin_p_7_3_c * x308 - sigma_kin_p_7_3_s * x309
    x326 = x138 * x325
    x327 = x141 * x326
    x328 = x215 * x324 + x270 * x318 + x275 * x321 + x285 * x327 + x312 * x314
    x329 = x21 * x23
    x330 = x20 * x24
    x331 = sigma_kin_p_2_2_c * x329 - sigma_kin_p_2_2_s * x330
    x332 = sigma_kin_v_2_1 * x33
    x333 = x331 * x332
    x334 = x26 * x28
    x335 = x334 * x35
    x336 = sigma_kin_p_3_2_c * x329 - sigma_kin_p_3_2_s * x330
    x337 = sigma_kin_v_3_1 * x336
    x338 = x337 * x39
    x339 = x338 * x41
    x340 = sigma_kin_p_4_2_c * x329 - sigma_kin_p_4_2_s * x330
    x341 = x234 * x61
    x342 = x340 * x341
    x343 = x75 * x77
    x344 = x343 * x81
    x345 = sigma_kin_p_5_2_c * x329 - sigma_kin_p_5_2_s * x330
    x346 = x111 * x228
    x347 = x345 * x346
    x348 = x121 * x123
    x349 = x127 * x348
    x350 = sigma_kin_p_6_2_c * x329 - sigma_kin_p_6_2_s * x330
    x351 = x157 * x205
    x352 = x350 * x351
    x353 = x169 * x171
    x354 = x175 * x353
    x355 = x132 * x187
    x356 = sigma_kin_p_7_2_c * x329 - sigma_kin_p_7_2_s * x330
    x357 = x133 * x134
    x358 = x356 * x357
    x359 = x355 * x358
    x360 = x151 * x359 + x254 * x339 + x333 * x335 + x342 * x344 + x347 * x349 + x352 * x354
    x361 = x133**2
    x362 = ddq_i7 * x144
    x363 = x134 * x138
    x364 = x140 * x142
    x365 = x363 * x364
    x366 = x148 * x365
    x367 = x362 * x366
    x368 = dq_i7 * x144
    x369 = x364 * x368
    x370 = 2 * dq_i2
    x371 = x138 * x148
    x372 = x370 * x371
    x373 = x356 * x372
    x374 = x134 * x148
    x375 = x369 * x374
    x376 = 2 * dq_i3
    x377 = x325 * x376
    x378 = 2 * dq_i4
    x379 = x286 * x378
    x380 = x142 * x148
    x381 = x363 * x368
    x382 = x380 * x381
    x383 = 2 * dq_i5
    x384 = x250 * x383
    x385 = x140 * x148
    x386 = x381 * x385
    x387 = dq_i7 * x366
    x388 = 2 * dq_i6
    x389 = x223 * x388
    x390 = x144 * x365
    x391 = dq_i7**2
    x392 = x201 * x391
    x393 = 2 * x392
    x394 = x144 * x366
    x395 = sigma_kin_v_7_7 * x394
    x396 = 2 * ddq_j7
    x397 = x18 * x31
    x398 = x31 * x9
    x399 = x332 * x37
    x400 = x30 * x332
    x401 = x311 * x58
    x402 = x311 * x54
    x403 = x341 * x83
    x404 = x341 * x79
    x405 = x129 * x346
    x406 = x125 * x346
    x407 = x177 * x351
    x408 = x173 * x351
    x409 = x192 * x355
    x410 = x135 * x409
    x411 = x151 * x410
    x412 = x151 * x187
    x413 = x190 * x412
    x414 = x399 + x400 + x401 + x402 + x403 + x404 + x405 + x406 + x407 + x408 + x411 + x413
    x415 = sigma_kin_v_7_1 * sigma_kin_v_7_6
    x416 = x135 * x361
    x417 = x141 * x416
    x418 = x145 * x417
    x419 = x197 * x418
    x420 = x201 * x419
    x421 = x202 * x420
    x422 = x388 * x421
    x423 = x171**2
    x424 = x170 * x423
    x425 = x167 * x424
    x426 = sigma_kin_v_6_1 * x425
    x427 = x150 * x416
    x428 = sigma_kin_v_7_1 * x427
    x429 = sigma_kin_v_6_6 * x426 + sigma_kin_v_7_6 * x428
    x430 = x172 * x176
    x431 = sigma_kin_v_6_6 * x205
    x432 = x136 * x192
    x433 = sigma_kin_v_7_6 * x151
    x434 = x430 * x431 + x432 * x433
    x435 = dq_i1 * x388
    x436 = x169 * x423
    x437 = x350 * x436
    x438 = x134 * x361
    x439 = x356 * x438
    x440 = x431 * x437 + x433 * x439
    x441 = dq_i6 * x370
    x442 = x162 * x424
    x443 = x159 * x442
    x444 = x322 * x443
    x445 = x279 * x444
    x446 = sigma_kin_v_6_1 * sigma_kin_v_6_6
    x447 = x138 * x417
    x448 = x325 * x447
    x449 = x283 * x448
    x450 = x415 * x449 + x445 * x446
    x451 = dq_i6 * x376
    x452 = x277 * x424
    x453 = x278 * x452
    x454 = x279 * x453
    x455 = x287 * x416
    x456 = x286 * x455
    x457 = x283 * x456
    x458 = x415 * x457 + x446 * x454
    x459 = dq_i6 * x378
    x460 = x160 * x442
    x461 = x249 * x460
    x462 = x251 * x418
    x463 = x219 * x462
    x464 = x415 * x463 + x446 * x461
    x465 = dq_i6 * x383
    x466 = x214 * x460
    x467 = x224 * x417
    x468 = x223 * x467
    x469 = x219 * x468
    x470 = x415 * x469 + x446 * x466
    x471 = dq_i6**2
    x472 = 2 * x471
    x473 = 2 * ddq_j6
    x474 = sigma_kin_p_7_7_c * x102 * x106 - sigma_kin_p_7_7_s * x103 * x105
    x475 = x187 * x474
    x476 = 2 * dq_i1
    x477 = x192 * x476
    x478 = x394 * x477
    x479 = x475 * x478
    x480 = x474 * (2 * x179 + x186)
    x481 = x133 * x364
    x482 = x144 * x481
    x483 = x356 * x482
    x484 = x372 * x483
    x485 = x325 * x482
    x486 = x376 * x485
    x487 = x374 * x486
    x488 = x379 * x380
    x489 = x133 * x363
    x490 = x144 * x489
    x491 = x480 * x490
    x492 = x384 * x385
    x493 = x133 * x366
    x494 = x389 * x493
    x495 = ddq_i1 * dq_j1
    x496 = 2 * x495
    x497 = x144 * x493
    x498 = x474 * x497
    x499 = x133 * x390
    x500 = x201 * x474
    x501 = x499 * x500
    x502 = dq_i7 * x501
    x503 = sigma_kin_p_7_7_c * x107 + sigma_kin_p_7_7_s * x104
    x504 = x368 * x503
    x505 = x493 * x504
    x506 = sigma_kin_v_7_2 * x263
    x507 = -x498 * x506
    x508 = sigma_kin_v_7_3 * x258
    x509 = -x498 * x508
    x510 = sigma_kin_v_7_4 * x237
    x511 = -x498 * x510
    x512 = sigma_kin_v_7_5 * x231
    x513 = -x498 * x512
    x514 = sigma_kin_v_7_6 * x208
    x515 = -x498 * x514
    x516 = x507 + x509 + x511 + x513 + x515
    x517 = dq_i7 * x187
    x518 = x187 * x493
    x519 = x133 * x367
    x520 = x153 * x474
    x521 = x519 * x520
    x522 = x501 * x517 + x504 * x518 - x521
    x523 = 4 * x499
    x524 = dq_j7 * x523
    x525 = sigma_kin_v_7_1 * sigma_kin_v_7_5
    x526 = x383 * x421
    x527 = sigma_kin_v_6_1 * sigma_kin_v_6_5
    x528 = x466 * x527 + x469 * x525
    x529 = x123**2
    x530 = x122 * x529
    x531 = x119 * x530
    x532 = sigma_kin_v_5_1 * x531
    x533 = sigma_kin_v_5_5 * x532 + sigma_kin_v_6_5 * x426 + sigma_kin_v_7_5 * x428
    x534 = x124 * x128
    x535 = sigma_kin_v_5_5 * x228
    x536 = sigma_kin_v_6_5 * x205
    x537 = sigma_kin_v_7_5 * x151
    x538 = x430 * x536 + x432 * x537 + x534 * x535
    x539 = dq_i1 * x383
    x540 = x121 * x529
    x541 = x345 * x540
    x542 = x437 * x536 + x439 * x537 + x535 * x541
    x543 = dq_i5 * x370
    x544 = x118 * x530
    x545 = x113 * x544
    x546 = x319 * x545
    x547 = x116 * x546
    x548 = sigma_kin_v_5_1 * sigma_kin_v_5_5
    x549 = x445 * x527 + x449 * x525 + x547 * x548
    x550 = dq_i5 * x376
    x551 = x114 * x544
    x552 = x273 * x551
    x553 = x454 * x527 + x457 * x525 + x548 * x552
    x554 = dq_i5 * x378
    x555 = x239 * x530
    x556 = x242 * x555
    x557 = x116 * x556
    x558 = x461 * x527 + x463 * x525 + x548 * x557
    x559 = dq_i5**2
    x560 = 2 * x559
    x561 = 2 * ddq_j5
    x562 = sigma_kin_v_7_1 * sigma_kin_v_7_4
    x563 = x378 * x421
    x564 = sigma_kin_v_6_1 * sigma_kin_v_6_4
    x565 = x460 * x564
    x566 = x219 * x562
    x567 = x214 * x565 + x468 * x566
    x568 = sigma_kin_v_5_1 * sigma_kin_v_5_4
    x569 = x249 * x565 + x462 * x566 + x557 * x568
    x570 = x77**2
    x571 = x570 * x76
    x572 = x571 * x73
    x573 = sigma_kin_v_4_1 * x572
    x574 = sigma_kin_v_4_4 * x573 + sigma_kin_v_5_4 * x532 + sigma_kin_v_6_4 * x426 + sigma_kin_v_7_4 * x428
    x575 = x78 * x82
    x576 = sigma_kin_v_4_4 * x234
    x577 = sigma_kin_v_5_4 * x228
    x578 = sigma_kin_v_6_4 * x205
    x579 = sigma_kin_v_7_4 * x151
    x580 = x430 * x578 + x432 * x579 + x534 * x577 + x575 * x576
    x581 = dq_i1 * x378
    x582 = x570 * x75
    x583 = x340 * x582
    x584 = x437 * x578 + x439 * x579 + x541 * x577 + x576 * x583
    x585 = dq_i4 * x370
    x586 = sigma_kin_v_4_1 * x571
    x587 = x317 * x586
    x588 = sigma_kin_v_4_4 * x587 + x445 * x564 + x449 * x562 + x547 * x568
    x589 = dq_i4 * x376
    x590 = x269 * x586
    x591 = sigma_kin_v_4_4 * x590 + x454 * x564 + x457 * x562 + x552 * x568
    x592 = dq_i4**2
    x593 = 2 * x592
    x594 = 2 * ddq_j4
    x595 = sigma_kin_v_7_1 * sigma_kin_v_7_3
    x596 = x376 * x421
    x597 = sigma_kin_v_6_1 * sigma_kin_v_6_3
    x598 = x460 * x597
    x599 = x219 * x595
    x600 = x214 * x598 + x468 * x599
    x601 = sigma_kin_v_5_1 * sigma_kin_v_5_3
    x602 = x249 * x598 + x462 * x599 + x557 * x601
    x603 = sigma_kin_v_4_3 * x590 + x454 * x597 + x457 * x595 + x552 * x601
    x604 = x52**2
    x605 = x51 * x604
    x606 = x42 * x605
    x607 = sigma_kin_v_3_1 * x606
    x608 = (
        sigma_kin_v_3_3 * x607
        + sigma_kin_v_4_3 * x573
        + sigma_kin_v_5_3 * x532
        + sigma_kin_v_6_3 * x426
        + sigma_kin_v_7_3 * x428
    )
    x609 = x53 * x57
    x610 = sigma_kin_v_3_3 * x255
    x611 = sigma_kin_v_4_3 * x234
    x612 = sigma_kin_v_5_3 * x228
    x613 = sigma_kin_v_6_3 * x205
    x614 = sigma_kin_v_7_3 * x151
    x615 = x430 * x613 + x432 * x614 + x534 * x612 + x575 * x611 + x609 * x610
    x616 = dq_i1 * x376
    x617 = x41 * x605
    x618 = sigma_kin_v_3_3 * x617
    x619 = x337 * x618 + x437 * x613 + x439 * x614 + x541 * x612 + x583 * x611
    x620 = dq_i3 * x370
    x621 = x50 * x604
    x622 = x310 * x621
    x623 = sigma_kin_v_4_3 * x587 + x445 * x597 + x449 * x595 + x547 * x601 + x610 * x622
    x624 = dq_i3**2
    x625 = 2 * x624
    x626 = 2 * ddq_j3
    x627 = dq_i1 * x370
    x628 = x261 * x36
    x629 = sigma_kin_v_3_2 * x255
    x630 = x609 * x629
    x631 = sigma_kin_v_4_2 * x234
    x632 = x575 * x631
    x633 = sigma_kin_v_5_2 * x228
    x634 = x534 * x633
    x635 = sigma_kin_v_6_2 * x205
    x636 = x430 * x635
    x637 = sigma_kin_v_7_2 * x151
    x638 = x432 * x637
    x639 = sigma_kin_v_7_1 * sigma_kin_v_7_2
    x640 = x202 * x370
    x641 = x420 * x640
    x642 = sigma_kin_v_6_1 * sigma_kin_v_6_2
    x643 = x460 * x642
    x644 = x219 * x639
    x645 = x214 * x643 + x468 * x644
    x646 = sigma_kin_v_5_1 * sigma_kin_v_5_2
    x647 = x249 * x643 + x462 * x644 + x557 * x646
    x648 = sigma_kin_v_4_2 * x590 + x454 * x642 + x457 * x639 + x552 * x646
    x649 = x28**2
    x650 = x27 * x649
    x651 = (
        sigma_kin_v_3_2 * x607
        + sigma_kin_v_4_2 * x573
        + sigma_kin_v_5_2 * x532
        + sigma_kin_v_6_2 * x426
        + sigma_kin_v_7_2 * x428
        + x260 * x650
    )
    x652 = sigma_kin_v_4_2 * x317
    x653 = x445 * x642 + x449 * x639 + x547 * x646 + x586 * x652 + x622 * x629
    x654 = x628 + x630 + x632 + x634 + x636 + x638
    x655 = x26 * x649
    x656 = x260 * x655
    x657 = x331 * x656
    x658 = sigma_kin_v_3_2 * x617
    x659 = x337 * x658
    x660 = x583 * x631
    x661 = x541 * x633
    x662 = x437 * x635
    x663 = x439 * x637
    x664 = x657 + x659 + x660 + x661 + x662 + x663
    x665 = dq_i2**2
    x666 = 2 * x665
    x667 = 2 * ddq_j2
    x668 = x17 * x8
    x669 = x291 * x36
    x670 = x294 * x57
    x671 = x297 * x82
    x672 = x128 * x300
    x673 = x176 * x303
    x674 = x192 * x306
    x675 = x421 * x476
    x676 = x301 * x460
    x677 = x213 * x676
    x678 = x219 * x304
    x679 = x467 * x678
    x680 = x116 * x298
    x681 = x555 * x680
    x682 = x418 * x678
    x683 = x295 * x571
    x684 = x268 * x683
    x685 = x273 * x298
    x686 = x279 * x301
    x687 = x452 * x686
    x688 = x283 * x304
    x689 = x455 * x688
    x690 = x293 * x621
    x691 = x290 * x655
    x692 = x292 * x617
    x693 = x296 * x582
    x694 = x299 * x540
    x695 = x302 * x436
    x696 = x305 * x438
    x697 = sigma_kin_p_7_6_c * x211 - sigma_kin_p_7_6_s * x210
    x698 = x153 * x697
    x699 = x143 * x219 * x362 * x417 * x698
    x700 = sigma_kin_p_6_6_c * x211 - sigma_kin_p_6_6_s * x210
    x701 = x213 * x700
    x702 = x162 * x217
    x703 = x701 * x702
    x704 = x430 * x703
    x705 = x224 * x697
    x706 = x136 * x477
    x707 = x221 * x706
    x708 = x467 * x697
    x709 = x643 * x701 + x644 * x708
    x710 = x598 * x701 + x599 * x708
    x711 = x565 * x701 + x566 * x708
    x712 = x460 * x701
    x713 = x219 * x708
    x714 = x525 * x713 + x527 * x712
    x715 = x415 * x713 + x446 * x712
    x716 = x160 * x162
    x717 = x430 * x716
    x718 = x301 * x31
    x719 = x701 * x718
    x720 = x31 * x678
    x721 = x141 * x720
    x722 = x705 * x721
    x723 = x221 * x432
    x724 = x437 * x716
    x725 = x221 * x439
    x726 = x304 * x31
    x727 = x149 * x726
    x728 = x448 * x727
    x729 = x216 * x701
    x730 = x149 * x196
    x731 = x448 * x730
    x732 = x456 * x705
    x733 = x31 * x676
    x734 = x163 * x165
    x735 = x247 * x700 * x734
    x736 = x251 * x697
    x737 = x144 * x417
    x738 = x720 * x737
    x739 = x216 * x460
    x740 = x220 * x417
    x741 = x144 * x740
    x742 = sigma_kin_p_6_6_c * x100 + sigma_kin_p_6_6_s * x97
    x743 = x164 * x700
    x744 = x212 * x743
    x745 = sigma_kin_p_7_6_c * x100 + sigma_kin_p_7_6_s * x97
    x746 = x223 * x697
    x747 = x143 * x746
    x748 = x417 * x747
    x749 = x213 * x742
    x750 = x212 * x739
    x751 = x467 * x745
    x752 = 4 * dq_j6
    x753 = sigma_kin_p_5_5_c * x241 - sigma_kin_p_5_5_s * x240
    x754 = x239 * x753
    x755 = x244 * x754
    x756 = x534 * x755
    x757 = sigma_kin_p_7_5_c * x241 - sigma_kin_p_7_5_s * x240
    x758 = x142 * x757
    x759 = x219 * x418
    x760 = x758 * x759
    x761 = sigma_kin_v_7_1 * x154
    x762 = sigma_kin_p_6_5_c * x241 - sigma_kin_p_6_5_s * x240
    x763 = x248 * x762
    x764 = x702 * x763
    x765 = x430 * x764
    x766 = x145 * x758
    x767 = x460 * x763
    x768 = x415 * x760 + x446 * x767
    x769 = x555 * x753
    x770 = x116 * x769
    x771 = x418 * x758
    x772 = x643 * x763 + x644 * x771 + x646 * x770
    x773 = x598 * x763 + x599 * x771 + x601 * x770
    x774 = x565 * x763 + x566 * x771 + x568 * x770
    x775 = x525 * x760 + x527 * x767 + x548 * x770
    x776 = x734 * x762
    x777 = x212 * x776
    x778 = x223 * x758
    x779 = x31 * x680
    x780 = x754 * x779
    x781 = x718 * x763
    x782 = x721 * x766
    x783 = x530 * x753
    x784 = x117 * x783
    x785 = x320 * x779
    x786 = x216 * x763
    x787 = x31 * x685
    x788 = x416 * x757
    x789 = x286 * x788
    x790 = x145 * x364
    x791 = x720 * x790
    x792 = x220 * x790
    x793 = sigma_kin_p_5_5_c * x93 + sigma_kin_p_5_5_s * x90
    x794 = x114 * x783
    x795 = x242 * x794
    x796 = x248 * (sigma_kin_p_6_5_c * x93 + sigma_kin_p_6_5_s * x90)
    x797 = x555 * x793
    x798 = x166 * x247 * x762
    x799 = sigma_kin_p_7_5_c * x93 + sigma_kin_p_7_5_s * x90
    x800 = x142 * x799
    x801 = x31 * x682
    x802 = x250 * x757
    x803 = x196 * x759
    x804 = 4 * dq_j5
    x805 = sigma_kin_p_4_4_c * x266 - sigma_kin_p_4_4_s * x265
    x806 = x268 * x805
    x807 = x270 * x806
    x808 = x575 * x807
    x809 = x118 * x534
    x810 = sigma_kin_p_5_4_c * x266 - sigma_kin_p_5_4_s * x265
    x811 = x115 * x243
    x812 = x810 * x811
    x813 = x114 * x812
    x814 = x809 * x813
    x815 = sigma_kin_p_7_4_c * x266 - sigma_kin_p_7_4_s * x265
    x816 = x455 * x815
    x817 = x283 * x816
    x818 = x280 * x430
    x819 = sigma_kin_p_6_4_c * x266 - sigma_kin_p_6_4_s * x265
    x820 = x277 * x819
    x821 = x818 * x820
    x822 = x287 * x815
    x823 = x284 * x822
    x824 = x452 * x819
    x825 = x279 * x824
    x826 = x415 * x817 + x446 * x825
    x827 = x551 * x810
    x828 = x115 * x827
    x829 = x525 * x817 + x527 * x825 + x548 * x828
    x830 = x586 * x806
    x831 = sigma_kin_v_4_2 * x830 + x639 * x817 + x642 * x825 + x646 * x828
    x832 = sigma_kin_v_4_3 * x830 + x595 * x817 + x597 * x825 + x601 * x828
    x833 = sigma_kin_v_4_4 * x830 + x562 * x817 + x564 * x825 + x568 * x828
    x834 = x214 * x718
    x835 = x225 * x816
    x836 = x216 * x824
    x837 = x298 * x31
    x838 = x115 * x810
    x839 = x837 * x838
    x840 = x249 * x718
    x841 = x416 * x815
    x842 = x250 * x841
    x843 = x295 * x31
    x844 = x806 * x843
    x845 = x114 * x839
    x846 = x31 * x686
    x847 = x820 * x846
    x848 = x31 * x688
    x849 = x822 * x848
    x850 = x118 * x541
    x851 = x280 * x437
    x852 = x31 * x683
    x853 = x63 * x71
    x854 = x315 * x805 * x853
    x855 = x586 * x61
    x856 = x424 * x819
    x857 = x161 * x856
    x858 = x323 * x846
    x859 = x326 * x815
    x860 = x140 * x416
    x861 = x848 * x860
    x862 = x284 * x416
    x863 = x140 * x862
    x864 = sigma_kin_p_4_4_c * x70 + sigma_kin_p_4_4_s * x67
    x865 = x64 * x805
    x866 = x267 * x865
    x867 = x268 * x864
    x868 = x267 * x855
    x869 = x551 * (sigma_kin_p_5_4_c * x70 + sigma_kin_p_5_4_s * x67)
    x870 = x115 * x869
    x871 = x272 * x827
    x872 = sigma_kin_p_6_4_c * x70 + sigma_kin_p_6_4_s * x67
    x873 = x160 * x856
    x874 = x278 * x873
    x875 = sigma_kin_p_7_4_c * x70 + sigma_kin_p_7_4_s * x67
    x876 = x286 * x815
    x877 = x139 * x876
    x878 = x416 * x877
    x879 = x452 * x872
    x880 = x455 * x875
    x881 = 4 * dq_j4
    x882 = sigma_kin_p_3_3_c * x309 - sigma_kin_p_3_3_s * x308
    x883 = x313 * x57 * x882
    x884 = x311 * x883
    x885 = sigma_kin_p_4_3_c * x309 - sigma_kin_p_4_3_s * x308
    x886 = x316 * x885
    x887 = x270 * x886
    x888 = x575 * x887
    x889 = sigma_kin_p_5_3_c * x309 - sigma_kin_p_5_3_s * x308
    x890 = x113 * x889
    x891 = x244 * x890
    x892 = x809 * x891
    x893 = sigma_kin_p_7_3_c * x309 - sigma_kin_p_7_3_s * x308
    x894 = x447 * x893
    x895 = x283 * x894
    x896 = sigma_kin_p_6_3_c * x309 - sigma_kin_p_6_3_s * x308
    x897 = x159 * x896
    x898 = x162 * x897
    x899 = x818 * x898
    x900 = x141 * x284
    x901 = x138 * x893
    x902 = x443 * x896
    x903 = x279 * x902
    x904 = x415 * x895 + x446 * x903
    x905 = x545 * x889
    x906 = x116 * x905
    x907 = x525 * x895 + x527 * x903 + x548 * x906
    x908 = x586 * x886
    x909 = sigma_kin_v_4_4 * x908 + x562 * x895 + x564 * x903 + x568 * x906
    x910 = x621 * x882
    x911 = sigma_kin_v_4_2 * x908 + x629 * x910 + x639 * x895 + x642 * x903 + x646 * x906
    x912 = sigma_kin_v_4_3 * x908 + x595 * x895 + x597 * x903 + x601 * x906 + x610 * x910
    x913 = x149 * x894
    x914 = x225 * x913
    x915 = x216 * x902
    x916 = x730 * x894
    x917 = x117 * x530 * x890
    x918 = x242 * x779
    x919 = x252 * x913
    x920 = x853 * x885
    x921 = x267 * x920
    x922 = x161 * x424 * x897
    x923 = x278 * x846
    x924 = x286 * x901
    x925 = x293 * x31
    x926 = x843 * x886
    x927 = x779 * x890
    x928 = x846 * x898
    x929 = x432 * x901
    x930 = x141 * x848
    x931 = x292 * x31
    x932 = x336 * x41
    x933 = x931 * x932
    x934 = x356 * x361
    x935 = x893 * x934
    x936 = x363 * x930
    x937 = x363 * x900
    x938 = sigma_kin_p_3_3_c * x49 + sigma_kin_p_3_3_s * x46
    x939 = x604 * x882
    x940 = x310 * x939
    x941 = x621 * x938
    x942 = x316 * (sigma_kin_p_4_3_c * x49 + sigma_kin_p_4_3_s * x46)
    x943 = x315 * x72 * x885
    x944 = x545 * (sigma_kin_p_5_3_c * x49 + sigma_kin_p_5_3_s * x46)
    x945 = x319 * x544 * x889
    x946 = x443 * (sigma_kin_p_6_3_c * x49 + sigma_kin_p_6_3_s * x46)
    x947 = x322 * x442 * x896
    x948 = sigma_kin_p_7_3_c * x49 + sigma_kin_p_7_3_s * x46
    x949 = x447 * x948
    x950 = x325 * x417 * x893
    x951 = 4 * dq_j3
    x952 = sigma_kin_p_2_2_c * x330 - sigma_kin_p_2_2_s * x329
    x953 = x334 * x36 * x952
    x954 = x332 * x953
    x955 = x41 * x609
    x956 = sigma_kin_p_3_2_c * x330 - sigma_kin_p_3_2_s * x329
    x957 = sigma_kin_v_3_1 * x956
    x958 = x39 * x957
    x959 = x955 * x958
    x960 = sigma_kin_p_4_2_c * x330 - sigma_kin_p_4_2_s * x329
    x961 = x343 * x82 * x960
    x962 = x341 * x961
    x963 = sigma_kin_p_5_2_c * x330 - sigma_kin_p_5_2_s * x329
    x964 = x128 * x348 * x963
    x965 = x346 * x964
    x966 = sigma_kin_p_7_2_c * x330 - sigma_kin_p_7_2_s * x329
    x967 = x438 * x966
    x968 = x154 * x967
    x969 = sigma_kin_p_6_2_c * x330 - sigma_kin_p_6_2_s * x329
    x970 = x176 * x353 * x969
    x971 = x351 * x970
    x972 = x412 * x966
    x973 = x436 * x969
    x974 = x431 * x973 + x433 * x967
    x975 = x540 * x963
    x976 = x535 * x975 + x536 * x973 + x537 * x967
    x977 = x582 * x960
    x978 = x576 * x977 + x577 * x975 + x578 * x973 + x579 * x967
    x979 = x611 * x977 + x612 * x975 + x613 * x973 + x614 * x967 + x618 * x957
    x980 = x631 * x977 + x633 * x975 + x635 * x973 + x637 * x967 + x656 * x952 + x658 * x957
    x981 = x162 * x973
    x982 = x160 * x981
    x983 = x721 * x967
    x984 = x217 * x981
    x985 = x221 * x967
    x986 = x239 * x918
    x987 = x269 * x843
    x988 = x118 * x975
    x989 = x114 * x988
    x990 = x277 * x923
    x991 = x288 * x967
    x992 = x41 * x622
    x993 = x931 * x956
    x994 = x317 * x977
    x995 = x361 * x966
    x996 = x325 * x995
    x997 = x290 * x31
    x998 = x296 * x31
    x999 = x299 * x31
    x1000 = x302 * x31
    x1001 = x305 * x31
    x1002 = x192 * x357
    x1003 = x1002 * x966
    x1004 = sigma_kin_p_2_2_c * x25 + sigma_kin_p_2_2_s * x22
    x1005 = x649 * x952
    x1006 = x1005 * x331
    x1007 = x1004 * x655
    x1008 = sigma_kin_p_3_2_c * x25 + sigma_kin_p_3_2_s * x22
    x1009 = x605 * x956
    x1010 = x1009 * x336
    x1011 = x1008 * x39
    x1012 = sigma_kin_p_4_2_c * x25 + sigma_kin_p_4_2_s * x22
    x1013 = x570 * x960
    x1014 = x1013 * x340
    x1015 = x1012 * x582
    x1016 = sigma_kin_p_5_2_c * x25 + sigma_kin_p_5_2_s * x22
    x1017 = x529 * x963
    x1018 = x1017 * x345
    x1019 = sigma_kin_p_6_2_c * x25 + sigma_kin_p_6_2_s * x22
    x1020 = x1016 * x540
    x1021 = x423 * x969
    x1022 = x1021 * x350
    x1023 = sigma_kin_p_7_2_c * x25 + sigma_kin_p_7_2_s * x22
    x1024 = x934 * x966
    x1025 = x1019 * x436
    x1026 = x1023 * x438
    x1027 = 4 * dq_j2
    x1028 = dq_j1 * x11
    x1029 = 2 * x201
    x1030 = x160 * x215
    x1031 = x137 * x721
    x1032 = x137 * x848
    x1033 = x310 * x314
    x1034 = x331 * x335
    x1035 = x340 * x344
    x1036 = x345 * x349
    x1037 = x350 * x354
    x1038 = x132 * x358
    x1039 = x192 * x194
    x1040 = x62 * x73
    x1041 = sigma_pot_1_c * x5 + sigma_pot_1_off + sigma_pot_1_s * x2
    x1042 = x112 * x119
    x1043 = sigma_kin_v_7_2 * x150
    x1044 = x1043 * x968
    x1045 = x158 * x167
    x1046 = x150 * x188
    x1047 = 8 * dq_j2
    x1048 = x201 * x517
    x1049 = sigma_kin_v_6_2 * sigma_kin_v_6_6
    x1050 = x1049 * x167
    x1051 = sigma_kin_v_7_2 * sigma_kin_v_7_6
    x1052 = x1051 * x150
    x1053 = x1050 * x973 + x1052 * x967
    x1054 = x1053 * x208
    x1055 = sigma_kin_v_6_2 * x157
    x1056 = x1055 * x160
    x1057 = x1056 * x981
    x1058 = sigma_kin_v_7_2 * x187
    x1059 = x1058 * x219
    x1060 = x1059 * x141
    x1061 = x1060 * x967
    x1062 = x1057 * x214 + x1061 * x225
    x1063 = sigma_kin_v_5_2 * sigma_kin_v_5_5
    x1064 = x1063 * x119
    x1065 = sigma_kin_v_6_2 * sigma_kin_v_6_5
    x1066 = x1065 * x167
    x1067 = sigma_kin_v_7_2 * sigma_kin_v_7_5
    x1068 = x1067 * x150
    x1069 = x1064 * x975 + x1066 * x973 + x1068 * x967
    x1070 = x1069 * x231
    x1071 = sigma_kin_v_4_2 * x73
    x1072 = sigma_kin_v_4_4 * x1071
    x1073 = sigma_kin_v_5_2 * sigma_kin_v_5_4
    x1074 = x1073 * x119
    x1075 = sigma_kin_v_6_2 * sigma_kin_v_6_4
    x1076 = x1075 * x167
    x1077 = sigma_kin_v_7_2 * sigma_kin_v_7_4
    x1078 = x1077 * x150
    x1079 = x1072 * x977 + x1074 * x975 + x1076 * x973 + x1078 * x967
    x1080 = x1079 * x237
    x1081 = sigma_kin_v_5_2 * x111
    x1082 = x1081 * x116
    x1083 = x1082 * x242
    x1084 = x1083 * x239
    x1085 = x1057 * x249 + x1061 * x252 + x1084 * x975
    x1086 = sigma_kin_v_3_2 * sigma_kin_v_3_3
    x1087 = x1086 * x617
    x1088 = sigma_kin_v_4_3 * x1071
    x1089 = sigma_kin_v_5_2 * sigma_kin_v_5_3
    x1090 = x1089 * x119
    x1091 = sigma_kin_v_6_2 * sigma_kin_v_6_3
    x1092 = x1091 * x167
    x1093 = sigma_kin_v_7_2 * sigma_kin_v_7_3
    x1094 = x1093 * x150
    x1095 = x1087 * x956 + x1088 * x977 + x1090 * x975 + x1092 * x973 + x1094 * x967
    x1096 = x1095 * x258
    x1097 = ddq_i1 * x131
    x1098 = sigma_kin_v_2_2**2
    x1099 = x1098 * x655
    x1100 = sigma_kin_v_3_2**2
    x1101 = x1100 * x617
    x1102 = sigma_kin_v_4_2**2
    x1103 = x1102 * x73
    x1104 = sigma_kin_v_5_2**2
    x1105 = x1104 * x119
    x1106 = sigma_kin_v_6_2**2
    x1107 = x1106 * x167
    x1108 = sigma_kin_v_7_2**2
    x1109 = x1108 * x150
    x1110 = x1099 * x952 + x1101 * x956 + x1103 * x977 + x1105 * x975 + x1107 * x973 + x1109 * x967
    x1111 = sigma_kin_v_4_2 * x269
    x1112 = x61 * x977
    x1113 = x1081 * x273
    x1114 = x1055 * x279
    x1115 = x1114 * x278
    x1116 = x1115 * x277
    x1117 = x1058 * x283
    x1118 = x1111 * x1112 + x1113 * x989 + x1116 * x973 + x1117 * x991
    x1119 = sigma_kin_v_3_2 * x39
    x1120 = x1119 * x956
    x1121 = x1082 * x320
    x1122 = x1114 * x323
    x1123 = x1117 * x141
    x1124 = x1123 * x363
    x1125 = x1112 * x652 + x1120 * x992 + x1121 * x988 + x1122 * x981 + x1124 * x996
    x1126 = sigma_kin_v_2_2 * x33
    x1127 = x1071 * x61
    x1128 = x1081 * x119
    x1129 = x1055 * x167
    x1130 = x193 * x357 * x966
    x1131 = x1058 * x1130 + x1120 * x955 + x1126 * x953 + x1127 * x961 + x1128 * x964 + x1129 * x970
    x1132 = x362 * x481
    x1133 = x133 * x368
    x1134 = x1133 * x371
    x1135 = x142 * x286
    x1136 = x1135 * x378
    x1137 = x140 * x250
    x1138 = x1137 * x383
    x1139 = dq_i7 * x389
    x1140 = x1139 * x481
    x1141 = x144 * x393
    x1142 = x1141 * x481
    x1143 = sigma_kin_v_7_7 * x482
    x1144 = x1143 * x371
    x1145 = x1007 * x1126
    x1146 = x1006 * x1126
    x1147 = x1011 * x658
    x1148 = x1010 * x1119
    x1149 = x1015 * x1127
    x1150 = x1014 * x1127
    x1151 = x1020 * x1128
    x1152 = x1018 * x1128
    x1153 = x1025 * x1129
    x1154 = x1022 * x1129
    x1155 = x1043 * x187
    x1156 = x1026 * x1155
    x1157 = x1024 * x1155
    x1158 = x1145 + x1146 + x1147 + x1148 + x1149 + x1150 + x1151 + x1152 + x1153 + x1154 + x1156 + x1157
    x1159 = sigma_kin_v_6_2 * x425
    x1160 = sigma_kin_v_7_2 * x427
    x1161 = sigma_kin_v_6_6 * x1159 + sigma_kin_v_7_6 * x1160
    x1162 = x1050 * x430 + x1052 * x432
    x1163 = x1050 * x437 + x1052 * x439
    x1164 = x1049 * x445 + x1051 * x449
    x1165 = x1049 * x454 + x1051 * x457
    x1166 = x1049 * x461 + x1051 * x463
    x1167 = x1049 * x466 + x1051 * x469
    x1168 = x187 * x483
    x1169 = x1168 * x372
    x1170 = x1169 * x474
    x1171 = x179 + x182 + x183 + x184 + x185
    x1172 = x474 * (x1171 + 2 * x180 + x181)
    x1173 = x1172 * x490
    x1174 = 2 * x263
    x1175 = sigma_kin_v_7_1 * x495
    x1176 = -x1175 * x498
    x1177 = x1176 + x511 + x513 + x515 + x522
    x1178 = x1065 * x466 + x1067 * x469
    x1179 = sigma_kin_v_5_2 * x531
    x1180 = sigma_kin_v_5_5 * x1179 + sigma_kin_v_6_5 * x1159 + sigma_kin_v_7_5 * x1160
    x1181 = x1064 * x534 + x1066 * x430 + x1068 * x432
    x1182 = x1064 * x541 + x1066 * x437 + x1068 * x439
    x1183 = x1063 * x547 + x1065 * x445 + x1067 * x449
    x1184 = x1063 * x552 + x1065 * x454 + x1067 * x457
    x1185 = x1063 * x557 + x1065 * x461 + x1067 * x463
    x1186 = x1075 * x466 + x1077 * x469
    x1187 = x1073 * x557 + x1075 * x461 + x1077 * x463
    x1188 = sigma_kin_v_4_2 * x572
    x1189 = sigma_kin_v_4_4 * x1188 + sigma_kin_v_5_4 * x1179 + sigma_kin_v_6_4 * x1159 + sigma_kin_v_7_4 * x1160
    x1190 = x1072 * x575 + x1074 * x534 + x1076 * x430 + x1078 * x432
    x1191 = x1071 * x583
    x1192 = sigma_kin_v_4_4 * x1191 + x1074 * x541 + x1076 * x437 + x1078 * x439
    x1193 = sigma_kin_v_4_4 * x571
    x1194 = x1073 * x547 + x1075 * x445 + x1077 * x449 + x1193 * x652
    x1195 = x1073 * x552 + x1075 * x454 + x1077 * x457 + x1111 * x1193
    x1196 = x1091 * x466 + x1093 * x469
    x1197 = x1089 * x557 + x1091 * x461 + x1093 * x463
    x1198 = sigma_kin_v_4_3 * x571
    x1199 = x1089 * x552 + x1091 * x454 + x1093 * x457 + x1111 * x1198
    x1200 = (
        sigma_kin_v_4_3 * x1188
        + sigma_kin_v_5_3 * x1179
        + sigma_kin_v_6_3 * x1159
        + sigma_kin_v_7_3 * x1160
        + x1086 * x606
    )
    x1201 = x1086 * x42
    x1202 = sigma_kin_v_4_3 * x575
    x1203 = x1071 * x1202 + x1090 * x534 + x1092 * x430 + x1094 * x432 + x1201 * x609
    x1204 = sigma_kin_v_4_3 * x1191 + x1087 * x336 + x1090 * x541 + x1092 * x437 + x1094 * x439
    x1205 = x1089 * x547 + x1091 * x445 + x1093 * x449 + x1198 * x652 + x1201 * x622
    x1206 = 2 * ddq_j1
    x1207 = x1099 * x331
    x1208 = x1101 * x336
    x1209 = x1103 * x583
    x1210 = x1105 * x541
    x1211 = x1107 * x437
    x1212 = x1109 * x439
    x1213 = x1106 * x460
    x1214 = x1213 * x213
    x1215 = x1108 * x219
    x1216 = x1215 * x467
    x1217 = x1104 * x116
    x1218 = x1217 * x555
    x1219 = x1215 * x418
    x1220 = x1102 * x571
    x1221 = x1220 * x268
    x1222 = x1106 * x279
    x1223 = x1222 * x452
    x1224 = x1108 * x283
    x1225 = x1224 * x455
    x1226 = x1100 * x42
    x1227 = x1226 * x621
    x1228 = x1098 * x29
    x1229 = x1103 * x78
    x1230 = x1105 * x124
    x1231 = x1107 * x172
    x1232 = x1109 * x136
    x1233 = x1056 * x162
    x1234 = x1233 * x701
    x1235 = x1234 * x437
    x1236 = x1060 * x705
    x1237 = x1236 * x439
    x1238 = x1091 * x712 + x1093 * x713
    x1239 = x1075 * x712 + x1077 * x713
    x1240 = x1065 * x712 + x1067 * x713
    x1241 = x1049 * x712 + x1051 * x713
    x1242 = x1106 * x32
    x1243 = x1242 * x701
    x1244 = x1215 * x32
    x1245 = x1244 * x141
    x1246 = x1245 * x705
    x1247 = x1108 * x32
    x1248 = x1247 * x149
    x1249 = x1248 * x448
    x1250 = x1055 * x701
    x1251 = x1058 * x149
    x1252 = x1251 * x448
    x1253 = x1213 * x32
    x1254 = x1244 * x737
    x1255 = x1055 * x460
    x1256 = x1059 * x737
    x1257 = x1082 * x754
    x1258 = x1257 * x541
    x1259 = sigma_kin_v_7_2 * x154
    x1260 = x1233 * x763
    x1261 = x1260 * x437
    x1262 = x1060 * x766
    x1263 = x1262 * x439
    x1264 = x1049 * x767 + x1051 * x760
    x1265 = x1089 * x770 + x1091 * x767 + x1093 * x760
    x1266 = x1073 * x770 + x1075 * x767 + x1077 * x760
    x1267 = x1063 * x770 + x1065 * x767 + x1067 * x760
    x1268 = x1217 * x32
    x1269 = x1268 * x754
    x1270 = x1242 * x763
    x1271 = x1245 * x766
    x1272 = x1268 * x320
    x1273 = x1055 * x763
    x1274 = x1104 * x32
    x1275 = x1274 * x273
    x1276 = x1244 * x790
    x1277 = x1059 * x790
    x1278 = x1219 * x32
    x1279 = x1058 * x759
    x1280 = sigma_kin_v_4_2 * x61
    x1281 = x1280 * x806
    x1282 = x1281 * x583
    x1283 = x1081 * x838
    x1284 = x114 * x1283
    x1285 = x1284 * x850
    x1286 = x1114 * x820
    x1287 = x1286 * x437
    x1288 = x1117 * x822
    x1289 = x1288 * x439
    x1290 = x1049 * x825 + x1051 * x817
    x1291 = x1063 * x828 + x1065 * x825 + x1067 * x817
    x1292 = sigma_kin_v_4_2 * x806
    x1293 = x1089 * x828 + x1091 * x825 + x1093 * x817 + x1198 * x1292
    x1294 = x1073 * x828 + x1075 * x825 + x1077 * x817 + x1193 * x1292
    x1295 = x1242 * x214
    x1296 = x1055 * x824
    x1297 = x1274 * x838
    x1298 = x1242 * x249
    x1299 = x1102 * x32
    x1300 = x1299 * x806
    x1301 = x114 * x1297
    x1302 = x1222 * x32
    x1303 = x1302 * x820
    x1304 = x1224 * x32
    x1305 = x1304 * x822
    x1306 = x1220 * x32
    x1307 = x1280 * x571
    x1308 = x1302 * x323
    x1309 = x1304 * x860
    x1310 = x1117 * x860
    x1311 = x1119 * x932
    x1312 = x1311 * x910
    x1313 = x1280 * x886
    x1314 = x1313 * x583
    x1315 = x1082 * x890
    x1316 = x1315 * x850
    x1317 = x1114 * x898
    x1318 = x1317 * x437
    x1319 = x1124 * x935
    x1320 = x1049 * x903 + x1051 * x895
    x1321 = x1063 * x906 + x1065 * x903 + x1067 * x895
    x1322 = sigma_kin_v_4_2 * x886
    x1323 = x1073 * x906 + x1075 * x903 + x1077 * x895 + x1193 * x1322
    x1324 = x1089 * x906 + x1091 * x903 + x1093 * x895 + x1198 * x1322 + x1201 * x910
    x1325 = x1055 * x902
    x1326 = x1251 * x894
    x1327 = x1268 * x242
    x1328 = x1302 * x278
    x1329 = x1226 * x32
    x1330 = x1119 * x42
    x1331 = x1299 * x886
    x1332 = x1268 * x890
    x1333 = x1302 * x898
    x1334 = x1304 * x141
    x1335 = x1100 * x32
    x1336 = x1335 * x932
    x1337 = x1334 * x363
    x1338 = x1034 * x1126
    x1339 = x1311 * x254
    x1340 = x1035 * x1127
    x1341 = x1036 * x1128
    x1342 = x137 * x154
    x1343 = x1037 * x1129
    x1344 = x1043 * x359
    x1345 = x1050 * x204 + x1052 * x137
    x1346 = x1064 * x227 + x1066 * x204 + x1068 * x137
    x1347 = x1072 * x233 + x1074 * x227 + x1076 * x204 + x1078 * x137
    x1348 = x1088 * x233 + x1090 * x227 + x1092 * x204 + x1094 * x137 + x1201 * x254
    x1349 = x1245 * x137
    x1350 = x1056 * x215
    x1351 = x1060 * x137
    x1352 = x1327 * x239
    x1353 = x1030 * x249
    x1354 = x1299 * x269
    x1355 = x233 * x61
    x1356 = x1328 * x277
    x1357 = x1304 * x137
    x1358 = x1117 * x137
    x1359 = x1098 * x32
    x1360 = x1103 * x32
    x1361 = x1105 * x32
    x1362 = x1107 * x32
    x1363 = x1109 * x32
    x1364 = x1245 * x967
    x1365 = x1335 * x956
    x1366 = x40 * x42
    x1367 = x571 * x62
    x1368 = 1.0 * x101 * x1041 * x108 * x85 * x94
    x1369 = x112 * x116
    x1370 = sigma_kin_v_7_3 * x154
    x1371 = x1370 * x895
    x1372 = x158 * x279
    x1373 = x188 * x283
    x1374 = 8 * dq_j3
    x1375 = sigma_kin_v_6_3 * sigma_kin_v_6_6
    x1376 = sigma_kin_v_7_3 * sigma_kin_v_7_6
    x1377 = x1375 * x903 + x1376 * x895
    x1378 = x1377 * x208
    x1379 = sigma_kin_v_6_3 * x157
    x1380 = x1379 * x902
    x1381 = sigma_kin_v_7_3 * x187
    x1382 = x1381 * x149
    x1383 = x1382 * x894
    x1384 = x1380 * x214 + x1383 * x225
    x1385 = sigma_kin_v_5_3 * sigma_kin_v_5_5
    x1386 = sigma_kin_v_6_3 * sigma_kin_v_6_5
    x1387 = sigma_kin_v_7_3 * sigma_kin_v_7_5
    x1388 = x1385 * x906 + x1386 * x903 + x1387 * x895
    x1389 = x1388 * x231
    x1390 = sigma_kin_v_4_3 * sigma_kin_v_4_4
    x1391 = x1390 * x571
    x1392 = sigma_kin_v_5_3 * sigma_kin_v_5_4
    x1393 = sigma_kin_v_6_3 * sigma_kin_v_6_4
    x1394 = sigma_kin_v_7_3 * sigma_kin_v_7_4
    x1395 = x1391 * x886 + x1392 * x906 + x1393 * x903 + x1394 * x895
    x1396 = x1395 * x237
    x1397 = sigma_kin_v_5_3 * x111
    x1398 = x116 * x1397
    x1399 = x1398 * x242
    x1400 = x1380 * x249 + x1383 * x252 + x1399 * x917
    x1401 = x1324 * x263
    x1402 = sigma_kin_v_3_3**2
    x1403 = x1402 * x42
    x1404 = sigma_kin_v_4_3**2
    x1405 = x1404 * x571
    x1406 = sigma_kin_v_5_3**2
    x1407 = x116 * x1406
    x1408 = sigma_kin_v_6_3**2
    x1409 = x1408 * x279
    x1410 = sigma_kin_v_7_3**2
    x1411 = x1410 * x283
    x1412 = x1403 * x910 + x1405 * x886 + x1407 * x905 + x1409 * x902 + x1411 * x894
    x1413 = x1198 * x61
    x1414 = x1397 * x273
    x1415 = x1379 * x279
    x1416 = x1415 * x278
    x1417 = x1381 * x283
    x1418 = x1417 * x860
    x1419 = x1413 * x921 + x1414 * x905 + x1416 * x922 + x1418 * x924
    x1420 = sigma_kin_v_3_3 * x39
    x1421 = x1420 * x42
    x1422 = x1202 * x61
    x1423 = x1398 * x890
    x1424 = x1415 * x898
    x1425 = x141 * x1417
    x1426 = x1421 * x883 + x1422 * x886 + x1423 * x809 + x1424 * x430 + x1425 * x929
    x1427 = x1420 * x932
    x1428 = sigma_kin_v_4_3 * x61
    x1429 = x1428 * x583
    x1430 = x1425 * x363
    x1431 = x1423 * x850 + x1424 * x437 + x1427 * x910 + x1429 * x886 + x1430 * x935
    x1432 = x1133 * x374
    x1433 = x1143 * x374
    x1434 = x1421 * x941
    x1435 = x1421 * x940
    x1436 = x1413 * x942
    x1437 = x1413 * x943
    x1438 = x1398 * x944
    x1439 = x1398 * x945
    x1440 = x1415 * x946
    x1441 = x1415 * x947
    x1442 = x1417 * x949
    x1443 = x1417 * x950
    x1444 = x1434 + x1435 + x1436 + x1437 + x1438 + x1439 + x1440 + x1441 + x1442 + x1443
    x1445 = sigma_kin_v_6_3 * x425
    x1446 = sigma_kin_v_7_3 * x427
    x1447 = sigma_kin_v_6_6 * x1445 + sigma_kin_v_7_6 * x1446
    x1448 = x1375 * x167
    x1449 = x1376 * x150
    x1450 = x1448 * x430 + x1449 * x432
    x1451 = x1448 * x437 + x1449 * x439
    x1452 = x1375 * x445 + x1376 * x449
    x1453 = x1375 * x454 + x1376 * x457
    x1454 = x1375 * x461 + x1376 * x463
    x1455 = x1375 * x466 + x1376 * x469
    x1456 = x187 * x374
    x1457 = x1456 * x486
    x1458 = x1457 * x474
    x1459 = x474 * (x1171 + x180 + 2 * x181)
    x1460 = x1459 * x490
    x1461 = 2 * x258
    x1462 = x1386 * x466 + x1387 * x469
    x1463 = sigma_kin_v_5_3 * x531
    x1464 = sigma_kin_v_5_5 * x1463 + sigma_kin_v_6_5 * x1445 + sigma_kin_v_7_5 * x1446
    x1465 = x119 * x1385
    x1466 = x1386 * x167
    x1467 = x1387 * x150
    x1468 = x1465 * x534 + x1466 * x430 + x1467 * x432
    x1469 = x1465 * x541 + x1466 * x437 + x1467 * x439
    x1470 = x1385 * x547 + x1386 * x445 + x1387 * x449
    x1471 = x1385 * x552 + x1386 * x454 + x1387 * x457
    x1472 = x1385 * x557 + x1386 * x461 + x1387 * x463
    x1473 = x1393 * x466 + x1394 * x469
    x1474 = x1392 * x557 + x1393 * x461 + x1394 * x463
    x1475 = sigma_kin_v_5_4 * x1463 + sigma_kin_v_6_4 * x1445 + sigma_kin_v_7_4 * x1446 + x1390 * x572
    x1476 = x1390 * x73
    x1477 = x119 * x1392
    x1478 = x1393 * x167
    x1479 = x1394 * x150
    x1480 = x1476 * x575 + x1477 * x534 + x1478 * x430 + x1479 * x432
    x1481 = x1476 * x583 + x1477 * x541 + x1478 * x437 + x1479 * x439
    x1482 = x1391 * x317 + x1392 * x547 + x1393 * x445 + x1394 * x449
    x1483 = x1391 * x269 + x1392 * x552 + x1393 * x454 + x1394 * x457
    x1484 = x1408 * x460
    x1485 = x1484 * x213
    x1486 = x1410 * x219
    x1487 = x1486 * x467
    x1488 = x1407 * x555
    x1489 = x1486 * x418
    x1490 = x1405 * x268
    x1491 = x1409 * x452
    x1492 = x1411 * x455
    x1493 = x1404 * x73
    x1494 = x119 * x1406
    x1495 = x1408 * x167
    x1496 = x1410 * x150
    x1497 = x1402 * x617
    x1498 = x1379 * x701
    x1499 = x1498 * x444
    x1500 = x1382 * x448
    x1501 = x1500 * x705
    x1502 = x1393 * x712 + x1394 * x713
    x1503 = x1386 * x712 + x1387 * x713
    x1504 = x1375 * x712 + x1376 * x713
    x1505 = x1408 * x38
    x1506 = x1505 * x701
    x1507 = x1486 * x38
    x1508 = x141 * x1507
    x1509 = x1508 * x705
    x1510 = x1379 * x160
    x1511 = x1510 * x162
    x1512 = x1511 * x701
    x1513 = x1381 * x219
    x1514 = x141 * x1513
    x1515 = x1514 * x705
    x1516 = x1410 * x38
    x1517 = x149 * x1516
    x1518 = x1517 * x448
    x1519 = x1484 * x38
    x1520 = x1507 * x737
    x1521 = x1379 * x460
    x1522 = x1513 * x737
    x1523 = x1398 * x320
    x1524 = x1523 * x784
    x1525 = x1379 * x763
    x1526 = x1525 * x444
    x1527 = x1500 * x766
    x1528 = x1375 * x767 + x1376 * x760
    x1529 = x1392 * x770 + x1393 * x767 + x1394 * x760
    x1530 = x1385 * x770 + x1386 * x767 + x1387 * x760
    x1531 = x1407 * x38
    x1532 = x1531 * x754
    x1533 = x1505 * x763
    x1534 = x1398 * x754
    x1535 = x1508 * x766
    x1536 = x1511 * x763
    x1537 = x1514 * x766
    x1538 = x1531 * x320
    x1539 = x1406 * x38
    x1540 = x1539 * x273
    x1541 = x1507 * x790
    x1542 = x1513 * x790
    x1543 = x1489 * x38
    x1544 = x1381 * x759
    x1545 = x1413 * x854
    x1546 = x1397 * x838
    x1547 = x1546 * x546
    x1548 = x1415 * x323
    x1549 = x1548 * x857
    x1550 = x138 * x377
    x1551 = x1375 * x825 + x1376 * x817
    x1552 = x1385 * x828 + x1386 * x825 + x1387 * x817
    x1553 = x1391 * x806 + x1392 * x828 + x1393 * x825 + x1394 * x817
    x1554 = x1505 * x214
    x1555 = x1379 * x824
    x1556 = x1539 * x838
    x1557 = x1505 * x249
    x1558 = x1404 * x38
    x1559 = x1558 * x806
    x1560 = x114 * x1556
    x1561 = x1409 * x38
    x1562 = x1561 * x820
    x1563 = x114 * x1546
    x1564 = x1411 * x38
    x1565 = x1564 * x822
    x1566 = x1415 * x820
    x1567 = x1417 * x822
    x1568 = x1405 * x38
    x1569 = x1561 * x323
    x1570 = x1564 * x860
    x1571 = x1033 * x1421
    x1572 = x1428 * x318
    x1573 = x1523 * x275
    x1574 = sigma_kin_v_7_3 * x150
    x1575 = x1548 * x215
    x1576 = x137 * x1417
    x1577 = x137 * x1449 + x1448 * x204
    x1578 = x137 * x1467 + x1465 * x227 + x1466 * x204
    x1579 = x137 * x1479 + x1476 * x233 + x1477 * x227 + x1478 * x204
    x1580 = x137 * x1508
    x1581 = x1510 * x215
    x1582 = x137 * x1514
    x1583 = x1531 * x242
    x1584 = x1583 * x239
    x1585 = x1399 * x239
    x1586 = x1558 * x269
    x1587 = x1428 * x269
    x1588 = x1561 * x278
    x1589 = x1588 * x277
    x1590 = x137 * x1564
    x1591 = x1416 * x277
    x1592 = x1402 * x38
    x1593 = x1592 * x932
    x1594 = x1493 * x38
    x1595 = x1428 * x73
    x1596 = x1494 * x38
    x1597 = x1495 * x38
    x1598 = x119 * x1397
    x1599 = x1496 * x38
    x1600 = x1379 * x167
    x1601 = x1403 * x38
    x1602 = x1420 * x956
    x1603 = x1602 * x992
    x1604 = x1428 * x994
    x1605 = x1523 * x988
    x1606 = x1548 * x981
    x1607 = x1448 * x973 + x1449 * x967
    x1608 = x1465 * x975 + x1466 * x973 + x1467 * x967
    x1609 = x1476 * x977 + x1477 * x975 + x1478 * x973 + x1479 * x967
    x1610 = x1508 * x967
    x1611 = x1510 * x981
    x1612 = x1514 * x967
    x1613 = x1592 * x956
    x1614 = x141 * x1564
    x1615 = x1614 * x363
    x1616 = x1574 * x187
    x1617 = x1558 * x886
    x1618 = x1531 * x890
    x1619 = x1561 * x898
    x1620 = sigma_kin_v_7_4 * x154
    x1621 = x1620 * x817
    x1622 = 8 * dq_j4
    x1623 = sigma_kin_v_6_4 * sigma_kin_v_6_6
    x1624 = sigma_kin_v_7_4 * sigma_kin_v_7_6
    x1625 = x1623 * x825 + x1624 * x817
    x1626 = x1625 * x208
    x1627 = sigma_kin_v_6_4 * x157
    x1628 = x1627 * x824
    x1629 = sigma_kin_v_7_4 * x187
    x1630 = x149 * x1629
    x1631 = x1628 * x214 + x1630 * x835
    x1632 = sigma_kin_v_5_4 * sigma_kin_v_5_5
    x1633 = sigma_kin_v_6_4 * sigma_kin_v_6_5
    x1634 = sigma_kin_v_7_4 * sigma_kin_v_7_5
    x1635 = x1632 * x828 + x1633 * x825 + x1634 * x817
    x1636 = x1635 * x231
    x1637 = x1294 * x263
    x1638 = x1553 * x258
    x1639 = sigma_kin_v_4_4**2
    x1640 = x1639 * x571
    x1641 = sigma_kin_v_5_4**2
    x1642 = sigma_kin_v_6_4**2
    x1643 = x1642 * x279
    x1644 = sigma_kin_v_7_4**2
    x1645 = x1644 * x283
    x1646 = x1640 * x806 + x1641 * x828 + x1643 * x824 + x1645 * x816
    x1647 = sigma_kin_v_5_4 * x111
    x1648 = x1647 * x838
    x1649 = x1629 * x219
    x1650 = x1649 * x790
    x1651 = x1628 * x249 + x1648 * x556 + x1650 * x842
    x1652 = sigma_kin_v_4_4 * x61
    x1653 = x1652 * x806
    x1654 = x114 * x1648
    x1655 = x1627 * x279
    x1656 = x1655 * x820
    x1657 = x1629 * x283
    x1658 = x1657 * x822
    x1659 = x1653 * x575 + x1654 * x809 + x1656 * x430 + x1658 * x432
    x1660 = x1653 * x583 + x1654 * x850 + x1656 * x437 + x1658 * x439
    x1661 = x1193 * x61
    x1662 = x1655 * x323
    x1663 = x1657 * x860
    x1664 = x1648 * x546 + x1661 * x854 + x1662 * x857 + x1663 * x859
    x1665 = x362 * x489
    x1666 = x1133 * x373
    x1667 = x1432 * x377
    x1668 = x202 * x490
    x1669 = x1139 * x489
    x1670 = x1141 * x489
    x1671 = x380 * x490
    x1672 = sigma_kin_v_7_7 * x1671
    x1673 = x1661 * x867
    x1674 = x1661 * x866
    x1675 = x1647 * x870
    x1676 = x1647 * x871
    x1677 = x1655 * x879
    x1678 = x1655 * x874
    x1679 = x1657 * x880
    x1680 = x1657 * x878
    x1681 = x1673 + x1674 + x1675 + x1676 + x1677 + x1678 + x1679 + x1680
    x1682 = sigma_kin_v_6_4 * x425
    x1683 = sigma_kin_v_7_4 * x427
    x1684 = sigma_kin_v_6_6 * x1682 + sigma_kin_v_7_6 * x1683
    x1685 = x1623 * x167
    x1686 = x150 * x1624
    x1687 = x1685 * x430 + x1686 * x432
    x1688 = x1685 * x437 + x1686 * x439
    x1689 = x1623 * x445 + x1624 * x449
    x1690 = x1623 * x454 + x1624 * x457
    x1691 = x1623 * x461 + x1624 * x463
    x1692 = x1623 * x466 + x1624 * x469
    x1693 = x488 * x490
    x1694 = x1693 * x475
    x1695 = x179 + x180 + x181 + x184 + x185
    x1696 = x474 * (x1695 + 2 * x182 + x183)
    x1697 = x490 * x492
    x1698 = 2 * x237
    x1699 = x1176 + x507 + x509 + x515 + x522
    x1700 = x1633 * x466 + x1634 * x469
    x1701 = sigma_kin_v_6_5 * x1682 + sigma_kin_v_7_5 * x1683 + x1632 * x531
    x1702 = x119 * x1632
    x1703 = x1633 * x167
    x1704 = x150 * x1634
    x1705 = x1702 * x534 + x1703 * x430 + x1704 * x432
    x1706 = x1702 * x541 + x1703 * x437 + x1704 * x439
    x1707 = x1632 * x547 + x1633 * x445 + x1634 * x449
    x1708 = x1632 * x552 + x1633 * x454 + x1634 * x457
    x1709 = x1632 * x557 + x1633 * x461 + x1634 * x463
    x1710 = x1642 * x460
    x1711 = x1710 * x213
    x1712 = x1644 * x219
    x1713 = x1712 * x467
    x1714 = x116 * x1641
    x1715 = x1714 * x555
    x1716 = x1712 * x418
    x1717 = x1639 * x73
    x1718 = x119 * x1641
    x1719 = x1642 * x167
    x1720 = x150 * x1644
    x1721 = x1627 * x701
    x1722 = x1721 * x453
    x1723 = x1633 * x712 + x1634 * x713
    x1724 = x1623 * x712 + x1624 * x713
    x1725 = x1642 * x60
    x1726 = x1725 * x701
    x1727 = x1712 * x60
    x1728 = x141 * x1727
    x1729 = x1728 * x705
    x1730 = x160 * x1627
    x1731 = x162 * x1730
    x1732 = x1731 * x701
    x1733 = x141 * x1649
    x1734 = x1733 * x705
    x1735 = x1644 * x60
    x1736 = x149 * x1735
    x1737 = x1736 * x448
    x1738 = x1630 * x448
    x1739 = x1710 * x60
    x1740 = x1727 * x737
    x1741 = x1627 * x460
    x1742 = x1649 * x737
    x1743 = x1647 * x273
    x1744 = x1743 * x769
    x1745 = x1627 * x763
    x1746 = x1745 * x453
    x1747 = x1623 * x767 + x1624 * x760
    x1748 = x1632 * x770 + x1633 * x767 + x1634 * x760
    x1749 = x1714 * x60
    x1750 = x1749 * x754
    x1751 = x1725 * x763
    x1752 = x116 * x1647
    x1753 = x1752 * x754
    x1754 = x1728 * x766
    x1755 = x1731 * x763
    x1756 = x1733 * x766
    x1757 = x1749 * x320
    x1758 = x1752 * x320
    x1759 = x1641 * x60
    x1760 = x1759 * x273
    x1761 = x1727 * x790
    x1762 = x1716 * x60
    x1763 = x1629 * x759
    x1764 = x1652 * x269
    x1765 = x1764 * x233
    x1766 = x1743 * x276
    x1767 = sigma_kin_v_7_4 * x150
    x1768 = x1655 * x278
    x1769 = x1768 * x277
    x1770 = x1769 * x204
    x1771 = x137 * x1657
    x1772 = x287 * x379
    x1773 = x137 * x1686 + x1685 * x204
    x1774 = x137 * x1704 + x1702 * x227 + x1703 * x204
    x1775 = x1725 * x214
    x1776 = x137 * x1728
    x1777 = x1730 * x215
    x1778 = x137 * x1733
    x1779 = x1749 * x242
    x1780 = x1779 * x239
    x1781 = x1752 * x242
    x1782 = x1781 * x239
    x1783 = x1717 * x60
    x1784 = x1652 * x73
    x1785 = x1718 * x60
    x1786 = x1719 * x60
    x1787 = x119 * x1647
    x1788 = x1720 * x60
    x1789 = x1627 * x167
    x1790 = x1639 * x60
    x1791 = x1643 * x60
    x1792 = x1791 * x323
    x1793 = x1645 * x60
    x1794 = x137 * x1793
    x1795 = x1790 * x269
    x1796 = x1791 * x278
    x1797 = x1796 * x277
    x1798 = x1764 * x977
    x1799 = x1743 * x989
    x1800 = x1769 * x973
    x1801 = x1685 * x973 + x1686 * x967
    x1802 = x1702 * x975 + x1703 * x973 + x1704 * x967
    x1803 = x1728 * x967
    x1804 = x1730 * x981
    x1805 = x1733 * x967
    x1806 = x1725 * x249
    x1807 = x141 * x1793
    x1808 = x1807 * x363
    x1809 = x141 * x1657
    x1810 = x1809 * x363
    x1811 = x1767 * x187
    x1812 = x1661 * x921
    x1813 = x1743 * x905
    x1814 = x1768 * x922
    x1815 = x1623 * x903 + x1624 * x895
    x1816 = x1632 * x906 + x1633 * x903 + x1634 * x895
    x1817 = x1627 * x902
    x1818 = x1630 * x894
    x1819 = x1790 * x886
    x1820 = x1652 * x886
    x1821 = x1749 * x890
    x1822 = x1791 * x898
    x1823 = x1752 * x890
    x1824 = x1655 * x898
    x1825 = x1640 * x60
    x1826 = x1793 * x860
    x1827 = x1759 * x838
    x1828 = x1790 * x806
    x1829 = x114 * x1827
    x1830 = x1791 * x820
    x1831 = x1793 * x822
    x1832 = 1.0 * x1041 * x108 * x85 * x86 * x87
    x1833 = sigma_kin_v_7_5 * x154
    x1834 = x1833 * x760
    x1835 = x158 * x460
    x1836 = x188 * x759
    x1837 = 8 * dq_j5
    x1838 = sigma_kin_v_6_5 * sigma_kin_v_6_6
    x1839 = sigma_kin_v_7_5 * sigma_kin_v_7_6
    x1840 = x1838 * x767 + x1839 * x760
    x1841 = x1840 * x208
    x1842 = sigma_kin_v_6_5 * x157
    x1843 = x1842 * x460
    x1844 = sigma_kin_v_7_5 * x187
    x1845 = x1844 * x219
    x1846 = x1845 * x737
    x1847 = x1843 * x777 + x1846 * x778
    x1848 = x1267 * x263
    x1849 = x1530 * x258
    x1850 = x1748 * x237
    x1851 = sigma_kin_v_5_5**2
    x1852 = x116 * x1851
    x1853 = sigma_kin_v_6_5**2
    x1854 = x1853 * x460
    x1855 = sigma_kin_v_7_5**2
    x1856 = x1855 * x219
    x1857 = x1856 * x418
    x1858 = x1852 * x769 + x1854 * x763 + x1857 * x758
    x1859 = sigma_kin_v_5_5 * x111
    x1860 = x116 * x1859
    x1861 = x1860 * x754
    x1862 = x160 * x1842
    x1863 = x162 * x1862
    x1864 = x1863 * x763
    x1865 = x141 * x1845
    x1866 = x1865 * x766
    x1867 = x1861 * x534 + x1864 * x430 + x1866 * x432
    x1868 = x1861 * x541 + x1864 * x437 + x1866 * x439
    x1869 = x1860 * x320
    x1870 = x1842 * x763
    x1871 = x149 * x1844
    x1872 = x1871 * x448
    x1873 = x1869 * x784 + x1870 * x444 + x1872 * x766
    x1874 = x1859 * x273
    x1875 = x1845 * x790
    x1876 = x1870 * x453 + x1874 * x769 + x1875 * x789
    x1877 = x385 * x490
    x1878 = sigma_kin_v_7_7 * x1877
    x1879 = x1860 * x797
    x1880 = x1860 * x795
    x1881 = x1843 * x796
    x1882 = x1843 * x798
    x1883 = x1844 * x759
    x1884 = x1883 * x800
    x1885 = x1883 * x802
    x1886 = x1879 + x1880 + x1881 + x1882 + x1884 + x1885
    x1887 = x1838 * x425 + x1839 * x427
    x1888 = x167 * x1838
    x1889 = x150 * x1839
    x1890 = x1888 * x430 + x1889 * x432
    x1891 = x1888 * x437 + x1889 * x439
    x1892 = x1838 * x445 + x1839 * x449
    x1893 = x1838 * x454 + x1839 * x457
    x1894 = x1838 * x461 + x1839 * x463
    x1895 = x1838 * x466 + x1839 * x469
    x1896 = x1697 * x475
    x1897 = x474 * (x1695 + x182 + 2 * x183)
    x1898 = 2 * x231
    x1899 = x1854 * x213
    x1900 = x1856 * x467
    x1901 = x119 * x1851
    x1902 = x167 * x1853
    x1903 = x150 * x1855
    x1904 = x1853 * x279
    x1905 = x1855 * x283
    x1906 = x1843 * x735
    x1907 = x1846 * x736
    x1908 = x1838 * x712 + x1839 * x713
    x1909 = x110 * x1853
    x1910 = x1909 * x701
    x1911 = x110 * x1856
    x1912 = x141 * x1911
    x1913 = x1912 * x705
    x1914 = x1863 * x701
    x1915 = x1865 * x705
    x1916 = x110 * x1855
    x1917 = x149 * x1916
    x1918 = x1917 * x448
    x1919 = x1842 * x701
    x1920 = x110 * x1854
    x1921 = x1911 * x737
    x1922 = x1860 * x242
    x1923 = x1922 * x239
    x1924 = x1923 * x227
    x1925 = sigma_kin_v_7_5 * x150
    x1926 = x1862 * x215
    x1927 = x1926 * x249
    x1928 = x137 * x1865
    x1929 = x1928 * x252
    x1930 = x137 * x1889 + x1888 * x204
    x1931 = x1909 * x214
    x1932 = x137 * x1912
    x1933 = x110 * x1901
    x1934 = x110 * x1902
    x1935 = x119 * x1859
    x1936 = x110 * x1903
    x1937 = x167 * x1842
    x1938 = x110 * x1852
    x1939 = x1938 * x320
    x1940 = x110 * x1904
    x1941 = x1940 * x323
    x1942 = x110 * x1905
    x1943 = x137 * x1942
    x1944 = x1842 * x279
    x1945 = x1944 * x323
    x1946 = x1844 * x283
    x1947 = x137 * x1946
    x1948 = x110 * x1851
    x1949 = x1948 * x273
    x1950 = x1940 * x278
    x1951 = x1950 * x277
    x1952 = x1944 * x278
    x1953 = x1952 * x277
    x1954 = x1938 * x242
    x1955 = x1954 * x239
    x1956 = x1923 * x975
    x1957 = x1862 * x981
    x1958 = x1957 * x249
    x1959 = x1865 * x967
    x1960 = x1959 * x252
    x1961 = x1888 * x973 + x1889 * x967
    x1962 = x1912 * x967
    x1963 = x141 * x1942
    x1964 = x1963 * x363
    x1965 = x141 * x1946
    x1966 = x1965 * x363
    x1967 = x1909 * x249
    x1968 = x187 * x1925
    x1969 = x1922 * x917
    x1970 = x1842 * x902
    x1971 = x1970 * x249
    x1972 = x1871 * x894
    x1973 = x1972 * x252
    x1974 = x1838 * x903 + x1839 * x895
    x1975 = x1938 * x890
    x1976 = x1940 * x898
    x1977 = x1860 * x890
    x1978 = x1944 * x898
    x1979 = x1942 * x860
    x1980 = x1946 * x860
    x1981 = x1859 * x838
    x1982 = x1981 * x556
    x1983 = x1842 * x824
    x1984 = x1983 * x249
    x1985 = x1838 * x825 + x1839 * x817
    x1986 = x1948 * x838
    x1987 = x114 * x1986
    x1988 = x1940 * x820
    x1989 = x114 * x1981
    x1990 = x1942 * x822
    x1991 = x1944 * x820
    x1992 = x1946 * x822
    x1993 = x1911 * x790
    x1994 = x1938 * x754
    x1995 = x1909 * x763
    x1996 = x1912 * x766
    x1997 = x110 * x1857
    x1998 = sigma_kin_v_7_6 * x699
    x1999 = x188 * x219
    x2000 = 8 * dq_j6
    x2001 = sigma_kin_v_7_6 * x187
    x2002 = x1241 * x263
    x2003 = x1504 * x258
    x2004 = x1724 * x237
    x2005 = x1908 * x231
    x2006 = sigma_kin_v_6_6**2
    x2007 = x2006 * x460
    x2008 = sigma_kin_v_7_6**2
    x2009 = x2008 * x219
    x2010 = x2007 * x701 + x2009 * x708
    x2011 = sigma_kin_v_6_6 * x157
    x2012 = x160 * x2011
    x2013 = x162 * x2012
    x2014 = x2013 * x701
    x2015 = x2001 * x219
    x2016 = x141 * x2015
    x2017 = x2016 * x705
    x2018 = x2014 * x430 + x2017 * x432
    x2019 = x2014 * x437 + x2017 * x439
    x2020 = x2011 * x701
    x2021 = x149 * x2001
    x2022 = x2021 * x448
    x2023 = x2020 * x444 + x2022 * x705
    x2024 = x2020 * x453 + x2021 * x732
    x2025 = x2011 * x460
    x2026 = x2015 * x737
    x2027 = x2025 * x735 + x2026 * x736
    x2028 = x2025 * x749
    x2029 = x2025 * x744
    x2030 = x2015 * x751
    x2031 = x2015 * x748
    x2032 = x2028 + x2029 + x2030 + x2031
    x2033 = dq_i7 * x481
    x2034 = dq_i7 * x489
    x2035 = x133 * x365
    x2036 = sigma_kin_v_7_7 * x493
    x2037 = x167 * x2006
    x2038 = x150 * x2008
    x2039 = x2006 * x279
    x2040 = x2008 * x283
    x2041 = x2009 * x418
    x2042 = x389 * x518
    x2043 = x2042 * x474
    x2044 = x474 * (x179 + x180 + x181 + x182 + x183 + 2 * x184 + x185)
    x2045 = 2 * x208
    x2046 = sigma_kin_v_7_6 * x150
    x2047 = x2012 * x215
    x2048 = x2047 * x214
    x2049 = x137 * x2016
    x2050 = x224 * x389
    x2051 = x156 * x2037
    x2052 = x156 * x2038
    x2053 = x167 * x2011
    x2054 = x156 * x2039
    x2055 = x2054 * x323
    x2056 = x156 * x2040
    x2057 = x137 * x2056
    x2058 = x2011 * x279
    x2059 = x2058 * x323
    x2060 = x2001 * x283
    x2061 = x137 * x2060
    x2062 = x2054 * x278
    x2063 = x2062 * x277
    x2064 = x2058 * x278
    x2065 = x2064 * x277
    x2066 = x156 * x2006
    x2067 = x156 * x2009
    x2068 = x141 * x2067
    x2069 = x137 * x2068
    x2070 = x2066 * x214
    x2071 = x2012 * x981
    x2072 = x2071 * x214
    x2073 = x2016 * x967
    x2074 = x141 * x2056
    x2075 = x2074 * x363
    x2076 = x141 * x2060
    x2077 = x2076 * x363
    x2078 = x2066 * x249
    x2079 = x2068 * x967
    x2080 = x187 * x2046
    x2081 = sigma_kin_v_7_6 * x154
    x2082 = x2011 * x902
    x2083 = x2082 * x214
    x2084 = x2021 * x894
    x2085 = x2054 * x898
    x2086 = x2058 * x898
    x2087 = x2056 * x860
    x2088 = x2060 * x860
    x2089 = x156 * x2008
    x2090 = x2011 * x824
    x2091 = x2090 * x214
    x2092 = x2054 * x820
    x2093 = x2056 * x822
    x2094 = x2058 * x820
    x2095 = x2060 * x822
    x2096 = x2067 * x790
    x2097 = x2015 * x790
    x2098 = x149 * x2089
    x2099 = x2025 * x777
    x2100 = x2066 * x763
    x2101 = x2068 * x766
    x2102 = x2013 * x763
    x2103 = x2016 * x766
    x2104 = x2098 * x448
    x2105 = x2011 * x763
    x2106 = x156 * x2007
    x2107 = x2067 * x737
    x2108 = x156 * x2041
    x2109 = x2001 * x759
    x2110 = x2066 * x701
    x2111 = x2068 * x705
    x2112 = x148 * x419
    x2113 = x2112 * x520
    x2114 = 4 * x2113
    x2115 = 4 * sigma_kin_v_7_7**2
    x2116 = x188 * x419
    x2117 = x148 * x503
    x2118 = 4 * x185 * x187 * x419
    x2119 = 8 * x199
    x2120 = x153 * x475
    x2121 = x148 * x2120
    x2122 = 8 * x418
    x2123 = dq_i4 * x187
    x2124 = dq_i5 * x187
    x2125 = x138 * x483
    x2126 = x134 * x485
    x2127 = x1135 * x490
    x2128 = x1137 * x490
    x2129 = x2035 * x223
    x2130 = x192 * x390
    x2131 = sigma_kin_v_7_7 * x149 * x499
    x2132 = x392 * x499
    x2133 = dq_i7 * x490
    x2134 = x185 * x474
    x2135 = dq_i1 * x144
    x2136 = x189 * x2135
    x2137 = x185 * x493
    x2138 = x132 * x497
    x2139 = x132 * x153
    x2140 = 2 * x519
    x2141 = x2135 * x366
    x2142 = x132 * x185
    x2143 = x393 * x499
    x2144 = dq_i2 * x1023
    x2145 = x185 * x497
    x2146 = x497 * x966
    x2147 = x153 * x966
    x2148 = dq_i2 * x371 * x966
    x2149 = x185 * x966
    x2150 = x144 * x518
    x2151 = x187 * x966
    x2152 = dq_i3 * x948
    x2153 = x497 * x893
    x2154 = x153 * x893
    x2155 = dq_i3 * x485 * x893
    x2156 = x185 * x893
    x2157 = x187 * x893
    x2158 = dq_i4 * x875
    x2159 = x497 * x815
    x2160 = x153 * x815
    x2161 = x1671 * x876
    x2162 = x185 * x815
    x2163 = x187 * x815
    x2164 = dq_i5 * x799
    x2165 = x497 * x757
    x2166 = x153 * x757
    x2167 = x1877 * x802
    x2168 = x185 * x757
    x2169 = x187 * x757
    x2170 = dq_i6 * x745
    x2171 = x497 * x697
    x2172 = dq_i6 * x746
    x2173 = x185 * x697
    x2174 = x187 * x697

    K_block_list = []
    K_block_list.append(
        4 * ddq_i1 * x10 * x307
        + 4
        * ddq_j1
        * (
            ddq_i1 * (x290 * x650 + x293 * x605 + x296 * x571 + x299 * x530 + x302 * x424 + x305 * x416 + x6 * x7**2)
            + x11 * x668
            + x11 * x669
            + x11 * x670
            + x11 * x671
            + x11 * x672
            + x11 * x673
            + x11 * x674
            - x12 * (x668 + x669 + x670 + x671 + x672 + x673 + x674)
            - x304 * x675
            - x435 * (x212 * x677 + x223 * x679)
            - x539 * (x242 * x681 + x249 * x676 + x251 * x682)
            - x581 * (x267 * x684 + x278 * x687 + x286 * x689 + x551 * x685)
            - x616 * (x310 * x690 + x317 * x683 + x444 * x686 + x448 * x688 + x546 * x680)
            - x627 * (x331 * x691 + x336 * x692 + x340 * x693 + x345 * x694 + x350 * x695 + x356 * x696)
        )
        - dq_i2 * x195 * x360
        - dq_i3 * x195 * x328
        - dq_i4 * x195 * x289
        - dq_i5 * x195 * x253
        - dq_i6 * x195 * x226
        - sigma_kin_v_7_1
        * x361
        * x395
        * x396
        * (-x367 + x369 * x373 + x375 * x377 + x379 * x382 + x384 * x386 + x387 * x389 + x390 * x393)
        + sigma_kin_v_7_1
        * x524
        * (
            -sigma_kin_v_7_1 * x496 * x498
            + x179 * x502
            + x179 * x505
            + x478 * x480
            - x479
            + x480 * x484
            + x480 * x487
            + x480 * x494
            + x488 * x491
            + x491 * x492
            + x516
            + x522
        )
        + x1027
        * (
            2 * dq_i1 * dq_i7 * dq_j1 * x134 * x139 * x141 * x143 * x145 * x148 * x201 * x304 * x361 * x966
            + 2
            * dq_i1
            * (
                x1000 * x970
                + x1001 * x1003
                + x1002 * x972
                + x953 * x997
                + x954
                + x955 * x993
                + x959
                + x961 * x998
                + x962
                + x964 * x999
                + x965
                + x971
            )
            + dq_i2
            * (
                sigma_kin_v_3_1 * x1011 * x617
                + x1000 * x1022
                + x1001 * x1024
                + x1004 * x31 * x691
                + x1005 * x333
                + x1006 * x997
                + x1007 * x332
                + x1008 * x31 * x692
                + x1009 * x338
                + x1010 * x931
                + x1012 * x31 * x693
                + x1013 * x342
                + x1014 * x998
                + x1015 * x341
                + x1016 * x31 * x694
                + x1017 * x347
                + x1018 * x999
                + x1019 * x31 * x695
                + x1020 * x346
                + x1021 * x352
                + x1023 * x31 * x696
                + x1025 * x351
                + x1026 * x412
                + x934 * x972
            )
            + 2
            * dq_i3
            * (
                x270 * x994
                + x321 * x988
                + x324 * x981
                + x785 * x988
                + x843 * x994
                + x858 * x981
                + x936 * x996
                + x937 * x996
                + x958 * x992
                + x992 * x993
            )
            + 2
            * dq_i4
            * (
                x271 * x977
                + x274 * x989
                + x282 * x973
                + x284 * x991
                + x787 * x989
                + x848 * x991
                + x973 * x990
                + x977 * x987
            )
            + 2 * dq_i5 * (x246 * x975 + x249 * x984 + x252 * x983 + x252 * x985 + x840 * x982 + x975 * x986)
            + 2 * dq_i6 * (x214 * x984 + x225 * x983 + x225 * x985 + x834 * x982)
            + 2 * dq_i7 * sigma_kin_v_7_1 * x134 * x139 * x141 * x143 * x145 * x148 * x187 * x201 * x361 * x966
            - x151 * x968
            - x208 * x974
            - x231 * x976
            - x237 * x978
            - x258 * x979
            - x263 * x980
            - x357 * x477 * x972
            - x476 * x954
            - x476 * x959
            - x476 * x962
            - x476 * x965
            - x476 * x971
            - x496 * (x691 * x952 + x692 * x956 + x693 * x960 + x694 * x963 + x695 * x969 + x696 * x966)
        )
        + x109 * x85 * (sigma_pot_1_c * x2 + sigma_pot_1_s * x5)
        + x112 * x126
        + x112 * x130
        + x13 * x18
        + x13 * x9
        + x131 * x155
        + x131 * x209
        + x131 * x232
        + x131 * x238
        + x131 * x259
        + x131 * x264
        - x131
        * (
            dq_i1 * x399
            + dq_i1 * x400
            + dq_i1 * x401
            + dq_i1 * x402
            + dq_i1 * x403
            + dq_i1 * x404
            + dq_i1 * x405
            + dq_i1 * x406
            + dq_i1 * x407
            + dq_i1 * x408
            + dq_i1 * x411
            + dq_i1 * x413
            - dq_i1
            * (
                x1000 * x177
                + x1001 * x1039
                + x120 * x300 * x31
                + x129 * x999
                + x168 * x303 * x31
                + x189 * x306 * x31
                + x19 * x291 * x31
                + x294 * x31 * x43
                + x297 * x31 * x74
                + x37 * x997
                + 2 * x397
                + 2 * x398
                + x414
                + x58 * x925
                + x83 * x998
            )
            + x1028 * x18
            + x1028 * x9
            - x1029 * x196 * x200 * x202
            - x1029 * x200 * x202 * x726
            + x155
            + x209
            + x232
            + x238
            + x259
            + x264
            + x307 * x496
            - x370 * (x1000 * x1037 + x1001 * x1038 + x1034 * x997 + x1035 * x998 + x1036 * x999 + x254 * x933 + x360)
            - x376 * (x1032 * x327 + x1033 * x925 + x215 * x858 + x275 * x785 + x318 * x843 + x328)
            - x378 * (x1032 * x288 + x204 * x990 + x233 * x987 + x276 * x787 + x289)
            - x383 * (x1030 * x840 + x1031 * x252 + x227 * x986 + x253)
            - x388 * (x1030 * x834 + x1031 * x225 + x226)
        )
        + x158 * x174
        + x158 * x178
        + x188 * x191
        + x188 * x193 * x194
        - x195 * x196 * x200 * x203
        + x30 * x34
        - 4 * x31 * (x397 + x398 + x414)
        + x34 * x37
        + x40 * x55
        + x40 * x59
        + x473
        * (
            ddq_i6 * x429
            + 2 * dq_i1 * dq_i6 * sigma_kin_v_6_1 * sigma_kin_v_6_6 * x160 * x162 * x164 * x166 * x170 * x171 * x176
            + 2
            * dq_i1
            * dq_i6
            * sigma_kin_v_7_1
            * sigma_kin_v_7_6
            * x133
            * x135
            * x139
            * x141
            * x143
            * x145
            * x149
            * x192
            - x415 * x422
            - x434 * x435
            - x440 * x441
            - x450 * x451
            - x458 * x459
            - x464 * x465
            - x470 * x472
        )
        + x561
        * (
            ddq_i5 * x533
            + 2 * dq_i1 * dq_i5 * sigma_kin_v_5_1 * sigma_kin_v_5_5 * x114 * x116 * x118 * x122 * x123 * x128
            + 2 * dq_i1 * dq_i5 * sigma_kin_v_6_1 * sigma_kin_v_6_5 * x160 * x162 * x164 * x166 * x170 * x171 * x176
            + 2
            * dq_i1
            * dq_i5
            * sigma_kin_v_7_1
            * sigma_kin_v_7_5
            * x133
            * x135
            * x139
            * x141
            * x143
            * x145
            * x149
            * x192
            - x465 * x528
            - x525 * x526
            - x538 * x539
            - x542 * x543
            - x549 * x550
            - x553 * x554
            - x558 * x560
        )
        + x594
        * (
            ddq_i4 * x574
            + 2 * dq_i1 * dq_i4 * sigma_kin_v_4_1 * sigma_kin_v_4_4 * x64 * x72 * x76 * x77 * x82
            + 2 * dq_i1 * dq_i4 * sigma_kin_v_5_1 * sigma_kin_v_5_4 * x114 * x116 * x118 * x122 * x123 * x128
            + 2 * dq_i1 * dq_i4 * sigma_kin_v_6_1 * sigma_kin_v_6_4 * x160 * x162 * x164 * x166 * x170 * x171 * x176
            + 2
            * dq_i1
            * dq_i4
            * sigma_kin_v_7_1
            * sigma_kin_v_7_4
            * x133
            * x135
            * x139
            * x141
            * x143
            * x145
            * x149
            * x192
            - x459 * x567
            - x554 * x569
            - x562 * x563
            - x580 * x581
            - x584 * x585
            - x588 * x589
            - x591 * x593
        )
        + x62 * x80
        + x62 * x84
        + x626
        * (
            ddq_i3 * x608
            + 2 * dq_i1 * dq_i3 * sigma_kin_v_3_1 * sigma_kin_v_3_3 * x42 * x51 * x52 * x57
            + 2 * dq_i1 * dq_i3 * sigma_kin_v_4_1 * sigma_kin_v_4_3 * x64 * x72 * x76 * x77 * x82
            + 2 * dq_i1 * dq_i3 * sigma_kin_v_5_1 * sigma_kin_v_5_3 * x114 * x116 * x118 * x122 * x123 * x128
            + 2 * dq_i1 * dq_i3 * sigma_kin_v_6_1 * sigma_kin_v_6_3 * x160 * x162 * x164 * x166 * x170 * x171 * x176
            + 2
            * dq_i1
            * dq_i3
            * sigma_kin_v_7_1
            * sigma_kin_v_7_3
            * x133
            * x135
            * x139
            * x141
            * x143
            * x145
            * x149
            * x192
            - x451 * x600
            - x550 * x602
            - x589 * x603
            - x595 * x596
            - x615 * x616
            - x619 * x620
            - x623 * x625
        )
        + x667
        * (
            ddq_i2 * x651
            - x441 * x645
            - x543 * x647
            - x585 * x648
            - x620 * x653
            + x627 * x628
            + x627 * x630
            + x627 * x632
            + x627 * x634
            + x627 * x636
            + x627 * x638
            - x627 * x654
            - x639 * x641
            - x664 * x666
        )
        + x752
        * (
            2 * dq_i1 * dq_i7 * dq_j1 * x135 * x139 * x141 * x143 * x144 * x148 * x201 * x304 * x361 * x697
            + 2 * dq_i1 * (x432 * x722 + x704 + x705 * x723 + x717 * x719)
            + 2 * dq_i2 * (x437 * x703 + x439 * x722 + x705 * x725 + x719 * x724)
            + 2 * dq_i3 * (x444 * x719 + x444 * x729 + x705 * x728 + x705 * x731)
            + 2 * dq_i4 * (x453 * x719 + x453 * x729 + x727 * x732 + x730 * x732)
            + 2 * dq_i5 * (x733 * x735 + x735 * x739 + x736 * x738 + x736 * x741)
            + dq_i6
            * (
                x220 * x751
                + x31 * x677 * x742
                + x31 * x679 * x745
                + x720 * x748
                + x733 * x744
                + x739 * x749
                + x740 * x747
                + x743 * x750
            )
            + 2 * dq_i7 * sigma_kin_v_7_1 * x135 * x139 * x141 * x143 * x144 * x148 * x187 * x201 * x361 * x697
            - sigma_kin_v_7_1 * x699
            - x208 * x715
            - x231 * x714
            - x237 * x711
            - x258 * x710
            - x263 * x709
            - x476 * x704
            - x496 * (x677 * x700 + x679 * x697)
            - x705 * x707
        )
        + x804
        * (
            2 * dq_i1 * dq_i7 * dq_j1 * x135 * x139 * x141 * x142 * x145 * x148 * x201 * x304 * x361 * x757
            + 2 * dq_i1 * (x432 * x782 + x534 * x780 + x717 * x781 + x723 * x766 + x756 + x765)
            + 2 * dq_i2 * (x437 * x764 + x439 * x782 + x541 * x755 + x541 * x780 + x724 * x781 + x725 * x766)
            + 2 * dq_i3 * (x321 * x784 + x444 * x781 + x444 * x786 + x728 * x766 + x731 * x766 + x784 * x785)
            + 2 * dq_i4 * (x274 * x769 + x453 * x781 + x453 * x786 + x769 * x787 + x789 * x791 + x789 * x792)
            + dq_i5
            * (
                x244 * x797
                + x245 * x794
                + x31 * x681 * x793
                + x733 * x796
                + x733 * x798
                + x739 * x796
                + x739 * x798
                + x779 * x795
                + x800 * x801
                + x800 * x803
                + x801 * x802
                + x802 * x803
            )
            + 2 * dq_i6 * (x733 * x777 + x738 * x778 + x741 * x778 + x750 * x776)
            + 2 * dq_i7 * sigma_kin_v_7_1 * x135 * x139 * x141 * x142 * x145 * x148 * x187 * x201 * x361 * x757
            - x208 * x768
            - x231 * x775
            - x237 * x774
            - x258 * x773
            - x263 * x772
            - x476 * x756
            - x476 * x765
            - x496 * (x676 * x763 + x681 * x753 + x682 * x758)
            - x707 * x766
            - x760 * x761
        )
        + x881
        * (
            2 * dq_i1 * dq_i7 * dq_j1 * x135 * x139 * x140 * x143 * x145 * x148 * x201 * x304 * x361 * x815
            + 2 * dq_i1 * (x430 * x847 + x432 * x823 + x432 * x849 + x575 * x844 + x808 + x809 * x845 + x814 + x821)
            + 2
            * dq_i2
            * (
                x437 * x847
                + x439 * x823
                + x439 * x849
                + x583 * x807
                + x583 * x844
                + x813 * x850
                + x820 * x851
                + x845 * x850
            )
            + 2
            * dq_i3
            * (
                x324 * x857
                + x546 * x812
                + x546 * x839
                + x852 * x854
                + x854 * x855
                + x857 * x858
                + x859 * x861
                + x859 * x863
            )
            + dq_i4
            * (
                x243 * x871
                + x280 * x879
                + x281 * x873
                + x284 * x880
                + x31 * x684 * x864
                + x31 * x687 * x872
                + x31 * x689 * x875
                + x811 * x869
                + x837 * x870
                + x837 * x871
                + x846 * x874
                + x848 * x878
                + x852 * x866
                + x855 * x867
                + x862 * x877
                + x865 * x868
            )
            + 2 * dq_i5 * (x249 * x836 + x556 * x812 + x556 * x839 + x791 * x842 + x792 * x842 + x824 * x840)
            + 2 * dq_i6 * (x214 * x836 + x727 * x835 + x730 * x835 + x824 * x834)
            + 2 * dq_i7 * sigma_kin_v_7_1 * x135 * x139 * x140 * x143 * x145 * x148 * x187 * x201 * x361 * x815
            - x208 * x826
            - x231 * x829
            - x237 * x833
            - x258 * x832
            - x263 * x831
            - x476 * x808
            - x476 * x814
            - x476 * x821
            - x496 * (x298 * x828 + x684 * x805 + x687 * x819 + x689 * x815)
            - x706 * x823
            - x761 * x817
        )
        + x951
        * (
            2 * dq_i1 * dq_i7 * dq_j1 * x135 * x138 * x141 * x143 * x145 * x148 * x201 * x304 * x361 * x893
            + 2
            * dq_i1
            * (
                x430 * x928
                + x575 * x926
                + x809 * x927
                + x883 * x925
                + x884
                + x888
                + x892
                + x899
                + x900 * x929
                + x929 * x930
            )
            + 2
            * dq_i2
            * (
                x339 * x910
                + x437 * x928
                + x583 * x887
                + x583 * x926
                + x850 * x891
                + x850 * x927
                + x851 * x898
                + x910 * x933
                + x935 * x936
                + x935 * x937
            )
            + dq_i3
            * (
                x244 * x944
                + x244 * x945
                + x280 * x946
                + x280 * x947
                + x284 * x949
                + x284 * x950
                + x31 * x690 * x938
                + x311 * x941
                + x312 * x939
                + x779 * x944
                + x779 * x945
                + x846 * x946
                + x846 * x947
                + x848 * x949
                + x848 * x950
                + x852 * x942
                + x852 * x943
                + x855 * x942
                + x855 * x943
                + x925 * x940
            )
            + 2
            * dq_i4
            * (
                x274 * x905
                + x281 * x922
                + x787 * x905
                + x852 * x921
                + x861 * x924
                + x863 * x924
                + x868 * x920
                + x922 * x923
            )
            + 2 * dq_i5 * (x245 * x917 + x249 * x915 + x252 * x916 + x726 * x919 + x840 * x902 + x917 * x918)
            + 2 * dq_i6 * (x214 * x915 + x225 * x916 + x726 * x914 + x834 * x902)
            + 2 * dq_i7 * sigma_kin_v_7_1 * x135 * x138 * x141 * x143 * x145 * x148 * x187 * x201 * x361 * x893
            - x208 * x904
            - x231 * x907
            - x237 * x909
            - x258 * x912
            - x263 * x911
            - x476 * x884
            - x476 * x888
            - x476 * x892
            - x476 * x899
            - x496 * (x680 * x905 + x683 * x886 + x686 * x902 + x688 * x894 + x690 * x882)
            - x706 * x900 * x901
            - x761 * x895
        )
    )
    K_block_list.append(
        4 * ddq_i2 * dq_j2**2 * x1110
        + 4
        * ddq_j2
        * (
            ddq_i2 * (x1098 * x650 + x1100 * x606 + x1102 * x572 + x1104 * x531 + x1106 * x425 + x1108 * x427)
            - x1108 * x641
            + x1207 * x665
            + x1208 * x665
            + x1209 * x665
            + x1210 * x665
            + x1211 * x665
            + x1212 * x665
            - x441 * (x1214 * x212 + x1216 * x223)
            - x543 * (x1213 * x249 + x1218 * x242 + x1219 * x251)
            - x585 * (x1104 * x552 + x1221 * x267 + x1223 * x278 + x1225 * x286)
            - x620 * (x1217 * x546 + x1220 * x317 + x1222 * x444 + x1224 * x448 + x1227 * x310)
            - x627 * (x1226 * x609 + x1228 * x36 + x1229 * x82 + x1230 * x128 + x1231 * x176 + x1232 * x192)
            - x666 * (x1207 + x1208 + x1209 + x1210 + x1211 + x1212)
        )
        - dq_i1 * x1047 * x1131
        - dq_i3 * x1047 * x1125
        - dq_i4 * x1047 * x1118
        - dq_i5 * x1047 * x1085
        - dq_i6 * x1047 * x1062
        + dq_j2 * x1097 * x980
        - sigma_kin_v_7_2 * x1047 * x1048 * x199 * x374 * x995
        - sigma_kin_v_7_2
        * x1144
        * x135
        * x396
        * (
            -x1132 * x371
            + x1134 * x1136
            + x1134 * x1138
            + x1140 * x371
            + x1142 * x138
            + x202 * x486
            + x369 * x371 * x477
        )
        + sigma_kin_v_7_2
        * x524
        * (
            -sigma_kin_v_7_2 * x1174 * x498
            - x1170
            + x1172 * x478
            + x1172 * x484
            + x1172 * x487
            + x1172 * x494
            + x1173 * x488
            + x1173 * x492
            + x1177
            + x180 * x502
            + x180 * x505
            + x509
        )
        + x1006 * x34
        + x1007 * x34
        + x1008 * x40 * x617
        + x1010 * x40
        + x1014 * x1040
        + x1015 * x1040
        + x1018 * x1042
        + x1020 * x1042
        + x1022 * x1045
        + x1024 * x1046
        + x1025 * x1045
        + x1026 * x1046
        + x1027 * x1044
        + x1027 * x1054
        + x1027 * x1070
        + x1027 * x1080
        + x1027 * x1096
        + x1027
        * (
            2
            * dq_i1
            * (x1003 * x1363 + x1131 + x1359 * x953 + x1360 * x961 + x1361 * x964 + x1362 * x970 + x1365 * x955)
            + 2 * dq_i2 * dq_i7 * dq_j2 * x1108 * x134 * x139 * x141 * x143 * x145 * x148 * x201 * x361 * x966
            - dq_i2 * x1145
            - dq_i2 * x1146
            - dq_i2 * x1147
            - dq_i2 * x1148
            - dq_i2 * x1149
            - dq_i2 * x1150
            - dq_i2 * x1151
            - dq_i2 * x1152
            - dq_i2 * x1153
            - dq_i2 * x1154
            - dq_i2 * x1156
            - dq_i2 * x1157
            + dq_i2
            * (
                x1004 * x1099 * x32
                + x1006 * x1359
                + x1008 * x1101 * x32
                + x1010 * x1335
                + x1014 * x1360
                + x1015 * x1360
                + x1018 * x1361
                + x1020 * x1361
                + x1022 * x1362
                + x1024 * x1363
                + x1025 * x1362
                + x1026 * x1363
                + x1158
            )
            + 2 * dq_i3 * (x1125 + x1272 * x988 + x1299 * x994 + x1308 * x981 + x1337 * x996 + x1365 * x992)
            + 2 * dq_i4 * (x1118 + x1275 * x989 + x1304 * x991 + x1354 * x977 + x1356 * x973)
            + 2 * dq_i5 * (x1085 + x1298 * x982 + x1352 * x975 + x1364 * x252)
            + 2 * dq_i6 * (x1062 + x1295 * x982 + x1364 * x225)
            + 2 * dq_i7 * sigma_kin_v_7_2 * x134 * x139 * x141 * x143 * x145 * x148 * x187 * x201 * x361 * x966
            - x1044
            - x1054
            - x1070
            - x1080
            - x1096
            - x1110 * x1174
            - x495 * x980
        )
        + x1041 * x109 * (sigma_pot_2_c * x25 + sigma_pot_2_s * x22)
        - 4 * x1158 * x32
        + x1206
        * (
            ddq_i1 * x651
            - x12 * x654
            - x435 * x645
            - x539 * x647
            - x581 * x648
            - x616 * x653
            + x627 * x657
            + x627 * x659
            + x627 * x660
            + x627 * x661
            + x627 * x662
            + x627 * x663
            - x627 * x664
            - x639 * x675
        )
        + x131
        * (
            dq_i1
            * (
                x1039 * x1363
                + x1043 * x410
                + x1055 * x174
                + x1055 * x178
                + x1058 * x191
                + x1081 * x126
                + x1081 * x130
                + x1119 * x55
                + x1119 * x59
                + x1126 * x30
                + x1126 * x37
                + x120 * x1230 * x32
                + x1228 * x19 * x32
                + x1229 * x32 * x74
                + x1231 * x168 * x32
                + x1232 * x189 * x32
                + x1280 * x80
                + x1280 * x84
                + x129 * x1361
                + x1329 * x54
                + x1329 * x58
                + x1359 * x37
                + x1360 * x83
                + x1362 * x177
            )
            + 2 * dq_i2 * dq_i7 * dq_j2 * x1108 * x132 * x133 * x135 * x139 * x141 * x143 * x145 * x148 * x201
            + 2
            * dq_i2
            * (
                x1034 * x1359
                + x1035 * x1360
                + x1036 * x1361
                + x1037 * x1362
                + x1038 * x1363
                + x1336 * x254
                + x1338
                + x1339
                + x1340
                + x1341
                + x1343
                + x1344
            )
            + 2
            * dq_i3
            * (
                x1033 * x1329
                + x1033 * x1330
                + x1121 * x275
                + x1122 * x215
                + x1272 * x275
                + x1299 * x318
                + x1308 * x215
                + x1355 * x652
                + x1357 * x327
                + x1358 * x327
            )
            + 2
            * dq_i4
            * (
                x1111 * x1355
                + x1113 * x276
                + x1116 * x204
                + x1275 * x276
                + x1354 * x233
                + x1356 * x204
                + x1357 * x288
                + x1358 * x288
            )
            + 2 * dq_i5 * (x1084 * x227 + x1242 * x1353 + x1349 * x252 + x1350 * x249 + x1351 * x252 + x1352 * x227)
            + 2 * dq_i6 * (x1030 * x1295 + x1349 * x225 + x1350 * x214 + x1351 * x225)
            + 2 * dq_i7 * sigma_kin_v_7_2 * x132 * x133 * x135 * x139 * x141 * x143 * x145 * x148 * x187 * x201
            - x1043 * x1342
            - x1174 * (x1226 * x254 + x1228 * x35 + x1229 * x81 + x1230 * x127 + x1231 * x175 + x1232 * x132)
            - x1338 * x370
            - x1339 * x370
            - x1340 * x370
            - x1341 * x370
            - x1343 * x370
            - x1344 * x370
            - x1345 * x208
            - x1346 * x231
            - x1347 * x237
            - x1348 * x258
            - x262 * x495
        )
        + x473
        * (
            ddq_i6 * x1161
            + 2 * dq_i2 * dq_i6 * sigma_kin_v_6_2 * sigma_kin_v_6_6 * x160 * x162 * x164 * x166 * x169 * x350 * x423
            + 2
            * dq_i2
            * dq_i6
            * sigma_kin_v_7_2
            * sigma_kin_v_7_6
            * x134
            * x139
            * x141
            * x143
            * x145
            * x149
            * x356
            * x361
            - x1051 * x422
            - x1162 * x435
            - x1163 * x441
            - x1164 * x451
            - x1165 * x459
            - x1166 * x465
            - x1167 * x472
        )
        + x561
        * (
            ddq_i5 * x1180
            + 2 * dq_i2 * dq_i5 * sigma_kin_v_5_2 * sigma_kin_v_5_5 * x114 * x116 * x118 * x121 * x345 * x529
            + 2 * dq_i2 * dq_i5 * sigma_kin_v_6_2 * sigma_kin_v_6_5 * x160 * x162 * x164 * x166 * x169 * x350 * x423
            + 2
            * dq_i2
            * dq_i5
            * sigma_kin_v_7_2
            * sigma_kin_v_7_5
            * x134
            * x139
            * x141
            * x143
            * x145
            * x149
            * x356
            * x361
            - x1067 * x526
            - x1178 * x465
            - x1181 * x539
            - x1182 * x543
            - x1183 * x550
            - x1184 * x554
            - x1185 * x560
        )
        + x594
        * (
            ddq_i4 * x1189
            + 2 * dq_i2 * dq_i4 * sigma_kin_v_4_2 * sigma_kin_v_4_4 * x340 * x570 * x64 * x72 * x75
            + 2 * dq_i2 * dq_i4 * sigma_kin_v_5_2 * sigma_kin_v_5_4 * x114 * x116 * x118 * x121 * x345 * x529
            + 2 * dq_i2 * dq_i4 * sigma_kin_v_6_2 * sigma_kin_v_6_4 * x160 * x162 * x164 * x166 * x169 * x350 * x423
            + 2
            * dq_i2
            * dq_i4
            * sigma_kin_v_7_2
            * sigma_kin_v_7_4
            * x134
            * x139
            * x141
            * x143
            * x145
            * x149
            * x356
            * x361
            - x1077 * x563
            - x1186 * x459
            - x1187 * x554
            - x1190 * x581
            - x1192 * x585
            - x1194 * x589
            - x1195 * x593
        )
        + x626
        * (
            ddq_i3 * x1200
            + 2 * dq_i2 * dq_i3 * sigma_kin_v_3_2 * sigma_kin_v_3_3 * x336 * x41 * x51 * x604
            + 2 * dq_i2 * dq_i3 * sigma_kin_v_4_2 * sigma_kin_v_4_3 * x340 * x570 * x64 * x72 * x75
            + 2 * dq_i2 * dq_i3 * sigma_kin_v_5_2 * sigma_kin_v_5_3 * x114 * x116 * x118 * x121 * x345 * x529
            + 2 * dq_i2 * dq_i3 * sigma_kin_v_6_2 * sigma_kin_v_6_3 * x160 * x162 * x164 * x166 * x169 * x350 * x423
            + 2
            * dq_i2
            * dq_i3
            * sigma_kin_v_7_2
            * sigma_kin_v_7_3
            * x134
            * x139
            * x141
            * x143
            * x145
            * x149
            * x356
            * x361
            - x1093 * x596
            - x1196 * x451
            - x1197 * x550
            - x1199 * x589
            - x1203 * x616
            - x1204 * x620
            - x1205 * x625
        )
        + x752
        * (
            2 * dq_i1 * (x1234 * x430 + x1236 * x432 + x1243 * x717 + x1246 * x432)
            + 2 * dq_i2 * dq_i7 * dq_j2 * x1108 * x135 * x139 * x141 * x143 * x144 * x148 * x201 * x361 * x697
            + 2 * dq_i2 * (x1235 + x1237 + x1243 * x724 + x1246 * x439)
            + 2 * dq_i3 * (x1243 * x444 + x1249 * x705 + x1250 * x444 + x1252 * x705)
            + 2 * dq_i4 * (x1243 * x453 + x1248 * x732 + x1250 * x453 + x1251 * x732)
            + 2 * dq_i5 * (x1253 * x735 + x1254 * x736 + x1255 * x735 + x1256 * x736)
            + dq_i6
            * (
                x1059 * x748
                + x1059 * x751
                + x1214 * x32 * x742
                + x1216 * x32 * x745
                + x1244 * x748
                + x1253 * x744
                + x1255 * x744
                + x1255 * x749
            )
            + 2 * dq_i7 * sigma_kin_v_7_2 * x135 * x139 * x141 * x143 * x144 * x148 * x187 * x201 * x361 * x697
            - sigma_kin_v_7_2 * x699
            - x1174 * (x1214 * x700 + x1216 * x697)
            - x1235 * x370
            - x1237 * x370
            - x1238 * x258
            - x1239 * x237
            - x1240 * x231
            - x1241 * x208
            - x495 * x709
        )
        + x804
        * (
            2 * dq_i1 * (x1257 * x534 + x1260 * x430 + x1262 * x432 + x1269 * x534 + x1270 * x717 + x1271 * x432)
            + 2 * dq_i2 * dq_i7 * dq_j2 * x1108 * x135 * x139 * x141 * x142 * x145 * x148 * x201 * x361 * x757
            + 2 * dq_i2 * (x1258 + x1261 + x1263 + x1269 * x541 + x1270 * x724 + x1271 * x439)
            + 2 * dq_i3 * (x1121 * x784 + x1249 * x766 + x1252 * x766 + x1270 * x444 + x1272 * x784 + x1273 * x444)
            + 2 * dq_i4 * (x1113 * x769 + x1270 * x453 + x1273 * x453 + x1275 * x769 + x1276 * x789 + x1277 * x789)
            + dq_i5
            * (
                x1082 * x795
                + x1082 * x797
                + x1218 * x32 * x793
                + x1253 * x796
                + x1253 * x798
                + x1255 * x796
                + x1255 * x798
                + x1268 * x795
                + x1278 * x800
                + x1278 * x802
                + x1279 * x800
                + x1279 * x802
            )
            + 2 * dq_i6 * (x1253 * x777 + x1254 * x778 + x1255 * x777 + x1256 * x778)
            + 2 * dq_i7 * sigma_kin_v_7_2 * x135 * x139 * x141 * x142 * x145 * x148 * x187 * x201 * x361 * x757
            - x1174 * (x1213 * x763 + x1218 * x753 + x1219 * x758)
            - x1258 * x370
            - x1259 * x760
            - x1261 * x370
            - x1263 * x370
            - x1264 * x208
            - x1265 * x258
            - x1266 * x237
            - x1267 * x231
            - x495 * x772
        )
        + x881
        * (
            2
            * dq_i1
            * (
                x1281 * x575
                + x1284 * x809
                + x1286 * x430
                + x1288 * x432
                + x1300 * x575
                + x1301 * x809
                + x1303 * x430
                + x1305 * x432
            )
            + 2 * dq_i2 * dq_i7 * dq_j2 * x1108 * x135 * x139 * x140 * x143 * x145 * x148 * x201 * x361 * x815
            + 2 * dq_i2 * (x1282 + x1285 + x1287 + x1289 + x1300 * x583 + x1301 * x850 + x1303 * x437 + x1305 * x439)
            + 2
            * dq_i3
            * (
                x1122 * x857
                + x1283 * x546
                + x1297 * x546
                + x1306 * x854
                + x1307 * x854
                + x1308 * x857
                + x1309 * x859
                + x1310 * x859
            )
            + dq_i4
            * (
                x1081 * x870
                + x1081 * x871
                + x1114 * x874
                + x1114 * x879
                + x1117 * x878
                + x1117 * x880
                + x1221 * x32 * x864
                + x1223 * x32 * x872
                + x1225 * x32 * x875
                + x1274 * x870
                + x1274 * x871
                + x1302 * x874
                + x1304 * x878
                + x1306 * x866
                + x1307 * x866
                + x1307 * x867
            )
            + 2 * dq_i5 * (x1276 * x842 + x1277 * x842 + x1283 * x556 + x1296 * x249 + x1297 * x556 + x1298 * x824)
            + 2 * dq_i6 * (x1248 * x835 + x1251 * x835 + x1295 * x824 + x1296 * x214)
            + 2 * dq_i7 * sigma_kin_v_7_2 * x135 * x139 * x140 * x143 * x145 * x148 * x187 * x201 * x361 * x815
            - x1174 * (x1104 * x828 + x1221 * x805 + x1223 * x819 + x1225 * x815)
            - x1259 * x817
            - x1282 * x370
            - x1285 * x370
            - x1287 * x370
            - x1289 * x370
            - x1290 * x208
            - x1291 * x231
            - x1293 * x258
            - x1294 * x237
            - x495 * x831
        )
        + x951
        * (
            2
            * dq_i1
            * (
                x1123 * x929
                + x1313 * x575
                + x1315 * x809
                + x1317 * x430
                + x1329 * x883
                + x1330 * x883
                + x1331 * x575
                + x1332 * x809
                + x1333 * x430
                + x1334 * x929
            )
            + 2 * dq_i2 * dq_i7 * dq_j2 * x1108 * x135 * x138 * x141 * x143 * x145 * x148 * x201 * x361 * x893
            + 2
            * dq_i2
            * (
                x1312
                + x1314
                + x1316
                + x1318
                + x1319
                + x1331 * x583
                + x1332 * x850
                + x1333 * x437
                + x1336 * x910
                + x1337 * x935
            )
            + dq_i3
            * (
                x1082 * x944
                + x1082 * x945
                + x1114 * x946
                + x1114 * x947
                + x1117 * x949
                + x1117 * x950
                + x1227 * x32 * x938
                + x1268 * x944
                + x1268 * x945
                + x1302 * x946
                + x1302 * x947
                + x1304 * x949
                + x1304 * x950
                + x1306 * x942
                + x1306 * x943
                + x1307 * x942
                + x1307 * x943
                + x1329 * x940
                + x1330 * x940
                + x1330 * x941
            )
            + 2
            * dq_i4
            * (
                x1113 * x905
                + x1115 * x922
                + x1275 * x905
                + x1306 * x921
                + x1307 * x921
                + x1309 * x924
                + x1310 * x924
                + x1328 * x922
            )
            + 2 * dq_i5 * (x1083 * x917 + x1247 * x919 + x1298 * x902 + x1325 * x249 + x1326 * x252 + x1327 * x917)
            + 2 * dq_i6 * (x1247 * x914 + x1295 * x902 + x1325 * x214 + x1326 * x225)
            + 2 * dq_i7 * sigma_kin_v_7_2 * x135 * x138 * x141 * x143 * x145 * x148 * x187 * x201 * x361 * x893
            - x1174 * (x1217 * x905 + x1220 * x886 + x1222 * x902 + x1224 * x894 + x1227 * x882)
            - x1259 * x895
            - x1312 * x370
            - x1314 * x370
            - x1316 * x370
            - x1318 * x370
            - x1319 * x370
            - x1320 * x208
            - x1321 * x231
            - x1323 * x237
            - x1324 * x258
            - x495 * x911
        )
    )
    K_block_list.append(
        4 * ddq_i3 * dq_j3**2 * x1412
        + 4
        * ddq_j3
        * (
            ddq_i3 * (x1402 * x606 + x1404 * x572 + x1406 * x531 + x1408 * x425 + x1410 * x427)
            + x113 * x116 * x118 * x122 * x1406 * x319 * x529 * x624
            + x135 * x138 * x141 * x1410 * x143 * x145 * x149 * x325 * x361 * x624
            + x1402 * x310 * x42 * x50 * x604 * x624
            + x1404 * x315 * x570 * x624 * x63 * x72 * x76
            + x1408 * x159 * x162 * x164 * x166 * x170 * x322 * x423 * x624
            - x1410 * x596
            - x451 * (x1485 * x212 + x1487 * x223)
            - x550 * (x1484 * x249 + x1488 * x242 + x1489 * x251)
            - x589 * (x1406 * x552 + x1490 * x267 + x1491 * x278 + x1492 * x286)
            - x616 * (x1403 * x609 + x1493 * x575 + x1494 * x534 + x1495 * x430 + x1496 * x432)
            - x620 * (x1493 * x583 + x1494 * x541 + x1495 * x437 + x1496 * x439 + x1497 * x336)
            - x625 * (x1403 * x622 + x1405 * x317 + x1407 * x546 + x1409 * x444 + x1411 * x448)
        )
        - dq_i1 * x1374 * x1426
        - dq_i2 * x1374 * x1431
        - dq_i4 * x1374 * x1419
        - dq_i5 * x1374 * x1400
        - dq_i6 * x1374 * x1384
        + dq_j3 * x1097 * x912
        - sigma_kin_v_7_3 * x1048 * x1374 * x143 * x371 * x418 * x893
        - sigma_kin_v_7_3
        * x139
        * x1433
        * x396
        * (-x1132 * x374 + x1136 * x1432 + x1138 * x1432 + x1140 * x374 + x1142 * x134 + x375 * x477 + x483 * x640)
        + sigma_kin_v_7_3
        * x524
        * (
            -sigma_kin_v_7_3 * x1461 * x498
            + x1177
            - x1458
            + x1459 * x478
            + x1459 * x484
            + x1459 * x487
            + x1459 * x494
            + x1460 * x488
            + x1460 * x492
            + x181 * x502
            + x181 * x505
            + x507
        )
        + x1027
        * (
            2
            * dq_i1
            * (
                x1003 * x1599
                + x1130 * x1381
                + x1594 * x961
                + x1595 * x961
                + x1596 * x964
                + x1597 * x970
                + x1598 * x964
                + x1600 * x970
                + x1602 * x955
                + x1613 * x955
            )
            + dq_i2
            * (
                x1008 * x1497 * x38
                + x1010 * x1420
                + x1010 * x1592
                + x1011 * x618
                + x1014 * x1594
                + x1014 * x1595
                + x1015 * x1594
                + x1015 * x1595
                + x1018 * x1596
                + x1018 * x1598
                + x1020 * x1596
                + x1020 * x1598
                + x1022 * x1597
                + x1022 * x1600
                + x1024 * x1599
                + x1024 * x1616
                + x1025 * x1597
                + x1025 * x1600
                + x1026 * x1599
                + x1026 * x1616
            )
            + 2 * dq_i3 * dq_i7 * dq_j3 * x134 * x139 * x141 * x1410 * x143 * x145 * x148 * x201 * x361 * x966
            + 2
            * dq_i3
            * (
                x1430 * x996
                + x1538 * x988
                + x1558 * x994
                + x1569 * x981
                + x1603
                + x1604
                + x1605
                + x1606
                + x1613 * x992
                + x1615 * x996
            )
            + 2
            * dq_i4
            * (
                x1414 * x989
                + x1417 * x991
                + x1540 * x989
                + x1564 * x991
                + x1586 * x977
                + x1587 * x977
                + x1589 * x973
                + x1591 * x973
            )
            + 2 * dq_i5 * (x1557 * x982 + x1584 * x975 + x1585 * x975 + x1610 * x252 + x1611 * x249 + x1612 * x252)
            + 2 * dq_i6 * (x1554 * x982 + x1610 * x225 + x1611 * x214 + x1612 * x225)
            + 2 * dq_i7 * sigma_kin_v_7_3 * x134 * x139 * x141 * x143 * x145 * x148 * x187 * x201 * x361 * x966
            - x1095 * x263
            - x1430 * x377 * x995
            - x1461 * (x1493 * x977 + x1494 * x975 + x1495 * x973 + x1496 * x967 + x1497 * x956)
            - x1574 * x968
            - x1603 * x376
            - x1604 * x376
            - x1605 * x376
            - x1606 * x376
            - x1607 * x208
            - x1608 * x231
            - x1609 * x237
            - x495 * x979
        )
        + x1206
        * (
            ddq_i1 * x608
            + 2 * dq_i1 * dq_i3 * sigma_kin_v_3_1 * sigma_kin_v_3_3 * x310 * x42 * x50 * x604
            + 2 * dq_i1 * dq_i3 * sigma_kin_v_4_1 * sigma_kin_v_4_3 * x315 * x570 * x63 * x72 * x76
            + 2 * dq_i1 * dq_i3 * sigma_kin_v_5_1 * sigma_kin_v_5_3 * x113 * x116 * x118 * x122 * x319 * x529
            + 2 * dq_i1 * dq_i3 * sigma_kin_v_6_1 * sigma_kin_v_6_3 * x159 * x162 * x164 * x166 * x170 * x322 * x423
            + 2
            * dq_i1
            * dq_i3
            * sigma_kin_v_7_1
            * sigma_kin_v_7_3
            * x135
            * x138
            * x141
            * x143
            * x145
            * x149
            * x325
            * x361
            - x12 * x615
            - x435 * x600
            - x539 * x602
            - x581 * x603
            - x595 * x675
            - x616 * x623
            - x619 * x627
        )
        + x131
        * (
            dq_i1
            * (
                x1039 * x1599
                + x125 * x1596
                + x126 * x1397
                + x129 * x1596
                + x130 * x1397
                + x1379 * x174
                + x1379 * x178
                + x1381 * x191
                + x1420 * x55
                + x1420 * x59
                + x1428 * x80
                + x1428 * x84
                + x1574 * x410
                + x1594 * x79
                + x1594 * x83
                + x1597 * x173
                + x1597 * x177
                + x1599 * x190
                + x1601 * x54
                + x1601 * x58
            )
            + 2
            * dq_i2
            * (
                x1035 * x1594
                + x1035 * x1595
                + x1036 * x1596
                + x1036 * x1598
                + x1037 * x1597
                + x1037 * x1600
                + x1038 * x1599
                + x1427 * x254
                + x1574 * x359
                + x1593 * x254
            )
            + 2 * dq_i3 * dq_i7 * dq_j3 * x132 * x133 * x135 * x139 * x141 * x1410 * x143 * x145 * x148 * x201
            + 2
            * dq_i3
            * (
                x1033 * x1601
                + x1538 * x275
                + x1558 * x318
                + x1569 * x215
                + x1571
                + x1572
                + x1573
                + x1575
                + x1576 * x327
                + x1590 * x327
            )
            + 2
            * dq_i4
            * (
                x1414 * x276
                + x1540 * x276
                + x1576 * x288
                + x1586 * x233
                + x1587 * x233
                + x1589 * x204
                + x1590 * x288
                + x1591 * x204
            )
            + 2 * dq_i5 * (x1353 * x1505 + x1580 * x252 + x1581 * x249 + x1582 * x252 + x1584 * x227 + x1585 * x227)
            + 2 * dq_i6 * (x1030 * x1554 + x1580 * x225 + x1581 * x214 + x1582 * x225)
            + 2 * dq_i7 * sigma_kin_v_7_3 * x132 * x133 * x135 * x139 * x141 * x143 * x145 * x148 * x187 * x201
            - x1342 * x1574
            - x1348 * x263
            - x141 * x1550 * x1576
            - x1461 * (x137 * x1496 + x1403 * x254 + x1493 * x233 + x1494 * x227 + x1495 * x204)
            - x1571 * x376
            - x1572 * x376
            - x1573 * x376
            - x1575 * x376
            - x1577 * x208
            - x1578 * x231
            - x1579 * x237
            - x257 * x495
        )
        + x1366 * x940
        + x1366 * x941
        + x1367 * x942
        + x1367 * x943
        + x1368 * x87 * (sigma_pot_3_c * x49 + sigma_pot_3_s * x46)
        + x1369 * x944
        + x1369 * x945
        + x1371 * x951
        + x1372 * x946
        + x1372 * x947
        + x1373 * x949
        + x1373 * x950
        + x1378 * x951
        + x1389 * x951
        + x1396 * x951
        + x1401 * x951
        - 4 * x1444 * x38
        + x473
        * (
            ddq_i6 * x1447
            + 2 * dq_i3 * dq_i6 * sigma_kin_v_6_3 * sigma_kin_v_6_6 * x159 * x162 * x164 * x166 * x170 * x322 * x423
            + 2
            * dq_i3
            * dq_i6
            * sigma_kin_v_7_3
            * sigma_kin_v_7_6
            * x135
            * x138
            * x141
            * x143
            * x145
            * x149
            * x325
            * x361
            - x1376 * x422
            - x1450 * x435
            - x1451 * x441
            - x1452 * x451
            - x1453 * x459
            - x1454 * x465
            - x1455 * x472
        )
        + x561
        * (
            ddq_i5 * x1464
            + 2 * dq_i3 * dq_i5 * sigma_kin_v_5_3 * sigma_kin_v_5_5 * x113 * x116 * x118 * x122 * x319 * x529
            + 2 * dq_i3 * dq_i5 * sigma_kin_v_6_3 * sigma_kin_v_6_5 * x159 * x162 * x164 * x166 * x170 * x322 * x423
            + 2
            * dq_i3
            * dq_i5
            * sigma_kin_v_7_3
            * sigma_kin_v_7_5
            * x135
            * x138
            * x141
            * x143
            * x145
            * x149
            * x325
            * x361
            - x1387 * x526
            - x1462 * x465
            - x1468 * x539
            - x1469 * x543
            - x1470 * x550
            - x1471 * x554
            - x1472 * x560
        )
        + x594
        * (
            ddq_i4 * x1475
            + 2 * dq_i3 * dq_i4 * sigma_kin_v_4_3 * sigma_kin_v_4_4 * x315 * x570 * x63 * x72 * x76
            + 2 * dq_i3 * dq_i4 * sigma_kin_v_5_3 * sigma_kin_v_5_4 * x113 * x116 * x118 * x122 * x319 * x529
            + 2 * dq_i3 * dq_i4 * sigma_kin_v_6_3 * sigma_kin_v_6_4 * x159 * x162 * x164 * x166 * x170 * x322 * x423
            + 2
            * dq_i3
            * dq_i4
            * sigma_kin_v_7_3
            * sigma_kin_v_7_4
            * x135
            * x138
            * x141
            * x143
            * x145
            * x149
            * x325
            * x361
            - x1394 * x563
            - x1473 * x459
            - x1474 * x554
            - x1480 * x581
            - x1481 * x585
            - x1482 * x589
            - x1483 * x593
        )
        + x667
        * (
            ddq_i2 * x1200
            + 2 * dq_i2 * dq_i3 * sigma_kin_v_3_2 * sigma_kin_v_3_3 * x310 * x42 * x50 * x604
            + 2 * dq_i2 * dq_i3 * sigma_kin_v_4_2 * sigma_kin_v_4_3 * x315 * x570 * x63 * x72 * x76
            + 2 * dq_i2 * dq_i3 * sigma_kin_v_5_2 * sigma_kin_v_5_3 * x113 * x116 * x118 * x122 * x319 * x529
            + 2 * dq_i2 * dq_i3 * sigma_kin_v_6_2 * sigma_kin_v_6_3 * x159 * x162 * x164 * x166 * x170 * x322 * x423
            + 2
            * dq_i2
            * dq_i3
            * sigma_kin_v_7_2
            * sigma_kin_v_7_3
            * x135
            * x138
            * x141
            * x143
            * x145
            * x149
            * x325
            * x361
            - x1093 * x641
            - x1196 * x441
            - x1197 * x543
            - x1199 * x585
            - x1203 * x627
            - x1204 * x666
            - x1205 * x620
        )
        + x752
        * (
            2 * dq_i1 * (x1506 * x717 + x1509 * x432 + x1512 * x430 + x1515 * x432)
            + 2 * dq_i2 * (x1506 * x724 + x1509 * x439 + x1512 * x437 + x1515 * x439)
            + 2 * dq_i3 * dq_i7 * dq_j3 * x135 * x139 * x141 * x1410 * x143 * x144 * x148 * x201 * x361 * x697
            + 2 * dq_i3 * (x1499 + x1501 + x1506 * x444 + x1518 * x705)
            + 2 * dq_i4 * (x1382 * x732 + x1498 * x453 + x1506 * x453 + x1517 * x732)
            + 2 * dq_i5 * (x1519 * x735 + x1520 * x736 + x1521 * x735 + x1522 * x736)
            + dq_i6
            * (
                x1485 * x38 * x742
                + x1487 * x38 * x745
                + x1507 * x748
                + x1513 * x748
                + x1513 * x751
                + x1519 * x744
                + x1521 * x744
                + x1521 * x749
            )
            + 2 * dq_i7 * sigma_kin_v_7_3 * x135 * x139 * x141 * x143 * x144 * x148 * x187 * x201 * x361 * x697
            - sigma_kin_v_7_3 * x699
            - x1238 * x263
            - x1461 * (x1485 * x700 + x1487 * x697)
            - x1499 * x376
            - x1501 * x376
            - x1502 * x237
            - x1503 * x231
            - x1504 * x208
            - x495 * x710
        )
        + x804
        * (
            2 * dq_i1 * (x1532 * x534 + x1533 * x717 + x1534 * x534 + x1535 * x432 + x1536 * x430 + x1537 * x432)
            + 2 * dq_i2 * (x1532 * x541 + x1533 * x724 + x1534 * x541 + x1535 * x439 + x1536 * x437 + x1537 * x439)
            + 2 * dq_i3 * dq_i7 * dq_j3 * x135 * x139 * x141 * x1410 * x142 * x145 * x148 * x201 * x361 * x757
            + 2 * dq_i3 * (x1518 * x766 + x1524 + x1526 + x1527 + x1533 * x444 + x1538 * x784)
            + 2 * dq_i4 * (x1414 * x769 + x1525 * x453 + x1533 * x453 + x1540 * x769 + x1541 * x789 + x1542 * x789)
            + dq_i5
            * (
                x1398 * x795
                + x1398 * x797
                + x1488 * x38 * x793
                + x1519 * x796
                + x1519 * x798
                + x1521 * x796
                + x1521 * x798
                + x1531 * x795
                + x1543 * x800
                + x1543 * x802
                + x1544 * x800
                + x1544 * x802
            )
            + 2 * dq_i6 * (x1519 * x777 + x1520 * x778 + x1521 * x777 + x1522 * x778)
            + 2 * dq_i7 * sigma_kin_v_7_3 * x135 * x139 * x141 * x142 * x145 * x148 * x187 * x201 * x361 * x757
            - x1265 * x263
            - x1370 * x760
            - x1461 * (x1484 * x763 + x1488 * x753 + x1489 * x758)
            - x1524 * x376
            - x1526 * x376
            - x1527 * x376
            - x1528 * x208
            - x1529 * x237
            - x1530 * x231
            - x495 * x773
        )
        + x881
        * (
            2
            * dq_i1
            * (
                x1422 * x806
                + x1559 * x575
                + x1560 * x809
                + x1562 * x430
                + x1563 * x809
                + x1565 * x432
                + x1566 * x430
                + x1567 * x432
            )
            + 2
            * dq_i2
            * (
                x1429 * x806
                + x1559 * x583
                + x1560 * x850
                + x1562 * x437
                + x1563 * x850
                + x1565 * x439
                + x1566 * x437
                + x1567 * x439
            )
            + 2 * dq_i3 * dq_i7 * dq_j3 * x135 * x139 * x140 * x1410 * x143 * x145 * x148 * x201 * x361 * x815
            + 2
            * dq_i3
            * (x1418 * x859 + x1545 + x1547 + x1549 + x1556 * x546 + x1568 * x854 + x1569 * x857 + x1570 * x859)
            + dq_i4
            * (
                x1397 * x870
                + x1397 * x871
                + x1413 * x866
                + x1413 * x867
                + x1415 * x874
                + x1415 * x879
                + x1417 * x878
                + x1417 * x880
                + x1490 * x38 * x864
                + x1491 * x38 * x872
                + x1492 * x38 * x875
                + x1539 * x870
                + x1539 * x871
                + x1561 * x874
                + x1564 * x878
                + x1568 * x866
            )
            + 2 * dq_i5 * (x1541 * x842 + x1542 * x842 + x1546 * x556 + x1555 * x249 + x1556 * x556 + x1557 * x824)
            + 2 * dq_i6 * (x1382 * x835 + x1517 * x835 + x1554 * x824 + x1555 * x214)
            + 2 * dq_i7 * sigma_kin_v_7_3 * x135 * x139 * x140 * x143 * x145 * x148 * x187 * x201 * x361 * x815
            - x1293 * x263
            - x1370 * x817
            - x140 * x1417 * x1550 * x841
            - x1461 * (x1406 * x828 + x1490 * x805 + x1491 * x819 + x1492 * x815)
            - x1545 * x376
            - x1547 * x376
            - x1549 * x376
            - x1551 * x208
            - x1552 * x231
            - x1553 * x237
            - x495 * x832
        )
        + x951
        * (
            2 * dq_i1 * (x1426 + x1601 * x883 + x1614 * x929 + x1617 * x575 + x1618 * x809 + x1619 * x430)
            + 2 * dq_i2 * (x1431 + x1593 * x910 + x1615 * x935 + x1617 * x583 + x1618 * x850 + x1619 * x437)
            + 2 * dq_i3 * dq_i7 * dq_j3 * x135 * x138 * x141 * x1410 * x143 * x145 * x148 * x201 * x361 * x893
            - dq_i3 * x1434
            - dq_i3 * x1435
            - dq_i3 * x1436
            - dq_i3 * x1437
            - dq_i3 * x1438
            - dq_i3 * x1439
            - dq_i3 * x1440
            - dq_i3 * x1441
            - dq_i3 * x1442
            - dq_i3 * x1443
            + dq_i3
            * (
                x1444
                + x1531 * x944
                + x1531 * x945
                + x1561 * x946
                + x1561 * x947
                + x1564 * x949
                + x1564 * x950
                + x1568 * x942
                + x1568 * x943
                + x1601 * x940
                + x1601 * x941
            )
            + 2 * dq_i4 * (x1419 + x1540 * x905 + x1568 * x921 + x1570 * x924 + x1588 * x922)
            + 2 * dq_i5 * (x1400 + x1516 * x919 + x1557 * x902 + x1583 * x917)
            + 2 * dq_i6 * (x1384 + x1516 * x914 + x1554 * x902)
            + 2 * dq_i7 * sigma_kin_v_7_3 * x135 * x138 * x141 * x143 * x145 * x148 * x187 * x201 * x361 * x893
            - x1371
            - x1378
            - x1389
            - x1396
            - x1401
            - x1412 * x1461
            - x495 * x912
        )
    )
    K_block_list.append(
        4 * ddq_i4 * dq_j4**2 * x1646
        + 4
        * ddq_j4
        * (
            ddq_i4 * (x1639 * x572 + x1641 * x531 + x1642 * x425 + x1644 * x427)
            + x114 * x115 * x118 * x122 * x1641 * x272 * x529 * x592
            + x135 * x139 * x140 * x143 * x145 * x149 * x1644 * x286 * x361 * x592
            + x160 * x161 * x164 * x1642 * x166 * x170 * x278 * x423 * x592
            + x1639 * x267 * x570 * x592 * x64 * x71 * x76
            - x1644 * x563
            - x459 * (x1711 * x212 + x1713 * x223)
            - x554 * (x1710 * x249 + x1715 * x242 + x1716 * x251)
            - x581 * (x1717 * x575 + x1718 * x534 + x1719 * x430 + x1720 * x432)
            - x585 * (x1717 * x583 + x1718 * x541 + x1719 * x437 + x1720 * x439)
            - x589 * (x1640 * x317 + x1643 * x444 + x1645 * x448 + x1714 * x546)
            - x593 * (x1640 * x269 + x1641 * x552 + x1643 * x453 + x1645 * x456)
        )
        - dq_i1 * x1622 * x1659
        - dq_i2 * x1622 * x1660
        - dq_i3 * x1622 * x1664
        - dq_i5 * x1622 * x1651
        - dq_i6 * x1622 * x1631
        + dq_j4 * x1097 * x833
        - sigma_kin_v_7_4 * x1048 * x1622 * x198 * x385 * x841
        - sigma_kin_v_7_4
        * x141
        * x1672
        * x396
        * (x142 * x1666 + x142 * x1667 + x142 * x1670 - x1665 * x380 + x1668 * x384 + x1669 * x380 + x382 * x477)
        + sigma_kin_v_7_4
        * x524
        * (
            -sigma_kin_v_7_4 * x1698 * x498
            + x1693 * x1696
            - x1694
            + x1696 * x1697
            + x1696 * x478
            + x1696 * x484
            + x1696 * x487
            + x1696 * x494
            + x1699
            + x182 * x502
            + x182 * x505
            + x513
        )
        + x1027
        * (
            2
            * dq_i1
            * (
                x1003 * x1788
                + x1130 * x1629
                + x1783 * x961
                + x1784 * x961
                + x1785 * x964
                + x1786 * x970
                + x1787 * x964
                + x1789 * x970
            )
            + dq_i2
            * (
                x1014 * x1783
                + x1014 * x1784
                + x1015 * x1783
                + x1015 * x1784
                + x1018 * x1785
                + x1018 * x1787
                + x1020 * x1785
                + x1020 * x1787
                + x1022 * x1786
                + x1022 * x1789
                + x1024 * x1788
                + x1024 * x1811
                + x1025 * x1786
                + x1025 * x1789
                + x1026 * x1788
                + x1026 * x1811
            )
            + 2
            * dq_i3
            * (
                x1652 * x994
                + x1662 * x981
                + x1757 * x988
                + x1758 * x988
                + x1790 * x994
                + x1792 * x981
                + x1808 * x996
                + x1810 * x996
            )
            + 2 * dq_i4 * dq_i7 * dq_j4 * x134 * x139 * x141 * x143 * x145 * x148 * x1644 * x201 * x361 * x966
            + 2
            * dq_i4
            * (x1657 * x991 + x1760 * x989 + x1793 * x991 + x1795 * x977 + x1797 * x973 + x1798 + x1799 + x1800)
            + 2 * dq_i5 * (x1780 * x975 + x1782 * x975 + x1803 * x252 + x1804 * x249 + x1805 * x252 + x1806 * x982)
            + 2 * dq_i6 * (x1775 * x982 + x1803 * x225 + x1804 * x214 + x1805 * x225)
            + 2 * dq_i7 * sigma_kin_v_7_4 * x134 * x139 * x141 * x143 * x145 * x148 * x187 * x201 * x361 * x966
            - x1079 * x263
            - x1609 * x258
            - x1657 * x1772 * x967
            - x1698 * (x1717 * x977 + x1718 * x975 + x1719 * x973 + x1720 * x967)
            - x1767 * x968
            - x1798 * x378
            - x1799 * x378
            - x1800 * x378
            - x1801 * x208
            - x1802 * x231
            - x495 * x978
        )
        + x112 * x870
        + x112 * x871
        + x1206
        * (
            ddq_i1 * x574
            + 2 * dq_i1 * dq_i4 * sigma_kin_v_4_1 * sigma_kin_v_4_4 * x267 * x570 * x64 * x71 * x76
            + 2 * dq_i1 * dq_i4 * sigma_kin_v_5_1 * sigma_kin_v_5_4 * x114 * x115 * x118 * x122 * x272 * x529
            + 2 * dq_i1 * dq_i4 * sigma_kin_v_6_1 * sigma_kin_v_6_4 * x160 * x161 * x164 * x166 * x170 * x278 * x423
            + 2
            * dq_i1
            * dq_i4
            * sigma_kin_v_7_1
            * sigma_kin_v_7_4
            * x135
            * x139
            * x140
            * x143
            * x145
            * x149
            * x286
            * x361
            - x12 * x580
            - x435 * x567
            - x539 * x569
            - x562 * x675
            - x581 * x591
            - x584 * x627
            - x588 * x616
        )
        + x131
        * (
            dq_i1
            * (
                x1039 * x1788
                + x125 * x1785
                + x126 * x1647
                + x129 * x1785
                + x130 * x1647
                + x1627 * x174
                + x1627 * x178
                + x1629 * x191
                + x1652 * x80
                + x1652 * x84
                + x173 * x1786
                + x1767 * x410
                + x177 * x1786
                + x1783 * x79
                + x1783 * x83
                + x1788 * x190
            )
            + 2
            * dq_i2
            * (
                x1035 * x1783
                + x1035 * x1784
                + x1036 * x1785
                + x1036 * x1787
                + x1037 * x1786
                + x1037 * x1789
                + x1038 * x1788
                + x1767 * x359
            )
            + 2
            * dq_i3
            * (
                x1652 * x318
                + x1662 * x215
                + x1757 * x275
                + x1758 * x275
                + x1771 * x327
                + x1790 * x318
                + x1792 * x215
                + x1794 * x327
            )
            + 2 * dq_i4 * dq_i7 * dq_j4 * x132 * x133 * x135 * x139 * x141 * x143 * x145 * x148 * x1644 * x201
            + 2
            * dq_i4
            * (x1760 * x276 + x1765 + x1766 + x1770 + x1771 * x288 + x1794 * x288 + x1795 * x233 + x1797 * x204)
            + 2 * dq_i5 * (x1353 * x1725 + x1776 * x252 + x1777 * x249 + x1778 * x252 + x1780 * x227 + x1782 * x227)
            + 2 * dq_i6 * (x1030 * x1775 + x1776 * x225 + x1777 * x214 + x1778 * x225)
            + 2 * dq_i7 * sigma_kin_v_7_4 * x132 * x133 * x135 * x139 * x141 * x143 * x145 * x148 * x187 * x201
            - x1342 * x1767
            - x1347 * x263
            - x1579 * x258
            - x1698 * (x137 * x1720 + x1717 * x233 + x1718 * x227 + x1719 * x204)
            - x1765 * x378
            - x1766 * x378
            - x1770 * x378
            - x1771 * x1772
            - x1773 * x208
            - x1774 * x231
            - x236 * x495
        )
        + x1367 * x866
        + x1367 * x867
        + x1368 * x86 * (sigma_pot_4_c * x70 + sigma_pot_4_s * x67)
        + x1372 * x874
        + x1372 * x879
        + x1373 * x878
        + x1373 * x880
        + x1621 * x881
        + x1626 * x881
        + x1636 * x881
        + x1637 * x881
        + x1638 * x881
        - 4 * x1681 * x60
        + x473
        * (
            ddq_i6 * x1684
            + 2 * dq_i4 * dq_i6 * sigma_kin_v_6_4 * sigma_kin_v_6_6 * x160 * x161 * x164 * x166 * x170 * x278 * x423
            + 2
            * dq_i4
            * dq_i6
            * sigma_kin_v_7_4
            * sigma_kin_v_7_6
            * x135
            * x139
            * x140
            * x143
            * x145
            * x149
            * x286
            * x361
            - x1624 * x422
            - x1687 * x435
            - x1688 * x441
            - x1689 * x451
            - x1690 * x459
            - x1691 * x465
            - x1692 * x472
        )
        + x561
        * (
            ddq_i5 * x1701
            + 2 * dq_i4 * dq_i5 * sigma_kin_v_5_4 * sigma_kin_v_5_5 * x114 * x115 * x118 * x122 * x272 * x529
            + 2 * dq_i4 * dq_i5 * sigma_kin_v_6_4 * sigma_kin_v_6_5 * x160 * x161 * x164 * x166 * x170 * x278 * x423
            + 2
            * dq_i4
            * dq_i5
            * sigma_kin_v_7_4
            * sigma_kin_v_7_5
            * x135
            * x139
            * x140
            * x143
            * x145
            * x149
            * x286
            * x361
            - x1634 * x526
            - x1700 * x465
            - x1705 * x539
            - x1706 * x543
            - x1707 * x550
            - x1708 * x554
            - x1709 * x560
        )
        + x626
        * (
            ddq_i3 * x1475
            + 2 * dq_i3 * dq_i4 * sigma_kin_v_4_3 * sigma_kin_v_4_4 * x267 * x570 * x64 * x71 * x76
            + 2 * dq_i3 * dq_i4 * sigma_kin_v_5_3 * sigma_kin_v_5_4 * x114 * x115 * x118 * x122 * x272 * x529
            + 2 * dq_i3 * dq_i4 * sigma_kin_v_6_3 * sigma_kin_v_6_4 * x160 * x161 * x164 * x166 * x170 * x278 * x423
            + 2
            * dq_i3
            * dq_i4
            * sigma_kin_v_7_3
            * sigma_kin_v_7_4
            * x135
            * x139
            * x140
            * x143
            * x145
            * x149
            * x286
            * x361
            - x1394 * x596
            - x1473 * x451
            - x1474 * x550
            - x1480 * x616
            - x1481 * x620
            - x1482 * x625
            - x1483 * x589
        )
        + x667
        * (
            ddq_i2 * x1189
            + 2 * dq_i2 * dq_i4 * sigma_kin_v_4_2 * sigma_kin_v_4_4 * x267 * x570 * x64 * x71 * x76
            + 2 * dq_i2 * dq_i4 * sigma_kin_v_5_2 * sigma_kin_v_5_4 * x114 * x115 * x118 * x122 * x272 * x529
            + 2 * dq_i2 * dq_i4 * sigma_kin_v_6_2 * sigma_kin_v_6_4 * x160 * x161 * x164 * x166 * x170 * x278 * x423
            + 2
            * dq_i2
            * dq_i4
            * sigma_kin_v_7_2
            * sigma_kin_v_7_4
            * x135
            * x139
            * x140
            * x143
            * x145
            * x149
            * x286
            * x361
            - x1077 * x641
            - x1186 * x441
            - x1187 * x543
            - x1190 * x627
            - x1192 * x666
            - x1194 * x620
            - x1195 * x585
        )
        + x752
        * (
            2 * dq_i1 * (x1726 * x717 + x1729 * x432 + x1732 * x430 + x1734 * x432)
            + 2 * dq_i2 * (x1726 * x724 + x1729 * x439 + x1732 * x437 + x1734 * x439)
            + 2 * dq_i3 * (x1721 * x444 + x1726 * x444 + x1737 * x705 + x1738 * x705)
            + 2 * dq_i4 * dq_i7 * dq_j4 * x135 * x139 * x141 * x143 * x144 * x148 * x1644 * x201 * x361 * x697
            + 2 * dq_i4 * (x1630 * x732 + x1722 + x1726 * x453 + x1736 * x732)
            + 2 * dq_i5 * (x1739 * x735 + x1740 * x736 + x1741 * x735 + x1742 * x736)
            + dq_i6
            * (
                x1649 * x748
                + x1649 * x751
                + x1711 * x60 * x742
                + x1713 * x60 * x745
                + x1727 * x748
                + x1739 * x744
                + x1741 * x744
                + x1741 * x749
            )
            + 2 * dq_i7 * sigma_kin_v_7_4 * x135 * x139 * x141 * x143 * x144 * x148 * x187 * x201 * x361 * x697
            - sigma_kin_v_7_4 * x699
            - x1239 * x263
            - x1502 * x258
            - x1630 * x379 * x455 * x705
            - x1698 * (x1711 * x700 + x1713 * x697)
            - x1722 * x378
            - x1723 * x231
            - x1724 * x208
            - x495 * x711
        )
        + x804
        * (
            2 * dq_i1 * (x1750 * x534 + x1751 * x717 + x1753 * x534 + x1754 * x432 + x1755 * x430 + x1756 * x432)
            + 2 * dq_i2 * (x1750 * x541 + x1751 * x724 + x1753 * x541 + x1754 * x439 + x1755 * x437 + x1756 * x439)
            + 2 * dq_i3 * (x1737 * x766 + x1738 * x766 + x1745 * x444 + x1751 * x444 + x1757 * x784 + x1758 * x784)
            + 2 * dq_i4 * dq_i7 * dq_j4 * x135 * x139 * x141 * x142 * x145 * x148 * x1644 * x201 * x361 * x757
            + 2 * dq_i4 * (x1650 * x789 + x1744 + x1746 + x1751 * x453 + x1760 * x769 + x1761 * x789)
            + dq_i5
            * (
                x1715 * x60 * x793
                + x1739 * x796
                + x1739 * x798
                + x1741 * x796
                + x1741 * x798
                + x1749 * x795
                + x1752 * x795
                + x1752 * x797
                + x1762 * x800
                + x1762 * x802
                + x1763 * x800
                + x1763 * x802
            )
            + 2 * dq_i6 * (x1739 * x777 + x1740 * x778 + x1741 * x777 + x1742 * x778)
            + 2 * dq_i7 * sigma_kin_v_7_4 * x135 * x139 * x141 * x142 * x145 * x148 * x187 * x201 * x361 * x757
            - x1266 * x263
            - x1529 * x258
            - x1620 * x760
            - x1650 * x379 * x788
            - x1698 * (x1710 * x763 + x1715 * x753 + x1716 * x758)
            - x1744 * x378
            - x1746 * x378
            - x1747 * x208
            - x1748 * x231
            - x495 * x774
        )
        + x881
        * (
            2 * dq_i1 * (x1659 + x1828 * x575 + x1829 * x809 + x1830 * x430 + x1831 * x432)
            + 2 * dq_i2 * (x1660 + x1828 * x583 + x1829 * x850 + x1830 * x437 + x1831 * x439)
            + 2 * dq_i3 * (x1664 + x1792 * x857 + x1825 * x854 + x1826 * x859 + x1827 * x546)
            + 2 * dq_i4 * dq_i7 * dq_j4 * x135 * x139 * x140 * x143 * x145 * x148 * x1644 * x201 * x361 * x815
            - dq_i4 * x1673
            - dq_i4 * x1674
            - dq_i4 * x1675
            - dq_i4 * x1676
            - dq_i4 * x1677
            - dq_i4 * x1678
            - dq_i4 * x1679
            - dq_i4 * x1680
            + dq_i4
            * (
                x1681
                + x1759 * x870
                + x1759 * x871
                + x1791 * x874
                + x1791 * x879
                + x1793 * x878
                + x1793 * x880
                + x1825 * x866
                + x1825 * x867
            )
            + 2 * dq_i5 * (x1651 + x1761 * x842 + x1806 * x824 + x1827 * x556)
            + 2 * dq_i6 * (x1631 + x1736 * x835 + x1775 * x824)
            + 2 * dq_i7 * sigma_kin_v_7_4 * x135 * x139 * x140 * x143 * x145 * x148 * x187 * x201 * x361 * x815
            - x1621
            - x1626
            - x1636
            - x1637
            - x1638
            - x1646 * x1698
            - x495 * x833
        )
        + x951
        * (
            2
            * dq_i1
            * (
                x1807 * x929
                + x1809 * x929
                + x1819 * x575
                + x1820 * x575
                + x1821 * x809
                + x1822 * x430
                + x1823 * x809
                + x1824 * x430
            )
            + 2
            * dq_i2
            * (
                x1808 * x935
                + x1810 * x935
                + x1819 * x583
                + x1820 * x583
                + x1821 * x850
                + x1822 * x437
                + x1823 * x850
                + x1824 * x437
            )
            + dq_i3
            * (
                x1655 * x946
                + x1655 * x947
                + x1657 * x949
                + x1657 * x950
                + x1661 * x942
                + x1661 * x943
                + x1749 * x944
                + x1749 * x945
                + x1752 * x944
                + x1752 * x945
                + x1791 * x946
                + x1791 * x947
                + x1793 * x949
                + x1793 * x950
                + x1825 * x942
                + x1825 * x943
            )
            + 2 * dq_i4 * dq_i7 * dq_j4 * x135 * x138 * x141 * x143 * x145 * x148 * x1644 * x201 * x361 * x893
            + 2
            * dq_i4
            * (x1663 * x924 + x1760 * x905 + x1796 * x922 + x1812 + x1813 + x1814 + x1825 * x921 + x1826 * x924)
            + 2 * dq_i5 * (x1735 * x919 + x1779 * x917 + x1781 * x917 + x1806 * x902 + x1817 * x249 + x1818 * x252)
            + 2 * dq_i6 * (x1735 * x914 + x1775 * x902 + x1817 * x214 + x1818 * x225)
            + 2 * dq_i7 * sigma_kin_v_7_4 * x135 * x138 * x141 * x143 * x145 * x148 * x187 * x201 * x361 * x893
            - x1323 * x263
            - x1395 * x258
            - x1620 * x895
            - x1663 * x379 * x901
            - x1698 * (x1640 * x886 + x1643 * x902 + x1645 * x894 + x1714 * x905)
            - x1812 * x378
            - x1813 * x378
            - x1814 * x378
            - x1815 * x208
            - x1816 * x231
            - x495 * x909
        )
    )
    K_block_list.append(
        4 * ddq_i5 * dq_j5**2 * x1858
        + 4
        * ddq_j5
        * (
            ddq_i5 * (x1851 * x531 + x1853 * x425 + x1855 * x427)
            + x114 * x116 * x117 * x122 * x1851 * x242 * x529 * x559
            + x135 * x139 * x141 * x142 * x145 * x149 * x1855 * x250 * x361 * x559
            + x160 * x162 * x163 * x166 * x170 * x1853 * x247 * x423 * x559
            - x1855 * x526
            - x465 * (x1899 * x212 + x1900 * x223)
            - x539 * (x1901 * x534 + x1902 * x430 + x1903 * x432)
            - x543 * (x1901 * x541 + x1902 * x437 + x1903 * x439)
            - x550 * (x1852 * x546 + x1904 * x444 + x1905 * x448)
            - x554 * (x1851 * x552 + x1904 * x453 + x1905 * x456)
            - x560 * (x1852 * x556 + x1854 * x249 + x1857 * x251)
        )
        - dq_i1 * x1837 * x1867
        - dq_i2 * x1837 * x1868
        - dq_i3 * x1837 * x1873
        - dq_i4 * x1837 * x1876
        - dq_i6 * x1837 * x1847
        + dq_j5 * x1097 * x775
        - sigma_kin_v_7_5 * x1048 * x139 * x1837 * x380 * x418 * x757
        - sigma_kin_v_7_5
        * x143
        * x1878
        * x396
        * (x140 * x1666 + x140 * x1667 + x140 * x1670 - x1665 * x385 + x1668 * x379 + x1669 * x385 + x386 * x477)
        + sigma_kin_v_7_5
        * x524
        * (
            -sigma_kin_v_7_5 * x1898 * x498
            + x1693 * x1897
            + x1697 * x1897
            + x1699
            + x183 * x502
            + x183 * x505
            - x1896
            + x1897 * x478
            + x1897 * x484
            + x1897 * x487
            + x1897 * x494
            + x511
        )
        + x101 * x1832 * (sigma_pot_5_c * x93 + sigma_pot_5_s * x90)
        + x1027
        * (
            2 * dq_i1 * (x1003 * x1936 + x1130 * x1844 + x1933 * x964 + x1934 * x970 + x1935 * x964 + x1937 * x970)
            + dq_i2
            * (
                x1018 * x1933
                + x1018 * x1935
                + x1020 * x1933
                + x1020 * x1935
                + x1022 * x1934
                + x1022 * x1937
                + x1024 * x1936
                + x1024 * x1968
                + x1025 * x1934
                + x1025 * x1937
                + x1026 * x1936
                + x1026 * x1968
            )
            + 2 * dq_i3 * (x1869 * x988 + x1939 * x988 + x1941 * x981 + x1945 * x981 + x1964 * x996 + x1966 * x996)
            + 2 * dq_i4 * (x1874 * x989 + x1942 * x991 + x1946 * x991 + x1949 * x989 + x1951 * x973 + x1953 * x973)
            + 2 * dq_i5 * dq_i7 * dq_j5 * x134 * x139 * x141 * x143 * x145 * x148 * x1855 * x201 * x361 * x966
            + 2 * dq_i5 * (x1955 * x975 + x1956 + x1958 + x1960 + x1962 * x252 + x1967 * x982)
            + 2 * dq_i6 * (x1931 * x982 + x1957 * x214 + x1959 * x225 + x1962 * x225)
            + 2 * dq_i7 * sigma_kin_v_7_5 * x134 * x139 * x141 * x143 * x145 * x148 * x187 * x201 * x361 * x966
            - x1069 * x263
            - x1608 * x258
            - x1802 * x237
            - x1898 * (x1901 * x975 + x1902 * x973 + x1903 * x967)
            - x1925 * x968
            - x1956 * x383
            - x1958 * x383
            - x1960 * x383
            - x1961 * x208
            - x495 * x976
        )
        - 4 * x110 * x1886
        + x1206
        * (
            ddq_i1 * x533
            + 2 * dq_i1 * dq_i5 * sigma_kin_v_5_1 * sigma_kin_v_5_5 * x114 * x116 * x117 * x122 * x242 * x529
            + 2 * dq_i1 * dq_i5 * sigma_kin_v_6_1 * sigma_kin_v_6_5 * x160 * x162 * x163 * x166 * x170 * x247 * x423
            + 2
            * dq_i1
            * dq_i5
            * sigma_kin_v_7_1
            * sigma_kin_v_7_5
            * x135
            * x139
            * x141
            * x142
            * x145
            * x149
            * x250
            * x361
            - x12 * x538
            - x435 * x528
            - x525 * x675
            - x539 * x558
            - x542 * x627
            - x549 * x616
            - x553 * x581
        )
        + x131
        * (
            dq_i1
            * (
                x1039 * x1936
                + x125 * x1933
                + x126 * x1859
                + x129 * x1933
                + x130 * x1859
                + x173 * x1934
                + x174 * x1842
                + x177 * x1934
                + x178 * x1842
                + x1844 * x191
                + x190 * x1936
                + x1925 * x410
            )
            + 2 * dq_i2 * (x1036 * x1933 + x1036 * x1935 + x1037 * x1934 + x1037 * x1937 + x1038 * x1936 + x1925 * x359)
            + 2 * dq_i3 * (x1869 * x275 + x1939 * x275 + x1941 * x215 + x1943 * x327 + x1945 * x215 + x1947 * x327)
            + 2 * dq_i4 * (x1874 * x276 + x1943 * x288 + x1947 * x288 + x1949 * x276 + x1951 * x204 + x1953 * x204)
            + 2 * dq_i5 * dq_i7 * dq_j5 * x132 * x133 * x135 * x139 * x141 * x143 * x145 * x148 * x1855 * x201
            + 2 * dq_i5 * (x1353 * x1909 + x1924 + x1927 + x1929 + x1932 * x252 + x1955 * x227)
            + 2 * dq_i6 * (x1030 * x1931 + x1926 * x214 + x1928 * x225 + x1932 * x225)
            + 2 * dq_i7 * sigma_kin_v_7_5 * x132 * x133 * x135 * x139 * x141 * x143 * x145 * x148 * x187 * x201
            - x1342 * x1925
            - x1346 * x263
            - x1578 * x258
            - x1774 * x237
            - x1898 * (x137 * x1903 + x1901 * x227 + x1902 * x204)
            - x1924 * x383
            - x1927 * x383
            - x1929 * x383
            - x1930 * x208
            - x230 * x495
        )
        + x1369 * x795
        + x1369 * x797
        + x1834 * x804
        + x1835 * x796
        + x1835 * x798
        + x1836 * x800
        + x1836 * x802
        + x1841 * x804
        + x1848 * x804
        + x1849 * x804
        + x1850 * x804
        + x473
        * (
            ddq_i6 * x1887
            + 2 * dq_i5 * dq_i6 * sigma_kin_v_6_5 * sigma_kin_v_6_6 * x160 * x162 * x163 * x166 * x170 * x247 * x423
            + 2
            * dq_i5
            * dq_i6
            * sigma_kin_v_7_5
            * sigma_kin_v_7_6
            * x135
            * x139
            * x141
            * x142
            * x145
            * x149
            * x250
            * x361
            - x1839 * x422
            - x1890 * x435
            - x1891 * x441
            - x1892 * x451
            - x1893 * x459
            - x1894 * x465
            - x1895 * x472
        )
        + x594
        * (
            ddq_i4 * x1701
            + 2 * dq_i4 * dq_i5 * sigma_kin_v_5_4 * sigma_kin_v_5_5 * x114 * x116 * x117 * x122 * x242 * x529
            + 2 * dq_i4 * dq_i5 * sigma_kin_v_6_4 * sigma_kin_v_6_5 * x160 * x162 * x163 * x166 * x170 * x247 * x423
            + 2
            * dq_i4
            * dq_i5
            * sigma_kin_v_7_4
            * sigma_kin_v_7_5
            * x135
            * x139
            * x141
            * x142
            * x145
            * x149
            * x250
            * x361
            - x1634 * x563
            - x1700 * x459
            - x1705 * x581
            - x1706 * x585
            - x1707 * x589
            - x1708 * x593
            - x1709 * x554
        )
        + x626
        * (
            ddq_i3 * x1464
            + 2 * dq_i3 * dq_i5 * sigma_kin_v_5_3 * sigma_kin_v_5_5 * x114 * x116 * x117 * x122 * x242 * x529
            + 2 * dq_i3 * dq_i5 * sigma_kin_v_6_3 * sigma_kin_v_6_5 * x160 * x162 * x163 * x166 * x170 * x247 * x423
            + 2
            * dq_i3
            * dq_i5
            * sigma_kin_v_7_3
            * sigma_kin_v_7_5
            * x135
            * x139
            * x141
            * x142
            * x145
            * x149
            * x250
            * x361
            - x1387 * x596
            - x1462 * x451
            - x1468 * x616
            - x1469 * x620
            - x1470 * x625
            - x1471 * x589
            - x1472 * x550
        )
        + x667
        * (
            ddq_i2 * x1180
            + 2 * dq_i2 * dq_i5 * sigma_kin_v_5_2 * sigma_kin_v_5_5 * x114 * x116 * x117 * x122 * x242 * x529
            + 2 * dq_i2 * dq_i5 * sigma_kin_v_6_2 * sigma_kin_v_6_5 * x160 * x162 * x163 * x166 * x170 * x247 * x423
            + 2
            * dq_i2
            * dq_i5
            * sigma_kin_v_7_2
            * sigma_kin_v_7_5
            * x135
            * x139
            * x141
            * x142
            * x145
            * x149
            * x250
            * x361
            - x1067 * x641
            - x1178 * x441
            - x1181 * x627
            - x1182 * x666
            - x1183 * x620
            - x1184 * x585
            - x1185 * x543
        )
        + x752
        * (
            2 * dq_i1 * (x1910 * x717 + x1913 * x432 + x1914 * x430 + x1915 * x432)
            + 2 * dq_i2 * (x1910 * x724 + x1913 * x439 + x1914 * x437 + x1915 * x439)
            + 2 * dq_i3 * (x1872 * x705 + x1910 * x444 + x1918 * x705 + x1919 * x444)
            + 2 * dq_i4 * (x1871 * x732 + x1910 * x453 + x1917 * x732 + x1919 * x453)
            + 2 * dq_i5 * dq_i7 * dq_j5 * x135 * x139 * x141 * x143 * x144 * x148 * x1855 * x201 * x361 * x697
            + 2 * dq_i5 * (x1906 + x1907 + x1920 * x735 + x1921 * x736)
            + dq_i6
            * (
                x110 * x1899 * x742
                + x110 * x1900 * x745
                + x1843 * x744
                + x1843 * x749
                + x1845 * x748
                + x1845 * x751
                + x1911 * x748
                + x1920 * x744
            )
            + 2 * dq_i7 * sigma_kin_v_7_5 * x135 * x139 * x141 * x143 * x144 * x148 * x187 * x201 * x361 * x697
            - sigma_kin_v_7_5 * x699
            - x1240 * x263
            - x1503 * x258
            - x1723 * x237
            - x1898 * (x1899 * x700 + x1900 * x697)
            - x1906 * x383
            - x1907 * x383
            - x1908 * x208
            - x495 * x714
        )
        + x804
        * (
            2 * dq_i1 * (x1867 + x1994 * x534 + x1995 * x717 + x1996 * x432)
            + 2 * dq_i2 * (x1868 + x1994 * x541 + x1995 * x724 + x1996 * x439)
            + 2 * dq_i3 * (x1873 + x1918 * x766 + x1939 * x784 + x1995 * x444)
            + 2 * dq_i4 * (x1876 + x1949 * x769 + x1993 * x789 + x1995 * x453)
            + 2 * dq_i5 * dq_i7 * dq_j5 * x135 * x139 * x141 * x142 * x145 * x148 * x1855 * x201 * x361 * x757
            - dq_i5 * x1879
            - dq_i5 * x1880
            - dq_i5 * x1881
            - dq_i5 * x1882
            - dq_i5 * x1884
            - dq_i5 * x1885
            + dq_i5 * (x1886 + x1920 * x796 + x1920 * x798 + x1938 * x795 + x1938 * x797 + x1997 * x800 + x1997 * x802)
            + 2 * dq_i6 * (x1847 + x1920 * x777 + x1921 * x778)
            + 2 * dq_i7 * sigma_kin_v_7_5 * x135 * x139 * x141 * x142 * x145 * x148 * x187 * x201 * x361 * x757
            - x1834
            - x1841
            - x1848
            - x1849
            - x1850
            - x1858 * x1898
            - x495 * x775
        )
        + x881
        * (
            2 * dq_i1 * (x1987 * x809 + x1988 * x430 + x1989 * x809 + x1990 * x432 + x1991 * x430 + x1992 * x432)
            + 2 * dq_i2 * (x1987 * x850 + x1988 * x437 + x1989 * x850 + x1990 * x439 + x1991 * x437 + x1992 * x439)
            + 2 * dq_i3 * (x1941 * x857 + x1945 * x857 + x1979 * x859 + x1980 * x859 + x1981 * x546 + x1986 * x546)
            + dq_i4
            * (
                x1859 * x870
                + x1859 * x871
                + x1940 * x874
                + x1940 * x879
                + x1942 * x878
                + x1942 * x880
                + x1944 * x874
                + x1944 * x879
                + x1946 * x878
                + x1946 * x880
                + x1948 * x870
                + x1948 * x871
            )
            + 2 * dq_i5 * dq_i7 * dq_j5 * x135 * x139 * x140 * x143 * x145 * x148 * x1855 * x201 * x361 * x815
            + 2 * dq_i5 * (x1875 * x842 + x1967 * x824 + x1982 + x1984 + x1986 * x556 + x1993 * x842)
            + 2 * dq_i6 * (x1871 * x835 + x1917 * x835 + x1931 * x824 + x1983 * x214)
            + 2 * dq_i7 * sigma_kin_v_7_5 * x135 * x139 * x140 * x143 * x145 * x148 * x187 * x201 * x361 * x815
            - x1291 * x263
            - x1552 * x258
            - x1635 * x237
            - x1833 * x817
            - x1875 * x384 * x841
            - x1898 * (x1851 * x828 + x1904 * x824 + x1905 * x816)
            - x1982 * x383
            - x1984 * x383
            - x1985 * x208
            - x495 * x829
        )
        + x951
        * (
            2 * dq_i1 * (x1963 * x929 + x1965 * x929 + x1975 * x809 + x1976 * x430 + x1977 * x809 + x1978 * x430)
            + 2 * dq_i2 * (x1964 * x935 + x1966 * x935 + x1975 * x850 + x1976 * x437 + x1977 * x850 + x1978 * x437)
            + dq_i3
            * (
                x1860 * x944
                + x1860 * x945
                + x1938 * x944
                + x1938 * x945
                + x1940 * x946
                + x1940 * x947
                + x1942 * x949
                + x1942 * x950
                + x1944 * x946
                + x1944 * x947
                + x1946 * x949
                + x1946 * x950
            )
            + 2 * dq_i4 * (x1874 * x905 + x1949 * x905 + x1950 * x922 + x1952 * x922 + x1979 * x924 + x1980 * x924)
            + 2 * dq_i5 * dq_i7 * dq_j5 * x135 * x138 * x141 * x143 * x145 * x148 * x1855 * x201 * x361 * x893
            + 2 * dq_i5 * (x1916 * x919 + x1954 * x917 + x1967 * x902 + x1969 + x1971 + x1973)
            + 2 * dq_i6 * (x1916 * x914 + x1931 * x902 + x1970 * x214 + x1972 * x225)
            + 2 * dq_i7 * sigma_kin_v_7_5 * x135 * x138 * x141 * x143 * x145 * x148 * x187 * x201 * x361 * x893
            - x1321 * x263
            - x1388 * x258
            - x1816 * x237
            - x1833 * x895
            - x1898 * (x1852 * x905 + x1904 * x902 + x1905 * x894)
            - x1969 * x383
            - x1971 * x383
            - x1973 * x383
            - x1974 * x208
            - x495 * x907
        )
    )
    K_block_list.append(
        4 * ddq_i6 * dq_j6**2 * x2010
        + 4
        * ddq_j6
        * (
            ddq_i6 * (x2006 * x425 + x2008 * x427)
            + x135 * x139 * x141 * x143 * x144 * x149 * x2008 * x223 * x361 * x471
            + x160 * x162 * x164 * x165 * x170 * x2006 * x212 * x423 * x471
            - x2008 * x422
            - x435 * (x2037 * x430 + x2038 * x432)
            - x441 * (x2037 * x437 + x2038 * x439)
            - x451 * (x2039 * x444 + x2040 * x448)
            - x459 * (x2039 * x453 + x2040 * x456)
            - x465 * (x2007 * x249 + x2041 * x251)
            - x472 * (x2007 * x214 + x2009 * x468)
        )
        - dq_i1 * x2000 * x2018
        - dq_i2 * x2000 * x2019
        - dq_i3 * x2000 * x2023
        - dq_i4 * x2000 * x2024
        - dq_i5 * x2000 * x2027
        + dq_j6 * x1097 * x715
        - sigma_kin_v_7_6
        * x145
        * x2036
        * x396
        * (
            -ddq_i7 * x493
            + x2033 * x373
            + x2033 * x374 * x377
            + x2034 * x488
            + x2034 * x492
            + x2035 * x393
            + x387 * x477
        )
        + sigma_kin_v_7_6
        * x524
        * (
            -sigma_kin_v_7_6 * x2045 * x498
            + x1176
            + x1693 * x2044
            + x1697 * x2044
            + x184 * x502
            + x184 * x505
            - x2043
            + x2044 * x478
            + x2044 * x484
            + x2044 * x487
            + x2044 * x494
            + x507
            + x509
            + x511
            + x513
            + x522
        )
        + x1027
        * (
            2 * dq_i1 * (x1003 * x2052 + x1130 * x2001 + x2051 * x970 + x2053 * x970)
            + dq_i2
            * (
                x1022 * x2051
                + x1022 * x2053
                + x1024 * x2052
                + x1024 * x2080
                + x1025 * x2051
                + x1025 * x2053
                + x1026 * x2052
                + x1026 * x2080
            )
            + 2 * dq_i3 * (x2055 * x981 + x2059 * x981 + x2075 * x996 + x2077 * x996)
            + 2 * dq_i4 * (x2056 * x991 + x2060 * x991 + x2063 * x973 + x2065 * x973)
            + 2 * dq_i5 * (x2071 * x249 + x2073 * x252 + x2078 * x982 + x2079 * x252)
            + 2 * dq_i6 * dq_i7 * dq_j6 * x134 * x139 * x141 * x143 * x145 * x148 * x2008 * x201 * x361 * x966
            + 2 * dq_i6 * (x2070 * x982 + x2072 + x2073 * x225 + x2079 * x225)
            + 2 * dq_i7 * sigma_kin_v_7_6 * x134 * x139 * x141 * x143 * x145 * x148 * x187 * x201 * x361 * x966
            - x1053 * x263
            - x1607 * x258
            - x1801 * x237
            - x1961 * x231
            - x2045 * (x2037 * x973 + x2038 * x967)
            - x2046 * x968
            - x2050 * x2073
            - x2072 * x388
            - x495 * x974
        )
        + x1206
        * (
            ddq_i1 * x429
            + 2 * dq_i1 * dq_i6 * sigma_kin_v_6_1 * sigma_kin_v_6_6 * x160 * x162 * x164 * x165 * x170 * x212 * x423
            + 2
            * dq_i1
            * dq_i6
            * sigma_kin_v_7_1
            * sigma_kin_v_7_6
            * x135
            * x139
            * x141
            * x143
            * x144
            * x149
            * x223
            * x361
            - x12 * x434
            - x415 * x675
            - x435 * x470
            - x440 * x627
            - x450 * x616
            - x458 * x581
            - x464 * x539
        )
        + x131
        * (
            dq_i1
            * (
                x1039 * x2052
                + x173 * x2051
                + x174 * x2011
                + x177 * x2051
                + x178 * x2011
                + x190 * x2052
                + x191 * x2001
                + x2046 * x410
            )
            + 2 * dq_i2 * (x1037 * x2051 + x1037 * x2053 + x1038 * x2052 + x2046 * x359)
            + 2 * dq_i3 * (x2055 * x215 + x2057 * x327 + x2059 * x215 + x2061 * x327)
            + 2 * dq_i4 * (x204 * x2063 + x204 * x2065 + x2057 * x288 + x2061 * x288)
            + 2 * dq_i5 * (x1353 * x2066 + x2047 * x249 + x2049 * x252 + x2069 * x252)
            + 2 * dq_i6 * dq_i7 * dq_j6 * x132 * x133 * x135 * x139 * x141 * x143 * x145 * x148 * x2008 * x201
            + 2 * dq_i6 * (x1030 * x2070 + x2048 + x2049 * x225 + x2069 * x225)
            + 2 * dq_i7 * sigma_kin_v_7_6 * x132 * x133 * x135 * x139 * x141 * x143 * x145 * x148 * x187 * x201
            - x1342 * x2046
            - x1345 * x263
            - x1577 * x258
            - x1773 * x237
            - x1930 * x231
            - x2045 * (x137 * x2038 + x2037 * x204)
            - x2048 * x388
            - x2049 * x2050
            - x207 * x495
        )
        - x139 * x2000 * x2001 * x203 * x708
        - 4 * x156 * x2032
        + x1832 * x94 * (sigma_pot_6_c * x100 + sigma_pot_6_s * x97)
        + x1835 * x744
        + x1835 * x749
        + x1998 * x752
        + x1999 * x748
        + x1999 * x751
        + x2002 * x752
        + x2003 * x752
        + x2004 * x752
        + x2005 * x752
        + x561
        * (
            ddq_i5 * x1887
            + 2 * dq_i5 * dq_i6 * sigma_kin_v_6_5 * sigma_kin_v_6_6 * x160 * x162 * x164 * x165 * x170 * x212 * x423
            + 2
            * dq_i5
            * dq_i6
            * sigma_kin_v_7_5
            * sigma_kin_v_7_6
            * x135
            * x139
            * x141
            * x143
            * x144
            * x149
            * x223
            * x361
            - x1839 * x526
            - x1890 * x539
            - x1891 * x543
            - x1892 * x550
            - x1893 * x554
            - x1894 * x560
            - x1895 * x465
        )
        + x594
        * (
            ddq_i4 * x1684
            + 2 * dq_i4 * dq_i6 * sigma_kin_v_6_4 * sigma_kin_v_6_6 * x160 * x162 * x164 * x165 * x170 * x212 * x423
            + 2
            * dq_i4
            * dq_i6
            * sigma_kin_v_7_4
            * sigma_kin_v_7_6
            * x135
            * x139
            * x141
            * x143
            * x144
            * x149
            * x223
            * x361
            - x1624 * x563
            - x1687 * x581
            - x1688 * x585
            - x1689 * x589
            - x1690 * x593
            - x1691 * x554
            - x1692 * x459
        )
        + x626
        * (
            ddq_i3 * x1447
            + 2 * dq_i3 * dq_i6 * sigma_kin_v_6_3 * sigma_kin_v_6_6 * x160 * x162 * x164 * x165 * x170 * x212 * x423
            + 2
            * dq_i3
            * dq_i6
            * sigma_kin_v_7_3
            * sigma_kin_v_7_6
            * x135
            * x139
            * x141
            * x143
            * x144
            * x149
            * x223
            * x361
            - x1376 * x596
            - x1450 * x616
            - x1451 * x620
            - x1452 * x625
            - x1453 * x589
            - x1454 * x550
            - x1455 * x451
        )
        + x667
        * (
            ddq_i2 * x1161
            + 2 * dq_i2 * dq_i6 * sigma_kin_v_6_2 * sigma_kin_v_6_6 * x160 * x162 * x164 * x165 * x170 * x212 * x423
            + 2
            * dq_i2
            * dq_i6
            * sigma_kin_v_7_2
            * sigma_kin_v_7_6
            * x135
            * x139
            * x141
            * x143
            * x144
            * x149
            * x223
            * x361
            - x1051 * x641
            - x1162 * x627
            - x1163 * x666
            - x1164 * x620
            - x1165 * x585
            - x1166 * x543
            - x1167 * x441
        )
        + x752
        * (
            2 * dq_i1 * (x2018 + x2110 * x717 + x2111 * x432)
            + 2 * dq_i2 * (x2019 + x2110 * x724 + x2111 * x439)
            + 2 * dq_i3 * (x2023 + x2104 * x705 + x2110 * x444)
            + 2 * dq_i4 * (x2024 + x2098 * x732 + x2110 * x453)
            + 2 * dq_i5 * (x2027 + x2106 * x735 + x2107 * x736)
            + 2 * dq_i6 * dq_i7 * dq_j6 * x135 * x139 * x141 * x143 * x144 * x148 * x2008 * x201 * x361 * x697
            - dq_i6 * x2028
            - dq_i6 * x2029
            - dq_i6 * x2030
            - dq_i6 * x2031
            + dq_i6 * (x2032 + x2067 * x748 + x2067 * x751 + x2106 * x744 + x2106 * x749)
            + 2 * dq_i7 * sigma_kin_v_7_6 * x135 * x139 * x141 * x143 * x144 * x148 * x187 * x201 * x361 * x697
            - x1998
            - x2002
            - x2003
            - x2004
            - x2005
            - x2010 * x2045
            - x495 * x715
        )
        + x804
        * (
            2 * dq_i1 * (x2100 * x717 + x2101 * x432 + x2102 * x430 + x2103 * x432)
            + 2 * dq_i2 * (x2100 * x724 + x2101 * x439 + x2102 * x437 + x2103 * x439)
            + 2 * dq_i3 * (x2022 * x766 + x2100 * x444 + x2104 * x766 + x2105 * x444)
            + 2 * dq_i4 * (x2096 * x789 + x2097 * x789 + x2100 * x453 + x2105 * x453)
            + dq_i5
            * (
                x2025 * x796
                + x2025 * x798
                + x2106 * x796
                + x2106 * x798
                + x2108 * x800
                + x2108 * x802
                + x2109 * x800
                + x2109 * x802
            )
            + 2 * dq_i6 * dq_i7 * dq_j6 * x135 * x139 * x141 * x142 * x145 * x148 * x2008 * x201 * x361 * x757
            + 2 * dq_i6 * (x2026 * x778 + x2099 + x2106 * x777 + x2107 * x778)
            + 2 * dq_i7 * sigma_kin_v_7_6 * x135 * x139 * x141 * x142 * x145 * x148 * x187 * x201 * x361 * x757
            - x1264 * x263
            - x1528 * x258
            - x1747 * x237
            - x1840 * x231
            - x2026 * x389 * x758
            - x2045 * (x2007 * x763 + x2041 * x758)
            - x2081 * x760
            - x2099 * x388
            - x495 * x768
        )
        + x881
        * (
            2 * dq_i1 * (x2092 * x430 + x2093 * x432 + x2094 * x430 + x2095 * x432)
            + 2 * dq_i2 * (x2092 * x437 + x2093 * x439 + x2094 * x437 + x2095 * x439)
            + 2 * dq_i3 * (x2055 * x857 + x2059 * x857 + x2087 * x859 + x2088 * x859)
            + dq_i4
            * (
                x2054 * x874
                + x2054 * x879
                + x2056 * x878
                + x2056 * x880
                + x2058 * x874
                + x2058 * x879
                + x2060 * x878
                + x2060 * x880
            )
            + 2 * dq_i5 * (x2078 * x824 + x2090 * x249 + x2096 * x842 + x2097 * x842)
            + 2 * dq_i6 * dq_i7 * dq_j6 * x135 * x139 * x140 * x143 * x145 * x148 * x2008 * x201 * x361 * x815
            + 2 * dq_i6 * (x2021 * x835 + x2070 * x824 + x2091 + x2098 * x835)
            + 2 * dq_i7 * sigma_kin_v_7_6 * x135 * x139 * x140 * x143 * x145 * x148 * x187 * x201 * x361 * x815
            - x1290 * x263
            - x1551 * x258
            - x1625 * x237
            - x1985 * x231
            - x2021 * x2050 * x816
            - x2045 * (x2039 * x824 + x2040 * x816)
            - x2081 * x817
            - x2091 * x388
            - x495 * x826
        )
        + x951
        * (
            2 * dq_i1 * (x2074 * x929 + x2076 * x929 + x2085 * x430 + x2086 * x430)
            + 2 * dq_i2 * (x2075 * x935 + x2077 * x935 + x2085 * x437 + x2086 * x437)
            + dq_i3
            * (
                x2054 * x946
                + x2054 * x947
                + x2056 * x949
                + x2056 * x950
                + x2058 * x946
                + x2058 * x947
                + x2060 * x949
                + x2060 * x950
            )
            + 2 * dq_i4 * (x2062 * x922 + x2064 * x922 + x2087 * x924 + x2088 * x924)
            + 2 * dq_i5 * (x2078 * x902 + x2082 * x249 + x2084 * x252 + x2089 * x919)
            + 2 * dq_i6 * dq_i7 * dq_j6 * x135 * x138 * x141 * x143 * x145 * x148 * x2008 * x201 * x361 * x893
            + 2 * dq_i6 * (x2070 * x902 + x2083 + x2084 * x225 + x2089 * x914)
            + 2 * dq_i7 * sigma_kin_v_7_6 * x135 * x138 * x141 * x143 * x145 * x148 * x187 * x201 * x361 * x893
            - x1320 * x263
            - x1377 * x258
            - x1815 * x237
            - x1974 * x231
            - x2045 * (x2039 * x902 + x2040 * x894)
            - x2050 * x2084
            - x2081 * x895
            - x2083 * x388
            - x495 * x904
        )
    )
    K_block_list.append(
        ddq_i7 * dq_j7**2 * x2112 * x2115 * x474
        - ddq_j7
        * x2115
        * x497
        * (dq_i7 * x484 + dq_i7 * x487 + x1139 * x493 + x2132 + x2133 * x488 + x2133 * x492 + x366 * x368 * x477 - x519)
        - dq_i1 * x2119 * x2121 * x432
        - dq_i2 * x1456 * x2119 * x520 * x934
        - dq_i3 * x143 * x2120 * x2122 * x325 * x371
        - 8 * dq_i6 * x139 * x2121 * x468
        + sigma_kin_v_7_1 * x1097 * x2113
        - sigma_kin_v_7_1
        * x1206
        * x2131
        * (-ddq_i1 * x499 + x12 * x2130 + x2125 * x627 + x2126 * x616 + x2127 * x581 + x2128 * x539 + x2129 * x435)
        - sigma_kin_v_7_2
        * x2131
        * x667
        * (-ddq_i2 * x499 + x2125 * x666 + x2126 * x620 + x2127 * x585 + x2128 * x543 + x2129 * x441 + x2130 * x627)
        - sigma_kin_v_7_3
        * x2131
        * x626
        * (-ddq_i3 * x499 + x2125 * x620 + x2126 * x625 + x2127 * x589 + x2128 * x550 + x2129 * x451 + x2130 * x616)
        - sigma_kin_v_7_4
        * x2131
        * x594
        * (-ddq_i4 * x499 + x2125 * x585 + x2126 * x589 + x2127 * x593 + x2128 * x554 + x2129 * x459 + x2130 * x581)
        - sigma_kin_v_7_5
        * x2131
        * x561
        * (-ddq_i5 * x499 + x2125 * x543 + x2126 * x550 + x2127 * x554 + x2128 * x560 + x2129 * x465 + x2130 * x539)
        - sigma_kin_v_7_6
        * x2131
        * x473
        * (-ddq_i6 * x499 + x2125 * x441 + x2126 * x451 + x2127 * x459 + x2128 * x465 + x2129 * x472 + x2130 * x435)
        + 1.0 * x101 * x1041 * x85 * x86 * x87 * x94 * (sigma_pot_7_c * x107 + sigma_pot_7_s * x104)
        + x1027
        * x1144
        * (
            x1168 * x2148
            - x1175 * x2146
            + x1457 * x966
            + x1693 * x2149
            + x1693 * x2151
            + x1697 * x2149
            + x1697 * x2151
            + x185 * x2148 * x483
            + x2042 * x966
            - x2140 * x2147
            + x2143 * x2147
            + x2144 * x2145
            + x2144 * x2150
            - x2146 * x506
            - x2146 * x508
            - x2146 * x510
            - x2146 * x512
            - x2146 * x514
            + x2149 * x478
            + x2149 * x487
            + x2149 * x494
            + x2151 * x478
        )
        + x131
        * x395
        * (
            -x1175 * x2138
            + x1693 * x2142
            + x1693 * x355
            + x1697 * x2142
            + x1697 * x355
            + x192 * x2141 * x2142
            + x2136 * x2137
            + x2136 * x518
            - x2138 * x506
            - x2138 * x508
            - x2138 * x510
            - x2138 * x512
            - x2138 * x514
            - x2139 * x2140
            + x2139 * x2143
            + x2141 * x409
            + x2142 * x484
            + x2142 * x487
            + x2142 * x494
            + x355 * x484
            + x355 * x487
            + x355 * x494
        )
        - x139 * x2122 * x2124 * x250 * x380 * x520
        + x1433
        * x951
        * (
            x1169 * x893
            - x1175 * x2153
            + x1456 * x2155
            + x1693 * x2156
            + x1693 * x2157
            + x1697 * x2156
            + x1697 * x2157
            + x185 * x2155 * x374
            + x2042 * x893
            - x2140 * x2154
            + x2143 * x2154
            + x2145 * x2152
            + x2150 * x2152
            - x2153 * x506
            - x2153 * x508
            - x2153 * x510
            - x2153 * x512
            - x2153 * x514
            + x2156 * x478
            + x2156 * x484
            + x2156 * x494
            + x2157 * x478
        )
        + x153
        * x523
        * (
            x1170
            + x1176
            + x1458
            + x153 * x391 * x497 * x503
            + x1693 * x2134
            + x1694
            + x1697 * x2134
            + x1896
            + x2043
            + x2132 * x520
            + x2134 * x478
            + x2134 * x484
            + x2134 * x487
            + x2134 * x494
            + x479
            + x516
            - 2 * x521
        )
        + x1672
        * x881
        * (
            dq_i4 * x185 * x2161
            + x1169 * x815
            - x1175 * x2159
            + x1457 * x815
            + x1697 * x2162
            + x1697 * x2163
            + x2042 * x815
            + x2123 * x2161
            - x2140 * x2160
            + x2143 * x2160
            + x2145 * x2158
            + x2150 * x2158
            - x2159 * x506
            - x2159 * x508
            - x2159 * x510
            - x2159 * x512
            - x2159 * x514
            + x2162 * x478
            + x2162 * x484
            + x2162 * x487
            + x2162 * x494
            + x2163 * x478
        )
        + x1878
        * x804
        * (
            dq_i5 * x185 * x2167
            + x1169 * x757
            - x1175 * x2165
            + x1457 * x757
            + x1693 * x2168
            + x1693 * x2169
            + x2042 * x757
            + x2124 * x2167
            - x2140 * x2166
            + x2143 * x2166
            + x2145 * x2164
            + x2150 * x2164
            - x2165 * x506
            - x2165 * x508
            - x2165 * x510
            - x2165 * x512
            - x2165 * x514
            + x2168 * x478
            + x2168 * x484
            + x2168 * x487
            + x2168 * x494
            + x2169 * x478
        )
        - 8 * x198 * x2123 * x286 * x385 * x416 * x520
        + x2036
        * x752
        * (
            x1169 * x697
            - x1175 * x2171
            + x1457 * x697
            + x1693 * x2173
            + x1693 * x2174
            + x1697 * x2173
            + x1697 * x2174
            + x2137 * x2172
            - x2140 * x698
            + x2143 * x698
            + x2145 * x2170
            + x2150 * x2170
            - x2171 * x506
            - x2171 * x508
            - x2171 * x510
            - x2171 * x512
            - x2171 * x514
            + x2172 * x518
            + x2173 * x478
            + x2173 * x484
            + x2173 * x487
            + x2174 * x478
        )
        + x2114 * x506
        + x2114 * x508
        + x2114 * x510
        + x2114 * x512
        + x2114 * x514
        + x2116 * x2117
        + x2116 * x500
        - x2117 * x2118
        - x2118 * x500
    )

    return K_block_list
