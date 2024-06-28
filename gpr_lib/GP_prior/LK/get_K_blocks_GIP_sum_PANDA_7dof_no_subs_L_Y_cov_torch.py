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

    x0 = dq_i1**2
    x1 = sigma_kin_v_1_1**2
    x2 = torch.sin(q_j1)
    x3 = torch.cos(q_i1)
    x4 = sigma_kin_p_1_1_c * x3
    x5 = torch.cos(q_j1)
    x6 = torch.sin(q_i1)
    x7 = sigma_kin_p_1_1_s * x6
    x8 = sigma_kin_p_1_1_off + x2 * x7 + x4 * x5
    x9 = x1 * x8 * (x2 * x4 - x5 * x7)
    x10 = torch.cos(q_i2)
    x11 = torch.cos(q_j2)
    x12 = x10 * x11
    x13 = torch.sin(q_i2)
    x14 = torch.sin(q_j2)
    x15 = x13 * x14
    x16 = sigma_kin_p_2_2_c * x12 + sigma_kin_p_2_2_off + sigma_kin_p_2_2_s * x15
    x17 = x16**2
    x18 = sigma_kin_p_2_1_c * x3
    x19 = sigma_kin_p_2_1_s * x6
    x20 = sigma_kin_p_2_1_off + x18 * x5 + x19 * x2
    x21 = x17 * x20 * (x18 * x2 - x19 * x5)
    x22 = dq_i1 * dq_j1
    x23 = dq_i2 * dq_j2
    x24 = sigma_kin_v_2_1 * x22 + sigma_kin_v_2_2 * x23
    x25 = 2 * x24**2
    x26 = sigma_kin_p_3_1_c * x3
    x27 = sigma_kin_p_3_1_s * x6
    x28 = sigma_kin_p_3_1_off + x2 * x27 + x26 * x5
    x29 = sigma_kin_p_3_2_c * x12 + sigma_kin_p_3_2_off + sigma_kin_p_3_2_s * x15
    x30 = x29**2
    x31 = torch.cos(q_i3)
    x32 = torch.cos(q_j3)
    x33 = x31 * x32
    x34 = torch.sin(q_i3)
    x35 = torch.sin(q_j3)
    x36 = x34 * x35
    x37 = sigma_kin_p_3_3_c * x33 + sigma_kin_p_3_3_off + sigma_kin_p_3_3_s * x36
    x38 = x37**2
    x39 = x30 * x38
    x40 = x28 * x39 * (x2 * x26 - x27 * x5)
    x41 = dq_i3 * dq_j3
    x42 = sigma_kin_v_3_1 * x22 + sigma_kin_v_3_2 * x23 + sigma_kin_v_3_3 * x41
    x43 = 2 * x42**2
    x44 = sigma_kin_p_4_1_c * x3
    x45 = sigma_kin_p_4_1_s * x6
    x46 = sigma_kin_p_4_1_off + x2 * x45 + x44 * x5
    x47 = sigma_kin_p_4_2_c * x12 + sigma_kin_p_4_2_off + sigma_kin_p_4_2_s * x15
    x48 = x47**2
    x49 = sigma_kin_p_4_3_c * x33 + sigma_kin_p_4_3_off + sigma_kin_p_4_3_s * x36
    x50 = x49**2
    x51 = torch.cos(q_i4)
    x52 = torch.cos(q_j4)
    x53 = x51 * x52
    x54 = torch.sin(q_i4)
    x55 = torch.sin(q_j4)
    x56 = x54 * x55
    x57 = sigma_kin_p_4_4_c * x53 + sigma_kin_p_4_4_off + sigma_kin_p_4_4_s * x56
    x58 = x57**2
    x59 = x48 * x50 * x58
    x60 = x46 * x59 * (x2 * x44 - x45 * x5)
    x61 = dq_i4 * dq_j4
    x62 = sigma_kin_v_4_1 * x22 + sigma_kin_v_4_2 * x23 + sigma_kin_v_4_3 * x41 + sigma_kin_v_4_4 * x61
    x63 = 2 * x62**2
    x64 = sigma_pot_1_c * x3
    x65 = sigma_pot_1_s * x6
    x66 = sigma_pot_2_c * x12 + sigma_pot_2_off + sigma_pot_2_s * x15
    x67 = sigma_pot_3_c * x33 + sigma_pot_3_off + sigma_pot_3_s * x36
    x68 = sigma_pot_4_c * x53 + sigma_pot_4_off + sigma_pot_4_s * x56
    x69 = torch.cos(q_i5)
    x70 = torch.cos(q_j5)
    x71 = x69 * x70
    x72 = torch.sin(q_i5)
    x73 = torch.sin(q_j5)
    x74 = x72 * x73
    x75 = sigma_pot_5_c * x71 + sigma_pot_5_off + sigma_pot_5_s * x74
    x76 = torch.cos(q_i6)
    x77 = torch.cos(q_j6)
    x78 = x76 * x77
    x79 = torch.sin(q_i6)
    x80 = torch.sin(q_j6)
    x81 = x79 * x80
    x82 = sigma_pot_6_c * x78 + sigma_pot_6_off + sigma_pot_6_s * x81
    x83 = torch.cos(q_i7)
    x84 = torch.cos(q_j7)
    x85 = x83 * x84
    x86 = torch.sin(q_i7)
    x87 = torch.sin(q_j7)
    x88 = x86 * x87
    x89 = sigma_pot_7_c * x85 + sigma_pot_7_off + sigma_pot_7_s * x88
    x90 = 1.0 * x67 * x68 * x75 * x82 * x89
    x91 = sigma_kin_p_5_1_c * x3
    x92 = sigma_kin_p_5_1_s * x6
    x93 = sigma_kin_p_5_1_off + x2 * x92 + x5 * x91
    x94 = sigma_kin_p_5_2_c * x12 + sigma_kin_p_5_2_off + sigma_kin_p_5_2_s * x15
    x95 = x94**2
    x96 = sigma_kin_p_5_3_c * x33 + sigma_kin_p_5_3_off + sigma_kin_p_5_3_s * x36
    x97 = x96**2
    x98 = sigma_kin_p_5_4_c * x53 + sigma_kin_p_5_4_off + sigma_kin_p_5_4_s * x56
    x99 = x98**2
    x100 = sigma_kin_p_5_5_c * x71 + sigma_kin_p_5_5_off + sigma_kin_p_5_5_s * x74
    x101 = x100**2
    x102 = x101 * x95 * x97 * x99
    x103 = x102 * x93 * (x2 * x91 - x5 * x92)
    x104 = dq_i5 * dq_j5
    x105 = (
        sigma_kin_v_5_1 * x22
        + sigma_kin_v_5_2 * x23
        + sigma_kin_v_5_3 * x41
        + sigma_kin_v_5_4 * x61
        + sigma_kin_v_5_5 * x104
    )
    x106 = 2 * x105**2
    x107 = dq_i7 * sigma_kin_v_7_7
    x108 = 2 * ddq_j7
    x109 = x107 * x108
    x110 = sigma_kin_p_7_1_c * x3
    x111 = sigma_kin_p_7_1_s * x6
    x112 = sigma_kin_p_7_1_off + x110 * x5 + x111 * x2
    x113 = x112**2
    x114 = sigma_kin_p_7_2_c * x12 + sigma_kin_p_7_2_off + sigma_kin_p_7_2_s * x15
    x115 = x114**2
    x116 = sigma_kin_p_7_3_c * x33 + sigma_kin_p_7_3_off + sigma_kin_p_7_3_s * x36
    x117 = x116**2
    x118 = sigma_kin_p_7_4_c * x53 + sigma_kin_p_7_4_off + sigma_kin_p_7_4_s * x56
    x119 = x118**2
    x120 = sigma_kin_p_7_5_c * x71 + sigma_kin_p_7_5_off + sigma_kin_p_7_5_s * x74
    x121 = x120**2
    x122 = sigma_kin_p_7_6_c * x78 + sigma_kin_p_7_6_off + sigma_kin_p_7_6_s * x81
    x123 = x122**2
    x124 = sigma_kin_p_7_7_c * x85 + sigma_kin_p_7_7_off + sigma_kin_p_7_7_s * x88
    x125 = x124**2
    x126 = x115 * x117 * x119 * x121 * x123 * x125
    x127 = x113 * x126
    x128 = sigma_kin_v_7_1 * x127
    x129 = dq_i1 * x128
    x130 = sigma_kin_p_6_1_c * x3
    x131 = sigma_kin_p_6_1_s * x6
    x132 = sigma_kin_p_6_1_off + x130 * x5 + x131 * x2
    x133 = sigma_kin_p_6_2_c * x12 + sigma_kin_p_6_2_off + sigma_kin_p_6_2_s * x15
    x134 = x133**2
    x135 = sigma_kin_p_6_3_c * x33 + sigma_kin_p_6_3_off + sigma_kin_p_6_3_s * x36
    x136 = x135**2
    x137 = sigma_kin_p_6_4_c * x53 + sigma_kin_p_6_4_off + sigma_kin_p_6_4_s * x56
    x138 = x137**2
    x139 = sigma_kin_p_6_5_c * x71 + sigma_kin_p_6_5_off + sigma_kin_p_6_5_s * x74
    x140 = x139**2
    x141 = sigma_kin_p_6_6_c * x78 + sigma_kin_p_6_6_off + sigma_kin_p_6_6_s * x81
    x142 = x141**2
    x143 = x134 * x136 * x138 * x140 * x142
    x144 = x132 * x143 * (x130 * x2 - x131 * x5)
    x145 = dq_i6 * dq_j6
    x146 = (
        sigma_kin_v_6_1 * x22
        + sigma_kin_v_6_2 * x23
        + sigma_kin_v_6_3 * x41
        + sigma_kin_v_6_4 * x61
        + sigma_kin_v_6_5 * x104
        + sigma_kin_v_6_6 * x145
    )
    x147 = 2 * x146**2
    x148 = x112 * x126 * (x110 * x2 - x111 * x5)
    x149 = dq_j7 * x107
    x150 = (
        sigma_kin_v_7_1 * x22
        + sigma_kin_v_7_2 * x23
        + sigma_kin_v_7_3 * x41
        + sigma_kin_v_7_4 * x61
        + sigma_kin_v_7_5 * x104
        + sigma_kin_v_7_6 * x145
        + x149
    )
    x151 = 2 * x150**2
    x152 = 4 * dq_i1
    x153 = sigma_kin_v_7_1 * x150
    x154 = x113 * x115 * x117 * x119 * x121
    x155 = x153 * x154
    x156 = x83 * x87
    x157 = x84 * x86
    x158 = x123 * x124 * (sigma_kin_p_7_7_c * x156 - sigma_kin_p_7_7_s * x157)
    x159 = 2 * ddq_j6
    x160 = dq_i6 * x159
    x161 = x132**2
    x162 = x143 * x161
    x163 = sigma_kin_v_6_1 * x162
    x164 = dq_i1 * (sigma_kin_v_6_6 * x163 + sigma_kin_v_7_6 * x128)
    x165 = sigma_kin_v_6_1 * x146
    x166 = x76 * x80
    x167 = x77 * x79
    x168 = x134 * x136 * x138 * x140 * x141 * x161 * (sigma_kin_p_6_6_c * x166 - sigma_kin_p_6_6_s * x167)
    x169 = x122 * x125 * (sigma_kin_p_7_6_c * x166 - sigma_kin_p_7_6_s * x167)
    x170 = 2 * ddq_j5
    x171 = dq_i5 * x170
    x172 = x93**2
    x173 = x102 * x172
    x174 = sigma_kin_v_5_1 * x173
    x175 = dq_i1 * (sigma_kin_v_5_5 * x174 + sigma_kin_v_6_5 * x163 + sigma_kin_v_7_5 * x128)
    x176 = 2 * ddq_j4
    x177 = dq_i4 * x176
    x178 = x46**2
    x179 = x178 * x59
    x180 = sigma_kin_v_4_1 * x179
    x181 = dq_i1 * (sigma_kin_v_4_4 * x180 + sigma_kin_v_5_4 * x174 + sigma_kin_v_6_4 * x163 + sigma_kin_v_7_4 * x128)
    x182 = sigma_kin_v_5_1 * x105
    x183 = x172 * x95 * x97
    x184 = x182 * x183
    x185 = x69 * x73
    x186 = x70 * x72
    x187 = x100 * x99 * (sigma_kin_p_5_5_c * x185 - sigma_kin_p_5_5_s * x186)
    x188 = x134 * x136 * x142 * x161
    x189 = x165 * x188
    x190 = x138 * x139 * (sigma_kin_p_6_5_c * x185 - sigma_kin_p_6_5_s * x186)
    x191 = x113 * x115 * x117 * x123 * x125
    x192 = x153 * x191
    x193 = x119 * x120 * (sigma_kin_p_7_5_c * x185 - sigma_kin_p_7_5_s * x186)
    x194 = 2 * ddq_j3
    x195 = dq_i3 * x194
    x196 = x28**2
    x197 = x196 * x39
    x198 = sigma_kin_v_3_1 * x197
    x199 = dq_i1 * (
        sigma_kin_v_3_3 * x198
        + sigma_kin_v_4_3 * x180
        + sigma_kin_v_5_3 * x174
        + sigma_kin_v_6_3 * x163
        + sigma_kin_v_7_3 * x128
    )
    x200 = 2 * ddq_j2
    x201 = dq_i2 * x200
    x202 = x20**2
    x203 = x17 * x202
    x204 = dq_i1 * (
        sigma_kin_v_2_1 * sigma_kin_v_2_2 * x203
        + sigma_kin_v_3_2 * x198
        + sigma_kin_v_4_2 * x180
        + sigma_kin_v_5_2 * x174
        + sigma_kin_v_6_2 * x163
        + sigma_kin_v_7_2 * x128
    )
    x205 = 2 * ddq_j1
    x206 = sigma_kin_v_4_1 * x62
    x207 = x51 * x55
    x208 = x52 * x54
    x209 = x178 * x48 * x50 * x57 * (sigma_kin_p_4_4_c * x207 - sigma_kin_p_4_4_s * x208)
    x210 = x101 * x98 * (sigma_kin_p_5_4_c * x207 - sigma_kin_p_5_4_s * x208)
    x211 = x137 * x140 * (sigma_kin_p_6_4_c * x207 - sigma_kin_p_6_4_s * x208)
    x212 = x118 * x121 * (sigma_kin_p_7_4_c * x207 - sigma_kin_p_7_4_s * x208)
    x213 = sigma_kin_v_3_1 * x42
    x214 = x196 * x213
    x215 = x31 * x35
    x216 = x32 * x34
    x217 = x30 * x37 * (sigma_kin_p_3_3_c * x215 - sigma_kin_p_3_3_s * x216)
    x218 = x178 * x58
    x219 = x206 * x218
    x220 = x48 * x49 * (sigma_kin_p_4_3_c * x215 - sigma_kin_p_4_3_s * x216)
    x221 = x101 * x172 * x99
    x222 = x182 * x221
    x223 = x95 * x96 * (sigma_kin_p_5_3_c * x215 - sigma_kin_p_5_3_s * x216)
    x224 = x138 * x140 * x142 * x161
    x225 = x165 * x224
    x226 = x134 * x135 * (sigma_kin_p_6_3_c * x215 - sigma_kin_p_6_3_s * x216)
    x227 = x113 * x119 * x121 * x123 * x125
    x228 = x153 * x227
    x229 = x115 * x116 * (sigma_kin_p_7_3_c * x215 - sigma_kin_p_7_3_s * x216)
    x230 = sigma_kin_v_2_1 * x24
    x231 = x10 * x14
    x232 = x11 * x13
    x233 = x16 * x202 * (sigma_kin_p_2_2_c * x231 - sigma_kin_p_2_2_s * x232)
    x234 = x29 * x38 * (sigma_kin_p_3_2_c * x231 - sigma_kin_p_3_2_s * x232)
    x235 = x47 * x50 * (sigma_kin_p_4_2_c * x231 - sigma_kin_p_4_2_s * x232)
    x236 = x94 * x97 * (sigma_kin_p_5_2_c * x231 - sigma_kin_p_5_2_s * x232)
    x237 = x133 * x136 * (sigma_kin_p_6_2_c * x231 - sigma_kin_p_6_2_s * x232)
    x238 = x114 * x117 * (sigma_kin_p_7_2_c * x231 - sigma_kin_p_7_2_s * x232)
    x239 = x196 * x234
    x240 = x218 * x235
    x241 = sigma_pot_1_off + x2 * x65 + x5 * x64
    x242 = x221 * x236
    x243 = sigma_kin_v_7_2 * x127
    x244 = x224 * x237
    x245 = x227 * x238
    x246 = 4 * dq_i2
    x247 = sigma_kin_v_7_2 * x150
    x248 = x154 * x158
    x249 = dq_j7 * x248
    x250 = sigma_kin_v_6_2 * x162
    x251 = sigma_kin_v_6_6 * x250 + sigma_kin_v_7_6 * x243
    x252 = sigma_kin_v_6_2 * x146
    x253 = x154 * x169
    x254 = sigma_kin_v_5_2 * x173
    x255 = sigma_kin_v_5_5 * x254 + sigma_kin_v_6_5 * x250 + sigma_kin_v_7_5 * x243
    x256 = sigma_kin_v_4_2 * x179
    x257 = sigma_kin_v_4_4 * x256 + sigma_kin_v_5_4 * x254 + sigma_kin_v_6_4 * x250 + sigma_kin_v_7_4 * x243
    x258 = sigma_kin_v_5_2 * x105
    x259 = x183 * x258
    x260 = x188 * x252
    x261 = x191 * x247
    x262 = (
        sigma_kin_v_3_2 * sigma_kin_v_3_3 * x197
        + sigma_kin_v_4_3 * x256
        + sigma_kin_v_5_3 * x254
        + sigma_kin_v_6_3 * x250
        + sigma_kin_v_7_3 * x243
    )
    x263 = sigma_kin_v_4_2 * x62
    x264 = sigma_kin_v_3_2 * x42
    x265 = x196 * x264
    x266 = x218 * x263
    x267 = x221 * x258
    x268 = x224 * x252
    x269 = x227 * x247
    x270 = sigma_kin_v_2_2 * x24
    x271 = x196 * x217
    x272 = x218 * x220
    x273 = 1.0 * x241 * x66 * x75 * x82 * x89
    x274 = x221 * x223
    x275 = sigma_kin_v_7_3 * x127
    x276 = x224 * x226
    x277 = x227 * x229
    x278 = 4 * dq_i3
    x279 = sigma_kin_v_7_3 * x150
    x280 = sigma_kin_v_6_3 * x162
    x281 = sigma_kin_v_6_6 * x280 + sigma_kin_v_7_6 * x275
    x282 = sigma_kin_v_6_3 * x146
    x283 = sigma_kin_v_5_3 * x173
    x284 = sigma_kin_v_5_5 * x283 + sigma_kin_v_6_5 * x280 + sigma_kin_v_7_5 * x275
    x285 = (
        sigma_kin_v_4_3 * sigma_kin_v_4_4 * x179
        + sigma_kin_v_5_4 * x283
        + sigma_kin_v_6_4 * x280
        + sigma_kin_v_7_4 * x275
    )
    x286 = sigma_kin_v_5_3 * x105
    x287 = x183 * x286
    x288 = x188 * x282
    x289 = x191 * x279
    x290 = sigma_kin_v_4_3 * x62
    x291 = sigma_kin_v_3_3 * x42
    x292 = x183 * x210
    x293 = sigma_kin_v_7_4 * x127
    x294 = x188 * x211
    x295 = x191 * x212
    x296 = 4 * dq_i4
    x297 = sigma_kin_v_7_4 * x150
    x298 = sigma_kin_v_6_4 * x162
    x299 = sigma_kin_v_6_6 * x298 + sigma_kin_v_7_6 * x293
    x300 = sigma_kin_v_6_4 * x146
    x301 = sigma_kin_v_5_4 * sigma_kin_v_5_5 * x173 + sigma_kin_v_6_5 * x298 + sigma_kin_v_7_5 * x293
    x302 = sigma_kin_v_5_4 * x105
    x303 = x183 * x302
    x304 = x188 * x300
    x305 = x191 * x297
    x306 = sigma_kin_v_4_4 * x62
    x307 = 1.0 * x241 * x66 * x67 * x68 * x89
    x308 = x183 * x187
    x309 = sigma_kin_v_7_5 * x127
    x310 = x188 * x190
    x311 = x191 * x193
    x312 = 4 * dq_i5
    x313 = sigma_kin_v_7_5 * x150
    x314 = sigma_kin_v_6_5 * sigma_kin_v_6_6 * x162 + sigma_kin_v_7_6 * x309
    x315 = sigma_kin_v_6_5 * x146
    x316 = sigma_kin_v_5_5 * x105
    x317 = x108 * x127
    x318 = sigma_kin_v_7_6 * x107
    x319 = 4 * dq_i6
    x320 = sigma_kin_v_7_6 * x150
    x321 = sigma_kin_v_6_6 * x146
    x322 = 4 * x150
    x323 = x107 * x322

    K_block_list = []
    K_block_list.append(
        2 * dq_j1**2 * x0 * x9
        - dq_j2 * x152 * (x214 * x234 + x219 * x235 + x222 * x236 + x225 * x237 + x228 * x238 + x230 * x233)
        - dq_j3 * x152 * (x214 * x217 + x219 * x220 + x222 * x223 + x225 * x226 + x228 * x229)
        - dq_j4 * x152 * (x184 * x210 + x189 * x211 + x192 * x212 + x206 * x209)
        - dq_j5 * x152 * (x184 * x187 + x189 * x190 + x192 * x193)
        - dq_j6 * x152 * (x155 * x169 + x165 * x168)
        - dq_j7 * x152 * x155 * x158
        + x0
        * x205
        * (
            sigma_kin_v_2_1**2 * x203
            + sigma_kin_v_3_1**2 * x197
            + sigma_kin_v_4_1**2 * x179
            + sigma_kin_v_5_1**2 * x173
            + sigma_kin_v_6_1**2 * x162
            + sigma_kin_v_7_1**2 * x127
            + x1 * x8**2
        )
        + x103 * x106
        + x109 * x129
        + x144 * x147
        + x148 * x151
        + x160 * x164
        + x171 * x175
        + x177 * x181
        + x195 * x199
        + x201 * x204
        + x21 * x25
        - 4 * x22 * (x103 * x182 + x144 * x165 + x148 * x153 + x206 * x60 + x21 * x230 + x213 * x40 + x22 * x9)
        + x40 * x43
        + x60 * x63
        + x66 * x90 * (x2 * x64 - x5 * x65)
    )
    K_block_list.append(
        dq_i2**2
        * x200
        * (
            sigma_kin_v_2_2**2 * x203
            + sigma_kin_v_3_2**2 * x197
            + sigma_kin_v_4_2**2 * x179
            + sigma_kin_v_5_2**2 * x173
            + sigma_kin_v_6_2**2 * x162
            + sigma_kin_v_7_2**2 * x127
        )
        + dq_i2 * x109 * x243
        + dq_i2 * x160 * x251
        + dq_i2 * x171 * x255
        + dq_i2 * x177 * x257
        + dq_i2 * x195 * x262
        + dq_i2 * x204 * x205
        - dq_j1 * x246 * (x103 * x258 + x144 * x252 + x148 * x247 + x21 * x270 + x263 * x60 + x264 * x40)
        - dq_j3 * x246 * (x217 * x265 + x220 * x266 + x223 * x267 + x226 * x268 + x229 * x269)
        - dq_j4 * x246 * (x209 * x263 + x210 * x259 + x211 * x260 + x212 * x261)
        - dq_j5 * x246 * (x187 * x259 + x190 * x260 + x193 * x261)
        - dq_j6 * x246 * (x168 * x252 + x247 * x253)
        + x106 * x242
        + x147 * x244
        + x151 * x245
        - 4 * x23 * (x233 * x270 + x234 * x265 + x235 * x266 + x236 * x267 + x237 * x268 + x238 * x269)
        + x233 * x25
        + x239 * x43
        + x240 * x63
        + x241 * x90 * (sigma_pot_2_c * x231 - sigma_pot_2_s * x232)
        - x246 * x247 * x249
    )
    K_block_list.append(
        dq_i3**2
        * x194
        * (
            sigma_kin_v_3_3**2 * x197
            + sigma_kin_v_4_3**2 * x179
            + sigma_kin_v_5_3**2 * x173
            + sigma_kin_v_6_3**2 * x162
            + sigma_kin_v_7_3**2 * x127
        )
        + dq_i3 * x109 * x275
        + dq_i3 * x160 * x281
        + dq_i3 * x171 * x284
        + dq_i3 * x177 * x285
        + dq_i3 * x199 * x205
        + dq_i3 * x201 * x262
        - dq_j1 * x278 * (x103 * x286 + x144 * x282 + x148 * x279 + x290 * x60 + x291 * x40)
        - dq_j2 * x278 * (x239 * x291 + x240 * x290 + x242 * x286 + x244 * x282 + x245 * x279)
        - dq_j4 * x278 * (x209 * x290 + x210 * x287 + x211 * x288 + x212 * x289)
        - dq_j5 * x278 * (x187 * x287 + x190 * x288 + x193 * x289)
        - dq_j6 * x278 * (x168 * x282 + x253 * x279)
        + x106 * x274
        + x147 * x276
        + x151 * x277
        - x249 * x278 * x279
        + x271 * x43
        + x272 * x63
        + x273 * x68 * (sigma_pot_3_c * x215 - sigma_pot_3_s * x216)
        - 4 * x41 * (x271 * x291 + x272 * x290 + x274 * x286 + x276 * x282 + x277 * x279)
    )
    K_block_list.append(
        dq_i4**2
        * x176
        * (
            sigma_kin_v_4_4**2 * x179
            + sigma_kin_v_5_4**2 * x173
            + sigma_kin_v_6_4**2 * x162
            + sigma_kin_v_7_4**2 * x127
        )
        + dq_i4 * x109 * x293
        + dq_i4 * x160 * x299
        + dq_i4 * x171 * x301
        + dq_i4 * x181 * x205
        + dq_i4 * x195 * x285
        + dq_i4 * x201 * x257
        - dq_j1 * x296 * (x103 * x302 + x144 * x300 + x148 * x297 + x306 * x60)
        - dq_j2 * x296 * (x240 * x306 + x242 * x302 + x244 * x300 + x245 * x297)
        - dq_j3 * x296 * (x272 * x306 + x274 * x302 + x276 * x300 + x277 * x297)
        - dq_j5 * x296 * (x187 * x303 + x190 * x304 + x193 * x305)
        - dq_j6 * x296 * (x168 * x300 + x253 * x297)
        + x106 * x292
        + x147 * x294
        + x151 * x295
        + x209 * x63
        - x249 * x296 * x297
        + x273 * x67 * (sigma_pot_4_c * x207 - sigma_pot_4_s * x208)
        - 4 * x61 * (x209 * x306 + x210 * x303 + x211 * x304 + x212 * x305)
    )
    K_block_list.append(
        dq_i5**2 * x170 * (sigma_kin_v_5_5**2 * x173 + sigma_kin_v_6_5**2 * x162 + sigma_kin_v_7_5**2 * x127)
        + dq_i5 * x109 * x309
        + dq_i5 * x160 * x314
        + dq_i5 * x175 * x205
        + dq_i5 * x177 * x301
        + dq_i5 * x195 * x284
        + dq_i5 * x201 * x255
        - dq_j1 * x312 * (x103 * x316 + x144 * x315 + x148 * x313)
        - dq_j2 * x312 * (x242 * x316 + x244 * x315 + x245 * x313)
        - dq_j3 * x312 * (x274 * x316 + x276 * x315 + x277 * x313)
        - dq_j4 * x312 * (x292 * x316 + x294 * x315 + x295 * x313)
        - dq_j6 * x312 * (x168 * x315 + x253 * x313)
        - 4 * x104 * (x308 * x316 + x310 * x315 + x311 * x313)
        + x106 * x308
        + x147 * x310
        + x151 * x311
        - x249 * x312 * x313
        + x307 * x82 * (sigma_pot_5_c * x185 - sigma_pot_5_s * x186)
    )
    K_block_list.append(
        dq_i6**2 * x159 * (sigma_kin_v_6_6**2 * x162 + sigma_kin_v_7_6**2 * x127)
        + dq_i6 * x164 * x205
        + dq_i6 * x171 * x314
        + dq_i6 * x177 * x299
        + dq_i6 * x195 * x281
        + dq_i6 * x201 * x251
        + dq_i6 * x317 * x318
        - dq_j1 * x319 * (x144 * x321 + x148 * x320)
        - dq_j2 * x319 * (x244 * x321 + x245 * x320)
        - dq_j3 * x319 * (x276 * x321 + x277 * x320)
        - dq_j4 * x319 * (x294 * x321 + x295 * x320)
        - dq_j5 * x319 * (x310 * x321 + x311 * x320)
        - 4 * x145 * (x168 * x321 + x253 * x320)
        + x147 * x168
        + x151 * x253
        - x249 * x319 * x320
        + x307 * x75 * (sigma_pot_6_c * x166 - sigma_pot_6_s * x167)
    )
    K_block_list.append(
        dq_i7**2 * sigma_kin_v_7_7**2 * x317
        - dq_j1 * x148 * x323
        - dq_j2 * x245 * x323
        - dq_j3 * x277 * x323
        - dq_j4 * x295 * x323
        - dq_j5 * x311 * x323
        - dq_j6 * x253 * x323
        + x107 * x129 * x205
        + x107 * x171 * x309
        + x107 * x177 * x293
        + x107 * x195 * x275
        + x107 * x201 * x243
        + x127 * x160 * x318
        - x149 * x248 * x322
        + x151 * x248
        + 1.0 * x241 * x66 * x67 * x68 * x75 * x82 * (sigma_pot_7_c * x156 - sigma_pot_7_s * x157)
    )

    return K_block_list
