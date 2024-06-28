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
    x2 = torch.cos(q_i1)
    x3 = torch.sin(q_j1)
    x4 = x2 * x3
    x5 = torch.cos(q_j1)
    x6 = torch.sin(q_i1)
    x7 = x5 * x6
    x8 = x2 * x5
    x9 = x3 * x6
    x10 = sigma_kin_p_1_1_c * x8 + sigma_kin_p_1_1_off + sigma_kin_p_1_1_s * x9
    x11 = x1 * x10 * (sigma_kin_p_1_1_c * x4 - sigma_kin_p_1_1_s * x7)
    x12 = dq_i1 * dq_j1
    x13 = dq_i2 * dq_j2
    x14 = sigma_kin_v_2_1 * x12 + sigma_kin_v_2_2 * x13
    x15 = x14**2
    x16 = torch.cos(q_i2)
    x17 = torch.cos(q_j2)
    x18 = x16 * x17
    x19 = torch.sin(q_i2)
    x20 = torch.sin(q_j2)
    x21 = x19 * x20
    x22 = sigma_kin_p_2_2_c * x18 + sigma_kin_p_2_2_off + sigma_kin_p_2_2_s * x21
    x23 = x22**2
    x24 = sigma_kin_p_2_1_c * x8 + sigma_kin_p_2_1_off + sigma_kin_p_2_1_s * x9
    x25 = x23 * x24 * (sigma_kin_p_2_1_c * x4 - sigma_kin_p_2_1_s * x7)
    x26 = dq_i3 * dq_j3
    x27 = sigma_kin_v_3_1 * x12 + sigma_kin_v_3_2 * x13 + sigma_kin_v_3_3 * x26
    x28 = x27**2
    x29 = sigma_kin_p_3_1_c * x8 + sigma_kin_p_3_1_off + sigma_kin_p_3_1_s * x9
    x30 = sigma_kin_p_3_2_c * x18 + sigma_kin_p_3_2_off + sigma_kin_p_3_2_s * x21
    x31 = x30**2
    x32 = torch.cos(q_i3)
    x33 = torch.cos(q_j3)
    x34 = x32 * x33
    x35 = torch.sin(q_i3)
    x36 = torch.sin(q_j3)
    x37 = x35 * x36
    x38 = sigma_kin_p_3_3_c * x34 + sigma_kin_p_3_3_off + sigma_kin_p_3_3_s * x37
    x39 = x38**2
    x40 = x31 * x39
    x41 = x29 * x40 * (sigma_kin_p_3_1_c * x4 - sigma_kin_p_3_1_s * x7)
    x42 = dq_i4 * dq_j4
    x43 = sigma_kin_v_4_1 * x12 + sigma_kin_v_4_2 * x13 + sigma_kin_v_4_3 * x26 + sigma_kin_v_4_4 * x42
    x44 = x43**2
    x45 = sigma_kin_p_4_1_c * x8 + sigma_kin_p_4_1_off + sigma_kin_p_4_1_s * x9
    x46 = sigma_kin_p_4_2_c * x18 + sigma_kin_p_4_2_off + sigma_kin_p_4_2_s * x21
    x47 = x46**2
    x48 = sigma_kin_p_4_3_c * x34 + sigma_kin_p_4_3_off + sigma_kin_p_4_3_s * x37
    x49 = x48**2
    x50 = torch.cos(q_i4)
    x51 = torch.cos(q_j4)
    x52 = x50 * x51
    x53 = torch.sin(q_i4)
    x54 = torch.sin(q_j4)
    x55 = x53 * x54
    x56 = sigma_kin_p_4_4_c * x52 + sigma_kin_p_4_4_off + sigma_kin_p_4_4_s * x55
    x57 = x56**2
    x58 = x47 * x49 * x57
    x59 = x45 * x58 * (sigma_kin_p_4_1_c * x4 - sigma_kin_p_4_1_s * x7)
    x60 = dq_i5 * dq_j5
    x61 = (
        sigma_kin_v_5_1 * x12
        + sigma_kin_v_5_2 * x13
        + sigma_kin_v_5_3 * x26
        + sigma_kin_v_5_4 * x42
        + sigma_kin_v_5_5 * x60
    )
    x62 = x61**2
    x63 = sigma_kin_p_5_1_c * x8 + sigma_kin_p_5_1_off + sigma_kin_p_5_1_s * x9
    x64 = sigma_kin_p_5_2_c * x18 + sigma_kin_p_5_2_off + sigma_kin_p_5_2_s * x21
    x65 = x64**2
    x66 = sigma_kin_p_5_3_c * x34 + sigma_kin_p_5_3_off + sigma_kin_p_5_3_s * x37
    x67 = x66**2
    x68 = sigma_kin_p_5_4_c * x52 + sigma_kin_p_5_4_off + sigma_kin_p_5_4_s * x55
    x69 = x68**2
    x70 = torch.cos(q_i5)
    x71 = torch.cos(q_j5)
    x72 = x70 * x71
    x73 = torch.sin(q_i5)
    x74 = torch.sin(q_j5)
    x75 = x73 * x74
    x76 = sigma_kin_p_5_5_c * x72 + sigma_kin_p_5_5_off + sigma_kin_p_5_5_s * x75
    x77 = x76**2
    x78 = x65 * x67 * x69 * x77
    x79 = x63 * x78 * (sigma_kin_p_5_1_c * x4 - sigma_kin_p_5_1_s * x7)
    x80 = sigma_kin_p_7_1_c * x8 + sigma_kin_p_7_1_off + sigma_kin_p_7_1_s * x9
    x81 = x80**2
    x82 = sigma_kin_p_7_2_c * x18 + sigma_kin_p_7_2_off + sigma_kin_p_7_2_s * x21
    x83 = x82**2
    x84 = sigma_kin_p_7_3_c * x34 + sigma_kin_p_7_3_off + sigma_kin_p_7_3_s * x37
    x85 = x84**2
    x86 = sigma_kin_p_7_4_c * x52 + sigma_kin_p_7_4_off + sigma_kin_p_7_4_s * x55
    x87 = x86**2
    x88 = sigma_kin_p_7_5_c * x72 + sigma_kin_p_7_5_off + sigma_kin_p_7_5_s * x75
    x89 = x88**2
    x90 = torch.cos(q_i6)
    x91 = torch.cos(q_j6)
    x92 = x90 * x91
    x93 = torch.sin(q_i6)
    x94 = torch.sin(q_j6)
    x95 = x93 * x94
    x96 = sigma_kin_p_7_6_c * x92 + sigma_kin_p_7_6_off + sigma_kin_p_7_6_s * x95
    x97 = x96**2
    x98 = torch.cos(q_j7)
    x99 = sigma_kin_p_7_7_c * torch.cos(q_i7)
    x100 = torch.sin(q_j7)
    x101 = sigma_kin_p_7_7_s * torch.sin(q_i7)
    x102 = sigma_kin_p_7_7_off + x100 * x101 + x98 * x99
    x103 = x102**2
    x104 = x103 * x83 * x85 * x87 * x89 * x97
    x105 = x104 * x81
    x106 = sigma_kin_v_7_1 * x105
    x107 = dq_i7 * sigma_kin_v_7_7
    x108 = ddq_j7 * x107
    x109 = dq_i6 * dq_j6
    x110 = (
        sigma_kin_v_6_1 * x12
        + sigma_kin_v_6_2 * x13
        + sigma_kin_v_6_3 * x26
        + sigma_kin_v_6_4 * x42
        + sigma_kin_v_6_5 * x60
        + sigma_kin_v_6_6 * x109
    )
    x111 = x110**2
    x112 = sigma_kin_p_6_1_c * x8 + sigma_kin_p_6_1_off + sigma_kin_p_6_1_s * x9
    x113 = sigma_kin_p_6_2_c * x18 + sigma_kin_p_6_2_off + sigma_kin_p_6_2_s * x21
    x114 = x113**2
    x115 = sigma_kin_p_6_3_c * x34 + sigma_kin_p_6_3_off + sigma_kin_p_6_3_s * x37
    x116 = x115**2
    x117 = sigma_kin_p_6_4_c * x52 + sigma_kin_p_6_4_off + sigma_kin_p_6_4_s * x55
    x118 = x117**2
    x119 = sigma_kin_p_6_5_c * x72 + sigma_kin_p_6_5_off + sigma_kin_p_6_5_s * x75
    x120 = x119**2
    x121 = sigma_kin_p_6_6_c * x92 + sigma_kin_p_6_6_off + sigma_kin_p_6_6_s * x95
    x122 = x121**2
    x123 = x114 * x116 * x118 * x120 * x122
    x124 = x112 * x123 * (sigma_kin_p_6_1_c * x4 - sigma_kin_p_6_1_s * x7)
    x125 = dq_j7 * x107
    x126 = (
        sigma_kin_v_7_1 * x12
        + sigma_kin_v_7_2 * x13
        + sigma_kin_v_7_3 * x26
        + sigma_kin_v_7_4 * x42
        + sigma_kin_v_7_5 * x60
        + sigma_kin_v_7_6 * x109
        + x125
    )
    x127 = x126**2
    x128 = x127 * x80
    x129 = sigma_kin_p_7_1_c * x4 - sigma_kin_p_7_1_s * x7
    x130 = x104 * x129
    x131 = 2 * dq_i1
    x132 = sigma_kin_v_7_1 * x126
    x133 = x100 * x99 - x101 * x98
    x134 = x81 * x83 * x85 * x87 * x89
    x135 = dq_j7 * x102 * x133 * x134 * x97
    x136 = ddq_j6 * dq_i6
    x137 = x112**2
    x138 = x123 * x137
    x139 = sigma_kin_v_6_1 * x138
    x140 = dq_i1 * (sigma_kin_v_6_6 * x139 + sigma_kin_v_7_6 * x106)
    x141 = x90 * x94
    x142 = x91 * x93
    x143 = x114 * x116 * x118 * x120 * x121 * x137 * (sigma_kin_p_6_6_c * x141 - sigma_kin_p_6_6_s * x142)
    x144 = sigma_kin_v_6_1 * x110
    x145 = sigma_kin_p_7_6_c * x141 - sigma_kin_p_7_6_s * x142
    x146 = x103 * x134 * x145 * x96
    x147 = ddq_j5 * dq_i5
    x148 = x63**2
    x149 = x148 * x78
    x150 = sigma_kin_v_5_1 * x149
    x151 = dq_i1 * (sigma_kin_v_5_5 * x150 + sigma_kin_v_6_5 * x139 + sigma_kin_v_7_5 * x106)
    x152 = ddq_j4 * dq_i4
    x153 = x45**2
    x154 = x153 * x58
    x155 = sigma_kin_v_4_1 * x154
    x156 = dq_i1 * (sigma_kin_v_4_4 * x155 + sigma_kin_v_5_4 * x150 + sigma_kin_v_6_4 * x139 + sigma_kin_v_7_4 * x106)
    x157 = x70 * x74
    x158 = x71 * x73
    x159 = x69 * x76 * (sigma_kin_p_5_5_c * x157 - sigma_kin_p_5_5_s * x158)
    x160 = x148 * x65 * x67
    x161 = sigma_kin_v_5_1 * x61
    x162 = x160 * x161
    x163 = x118 * x119 * (sigma_kin_p_6_5_c * x157 - sigma_kin_p_6_5_s * x158)
    x164 = x114 * x116 * x122 * x137
    x165 = x144 * x164
    x166 = sigma_kin_p_7_5_c * x157 - sigma_kin_p_7_5_s * x158
    x167 = x166 * x87 * x88
    x168 = x103 * x81 * x83 * x85 * x97
    x169 = x132 * x168
    x170 = ddq_j3 * dq_i3
    x171 = x29**2
    x172 = x171 * x40
    x173 = sigma_kin_v_3_1 * x172
    x174 = dq_i1 * (
        sigma_kin_v_3_3 * x173
        + sigma_kin_v_4_3 * x155
        + sigma_kin_v_5_3 * x150
        + sigma_kin_v_6_3 * x139
        + sigma_kin_v_7_3 * x106
    )
    x175 = ddq_j2 * dq_i2
    x176 = x24**2
    x177 = x176 * x23
    x178 = dq_i1 * (
        sigma_kin_v_2_1 * sigma_kin_v_2_2 * x177
        + sigma_kin_v_3_2 * x173
        + sigma_kin_v_4_2 * x155
        + sigma_kin_v_5_2 * x150
        + sigma_kin_v_6_2 * x139
        + sigma_kin_v_7_2 * x106
    )
    x179 = x50 * x54
    x180 = x51 * x53
    x181 = x153 * x47 * x49 * x56 * (sigma_kin_p_4_4_c * x179 - sigma_kin_p_4_4_s * x180)
    x182 = sigma_kin_v_4_1 * x43
    x183 = x68 * x77 * (sigma_kin_p_5_4_c * x179 - sigma_kin_p_5_4_s * x180)
    x184 = x117 * x120 * (sigma_kin_p_6_4_c * x179 - sigma_kin_p_6_4_s * x180)
    x185 = sigma_kin_p_7_4_c * x179 - sigma_kin_p_7_4_s * x180
    x186 = x185 * x86 * x89
    x187 = x32 * x36
    x188 = x33 * x35
    x189 = x31 * x38 * (sigma_kin_p_3_3_c * x187 - sigma_kin_p_3_3_s * x188)
    x190 = sigma_kin_v_3_1 * x27
    x191 = x171 * x190
    x192 = x47 * x48 * (sigma_kin_p_4_3_c * x187 - sigma_kin_p_4_3_s * x188)
    x193 = x153 * x57
    x194 = x182 * x193
    x195 = x65 * x66 * (sigma_kin_p_5_3_c * x187 - sigma_kin_p_5_3_s * x188)
    x196 = x148 * x69 * x77
    x197 = x161 * x196
    x198 = x114 * x115 * (sigma_kin_p_6_3_c * x187 - sigma_kin_p_6_3_s * x188)
    x199 = x118 * x120 * x122 * x137
    x200 = x144 * x199
    x201 = sigma_kin_p_7_3_c * x187 - sigma_kin_p_7_3_s * x188
    x202 = x201 * x83 * x84
    x203 = x103 * x81 * x87 * x89 * x97
    x204 = x132 * x203
    x205 = x16 * x20
    x206 = x17 * x19
    x207 = x176 * x22 * (sigma_kin_p_2_2_c * x205 - sigma_kin_p_2_2_s * x206)
    x208 = sigma_kin_v_2_1 * x14
    x209 = x30 * x39 * (sigma_kin_p_3_2_c * x205 - sigma_kin_p_3_2_s * x206)
    x210 = x46 * x49 * (sigma_kin_p_4_2_c * x205 - sigma_kin_p_4_2_s * x206)
    x211 = x64 * x67 * (sigma_kin_p_5_2_c * x205 - sigma_kin_p_5_2_s * x206)
    x212 = x113 * x116 * (sigma_kin_p_6_2_c * x205 - sigma_kin_p_6_2_s * x206)
    x213 = sigma_kin_p_7_2_c * x205 - sigma_kin_p_7_2_s * x206
    x214 = x213 * x82 * x85
    x215 = x130 * x80
    x216 = x171 * x28
    x217 = x193 * x44
    x218 = x196 * x62
    x219 = sigma_kin_v_7_2 * x105
    x220 = x111 * x199
    x221 = x127 * x203
    x222 = 2 * dq_i2
    x223 = sigma_kin_v_7_2 * x126
    x224 = sigma_kin_v_6_2 * x138
    x225 = sigma_kin_v_6_6 * x224 + sigma_kin_v_7_6 * x219
    x226 = sigma_kin_v_6_2 * x110
    x227 = sigma_kin_v_5_2 * x149
    x228 = sigma_kin_v_5_5 * x227 + sigma_kin_v_6_5 * x224 + sigma_kin_v_7_5 * x219
    x229 = sigma_kin_v_4_2 * x154
    x230 = sigma_kin_v_4_4 * x229 + sigma_kin_v_5_4 * x227 + sigma_kin_v_6_4 * x224 + sigma_kin_v_7_4 * x219
    x231 = sigma_kin_v_5_2 * x61
    x232 = x160 * x231
    x233 = x164 * x226
    x234 = x168 * x223
    x235 = (
        sigma_kin_v_3_2 * sigma_kin_v_3_3 * x172
        + sigma_kin_v_4_3 * x229
        + sigma_kin_v_5_3 * x227
        + sigma_kin_v_6_3 * x224
        + sigma_kin_v_7_3 * x219
    )
    x236 = sigma_kin_v_4_2 * x43
    x237 = sigma_kin_v_3_2 * x27
    x238 = x171 * x237
    x239 = x193 * x236
    x240 = x196 * x231
    x241 = x199 * x226
    x242 = x203 * x223
    x243 = sigma_kin_v_2_2 * x14
    x244 = sigma_kin_v_7_3 * x105
    x245 = 2 * dq_i3
    x246 = sigma_kin_v_7_3 * x126
    x247 = sigma_kin_v_6_3 * x138
    x248 = sigma_kin_v_6_6 * x247 + sigma_kin_v_7_6 * x244
    x249 = sigma_kin_v_6_3 * x110
    x250 = sigma_kin_v_5_3 * x149
    x251 = sigma_kin_v_5_5 * x250 + sigma_kin_v_6_5 * x247 + sigma_kin_v_7_5 * x244
    x252 = (
        sigma_kin_v_4_3 * sigma_kin_v_4_4 * x154
        + sigma_kin_v_5_4 * x250
        + sigma_kin_v_6_4 * x247
        + sigma_kin_v_7_4 * x244
    )
    x253 = sigma_kin_v_5_3 * x61
    x254 = x160 * x253
    x255 = x164 * x249
    x256 = x168 * x246
    x257 = sigma_kin_v_4_3 * x43
    x258 = sigma_kin_v_3_3 * x27
    x259 = x171 * x258
    x260 = x193 * x257
    x261 = x196 * x253
    x262 = x199 * x249
    x263 = x203 * x246
    x264 = x160 * x62
    x265 = sigma_kin_v_7_4 * x105
    x266 = x111 * x164
    x267 = x127 * x168
    x268 = 2 * dq_i4
    x269 = sigma_kin_v_7_4 * x126
    x270 = sigma_kin_v_6_4 * x138
    x271 = sigma_kin_v_6_6 * x270 + sigma_kin_v_7_6 * x265
    x272 = sigma_kin_v_6_4 * x110
    x273 = sigma_kin_v_5_4 * sigma_kin_v_5_5 * x149 + sigma_kin_v_6_5 * x270 + sigma_kin_v_7_5 * x265
    x274 = sigma_kin_v_5_4 * x61
    x275 = x160 * x274
    x276 = x164 * x272
    x277 = x168 * x269
    x278 = sigma_kin_v_4_4 * x43
    x279 = x193 * x278
    x280 = x196 * x274
    x281 = x199 * x272
    x282 = x203 * x269
    x283 = sigma_kin_v_7_5 * x105
    x284 = 2 * dq_i5
    x285 = sigma_kin_v_7_5 * x126
    x286 = sigma_kin_v_6_5 * sigma_kin_v_6_6 * x138 + sigma_kin_v_7_6 * x283
    x287 = sigma_kin_v_6_5 * x110
    x288 = sigma_kin_v_5_5 * x61
    x289 = x196 * x288
    x290 = x199 * x287
    x291 = x203 * x285
    x292 = x160 * x288
    x293 = x164 * x287
    x294 = x168 * x285
    x295 = 2 * dq_i6
    x296 = sigma_kin_v_7_6 * x126
    x297 = sigma_kin_v_6_6 * x110
    x298 = x199 * x297
    x299 = x203 * x296
    x300 = x164 * x297
    x301 = x168 * x296
    x302 = x82 * x84 * x86 * x88 * x96
    x303 = x102 * x302
    x304 = x303 * x80
    x305 = x107 * x304
    x306 = x133 * x302
    x307 = x107 * x126
    x308 = 2 * x80
    x309 = x102 * x307 * x308 * x86 * x88 * x96
    x310 = x102 * x307 * x308 * x82 * x84 * x96

    K_block_list = []
    K_block_list.append(
        2
        * ddq_j1
        * x0
        * (
            sigma_kin_v_2_1**2 * x177
            + sigma_kin_v_3_1**2 * x172
            + sigma_kin_v_4_1**2 * x154
            + sigma_kin_v_5_1**2 * x149
            + sigma_kin_v_6_1**2 * x138
            + sigma_kin_v_7_1**2 * x105
            + x1 * x10**2
        )
        + 2 * dq_i1 * x106 * x108
        + 2 * dq_j1**2 * x0 * x11
        - 2 * dq_j2 * x131 * (x191 * x209 + x194 * x210 + x197 * x211 + x200 * x212 + x204 * x214 + x207 * x208)
        - 2 * dq_j3 * x131 * (x189 * x191 + x192 * x194 + x195 * x197 + x198 * x200 + x202 * x204)
        - 2 * dq_j4 * x131 * (x162 * x183 + x165 * x184 + x169 * x186 + x181 * x182)
        - 2 * dq_j5 * x131 * (x159 * x162 + x163 * x165 + x167 * x169)
        - 2 * dq_j6 * x131 * (x132 * x146 + x143 * x144)
        + 2 * x111 * x124
        - 4 * x12 * (x11 * x12 + x124 * x144 + x132 * x215 + x161 * x79 + x182 * x59 + x190 * x41 + x208 * x25)
        + 2 * x128 * x130
        - 2 * x131 * x132 * x135
        + 2 * x136 * x140
        + 2 * x147 * x151
        + 2 * x15 * x25
        + 2 * x152 * x156
        + 2 * x170 * x174
        + 2 * x175 * x178
        + 2 * x28 * x41
        + 2 * x44 * x59
        + 2 * x62 * x79
    )
    K_block_list.append(
        2 * ddq_j1 * dq_i2 * x178
        + 2
        * ddq_j2
        * dq_i2**2
        * (
            sigma_kin_v_2_2**2 * x177
            + sigma_kin_v_3_2**2 * x172
            + sigma_kin_v_4_2**2 * x154
            + sigma_kin_v_5_2**2 * x149
            + sigma_kin_v_6_2**2 * x138
            + sigma_kin_v_7_2**2 * x105
        )
        + 2 * dq_i2 * x108 * x219
        + 2 * dq_i2 * x136 * x225
        + 2 * dq_i2 * x147 * x228
        + 2 * dq_i2 * x152 * x230
        + 2 * dq_i2 * x170 * x235
        - 2 * dq_j1 * x222 * (x124 * x226 + x215 * x223 + x231 * x79 + x236 * x59 + x237 * x41 + x243 * x25)
        - 2 * dq_j3 * x222 * (x189 * x238 + x192 * x239 + x195 * x240 + x198 * x241 + x202 * x242)
        - 2 * dq_j4 * x222 * (x181 * x236 + x183 * x232 + x184 * x233 + x186 * x234)
        - 2 * dq_j5 * x222 * (x159 * x232 + x163 * x233 + x167 * x234)
        - 2 * dq_j6 * x222 * (x143 * x226 + x146 * x223)
        - 4 * x13 * (x207 * x243 + x209 * x238 + x210 * x239 + x211 * x240 + x212 * x241 + x214 * x242)
        - 2 * x135 * x222 * x223
        + 2 * x15 * x207
        + 2 * x209 * x216
        + 2 * x210 * x217
        + 2 * x211 * x218
        + 2 * x212 * x220
        + 2 * x214 * x221
    )
    K_block_list.append(
        2 * ddq_j1 * dq_i3 * x174
        + 2
        * ddq_j3
        * dq_i3**2
        * (
            sigma_kin_v_3_3**2 * x172
            + sigma_kin_v_4_3**2 * x154
            + sigma_kin_v_5_3**2 * x149
            + sigma_kin_v_6_3**2 * x138
            + sigma_kin_v_7_3**2 * x105
        )
        + 2 * dq_i3 * x108 * x244
        + 2 * dq_i3 * x136 * x248
        + 2 * dq_i3 * x147 * x251
        + 2 * dq_i3 * x152 * x252
        + 2 * dq_i3 * x175 * x235
        - 2 * dq_j1 * x245 * (x124 * x249 + x215 * x246 + x253 * x79 + x257 * x59 + x258 * x41)
        - 2 * dq_j2 * x245 * (x209 * x259 + x210 * x260 + x211 * x261 + x212 * x262 + x214 * x263)
        - 2 * dq_j4 * x245 * (x181 * x257 + x183 * x254 + x184 * x255 + x186 * x256)
        - 2 * dq_j5 * x245 * (x159 * x254 + x163 * x255 + x167 * x256)
        - 2 * dq_j6 * x245 * (x143 * x249 + x146 * x246)
        - 2 * x135 * x245 * x246
        + 2 * x189 * x216
        + 2 * x192 * x217
        + 2 * x195 * x218
        + 2 * x198 * x220
        + 2 * x202 * x221
        - 4 * x26 * (x189 * x259 + x192 * x260 + x195 * x261 + x198 * x262 + x202 * x263)
    )
    K_block_list.append(
        2 * ddq_j1 * dq_i4 * x156
        + 2
        * ddq_j4
        * dq_i4**2
        * (
            sigma_kin_v_4_4**2 * x154
            + sigma_kin_v_5_4**2 * x149
            + sigma_kin_v_6_4**2 * x138
            + sigma_kin_v_7_4**2 * x105
        )
        + 2 * dq_i4 * x108 * x265
        + 2 * dq_i4 * x136 * x271
        + 2 * dq_i4 * x147 * x273
        + 2 * dq_i4 * x170 * x252
        + 2 * dq_i4 * x175 * x230
        - 2 * dq_j1 * x268 * (x124 * x272 + x215 * x269 + x274 * x79 + x278 * x59)
        - 2 * dq_j2 * x268 * (x210 * x279 + x211 * x280 + x212 * x281 + x214 * x282)
        - 2 * dq_j3 * x268 * (x192 * x279 + x195 * x280 + x198 * x281 + x202 * x282)
        - 2 * dq_j5 * x268 * (x159 * x275 + x163 * x276 + x167 * x277)
        - 2 * dq_j6 * x268 * (x143 * x272 + x146 * x269)
        - 2 * x135 * x268 * x269
        + 2 * x181 * x44
        + 2 * x183 * x264
        + 2 * x184 * x266
        + 2 * x186 * x267
        - 4 * x42 * (x181 * x278 + x183 * x275 + x184 * x276 + x186 * x277)
    )
    K_block_list.append(
        2 * ddq_j1 * dq_i5 * x151
        + 2
        * ddq_j5
        * dq_i5**2
        * (sigma_kin_v_5_5**2 * x149 + sigma_kin_v_6_5**2 * x138 + sigma_kin_v_7_5**2 * x105)
        + 2 * dq_i5 * x108 * x283
        + 2 * dq_i5 * x136 * x286
        + 2 * dq_i5 * x152 * x273
        + 2 * dq_i5 * x170 * x251
        + 2 * dq_i5 * x175 * x228
        - 2 * dq_j1 * x284 * (x124 * x287 + x215 * x285 + x288 * x79)
        - 2 * dq_j2 * x284 * (x211 * x289 + x212 * x290 + x214 * x291)
        - 2 * dq_j3 * x284 * (x195 * x289 + x198 * x290 + x202 * x291)
        - 2 * dq_j4 * x284 * (x183 * x292 + x184 * x293 + x186 * x294)
        - 2 * dq_j6 * x284 * (x143 * x287 + x146 * x285)
        - 2 * x135 * x284 * x285
        + 2 * x159 * x264
        + 2 * x163 * x266
        + 2 * x167 * x267
        - 4 * x60 * (x159 * x292 + x163 * x293 + x167 * x294)
    )
    K_block_list.append(
        2 * ddq_j1 * dq_i6 * x140
        + 2 * ddq_j6 * dq_i6**2 * (sigma_kin_v_6_6**2 * x138 + sigma_kin_v_7_6**2 * x105)
        + 2 * dq_i6 * sigma_kin_v_7_6 * x105 * x108
        + 2 * dq_i6 * x147 * x286
        + 2 * dq_i6 * x152 * x271
        + 2 * dq_i6 * x170 * x248
        + 2 * dq_i6 * x175 * x225
        - 2 * dq_j1 * x295 * (x124 * x297 + x215 * x296)
        - 2 * dq_j2 * x295 * (x212 * x298 + x214 * x299)
        - 2 * dq_j3 * x295 * (x198 * x298 + x202 * x299)
        - 2 * dq_j4 * x295 * (x184 * x300 + x186 * x301)
        - 2 * dq_j5 * x295 * (x163 * x300 + x167 * x301)
        - 4 * x109 * (x143 * x297 + x146 * x296)
        + 2 * x111 * x143
        + 2 * x127 * x146
        - 2 * x135 * x295 * x296
    )
    K_block_list.append(
        x303
        * x308
        * (
            ddq_j1 * dq_i1 * sigma_kin_v_7_1 * x305
            + ddq_j7 * dq_i7**2 * sigma_kin_v_7_7**2 * x304
            - 2 * dq_j1 * x129 * x303 * x307
            - dq_j2 * x213 * x309 * x84
            - dq_j3 * x201 * x309 * x82
            - dq_j4 * x185 * x310 * x88
            - dq_j5 * x166 * x310 * x86
            - dq_j6 * x102 * x145 * x307 * x308 * x82 * x84 * x86 * x88
            + sigma_kin_v_7_2 * x175 * x305
            + sigma_kin_v_7_3 * x170 * x305
            + sigma_kin_v_7_4 * x152 * x305
            + sigma_kin_v_7_5 * x147 * x305
            + sigma_kin_v_7_6 * x136 * x305
            - x125 * x126 * x306 * x308
            + x128 * x306
        )
    )

    return K_block_list
