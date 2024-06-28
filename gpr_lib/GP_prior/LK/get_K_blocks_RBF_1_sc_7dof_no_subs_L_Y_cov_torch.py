import torch


def get_K_blocks_RBF_1_sc_7dof_no_subs(X1, X2, pos_indices, vel_indices, acc_indices, lL, sL):

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

    lL1 = lL[0]
    lL8 = lL[7]
    lL15 = lL[14]
    lL2 = lL[1]
    lL9 = lL[8]
    lL16 = lL[15]
    lL3 = lL[2]
    lL10 = lL[9]
    lL17 = lL[16]
    lL4 = lL[3]
    lL11 = lL[10]
    lL18 = lL[17]
    lL5 = lL[4]
    lL12 = lL[11]
    lL19 = lL[18]
    lL6 = lL[5]
    lL13 = lL[12]
    lL20 = lL[19]
    lL7 = lL[6]
    lL14 = lL[13]
    lL21 = lL[20]

    x0 = torch.sin(q_j1)
    x1 = lL1 ** (-2.0)
    x2 = torch.cos(q_j1)
    x3 = -x2 + torch.cos(q_i1)
    x4 = x0 * x1 * x3
    x5 = lL8 ** (-2.0)
    x6 = -x0 + torch.sin(q_i1)
    x7 = x2 * x5 * x6
    x8 = lL16 ** (-2.0)
    x9 = ddq_j2 * x8
    x10 = dq_i2 - dq_j2
    x11 = dq_i1 - dq_j1
    x12 = lL15 ** (-2.0)
    x13 = 2 * x12
    x14 = x11 * x13
    x15 = x10 * x14
    x16 = dq_i3 - dq_j3
    x17 = lL17 ** (-2.0)
    x18 = ddq_j3 * x17
    x19 = x16 * x18
    x20 = dq_i4 - dq_j4
    x21 = lL18 ** (-2.0)
    x22 = ddq_j4 * x21
    x23 = x20 * x22
    x24 = dq_i5 - dq_j5
    x25 = lL19 ** (-2.0)
    x26 = ddq_j5 * x25
    x27 = x24 * x26
    x28 = dq_i6 - dq_j6
    x29 = lL20 ** (-2.0)
    x30 = ddq_j6 * x29
    x31 = x28 * x30
    x32 = dq_i7 - dq_j7
    x33 = lL21 ** (-2.0)
    x34 = ddq_j7 * x33
    x35 = x32 * x34
    x36 = dq_j1 * (-x4 + x7)
    x37 = torch.cos(q_j2)
    x38 = lL9 ** (-2.0)
    x39 = torch.sin(q_j2)
    x40 = -x39 + torch.sin(q_i2)
    x41 = x37 * x38 * x40
    x42 = lL2 ** (-2.0)
    x43 = -x37 + torch.cos(q_i2)
    x44 = x39 * x42 * x43
    x45 = dq_j2 * (x41 - x44)
    x46 = torch.sin(q_j3)
    x47 = lL3 ** (-2.0)
    x48 = torch.cos(q_j3)
    x49 = -x48 + torch.cos(q_i3)
    x50 = lL10 ** (-2.0)
    x51 = -x46 + torch.sin(q_i3)
    x52 = x46 * x47 * x49 - x48 * x50 * x51
    x53 = dq_j3 * x52
    x54 = torch.sin(q_j4)
    x55 = lL4 ** (-2.0)
    x56 = torch.cos(q_j4)
    x57 = -x56 + torch.cos(q_i4)
    x58 = lL11 ** (-2.0)
    x59 = -x54 + torch.sin(q_i4)
    x60 = x54 * x55 * x57 - x56 * x58 * x59
    x61 = dq_j4 * x60
    x62 = torch.sin(q_j5)
    x63 = lL5 ** (-2.0)
    x64 = torch.cos(q_j5)
    x65 = -x64 + torch.cos(q_i5)
    x66 = lL12 ** (-2.0)
    x67 = -x62 + torch.sin(q_i5)
    x68 = x62 * x63 * x65 - x64 * x66 * x67
    x69 = dq_j5 * x68
    x70 = torch.sin(q_j6)
    x71 = lL6 ** (-2.0)
    x72 = torch.cos(q_j6)
    x73 = -x72 + torch.cos(q_i6)
    x74 = lL13 ** (-2.0)
    x75 = -x70 + torch.sin(q_i6)
    x76 = x70 * x71 * x73 - x72 * x74 * x75
    x77 = dq_j6 * x76
    x78 = torch.sin(q_j7)
    x79 = lL7 ** (-2.0)
    x80 = torch.cos(q_j7)
    x81 = -x80 + torch.cos(q_i7)
    x82 = lL14 ** (-2.0)
    x83 = -x78 + torch.sin(q_i7)
    x84 = x78 * x79 * x81 - x80 * x82 * x83
    x85 = dq_j7 * x84
    x86 = (
        2
        * sL
        * torch.exp(
            -x1 * x3**2
            - x10**2 * x8
            - x11**2 * x12
            - x16**2 * x17
            - x20**2 * x21
            - x24**2 * x25
            - x28**2 * x29
            - x32**2 * x33
            - x38 * x40**2
            - x42 * x43**2
            - x47 * x49**2
            - x5 * x6**2
            - x50 * x51**2
            - x55 * x57**2
            - x58 * x59**2
            - x63 * x65**2
            - x66 * x67**2
            - x71 * x73**2
            - x74 * x75**2
            - x79 * x81**2
            - x82 * x83**2
        )
    )
    x87 = 2 * x8
    x88 = x10 * x87
    x89 = 2 * x17
    x90 = x16 * x89
    x91 = x16 * x17
    x92 = ddq_j1 * x14
    x93 = ddq_j2 * x88
    x94 = 2 * x21
    x95 = x20 * x94
    x96 = x20 * x21
    x97 = ddq_j3 * x90
    x98 = 2 * x25
    x99 = x24 * x98
    x100 = x24 * x25
    x101 = ddq_j4 * x95
    x102 = 2 * x29
    x103 = x102 * x28
    x104 = x28 * x29
    x105 = ddq_j5 * x99
    x106 = 2 * x33
    x107 = x106 * x32
    x108 = x32 * x33

    K_block_list = []
    K_block_list.append(
        x86
        * (
            -ddq_j1 * x12 * (-(x11**2) * x13 + 1)
            + x14 * x19
            + x14 * x23
            + x14 * x27
            + x14 * x31
            + x14 * x35
            + x14 * x36
            + x14 * x45
            - x14 * x53
            - x14 * x61
            - x14 * x69
            - x14 * x77
            - x14 * x85
            + x15 * x9
            + x4
            - x7
        )
    )
    K_block_list.append(
        x86
        * (
            ddq_j1 * x15 * x8
            + x19 * x88
            + x23 * x88
            + x27 * x88
            + x31 * x88
            + x35 * x88
            + x36 * x88
            - x41
            + x44
            + x45 * x88
            - x53 * x88
            - x61 * x88
            - x69 * x88
            - x77 * x88
            - x85 * x88
            - x9 * (-(x10**2) * x87 + 1)
        )
    )
    K_block_list.append(
        x86
        * (
            -x18 * (-(x16**2) * x89 + 1)
            + x23 * x90
            + x27 * x90
            + x31 * x90
            + x35 * x90
            + x36 * x90
            + x45 * x90
            + x52
            - x53 * x90
            - x61 * x90
            - x69 * x90
            - x77 * x90
            - x85 * x90
            + x91 * x92
            + x91 * x93
        )
    )
    K_block_list.append(
        x86
        * (
            -x22 * (-(x20**2) * x94 + 1)
            + x27 * x95
            + x31 * x95
            + x35 * x95
            + x36 * x95
            + x45 * x95
            - x53 * x95
            + x60
            - x61 * x95
            - x69 * x95
            - x77 * x95
            - x85 * x95
            + x92 * x96
            + x93 * x96
            + x96 * x97
        )
    )
    K_block_list.append(
        x86
        * (
            x100 * x101
            + x100 * x92
            + x100 * x93
            + x100 * x97
            - x26 * (-(x24**2) * x98 + 1)
            + x31 * x99
            + x35 * x99
            + x36 * x99
            + x45 * x99
            - x53 * x99
            - x61 * x99
            + x68
            - x69 * x99
            - x77 * x99
            - x85 * x99
        )
    )
    K_block_list.append(
        x86
        * (
            x101 * x104
            + x103 * x35
            + x103 * x36
            + x103 * x45
            - x103 * x53
            - x103 * x61
            - x103 * x69
            - x103 * x77
            - x103 * x85
            + x104 * x105
            + x104 * x92
            + x104 * x93
            + x104 * x97
            - x30 * (-x102 * x28**2 + 1)
            + x76
        )
    )
    K_block_list.append(
        x86
        * (
            ddq_j6 * x103 * x108
            + x101 * x108
            + x105 * x108
            + x107 * x36
            + x107 * x45
            - x107 * x53
            - x107 * x61
            - x107 * x69
            - x107 * x77
            - x107 * x85
            + x108 * x92
            + x108 * x93
            + x108 * x97
            - x34 * (-x106 * x32**2 + 1)
            + x84
        )
    )

    return K_block_list
