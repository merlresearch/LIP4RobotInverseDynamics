# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import torch
from torch import cos, sin


def H(X, pos_dim, vel_dim, acc_dim, dtype, device):
    q = X[:, pos_dim]
    dq = X[:, vel_dim]
    ddq = X[:, acc_dim]
    #
    H_num = torch.zeros([q.shape[0], 490], dtype=dtype, device=device)
    #
    x0 = torch.sin(q[:, 1])
    x1 = -ddq[:, 0]
    x2 = torch.cos(q[:, 1])
    x3 = -dq[:, 0]
    x4 = x2 * x3
    x5 = dq[:, 1] * x4
    x6 = x0 * x1 + x5
    x7 = -x6
    x8 = x0 * x3
    x9 = dq[:, 1] * x8
    x10 = -x9
    x11 = x10 * x2
    x12 = dq[:, 0] * dq[:, 1] * x0 + x1 * x2
    x13 = -x0
    x14 = -x2
    x15 = x5 + x6
    x16 = x4 * x8
    x17 = dq[:, 1] ** 2
    x18 = x8**2
    x19 = -x18
    x20 = -x12
    x21 = -x5
    x22 = -x17
    x23 = x4**2
    x24 = -x16
    x25 = -9.81 * x0
    x26 = -9.81 * x2
    x27 = -x26
    x28 = torch.cos(q[:, 2])
    x29 = torch.sin(q[:, 2])
    x30 = -x29
    x31 = dq[:, 1] * x28 + x30 * x8
    x32 = ddq[:, 1] * x29 + dq[:, 2] * x31 + x28 * x6
    x33 = dq[:, 1] * x29 + x28 * x8
    x34 = dq[:, 2] - x4
    x35 = x33 * x34
    x36 = x31 * x33
    x37 = -x36
    x38 = -x35
    x39 = -x33
    x40 = ddq[:, 1] * x28 + dq[:, 2] * x39 + x29 * x7
    x41 = x38 + x40
    x42 = x31 * x34
    x43 = x32 + x42
    x44 = x31**2
    x45 = -x44
    x46 = x33**2
    x47 = x45 + x46
    x48 = ddq[:, 2] + x20
    x49 = x36 + x48
    x50 = x28 * x49
    x51 = x34**2
    x52 = -x46
    x53 = x51 + x52
    x54 = -x42
    x55 = x32 + x54
    x56 = -x51
    x57 = x44 + x56
    x58 = x37 + x48
    x59 = x35 + x40
    x60 = -x48
    x61 = 0.316 * ddq[:, 1] - 0.316 * x16 + x25
    x62 = -0.316 * x15
    x63 = x28 * x62 + x30 * x61
    x64 = -x63
    x65 = 0.316 * x19 + 0.316 * x22 + x27
    x66 = -x65
    x67 = x45 + x56
    x68 = x36 + x60
    x69 = x52 + x56
    x70 = x28 * x61 + x29 * x62
    x71 = -x70
    x72 = -x32
    x73 = x42 + x72
    x74 = -0.316 * x29
    x75 = torch.cos(q[:, 3])
    x76 = torch.sin(q[:, 3])
    x77 = x34 * x75 + x39 * x76
    x78 = dq[:, 3] * x77 + x32 * x75 + x48 * x76
    x79 = x33 * x75 + x34 * x76
    x80 = dq[:, 3] - x31
    x81 = x79 * x80
    x82 = -x76
    x83 = x75 * x78 + x81 * x82
    x84 = x77 * x79
    x85 = -x84
    x86 = -x85
    x87 = x75 * x81 + x76 * x78
    x88 = -x79
    x89 = dq[:, 3] * x88 + x48 * x75 + x72 * x76
    x90 = -x81
    x91 = x89 + x90
    x92 = x77 * x80
    x93 = x78 + x92
    x94 = x75 * x91 + x82 * x93
    x95 = x77**2
    x96 = -x95
    x97 = x79**2
    x98 = x96 + x97
    x99 = -x98
    x100 = x75 * x93 + x76 * x91
    x101 = -x40
    x102 = ddq[:, 3] + x101
    x103 = x102 + x84
    x104 = x80**2
    x105 = -x97
    x106 = x104 + x105
    x107 = x103 * x75 + x106 * x82
    x108 = -x92
    x109 = x108 + x78
    x110 = -x109
    x111 = x103 * x76 + x106 * x75
    x112 = -x89
    x113 = x108 * x75 + x112 * x76
    x114 = x108 * x76 + x75 * x89
    x115 = -x104
    x116 = x115 + x95
    x117 = x102 + x85
    x118 = x116 * x75 + x117 * x82
    x119 = x81 + x89
    x120 = -x119
    x121 = x116 * x76 + x117 * x75
    x122 = x75 * x92 + x82 * x90
    x123 = -x102
    x124 = x75 * x90 + x76 * x92
    x125 = -0.0825 * x49 + x64
    x126 = -x125
    x127 = x126 * x82
    x128 = 0.0825 * x101 + 0.0825 * x35 + x65
    x129 = 0.0825 * x67 + x70
    x130 = x128 * x75 + x129 * x82
    x131 = -x130
    x132 = x115 + x96
    x133 = -0.0825 * x132
    x134 = -0.0825 * x103
    x135 = x131 + x133 * x76 + x134 * x75
    x136 = x103 * x82 + x132 * x75
    x137 = x112 + x81
    x138 = -x137
    x139 = -0.0825 * x137
    x140 = x126 * x75 + x139
    x141 = x125 * x75
    x142 = x128 * x76 + x129 * x75
    x143 = -x142
    x144 = x123 + x84
    x145 = x105 + x115
    x146 = -x143 - 0.0825 * x144 * x76 - 0.0825 * x145 * x75
    x147 = x144 * x75 + x145 * x82
    x148 = -x93
    x149 = x125 * x76 - 0.0825 * x93
    x150 = x131 * x75 + x142 * x82
    x151 = -x78
    x152 = x151 + x92
    x153 = -0.0825 * x119 * x76 - 0.0825 * x152 * x75
    x154 = x119 * x75 + x152 * x82
    x155 = x105 + x96
    x156 = -x155
    x157 = x142 * x75
    x158 = x131 * x76 - 0.0825 * x155 + x157
    x159 = -0.0825 * x130 * x75 - 0.0825 * x142 * x76
    x160 = x130 * x82 + x157
    x161 = -0.0825 * x125
    x162 = torch.cos(q[:, 4])
    x163 = torch.sin(q[:, 4])
    x164 = -x80
    x165 = x162 * x164 + x163 * x88
    x166 = dq[:, 4] * x165 + x123 * x163 + x162 * x78
    x167 = -x163
    x168 = x162 * x79 + x163 * x164
    x169 = dq[:, 4] + x77
    x170 = x168 * x169
    x171 = x162 * x166 + x167 * x170
    x172 = x165 * x168
    x173 = -x172
    x174 = x171 * x75 + x173 * x82
    x175 = -x162
    x176 = x166 * x167 + x170 * x175
    x177 = -x176
    x178 = x171 * x76 + x173 * x75
    x179 = -x168
    x180 = dq[:, 4] * x179 + x123 * x162 + x151 * x163
    x181 = -x170
    x182 = x180 + x181
    x183 = x165 * x169
    x184 = x166 + x183
    x185 = x162 * x182 + x167 * x184
    x186 = x165**2
    x187 = -x186
    x188 = x168**2
    x189 = x187 + x188
    x190 = x185 * x75 + x189 * x82
    x191 = x167 * x182 + x175 * x184
    x192 = -x191
    x193 = x185 * x76 + x189 * x75
    x194 = ddq[:, 4] + x89
    x195 = x172 + x194
    x196 = x162 * x195
    x197 = x169**2
    x198 = -x188
    x199 = x197 + x198
    x200 = x167 * x199 + x196
    x201 = -x183
    x202 = x166 + x201
    x203 = x200 * x75 + x202 * x82
    x204 = x167 * x195
    x205 = x175 * x199 + x204
    x206 = -x205
    x207 = x200 * x76 + x202 * x75
    x208 = x162 * x201 + x167 * x180
    x209 = x172 * x82 + x208 * x75
    x210 = -x180
    x211 = x162 * x210 + x167 * x201
    x212 = -x211
    x213 = x172 * x75 + x208 * x76
    x214 = -x197
    x215 = x186 + x214
    x216 = x173 + x194
    x217 = x162 * x215 + x167 * x216
    x218 = x170 + x180
    x219 = x217 * x75 + x218 * x82
    x220 = x167 * x215 + x175 * x216
    x221 = -x220
    x222 = x217 * x76 + x218 * x75
    x223 = x162 * x183 + x167 * x181
    x224 = -x194
    x225 = x223 * x75 + x224 * x76
    x226 = x167 * x183 + x175 * x181
    x227 = -x226
    x228 = x194 * x75 + x223 * x76
    x229 = x130 + x134 + 0.384 * x145
    x230 = -x229
    x231 = x187 + x214
    x232 = x163 * x231
    x233 = x167 * x230 - 0.384 * x196 - 0.384 * x232
    x234 = x133 + x142 + 0.384 * x144
    x235 = x125 + x139 + 0.384 * x93
    x236 = x167 * x234 + x175 * x235
    x237 = -0.0825 * x196 - 0.0825 * x232 + x236
    x238 = x233 * x75 + x237 * x82
    x239 = x170 + x210
    x240 = -0.0825 * x239
    x241 = 0.384 * x163
    x242 = x162 * x231
    x243 = x175 * x230 + x195 * x241 + x240 - 0.384 * x242
    x244 = x204 + x242
    x245 = x240 * x75 - x243 - 0.0825 * x244 * x76
    x246 = x239 * x82 + x244 * x75
    x247 = x167 * x231 + x175 * x195
    x248 = -x247
    x249 = x233 * x76 + x237 * x75 - 0.0825 * x247
    x250 = x172 + x224
    x251 = x163 * x250
    x252 = x198 + x214
    x253 = x162 * x252
    x254 = x162 * x229 - 0.384 * x251 - 0.384 * x253
    x255 = x162 * x234 + x167 * x235
    x256 = -x255
    x257 = -0.0825 * x251 - 0.0825 * x253 + x256
    x258 = x254 * x75 + x257 * x82
    x259 = -0.0825 * x184
    x260 = x162 * x250
    x261 = x167 * x229 + x241 * x252 + x259 - 0.384 * x260
    x262 = x167 * x252 + x260
    x263 = x259 * x75 - x261 - 0.0825 * x262 * x76
    x264 = x184 * x82 + x262 * x75
    x265 = x167 * x250 + x175 * x252
    x266 = -x265
    x267 = x254 * x76 + x257 * x75 - 0.0825 * x265
    x268 = x163 * x218
    x269 = -x166
    x270 = x183 + x269
    x271 = x162 * x270
    x272 = -x236
    x273 = x162 * x272 + x167 * x255
    x274 = -0.384 * x268 - 0.384 * x271 + x273
    x275 = -0.0825 * x268 - 0.0825 * x271
    x276 = x274 * x75 + x275 * x82
    x277 = x187 + x198
    x278 = -0.0825 * x277
    x279 = x162 * x218
    x280 = x167 * x272 + x175 * x255 + x241 * x270 + x278 - 0.384 * x279
    x281 = x167 * x270 + x279
    x282 = x278 * x75 - x280 - 0.0825 * x281 * x76
    x283 = x277 * x82 + x281 * x75
    x284 = x167 * x218 + x175 * x270
    x285 = -x284
    x286 = x274 * x76 + x275 * x75 - 0.0825 * x284
    x287 = x163 * x255
    x288 = x162 * x236
    x289 = -0.384 * x287 - 0.384 * x288
    x290 = -0.0825 * x287 - 0.0825 * x288
    x291 = x289 * x75 + x290 * x82
    x292 = -0.0825 * x229
    x293 = x162 * x255
    x294 = x236 * x241 + x292 - 0.384 * x293
    x295 = x167 * x236 + x293
    x296 = x292 * x75 - x294 - 0.0825 * x295 * x76
    x297 = x229 * x82 + x295 * x75
    x298 = -x273
    x299 = -0.0825 * x273 + x289 * x76 + x290 * x75
    x300 = torch.cos(q[:, 5])
    x301 = torch.sin(q[:, 5])
    x302 = x169 * x300 + x179 * x301
    x303 = dq[:, 5] * x302 + x166 * x300 + x194 * x301
    x304 = -x301
    x305 = x168 * x300 + x169 * x301
    x306 = dq[:, 5] - x165
    x307 = x305 * x306
    x308 = x300 * x303 + x304 * x307
    x309 = x302 * x305
    x310 = -x309
    x311 = -x310
    x312 = x162 * x308 + x167 * x311
    x313 = x300 * x307 + x301 * x303
    x314 = x312 * x75 + x313 * x82
    x315 = x167 * x308 + x175 * x311
    x316 = -x315
    x317 = x312 * x76 + x313 * x75
    x318 = -x307
    x319 = -x305
    x320 = dq[:, 5] * x319 + x194 * x300 + x269 * x301
    x321 = x318 + x320
    x322 = x302 * x306
    x323 = x303 + x322
    x324 = x300 * x321 + x304 * x323
    x325 = x302**2
    x326 = -x325
    x327 = x305**2
    x328 = x326 + x327
    x329 = -x328
    x330 = x162 * x324 + x167 * x329
    x331 = x300 * x323 + x301 * x321
    x332 = x330 * x75 + x331 * x82
    x333 = x167 * x324 + x175 * x329
    x334 = -x333
    x335 = x330 * x76 + x331 * x75
    x336 = ddq[:, 5] + x210
    x337 = x309 + x336
    x338 = x300 * x337
    x339 = x306**2
    x340 = -x327
    x341 = x339 + x340
    x342 = x304 * x341 + x338
    x343 = -x322
    x344 = x303 + x343
    x345 = -x344
    x346 = x162 * x342 + x167 * x345
    x347 = x300 * x341 + x301 * x337
    x348 = x346 * x75 + x347 * x82
    x349 = x167 * x342 + x175 * x345
    x350 = -x349
    x351 = x346 * x76 + x347 * x75
    x352 = -x320
    x353 = x300 * x343 + x301 * x352
    x354 = x162 * x353 + x167 * x310
    x355 = x300 * x320 + x301 * x343
    x356 = x354 * x75 + x355 * x82
    x357 = x167 * x353 + x175 * x310
    x358 = -x357
    x359 = x354 * x76 + x355 * x75
    x360 = -x339
    x361 = x325 + x360
    x362 = x310 + x336
    x363 = x300 * x361 + x304 * x362
    x364 = x307 + x320
    x365 = -x364
    x366 = x162 * x363 + x167 * x365
    x367 = x300 * x362 + x301 * x361
    x368 = x366 * x75 + x367 * x82
    x369 = x167 * x363 + x175 * x365
    x370 = -x369
    x371 = x366 * x76 + x367 * x75
    x372 = x300 * x322 + x304 * x318
    x373 = -x336
    x374 = x162 * x372 + x167 * x373
    x375 = x300 * x318 + x301 * x322
    x376 = x374 * x75 + x375 * x82
    x377 = x167 * x372 + x175 * x373
    x378 = -x377
    x379 = x374 * x76 + x375 * x75
    x380 = -x272
    x381 = x304 * x380
    x382 = x229 * x300 + x256 * x301
    x383 = -x382
    x384 = x326 + x360
    x385 = x300 * x384 + x304 * x337
    x386 = x163 * x385
    x387 = x307 + x352
    x388 = -x387
    x389 = x162 * x388
    x390 = x162 * x381 + x167 * x383 - 0.384 * x386 - 0.384 * x389
    x391 = x300 * x380
    x392 = -0.0825 * x386 - 0.0825 * x389 + x391
    x393 = x390 * x75 + x392 * x82
    x394 = x301 * x384 + x338
    x395 = -0.0825 * x394
    x396 = x162 * x385
    x397 = x167 * x381 + x175 * x383 + x241 * x388 + x395 - 0.384 * x396
    x398 = x167 * x388 + x396
    x399 = x395 * x75 - x397 - 0.0825 * x398 * x76
    x400 = x394 * x82 + x398 * x75
    x401 = x167 * x385 + x175 * x388
    x402 = -x401
    x403 = x390 * x76 + x392 * x75 - 0.0825 * x401
    x404 = x272 * x300
    x405 = x229 * x301 + x255 * x300
    x406 = -x405
    x407 = -x406
    x408 = x309 + x373
    x409 = x340 + x360
    x410 = x300 * x408 + x304 * x409
    x411 = x163 * x410
    x412 = -x323
    x413 = x162 * x412
    x414 = x162 * x404 + x167 * x407 - 0.384 * x411 - 0.384 * x413
    x415 = x272 * x301
    x416 = -0.0825 * x411 - 0.0825 * x413 + x415
    x417 = x414 * x75 + x416 * x82
    x418 = x300 * x409 + x301 * x408
    x419 = -0.0825 * x418
    x420 = x162 * x410
    x421 = x167 * x404 + x175 * x407 + x241 * x412 + x419 - 0.384 * x420
    x422 = x167 * x412 + x420
    x423 = x419 * x75 - x421 - 0.0825 * x422 * x76
    x424 = x418 * x82 + x422 * x75
    x425 = x167 * x410 + x175 * x412
    x426 = -x425
    x427 = x414 * x76 + x416 * x75 - 0.0825 * x425
    x428 = x300 * x383 + x304 * x405
    x429 = -x303
    x430 = x322 + x429
    x431 = x300 * x364 + x304 * x430
    x432 = x163 * x431
    x433 = -x326 - x340
    x434 = x162 * x433
    x435 = x162 * x428 - 0.384 * x432 - 0.384 * x434
    x436 = x300 * x405 + x301 * x383
    x437 = -0.0825 * x432 - 0.0825 * x434 + x436
    x438 = x435 * x75 + x437 * x82
    x439 = x300 * x430 + x301 * x364
    x440 = -0.0825 * x439
    x441 = x162 * x431
    x442 = x167 * x428 + x241 * x433 + x440 - 0.384 * x441
    x443 = x167 * x433 + x441
    x444 = x440 * x75 - x442 - 0.0825 * x443 * x76
    x445 = x439 * x82 + x443 * x75
    x446 = x167 * x431 + x175 * x433
    x447 = -x446
    x448 = x435 * x76 + x437 * x75 - 0.0825 * x446
    x449 = x162 * x380
    x450 = x163 * x436
    x451 = -0.384 * x449 - 0.384 * x450
    x452 = -0.0825 * x449 - 0.0825 * x450
    x453 = x451 * x75 + x452 * x82
    x454 = x300 * x382 + x301 * x405
    x455 = -0.0825 * x454
    x456 = x162 * x436
    x457 = x241 * x380 + x455 - 0.384 * x456
    x458 = x167 * x380 + x456
    x459 = x455 * x75 - x457 - 0.0825 * x458 * x76
    x460 = x454 * x82 + x458 * x75
    x461 = x167 * x436 + x175 * x380
    x462 = -x461
    x463 = x451 * x76 + x452 * x75 - 0.0825 * x461
    x464 = torch.cos(q[:, 6])
    x465 = torch.sin(q[:, 6])
    x466 = x306 * x464 + x319 * x465
    x467 = dq[:, 6] * x466 + x303 * x464 + x336 * x465
    x468 = -x465
    x469 = x305 * x464 + x306 * x465
    x470 = dq[:, 6] - x302
    x471 = x469 * x470
    x472 = x464 * x467 + x468 * x471
    x473 = x466 * x469
    x474 = -x473
    x475 = -x474
    x476 = x300 * x472 + x304 * x475
    x477 = x464 * x471 + x465 * x467
    x478 = -x477
    x479 = x162 * x476 + x167 * x478
    x480 = x300 * x475 + x301 * x472
    x481 = x479 * x75 + x480 * x82
    x482 = x167 * x476 + x175 * x478
    x483 = -x482
    x484 = x479 * x76 + x480 * x75
    x485 = -x471
    x486 = -dq[:, 6] * x469 + x336 * x464 + x429 * x465
    x487 = x485 + x486
    x488 = x466 * x470
    x489 = x467 + x488
    x490 = x464 * x487 + x468 * x489
    x491 = x466**2
    x492 = -x491
    x493 = x469**2
    x494 = x492 + x493
    x495 = -x494
    x496 = x300 * x490 + x304 * x495
    x497 = x464 * x489 + x465 * x487
    x498 = -x497
    x499 = x162 * x496 + x167 * x498
    x500 = x300 * x495 + x301 * x490
    x501 = x499 * x75 + x500 * x82
    x502 = x167 * x496 + x175 * x498
    x503 = -x502
    x504 = x499 * x76 + x500 * x75
    x505 = ddq[:, 6] + x352
    x506 = x473 + x505
    x507 = x464 * x506
    x508 = x470**2
    x509 = -x493
    x510 = x508 + x509
    x511 = x468 * x510 + x507
    x512 = -x488
    x513 = x467 + x512
    x514 = -x513
    x515 = x300 * x511 + x304 * x514
    x516 = x464 * x510 + x465 * x506
    x517 = -x516
    x518 = x162 * x515 + x167 * x517
    x519 = x300 * x514 + x301 * x511
    x520 = x518 * x75 + x519 * x82
    x521 = x167 * x515 + x175 * x517
    x522 = -x521
    x523 = x518 * x76 + x519 * x75
    x524 = -x486
    x525 = x464 * x512 + x465 * x524
    x526 = x300 * x525 + x304 * x474
    x527 = x464 * x486 + x465 * x512
    x528 = -x527
    x529 = x162 * x526 + x167 * x528
    x530 = x300 * x474 + x301 * x525
    x531 = x529 * x75 + x530 * x82
    x532 = x167 * x526 + x175 * x528
    x533 = -x532
    x534 = x529 * x76 + x530 * x75
    x535 = -x508
    x536 = x491 + x535
    x537 = x474 + x505
    x538 = x464 * x536 + x468 * x537
    x539 = x471 + x486
    x540 = -x539
    x541 = x300 * x538 + x304 * x540
    x542 = x464 * x537 + x465 * x536
    x543 = -x542
    x544 = x162 * x541 + x167 * x543
    x545 = x300 * x540 + x301 * x538
    x546 = x544 * x75 + x545 * x82
    x547 = x167 * x541 + x175 * x543
    x548 = -x547
    x549 = x544 * x76 + x545 * x75
    x550 = x464 * x488 + x468 * x485
    x551 = -x505
    x552 = x300 * x550 + x304 * x551
    x553 = x464 * x485 + x465 * x488
    x554 = -x553
    x555 = x162 * x552 + x167 * x554
    x556 = x300 * x551 + x301 * x550
    x557 = x555 * x75 + x556 * x82
    x558 = x167 * x552 + x175 * x554
    x559 = -x558
    x560 = x555 * x76 + x556 * x75
    x561 = -0.088 * x337 + x383
    x562 = -x561
    x563 = x468 * x562
    x564 = x272 + 0.088 * x387
    x565 = 0.088 * x384 + x405
    x566 = x464 * x564 + x468 * x565
    x567 = -x566
    x568 = x492 + x535
    x569 = x465 * x568
    x570 = -0.088 * x507 + x567 - 0.088 * x569
    x571 = x300 * x563 + x304 * x570
    x572 = x471 + x524
    x573 = x464 * x562 - 0.088 * x572
    x574 = -x573
    x575 = x464 * x568 + x468 * x506
    x576 = -x572
    x577 = x300 * x575 + x304 * x576
    x578 = x163 * x577
    x579 = -x507 - x569
    x580 = x162 * x579
    x581 = x162 * x571 + x167 * x574 - 0.384 * x578 - 0.384 * x580
    x582 = x300 * x570 + x301 * x563
    x583 = -0.0825 * x578 - 0.0825 * x580 + x582
    x584 = x581 * x75 + x583 * x82
    x585 = x300 * x576 + x301 * x575
    x586 = -0.0825 * x585
    x587 = x162 * x577
    x588 = x167 * x571 + x175 * x574 + x241 * x579 + x586 - 0.384 * x587
    x589 = x167 * x579 + x587
    x590 = x586 * x75 - x588 - 0.0825 * x589 * x76
    x591 = x585 * x82 + x589 * x75
    x592 = x167 * x577 + x175 * x579
    x593 = -x592
    x594 = x581 * x76 + x583 * x75 - 0.0825 * x592
    x595 = x464 * x561
    x596 = x464 * x565 + x465 * x564
    x597 = -x596
    x598 = x473 + x551
    x599 = x465 * x598
    x600 = x509 + x535
    x601 = x464 * x600
    x602 = -x597 - 0.088 * x599 - 0.088 * x601
    x603 = x300 * x595 + x304 * x602
    x604 = x465 * x561 - 0.088 * x489
    x605 = -x604
    x606 = x464 * x598 + x468 * x600
    x607 = -x489
    x608 = x300 * x606 + x304 * x607
    x609 = x163 * x608
    x610 = -x599 - x601
    x611 = x162 * x610
    x612 = x162 * x603 + x167 * x605 - 0.384 * x609 - 0.384 * x611
    x613 = x300 * x602 + x301 * x595
    x614 = -0.0825 * x609 - 0.0825 * x611 + x613
    x615 = x612 * x75 + x614 * x82
    x616 = x300 * x607 + x301 * x606
    x617 = -0.0825 * x616
    x618 = x162 * x608
    x619 = x167 * x603 + x175 * x605 + x241 * x610 + x617 - 0.384 * x618
    x620 = x167 * x610 + x618
    x621 = x617 * x75 - x619 - 0.0825 * x620 * x76
    x622 = x616 * x82 + x620 * x75
    x623 = x167 * x608 + x175 * x610
    x624 = -x623
    x625 = x612 * x76 + x614 * x75 - 0.0825 * x623
    x626 = x464 * x567 + x468 * x596
    x627 = x465 * x539
    x628 = -x467 + x488
    x629 = x464 * x628
    x630 = -0.088 * x627 - 0.088 * x629
    x631 = x300 * x626 + x304 * x630
    x632 = x492 + x509
    x633 = x464 * x596
    x634 = x465 * x567 - 0.088 * x632 + x633
    x635 = -x634
    x636 = x464 * x539 + x468 * x628
    x637 = -x632
    x638 = x300 * x636 + x304 * x637
    x639 = x163 * x638
    x640 = -x627 - x629
    x641 = x162 * x640
    x642 = x162 * x631 + x167 * x635 - 0.384 * x639 - 0.384 * x641
    x643 = x300 * x630 + x301 * x626
    x644 = -0.0825 * x639 - 0.0825 * x641 + x643
    x645 = x642 * x75 + x644 * x82
    x646 = x300 * x637 + x301 * x636
    x647 = -0.0825 * x646
    x648 = x162 * x638
    x649 = x167 * x631 + x175 * x635 + x241 * x640 + x647 - 0.384 * x648
    x650 = x167 * x640 + x648
    x651 = x647 * x75 - x649 - 0.0825 * x650 * x76
    x652 = x646 * x82 + x650 * x75
    x653 = x167 * x638 + x175 * x640
    x654 = -x653
    x655 = x642 * x76 + x644 * x75 - 0.0825 * x653
    x656 = x465 * x596
    x657 = x464 * x566
    x658 = -0.088 * x656 - 0.088 * x657
    x659 = x304 * x658
    x660 = -0.088 * x561
    x661 = -x660
    x662 = x468 * x566 + x633
    x663 = x300 * x662 + x304 * x562
    x664 = x163 * x663
    x665 = -x656 - x657
    x666 = x162 * x665
    x667 = x162 * x659 + x167 * x661 - 0.384 * x664 - 0.384 * x666
    x668 = x300 * x658
    x669 = -0.0825 * x664 - 0.0825 * x666 + x668
    x670 = x667 * x75 + x669 * x82
    x671 = x300 * x562 + x301 * x662
    x672 = -0.0825 * x671
    x673 = x162 * x663
    x674 = x167 * x659 + x175 * x661 + x241 * x665 + x672 - 0.384 * x673
    x675 = x167 * x665 + x673
    x676 = x672 * x75 - x674 - 0.0825 * x675 * x76
    x677 = x671 * x82 + x675 * x75
    x678 = x167 * x663 + x175 * x665
    x679 = -x678
    x680 = x667 * x76 + x669 * x75 - 0.0825 * x678
    x681 = x29 * x49
    x682 = x28 * x70
    #
    H_num[:, 0] = 0
    H_num[:, 1] = 0
    H_num[:, 2] = 0
    H_num[:, 3] = 0
    H_num[:, 4] = 0
    H_num[:, 5] = ddq[:, 0]
    H_num[:, 6] = 0
    H_num[:, 7] = 0
    H_num[:, 8] = 0
    H_num[:, 9] = 0
    H_num[:, 10] = x0 * x7 + x11
    H_num[:, 11] = x13 * (x10 + x12) + x14 * x15
    H_num[:, 12] = x13 * (ddq[:, 1] + x16) + x14 * (x17 + x19)
    H_num[:, 13] = x13 * x21 + x2 * x20
    H_num[:, 14] = x13 * (x22 + x23) + x14 * (ddq[:, 1] + x24)
    H_num[:, 15] = -x11 + x13 * x5
    H_num[:, 16] = 0
    H_num[:, 17] = 0
    H_num[:, 18] = x13 * x27 + x14 * x25
    H_num[:, 19] = 0
    H_num[:, 20] = x13 * (x28 * x32 + x30 * x35) - x14 * x37
    H_num[:, 21] = x13 * (x28 * x41 + x30 * x43) - x14 * x47
    H_num[:, 22] = x13 * (x30 * x53 + x50) - x14 * x55
    H_num[:, 23] = x13 * (x28 * x54 + x30 * x40) + x14 * x37
    H_num[:, 24] = x13 * (x28 * x57 + x30 * x58) - x14 * x59
    H_num[:, 25] = x13 * (x28 * x42 + x30 * x38) + x14 * x60
    H_num[:, 26] = x13 * (-0.316 * x29 * x67 + x30 * x66 - 0.316 * x50) + x14 * x64
    H_num[:, 27] = x13 * (x28 * x65 - 0.316 * x28 * x69 - 0.316 * x29 * x68) - x14 * x71
    H_num[:, 28] = x13 * (x28 * x64 - 0.316 * x28 * x73 - 0.316 * x29 * x59 + x30 * x70)
    H_num[:, 29] = x13 * (-0.316 * x28 * x63 + x70 * x74)
    H_num[:, 30] = x13 * (x28 * x83 + x30 * x86) - x14 * x87
    H_num[:, 31] = -x100 * x14 + x13 * (x28 * x94 + x30 * x99)
    H_num[:, 32] = -x111 * x14 + x13 * (x107 * x28 + x110 * x30)
    H_num[:, 33] = -x114 * x14 + x13 * (x113 * x28 + x30 * x85)
    H_num[:, 34] = -x121 * x14 + x13 * (x118 * x28 + x120 * x30)
    H_num[:, 35] = -x124 * x14 + x13 * (x122 * x28 + x123 * x30)
    H_num[:, 36] = x13 * (x127 * x28 + x135 * x30 + x136 * x74 - 0.316 * x138 * x28) - x14 * x140
    H_num[:, 37] = x13 * (x141 * x28 + x146 * x30 + x147 * x74 - 0.316 * x148 * x28) - x14 * x149
    H_num[:, 38] = x13 * (x150 * x28 + x153 * x30 + x154 * x74 - 0.316 * x156 * x28) - x14 * x158
    H_num[:, 39] = x13 * (-0.316 * x126 * x28 + x159 * x30 + x160 * x74) - x14 * x161
    H_num[:, 40] = x13 * (x174 * x28 + x177 * x30) - x14 * x178
    H_num[:, 41] = x13 * (x190 * x28 + x192 * x30) - x14 * x193
    H_num[:, 42] = x13 * (x203 * x28 + x206 * x30) - x14 * x207
    H_num[:, 43] = x13 * (x209 * x28 + x212 * x30) - x14 * x213
    H_num[:, 44] = x13 * (x219 * x28 + x221 * x30) - x14 * x222
    H_num[:, 45] = x13 * (x225 * x28 + x227 * x30) - x14 * x228
    H_num[:, 46] = x13 * (x238 * x28 + x245 * x30 + x246 * x74 - 0.316 * x248 * x28) - x14 * x249
    H_num[:, 47] = x13 * (x258 * x28 + x263 * x30 + x264 * x74 - 0.316 * x266 * x28) - x14 * x267
    H_num[:, 48] = x13 * (x276 * x28 - 0.316 * x28 * x285 + x282 * x30 + x283 * x74) - x14 * x286
    H_num[:, 49] = x13 * (x28 * x291 - 0.316 * x28 * x298 + x296 * x30 + x297 * x74) - x14 * x299
    H_num[:, 50] = x13 * (x28 * x314 + x30 * x316) - x14 * x317
    H_num[:, 51] = x13 * (x28 * x332 + x30 * x334) - x14 * x335
    H_num[:, 52] = x13 * (x28 * x348 + x30 * x350) - x14 * x351
    H_num[:, 53] = x13 * (x28 * x356 + x30 * x358) - x14 * x359
    H_num[:, 54] = x13 * (x28 * x368 + x30 * x370) - x14 * x371
    H_num[:, 55] = x13 * (x28 * x376 + x30 * x378) - x14 * x379
    H_num[:, 56] = x13 * (x28 * x393 - 0.316 * x28 * x402 + x30 * x399 + x400 * x74) - x14 * x403
    H_num[:, 57] = x13 * (x28 * x417 - 0.316 * x28 * x426 + x30 * x423 + x424 * x74) - x14 * x427
    H_num[:, 58] = x13 * (x28 * x438 - 0.316 * x28 * x447 + x30 * x444 + x445 * x74) - x14 * x448
    H_num[:, 59] = x13 * (x28 * x453 - 0.316 * x28 * x462 + x30 * x459 + x460 * x74) - x14 * x463
    H_num[:, 60] = x13 * (x28 * x481 + x30 * x483) - x14 * x484
    H_num[:, 61] = x13 * (x28 * x501 + x30 * x503) - x14 * x504
    H_num[:, 62] = x13 * (x28 * x520 + x30 * x522) - x14 * x523
    H_num[:, 63] = x13 * (x28 * x531 + x30 * x533) - x14 * x534
    H_num[:, 64] = x13 * (x28 * x546 + x30 * x548) - x14 * x549
    H_num[:, 65] = x13 * (x28 * x557 + x30 * x559) - x14 * x560
    H_num[:, 66] = x13 * (x28 * x584 - 0.316 * x28 * x593 + x30 * x590 + x591 * x74) - x14 * x594
    H_num[:, 67] = x13 * (x28 * x615 - 0.316 * x28 * x624 + x30 * x621 + x622 * x74) - x14 * x625
    H_num[:, 68] = x13 * (x28 * x645 - 0.316 * x28 * x654 + x30 * x651 + x652 * x74) - x14 * x655
    H_num[:, 69] = x13 * (x28 * x670 - 0.316 * x28 * x679 + x30 * x676 + x677 * x74) - x14 * x680
    H_num[:, 70] = 0
    H_num[:, 71] = 0
    H_num[:, 72] = 0
    H_num[:, 73] = 0
    H_num[:, 74] = 0
    H_num[:, 75] = 0
    H_num[:, 76] = 0
    H_num[:, 77] = 0
    H_num[:, 78] = 0
    H_num[:, 79] = 0
    H_num[:, 80] = x24
    H_num[:, 81] = x18 - x23
    H_num[:, 82] = x21 + x6
    H_num[:, 83] = x16
    H_num[:, 84] = x12 + x9
    H_num[:, 85] = ddq[:, 1]
    H_num[:, 86] = x26
    H_num[:, 87] = -x25
    H_num[:, 88] = 0
    H_num[:, 89] = 0
    H_num[:, 90] = x28 * x35 + x29 * x32
    H_num[:, 91] = x28 * x43 + x29 * x41
    H_num[:, 92] = x28 * x53 + x681
    H_num[:, 93] = x28 * x40 + x29 * x54
    H_num[:, 94] = x28 * x58 + x29 * x57
    H_num[:, 95] = x28 * x38 + x29 * x42
    H_num[:, 96] = x28 * x66 + 0.316 * x28 * x67 - 0.316 * x681
    H_num[:, 97] = 0.316 * x28 * x68 + x29 * x65 + x69 * x74
    H_num[:, 98] = 0.316 * x28 * x59 + x29 * x64 + x682 + x73 * x74
    H_num[:, 99] = x63 * x74 + 0.316 * x682
    H_num[:, 100] = x28 * x86 + x29 * x83
    H_num[:, 101] = x28 * x99 + x29 * x94
    H_num[:, 102] = x107 * x29 + x110 * x28
    H_num[:, 103] = x113 * x29 + x28 * x85
    H_num[:, 104] = x118 * x29 + x120 * x28
    H_num[:, 105] = x122 * x29 + x123 * x28
    H_num[:, 106] = x127 * x29 + x135 * x28 + 0.316 * x136 * x28 + x138 * x74
    H_num[:, 107] = x141 * x29 + x146 * x28 + 0.316 * x147 * x28 + x148 * x74
    H_num[:, 108] = x150 * x29 + x153 * x28 + 0.316 * x154 * x28 + x156 * x74
    H_num[:, 109] = x126 * x74 + x159 * x28 + 0.316 * x160 * x28
    H_num[:, 110] = x174 * x29 + x177 * x28
    H_num[:, 111] = x190 * x29 + x192 * x28
    H_num[:, 112] = x203 * x29 + x206 * x28
    H_num[:, 113] = x209 * x29 + x212 * x28
    H_num[:, 114] = x219 * x29 + x221 * x28
    H_num[:, 115] = x225 * x29 + x227 * x28
    H_num[:, 116] = x238 * x29 + x245 * x28 + 0.316 * x246 * x28 + x248 * x74
    H_num[:, 117] = x258 * x29 + x263 * x28 + 0.316 * x264 * x28 + x266 * x74
    H_num[:, 118] = x276 * x29 + x28 * x282 + 0.316 * x28 * x283 + x285 * x74
    H_num[:, 119] = x28 * x296 + 0.316 * x28 * x297 + x29 * x291 + x298 * x74
    H_num[:, 120] = x28 * x316 + x29 * x314
    H_num[:, 121] = x28 * x334 + x29 * x332
    H_num[:, 122] = x28 * x350 + x29 * x348
    H_num[:, 123] = x28 * x358 + x29 * x356
    H_num[:, 124] = x28 * x370 + x29 * x368
    H_num[:, 125] = x28 * x378 + x29 * x376
    H_num[:, 126] = x28 * x399 + 0.316 * x28 * x400 + x29 * x393 + x402 * x74
    H_num[:, 127] = x28 * x423 + 0.316 * x28 * x424 + x29 * x417 + x426 * x74
    H_num[:, 128] = x28 * x444 + 0.316 * x28 * x445 + x29 * x438 + x447 * x74
    H_num[:, 129] = x28 * x459 + 0.316 * x28 * x460 + x29 * x453 + x462 * x74
    H_num[:, 130] = x28 * x483 + x29 * x481
    H_num[:, 131] = x28 * x503 + x29 * x501
    H_num[:, 132] = x28 * x522 + x29 * x520
    H_num[:, 133] = x28 * x533 + x29 * x531
    H_num[:, 134] = x28 * x548 + x29 * x546
    H_num[:, 135] = x28 * x559 + x29 * x557
    H_num[:, 136] = x28 * x590 + 0.316 * x28 * x591 + x29 * x584 + x593 * x74
    H_num[:, 137] = x28 * x621 + 0.316 * x28 * x622 + x29 * x615 + x624 * x74
    H_num[:, 138] = x28 * x651 + 0.316 * x28 * x652 + x29 * x645 + x654 * x74
    H_num[:, 139] = x28 * x676 + 0.316 * x28 * x677 + x29 * x670 + x679 * x74
    H_num[:, 140] = 0
    H_num[:, 141] = 0
    H_num[:, 142] = 0
    H_num[:, 143] = 0
    H_num[:, 144] = 0
    H_num[:, 145] = 0
    H_num[:, 146] = 0
    H_num[:, 147] = 0
    H_num[:, 148] = 0
    H_num[:, 149] = 0
    H_num[:, 150] = 0
    H_num[:, 151] = 0
    H_num[:, 152] = 0
    H_num[:, 153] = 0
    H_num[:, 154] = 0
    H_num[:, 155] = 0
    H_num[:, 156] = 0
    H_num[:, 157] = 0
    H_num[:, 158] = 0
    H_num[:, 159] = 0
    H_num[:, 160] = x37
    H_num[:, 161] = x47
    H_num[:, 162] = x55
    H_num[:, 163] = x36
    H_num[:, 164] = x59
    H_num[:, 165] = x48
    H_num[:, 166] = x63
    H_num[:, 167] = x71
    H_num[:, 168] = 0
    H_num[:, 169] = 0
    H_num[:, 170] = x87
    H_num[:, 171] = x100
    H_num[:, 172] = x111
    H_num[:, 173] = x114
    H_num[:, 174] = x121
    H_num[:, 175] = x124
    H_num[:, 176] = x140
    H_num[:, 177] = x149
    H_num[:, 178] = x158
    H_num[:, 179] = x161
    H_num[:, 180] = x178
    H_num[:, 181] = x193
    H_num[:, 182] = x207
    H_num[:, 183] = x213
    H_num[:, 184] = x222
    H_num[:, 185] = x228
    H_num[:, 186] = x249
    H_num[:, 187] = x267
    H_num[:, 188] = x286
    H_num[:, 189] = x299
    H_num[:, 190] = x317
    H_num[:, 191] = x335
    H_num[:, 192] = x351
    H_num[:, 193] = x359
    H_num[:, 194] = x371
    H_num[:, 195] = x379
    H_num[:, 196] = x403
    H_num[:, 197] = x427
    H_num[:, 198] = x448
    H_num[:, 199] = x463
    H_num[:, 200] = x484
    H_num[:, 201] = x504
    H_num[:, 202] = x523
    H_num[:, 203] = x534
    H_num[:, 204] = x549
    H_num[:, 205] = x560
    H_num[:, 206] = x594
    H_num[:, 207] = x625
    H_num[:, 208] = x655
    H_num[:, 209] = x680
    H_num[:, 210] = 0
    H_num[:, 211] = 0
    H_num[:, 212] = 0
    H_num[:, 213] = 0
    H_num[:, 214] = 0
    H_num[:, 215] = 0
    H_num[:, 216] = 0
    H_num[:, 217] = 0
    H_num[:, 218] = 0
    H_num[:, 219] = 0
    H_num[:, 220] = 0
    H_num[:, 221] = 0
    H_num[:, 222] = 0
    H_num[:, 223] = 0
    H_num[:, 224] = 0
    H_num[:, 225] = 0
    H_num[:, 226] = 0
    H_num[:, 227] = 0
    H_num[:, 228] = 0
    H_num[:, 229] = 0
    H_num[:, 230] = 0
    H_num[:, 231] = 0
    H_num[:, 232] = 0
    H_num[:, 233] = 0
    H_num[:, 234] = 0
    H_num[:, 235] = 0
    H_num[:, 236] = 0
    H_num[:, 237] = 0
    H_num[:, 238] = 0
    H_num[:, 239] = 0
    H_num[:, 240] = x85
    H_num[:, 241] = x98
    H_num[:, 242] = x109
    H_num[:, 243] = x84
    H_num[:, 244] = x119
    H_num[:, 245] = x102
    H_num[:, 246] = x130
    H_num[:, 247] = x143
    H_num[:, 248] = 0
    H_num[:, 249] = 0
    H_num[:, 250] = x176
    H_num[:, 251] = x191
    H_num[:, 252] = x205
    H_num[:, 253] = x211
    H_num[:, 254] = x220
    H_num[:, 255] = x226
    H_num[:, 256] = x243
    H_num[:, 257] = x261
    H_num[:, 258] = x280
    H_num[:, 259] = x294
    H_num[:, 260] = x315
    H_num[:, 261] = x333
    H_num[:, 262] = x349
    H_num[:, 263] = x357
    H_num[:, 264] = x369
    H_num[:, 265] = x377
    H_num[:, 266] = x397
    H_num[:, 267] = x421
    H_num[:, 268] = x442
    H_num[:, 269] = x457
    H_num[:, 270] = x482
    H_num[:, 271] = x502
    H_num[:, 272] = x521
    H_num[:, 273] = x532
    H_num[:, 274] = x547
    H_num[:, 275] = x558
    H_num[:, 276] = x588
    H_num[:, 277] = x619
    H_num[:, 278] = x649
    H_num[:, 279] = x674
    H_num[:, 280] = 0
    H_num[:, 281] = 0
    H_num[:, 282] = 0
    H_num[:, 283] = 0
    H_num[:, 284] = 0
    H_num[:, 285] = 0
    H_num[:, 286] = 0
    H_num[:, 287] = 0
    H_num[:, 288] = 0
    H_num[:, 289] = 0
    H_num[:, 290] = 0
    H_num[:, 291] = 0
    H_num[:, 292] = 0
    H_num[:, 293] = 0
    H_num[:, 294] = 0
    H_num[:, 295] = 0
    H_num[:, 296] = 0
    H_num[:, 297] = 0
    H_num[:, 298] = 0
    H_num[:, 299] = 0
    H_num[:, 300] = 0
    H_num[:, 301] = 0
    H_num[:, 302] = 0
    H_num[:, 303] = 0
    H_num[:, 304] = 0
    H_num[:, 305] = 0
    H_num[:, 306] = 0
    H_num[:, 307] = 0
    H_num[:, 308] = 0
    H_num[:, 309] = 0
    H_num[:, 310] = 0
    H_num[:, 311] = 0
    H_num[:, 312] = 0
    H_num[:, 313] = 0
    H_num[:, 314] = 0
    H_num[:, 315] = 0
    H_num[:, 316] = 0
    H_num[:, 317] = 0
    H_num[:, 318] = 0
    H_num[:, 319] = 0
    H_num[:, 320] = x173
    H_num[:, 321] = x189
    H_num[:, 322] = x202
    H_num[:, 323] = x172
    H_num[:, 324] = x218
    H_num[:, 325] = x194
    H_num[:, 326] = x236
    H_num[:, 327] = x256
    H_num[:, 328] = 0
    H_num[:, 329] = 0
    H_num[:, 330] = x313
    H_num[:, 331] = x331
    H_num[:, 332] = x347
    H_num[:, 333] = x355
    H_num[:, 334] = x367
    H_num[:, 335] = x375
    H_num[:, 336] = x391
    H_num[:, 337] = x415
    H_num[:, 338] = x436
    H_num[:, 339] = 0
    H_num[:, 340] = x480
    H_num[:, 341] = x500
    H_num[:, 342] = x519
    H_num[:, 343] = x530
    H_num[:, 344] = x545
    H_num[:, 345] = x556
    H_num[:, 346] = x582
    H_num[:, 347] = x613
    H_num[:, 348] = x643
    H_num[:, 349] = x668
    H_num[:, 350] = 0
    H_num[:, 351] = 0
    H_num[:, 352] = 0
    H_num[:, 353] = 0
    H_num[:, 354] = 0
    H_num[:, 355] = 0
    H_num[:, 356] = 0
    H_num[:, 357] = 0
    H_num[:, 358] = 0
    H_num[:, 359] = 0
    H_num[:, 360] = 0
    H_num[:, 361] = 0
    H_num[:, 362] = 0
    H_num[:, 363] = 0
    H_num[:, 364] = 0
    H_num[:, 365] = 0
    H_num[:, 366] = 0
    H_num[:, 367] = 0
    H_num[:, 368] = 0
    H_num[:, 369] = 0
    H_num[:, 370] = 0
    H_num[:, 371] = 0
    H_num[:, 372] = 0
    H_num[:, 373] = 0
    H_num[:, 374] = 0
    H_num[:, 375] = 0
    H_num[:, 376] = 0
    H_num[:, 377] = 0
    H_num[:, 378] = 0
    H_num[:, 379] = 0
    H_num[:, 380] = 0
    H_num[:, 381] = 0
    H_num[:, 382] = 0
    H_num[:, 383] = 0
    H_num[:, 384] = 0
    H_num[:, 385] = 0
    H_num[:, 386] = 0
    H_num[:, 387] = 0
    H_num[:, 388] = 0
    H_num[:, 389] = 0
    H_num[:, 390] = 0
    H_num[:, 391] = 0
    H_num[:, 392] = 0
    H_num[:, 393] = 0
    H_num[:, 394] = 0
    H_num[:, 395] = 0
    H_num[:, 396] = 0
    H_num[:, 397] = 0
    H_num[:, 398] = 0
    H_num[:, 399] = 0
    H_num[:, 400] = x310
    H_num[:, 401] = x328
    H_num[:, 402] = x344
    H_num[:, 403] = x309
    H_num[:, 404] = x364
    H_num[:, 405] = x336
    H_num[:, 406] = x382
    H_num[:, 407] = x406
    H_num[:, 408] = 0
    H_num[:, 409] = 0
    H_num[:, 410] = x477
    H_num[:, 411] = x497
    H_num[:, 412] = x516
    H_num[:, 413] = x527
    H_num[:, 414] = x542
    H_num[:, 415] = x553
    H_num[:, 416] = x573
    H_num[:, 417] = x604
    H_num[:, 418] = x634
    H_num[:, 419] = x660
    H_num[:, 420] = 0
    H_num[:, 421] = 0
    H_num[:, 422] = 0
    H_num[:, 423] = 0
    H_num[:, 424] = 0
    H_num[:, 425] = 0
    H_num[:, 426] = 0
    H_num[:, 427] = 0
    H_num[:, 428] = 0
    H_num[:, 429] = 0
    H_num[:, 430] = 0
    H_num[:, 431] = 0
    H_num[:, 432] = 0
    H_num[:, 433] = 0
    H_num[:, 434] = 0
    H_num[:, 435] = 0
    H_num[:, 436] = 0
    H_num[:, 437] = 0
    H_num[:, 438] = 0
    H_num[:, 439] = 0
    H_num[:, 440] = 0
    H_num[:, 441] = 0
    H_num[:, 442] = 0
    H_num[:, 443] = 0
    H_num[:, 444] = 0
    H_num[:, 445] = 0
    H_num[:, 446] = 0
    H_num[:, 447] = 0
    H_num[:, 448] = 0
    H_num[:, 449] = 0
    H_num[:, 450] = 0
    H_num[:, 451] = 0
    H_num[:, 452] = 0
    H_num[:, 453] = 0
    H_num[:, 454] = 0
    H_num[:, 455] = 0
    H_num[:, 456] = 0
    H_num[:, 457] = 0
    H_num[:, 458] = 0
    H_num[:, 459] = 0
    H_num[:, 460] = 0
    H_num[:, 461] = 0
    H_num[:, 462] = 0
    H_num[:, 463] = 0
    H_num[:, 464] = 0
    H_num[:, 465] = 0
    H_num[:, 466] = 0
    H_num[:, 467] = 0
    H_num[:, 468] = 0
    H_num[:, 469] = 0
    H_num[:, 470] = 0
    H_num[:, 471] = 0
    H_num[:, 472] = 0
    H_num[:, 473] = 0
    H_num[:, 474] = 0
    H_num[:, 475] = 0
    H_num[:, 476] = 0
    H_num[:, 477] = 0
    H_num[:, 478] = 0
    H_num[:, 479] = 0
    H_num[:, 480] = x474
    H_num[:, 481] = x494
    H_num[:, 482] = x513
    H_num[:, 483] = x473
    H_num[:, 484] = x539
    H_num[:, 485] = x505
    H_num[:, 486] = x566
    H_num[:, 487] = x597
    H_num[:, 488] = 0
    H_num[:, 489] = 0
    #
    return H_num
