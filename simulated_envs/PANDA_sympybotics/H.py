# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from math import cos, sin


def H(q, dq, ddq):
    #
    H_num = [0] * 490
    #
    x0 = sin(q[1])
    x1 = -dq[0]
    x2 = x0 * x1
    x3 = dq[1] * x2
    x4 = -x3
    x5 = cos(q[1])
    x6 = x4 * x5
    x7 = x1 * x5
    x8 = dq[1] * x7
    x9 = -ddq[0]
    x10 = x0 * x9 + x8
    x11 = -x10
    x12 = dq[0] * dq[1] * x0 + x5 * x9
    x13 = -x0
    x14 = -x5
    x15 = x10 + x8
    x16 = dq[1] ** 2
    x17 = x2**2
    x18 = -x17
    x19 = x2 * x7
    x20 = -x12
    x21 = -x8
    x22 = x7**2
    x23 = -x16
    x24 = -x19
    x25 = -9.81 * x5
    x26 = -x25
    x27 = -9.81 * x0
    x28 = sin(q[2])
    x29 = -x28
    x30 = cos(q[2])
    x31 = dq[1] * x28 + x2 * x30
    x32 = dq[2] - x7
    x33 = x31 * x32
    x34 = dq[1] * x30 + x2 * x29
    x35 = ddq[1] * x28 + dq[2] * x34 + x10 * x30
    x36 = x31 * x34
    x37 = -x36
    x38 = x31**2
    x39 = x34**2
    x40 = -x39
    x41 = x38 + x40
    x42 = -x31
    x43 = ddq[1] * x30 + dq[2] * x42 + x11 * x28
    x44 = -x33
    x45 = x43 + x44
    x46 = x32 * x34
    x47 = x35 + x46
    x48 = -x38
    x49 = x32**2
    x50 = x48 + x49
    x51 = ddq[2] + x20
    x52 = x36 + x51
    x53 = x30 * x52
    x54 = -x46
    x55 = x35 + x54
    x56 = x33 + x43
    x57 = x37 + x51
    x58 = -x49
    x59 = x39 + x58
    x60 = -x51
    x61 = x40 + x58
    x62 = 0.316 * x18 + 0.316 * x23 + x26
    x63 = -x62
    x64 = -0.316 * x15
    x65 = 0.316 * ddq[1] - 0.316 * x19 + x27
    x66 = x29 * x65 + x30 * x64
    x67 = -x66
    x68 = x28 * x64 + x30 * x65
    x69 = -x68
    x70 = x36 + x60
    x71 = x48 + x58
    x72 = -x35
    x73 = x46 + x72
    x74 = -0.316 * x28
    x75 = cos(q[3])
    x76 = sin(q[3])
    x77 = x32 * x75 + x42 * x76
    x78 = dq[3] * x77 + x35 * x75 + x51 * x76
    x79 = x31 * x75 + x32 * x76
    x80 = dq[3] - x34
    x81 = x79 * x80
    x82 = -x76
    x83 = x75 * x78 + x81 * x82
    x84 = x77 * x79
    x85 = -x84
    x86 = -x85
    x87 = x75 * x81 + x76 * x78
    x88 = x79**2
    x89 = x77**2
    x90 = -x89
    x91 = x88 + x90
    x92 = -x91
    x93 = -x81
    x94 = -x79
    x95 = dq[3] * x94 + x51 * x75 + x72 * x76
    x96 = x93 + x95
    x97 = x77 * x80
    x98 = x78 + x97
    x99 = x75 * x96 + x82 * x98
    x100 = x75 * x98 + x76 * x96
    x101 = -x97
    x102 = x101 + x78
    x103 = -x102
    x104 = x80**2
    x105 = -x88
    x106 = x104 + x105
    x107 = -x43
    x108 = ddq[3] + x107
    x109 = x108 + x84
    x110 = x106 * x82 + x109 * x75
    x111 = x106 * x75 + x109 * x76
    x112 = x101 * x76 + x75 * x95
    x113 = -x95
    x114 = x101 * x75 + x113 * x76
    x115 = x81 + x95
    x116 = -x115
    x117 = -x104
    x118 = x117 + x89
    x119 = x108 + x85
    x120 = x118 * x75 + x119 * x82
    x121 = x118 * x76 + x119 * x75
    x122 = x75 * x93 + x76 * x97
    x123 = x75 * x97 + x82 * x93
    x124 = -x108
    x125 = -0.0825 * x52 + x67
    x126 = -x125
    x127 = x113 + x81
    x128 = -0.0825 * x127
    x129 = x126 * x75 + x128
    x130 = x117 + x90
    x131 = x109 * x82 + x130 * x75
    x132 = -x127
    x133 = x126 * x82
    x134 = -0.0825 * x109
    x135 = 0.0825 * x107 + 0.0825 * x33 + x62
    x136 = 0.0825 * x61 + x68
    x137 = x135 * x75 + x136 * x82
    x138 = -x137
    x139 = -0.0825 * x130
    x140 = x134 * x75 + x138 + x139 * x76
    x141 = x125 * x76 - 0.0825 * x98
    x142 = x124 + x84
    x143 = x105 + x117
    x144 = x142 * x75 + x143 * x82
    x145 = x135 * x76 + x136 * x75
    x146 = -x145
    x147 = -0.0825 * x142 * x76 - 0.0825 * x143 * x75 - x146
    x148 = x125 * x75
    x149 = -x98
    x150 = x138 * x75 + x145 * x82
    x151 = -x78
    x152 = x151 + x97
    x153 = x115 * x75 + x152 * x82
    x154 = -0.0825 * x115 * x76 - 0.0825 * x152 * x75
    x155 = x105 + x90
    x156 = -x155
    x157 = x145 * x75
    x158 = x138 * x76 - 0.0825 * x155 + x157
    x159 = -0.0825 * x137 * x75 - 0.0825 * x145 * x76
    x160 = x137 * x82 + x157
    x161 = -0.0825 * x125
    x162 = sin(q[4])
    x163 = -x80
    x164 = cos(q[4])
    x165 = x162 * x163 + x164 * x79
    x166 = x162 * x94 + x163 * x164
    x167 = x165 * x166
    x168 = -x167
    x169 = -x162
    x170 = dq[4] + x77
    x171 = x165 * x170
    x172 = dq[4] * x166 + x124 * x162 + x164 * x78
    x173 = x164 * x172 + x169 * x171
    x174 = x168 * x82 + x173 * x75
    x175 = -x164
    x176 = x169 * x172 + x171 * x175
    x177 = -x176
    x178 = x168 * x75 + x173 * x76
    x179 = x166 * x170
    x180 = x172 + x179
    x181 = -x165
    x182 = dq[4] * x181 + x124 * x164 + x151 * x162
    x183 = -x171
    x184 = x182 + x183
    x185 = x164 * x184 + x169 * x180
    x186 = x166**2
    x187 = -x186
    x188 = x165**2
    x189 = x187 + x188
    x190 = x185 * x76 + x189 * x75
    x191 = x185 * x75 + x189 * x82
    x192 = x169 * x184 + x175 * x180
    x193 = -x192
    x194 = x170**2
    x195 = -x188
    x196 = x194 + x195
    x197 = ddq[4] + x95
    x198 = x167 + x197
    x199 = x164 * x198
    x200 = x169 * x196 + x199
    x201 = -x179
    x202 = x172 + x201
    x203 = x200 * x76 + x202 * x75
    x204 = x200 * x75 + x202 * x82
    x205 = x169 * x198
    x206 = x175 * x196 + x205
    x207 = -x206
    x208 = -x182
    x209 = x164 * x208 + x169 * x201
    x210 = -x209
    x211 = x164 * x201 + x169 * x182
    x212 = x167 * x82 + x211 * x75
    x213 = x167 * x75 + x211 * x76
    x214 = x171 + x182
    x215 = x168 + x197
    x216 = -x194
    x217 = x186 + x216
    x218 = x164 * x217 + x169 * x215
    x219 = x214 * x75 + x218 * x76
    x220 = x214 * x82 + x218 * x75
    x221 = x169 * x217 + x175 * x215
    x222 = -x221
    x223 = x164 * x179 + x169 * x183
    x224 = x197 * x75 + x223 * x76
    x225 = x169 * x179 + x175 * x183
    x226 = -x225
    x227 = -x197
    x228 = x223 * x75 + x227 * x76
    x229 = x171 + x208
    x230 = x187 + x216
    x231 = x164 * x230
    x232 = x205 + x231
    x233 = x229 * x82 + x232 * x75
    x234 = x169 * x230 + x175 * x198
    x235 = -x234
    x236 = x139 + 0.384 * x142 + x145
    x237 = x125 + x128 + 0.384 * x98
    x238 = x169 * x236 + x175 * x237
    x239 = x162 * x230
    x240 = -0.0825 * x199 + x238 - 0.0825 * x239
    x241 = x134 + x137 + 0.384 * x143
    x242 = -x241
    x243 = x169 * x242 - 0.384 * x199 - 0.384 * x239
    x244 = x240 * x82 + x243 * x75
    x245 = -0.0825 * x229
    x246 = 0.384 * x162
    x247 = x175 * x242 + x198 * x246 - 0.384 * x231 + x245
    x248 = -0.0825 * x232 * x76 + x245 * x75 - x247
    x249 = -0.0825 * x234 + x240 * x75 + x243 * x76
    x250 = x195 + x216
    x251 = x164 * x250
    x252 = x167 + x227
    x253 = x162 * x252
    x254 = x164 * x241 - 0.384 * x251 - 0.384 * x253
    x255 = x169 * x252 + x175 * x250
    x256 = x164 * x236 + x169 * x237
    x257 = -x256
    x258 = -0.0825 * x251 - 0.0825 * x253 + x257
    x259 = x254 * x76 - 0.0825 * x255 + x258 * x75
    x260 = x164 * x252
    x261 = -0.0825 * x180
    x262 = x169 * x241 + x246 * x250 - 0.384 * x260 + x261
    x263 = x169 * x250 + x260
    x264 = x261 * x75 - x262 - 0.0825 * x263 * x76
    x265 = -x255
    x266 = x254 * x75 + x258 * x82
    x267 = x180 * x82 + x263 * x75
    x268 = x162 * x214
    x269 = -x172
    x270 = x179 + x269
    x271 = x164 * x270
    x272 = -0.0825 * x268 - 0.0825 * x271
    x273 = x169 * x214 + x175 * x270
    x274 = -x238
    x275 = x164 * x274 + x169 * x256
    x276 = -0.384 * x268 - 0.384 * x271 + x275
    x277 = x272 * x75 - 0.0825 * x273 + x276 * x76
    x278 = x187 + x195
    x279 = x164 * x214
    x280 = x169 * x270 + x279
    x281 = x278 * x82 + x280 * x75
    x282 = x272 * x82 + x276 * x75
    x283 = -0.0825 * x278
    x284 = x169 * x274 + x175 * x256 + x246 * x270 - 0.384 * x279 + x283
    x285 = -0.0825 * x280 * x76 + x283 * x75 - x284
    x286 = -x273
    x287 = x162 * x256
    x288 = x164 * x238
    x289 = -0.384 * x287 - 0.384 * x288
    x290 = -0.0825 * x287 - 0.0825 * x288
    x291 = -0.0825 * x275 + x289 * x76 + x290 * x75
    x292 = x289 * x75 + x290 * x82
    x293 = x164 * x256
    x294 = x169 * x238 + x293
    x295 = x241 * x82 + x294 * x75
    x296 = -x275
    x297 = -0.0825 * x241
    x298 = x238 * x246 - 0.384 * x293 + x297
    x299 = -0.0825 * x294 * x76 + x297 * x75 - x298
    x300 = sin(q[5])
    x301 = cos(q[5])
    x302 = x170 * x301 + x181 * x300
    x303 = dq[5] * x302 + x172 * x301 + x197 * x300
    x304 = x165 * x301 + x170 * x300
    x305 = dq[5] - x166
    x306 = x304 * x305
    x307 = x300 * x303 + x301 * x306
    x308 = -x300
    x309 = x301 * x303 + x306 * x308
    x310 = x302 * x304
    x311 = -x310
    x312 = -x311
    x313 = x164 * x309 + x169 * x312
    x314 = x307 * x75 + x313 * x76
    x315 = x169 * x309 + x175 * x312
    x316 = -x315
    x317 = x307 * x82 + x313 * x75
    x318 = -x306
    x319 = -x304
    x320 = dq[5] * x319 + x197 * x301 + x269 * x300
    x321 = x318 + x320
    x322 = x302 * x305
    x323 = x303 + x322
    x324 = x301 * x321 + x308 * x323
    x325 = x302**2
    x326 = -x325
    x327 = x304**2
    x328 = x326 + x327
    x329 = -x328
    x330 = x164 * x324 + x169 * x329
    x331 = x300 * x321 + x301 * x323
    x332 = x330 * x76 + x331 * x75
    x333 = x169 * x324 + x175 * x329
    x334 = -x333
    x335 = x330 * x75 + x331 * x82
    x336 = -x322
    x337 = x303 + x336
    x338 = -x337
    x339 = ddq[5] + x208
    x340 = x310 + x339
    x341 = x301 * x340
    x342 = -x327
    x343 = x305**2
    x344 = x342 + x343
    x345 = x308 * x344 + x341
    x346 = x164 * x345 + x169 * x338
    x347 = x300 * x340 + x301 * x344
    x348 = x346 * x75 + x347 * x82
    x349 = x169 * x345 + x175 * x338
    x350 = -x349
    x351 = x346 * x76 + x347 * x75
    x352 = x300 * x336 + x301 * x320
    x353 = -x320
    x354 = x300 * x353 + x301 * x336
    x355 = x164 * x354 + x169 * x311
    x356 = x352 * x82 + x355 * x75
    x357 = x169 * x354 + x175 * x311
    x358 = -x357
    x359 = x352 * x75 + x355 * x76
    x360 = x311 + x339
    x361 = -x343
    x362 = x325 + x361
    x363 = x300 * x362 + x301 * x360
    x364 = x306 + x320
    x365 = -x364
    x366 = x301 * x362 + x308 * x360
    x367 = x164 * x366 + x169 * x365
    x368 = x363 * x75 + x367 * x76
    x369 = x363 * x82 + x367 * x75
    x370 = x169 * x366 + x175 * x365
    x371 = -x370
    x372 = x300 * x322 + x301 * x318
    x373 = x301 * x322 + x308 * x318
    x374 = -x339
    x375 = x164 * x373 + x169 * x374
    x376 = x372 * x75 + x375 * x76
    x377 = x372 * x82 + x375 * x75
    x378 = x169 * x373 + x175 * x374
    x379 = -x378
    x380 = x326 + x361
    x381 = x301 * x380 + x308 * x340
    x382 = x306 + x353
    x383 = -x382
    x384 = x169 * x381 + x175 * x383
    x385 = x164 * x383
    x386 = x162 * x381
    x387 = -x274
    x388 = x308 * x387
    x389 = x241 * x301 + x257 * x300
    x390 = -x389
    x391 = x164 * x388 + x169 * x390 - 0.384 * x385 - 0.384 * x386
    x392 = x301 * x387
    x393 = -0.0825 * x385 - 0.0825 * x386 + x392
    x394 = -0.0825 * x384 + x391 * x76 + x393 * x75
    x395 = -x384
    x396 = x391 * x75 + x393 * x82
    x397 = x300 * x380 + x341
    x398 = -0.0825 * x397
    x399 = x164 * x381
    x400 = x169 * x388 + x175 * x390 + x246 * x383 + x398 - 0.384 * x399
    x401 = x169 * x383 + x399
    x402 = x398 * x75 - x400 - 0.0825 * x401 * x76
    x403 = x397 * x82 + x401 * x75
    x404 = x274 * x300
    x405 = x342 + x361
    x406 = x310 + x374
    x407 = x301 * x406 + x308 * x405
    x408 = x162 * x407
    x409 = -x323
    x410 = x164 * x409
    x411 = x404 - 0.0825 * x408 - 0.0825 * x410
    x412 = x169 * x407 + x175 * x409
    x413 = x274 * x301
    x414 = x241 * x300 + x256 * x301
    x415 = -x414
    x416 = -x415
    x417 = x164 * x413 + x169 * x416 - 0.384 * x408 - 0.384 * x410
    x418 = x411 * x75 - 0.0825 * x412 + x417 * x76
    x419 = -x412
    x420 = x164 * x407
    x421 = x169 * x409 + x420
    x422 = x300 * x406 + x301 * x405
    x423 = x421 * x75 + x422 * x82
    x424 = x411 * x82 + x417 * x75
    x425 = -0.0825 * x422
    x426 = x169 * x413 + x175 * x416 + x246 * x409 - 0.384 * x420 + x425
    x427 = -0.0825 * x421 * x76 + x425 * x75 - x426
    x428 = -x303
    x429 = x322 + x428
    x430 = x301 * x364 + x308 * x429
    x431 = x162 * x430
    x432 = -x326 - x342
    x433 = x164 * x432
    x434 = x301 * x390 + x308 * x414
    x435 = x164 * x434 - 0.384 * x431 - 0.384 * x433
    x436 = x300 * x390 + x301 * x414
    x437 = -0.0825 * x431 - 0.0825 * x433 + x436
    x438 = x435 * x75 + x437 * x82
    x439 = x169 * x430 + x175 * x432
    x440 = -x439
    x441 = x300 * x364 + x301 * x429
    x442 = -0.0825 * x441
    x443 = x164 * x430
    x444 = x169 * x434 + x246 * x432 + x442 - 0.384 * x443
    x445 = x169 * x432 + x443
    x446 = x442 * x75 - x444 - 0.0825 * x445 * x76
    x447 = x441 * x82 + x445 * x75
    x448 = x435 * x76 + x437 * x75 - 0.0825 * x439
    x449 = x164 * x436
    x450 = x300 * x414 + x301 * x389
    x451 = -0.0825 * x450
    x452 = x246 * x387 - 0.384 * x449 + x451
    x453 = x169 * x387 + x449
    x454 = x451 * x75 - x452 - 0.0825 * x453 * x76
    x455 = x164 * x387
    x456 = x162 * x436
    x457 = -0.384 * x455 - 0.384 * x456
    x458 = -0.0825 * x455 - 0.0825 * x456
    x459 = x457 * x75 + x458 * x82
    x460 = x169 * x436 + x175 * x387
    x461 = -x460
    x462 = x450 * x82 + x453 * x75
    x463 = x457 * x76 + x458 * x75 - 0.0825 * x460
    x464 = sin(q[6])
    x465 = -x464
    x466 = cos(q[6])
    x467 = x304 * x466 + x305 * x464
    x468 = dq[6] - x302
    x469 = x467 * x468
    x470 = x305 * x466 + x319 * x464
    x471 = dq[6] * x470 + x303 * x466 + x339 * x464
    x472 = x465 * x469 + x466 * x471
    x473 = x467 * x470
    x474 = -x473
    x475 = -x474
    x476 = x300 * x472 + x301 * x475
    x477 = x301 * x472 + x308 * x475
    x478 = x464 * x471 + x466 * x469
    x479 = -x478
    x480 = x164 * x477 + x169 * x479
    x481 = x476 * x75 + x480 * x76
    x482 = x169 * x477 + x175 * x479
    x483 = -x482
    x484 = x476 * x82 + x480 * x75
    x485 = x470**2
    x486 = -x485
    x487 = x467**2
    x488 = x486 + x487
    x489 = -x488
    x490 = -x469
    x491 = -dq[6] * x467 + x339 * x466 + x428 * x464
    x492 = x490 + x491
    x493 = x468 * x470
    x494 = x471 + x493
    x495 = x465 * x494 + x466 * x492
    x496 = x300 * x495 + x301 * x489
    x497 = x464 * x492 + x466 * x494
    x498 = -x497
    x499 = x301 * x495 + x308 * x489
    x500 = x164 * x499 + x169 * x498
    x501 = x496 * x75 + x500 * x76
    x502 = x169 * x499 + x175 * x498
    x503 = -x502
    x504 = x496 * x82 + x500 * x75
    x505 = ddq[6] + x353
    x506 = x473 + x505
    x507 = x466 * x506
    x508 = x468**2
    x509 = -x487
    x510 = x508 + x509
    x511 = x465 * x510 + x507
    x512 = -x493
    x513 = x471 + x512
    x514 = -x513
    x515 = x300 * x511 + x301 * x514
    x516 = x301 * x511 + x308 * x514
    x517 = x464 * x506 + x466 * x510
    x518 = -x517
    x519 = x164 * x516 + x169 * x518
    x520 = x515 * x82 + x519 * x75
    x521 = x169 * x516 + x175 * x518
    x522 = -x521
    x523 = x515 * x75 + x519 * x76
    x524 = -x491
    x525 = x464 * x524 + x466 * x512
    x526 = x300 * x525 + x301 * x474
    x527 = x464 * x512 + x466 * x491
    x528 = -x527
    x529 = x301 * x525 + x308 * x474
    x530 = x164 * x529 + x169 * x528
    x531 = x526 * x75 + x530 * x76
    x532 = x169 * x529 + x175 * x528
    x533 = -x532
    x534 = x526 * x82 + x530 * x75
    x535 = -x508
    x536 = x485 + x535
    x537 = x474 + x505
    x538 = x465 * x537 + x466 * x536
    x539 = x469 + x491
    x540 = -x539
    x541 = x300 * x538 + x301 * x540
    x542 = x464 * x536 + x466 * x537
    x543 = -x542
    x544 = x301 * x538 + x308 * x540
    x545 = x164 * x544 + x169 * x543
    x546 = x541 * x82 + x545 * x75
    x547 = x169 * x544 + x175 * x543
    x548 = -x547
    x549 = x541 * x75 + x545 * x76
    x550 = x464 * x493 + x466 * x490
    x551 = -x550
    x552 = -x505
    x553 = x465 * x490 + x466 * x493
    x554 = x301 * x553 + x308 * x552
    x555 = x164 * x554 + x169 * x551
    x556 = x300 * x553 + x301 * x552
    x557 = x555 * x76 + x556 * x75
    x558 = x169 * x554 + x175 * x551
    x559 = -x558
    x560 = x555 * x75 + x556 * x82
    x561 = x469 + x524
    x562 = -0.088 * x340 + x390
    x563 = -x562
    x564 = x466 * x563 - 0.088 * x561
    x565 = -x564
    x566 = -x561
    x567 = x486 + x535
    x568 = x465 * x506 + x466 * x567
    x569 = x301 * x568 + x308 * x566
    x570 = x162 * x569
    x571 = x464 * x567
    x572 = -x507 - x571
    x573 = x164 * x572
    x574 = x274 + 0.088 * x382
    x575 = 0.088 * x380 + x414
    x576 = x465 * x575 + x466 * x574
    x577 = -x576
    x578 = -0.088 * x507 - 0.088 * x571 + x577
    x579 = x465 * x563
    x580 = x301 * x579 + x308 * x578
    x581 = x164 * x580 + x169 * x565 - 0.384 * x570 - 0.384 * x573
    x582 = x300 * x579 + x301 * x578
    x583 = -0.0825 * x570 - 0.0825 * x573 + x582
    x584 = x169 * x569 + x175 * x572
    x585 = x581 * x76 + x583 * x75 - 0.0825 * x584
    x586 = x164 * x569
    x587 = x169 * x572 + x586
    x588 = x300 * x568 + x301 * x566
    x589 = -0.0825 * x588
    x590 = x169 * x580 + x175 * x565 + x246 * x572 - 0.384 * x586 + x589
    x591 = -0.0825 * x587 * x76 + x589 * x75 - x590
    x592 = -x584
    x593 = x581 * x75 + x583 * x82
    x594 = x587 * x75 + x588 * x82
    x595 = x473 + x552
    x596 = x464 * x595
    x597 = x509 + x535
    x598 = x466 * x597
    x599 = -x596 - x598
    x600 = x164 * x599
    x601 = -x494
    x602 = x465 * x597 + x466 * x595
    x603 = x301 * x602 + x308 * x601
    x604 = x162 * x603
    x605 = x466 * x562
    x606 = x464 * x574 + x466 * x575
    x607 = -x606
    x608 = -0.088 * x596 - 0.088 * x598 - x607
    x609 = x300 * x605 + x301 * x608
    x610 = -0.0825 * x600 - 0.0825 * x604 + x609
    x611 = x169 * x603 + x175 * x599
    x612 = x301 * x605 + x308 * x608
    x613 = x464 * x562 - 0.088 * x494
    x614 = -x613
    x615 = x164 * x612 + x169 * x614 - 0.384 * x600 - 0.384 * x604
    x616 = x610 * x75 - 0.0825 * x611 + x615 * x76
    x617 = x610 * x82 + x615 * x75
    x618 = x300 * x602 + x301 * x601
    x619 = x164 * x603
    x620 = x169 * x599 + x619
    x621 = x618 * x82 + x620 * x75
    x622 = -x611
    x623 = -0.0825 * x618
    x624 = x169 * x612 + x175 * x614 + x246 * x599 - 0.384 * x619 + x623
    x625 = -0.0825 * x620 * x76 + x623 * x75 - x624
    x626 = -x471 + x493
    x627 = x465 * x626 + x466 * x539
    x628 = x486 + x509
    x629 = -x628
    x630 = x300 * x627 + x301 * x629
    x631 = -0.0825 * x630
    x632 = x466 * x606
    x633 = x464 * x577 - 0.088 * x628 + x632
    x634 = -x633
    x635 = x465 * x606 + x466 * x577
    x636 = x464 * x539
    x637 = x466 * x626
    x638 = -0.088 * x636 - 0.088 * x637
    x639 = x301 * x635 + x308 * x638
    x640 = -x636 - x637
    x641 = x301 * x627 + x308 * x629
    x642 = x164 * x641
    x643 = x169 * x639 + x175 * x634 + x246 * x640 + x631 - 0.384 * x642
    x644 = x169 * x640 + x642
    x645 = x631 * x75 - x643 - 0.0825 * x644 * x76
    x646 = x630 * x82 + x644 * x75
    x647 = x169 * x641 + x175 * x640
    x648 = -x647
    x649 = x164 * x640
    x650 = x162 * x641
    x651 = x164 * x639 + x169 * x634 - 0.384 * x649 - 0.384 * x650
    x652 = x300 * x635 + x301 * x638
    x653 = -0.0825 * x649 - 0.0825 * x650 + x652
    x654 = x651 * x75 + x653 * x82
    x655 = -0.0825 * x647 + x651 * x76 + x653 * x75
    x656 = x466 * x576
    x657 = x464 * x606
    x658 = -0.088 * x656 - 0.088 * x657
    x659 = x301 * x658
    x660 = -x656 - x657
    x661 = x164 * x660
    x662 = x465 * x576 + x632
    x663 = x301 * x662 + x308 * x563
    x664 = x162 * x663
    x665 = x659 - 0.0825 * x661 - 0.0825 * x664
    x666 = x308 * x658
    x667 = -0.088 * x562
    x668 = -x667
    x669 = x164 * x666 + x169 * x668 - 0.384 * x661 - 0.384 * x664
    x670 = x169 * x663 + x175 * x660
    x671 = x665 * x75 + x669 * x76 - 0.0825 * x670
    x672 = -x670
    x673 = x164 * x663
    x674 = x169 * x660 + x673
    x675 = x300 * x662 + x301 * x563
    x676 = x674 * x75 + x675 * x82
    x677 = -0.0825 * x675
    x678 = x169 * x666 + x175 * x668 + x246 * x660 - 0.384 * x673 + x677
    x679 = -0.0825 * x674 * x76 + x677 * x75 - x678
    x680 = x665 * x82 + x669 * x75
    x681 = x28 * x52
    x682 = x30 * x68
    #
    H_num[0] = 0
    H_num[1] = 0
    H_num[2] = 0
    H_num[3] = 0
    H_num[4] = 0
    H_num[5] = ddq[0]
    H_num[6] = 0
    H_num[7] = 0
    H_num[8] = 0
    H_num[9] = 0
    H_num[10] = x0 * x11 + x6
    H_num[11] = x13 * (x12 + x4) + x14 * x15
    H_num[12] = x13 * (ddq[1] + x19) + x14 * (x16 + x18)
    H_num[13] = x13 * x21 + x20 * x5
    H_num[14] = x13 * (x22 + x23) + x14 * (ddq[1] + x24)
    H_num[15] = x13 * x8 - x6
    H_num[16] = 0
    H_num[17] = 0
    H_num[18] = x13 * x26 + x14 * x27
    H_num[19] = 0
    H_num[20] = x13 * (x29 * x33 + x30 * x35) - x14 * x37
    H_num[21] = x13 * (x29 * x47 + x30 * x45) - x14 * x41
    H_num[22] = x13 * (x29 * x50 + x53) - x14 * x55
    H_num[23] = x13 * (x29 * x43 + x30 * x54) + x14 * x37
    H_num[24] = x13 * (x29 * x57 + x30 * x59) - x14 * x56
    H_num[25] = x13 * (x29 * x44 + x30 * x46) + x14 * x60
    H_num[26] = x13 * (-0.316 * x28 * x61 + x29 * x63 - 0.316 * x53) + x14 * x67
    H_num[27] = x13 * (-0.316 * x28 * x70 + x30 * x62 - 0.316 * x30 * x71) - x14 * x69
    H_num[28] = x13 * (-0.316 * x28 * x56 + x29 * x68 + x30 * x67 - 0.316 * x30 * x73)
    H_num[29] = x13 * (-0.316 * x30 * x66 + x68 * x74)
    H_num[30] = x13 * (x29 * x86 + x30 * x83) - x14 * x87
    H_num[31] = -x100 * x14 + x13 * (x29 * x92 + x30 * x99)
    H_num[32] = -x111 * x14 + x13 * (x103 * x29 + x110 * x30)
    H_num[33] = -x112 * x14 + x13 * (x114 * x30 + x29 * x85)
    H_num[34] = -x121 * x14 + x13 * (x116 * x29 + x120 * x30)
    H_num[35] = -x122 * x14 + x13 * (x123 * x30 + x124 * x29)
    H_num[36] = -x129 * x14 + x13 * (x131 * x74 - 0.316 * x132 * x30 + x133 * x30 + x140 * x29)
    H_num[37] = x13 * (x144 * x74 + x147 * x29 + x148 * x30 - 0.316 * x149 * x30) - x14 * x141
    H_num[38] = x13 * (x150 * x30 + x153 * x74 + x154 * x29 - 0.316 * x156 * x30) - x14 * x158
    H_num[39] = x13 * (-0.316 * x126 * x30 + x159 * x29 + x160 * x74) - x14 * x161
    H_num[40] = x13 * (x174 * x30 + x177 * x29) - x14 * x178
    H_num[41] = x13 * (x191 * x30 + x193 * x29) - x14 * x190
    H_num[42] = x13 * (x204 * x30 + x207 * x29) - x14 * x203
    H_num[43] = x13 * (x210 * x29 + x212 * x30) - x14 * x213
    H_num[44] = x13 * (x220 * x30 + x222 * x29) - x14 * x219
    H_num[45] = x13 * (x226 * x29 + x228 * x30) - x14 * x224
    H_num[46] = x13 * (x233 * x74 - 0.316 * x235 * x30 + x244 * x30 + x248 * x29) - x14 * x249
    H_num[47] = x13 * (x264 * x29 - 0.316 * x265 * x30 + x266 * x30 + x267 * x74) - x14 * x259
    H_num[48] = x13 * (x281 * x74 + x282 * x30 + x285 * x29 - 0.316 * x286 * x30) - x14 * x277
    H_num[49] = x13 * (x29 * x299 + x292 * x30 + x295 * x74 - 0.316 * x296 * x30) - x14 * x291
    H_num[50] = x13 * (x29 * x316 + x30 * x317) - x14 * x314
    H_num[51] = x13 * (x29 * x334 + x30 * x335) - x14 * x332
    H_num[52] = x13 * (x29 * x350 + x30 * x348) - x14 * x351
    H_num[53] = x13 * (x29 * x358 + x30 * x356) - x14 * x359
    H_num[54] = x13 * (x29 * x371 + x30 * x369) - x14 * x368
    H_num[55] = x13 * (x29 * x379 + x30 * x377) - x14 * x376
    H_num[56] = x13 * (x29 * x402 - 0.316 * x30 * x395 + x30 * x396 + x403 * x74) - x14 * x394
    H_num[57] = x13 * (x29 * x427 - 0.316 * x30 * x419 + x30 * x424 + x423 * x74) - x14 * x418
    H_num[58] = x13 * (x29 * x446 + x30 * x438 - 0.316 * x30 * x440 + x447 * x74) - x14 * x448
    H_num[59] = x13 * (x29 * x454 + x30 * x459 - 0.316 * x30 * x461 + x462 * x74) - x14 * x463
    H_num[60] = x13 * (x29 * x483 + x30 * x484) - x14 * x481
    H_num[61] = x13 * (x29 * x503 + x30 * x504) - x14 * x501
    H_num[62] = x13 * (x29 * x522 + x30 * x520) - x14 * x523
    H_num[63] = x13 * (x29 * x533 + x30 * x534) - x14 * x531
    H_num[64] = x13 * (x29 * x548 + x30 * x546) - x14 * x549
    H_num[65] = x13 * (x29 * x559 + x30 * x560) - x14 * x557
    H_num[66] = x13 * (x29 * x591 - 0.316 * x30 * x592 + x30 * x593 + x594 * x74) - x14 * x585
    H_num[67] = x13 * (x29 * x625 + x30 * x617 - 0.316 * x30 * x622 + x621 * x74) - x14 * x616
    H_num[68] = x13 * (x29 * x645 - 0.316 * x30 * x648 + x30 * x654 + x646 * x74) - x14 * x655
    H_num[69] = x13 * (x29 * x679 - 0.316 * x30 * x672 + x30 * x680 + x676 * x74) - x14 * x671
    H_num[70] = 0
    H_num[71] = 0
    H_num[72] = 0
    H_num[73] = 0
    H_num[74] = 0
    H_num[75] = 0
    H_num[76] = 0
    H_num[77] = 0
    H_num[78] = 0
    H_num[79] = 0
    H_num[80] = x24
    H_num[81] = x17 - x22
    H_num[82] = x10 + x21
    H_num[83] = x19
    H_num[84] = x12 + x3
    H_num[85] = ddq[1]
    H_num[86] = x25
    H_num[87] = -x27
    H_num[88] = 0
    H_num[89] = 0
    H_num[90] = x28 * x35 + x30 * x33
    H_num[91] = x28 * x45 + x30 * x47
    H_num[92] = x30 * x50 + x681
    H_num[93] = x28 * x54 + x30 * x43
    H_num[94] = x28 * x59 + x30 * x57
    H_num[95] = x28 * x46 + x30 * x44
    H_num[96] = 0.316 * x30 * x61 + x30 * x63 - 0.316 * x681
    H_num[97] = x28 * x62 + 0.316 * x30 * x70 + x71 * x74
    H_num[98] = x28 * x67 + 0.316 * x30 * x56 + x682 + x73 * x74
    H_num[99] = x66 * x74 + 0.316 * x682
    H_num[100] = x28 * x83 + x30 * x86
    H_num[101] = x28 * x99 + x30 * x92
    H_num[102] = x103 * x30 + x110 * x28
    H_num[103] = x114 * x28 + x30 * x85
    H_num[104] = x116 * x30 + x120 * x28
    H_num[105] = x123 * x28 + x124 * x30
    H_num[106] = 0.316 * x131 * x30 + x132 * x74 + x133 * x28 + x140 * x30
    H_num[107] = 0.316 * x144 * x30 + x147 * x30 + x148 * x28 + x149 * x74
    H_num[108] = x150 * x28 + 0.316 * x153 * x30 + x154 * x30 + x156 * x74
    H_num[109] = x126 * x74 + x159 * x30 + 0.316 * x160 * x30
    H_num[110] = x174 * x28 + x177 * x30
    H_num[111] = x191 * x28 + x193 * x30
    H_num[112] = x204 * x28 + x207 * x30
    H_num[113] = x210 * x30 + x212 * x28
    H_num[114] = x220 * x28 + x222 * x30
    H_num[115] = x226 * x30 + x228 * x28
    H_num[116] = 0.316 * x233 * x30 + x235 * x74 + x244 * x28 + x248 * x30
    H_num[117] = x264 * x30 + x265 * x74 + x266 * x28 + 0.316 * x267 * x30
    H_num[118] = x28 * x282 + 0.316 * x281 * x30 + x285 * x30 + x286 * x74
    H_num[119] = x28 * x292 + 0.316 * x295 * x30 + x296 * x74 + x299 * x30
    H_num[120] = x28 * x317 + x30 * x316
    H_num[121] = x28 * x335 + x30 * x334
    H_num[122] = x28 * x348 + x30 * x350
    H_num[123] = x28 * x356 + x30 * x358
    H_num[124] = x28 * x369 + x30 * x371
    H_num[125] = x28 * x377 + x30 * x379
    H_num[126] = x28 * x396 + x30 * x402 + 0.316 * x30 * x403 + x395 * x74
    H_num[127] = x28 * x424 + 0.316 * x30 * x423 + x30 * x427 + x419 * x74
    H_num[128] = x28 * x438 + x30 * x446 + 0.316 * x30 * x447 + x440 * x74
    H_num[129] = x28 * x459 + x30 * x454 + 0.316 * x30 * x462 + x461 * x74
    H_num[130] = x28 * x484 + x30 * x483
    H_num[131] = x28 * x504 + x30 * x503
    H_num[132] = x28 * x520 + x30 * x522
    H_num[133] = x28 * x534 + x30 * x533
    H_num[134] = x28 * x546 + x30 * x548
    H_num[135] = x28 * x560 + x30 * x559
    H_num[136] = x28 * x593 + x30 * x591 + 0.316 * x30 * x594 + x592 * x74
    H_num[137] = x28 * x617 + 0.316 * x30 * x621 + x30 * x625 + x622 * x74
    H_num[138] = x28 * x654 + x30 * x645 + 0.316 * x30 * x646 + x648 * x74
    H_num[139] = x28 * x680 + 0.316 * x30 * x676 + x30 * x679 + x672 * x74
    H_num[140] = 0
    H_num[141] = 0
    H_num[142] = 0
    H_num[143] = 0
    H_num[144] = 0
    H_num[145] = 0
    H_num[146] = 0
    H_num[147] = 0
    H_num[148] = 0
    H_num[149] = 0
    H_num[150] = 0
    H_num[151] = 0
    H_num[152] = 0
    H_num[153] = 0
    H_num[154] = 0
    H_num[155] = 0
    H_num[156] = 0
    H_num[157] = 0
    H_num[158] = 0
    H_num[159] = 0
    H_num[160] = x37
    H_num[161] = x41
    H_num[162] = x55
    H_num[163] = x36
    H_num[164] = x56
    H_num[165] = x51
    H_num[166] = x66
    H_num[167] = x69
    H_num[168] = 0
    H_num[169] = 0
    H_num[170] = x87
    H_num[171] = x100
    H_num[172] = x111
    H_num[173] = x112
    H_num[174] = x121
    H_num[175] = x122
    H_num[176] = x129
    H_num[177] = x141
    H_num[178] = x158
    H_num[179] = x161
    H_num[180] = x178
    H_num[181] = x190
    H_num[182] = x203
    H_num[183] = x213
    H_num[184] = x219
    H_num[185] = x224
    H_num[186] = x249
    H_num[187] = x259
    H_num[188] = x277
    H_num[189] = x291
    H_num[190] = x314
    H_num[191] = x332
    H_num[192] = x351
    H_num[193] = x359
    H_num[194] = x368
    H_num[195] = x376
    H_num[196] = x394
    H_num[197] = x418
    H_num[198] = x448
    H_num[199] = x463
    H_num[200] = x481
    H_num[201] = x501
    H_num[202] = x523
    H_num[203] = x531
    H_num[204] = x549
    H_num[205] = x557
    H_num[206] = x585
    H_num[207] = x616
    H_num[208] = x655
    H_num[209] = x671
    H_num[210] = 0
    H_num[211] = 0
    H_num[212] = 0
    H_num[213] = 0
    H_num[214] = 0
    H_num[215] = 0
    H_num[216] = 0
    H_num[217] = 0
    H_num[218] = 0
    H_num[219] = 0
    H_num[220] = 0
    H_num[221] = 0
    H_num[222] = 0
    H_num[223] = 0
    H_num[224] = 0
    H_num[225] = 0
    H_num[226] = 0
    H_num[227] = 0
    H_num[228] = 0
    H_num[229] = 0
    H_num[230] = 0
    H_num[231] = 0
    H_num[232] = 0
    H_num[233] = 0
    H_num[234] = 0
    H_num[235] = 0
    H_num[236] = 0
    H_num[237] = 0
    H_num[238] = 0
    H_num[239] = 0
    H_num[240] = x85
    H_num[241] = x91
    H_num[242] = x102
    H_num[243] = x84
    H_num[244] = x115
    H_num[245] = x108
    H_num[246] = x137
    H_num[247] = x146
    H_num[248] = 0
    H_num[249] = 0
    H_num[250] = x176
    H_num[251] = x192
    H_num[252] = x206
    H_num[253] = x209
    H_num[254] = x221
    H_num[255] = x225
    H_num[256] = x247
    H_num[257] = x262
    H_num[258] = x284
    H_num[259] = x298
    H_num[260] = x315
    H_num[261] = x333
    H_num[262] = x349
    H_num[263] = x357
    H_num[264] = x370
    H_num[265] = x378
    H_num[266] = x400
    H_num[267] = x426
    H_num[268] = x444
    H_num[269] = x452
    H_num[270] = x482
    H_num[271] = x502
    H_num[272] = x521
    H_num[273] = x532
    H_num[274] = x547
    H_num[275] = x558
    H_num[276] = x590
    H_num[277] = x624
    H_num[278] = x643
    H_num[279] = x678
    H_num[280] = 0
    H_num[281] = 0
    H_num[282] = 0
    H_num[283] = 0
    H_num[284] = 0
    H_num[285] = 0
    H_num[286] = 0
    H_num[287] = 0
    H_num[288] = 0
    H_num[289] = 0
    H_num[290] = 0
    H_num[291] = 0
    H_num[292] = 0
    H_num[293] = 0
    H_num[294] = 0
    H_num[295] = 0
    H_num[296] = 0
    H_num[297] = 0
    H_num[298] = 0
    H_num[299] = 0
    H_num[300] = 0
    H_num[301] = 0
    H_num[302] = 0
    H_num[303] = 0
    H_num[304] = 0
    H_num[305] = 0
    H_num[306] = 0
    H_num[307] = 0
    H_num[308] = 0
    H_num[309] = 0
    H_num[310] = 0
    H_num[311] = 0
    H_num[312] = 0
    H_num[313] = 0
    H_num[314] = 0
    H_num[315] = 0
    H_num[316] = 0
    H_num[317] = 0
    H_num[318] = 0
    H_num[319] = 0
    H_num[320] = x168
    H_num[321] = x189
    H_num[322] = x202
    H_num[323] = x167
    H_num[324] = x214
    H_num[325] = x197
    H_num[326] = x238
    H_num[327] = x257
    H_num[328] = 0
    H_num[329] = 0
    H_num[330] = x307
    H_num[331] = x331
    H_num[332] = x347
    H_num[333] = x352
    H_num[334] = x363
    H_num[335] = x372
    H_num[336] = x392
    H_num[337] = x404
    H_num[338] = x436
    H_num[339] = 0
    H_num[340] = x476
    H_num[341] = x496
    H_num[342] = x515
    H_num[343] = x526
    H_num[344] = x541
    H_num[345] = x556
    H_num[346] = x582
    H_num[347] = x609
    H_num[348] = x652
    H_num[349] = x659
    H_num[350] = 0
    H_num[351] = 0
    H_num[352] = 0
    H_num[353] = 0
    H_num[354] = 0
    H_num[355] = 0
    H_num[356] = 0
    H_num[357] = 0
    H_num[358] = 0
    H_num[359] = 0
    H_num[360] = 0
    H_num[361] = 0
    H_num[362] = 0
    H_num[363] = 0
    H_num[364] = 0
    H_num[365] = 0
    H_num[366] = 0
    H_num[367] = 0
    H_num[368] = 0
    H_num[369] = 0
    H_num[370] = 0
    H_num[371] = 0
    H_num[372] = 0
    H_num[373] = 0
    H_num[374] = 0
    H_num[375] = 0
    H_num[376] = 0
    H_num[377] = 0
    H_num[378] = 0
    H_num[379] = 0
    H_num[380] = 0
    H_num[381] = 0
    H_num[382] = 0
    H_num[383] = 0
    H_num[384] = 0
    H_num[385] = 0
    H_num[386] = 0
    H_num[387] = 0
    H_num[388] = 0
    H_num[389] = 0
    H_num[390] = 0
    H_num[391] = 0
    H_num[392] = 0
    H_num[393] = 0
    H_num[394] = 0
    H_num[395] = 0
    H_num[396] = 0
    H_num[397] = 0
    H_num[398] = 0
    H_num[399] = 0
    H_num[400] = x311
    H_num[401] = x328
    H_num[402] = x337
    H_num[403] = x310
    H_num[404] = x364
    H_num[405] = x339
    H_num[406] = x389
    H_num[407] = x415
    H_num[408] = 0
    H_num[409] = 0
    H_num[410] = x478
    H_num[411] = x497
    H_num[412] = x517
    H_num[413] = x527
    H_num[414] = x542
    H_num[415] = x550
    H_num[416] = x564
    H_num[417] = x613
    H_num[418] = x633
    H_num[419] = x667
    H_num[420] = 0
    H_num[421] = 0
    H_num[422] = 0
    H_num[423] = 0
    H_num[424] = 0
    H_num[425] = 0
    H_num[426] = 0
    H_num[427] = 0
    H_num[428] = 0
    H_num[429] = 0
    H_num[430] = 0
    H_num[431] = 0
    H_num[432] = 0
    H_num[433] = 0
    H_num[434] = 0
    H_num[435] = 0
    H_num[436] = 0
    H_num[437] = 0
    H_num[438] = 0
    H_num[439] = 0
    H_num[440] = 0
    H_num[441] = 0
    H_num[442] = 0
    H_num[443] = 0
    H_num[444] = 0
    H_num[445] = 0
    H_num[446] = 0
    H_num[447] = 0
    H_num[448] = 0
    H_num[449] = 0
    H_num[450] = 0
    H_num[451] = 0
    H_num[452] = 0
    H_num[453] = 0
    H_num[454] = 0
    H_num[455] = 0
    H_num[456] = 0
    H_num[457] = 0
    H_num[458] = 0
    H_num[459] = 0
    H_num[460] = 0
    H_num[461] = 0
    H_num[462] = 0
    H_num[463] = 0
    H_num[464] = 0
    H_num[465] = 0
    H_num[466] = 0
    H_num[467] = 0
    H_num[468] = 0
    H_num[469] = 0
    H_num[470] = 0
    H_num[471] = 0
    H_num[472] = 0
    H_num[473] = 0
    H_num[474] = 0
    H_num[475] = 0
    H_num[476] = 0
    H_num[477] = 0
    H_num[478] = 0
    H_num[479] = 0
    H_num[480] = x474
    H_num[481] = x488
    H_num[482] = x513
    H_num[483] = x473
    H_num[484] = x539
    H_num[485] = x505
    H_num[486] = x576
    H_num[487] = x607
    H_num[488] = 0
    H_num[489] = 0
    #
    return H_num
