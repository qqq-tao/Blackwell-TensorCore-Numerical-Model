#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
# import matplotlib.pyplot as plt
# import os

import fpemu

# from cuda_runner import CudaEnv
# from CuAsm import CuAsmLogger

if __name__ == '__main__':

    # CuAsmLogger.initLogger(stdout_level=0)

    grd = 1
    blk = 256
    N = grd * blk

    # A = np.r_[0:N].astype(dtype=np.uint8)
    # B = np.zeros(N, dtype=np.uint16)
    # B2 = np.zeros(N, dtype=np.uint16)

    # fpemu.cvt_fp8_to_half(A, B2, True)
    A = np.r_[0:N].astype(dtype=np.uint32)
    B = np.zeros(N, dtype=np.uint16)
    B2 = np.zeros(N, dtype=np.uint16)

    fpemu.cvt_float_to_bf16(A, B2, True)

    # args = [('ptr.i', A),
    #         ('ptr.o', B),
    #         ('u64',  N),
    #         ('s32', 1)
    #         ]

    # with open('t.cu', 'r+') as fin:
    #     ks = fin.read()

    # print(ks)

    # with CudaEnv(0) as env:
    # kernel = env.buildCudaC(ks, 'k' ) # , headers=[b'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5\\include'], includesNames=[b'cuda_fp8.h']
    # kernel = env.build_ptx(ks, 'k')
        # env.run(kernel, args, (grd, 1, 1), (blk,1,1))

    Bd = B - B2
    print(Bd)
