# Image denoising using MRF model for 3D data
from PIL import Image
import numpy as np
from pylab import *


def MRF_denoise(noisy):
    # Start MRF
    (M, N, P) = noisy.shape
    y_old = noisy
    y = np.zeros((M, N, P))

    while SNR(y_old, y) > 0.01:
        print(SNR(y_old, y))
        for i in range(M):
            for j in range(N):
                for k in range(P):
                    index = neighbor(i, j, k, M, N, P)

                    a = cost(1, noisy[i, j, k], y_old, index)
                    b = cost(0, noisy[i, j, k], y_old, index)

                    if a > b:
                        y[i, j, k] = 1
                    else:
                        y[i, j, k] = 0
        y_old = y.copy()
    print(SNR(y_old, y))
    return y


def SNR(A, B):
    if A.shape == B.shape:
        return np.sum(np.abs(A - B)) / A.size
    else:
        raise Exception("Two matrices must have the same size!")


def delta(a, b):
    return 1 if a == b else 0


def neighbor(i, j, k, M, N, P):
    # 找到正确的邻居
    neighbor = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                if di == 0 and dj == 0 and dk == 0:
                    continue  # 不包括自身
                ni, nj, nk = i + di, j + dj, k + dk
                if 0 <= ni < M and 0 <= nj < N and 0 <= nk < P:
                    neighbor.append((ni, nj, nk))
    return neighbor


def cost(y, x, y_old, index):
    alpha = 1
    beta = 10
    return alpha * delta(y, x) + beta * sum(delta(y, y_old[i]) for i in index)
