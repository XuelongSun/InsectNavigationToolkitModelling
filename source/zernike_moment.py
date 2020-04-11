# @File: zernike_moment.py
# @Info: the implementation of zernike moment
# @Author: Xuelong Sun, UoL, UK
# @Time: 2020-02-17

import numpy as np
from math import factorial

# Pre-calculate the factorial to improve the speed
Nmax = 60
Mmax = 20
Pmax = int(Nmax/2)
Qmax = int((Nmax + Mmax)/2)

FAC_S = np.zeros([Pmax+1])
for s in range(Pmax+1):
    FAC_S[s] = factorial(s)

FAC_N_S = np.zeros([Nmax,Pmax+1])
for n in range(Nmax):
    for s in range(int(n/2)+1):
        FAC_N_S[n,s] = factorial(n - s)

FAC_Q_S = np.zeros([Qmax,Pmax+1])
for q in range(Qmax):
    for s in range(np.min([q+1,Pmax+1])):
        FAC_Q_S[q,s] = factorial(q - s)

FAC_P_S = np.zeros([Pmax,Pmax+1])
for p in range(Pmax):
    for s in range(p+1):
        FAC_P_S[p,s] = factorial(p - s)

# pre-calculate the polar coordinates for fixed image size
pc_N = 208
x = range(pc_N)
y = x
pc_X, pc_Y = np.meshgrid(x, y)
pc_R = np.sqrt((2 * pc_X - pc_N + 1) ** 2 + (2 * pc_Y - pc_N + 1) ** 2) / pc_N
pc_Theta = np.arctan2(pc_N - 1 - 2 * pc_Y, 2 * pc_X - pc_N + 1)
pc_R = np.where(pc_R <= 1, 1, 0) * pc_R


def radial_poly(r, n, m):
    rad = np.zeros(r.shape, r.dtype)
    P = int((n - abs(m)) / 2)
    Q = int((n + abs(m)) / 2)
    for s in range(P + 1):
        c = (-1) ** s * FAC_N_S[n,s]
        c /= FAC_S[s] * FAC_Q_S[Q,s] * FAC_P_S[P,s]
        rad += c * r ** (n - 2 * s)
    return rad


def zernike_moment(src, n, m):
    """
        get the ZM coefficient with order n and repeat m
        :param src: source image
        :param m: the order
        :param n: the repeat
        :return: the amplitude and the phase
        """
    if src.dtype != np.float32:
        src = np.where(src > 0, 0, 1).astype(np.float32)
    if len(src.shape) == 3:
        print('the input image src should be in gray')
        return

    # get the radial polynomial
    Rad = radial_poly(pc_R, n, m)

    Product = src * Rad * np.exp(-1j * m * pc_Theta)
    # calculate the moments
    pc_Z = Product.sum()

    # count the number of pixels inside the unit circle
    cnt = np.count_nonzero(pc_R) + 1
    # normalize the amplitude of moments
    pc_Z = (n + 1) * pc_Z / cnt
    # calculate the amplitude of the moment
    pc_A = abs(pc_Z)
    # calculate the phase of the moment (in degrees)
    pc_Phi = np.angle(pc_Z) * 180 / np.pi

    return pc_Z, pc_A, pc_Phi

