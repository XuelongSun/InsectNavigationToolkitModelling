# @File: image_processing.py
# @Info: Useful functions for image pre-processing
# @Author: Xuelong Sun, UoL, UK
# @Time: 2019-08-17

import numpy as np

import cv2
from PIL import Image
import io

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection

from zernike_moment import zernike_moment


def cart2sph(X, Y, Z):
    """Converts cartesian to spherical coordinates.
    Works on matrices so we can pass in e.g. X with rows of len 3 for polygons."""

    XY = X ** 2 + Y ** 2
    TH = np.arctan2(Y, X)  # theta: azimuth
    PHI = np.arctan2(Z, np.sqrt(XY))  # phi: elevation from XY plane up
    R = np.sqrt(XY + Z ** 2)  # r

    return TH, PHI, R


def pi2pi(theta):
    """Constrains value to lie between -pi and pi."""
    return np.mod(theta + np.pi, 2 * np.pi) - np.pi


def img_wrapper(img, r_scale):
    """
    wrap the image to the disc
    :param img: input image
    :param r_scale: the radius of the wrapped image with respect to the width
    :return: the wrapped image
    """
    img_h = img.shape[0]
    img_w = img.shape[1]
    R = int(img_h * r_scale)
    deta_r = img_h / float(R)
    deta_theta = 2.0 * np.pi / img_w
    img_polar = np.zeros([2 * R, 2 * R], np.uint8)
    for i in range(2 * R):
        for j in range(2 * R):
            x = i - R
            y = R - j
            r = np.sqrt(x ** 2 + y ** 2)
            if r < R:
                theta = np.arctan2(y, x)
                #         print r, theta
                m = int(np.rint(r * deta_r))
                n = int(np.rint(theta / deta_theta))
                if n >= img_w:
                    n = img_w - 1
                if m >= img_h:
                    m = img_h - 1
                img_polar[i, j] = img[m, n]
    return img_polar


def get_img_view(world, x, y, z, th, res=1, hfov_d=360, v_max=np.pi / 2, v_min=-np.pi / 12,
                 wrap=False, blur=False, blur_kernel_size=3):
    """
    reconstruct the image view from the simulated world at position(x,y,z) with heading (th)
    :param world: the simulated world, a dictionary with keys ['X','Y','Z']
    :param x: the x-position of the eye
    :param y: the y-position of the eye
    :param z: the z-position of the eye,
    :param th: the heading
    :param res: resolution
    :param hfov_d: horizontal filed of view in deg
    :param v_max: the max vertical view
    :param v_min: the min vertical view
    :param wrap: whether to wrap the img or not
    :param blur: whether to blur the img or not
    :param blur_kernel_size: the size of the kernel to blur the image
    :return: the reconstructed image
    """

    X, Y, Z = world['X'], world['Y'], world['Z']
    dpi = 100
    hfov_deg = hfov_d
    hfov = np.deg2rad(hfov_deg)
    h_min = -hfov / 2
    h_max = hfov / 2

    vfov = v_max - v_min
    vfov_deg = np.rad2deg(vfov)

    resolution = res
    sky_colour = 'white'
    ground_colour = (0.1, 0.1, 0.1, 1)
    grass_colour = 'gray'
    grass_cmap = LinearSegmentedColormap.from_list('mycmap', [(0, (0, 0, 0, 1)), (1, grass_colour)])

    c = np.ones(Z.shape[0]) * 0.5

    image_ratio = vfov / hfov
    h_pixels = hfov_deg / resolution
    v_pixels = h_pixels * image_ratio

    im_width = h_pixels / dpi
    im_height = v_pixels / dpi

    fig = Figure(frameon=False, figsize=(im_width, im_height))
    ax = fig.add_axes([0., 0., 1., 1.])
    ax.set_xlim(h_min, h_max)
    ax.set_ylim(v_min, v_max)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor(sky_colour)

    canvas = FigureCanvasAgg(fig)
    ground_verts = [[(h_min, v_min), (h_max, v_min), (h_max, 0), (h_min, 0)]]

    g = PolyCollection(ground_verts, facecolor=ground_colour, edgecolor='none')
    ax.add_collection(g)

    TH, PHI, R = cart2sph(X - x, Y - y, np.abs(Z) - z)
    TH_rel = pi2pi(TH - th)

    # fix the grass
    ind = (np.max(TH_rel, axis=1) - np.min(TH_rel, axis=1)) > np.pi
    TH_ext = np.vstack((TH_rel, np.mod(TH_rel[ind, :] - 2 * np.pi, -2 * np.pi)))
    n_blades = np.sum(ind)
    padded_ind = np.lib.pad(ind, (0, n_blades), 'constant')
    TH_ext[padded_ind, :] = np.mod(TH_rel[ind, :] + 2 * np.pi, 2 * np.pi)

    PHI_ext = np.vstack((PHI, PHI[ind, :]))
    R_ext = np.vstack((R, R[ind, :]))

    grass_verts = np.dstack((TH_ext, PHI_ext))
    p = PolyCollection(grass_verts, array=c, cmap=grass_cmap, edgecolors='none')
    ax.add_collection(p)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', pad_inches=0, dpi=dpi)
    buf.seek(0)
    im = Image.open(buf)
    im_array = np.asarray(im)[:, :, 0:3]

    # grey scale and blurred image
    img_cv = cv2.cvtColor(np.asarray(im_array), cv2.COLOR_RGB2BGR)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    if wrap:
        img_cv = img_wrapper(img_cv, 1)
    if blur:
        img_cv = cv2.blur(img_cv, (blur_kernel_size, blur_kernel_size))
    return img_cv


def visual_sense(world, x, y, th, z=0.01, hfov=360, nmax=16, blur=False, kernel_size=3):
    """
    get the frequency encoding info (zernike moment) at position(x,y,z) with heading (th)
    :param world: the simulated world
    :param x: the x-position of the eye
    :param y: the y-position of the eye
    :param th: the heading
    :param z: the z-position of the eye
    :param hfov: horizontal filed of view in deg
    :param nmax: max order of the Zernike Moment coefficients
    :param blur: whether to blur the img or not
    :param kernel_size: the size of the kernel to blur the image
    :return: the amplitudes(A) and phase(P) of ZM
    """
    if nmax%2:
        coeff_num = int(((1+nmax)/2)*((3+nmax)/2))
    else:
        coeff_num = int((nmax/2.0+1)**2)
    A = np.zeros(coeff_num)
    P = np.zeros(coeff_num)
    img_wrap = get_img_view(world, x, y, z, th, hfov_d=hfov, wrap=True, blur=blur, blur_kernel_size=kernel_size)
    index = 0
    for n in range(nmax+1):
        for m in range(n+1):
            if (n-abs(m))%2 == 0:
                M, A[index], P[index] = zernike_moment(255-img_wrap, n, m)
                index+=1
    return A, P

