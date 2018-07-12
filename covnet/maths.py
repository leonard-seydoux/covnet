#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Calculation of the covariance matrix from raw seismograms.

import numpy as np
import copy

from numba import jit, complex128, int32
from math import factorial


def phase(data, **kwargs):
    return np.exp(1j * np.angle(data))


def xouter(complex_vector):
    """Fast numpy hermitian outer product."""
    return complex_vector * complex_vector.conj()[:, None]


def detrend_spectrum(x, smooth=11, order=1, epsilon=1e-10):

    n_frequencies, n_times = x.shape
    for t in range(n_times):
        x_smooth = savitzky_golay(np.abs(x[:, t]), smooth, order)
        x[:, t] /= (x_smooth + epsilon)
    return x


def savitzky_golay(y, window_size, order, deriv=0, rate=1):

    import numpy as np

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))

    except ValueError:
        raise ValueError("window_size and order have to be of type int")

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")

    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order + 1)
    half_window = (window_size - 1) // 2

    # Precompute coefficients
    b = np.mat([[k**i for i in order_range]
                for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)

    # Pad the signal at the extremes with values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))

    return np.convolve(m[::-1], y, mode='valid')


@jit(complex128(int32, complex128, int32, int32))
def xcov(wid, spectra_full, overlap, average):
    """
    Calculation of the array covariance matrix from the array data vectors
    stored in the spectra matrix (should be n_traces x n_windows x n_freq),
    over one set of averaged windows.
    """
    n_traces, n_windows, n_frequencies = spectra_full.shape
    beg = overlap * wid
    end = beg + average
    spectra = copy.deepcopy(spectra_full[:, beg:end, :])

    X = spectra[:, None, 0, :] * np.conj(spectra[:, 0, :])
    for swid in range(1, average):
        X += spectra[:, None, swid, :] * np.conj(spectra[:, swid, :])
    return X
