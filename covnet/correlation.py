#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from matplotlib import pyplot as plt


def calculate(times, covariance, fs=1):
    """Calculate covariance matrix from the given spectra.

    Arguments:
    ----------
        average (int): number of averaging spectral windows.

        overlap (float): overlaping ratio between consecutive spectral
            windows. Default to 0.5.

        standardize (bool): use of standart deviation normalization.
            Default to False (no normalization applied).

    """

    # Extract upper triangular
    covariance = covariance.triu(k=1)

    # Inverse Fourier transform
    correlation = np.fft.fftshift(
        np.fft.ifft(covariance, axis=-2), axes=-2).real

    # Calculate lags
    n_lags = correlation.shape[-2]
    n_lag_symm = (n_lags - 1) // 2
    lags = np.arange(-n_lag_symm, n_lag_symm + 1) / fs

    return lags, correlation.view(CorrelationMatrix)


class CorrelationMatrix(np.ndarray):
    """Covariance matrix.

    This class is a subclass of :class:`~numpy.ndarray`. This means that
    all the methods available with regular numpy arrays are also available
    here, plus additional arrayprocessing-oriented methods. Any numpy method
    or function applied to a
    :class:`~arrayprocessing.covariance.CovarianceMatrix` instance returns a
    :class:`~arrayprocessing.covariance.CovarianceMatrix` instance.

    """

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def hilbert(self):
        return np.abs(hilbert(self, axis=0)).view(CorrelationMatrix)

    def smooth(self, sigma=5):
        return gaussian_filter1d(self, sigma, axis=0).view(CorrelationMatrix)


def show_correlation(times, correlation, ax=None, cax=None,
                     flim=None, step=.5, figsize=(6, 5), **kwargs):

    # Axes
    if ax is None:
        gs = dict(width_ratios=[50, 1])
        fig, (ax, cax) = plt.subplots(1, 2, figsize=figsize, gridspec_kw=gs)

    # Safe
    correlation = np.squeeze(correlation)
    correlation /= correlation.max(axis=-2)[None, :]

    # Image
    pairs = np.arange(correlation.shape[-1] + 1)
    kwargs.setdefault('rasterized', True)
    img = ax.pcolormesh(times, pairs, correlation.T, **kwargs)

    # Colorbar
    plt.colorbar(img, cax=cax)
    cax.set_ylabel('correlation')

    # Date ticks
    ax.set_xlim(times[[0, -1]])
    ax.set_xlabel('Lags (sec)')

    # Frequencies
    ax.set_ylabel('Station pairs')

    return ax, cax
