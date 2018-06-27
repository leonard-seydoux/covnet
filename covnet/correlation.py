#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

from . import logtable


def calculate(times, covariance):
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
    correlation = np.fft.fftshift(np.fft.ifft(covariance, axis=-2)).real

    # Calculate lags
    lags = np.linspace(-2., 2., correlation.shape[-2])

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
