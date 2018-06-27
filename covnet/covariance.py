#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import dates as md
from scipy.linalg import eigvalsh, eigh

from . import logtable
from .maths import xcov


def calculate(times, spectra, average=10, overlap=.5):
    """Calculate covariance matrix from the given spectra.

    Arguments:
    ----------
        average (int): number of averaging spectral windows.

        overlap (float): overlaping ratio between consecutive spectral
            windows. Default to 0.5.

        standardize (bool): use of standart deviation normalization.
            Default to False (no normalization applied).

    """

    # Parametrization
    overlap = int(average * overlap)

    # Reshape spectra in order to (n_stations, n_times, n_frequencies)
    spectra = spectra.transpose([0, 2, 1])
    n_traces, n_windows, n_frequencies = spectra.shape

    # Times
    t_end = times[-1]
    times = times[:-1]
    times = times[:-average:overlap]
    n_average = len(times)
    times = np.hstack((times, t_end))

    # Initialization
    cov_shape = (n_average, n_traces, n_traces, n_frequencies)
    covariance = np.zeros(cov_shape, dtype=complex)

    # Compute
    waitbar = logtable.waitbar('Covariance', n_average)
    for t in range(n_average):
        covariance[t] = xcov(t, spectra, overlap, average)
        waitbar.progress(t)

    return times, covariance.view(CovarianceMatrix).transpose([0, -1, 1, 2])


class CovarianceMatrix(np.ndarray):
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

    def eigenvalues(self, norm=max):
        """Eigenvalue decomposition.

        The eigenvalue decomposition is performed onto the two last dimensions
        of the :class:`~arrayprocessing.covariance.CovarianceMatrix` object,
        i.e. if the shape of the matrix is ``(a, b, n, n)``, the decomposition
        is performed loop-wise onto the ``a * b`` ``(n, n)`` sub-matrices.

        The function used for eigenvalue decomposition is
        :func:`scipy.linalg.eigvalsh`. It assumes that the input matrix is 2D
        and hermitian. The decomposition is performed onto the lower triangular
        part.

        Keyword arguments
        -----------------

        norm : function
            The function used to normalize the eigenvalues. Can be :func:`max`,
            (default), :func:`sum` or any other functions.

        Returns
        -------

        :class:`np.ndarray`
            The eigenvalues array of shape ``(a, b, n)``.

        """

        # Initialization
        matrices = self._flat()
        eigenvalues = np.zeros((matrices.shape[0], matrices.shape[-1]))

        # Calculation over submatrices
        wb = logtable.waitbar('Eigenvalues', matrices.shape[0])
        for i, matrix in enumerate(matrices):
            wb.progress(i)
            eigenvalues[i] = np.abs(eigvalsh(matrix)[::-1])
            eigenvalues[i] /= norm(eigenvalues[i])

        return eigenvalues.reshape(self.shape[:-1])

    def eigenvectors(self, rank=0):
        """Extract eigenvectors of given rank.

        The eigenvector decomposition is performed onto the two last dimensions
        of the :class:`~arrayprocessing.covariance.CovarianceMatrix` object,
        i.e. if the shape of the matrix is ``(a, b, n, n)``, the decomposition
        is performed loop-wise onto the ``a * b`` ``(n, n)`` sub-matrices.

        The function used for eigenvalue decomposition is
        :func:`scipy.linalg.eigh`. It assumes that the input matrix is 2D
        and hermitian. The decomposition is performed onto the lower triangular
        part.

        Keyword arguments
        -----------------

        rank : int
            Eigenvector rank. Default is 0 (first eigenvector).

        Returns
        -------

        :class:`np.ndarray`
            The eigenvector array of shape ``(a, b, n)``.

        """

        # Initialization
        matrices = self._flat()
        eigenvectors = np.zeros((matrices.shape[0], matrices.shape[-1]),
                                dtype=complex)

        # Calculation over submatrices
        wb = logtable.waitbar('Eigenvectors', matrices.shape[0])
        for i, m in enumerate(matrices):
            wb.progress(i)
            # d, e = eigh(m)
            # eigenvectors[i] = np.sqrt(d[-1 - rank]) * m[:, -1 - rank]
            eigenvectors[i] = eigh(m)[1][:, -1 - rank]

        return eigenvectors.reshape(self.shape[:-1])

    def spectral_width(self):
        """Eigenvalue spectrum width of distribution.

        The measured is performed onto all the covariance matrices from
        the eigenvalues obtained with the method
        :meth:`arrayprocessing.covariance.CovarianceMatrix.eigenvalues`.
        For a given matrix :math:`n \\times n` matrix :math:`M` with
        eigenvalues :math:`\\lambda_i` where :math:`i=1\\ldots n`, the
        spectral width :math:`\\sigma` is obtained with the formula

        .. math::

            \\sigma = \\frac{\\sum_{i=0}^n i \\lambda_i}{\\sum_{i=0}^n
            \\lambda_i}

        Returns
        -------

        :class:`np.ndarray`
            The spectral width of shape ``(a, b)``.

        """

        eigenvalues = self.eigenvalues(norm=sum)
        indices = np.arange(self.shape[-1])
        return np.multiply(eigenvalues, indices).sum(axis=-1)

    def entropy(self, epsilon=1e-10):
        """Entropy of information.

        The measured is performed onto all the covariance matrices from
        the eigenvalues obtained with the method
        :meth:`arrayprocessing.covariance.CovarianceMatrix.eigenvalues`.
        For a given matrix :math:`n \\times n` matrix :math:`M` with
        eigenvalues :math:`\\lambda_i` where :math:`i=1\\ldots n`, the
        information entropy (of Shannon) :math:`h` is obtained with the formula

        .. math::

            h = - \\sum_{i=0}^n i \\lambda_i \\log\\lambda_i

        Returns
        -------

        :class:`np.ndarray`
            The entropy of shape ``(a, b)``.

        """

        eigenvalues = self.eigenvalues(norm=sum)
        log_eigenvalues = np.log(eigenvalues + epsilon)
        return - np.sum(eigenvalues * log_eigenvalues, axis=-1)

    def _flat(self):
        """Covariance matrices with flatten first dimensions.

        Returns
        -------

        :class:`np.ndarray`
            The covariance matrices in a shape ``(a * b, n, n)``.
        """
        return self.reshape(-1, *self.shape[-2:])

    def triu(self, **kwargs):

        trii, trij = np.triu_indices(self.shape[-1], **kwargs)
        return self[..., trii, trij]


def show_coherence(times, frequencies, coherence, ax=None, cax=None,
                   flim=None, step=.5, figsize=(6, 5), **kwargs):
    """Pcolormesh the spectrogram of a single seismic trace.

    The spectrogram (modulus of the short-time Fourier transform) is
    extracted from the complex spectrogram previously calculated from
    the :meth:`arrayprocessing.data.stft` method.

    The spectrogram is represented in log-scale amplitude normalized by
    the maximal amplitude (dB re max).

    The date axis is automatically defined with Matplotlib's dates.

    Parameters
    ----------

    times : :class:`np.ndarray`
        The starting times of the windows

    frequencies : :class:`np.ndarray`
        The frequency vector.

    spectra : :class:`np.ndarray`
        The spectrogram matrix of shape ``(n_station, n_frequencies, n_times)``

    Keyword arguments
    -----------------

    code : int or str
        Index or code of the seismic station.

    step : float
        The step between windows in fraction of segment duration.
        By default, assumes a step of .5 meaning 50% of overlap.

    ax : :class:`matplotlib.axes.Axes`
        Previously instanciated axes. Default to None, and the axes are
        created.

    cax : :class:`matplotlib.axes.Axes`
        Axes for the colorbar. Default to None, and the axes are created.
        These axes should be given if ``ax`` is not None.

    kwargs : dict
        Other keyword arguments passed to
        :func:`matplotlib.pyplot.pcolormesh`

    Return
    ------

        If the path_figure kwargs is set to None (default), the following
        objects are returned:

        fig (matplotlib.pyplot.Figure) the figure instance.
        ax (matplotlib.pyplot.Axes) axes of the spectrogram.
        cax (matplotlib.pyplot.Axes) axes of the colorbar.

    """

    # Axes
    if ax is None:
        gs = dict(width_ratios=[50, 1])
        fig, (ax, cax) = plt.subplots(1, 2, figsize=figsize, gridspec_kw=gs)

    # Safe
    coherence = np.squeeze(coherence)

    # Image
    kwargs.setdefault('rasterized', True)
    img = ax.pcolormesh(times, frequencies, coherence.T, **kwargs)

    # Colorbar
    plt.colorbar(img, cax=cax)
    cax.set_ylabel('Coherence')

    # Date ticks
    ax.set_xlim(times[[0, -1]])
    xticks = md.AutoDateLocator()
    ax.xaxis.set_major_locator(xticks)
    ax.xaxis.set_major_formatter(md.AutoDateFormatter(xticks))

    # Frequencies
    ax.set_yscale('log')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim(frequencies[[1, -1]])

    return ax, cax
