#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Calculation of the covariance matrix from raw seismograms.

import numpy as np

from scipy.special import jv

from .logtable import waitbar
from .covariance import CovarianceMatrix
from .maths import xouter


def planewave(net, frequency, slowness, azimuth, cov=True):
    """Synthetic monochromatic plane wave.

    The plane wave :math:`\\psi_{i}` recorded at station :math:`i` is defined
    in a homogeneous medium of slowness :math:`s_0` at frequency :math:`f_0` as

    .. math::

        \\psi_{i} = \\exp(2\\imath \\pi f_0 s_0 (\\sin(\\theta) x_i +
        \\cos(theta)y_i))

    where :math:`theta` is the azimuth of the plane wave from the north
    direction.

    Parameters
    ----------

    net : :class:`arrayprocessing.antenna.Antenna`
        The seismic array metadata.

    frequency : float
        The frequency in hertz.

    slowness : float
        The space-independant slowness in s/km.

    azimuth : float
        The azimuth from the North in degrees.

    Keyword arguments
    -----------------

    cov : bool
        If True (default), then the covariance is returned (e.g. outer
        conjugate product of the wavefield with itself). Else, the wavefield
        is returned.

    Returns
    -------

    :class:`~arrayprocessing.covariance.CovarianceMatrix`
        The covariance matrix object.

    """

    # Coordinates
    x, y = net.x, net.y

    # Phase
    wavenumber = 2 * np.pi * frequency * slowness
    azimuth = np.radians(azimuth)
    scalar_product = np.sin(azimuth) * x + np.cos(azimuth) * y

    # Wavefield
    wavefield = np.exp(-1j * wavenumber * scalar_product)

    if cov is True:
        return xouter(wavefield).astype(complex).view(CovarianceMatrix)
    else:
        return wavefield


def surface_noise(net, frequency, slowness):
    """Synthetic monochromatic theoretical surface noise.

    Parameters
    ----------

    net : :class:`arrayprocessing.antenna.Antenna`
        The seismic array metadata.

    frequency : float
        The frequency in hertz.

    slowness : float
        The space-independant slowness in s/km.

    Keyword arguments
    -----------------

    cov : bool
        If True (default), then the covariance is returned (e.g. outer
        conjugate product of the wavefield with itself). Else, the wavefield
        is returned.

    Returns
    -------

    :class:`~arrayprocessing.covariance.CovarianceMatrix`
        The covariance matrix object.

    """

    distances = net.distances()
    wavenumber = 2 * np.pi * frequency * slowness
    covariance = jv(0, wavenumber * distances).astype(complex)
    return covariance.view(CovarianceMatrix)


def volume_noise(net, frequency, slowness):
    """Synthetic monochromatic theoretical volume noise.

    Parameters
    ----------

    net : :class:`arrayprocessing.antenna.Antenna`
        The seismic array metadata.

    frequency : float
        The frequency in hertz.

    slowness : float
        The space-independant slowness in s/km.

    Keyword arguments
    -----------------

    cov : bool
        If True (default), then the covariance is returned (e.g. outer
        conjugate product of the wavefield with itself). Else, the wavefield
        is returned.

    Returns
    -------

    :class:`~arrayprocessing.covariance.CovarianceMatrix`
        The covariance matrix object.

    """

    distances = net.distances()
    wavenumber = 2 * np.pi * frequency * slowness
    covariance = np.sinc(wavenumber * distances).astype(complex)
    return covariance.view(CovarianceMatrix)


def estimated_surface_noise(net, frequency, slowness, n_sources=200,
                            n_snapshots=100):
    """ Estimate surface noise with random plane waves"""

    azimuths = np.linspace(0, 2 * np.pi, n_sources, endpoint=False)
    x, y = net.x, net.y
    covariance = np.zeros((net.dim, net.dim), dtype=complex)

    wb = waitbar('Surface noise estimate')
    for snapshot in range(n_snapshots):

        wb.progress((snapshot + 1) / n_snapshots)
        wavenumber = 2 * np.pi * frequency * slowness
        snapshots = np.zeros(net.dim, dtype=complex)
        phases = 2 * np.pi * np.random.rand(n_sources)

        for azimuth_id, (azimuth, phase) in enumerate(zip(azimuths, phases)):
            scalar_product = np.sin(azimuth) * x + np.cos(azimuth) * y
            snapshots += np.exp(-1j * wavenumber * scalar_product - 1j * phase)

        snapshots /= n_sources
        covariance += snapshots * snapshots.conj()[:, None]
        covariance /= n_snapshots
    return covariance.view(CovarianceMatrix).astype(complex)


def estimated_volume_noise(net, frequency, slowness, n_sources=200,
                           n_snapshots=100):
    """ Estimate volume noise with random plane waves"""

    azimuths = np.linspace(0, 2 * np.pi, n_sources, endpoint=False)
    x, y = net.x, net.y
    covariance = np.zeros((net.dim, net.dim), dtype=complex)

    wb = waitbar('Volume noise estimate')
    for snapshot in range(n_snapshots):

        wb.progress((snapshot + 1) / n_snapshots)
        wavenumber = 2 * np.pi * frequency * slowness
        snapshots = np.zeros(net.dim, dtype=complex)
        phases = 2 * np.pi * np.random.rand(n_sources)

        for azimuth_id, (azimuth, phase) in enumerate(zip(azimuths, phases)):
            scalar_product = np.sin(azimuth) * x + np.cos(azimuth) * y
            k = wavenumber * np.random.rand(1)
            snapshots += np.exp(-1j * k * scalar_product - 1j * phase)

        snapshots /= n_sources
        covariance += snapshots * snapshots.conj()[:, None]
        covariance /= n_snapshots
    return covariance.view(CovarianceMatrix).astype(complex)


def spherical(net, frequency, slowness, xyz=None, llz=None, depth=0):
    """
    Monochromatic spherical wave. Return wavefield.
    XY coordinates must be in km.
    LL means lon, lat.
    """
    x, y = net.x, net.y

    if llz is not None:
        xy = ap.net.geo2xy(*llz[:2], reference=net.get_reference())
        z = llz[-1]
        xyz = (xy[0], xy[1], z)

    depth = depth + 0 * y
    r = np.sqrt((x - xyz[0]) ** 2 + (y - xyz[1]) ** 2 + (depth - xyz[2]) ** 2)
    wavenumber = 2 * np.pi * frequency * slowness
    focal = 1 / (r + 1e-6) * np.exp(-1j * wavenumber * r)
    covariance = xouter(focal)
    return covariance.view(CovarianceMatrix).astype(complex)


def spherical_wave(net, frequency, slowness, xyz=None, llz=None, depth=0):
    """
    Monochromatic spherical wave. Return wavefield.
    XY coordinates must be in km.
    LL means lon, lat.
    """
    x, y = net.x, net.y

    if llz is not None:
        xy = ap.net.geo2xy(*llz[:2], reference=net.get_reference())
        z = llz[-1]
        xyz = (xy[0], xy[1], z)

    depth = depth + 0 * y
    r = np.sqrt((x - xyz[0]) ** 2 + (y - xyz[1]) ** 2 + (depth - xyz[2]) ** 2)
    wavenumber = 2 * np.pi * frequency * slowness
    focal = 1 / (r + 1e-6) * np.exp(-1j * wavenumber * r)
    return focal


def cylindrical(net, frequency, slowness, coordinate=(0.0, 0.0)):
    """
    Monochromatic cylindrical wave. Return covariance matrix.
    Coordinates must be in km.
    """

    x, y = net.x, net.y
    r = np.sqrt((x - coordinate[0]) ** 2 + (y - coordinate[1]) ** 2)
    wavenumber = 2 * np.pi * frequency * slowness
    focal = 1 / np.sqrt(r + 1e-6) * np.exp(-1j * wavenumber * r)
    covariance = xouter(focal)
    return covariance.view(CovarianceMatrix).astype(complex)


def cylindrical_wave(net, frequency, slowness, coordinate=(0.0, 0.0)):
    """
    Monochromatic cylindrical wave. Return wavefield.
    Coordinates must be in km.
    """

    x, y = net.x, net.y
    r = np.sqrt((x - coordinate[0]) ** 2 + (y - coordinate[1]) ** 2)
    wavenumber = 2 * np.pi * frequency * slowness
    focal = 1 / np.sqrt(r + 1e-6) * np.exp(-1j * wavenumber * r)
    return focal


def random_noise(net, n=100):
    """Random independant normal noise.

    Parameters
    ----------

    net : :class:`arrayprocessing.antenna.Antenna`
        The seismic array metadata.

    Keyword arguments
    -----------------

    n : int
        Number of snapshots used for the estimation.

    Returns
    -------

    :class:`~arrayprocessing.covariance.CovarianceMatrix`
        The covariance matrix object.
    """

    # Initialization
    N = len(net)
    random = np.random.randn
    covariance = np.zeros((N, N), dtype=complex)

    # Estimation
    wb = waitbar('Noise estimate', n)
    for shot in range(n):
        wb.progress(shot)
        snapshot = random(N) + 1j * random(N)
        covariance += snapshot * snapshot.conj()[:, None]
    covariance /= n

    return covariance.view(CovarianceMatrix)
