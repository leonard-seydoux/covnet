#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Manipulate geographical and cartesian coordinate systems.

"""

import numpy as np


def deg2cart(lon, lat, ref=None, earth_radius=6371.):
    """ Geographical to cartesian coordinates, assuming spherical Earth.

    The calculation is based on the approximate great-circle distance \
    formulae and a spherical Earth of radius :math:`R`. The cartesian \
    coordinates :math:`(x, y)` are obtained from the longitude \
    :math:`\\varphi` and the latitude :math:`\lambda`, and from the \
    reference coordinates :math:`(\\varphi_0, \lambda_0)` as

    .. math::

        x &= 2 \pi R (\\varphi_0 - \\varphi) \\cos(\lambda + \lambda_0)

        y &= 2 \pi R (\lambda_0 - \lambda)


    Parameters
    ----------
    lon : float or array-like
        Geographical longitude in decimal degrees.
    lat : float or array-like
        Geographical latitude in decimal degrees.


    Keyword Arguments
    -----------------
    ref : tuple
        Reference geographical coordinates for great-circle distance \
        calculation. Default is average of longitude and latitude (barycenter).

    earth_radius : float
        Radius of the spherical Earth in km. Default is 6371.0 km.

    Returns
    -------
    x: float or array
        East-west distance from reference in km.
    y: float or array
        North-south distance from reference in km.

    Example
    -------
    Convert coordinates in lists `lon` and `lat` with reference point (0, 0):

    .. code-block:: python

        lon = [-1., 0., 1.]
        lat = [-10., 0., 20.]
        x, y = deg2cart(lon, lat, ref=(0,0))
        print(x, y)

    .. code-block:: python

        [ 110.77179638    0.         -109.50562586]
        [ 1111.94926645    -0.         -2223.89853289]

    """

    # Convert to arrays
    if isinstance(lon, list):
        lon = np.array(lon)
        lat = np.array(lat)

    # Get reference coordinates
    if ref is None:
        lon_0 = np.mean(lon)
        lat_0 = np.mean(lat)
    else:
        lon_0, lat_0 = ref

    # Convert to radians
    lon = np.radians(lon)
    lat = np.radians(lat)
    lon_0 = np.radians(lon_0)
    lat_0 = np.radians(lat_0)

    # Calculus
    x = earth_radius * np.cos(lon_0 - lon) * np.sin(lat - lat_0)
    y = earth_radius * np.sin(lon_0 - lon) * np.cos(lat - lat_0)

    return x, y


if __name__ == '__main__':

    lon = [-1, 0., 1]
    lat = [0, 0, 0]
    x, y = deg2cart(lon, lat, ref=(0, 0))
    print(x, y)
