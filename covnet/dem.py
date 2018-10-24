#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Read and show DEM obtained at "https://www.gmrt.org/GMRTMapTool/".

import numpy as np
import urllib

from matplotlib.colors import LinearSegmentedColormap


def read(path, depth_factor=1.):
    """ Read files downloaded as ASCII from https://www.gmrt.org/GMRTMapTool/.

    Automatically extracts the extent of the grid map, and elevation.

    Parameters
    ----------
    path : :obj:`str`
        The path to the digital elevation model in ArcASCII format. \
        This kind of format can be downloaded manually from \
        https://www.gmrt.org/GMRTMapTool/.

    Keyword arguments
    -----------------
    depth_factor : float
        Depth factor is used to scale the elevation values \
        (i.e. turn meters to kilometers, or altitude to \
        depth). Default is 1.0.

    Returns
    -------
    :obj:`np.ndarray`, :obj:`np.ndarray`, :obj:`np.ndarray`
        The longitudes and latitudes vectors and the elevation matrix.

    """

    with open(path, 'r') as file:
        n_lon = int(file.readline().split()[-1])
        n_lat = int(file.readline().split()[-1])
        west = float(file.readline().split()[-1])
        south = float(file.readline().split()[-1])
        cell_size = float(file.readline().split()[-1])
        no_data_value = int(file.readline().split()[-1])

    # Coordinates
    longitudes = np.linspace(west, west + n_lon * cell_size, n_lon)
    latitudes = np.linspace(south, south + n_lat * cell_size, n_lat - 1)

    # Extract elevation (m)
    elevation = np.loadtxt(path, skiprows=7)
    elevation[elevation == no_data_value] = 0
    elevation = elevation[::-1] * depth_factor

    return longitudes, latitudes, elevation


def read_cpt(path, symmetric=False, reverse=False, levels=256):
    """Turns a color palette table into a matplotlib segmented colormap.

    Parameters
    ----------
    path : :obj:`str`
        Path to the cpt file.

    Keyword arguments
    -----------------
    reverse : bool
        If True, the returned colormap is reversed.

    symmetric : bool
        If True, concatenates the colormap contains with a mirrored version \
        of itself

    levels : int
        Number of levels used to sample the colormap. Default is 256 levels.

    Returns
    -------
    :obj:`matplotlib.colors.LinearSegmentedColormap`
        The matplotlib linearly segmented colormap.

    """

    # Initialization
    color_dict = {}

    # Try to read a url
    try:
        url = 'http://soliton.vm.bytemark.co.uk/pub/'
        url += path + '.cpt'
        response = urllib.urlopen(url)
        rgb = np.genfromtxt(response, skip_header=3, skip_footer=3)

    except:

        # Second url read trial
        try:
            url = path
            response = urllib.urlopen(url)
            rgb = np.genfromtxt(response, skip_header=3, skip_footer=3)

        # Try to read from local file
        except:
            f = open(path, 'r')
            rgb = np.loadtxt(f)
            f.close()

    # Normalize RGB values
    rgb = rgb / 255.
    if reverse is True:
        rgb = rgb[::-1]

    # Assign color values into dictionary
    s = np.shape(rgb)
    colors = ['red', 'green', 'blue']

    # For each RGB, assign coefficient
    for c in colors:

        i = colors.index(c)
        x = rgb[:, i + 1]

        if symmetric:
            color_dict[c] = np.zeros((2 * s[0], 3), dtype=float)
            color_dict[c][:, 0] = np.linspace(0, 1, 2 * s[0])
            vec = np.concatenate((x, x[::-1]))
        else:
            color_dict[c] = np.zeros((s[0], 3), dtype=float)
            color_dict[c][:, 0] = np.linspace(0, 1, s[0])
            vec = x
        color_dict[c][:, 1] = vec
        color_dict[c][:, 2] = vec

    return LinearSegmentedColormap('palette', color_dict, levels)
