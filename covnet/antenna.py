#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from numpy.lib.recfunctions import append_fields
from . import geo


def read(file_path, depth_factor=1e-3, **kwargs):
    """ Read seismic array metadata."""

    # Default options
    kwargs.setdefault('dtype', None)
    kwargs.setdefault('encoding', None)
    kwargs.setdefault('names', True)

    # Get metadata
    meta = np.genfromtxt(file_path, **kwargs)
    meta = meta.view(np.recarray)

    # Cartesian coordinates
    x, y = geo.deg2cart(meta['lon'], meta['lat'])
    if 'alt' in meta.dtype.names:
        z = -meta['alt'] * depth_factor
    else:
        z = np.zeros(meta.shape)

    # Append cartesian coordinates in meta
    meta = append_fields(meta, 'x', x)
    meta = append_fields(meta, 'y', y)
    meta = append_fields(meta, 'z', z)

    return meta.view(np.recarray).view(Antenna)


class Antenna(np.recarray):

    @property
    def dim(self):
        return len(self)

    @property
    def barycenter(self):
        """ Get array barycenter.

        Returns
        -------
        :obj:`tuple` of :class:`numpy.ndarray`
            Geographical coordinates of the barycenter ``(lon, lat, z)``

        """
        return np.mean(self.lon), np.mean(self.lat), np.mean(self.z)

    def distances(self, triangular=False, k=1):

        # Extract cartesian distances
        x2 = np.array(self['x'] - self['x'][:, None]) ** 2
        y2 = np.array(self['y'] - self['y'][:, None]) ** 2
        z2 = np.array(self['z'] - self['z'][:, None]) ** 2

        # Calculate distance matrix
        distances = np.sqrt(x2 + y2 + z2)

        if triangular is True:
            return distances[np.triu_indices(self.shape[0], k=k)]
        else:
            return distances

    def distances_from(self, lon_ref, lat_ref, z_ref):
        """Distances between each station and a reference point.

        Parameters
        ----------
        lon_ref : float
            The longitude of the reference point in decimal degrees.

        lat_ref : float
            The latitude of the reference point in decimal degrees.

        z_ref : float
            The depth of the reference point in km.

        Return
        ------
        :class:`numpy.ndarray`
            The distance between each array element and the reference point.


        Example
        -------

        .. execute_code::
            :hide_headers:

            import arrayprocessing as ap

            # Read csv station file
            csv_file = '../docs/demo/inputs/_US-TA-OpStationList.csv'
            network = ap.antenna.read_csv(csv_file, depth_factor=1e-3, sep=";")

            # Calculate distances of each stations to the barycenter
            distances = network.distances_from(*network.barycenter())
            print(distances)

        """

        # Get common barycenter
        *barycenter, _ = self.barycenter

        # Cartesian coordinates
        x, y = geo.deg2cart(self.lon, self.lat, ref=barycenter)
        z = np.array(self.z)
        x_ref, y_ref = geo.deg2cart(lon_ref, lat_ref, ref=barycenter)

        return np.sqrt((x - x_ref)**2 + (y - y_ref)**2 + (z - z_ref)**2)

    def select(self, names):

        selection = list()
        for name in names:
            selection.append(list(self['name']).index(name))

        return self[selection]
