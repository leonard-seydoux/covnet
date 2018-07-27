#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Spatial beam forming.


import numpy as np
# import arrayprocessing as ap
import matplotlib.pyplot as plt
import obspy.taup as taup

from . import logtable
from . import mapper


def initialize(shape=(10, 10, 10)):
    return np.zeros(shape).view(Beam)


class Beam(np.ndarray):

    def set_extent(self, west, east, south, north, depth_top, depth_max):
        """ Limits of the beamforming domain.

        Args
        ----
            west, east, south, north, depth_top and depth_max (float):
                Extent of the 3D map in degrees for the azimuths and km for
                the depths. The depth is given in km and should be negative for
                elevation.
        """

        self.extent = west, east, south, north
        self.lon = np.linspace(west, east, self.shape[0])
        self.lat = np.linspace(south, north, self.shape[1])
        self.dep = np.linspace(depth_top, depth_max, self.shape[2])
        self.grid = np.meshgrid(self.lon, self.lat, self.dep)
        self.grid_size = len(self.grid[0].ravel())

    def calculate_ttimes(self, stations, model=None, path='ttimes.npy'):

        # Initialization
        trii, trij = stations.triu
        n_lon = self.shape[0]
        n_lat = self.shape[1]
        n_dep = self.shape[2]
        ttimes = np.zeros((stations.dim, n_lon, n_lat, n_dep))

        # Compute
        wb = logtable.waitbar('Travel times')
        for grid in range(n_lon * n_lat * n_dep):

            wb.progress((grid + 1) / (n_lon * n_lat * n_dep))
            i, j, k = np.unravel_index(grid, (n_lon, n_lat, n_dep))
            src = self.lon[i], self.lat[j], self.dep[k]

            for sta in range(stations.dim):
                distance = taup.taup_geo.calc_dist(
                    src[1], src[0], stations.lat[sta], stations.lon[sta],
                    6378137.0, 0.0)

                arrivals = model.get_travel_times(
                    source_depth_in_km=src[2], distance_in_degree=distance,
                    phase_list=['s', 'S'])

                ttimes[sta, i, j, k] = arrivals[0].time

        np.save(path, ttimes)

    def calculate(self, xcorr, fs, stations, ttimes, close=None):
        """ Shift cross-correlation for each source in grid.
        """

        # Initialization
        beam_max = 0
        trii, trij = np.triu_indices(stations.dim, k=1)
        # xcorr_shifted_best = 0
        n_lon = self.shape[0]
        n_lat = self.shape[1]
        n_dep = self.shape[2]
        center = (xcorr.shape[1] - 1) // 2 + 1

        # Compute
        wb = logtable.waitbar('Beam', n_lon * n_lat * n_dep)
        for k in range(n_lon * n_lat * n_dep):
            wb.progress(k)

            # Differential travel times
            i, j, k = np.unravel_index(k, (n_lon, n_lat, n_dep))
            # src_distance = ttimes['distances'][:, i, j]
            # sdid = [np.abs(ttimes['epicentral_distances'] - d).argmin()
            #         for d in src_distance]
            # tt = np.array([ttimes['ttimes'][s, sdid[s], k]
            #                for s in range(stations.dim)])
            tt = ttimes[:, i, j, k]
            tt = tt[:, None] - tt
            tt = -tt[trii, trij]

            if np.any(np.isnan(tt)):
                continue

            if close is not None:
                tt = tt[close]

            dt_int = -(fs * tt).astype(int)
            dt_int = center + dt_int

            # Max
            beam = xcorr[range(xcorr.shape[0]), dt_int].sum()
            self[i, j, k] = beam
            if beam_max < beam:
                beam_max = beam

    def calculate_old(self, xcorr, fs, net, slowness, close=None):
        """ Shift cross-correlation for each source in grid.

        Args
        ----
            xcorr (np.ndarray): cross-correlation upper-triangular matrix
                of shape (n_triu, n_times), where n_triu is the number of upper
                triangular elements, and n_times the number of lag times.
            fs (float): the correlation sampling rates for calculating the
                integer moveout with respect to a source in a constant-velocity
                model.
            net (ap.Antenna): the seismic array.
            close (list of bool, optional): the indexes of closely selected
                net.

        Return
        ------
            xcorr_best (np.ndarray): best-shifted correlation functions with
                respect to the beam maximal value.
        """

        # Initialization
        beam_max = 0
        trii, trij = np.triu_indices(net.dim, k=1)
        n_lon = self.shape[0]
        n_lat = self.shape[1]
        n_dep = self.shape[2]

        # Compute
        wb = logtable.waitbar('Beam', np.prod(self.shape))
        for k in range(np.prod(self.shape)):

            # Unravel indexes
            wb.progress(k)
            i, j, k = np.unravel_index(k, (n_lon, n_lat, n_dep))
            src = self.lon[i], self.lat[j], self.dep[k]

            # Moveouts
            dxyz = net.distances_from(*src)
            differential = dxyz[:, None] - dxyz
            differential = differential[trii, trij]
            if close is not None:
                differential = differential[close]
            dt_int = -(fs * differential * slowness).astype(int)

            # Move
            rows, column_indices = np.ogrid[:xcorr.shape[0], :xcorr.shape[1]]
            dt_int[np.abs(dt_int) > xcorr.shape[1]] = xcorr.shape[1] - 1
            dt_int[dt_int < 0] += xcorr.shape[1]
            column_indices = column_indices - dt_int[:, np.newaxis]
            xcorr_shifted = xcorr[rows, column_indices]

            # Max
            beam = np.sum(xcorr_shifted, axis=0)[xcorr.shape[1] // 2]
            self[i, j, k] = beam
            if beam_max < beam:
                beam_max = beam
                xcorr_best = xcorr_shifted
                lon_max, lat_max, dep_max = src

        return xcorr_best

    def load(self, beamfile):
        """ Read already calculated beam file in npy format.
        """
        # Read
        n_lon = self.shape[0]
        n_lat = self.shape[1]
        n_dep = self.shape[2]
        beam = np.load(beamfile)

        # Compute
        wb = logtable.waitbar('Beam', n_lon * n_lat * n_dep)
        for k in range(np.prod(self.shape)):
            # Unravel indexes
            wb.progress(k)
            i, j, k = np.unravel_index(k, (n_lon, n_lat, n_dep))
            self[i, j, k] = beam[i, j, k]

    def show(self, stations, path=None, **kwargs):

        # Extents
        west, east = self.lon[[0, -1]]
        south, north = self.lat[[0, -1]]
        extent = west, east, south, north
        depth_lim = depth_min, depth_max = self.dep[[0, -1]]

        # Normalization
        beam = self
        beam[..., self.dep < 1] = np.nan
        beam = (beam - np.nanmin(beam)) / (np.nanmax(beam) - np.nanmin(beam))
        beam[np.isnan(beam)] = 0
        beam[np.isinf(beam)] = 0
        beam = beam ** 2
        # beam *= .9

        # Position of maximum
        imax, jmax, kmax = np.unravel_index(beam.argmax(), beam.shape)

        # Setting map
        axes = mapper.Map3(extent=extent, zlim=depth_lim)

        # Image keyword arguments
        kwargs.setdefault('origin', 'lower')
        kwargs.setdefault('vmax', 1)
        kwargs.setdefault('vmin', 0)
        kwargs.setdefault('aspect', 'auto')
        kwargs.setdefault('cmap', 'viridis')
        # kwargs.setdefault('interpolation', 'spline16')
        kwargs.setdefault('interpolation', 'None')

        # Show beam
        img = axes[0].imshow(beam[..., kmax].T, **kwargs)
        axes[0].plot(self.lon[imax], self.lat[jmax], 'w*', mec='w', ms=3)
        # axes[0].add_lands()
        img.set_extent((west, east, south, north))
        cb = plt.colorbar(img, cax=axes[-1], orientation='horizontal')
        cb.set_label('Beam', fontsize=10)
        cb.set_ticks([0, kwargs['vmax'] / 2, kwargs['vmax']])
        cb.ax.tick_params(which='both', direction='out', labelsize=8)

        # Longitude / latitude
        img = axes[1].imshow(beam[:, jmax, :].T, **kwargs)
        axes[1].plot(self.lon[imax], self.dep[kmax], 'w*', mec='w', ms=3)
        img.set_extent((west, east, depth_min, depth_max))

        # Depth / latitude
        img = axes[2].imshow(beam[imax, ...], **kwargs)
        axes[2].plot(self.dep[kmax], self.lat[jmax], 'w*', mec='w', ms=3)
        img.set_extent((depth_min, depth_max, south, north))

        # Stations
        axes[0].plot(stations.lon, stations.lat, 'kv', ms=3,
                     transform=axes[0].projection)

        # Save figure
        if path is not None:
            plt.savefig(path, bbox_inches='tight', dpi=600)
        else:
            return axes
