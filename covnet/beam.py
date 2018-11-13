#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Spatial beam forming.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import obspy.taup as taup

from numba import jit
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

    def calculate_ttimes(self, stations, model=None, save_path='ttimes.npy'):

        # Initialization
        trii, trij = np.triu_indices(stations.dim, k=1)
        n_lon = self.shape[0]
        n_lat = self.shape[1]
        n_dep = self.shape[2]
        ttimes = np.zeros((stations.dim, n_lon, n_lat, n_dep))
        earthradius = 6378137.0
        phase_list = ['s', 'S']

        # Compute
        wb = logtable.waitbar('Travel times', n_lon * n_lat * n_dep)
        for grid in range(n_lon * n_lat * n_dep):
            wb.progress(grid)

            i, j, k = np.unravel_index(grid, (n_lon, n_lat, n_dep))
            src = self.lon[i], self.lat[j], self.dep[k]

            for sta in range(stations.dim):
                distance = taup.taup_geo.calc_dist(
                    src[1], src[0], stations.lat[sta], stations.lon[sta],
                    earthradius, 0.0)

                arrivals = model.get_travel_times(
                    source_depth_in_km=src[2] - self.dep[0],
                    distance_in_degree=distance,
                    phase_list=phase_list,
                    receiver_depth_in_km=stations.z[sta] - self.dep[0])

                try:
                    ttimes[sta, i, j, k] = arrivals[0].time
                except IndexError:
                    ttimes[sta, i, j, k] = np.nan

        np.save(save_path, ttimes)

    def calculate_heterogeneous(self, xcorr, fs, stations, ttimes, close=None):
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

                # Keep that into memory
                beam_max = beam
                dt_int_abs = -(dt_int - center)

        # Move
        rows, column_indices = np.ogrid[:xcorr.shape[0], :xcorr.shape[1]]
        dt_int_abs[np.abs(dt_int_abs) > xcorr.shape[1]] = xcorr.shape[1] - 1
        dt_int_abs[dt_int_abs < 0] += xcorr.shape[1]
        column_indices = column_indices - dt_int_abs[:, np.newaxis]
        xcorr_best = xcorr[rows, column_indices]

        return xcorr_best.T

    def calculate_homogeneous(self, xcorr, fs, net, slowness, close=None):
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
        center = (xcorr.shape[1] - 1) // 2 + 1
        fss = fs * slowness
        # rows = range(xcorr.shape[0])

        # Compute
        wb = logtable.waitbar('Beam', np.prod(self.shape))
        for k in range(np.prod(self.shape)):

            # Unravel indexes
            wb.progress(k)
            i, j, k = np.unravel_index(k, (n_lon, n_lat, n_dep))
            src = self.lon[i], self.lat[j], self.dep[k]

            # Moveouts
            dxyz = net.distances_from(*src)
            ddiff = dxyz[:, None] - dxyz
            # ddiff = ddiff[trii, trij]
            # if close is not None:
            #     ddiff = ddiff[close]
            dt_int = center + (fss * ddiff[trii, trij]).astype(int)
            # dt_int = center + dt_int

            # Max
            # beam = (xcorr[range(xcorr.shape[0]), dt_int] ** 2).sum()
            self[i, j, k] = (xcorr[range(xcorr.shape[0]), dt_int] ** 2).sum()

            if beam_max < self[i, j, k]:

                # Keep that into memory
                beam_max = self[i, j, k]
                dt_int_abs = -(dt_int - center)

        # Move
        rows, column_indices = np.ogrid[:xcorr.shape[0], :xcorr.shape[1]]
        dt_int_abs[np.abs(dt_int_abs) > xcorr.shape[1]] = xcorr.shape[1] - 1
        dt_int_abs[dt_int_abs < 0] += xcorr.shape[1]
        column_indices = column_indices - dt_int_abs[:, np.newaxis]
        xcorr_best = xcorr[rows, column_indices]

        return xcorr_best.T

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

    def show(self, stations, path=None, normalize=True,
             axes_kw=dict(), **kwargs):

        # Extents
        west, east = self.lon[[0, -1]]
        south, north = self.lat[[0, -1]]
        extent = west, east, south, north
        depth_lim = depth_min, depth_max = self.dep[[0, -1]]

        # Normalization
        beam = self
        beam[..., self.dep < 1] = np.nan
        if normalize is True:
            beam = (beam - np.nanmin(beam)) /\
                (np.nanmax(beam) - np.nanmin(beam))
        beam[np.isnan(beam)] = 0
        beam[np.isinf(beam)] = 0
        beam = beam ** 2

        # Position of maximum
        imax, jmax, kmax = np.unravel_index(beam.argmax(), beam.shape)

        # Setting map
        axes_kw.setdefault('figsize', 2.5)
        axes_kw.setdefault('n_lon', 5)
        axes_kw.setdefault('n_lat', 5)
        axes = mapper.Map3(extent=extent, zlim=depth_lim, **axes_kw)

        # Image keyword arguments
        kwargs.setdefault('origin', 'lower')
        kwargs.setdefault('aspect', 'auto')
        kwargs.setdefault('cmap', 'viridis')
        # kwargs.setdefault('interpolation', 'spline16')
        kwargs.setdefault('interpolation', 'None')

        # Show beam
        img = axes[0].imshow(beam[..., kmax].T, **kwargs)
        axes[0].plot(self.lon[imax], self.lat[jmax], 'w*', mec='k', ms=5,
                     clip_on=False)
        # axes[0].add_lands()
        img.set_extent((west, east, south, north))
        cb = plt.colorbar(img, cax=axes[-1], orientation='horizontal')
        cb.set_label('Beam', fontsize=10)
        # cb.set_ticks([0, kwargs['vmax'] / 2, kwargs['vmax']])
        cb.ax.tick_params(which='both', direction='out', labelsize=8)

        # Longitude / depth
        img = axes[1].imshow(beam[:, jmax, :].T, **kwargs)
        axes[1].plot(self.lon[imax], self.dep[kmax], 'w*', mec='k', ms=5,
                     clip_on=False)
        img.set_extent((west, east, depth_min, depth_max))

        # Depth / latitude
        img = axes[2].imshow(beam[imax, ...], **kwargs)
        axes[2].plot(self.dep[kmax], self.lat[jmax], 'w*', mec='k', ms=5,
                     clip_on=False)
        img.set_extent((depth_min, depth_max, south, north))

        # Stations
        axes[0].plot(stations.lon, stations.lat, 'kv', ms=3,
                     transform=axes[0].projection)

        # Save figure
        if path is not None:
            plt.savefig(path, bbox_inches='tight', dpi=600)
        else:
            return axes

    def contour(self, stations, path=None, normalize=True,
                axes_kw=dict(), levels=10, low=4,  **kwargs):

        # Extents
        west, east = self.lon[[0, -1]]
        south, north = self.lat[[0, -1]]
        extent = west, east, south, north
        depth_lim = depth_min, depth_max = self.dep[[0, -1]]

        # Normalization
        beam = self
        beam[..., self.dep < 1] = np.nan
        if normalize is True:
            beam = (beam - np.nanmin(beam)) /\
                (np.nanmax(beam) - np.nanmin(beam))
        beam[np.isnan(beam)] = 0
        beam[np.isinf(beam)] = 0
        beam = beam ** 2

        # Position of maximum
        imax, jmax, kmax = np.unravel_index(beam.argmax(), beam.shape)

        # Setting map
        axes_kw.setdefault('figsize', 2.5)
        axes_kw.setdefault('n_lon', 5)
        axes_kw.setdefault('n_lat', 5)
        axes = mapper.Map3(extent=extent, zlim=depth_lim, **axes_kw)

        # Image keyword arguments
        cmap = plt.cm.get_cmap(kwargs['cmap'])(np.linspace(0, 1, levels))
        for i in range(low):
            cmap[i, :] = [1, 1, 1, 1]
        for i in range(1, levels):
            cmap[i, -1] = np.sqrt(i / levels)
        cmap_save = cmap
        cmap = mcolors.LinearSegmentedColormap.from_list('cmap', cmap)

        # Show beam
        img = axes[0].contourf(
            self.lon, self.lat, beam[..., kmax].T, levels, cmap=cmap)

        axes[0].plot(self.lon[imax], self.lat[jmax], '*', mec='k', ms=6,
                     clip_on=False, mfc='k', zorder=20)
        # axes[0].fancy_ticks()
        cb = plt.colorbar(img, cax=axes[-1], orientation='horizontal')
        cb.set_label('Beam', fontsize=8)
        cb.set_ticks([0, 1 / 2, 1])
        cb.ax.tick_params(which='both', direction='out', labelsize=8)

        # Longitude / depth
        img = axes[1].contourf(self.lon, self.dep, beam[
                               :, jmax, :].T, levels, cmap=cmap)
        axes[1].plot(self.lon[imax], self.dep[kmax], '*', mec='k', ms=6,
                     clip_on=False, mfc='k', zorder=20)

        # Depth / latitude
        img = axes[2].contourf(self.dep, self.lat, beam[
                               imax, ...], levels, cmap=cmap)
        axes[2].plot(self.dep[kmax], self.lat[jmax], '*', mec='k', ms=6,
                     clip_on=False, mfc='k', zorder=20)

        # Stations
        axes[0].plot(stations.lon, stations.lat, 'v', ms=4, mfc='w',
                     mec='k', transform=axes[0].projection, zorder=20)

        # Save figure
        if path is not None:
            plt.savefig(path, bbox_inches='tight', dpi=600)
        else:
            return axes
