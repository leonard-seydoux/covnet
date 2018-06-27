#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Spatial beam forming.


import numpy as np
import arrayprocessing as ap
import matplotlib.pyplot as plt
import obspy.taup as taup


def create_map(figsize=2.5, extent=(131.5, 135, 32.5, 34.5),
               depth_lim=[0, 100]):
    """ Creation of a lon, lat and depth map.
    """
    ap.logtable.full_row('Geomapping')

    # Create
    ratio = (extent[1] - extent[0]) / (extent[3] - extent[2])
    ax = ap.Map(figsize=(figsize * ratio, figsize), extent=extent)
    ax.ticks(n_lat=5, n_lon=5)
    fig = ax.figure

    # Latitude depth
    ax_lat = fig.add_axes([1 + 0.1 / ratio, 0, 0.5 / ratio, 1])
    ax_lat.set_ylim(extent[2], extent[3])
    ax_lat.set_yticks(ax.get_yticks())
    ax_lat.set_yticklabels(ax.get_yticklabels())
    ax_lat.yaxis.tick_right()
    ax_lat.yaxis.set_ticks_position('both')
    ax_lat.set_xlim(depth_lim)
    ax_lat.set_xlabel('Depth (km)', fontsize=8)
    ax_lat.set_xticks(np.arange(depth_lim[0], depth_lim[1] + 1, 5))

    # Longitude depth
    ax_lon = fig.add_axes([0, -.6, 1, 0.5])
    ax_lon.set_xlim(extent[0], extent[1])
    ax_lon.set_xticks(ax.get_xticks())
    ax_lon.set_xticklabels(ax.get_xticklabels())
    ax_lon.set_ylim(depth_lim)
    ax_lat.set_xlabel('Depth (km)', fontsize=8)
    ax_lon.invert_yaxis()
    ax_lon.set_ylabel('Depth (km)', fontsize=8)
    ax_lon.set_yticks(np.arange(depth_lim[0], depth_lim[1] + 1, 5))

    # Colorbar
    cax = fig.add_axes([1 + 0.1 / ratio, -.3, 0.5 / ratio, 0.04])

    # Lasly
    ax.set_xticklabels([''])

    return (ax, ax_lon, ax_lat, cax), fig


# Simple
def shift_simple(traces, sampling_rate, stations, source, slowness):
    """
    Moves out traces matrix w.r.t. source location and depth.
    """

    # Copy
    # frequencies = np.linspace(0, stream[0].stats.sampling_rate, npts)

    # Distance of stations to source
    xy2src = ap.antenna.geo2xy(stations.lon, stations.lat, source[:2])
    distances = np.sqrt(xy2src[0] ** 2 + xy2src[1] ** 2 + source[2] ** 2)
    moveout = np.round(distances * slowness * sampling_rate).astype(int)
    moveout[moveout == 0] = 1
    traces = traces[:, :-max(moveout)]
    shifted = np.zeros(shape=traces.shape)

    # Calculate shift
    for trace_id in range(stations.dim):

        shifted[trace_id, :-moveout[trace_id]] = \
            traces[trace_id, moveout[trace_id]:]

    shifted /= np.sum(np.abs(shifted), axis=-1)[:, None]
    return shifted


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
        wb = ap.logtable.waitbar('Travel times')
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

    def calculate_with_ttimes(self, xcorr, fs, stations, ttimes, close=None):
        """ Shift cross-correlation for each source in grid.
        """

        # Initialization
        beam_max = 0
        trii, trij = np.triu_indices(stations.dim, k=1)
        xcorr_shifted_best = 0
        n_lon = self.shape[0]
        n_lat = self.shape[1]
        n_dep = self.shape[2]

        # Compute
        wb = ap.logtable.waitbar('Beam')
        for k in range(n_lon * n_lat * n_dep):

            wb.progress((k + 1) / (n_lon * n_lat * n_dep))
            i, j, k = np.unravel_index(k, (n_lon, n_lat, n_dep))
            src_distance = ttimes['distances'][:, i, j]
            sdid = [np.abs(ttimes['epicentral_distances'] - d).argmin()
                    for d in src_distance]
            tt = np.array([ttimes['ttimes'][s, sdid[s], k]
                           for s in range(stations.dim)])
            tt = tt[:, None] - tt
            tt = tt[trii, trij]

            if np.any(np.isnan(tt)):
                continue

            if close is not None:
                tt = tt[close]

            dt_int = -(fs * tt).astype(int)
            # dt_int = -(fs * tt).astype(int)

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
                xcorr_shifted_best = xcorr_shifted

        return xcorr_shifted_best

    def calculate(self, xcorr, fs, stations, slowness, close=None):
        """ Shift cross-correlation for each source in grid.

        Args
        ----
            xcorr (np.ndarray): cross-correlation upper-triangular matrix
                of shape (n_triu, n_times), where n_triu is the number of upper
                triangular elements, and n_times the number of lag times.
            fs (float): the correlation sampling rates for calculating the
                integer moveout with respect to a source in a constant-velocity
                model.
            stations (ap.Antenna): the seismic array.
            close (list of bool, optional): the indexes of closely selected
                stations.

        Return
        ------
            xcorr_best (np.ndarray): best-shifted correlation functions with
                respect to the beam maximal value.
        """

        # Initialization
        z = stations.z
        beam_max = 0
        trii, trij = np.triu_indices(stations.dim, k=1)
        n_lon = self.shape[0]
        n_lat = self.shape[1]
        n_dep = self.shape[2]

        # Compute
        wb = ap.logtable.waitbar('Beam')
        for k in range(np.prod(self.shape)):

            # Unravel indexes
            wb.progress((k + 1) / (n_lon * n_lat * n_dep))
            i, j, k = np.unravel_index(k, (n_lon, n_lat, n_dep))
            src = self.lon[i], self.lat[j], self.dep[k]

            # Moveouts
            dxy = ap.antenna.geo2xy(stations.lon, stations.lat, src[:2])
            distances = np.sqrt(dxy[0] ** 2 + dxy[1] ** 2 + (z - src[2]) ** 2)
            differential = distances[:, None] - distances
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
        beam *= .9

        # Position of maximum
        imax, jmax, kmax = np.unravel_index(beam.argmax(), beam.shape)

        # Setting map
        (axes), fig = create_map(figsize=2, depth_lim=depth_lim, extent=extent)

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
        axes[0].add_lands()
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
            return fig, axes
