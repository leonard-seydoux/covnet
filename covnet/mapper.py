#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Mapping tools derived from the :mod:`cartopy` library.

"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

from matplotlib import ticker
from matplotlib import patheffects
from cartopy import crs
from cartopy.mpl import geoaxes
from copy import deepcopy
from cartopy.feature import NaturalEarthFeature as nef
from cartopy.crs import TransverseMercator
from matplotlib.colors import LightSource

from . import dem


def dd2dms(angle, sec=False):
    """ Decimal degrees to degrees, minutes and seconds.

    Parameters
    ----------
    angle : float
        Decimal degrees.

    Keyword arguments
    -----------------
    sec : bool
        Whether or not to compute seconds

    Returns
    -------
    tuple
        Degrees, minutes and seconds.

    Example
    -------

    .. execute_code::
        :hide_headers:

        import arrayprocessing as ap

        dms = ap.mapper.dd2dms(40.76, sec=True)
        print(dms)

    """

    # Split whole integer and decimal parts
    decimal_part, degrees = math.modf(angle)

    # Degrees are integer part
    degrees = int(degrees)

    # Multiply the decimal part by 60: 0.3478 * 60 = 20.868
    # Split the whole number part of the total as the minutes: 20
    # abs() absoulte value - no negative
    minutes = abs(int(math.modf(decimal_part * 60)[1]))

    # Multiply the decimal part of the split above by 60 to get the seconds
    # 0.868 x 60 = 52.08, round excess decimal places to 2 places
    # abs() absoulte value - no negative
    if sec is True:
        seconds = abs(round(math.modf(decimal_part * 60)[0] * 60, 2))

    if sec is True:
        return degrees, minutes, seconds
    else:
        return degrees, minutes


def minus_formatter(s):
    """ Mathematical minus formatter.

    Replace minus sign with the mathematical unicode 2212 character.

    Parameters
    ----------
        s : str
            A string into which to look for minus sign.

    Returns
    -------
    str
        Minus-formatted string.

    Example
    -------

    .. execute_code::
        :hide_headers:

        import arrayprocessing as ap

        s = '-30'
        s_formatted = ap.mapper.minus_formatter(s)
        print("Natural:", s)
        print("Formatted:", s_formatted)

    """
    return "{}".format(s).replace("-", u"\u2212")


def dmsfmt(degrees, minutes):
    """ String formatting of degrees and minutes.

    Parameters
    ----------
    degrees : int
        Degrees

    minutes : int
        Minutes

    Return
    ------
    str
        The formatted degrees and minute string.

    Example
    -------

    .. execute_code::
        :hide_headers:

        import arrayprocessing as ap

        dms = ap.mapper.dmsfmt(30, 40)
        print(dms)

    """

    dms = '{}\N{DEGREE SIGN}{}'
    if minutes > 0:
        fmt = minus_formatter(dms.format(degrees, minutes))
    else:
        fmt = minus_formatter(dms.format(degrees, ''))

    return fmt


class Map(geoaxes.GeoAxes):
    """ Create a map based on cartopy GeoAxes with additional utilities.

    Keyword arguments
    -----------------

    figsize : :obj:`tuple`
        The size of the figure in inches.

    ax : :obj:`pyplot.Axes`
        Axes for plotting the map. In order to create axes with correct \
        projection, a copy of `ax` is created, and the original `ax` deleted. \
        This can be useful for instance if the map is to be included within \
        subplots.

    extent : :obj:`tuple` or :obj:`list`
        The geographical extent of the axes in degrees. \
        Uses the same convention that with :mod:`cartopy` \
        e.g. extent = (west, east, south, north).

    projection : :class:`cartopy.crs`
        Projection of the map. Default to `crs.Platecarree().`\
        For more details about projections, please \
        refer to: \
        http://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html


    """

    def __init__(self, figsize=[4, 5], ax=None, extent=None,
                 projection=crs.PlateCarree()):

        # Create axes if none are given
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0, 0, 1, 1], projection=projection)
            ax.outline_patch.set_lw(0.5)
            self.__dict__ = ax.__dict__

        else:
            fig = ax.figure
            ax.set_axis_off()
            ax = fig.add_axes(ax.get_position(), projection=projection)
            ax.outline_patch.set_lw(0.5)
            self.__dict__ = ax.__dict__

        # Set extents
        if extent is not None:
            self.set_extent(extent)

    def add_global_location(self, position=[.7, -.1, .45, .45], label=None,
                            land_color='0.6', ocean_color='#efefef', **kwargs):
        """ Insert the global position shown in a small Earth inset.

        The projection used for the globe axes is :obj:`crs.Orthographic`, \
        latitude and longitude centered onto the map central point. \
        The shadow is obtained by axes overlaying on the top.


        Keyword arguments
        -----------------
        position : :obj:`tuple`
            The position of the globe axes in the figure. \
            Default to [.7, -.1, .45, .45] (e.g. bottom-right).

        land_color : :obj:`str` or :obj:`tuple`
            Color of land shapes. Default is gray (0.6).

        ocean_color : :obj:`str` or :obj:`tuple`
            Color of oceans. Default is very light grey (#efefef).

        **kwargs : :obj:`dict`
            Other kwargs passed to the :meth:`plt.plot()` function for the \
            location label style. Red square by default.

        """

        # Get local location from extent (bottom-left corner)
        extent = self.get_extent()

        # Global projection
        central_longitude = (extent[1] + extent[0]) / 2
        central_latitude = (extent[3] + extent[2]) / 2
        globe = crs.Orthographic(central_longitude=central_longitude,
                                 central_latitude=central_latitude)

        # Create axes
        ax = self.figure.add_axes(position, projection=globe)
        ax.outline_patch.set_lw(0.5)

        # Add lands and oceans
        lands = nef('physical', 'land', '110m')
        oceans = nef('physical', 'ocean', '110m')
        ax.add_feature(lands, facecolor=land_color, lw=0)
        ax.add_feature(oceans, facecolor=ocean_color, lw=0)
        ax.set_global()

        # Add gridlines
        gl = ax.gridlines(linewidth=0.2, linestyle='-')
        gl.xlocator = ticker.FixedLocator(range(-180, 181, 20))
        gl.ylocator = ticker.FixedLocator(range(-90, 91, 15))

        # Add second axes for the shadow
        globe = crs.Orthographic(central_longitude=150, central_latitude=0)
        ax = self.figure.add_axes(position, projection=globe)
        ax.background_patch.set_fill(False)
        ax.outline_patch.set_lw(0)
        ax.outline_patch.set_edgecolor('0.5')

        # Shadow
        shift = 5
        night = np.arctan(np.linspace(-2 * shift, 2 * shift, 500) + shift) - \
            np.arctan(np.linspace(-2 * shift, 2 * shift, 500) - shift)
        night, _ = np.meshgrid(night, night)
        ax.imshow(night, extent=(-180, 180, -90, 90),
                  zorder=10, cmap='Greys',
                  alpha=0.5, transform=crs.PlateCarree(),
                  interpolation='bicubic')

        # Add square centered on axes
        # The symbol in plotted on the top of the shadow axes
        # in order to be clearly seen.
        # Because the background axes are centered in the map center,
        # we can just plot the square in the middle of the map:
        ax.plot(0, 0, 's', alpha=1, zorder=20, **kwargs)

        # If label is not none, label
        if label is not None:
            ax.text(0, 0, label + '\n', ha='center', size=6,
                    weight='bold', style='italic')

    def add_nef(self, kind='physical', which='land', res='10m', **kwargs):
        """ Add natural earth feature to the axes.

        This function combines the \
        :meth:`cartopy.feature.NaturalEarthFeature` function and \
        the :meth:`axes.add_feature()` method.


        Keyword arguments
        -----------------
        kind : :obj:`str`
            Natural Earth feature type (e.g. 'physical', 'cultural', ...). \
            You can check the available features at : \
            http://www.naturalearthdata.com/features/

        which : :obj:`str`
            Which natural Earth feature (e.g. for 'physical' get 'land'). \
            You can check the available features at : \
            http://www.naturalearthdata.com/features/

        res : :obj:`str`
            Natural Earth resolution (e.g. '10m', '50m' or '110m').

        **kwargs : :obj:`dict`
            The plotting style passed to the :meth:`add_feature` method.

        """

        # Get the feature
        feature = nef(kind, which, res)

        # Add to axe
        self.add_feature(feature, **kwargs)

    def fancy_ticks(self, thickness=0.03, n_lon=7, n_lat=5, size=9):
        """ Add fancy ticks to the map (similar to the fancy GMT basemap).

        Keyword arguments
        -----------------
        thickness : float
            The thickness of the fancy bar (in figure width ratio).

        n_lon : int
            The number of longitudes ticks. Default 5.

        n_lat : int
            The number of latitudes ticks. Default 5.

        size : int
            Font size for the labels in points.

        """

        # Extract map meta from ax
        inner = self.get_extent(crs=crs.PlateCarree())
        proj = crs.PlateCarree()

        # Define outer limits
        outer = [dc for dc in deepcopy(inner)]

        [w, h] = self.figure.get_size_inches()
        fig_ratio = h / w
        width = inner[1] - inner[0]
        height = inner[3] - inner[2]
        ratio = height / width
        outer[0] -= width * thickness * fig_ratio
        outer[1] += width * thickness * fig_ratio
        outer[2] -= height * thickness
        outer[3] += height * thickness

        # Inner limits
        inner_lon = np.linspace(inner[0], inner[1], n_lon)
        inner_lat = np.linspace(inner[2], inner[3], n_lat)

        # Black and white styles
        w = dict(lw=.5, edgecolor='k', facecolor='w', transform=proj, zorder=9)
        b = dict(lw=0, facecolor='k', clip_on=False, transform=proj, zorder=10)

        # White frame
        self.fill_between([outer[0], outer[1]], outer[2], inner[2], **w)
        self.fill_between([outer[0], outer[1]], outer[3], inner[3], **w)
        self.fill_between([outer[0], inner[0]], inner[2], inner[3], **w)
        self.fill_between([outer[1], inner[1]], inner[2], inner[3], **w)

        # Create frame
        bottom_heigth = (outer[2], inner[2])
        top_heigth = (outer[3], inner[3])
        for index, limits in enumerate(zip(inner_lon[:-1], inner_lon[1:])):
            self.fill_between(limits, *bottom_heigth, **w)
            self.fill_between(limits, *top_heigth, **w)
            if index % 2 == 0:
                self.fill_between(limits, *bottom_heigth, **b)
                self.fill_between(limits, *top_heigth, **b)

        left_width = (outer[0], inner[0])
        right_width = (outer[1], inner[1])
        for index, height in enumerate(zip(inner_lat[:-1], inner_lat[1:])):
            self.fill_between(left_width, *height, **w)
            self.fill_between(right_width, *height, **w)
            if index % 2 == 0:
                self.fill_between(left_width, *height, **b)
                self.fill_between(right_width, *height, **b)

        self.set_xticks(inner_lon, crs=crs.PlateCarree())
        self.set_yticks(inner_lat, crs=crs.PlateCarree())
        self.xaxis.set_tick_params(length=0, labelsize=size)
        self.yaxis.set_tick_params(length=0, labelsize=size)
        lons = [dmsfmt(*dd2dms(l)) for l in inner_lon]
        lats = [dmsfmt(*dd2dms(l)) for l in inner_lat]
        self.set_xticklabels(lons)
        self.set_yticklabels(lats)
        self.set_extent(outer)

    def matshow(self, matrix, extent=None, **kwargs):
        """ Imshow with automatic parameters.

        Parameters
        ----------

        matrix : :obj:`np.ndarray`
            The matrix to be displayed. Directly passed to :meth:`plt.imshow` \
            function.

        Keyword arguments
        -----------------
        extent : :obj:`list`
            The extents of the matrix. Default are the map extents.

        kwargs : :obj:`dict`
            The options passed to the `plt.imshow` function. Defaults set \
            the origin of the matrix to the bottom-left corner, the transform \
            is adapted to the actual axes projection, and the extent of the \
            image to the map frame, unless extent is given.

        Return
        ------
        :class:`matplotlib.image.AxesImage`
            The image instance.

        """

        # Extent
        extent = self.get_extent() if extent is None else extent

        # Default imshow parameters
        kwargs.setdefault('transform', crs.PlateCarree())
        kwargs.setdefault('origin', 'lower')
        kwargs.setdefault('extent', extent)

        # Show image
        return self.imshow(matrix, **kwargs)

    def symbols(self, *args, **kwargs):
        """ Plot wrapper with map-like parameters.

        Please refer to the :meth:`plt.plot()` documentation:
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html?highlight=plot#matplotlib.pyplot.plot

        By default, the `kwargs` include an `transform` key with the
        actual axes projection.

        """

        # Set projection
        kwargs.setdefault('transform', crs.PlateCarree())
        return self.plot(*args, **kwargs)

    def ticks(self, n_lon=7, n_lat=5):
        """ Add normal ticks to the map.

        Keyword arguments
        -----------------
        n_lon : int
            The number of longitudes ticks.

        n_lat : int
            The number of latitudes ticks.

        """

        # Extract map meta from ax
        extent = self.get_extent(crs=crs.PlateCarree())
        extent_lon = np.linspace(extent[0], extent[1], n_lon)
        extent_lat = np.linspace(extent[2], extent[3], n_lat)

        self.set_xticks(extent_lon, crs=crs.PlateCarree())
        self.set_yticks(extent_lat, crs=crs.PlateCarree())
        lonlabels = [dmsfmt(*dd2dms(l)) for l in extent_lon]
        latlabels = [dmsfmt(*dd2dms(l)) for l in extent_lat]
        self.set_xticklabels(lonlabels)
        self.set_yticklabels(latlabels)

    def ticks_decimals(self, n_lon=7, n_lat=5, decimals=2):
        """ Add normal ticks to the map.

        Keyword arguments
        -----------------
        n_lon : int
            The number of longitudes ticks.

        n_lat : int
            The number of latitudes ticks.

        """

        # Extract map meta from ax
        extent = self.get_extent()
        extent_lon = np.linspace(extent[0], extent[1], n_lon)
        extent_lat = np.linspace(extent[2], extent[3], n_lat)

        self.set_xticks(extent_lon)
        self.set_yticks(extent_lat)
        lonlabels = [dmsfmt(np.round(l, decimals), 0) for l in extent_lon]
        latlabels = [dmsfmt(np.round(l, decimals), 0) for l in extent_lat]
        self.set_xticklabels(lonlabels)
        self.set_yticklabels(latlabels)

    def set_grid(self, extent, n_lon=5, n_lat=5):
        """ Add a grid to the current axes.

        Keyword arguments
        -----------------
        n_lon : int
            The number of longitudes ticks.

        n_lat : int
            The number of latitudes ticks.

        """

        # Get extents
        extent_lon = np.linspace(extent[0], extent[1], n_lon)
        extent_lat = np.linspace(extent[2], extent[3], n_lat)
        lon = np.linspace(extent[0], extent[1], 100)
        lat = np.linspace(extent[2], extent[3], 100)

        for la in range(n_lat):
            self.plot(lon, extent_lat[la] * np.ones(100),
                      transform=crs.Geodetic(), ls='--', lw=.5, c='k')

        for lo in range(n_lon):
            self.plot(extent_lon[lo] * np.ones(100), lat,
                      transform=crs.Geodetic(), ls='--', lw=.5, c='k')

    def label(self, x, y, labels, spread=0.01, fontsize=6, color='k'):
        """ Display labels with best-place spreading.

        Parameters
        ----------
        x, y : :obj:`np.array` or float
            The original coordinates for each label

        labels : :obj:`list`
            A list of the labels in :obj:`str` format.

        Keyword arguments
        -----------------

        spread : float
            Controls the spreading of the labels. If close to 0, then the \
            labels are collapsed to their original position. Default is .01.

        fontsize : int
            Size of the labels font.

        color : :obj:`str` or :obj:`tuple`
            Color of the text label.

        """

        G = nx.DiGraph()
        data_nodes = []
        init_pos = {}
        for xi, yi, label in zip(x, y, labels):
            data_str = 'data_{0}'.format(label)
            G.add_node(data_str)
            G.add_node(label)
            G.add_edge(label, data_str)
            data_nodes.append(data_str)
            init_pos[data_str] = (xi, yi)
            init_pos[label] = (xi, yi)

        pos = nx.spring_layout(G, pos=init_pos, fixed=data_nodes, k=spread)

        # undo spring_layout's rescaling
        pos_after = np.vstack([pos[d] for d in data_nodes])
        pos_before = np.vstack([init_pos[d] for d in data_nodes])
        scale, shift_x = np.polyfit(pos_after[:, 0], pos_before[:, 0], 1)
        scale, shift_y = np.polyfit(pos_after[:, 1], pos_before[:, 1], 1)
        shift = np.array([shift_x, shift_y])
        for key, val in pos.items():
            pos[key] = (val * scale) + shift

        for label, data_str in G.edges():
            t = self.annotate(
                label, xy=pos[data_str],
                xycoords=crs.PlateCarree()._as_mpl_transform(self),
                xytext=pos[label],
                textcoords=crs.PlateCarree()._as_mpl_transform(self),
                color=color,
                arrowprops=dict(
                    arrowstyle="-", linewidth=0.2, connectionstyle="arc",
                    color='k'),
                bbox=dict(
                    boxstyle='square,pad=0', facecolor=(0, 0, 0, 0),
                    linewidth=0))

            t.set_fontname('Consolas')
            t.set_fontsize(fontsize)
            t.set_fontweight('bold')
            t.set_path_effects([patheffects.Stroke(
                linewidth=0.6, foreground='w'), patheffects.Normal()])

    def scale_bar(self, length=50, location=(0.5, 0.06), lw=1, **kwargs):
        """ Add a scale bar to the axes.

        Keyword arguments
        -----------------

        length : float
            Length of the bar in km.

        location : :obj:`tuple`
            Location of the bar in axes nomarlized coordinates.

        lw : float
            Scale bar line width in points.

        """

        # Get the limits of the axis in lat long
        west, east, south, north = self.get_extent(self.projection)

        # Make tmc horizontally centred on the middle of the map,
        # vertically at scale bar location
        horizontal = (east + west) / 2
        vertical = south + (north - south) * location[1]
        tmc = TransverseMercator(horizontal, vertical)

        # Get the extent of the plotted area in coordinates in metres
        left, right, bottom, top = self.get_extent(tmc)

        # Turn the specified scalebar location into coordinates in metres
        bar_x = left + (right - left) * location[0]
        bar_y = bottom + (top - bottom) * location[1]

        # Generate the x coordinate for the ends of the scalebar
        left_x = [bar_x - length * 500, bar_x + length * 500]

        # Plot the scalebar
        self.plot(left_x, 2 * [bar_y], '|-',
                  transform=tmc, color='k', lw=lw, mew=lw, **kwargs)

        # Plot the scalebar label
        bar_text = str(length) + ' km'
        text_y = bottom + (top - bottom) * (location[1] + 0.01)
        self.text(bar_x, text_y, bar_text, transform=tmc, ha='center',
                  va='bottom', weight='bold', **kwargs)

    def add_dem(self, dem_file, cpt_topography=None,
                cpt_bathymetry=None, sun_azimuth=230,
                sun_altitude=15, topo_exag=100, bathy_exag=100):
        """ Add the digital elevation model.

        Parameters
        ----------

        dem_file : :obj:`str`
            The path to the digital elevation model in ArcASCII format. \
            This kind of format can be downloaded manually from \
            https://www.gmrt.org/GMRTMapTool/.

        Keyword arguments
        -----------------

        cpt_topography : :obj:`str`
            Path to the color palette table for topography.

        cpt_bathymetry : :obj:`str`
            Path to the color palette table for topography.

        sun_azimuth : float
            Azimuth of the sun in degrees for hill shading.

        sun_altitude : float
            Altitude of the sun in degrees for hill shading.

        topo_exag : float
            Vertical exageration of the topography.

        bathy_exag : float
            Vertical exageration of the bathymetry.

        """

        # Read dem
        lon, lat, elevation = dem.read(dem_file)
        georef = [lon[0], lon[-1], lat[0], lat[-1]]
        cmap_topo = dem.read_cpt(cpt_topography)
        cmap_bathy = dem.read_cpt(cpt_bathymetry)

        # Sun properties
        sun = LightSource(azdeg=sun_azimuth, altdeg=sun_altitude)
        sun_kw = dict(blend_mode='soft')

        # Topography
        topo = elevation.copy()
        topo[topo < 0] = 0
        img = (topo - np.nanmin(topo)) / (np.nanmax(topo) - np.nanmin(topo))
        img = sun.shade(img, cmap=cmap_topo, vert_exag=topo_exag,
                        fraction=1.1, **sun_kw)
        self.matshow(img, extent=georef)

        # Colorbar topography
        img = self.matshow(topo, cmap=cmap_topo)
        img.remove()
        cax = self.figure.add_axes([1.1, 0.5, 0.03, 0.4])
        plt.colorbar(img, cax=cax, extend='max')

        # Bathymetry
        bathy = elevation.copy()
        bathy[bathy >= 0] = 0
        img = (bathy - np.nanmin(bathy)) /\
            (np.nanmax(bathy) - np.nanmin(bathy))
        img = sun.shade(img, cmap=cmap_bathy,
                        vert_exag=bathy_exag, fraction=1, **sun_kw)
        img[bathy == 0, -1] = 0
        self.matshow(img, extent=georef)

        # Colorbar bathymetry
        img = self.matshow(bathy, cmap=cmap_bathy)
        img.remove()
        cax = self.figure.add_axes([1.1, 0.1, 0.03, 0.4])
        plt.colorbar(img, cax=cax, extend='min')

        # Colorbars label
        cb_label = 'Elevation (m)'
        self.figure.text(1.3, .5, cb_label, va='center',
                         rotation=90, weight='normal')


def Map3(figsize=2.5, extent=(131.5, 135, 32.5, 34.5), zlim=[0, 100],
         n_lat=6, n_lon=5):
    """ Creation of a lon, lat and depth map.

    Args
    ----

        figsize (tuple): the size of the figure (approx) in inches.
        extent (tuple): the extent in longitudes and latitudes
        zlim (tuple): the depth limits

    Return
    ------
        axes_lonlat (cartopy.GeoAxes): latitudes vs. longitude axes,
        axes_deplon (plt.Axes): depth vs. longitude axes,
        axes_latdep (plt.Axes): latitude vs. depth axes,
        cax (plt.Axes): colorbar spot
    """

    # Create
    ratio = (extent[1] - extent[0]) / (extent[3] - extent[2])
    ax = Map(figsize=(figsize * ratio, figsize), extent=extent)
    ax.ticks_decimals(n_lat=n_lat, n_lon=n_lon, decimals=2)

    # Latitude depth
    ax_lat = ax.figure.add_axes([1 + 0.1 / ratio, 0, 0.5 / ratio, 1])
    ax_lat.set_ylim(extent[2], extent[3])
    ax_lat.set_yticks(ax.get_yticks())
    ax_lat.set_yticklabels(ax.get_yticklabels())
    ax_lat.yaxis.tick_right()
    ax_lat.yaxis.set_ticks_position('both')
    ax_lat.set_xlim(zlim)
    ax_lat.set_xlabel('Depth (km)')
    ax_lat.set_xticks(np.arange(zlim[0], zlim[1] + 1, 5))

    # Longitude depth
    ax_lon = ax.figure.add_axes([0, -.6, 1, 0.5])
    ax_lon.set_xlim(extent[0], extent[1])
    ax_lon.set_xticks(ax.get_xticks())
    ax_lon.set_xticklabels(ax.get_xticklabels())
    ax_lon.set_ylim(zlim)
    ax_lat.set_xlabel('Depth (km)')
    ax_lon.invert_yaxis()
    ax_lon.set_ylabel('Depth (km)')
    ax_lon.set_yticks(np.arange(zlim[0], zlim[1] + 1, 5))

    # Colorbar
    cax = ax.figure.add_axes([1 + 0.1 / ratio, -.4, 0.5 / ratio, 0.04])

    # Lastly
    ax.set_xticklabels([''])

    return ax, ax_lon, ax_lat, cax
