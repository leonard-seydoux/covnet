#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from numpy.lib.recfunctions import append_fields
from datetime import datetime
from matplotlib.dates import date2num, datestr2num


def read(file_path, **kwargs):
    """ Read seismic array metadata."""

    # Default options
    kwargs.setdefault('dtype', None)
    kwargs.setdefault('encoding', None)
    kwargs.setdefault('names', True)

    # Get metadata
    meta = np.genfromtxt(file_path, **kwargs)
    meta = meta.view(np.recarray)

    # Append cartesian coordinates in meta
    times = list()
    for d in meta['date']:
        time = datetime.strptime(d, '%d.%m.%Y %H:%M:%S')
        times.append(date2num(time))
    meta = append_fields(meta, 'datetime', times)

    return meta.view(np.recarray).view(Events)


class Events(np.recarray):

    def select(self, start=None, end=None):

        start = datestr2num(start) if start is not None else 0
        end = datestr2num(end) if end is not None else np.inf
        selection = (self['datetime'] > start) & (self['datetime'] < end)

        return self[selection]
