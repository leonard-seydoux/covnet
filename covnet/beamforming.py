#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Calculation of the covariance matrix from raw seismograms.

import numpy as np

from scipy.linalg import svd


def project(vector_left, matrix, vector_right):
    return abs(vector_left.dot(matrix).dot(vector_right))


class Beam():

    def __init__(self, net, frequency=None, slowness_max=None,
                 dimension=100, flip=False):

        self._dimension = dimension
        self._slowness = np.linspace(-slowness_max, slowness_max, dimension)
        self._slowness_grid = np.meshgrid(self._slowness, self._slowness)
        self._frequency = frequency
        angular_frequency = 2 * np.pi * frequency
        if flip:
            angular_frequency *= -1
        phase_x = np.outer(self._slowness_grid[0].ravel(), net.x)
        phase_y = np.outer(self._slowness_grid[1].ravel(), net.y)
        self._beamformer = np.exp(1j * angular_frequency * (phase_x + phase_y))
        self._beamformer_conj = self._beamformer**(-1)
        self._beam = np.zeros(dimension**2)

    def __add__(self, other):
        beam = self._beam + other._beam
        beam_out = self
        beam_out._beam = beam
        return beam_out

    def compute_classical(self, covariance):

        for s in range(self._dimension**2):
            # print(s, flush=True)
            self._beam[s] = project(self._beamformer_conj[s, :], covariance,
                                    self._beamformer[s, :])
        return np.reshape(self._beam, (self._dimension, self._dimension))

    def compute_music(self, covariance, rank=1, epsilon=1e-10):

        # covariance = covariance.get_data(self._frequency)
        eigenvectors, eigenvalues, _ = svd(covariance)
        eigenvalues[:rank] = 0.0
        eigenvalues[rank:] = 1.0
        eigenvalues = np.diag(eigenvalues)
        covariance = eigenvectors.dot(eigenvalues).dot(np.conj(eigenvectors.T))
        for s in range(self._dimension**2):
            self._beam[s] = project(self._beamformer_conj[s, :], covariance,
                                    self._beamformer[s, :])
            self._beam[s] = 1 / np.abs(self._beam[s] + epsilon)
        return np.reshape(self._beam, (self._dimension, self._dimension))

    def pcolormesh(self, ax, **kwargs):
        beam = np.reshape(self._beam, (self._dimension, self._dimension))
        beam = np.rot90(beam, 2)
        beam = (beam - beam.min()) / (beam.max() - beam.min())
        kwargs = {**kwargs, **dict(rasterized=True, vmin=0.0, vmax=1.0)}
        img = ax.pcolormesh(self._slowness, self._slowness, beam**2, **kwargs)
        ax.set_aspect('equal')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        grid_style = dict(lw=0.3, dashes=[6, 4], c='w')
        ax.plot(2 * [0], xlim, **grid_style)
        ax.plot(ylim, 2 * [0], **grid_style)
        ax.plot(xlim, ylim, **grid_style)
        ax.plot(xlim, [-y for y in ylim], **grid_style)
        ax.set_xticks([xlim[0], xlim[0] / 2, 0, xlim[-1] / 2, xlim[-1]])
        ax.set_yticks([ylim[0], ylim[0] / 2, 0, ylim[-1] / 2, ylim[-1]])
        return img


# class CylindricalBeam():

#     def __init__(self, antenna, frequency=None,
#                  slowness_range=np.arange(0.1, 1, 20),
#                  lon_range=np.arange(-180, 180, 10),
#                  lat_range=np.arange(-90, 90, 10)):

#     # def calculate(self, data_vector):
    #     self._slowness = np.array(slowness_range)
    #     self._frequency = frequency
