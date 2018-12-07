# -*- coding: utf-8 -*-
"""
Preprocessor subroutines
-------------------------

This module contains functions to preprocess the input files to compute
a Finite Element Analysis.

"""
from __future__ import absolute_import, division, print_function
import sys
import numpy as np


def rect_grid(length, height, nx, ny, eletype=None):
    """Generate a structured mesh for a rectangle

    The coordinates of the nodes will be defined in the
    domain [-length/2, length/2] x [-height/2, height/2].

    Parameters
    ----------
        length : float
            Length of the domain.
        height : gloat
            Height of the domain.
        nx : int
            Number of elements in the x direction.
        ny : int
            Number of elements in the y direction.
        eletype : None
            It does nothing right now.

    Returns
    -------
        x : ndarray (float)
            x-coordinates for the nodes.
        y : ndarray (float)
            y-coordinates for the nodes.
        els : ndarray
            Array with element data.

    """

    y, x = np.mgrid[-height/2:height/2:(ny + 1)*1j,
                    -length/2:length/2:(nx + 1)*1j]
    els = np.zeros((nx*ny, 7), dtype=int)
    els[:, 1] = 1  # rectangular grid
    for row in range(ny):
        for col in range(nx):
            cont = row*nx + col
            els[cont, 0] = cont
            els[cont, 3:7] = [cont + row, cont + row + 1,
                              cont + row + nx + 2, cont + row + nx + 1]
    return x.flatten(), y.flatten(), els
