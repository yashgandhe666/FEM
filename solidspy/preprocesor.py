"""
Preprocessor subroutines
-------------------------

This module contains functions to preprocess the input files to compute
a Finite Element Analysis.

"""
from __future__ import absolute_import, division, print_function
import sys
import numpy as np


def rect_grid(l, h, nx, ny):
    """Generate a structured mesh for a rectangle

    The coordinates of the nodes will be defined in the
    domain [-l/2, l/2] x [-h/2, h/2].

    Parameters
    ----------
        l : float
            Length of the domain.
        h : float
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

    y, x = np.mgrid[-h/2:h/2:(ny + 1)*1j, -l/2:l/2:(nx + 1)*1j]
    ele = np.zeros((nx*ny, 7), dtype=int)
    ele[:, 1] = 1  # rectangular grid

    for row in range(ny):
        for col in range(nx):
            cont = row*nx + col
            ele[cont, 0] = cont
            ele[cont, 3:7] = [cont + row, cont + row + 1,
                              cont + row + nx + 2, cont + row + nx + 1]
    return x.flatten(), y.flatten(), ele
