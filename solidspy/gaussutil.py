# coding=utf-8
"""
Numeric integration routines
----------------------------
Weights and coordinates for Gauss-Legendre quadrature. The values can be found on pg. 461 of Bathe, Klaus-Jürgen.
"""
from __future__ import absolute_import, division, print_function
import numpy as np


def gpoints2x2():
    """Gauss points for a 2 by 2 grid

    Returns
    -------
    xw : ndarray
      Weights for the Gauss-Legendre quadrature.
    xp : ndarray
      Points for the Gauss-Legendre quadrature.

    """
    xw = np.zeros([4])
    xp = np.zeros([4, 2])
    xw[:] = 1.0
    xp[0, 0] = -0.577350269189626
    xp[1, 0] = 0.577350269189626
    xp[2, 0] = -0.577350269189626
    xp[3, 0] = 0.577350269189626

    xp[0, 1] = 0.577350269189626
    xp[1, 1] = 0.577350269189626
    xp[2, 1] = -0.577350269189626
    xp[3, 1] = -0.577350269189626

    return xw, xp
