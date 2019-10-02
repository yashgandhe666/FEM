"""
Element subroutines
-------------------
Each UEL subroutine computes the local stiffness matrix for a given
finite element.

New elements can be added by including additional subroutines.

"""
from __future__ import absolute_import, division, print_function
import numpy as np
import sys
sys.path.insert(0, '/Users/yashgandhe/Coding/fem/solidspy')
import femutil as fem
import gaussutil as gau


def uel4nquad(coord, enu, Emod):
    """Quadrilateral element with 4 nodes

    Parameters
    ----------
    coord : ndarray
      Coordinates for the nodes of the element (4, 2).
    enu : float
      Poisson coefficient (-1, 0.5).
    Emod : float
      Young modulus (>0).

    Returns
    -------
    kl : ndarray
      Local stiffness matrix for the element (8, 8).
    """
    kl = np.zeros([8, 8])
    C = fem.umat(enu, Emod)
    XW, XP = gau.gpoints2x2()
    ngpts = 4
    for i in range(0, ngpts):
        ri = XP[i, 0]
        si = XP[i, 1]
        alf = XW[i]
        ddet, B = fem.stdm4NQ(ri, si, coord)
        kl = kl + np.dot(np.dot(B.T,C), B)*alf*ddet
    return kl

