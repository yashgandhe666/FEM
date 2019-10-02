"""
FEM routines
------------

Functions to compute kinematics variables for the Finite
Element Analysis.

The element included is:
    1. 4 node bilinear quadrilateral.

The notation used is similar to the one used by Bathe [1]_.


References
----------
.. [1] Bathe, Klaus-Jürgen. Finite element procedures. Prentice Hall,
   Pearson Education, 2006.

"""
from __future__ import absolute_import, division, print_function
import sys
sys.path.insert(0, '/Users/yashgandhe/Coding/fem/solidspy')
import gaussutil as gau
import numpy as np


def eletype(iet):
    """Assigns number to degrees of freedom

    According to iet assigns number of degrees of freedom, number of
    nodes and minimum required number of integration points.

    Parameters
    ----------
    iet :  int
      Type of element. These are:
        1. 4 node bilinear quadrilateral.

    Returns
    -------
    ndof : int
      Number of degrees of freedom for the selected element.
    nnodes : int
      Number of nodes for the selected element.
    ngpts : int
      Number of Gauss points for the selected element.

    """
    if iet == 1:
        ndof = 8
        nnodes = 4
        ngpts = 4

    return ndof, nnodes, ngpts

def jacoper(dhdx, coord):
    """
    Compute the Jacobian of the transformation evaluated at
    the Gauss point

    Parameters
    ----------
    dhdx : ndarray
      Derivatives of the interpolation function with respect to the
      natural coordinates.
    coord : ndarray
      Coordinates of the nodes of the element (nn, 2).

    Returns
    -------
    jaco_inv : ndarray (2, 2)
      Jacobian of the transformation evaluated at `(r, s)`.

    """
    jaco = dhdx.dot(coord)
    det = np.linalg.det(jaco)
    if np.isclose(np.abs(det), 0.0):
        msg = "Jacobian close to zero. Check the shape of your elements!"
        raise ValueError(msg)
    jaco_inv = np.linalg.inv(jaco)
    if det < 0.0:
        msg = "Jacobian is negative. Check your elements orientation!"
        raise ValueError(msg)
    return det, jaco_inv

#%% Material routines
def umat(nu, E):
    """2D Elasticity constitutive matrix in plane stress

    For plane strain use effective properties.

    Parameters
    ----------
    nu : float
      Poisson coefficient (-1, 0.5).
    E : float
      Young modulus (>0).

    Returns
    -------
    C : ndarray
      Constitutive tensor in Voigt notation.

    """
    C = np.zeros((3, 3))
    enu = E/(1 - nu**2)
    mnu = (1 - nu)/2
    C[0, 0] = enu
    C[0, 1] = nu*enu
    C[1, 0] = C[0, 1]
    C[1, 1] = enu
    C[2, 2] = enu*mnu

    return C


def stdm4NQ(r, s, coord):
    """Strain-displacement interpolator B for a 4-noded quad element

    Parameters
    ----------
    r : float
      r component in the natural space.
    s : float
      s component in the natural space.
    coord : ndarray
      Coordinates of the nodes of the element (4, 2).

    Returns
    -------
    ddet : float
      Determinant evaluated at `(r, s)`.
    B : ndarray
      Strain-displacement interpolator evaluated at `(r, s)`.

    """
    nn = 4
    B = np.zeros((3, 2*nn))
    dhdx = 0.25*np.array([
            [s - 1, -s + 1, s + 1, -s - 1],
            [r - 1, -r - 1, r + 1, -r + 1]])
    det, jaco_inv = jacoper(dhdx, coord)
    dhdx = np.dot(jaco_inv, dhdx)
    B[0, ::2] = dhdx[0, :]
    B[1, 1::2] = dhdx[1, :]
    B[2, ::2] = dhdx[1, :]
    B[2, 1::2] = dhdx[0, :]
    return det, B