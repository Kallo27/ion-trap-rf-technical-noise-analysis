#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Functions to calculate shimsets for shuttling

'''
__author__ = "Jakob Wahl"
__version__ = "0.1.0"

import numpy as np 
from electrode import System
import cvxopt, cvxopt.modeling

from electrode.electrode import PolygonPixelElectrode
from electrode.utils import (expand_tensor, norm, rotate_tensor,
    mathieu, name_to_deriv)
from electrode.pattern_constraints import (PatternRangeConstraint,
        PotentialObjective)
from electrode import colors

class ShimsetCalc(System):
    def __init__(self, *args, **kwargs):
        System.__init__(self, *args, **kwargs)

    def shims_space(self, x_coord_deriv, objectives=[], constraints=None,
            **kwargs):
        """Determine shim vectors.

        Solves the shim equations (orthogonalizes) simultaneously at
        all points for all given derivatives. 

        Parameters
        ----------
        x_coord_deriv : list of tuples (x, coord, derivative)
            `x` being array_like, shape (3,), `coord` either None or a
            array_like shape (3, 3) coordinate system rotation matrix,
            and `derivative` a string for a partial derivative.
            For possible values see `utils._derivative_names`.
        objectives : list of `pattern_constraints.Constraint`
            Additional objectives. Use this for e.g.
            `pattern_constraints.MultiPotentialObjective`.
        constraints : None or list of `pattern_constraints.Constraint`
            List of constraints. If None, the pattern electrode
            potential values are constrained between -1 and 1.
        **kwargs : any
            Passed to `self.optimize`.

        Returns
        -------
        vectors : array, shape (n, m)
            The ith row is a shim vector that achieves a unit of the ith
            constraint's effect. `n` being the number of objectives in
            the (`len(x_coord_deriv) + len(objectives)`) and m is the
            number of electrodes (`len(self)`).
        """
        obj = [PotentialObjective(x=x, derivative=deriv, value=0,
            rotation=coord) for x, coord, deriv in x_coord_deriv]
        obj += objectives
        if constraints is None:
            constraints = [PatternRangeConstraint(min=-1, max=1)]
        vectors = np.empty((len(obj), len(self)),
                np.double)
        matrices = [] 
        for i, objective in enumerate(obj):
            objective.value = 1
            solution = self.solution_lse(constraints+obj, verbose=False,
                    **kwargs)
            objective.value = 0
            matrices.append(solution)
            #  vectors[i] = p/c
        return matrices 

    def solution_lse(self, constraints, rcond=1e-9, verbose=True, **kwargs):
        """Find electrode potentials that maximize given
        constraints/objectives.
        
        Parameters
        ----------
        constraints : list of `pattern_constraints.Constraint`
            Constraints and objectives.
        rcond : float
            Cutoff for small singular values. See `scipy.linalg.pinv`
            for details.
        verbose : bool
            Passed to the solver.
        
        Returns
        -------
        potentials : array, shape (n,)
            Electrode potentials that maximize the objective and
            fulfill the constraints. `n = len(self)`.
        c : float
            Solution strength. `c` times the objective value could
            be achieved using `potentials`.
        """

        p = cvxopt.modeling.variable(len(self))
        obj = []
        ctrs = []
        for ci in constraints:
            obj.extend(ci.objective(self, p))
            ctrs.extend(ci.constraints(self, p))
        B = np.array([i[0] for i in obj])
        b = np.array([i[1] for i in obj])
        # the inhomogeneous solution
        Bp = np.linalg.pinv(B, rcond=rcond)
        g = np.dot(Bp, b)
        g2 = np.inner(g, g)
        return B, b, g

def basis_voltages_solution_space(B, b, g, rcond=1e-15):
    '''
    Calculates a basis as a addtioon to g as a soltion space. 
    The full solution for the voltages ist V = g + Span(basis)
    Input:
    ------
    B matrix
    b vector
    g

    Returns:
    -------
    basis
    '''
    Bp = np.linalg.pinv(B, rcond=rcond)
    B1 = (np.identity(len(g)) - np.dot(Bp, B))
    U, s, V = np.linalg.svd(B1)
    svd = U @ np.diag(s)
    # round svd and recalculate rank
    svd = np.round(svd, 10)
    rank = np.linalg.matrix_rank(svd)
    # drop columns that are not necessary for basis
    basis = svd.T[:rank].T
    return basis, rank

def shim_vector(basis, g, p):
    '''
    Return shims set for given prefactors b

    Input:
    ------
    basis: np.array
    Basis fo Span{B1}
    g: vector
    particular solution of B @ x = b

    '''
    if np.round(p[0], 10) == 0: 
        raise ValueError('p[0] must be !=0 but ist {}'.format(p[0]))

    # normalize g
    g_norm = g / np.sqrt(np.inner(g, g))
    vec_space = np.vstack([g_norm, basis.T])
    return vec_space.T @ p

def full_basis(basis, g):
    '''
    Return  full basis (stacks basis and g)

    Input:
    ------
    basis: np.array
    Basis fo Span{B1}
    g: vector
    particular solution of B @ x = b

    '''
    # normalize g
    g_norm = g / np.sqrt(np.inner(g, g))
    vec_space = np.vstack([g_norm, basis.T])
    return vec_space.T

def calc_shims_solution_space(x0, 
               system, 
               derivs="z xx", 
               rf_ignores=['cover'],
               constraints=None,
               direction='all', 
               **kwargs):
    '''
    Function that returns a dict with a element for each derivs.
    The element for each div contain the Matrix B, vector b and vector g.
    B is Matrix which contains to contribution of each electrode to the respective deriv.
    b is the vector that constraints the solution, 
    e.g. [0, 0, 0, 1], if derivs are 'x, y, z, xx' and you want to have only the 'xx' potential.
    g is the solution to B*g=b with the smalles euclidic two norm.

    Input:
    ------
    x0: np.array
        Coordinates where to calculate shimssets
    system: electrode.System
        System of electrodes
    derivs: str
        derivatives to calculate shimsets for
    rf_ignores: list
        list of electrodes to ignore
    constraints: np.array
        passed to shimset calc function of electrode package
    direction: str
        if 'all' or None, matrices are returned for all directions in derivs.
        if directions is specifiied, only the matrices for the specified directions are returned.
        Note: direction must be element in derivs.
    '''

    derivs_arr = derivs.split()
    s1 = ShimsetCalc([e for e in system if not (e.rf or e.name in rf_ignores)])
    matrices = {}
    #  self.shimElectrodes = s1.names
    sol = s1.shims_space([(x0, constraints, deriv) for deriv in derivs_arr], **kwargs)
    for i, deriv in enumerate(derivs_arr):
        matrices[deriv] = sol[i]
    if direction == 'all' or direction == None:
        return matrices
    else:
        return matrices[direction]

def connect(x, x1, v0, v1):
    '''
    Interpolate between to shimsets with cubic spline.
    Starting point is x=0, where voltages of v0 are applied.
    End point is x=x1, where voltages of v1 are applied.

    Input:
    ------
    x: np.array
        Coordinates where to calculate shimssets
    x1: float
        Coordinate of end point
    v0: np.array
        Voltages at starting point
    v1: np.array; it must hold that v0.shape == v1.shape
        Voltages at end point
    
    Output:
    -------
    np.array of shape (len(v0), len(x))
        interpolated voltages for each value in 'x'
    '''

    x = x/x1
    a = - 2 * (v1 - v0)
    b = - 3 * (v0 - v1)
    return np.outer(a, x**3) + np.outer(b, x**2) + np.outer(v0, np.ones(len(x)))

def closest_set(V, v0):
    '''
    Find the closest shimset to v0 in the space spanned by V.
    
    Input:
    ------
    V: np.array
        Matrix of shape (len(v0), n) whose columns span the space of shimsets
    v0: np.array
        Voltage vector. (E.g interpolated voltages)
    '''
    Vp = np.linalg.pinv(V)
    return  Vp @ v0


def choose_shims_close_to_fit(system, derivs, x, close_to):
    '''
    Calculate solution space for shims at position x. 
    Return the one that is closest to close_to.
    '''
    matrices = calc_shims(x, system, derivs, return_values='full')
    adpated_shims = {}
    for d in matrices:
        B, b, g = matrices[d]
        basis, rank = basis_voltages_solution_space(B, b, g)
        p1 = closest_set(basis, close_to[d]-g)
        shim_vector = basis @ p1 + g
        adpated_shims[d] = shim_vector
    return adpated_shims

def calc_shims(x0s, system, derivs, return_values='shims'):
    '''
    Calc shims for arbitrary number of ion positions.
    Calculate shimsets and scale to 1 V/mm or 1 V/mm2 if lengthscale is 100µm.
    '''
    if np.ndim(x0s) == 1:
        x0s = np.array([x0s])
    if np.ndim(x0s) == 2:
        x0s = np.array(x0s)
    else:
        raise ValueError('x0s must be either of shape (3, ) (one x0) or of shape (n, 3) (n x0s), but is of shape {}'.format(np.array(x0s).shape))

    ds = derivs.split(' ')

    for i, x0 in enumerate(x0s):
        B = calc_shims_solution_space(x0, system=system,  derivs=derivs)[ds[0]][0]
        if i == 0:
            Bs = np.copy(B)
        if i > 0:
            Bs = np.append(Bs, B, axis=0)
    Bp = np.linalg.pinv(Bs, rcond=1e-9,)
    
    names = []
    splitmark = '-'
    for j, x in enumerate(x0s):
        for d in ds:
            if len(x0s) > 1:
                names.append(d + splitmark + 'x{}'.format(j))
            else: 
                names.append(d)

    shimsets = {}
    bs = np.identity(len(ds) * len(x0s))
    for j, b in enumerate(bs):
        g = np.dot(Bp, b) / 10**len(names[j].split(splitmark)[0])

        if return_values == 'shims':
            shimsets[names[j]] = g
        elif return_values == 'full':
            shimsets[names[j]] = [Bs, b, g]

    return shimsets

def map_to_system(sets, s0, s_full):
    '''
    Map shimsets 'sets' that were calculated for system 's0'
    to system s_full
    Attention: this alters the voltages that are applied dcs in the systems.
    '''

    full_set = {}
    for name, sett in sets.items():
        s_full.dcs = np.zeros(len(s_full.dcs))
        s0.dcs = sett
        full_set[name] = s_full.dcs
    return full_set
