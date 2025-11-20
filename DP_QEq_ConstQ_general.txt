#!/usr/bin/env python3
"""
QEq Calculator - Charge Equilibration with ASE Interface
=========================================================

A comprehensive implementation of the Charge Equilibration (QEq) method for
computing atomic partial charges, energies, and forces in molecular/materials systems.

Features:
---------
* Three electrostatics methods: PME, Direct Coulomb, Ewald summation
* Two solver algorithms: Projected Gradient (L-BFGS) and Direct Matrix Inversion
* Full integration with ASE (Atomic Simulation Environment)
* JAX-accelerated computations with automatic differentiation
* Support for periodic boundary conditions

Methods:
--------
PME (Particle Mesh Ewald):
    - Fast long-range electrostatics via reciprocal space
    - Best for large periodic systems
    - O(N log N) complexity

Direct Coulomb:
    - Real-space pairwise interactions with Gaussian screening
    - Suitable for medium-sized systems
    - O(N²) complexity

Ewald Summation:
    - Classical Ewald method with reciprocal space summation
    - Alternative to PME with different convergence properties
    - O(N^1.5) complexity

Solvers:
--------
Projected Gradient ('pg'):
    - L-BFGS optimization with charge conservation constraint
    - Iterative, efficient for large systems
    - Default solver

Direct Matrix ('matrix'):
    - One-step linear system solution
    - Exact solution (no iteration)
    - Best for Direct method, approximation for PME/Ewald

Units:
------
- Energy: eV
- Charges: elementary charge (e)
- Distances: Angstrom
- chi (electronegativity): eV
- eta (hardness): eV

Usage Example:
--------------
    from ase.io import read
    from DP_QEq_ConstQ_general import QEqCalculator
    
    # Load structure
    atoms = read('structure.xyz')
    
    # Create calculator
    calc = QEqCalculator(total_charge=0.0, rcut=6.0, method='pme', solver='pg')
    atoms.calc = calc
    
    # Calculate properties
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    charges = calc.get_charges()
"""

import sys
import os
import time
import json
from typing import TYPE_CHECKING, Optional, Dict, List

# Use extracted DMFF modules
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from jax.scipy.special import erfc
import jaxopt
import ase.io as IO
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii

from dmff_extracted.utils import pair_buffer_scales, regularize_pairs
from dmff_extracted.admp.recip import generate_pme_recip, Ck_1
from dmff_extracted.admp.pme import energy_pme

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Coulomb constant: e²/(4πε₀) in eV·Å
COULOMB_CONST = 1389.35455846

# Conversion factor: eV per kJ/mol
EV_PER_KJMOL = 96.4869

# Default QEq parameters for common elements (chi and eta in eV)
element_defaults = {
    'H': {'chi': 5.3200, 'eta': 7.4366},
    'C': {'chi': 5.8678, 'eta': 7.0000},
    'N': {'chi': 6.9000, 'eta': 11.7600},
    'O': {'chi': 8.5000, 'eta': 8.9989},
    'Li': {'chi': -3.0000, 'eta': 10.0241},
    'P': {'chi': 1.8000, 'eta': 7.0946},
    'F': {'chi': 9.0000, 'eta': 8.0000},
}

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def load_qeq_params(params_file: str) -> Dict:
    """Load QEq parameters (chi and eta) from JSON file
    
    Parameters
    ----------
    params_file : str
        Path to JSON file containing QEq parameters
        
    Returns
    -------
    params : dict
        Dictionary with element symbols as keys and dict of chi/eta as values
    """
    with open(params_file, 'r') as f:
        return json.load(f)


def determine_chi(box, positions: np.ndarray, symbols: list) -> np.ndarray:
    """Determine electronegativity values for atoms based on their symbols
    
    Parameters
    ----------
    symbols : list
        Chemical symbols for each atom
        
    Returns
    -------
    chi : np.ndarray
        Electronegativity values in eV
    """
    return np.array([element_defaults[symbol]['chi'] for symbol in symbols])


# ==============================================================================
# NEIGHBOR LIST CLASS
# ==============================================================================

class NeighborListJAX:
    """Efficient neighbor list implementation for JAX-compatible operations"""
    
    def __init__(self, box, rcut, cov_map, padding=True, max_shape=0):
        """Initialize neighbor list
        
        Parameters
        ----------
        box : array
            Simulation box matrix
        rcut : float
            Cutoff radius for neighbor detection
        cov_map : array
            Covalent bond mapping
        padding : bool
            Whether to pad the neighbor list
        max_shape : int
            Maximum number of pairs (0 for automatic)
        """
        self.box = box
        self.rcut = rcut
        self.capacity_multiplier = None
        self.padding = padding
        self.cov_map = cov_map
        self.max_shape = max_shape
    
    def _do_cov_map(self, pairs):
        """Apply covalent mapping to pair list"""
        nbond = self.cov_map[pairs[:, 0], pairs[:, 1]]
        return jnp.concatenate([pairs, nbond[:, None]], axis=1)
    
    def _compute_neighbor_list(self, coords, box, rcut):
        """Compute neighbor list using vectorized operations
        
        Parameters
        ----------
        coords : array
            Atomic coordinates
        box : array
            Simulation box matrix
        rcut : float
            Cutoff radius
            
        Returns
        -------
        nlist : array
            Neighbor list as (N_pairs, 2) array
        """
        natoms = coords.shape[0]
        box_inv = np.linalg.inv(box)
        
        # Create all pair indices (i < j)
        i_indices, j_indices = np.triu_indices(natoms, k=1)
        
        # Compute displacement vectors with PBC
        dpos = coords[i_indices] - coords[j_indices]
        dpos_scaled = dpos @ box_inv.T
        dpos_scaled -= np.floor(dpos_scaled + 0.5)
        dpos_pbc = dpos_scaled @ box.T
        
        # Find pairs within cutoff
        distances = np.linalg.norm(dpos_pbc, axis=1)
        mask = distances < rcut
        
        return np.column_stack([i_indices[mask], j_indices[mask]]).astype(np.int32)
    
    def allocate(self, coords, box=None):
        """Allocate neighbor list for given coordinates
        
        Parameters
        ----------
        coords : array
            Atomic coordinates
        box : array, optional
            Simulation box (uses self.box if None)
        """
        self._positions = coords
        current_box = box if box is not None else self.box
        
        nlist = self._compute_neighbor_list(coords, current_box, self.rcut)
        
        # Determine capacity
        if self.capacity_multiplier is None:
            if self.max_shape == 0:
                self.capacity_multiplier = int(nlist.shape[0] * 1.3)
            else:
                self.capacity_multiplier = self.max_shape
        
        if not self.padding:
            self._pairs = self._do_cov_map(nlist)
            return self._pairs

        if self.max_shape == 0:
            self.capacity_multiplier = max(self.capacity_multiplier, nlist.shape[0])
        else:
            self.capacity_multiplier = self.max_shape

        # Apply padding if needed
        padding_width = self.capacity_multiplier - nlist.shape[0]
        if padding_width == 0:
            self._pairs = self._do_cov_map(nlist)
        elif padding_width > 0:
            padding = np.ones((padding_width, 2), dtype=np.int32) * coords.shape[0]
            nlist = np.vstack((nlist, padding))
            self._pairs = self._do_cov_map(nlist)
        else:
            raise ValueError("Padding width < 0: increase max_shape")
            
        return self._pairs

    def update(self, positions, box=None):
        """Update neighbor list with new positions"""
        self.allocate(positions, box)
    
    @property
    def pairs(self):
        """Get neighbor pairs"""
        return self._pairs

    @property
    def scaled_pairs(self):
        """Get scaled neighbor pairs"""
        return self._pairs

    @property
    def positions(self):
        """Get cached positions"""
        return self._positions


# ==============================================================================
# DISTANCE AND PME HELPER FUNCTIONS
# ==============================================================================

@jit
def ds_pairs(positions, box, pairs):
    """Calculate distances between atom pairs with periodic boundary conditions
    
    Parameters
    ----------
    positions : array
        Atomic positions
    box : array
        Simulation box matrix
    pairs : array
        Neighbor pairs
        
    Returns
    -------
    ds : array
        Distances for each pair
    """
    pos1 = positions[pairs[:, 0].astype(int)]
    pos2 = positions[pairs[:, 1].astype(int)]
    box_inv = jnp.linalg.inv(box)
    
    # Apply minimum image convention
    dpos = pos1 - pos2
    dpos = dpos.dot(box_inv)
    dpos -= jnp.floor(dpos + 0.5)
    dr = dpos.dot(box)
    
    return jnp.linalg.norm(dr, axis=1)


def generate_get_energy(kappa, K1, K2, K3):
    """Generate PME energy function with given parameters
    
    Parameters
    ----------
    kappa : float
        Ewald splitting parameter
    K1, K2, K3 : int
        Grid dimensions for PME
        
    Returns
    -------
    get_energy : function
        PME energy function
    """
    pme_recip_fn = generate_pme_recip(
        Ck_fn=Ck_1,
        kappa=kappa / 10,
        gamma=False,
        pme_order=6,
        K1=K1,
        K2=K2,
        K3=K3,
        lmax=0,
    )
    
    def get_energy_kernel(positions, box, pairs, charges, mscales):
        """PME energy kernel
        
        Parameters
        ----------
        positions : array
            Atom positions
        box : array
            Simulation box
        pairs : array
            Neighbor pairs
        charges : array
            Atomic charges
        mscales : array
            Scaling factors
            
        Returns
        -------
        float
            PME energy in kJ/mol
        """
        atomChargesT = jnp.reshape(charges, (-1, 1))
        return energy_pme(
            positions * 10,
            box * 10,
            pairs,
            atomChargesT,
            None, None, None,
            mscales,
            None, None, None,
            pme_recip_fn,
            kappa / 10,
            K1, K2, K3,
            0,
            False,
        )
    
    return lambda positions, box, pairs, charges, mscales: \
        get_energy_kernel(positions, box, pairs, charges, mscales)


# ==============================================================================
# QEQ ENERGY FUNCTIONS
# ==============================================================================

@jit 
def get_Energy_Qeq_PME(charges, positions, box, pairs, alpha, chi, hardness):
    """Calculate QEq energy using PME method
    
    Parameters
    ----------
    charges : array
        Atomic charges (e)
    positions : array
        Atomic positions (Angstrom)
    box : array
        Simulation box matrix (Angstrom)
    pairs : array
        Neighbor pair indices
    alpha : array
        Gaussian width parameters (Angstrom)
    chi : array
        Electronegativity (eV)
    hardness : array
        Chemical hardness (eV)
        
    Returns
    -------
    energy : float
        Total QEq energy (eV)
    """
    @jit 
    def get_Energy_PME():
        """PME electrostatic energy (returns kJ/mol)"""
        pme = generate_get_energy(4.3804348, 45, 123, 22)
        return pme(positions/10, box/10, pairs, charges, 
                  mscales=jnp.array([1., 1., 1., 1., 1., 1.]))
    
    @jit 
    def get_Energy_Correction():
        """Gaussian screening correction"""
        ds = ds_pairs(positions, box, pairs)
        buffer_scales = pair_buffer_scales(pairs)
        alpha_ij = jnp.sqrt(alpha[pairs[:,0]]**2 + alpha[pairs[:,1]]**2)
        screening = erfc(ds / (jnp.sqrt(2) * alpha_ij))
        
        e_corr_pair = (charges[pairs[:,0]] * charges[pairs[:,1]] * 
                      screening * COULOMB_CONST / ds * buffer_scales)
        e_corr_self = charges * charges * COULOMB_CONST / (2 * jnp.sqrt(jnp.pi) * alpha)
        return -jnp.sum(e_corr_pair) + jnp.sum(e_corr_self)
    
    @jit
    def get_Energy_Onsite():
        """On-site (Thomas-Fermi) energy"""
        E_tf = (chi * charges + 0.5 * hardness * charges * charges) * EV_PER_KJMOL
        return jnp.sum(E_tf)
    
    @jit 
    def get_dipole_correction():
        """Dipole correction for non-periodic systems"""
        V = jnp.linalg.det(box)
        pre_corr = 2 * jnp.pi / V * COULOMB_CONST
        Mz = jnp.sum(charges * positions[:, 1])
        return pre_corr * Mz**2

    # All energy terms converted to eV
    return (get_Energy_PME() + get_Energy_Correction() + 
            get_Energy_Onsite() + get_dipole_correction()) / EV_PER_KJMOL

@jit 
def get_Energy_Qeq_Direct(charges, positions, box, pairs, alpha, chi, hardness):
    """Calculate QEq energy using direct Coulomb method
    
    Parameters
    ----------
    charges : array
        Atomic charges (e)
    positions : array
        Atomic positions (Angstrom)
    box : array
        Simulation box matrix (Angstrom)
    pairs : array
        Neighbor pair indices
    alpha : array
        Gaussian width parameters (Angstrom)
    chi : array
        Electronegativity (eV)
    hardness : array
        Chemical hardness (eV)
        
    Returns
    -------
    energy : float
        Total QEq energy (eV)
    """
    @jit
    def get_Energy_Coulomb():
        """Coulomb energy with Gaussian screening"""
        ds = ds_pairs(positions, box, pairs)
        buffer_scales = pair_buffer_scales(pairs)
        alpha_ij = jnp.sqrt(alpha[pairs[:,0]]**2 + alpha[pairs[:,1]]**2)
        screening = erfc(ds / (jnp.sqrt(2) * alpha_ij))
        
        e_coulomb_pair = (charges[pairs[:,0]] * charges[pairs[:,1]] * 
                         screening * COULOMB_CONST / ds * buffer_scales)
        e_coulomb_self = charges * charges * COULOMB_CONST / (2 * jnp.sqrt(jnp.pi) * alpha)
        return jnp.sum(e_coulomb_pair) + jnp.sum(e_coulomb_self)
    
    @jit
    def get_Energy_Onsite():
        """On-site (Thomas-Fermi) energy"""
        E_tf = (chi * charges + 0.5 * hardness * charges * charges) * EV_PER_KJMOL
        return jnp.sum(E_tf)

    return (get_Energy_Coulomb() + get_Energy_Onsite()) / EV_PER_KJMOL

@jit
def get_Energy_Qeq_Ewald(charges, positions, box, pairs, alpha, chi, hardness, ewald_alpha=0.3):
    """Calculate QEq energy using Ewald summation method
    
    Parameters
    ----------
    charges : array
        Atomic charges (e)
    positions : array
        Atomic positions (Angstrom)
    box : array
        Simulation box matrix (Angstrom)
    pairs : array
        Neighbor pair indices
    alpha : array
        Gaussian width parameters (Angstrom)
    chi : array
        Electronegativity (eV)
    hardness : array
        Chemical hardness (eV)
    ewald_alpha : float
        Ewald splitting parameter (1/Angstrom), default 0.3
        
    Returns
    -------
    energy : float
        Total QEq energy (eV)
    """
    @jit
    def get_Energy_RealSpace():
        """Real-space part of Ewald summation"""
        ds = ds_pairs(positions, box, pairs)
        buffer_scales = pair_buffer_scales(pairs)
        
        # Combine Gaussian and Ewald screening
        alpha_ij = jnp.sqrt(alpha[pairs[:,0]]**2 + alpha[pairs[:,1]]**2)
        gaussian_screening = erfc(ds / (jnp.sqrt(2) * alpha_ij))
        ewald_screening = erfc(ewald_alpha * ds)
        
        e_real_pair = (charges[pairs[:,0]] * charges[pairs[:,1]] * 
                      gaussian_screening * ewald_screening * COULOMB_CONST / ds * buffer_scales)
        e_self_gaussian = charges * charges * COULOMB_CONST / (2 * jnp.sqrt(jnp.pi) * alpha)
        return jnp.sum(e_real_pair) + jnp.sum(e_self_gaussian)
    
    @jit
    def get_Energy_ReciprocalSpace():
        """Reciprocal-space part of Ewald summation (dipole approximation)"""
        V = jnp.linalg.det(box)
        pre_factor = 2 * jnp.pi / V * COULOMB_CONST
        dipole_moment = jnp.sum(charges[:, None] * positions, axis=0)
        return pre_factor * jnp.sum(dipole_moment**2)
    
    @jit
    def get_Energy_SelfCorrection():
        """Self-energy correction for Ewald summation"""
        return -ewald_alpha / jnp.sqrt(jnp.pi) * jnp.sum(charges**2) * COULOMB_CONST
    
    @jit
    def get_Energy_Onsite():
        """On-site (Thomas-Fermi) energy"""
        E_tf = (chi * charges + 0.5 * hardness * charges * charges) * EV_PER_KJMOL
        return jnp.sum(E_tf)
    
    return (get_Energy_RealSpace() + get_Energy_ReciprocalSpace() + 
            get_Energy_SelfCorrection() + get_Energy_Onsite()) / EV_PER_KJMOL


# ==============================================================================
# CHARGE OPTIMIZATION - PROJECTED GRADIENT METHOD
# ==============================================================================

def fn_value_and_proj_grad(func, constraint_matrix, has_aux=False):
    """Create projected gradient function for constrained optimization
    
    Parameters
    ----------
    func : callable
        Energy function to minimize
    constraint_matrix : array
        Constraint matrix (typically ones for charge conservation)
    has_aux : bool
        Whether function returns auxiliary data
        
    Returns
    -------
    value_and_proj_grad : callable
        Function returning (value, projected_gradient)
    """
    def value_and_proj_grad(*arg, **kwargs):
        value, grad = jax.value_and_grad(func, has_aux=has_aux)(*arg, **kwargs)
        # Project gradient onto constraint surface
        a = jnp.matmul(constraint_matrix, grad.reshape(-1, 1))
        b = jnp.sum(constraint_matrix * constraint_matrix, axis=1, keepdims=True)
        delta_grad = jnp.matmul((a / b).T, constraint_matrix)
        proj_grad = grad - delta_grad.reshape(-1)
        return value, proj_grad
    return value_and_proj_grad


# ==============================================================================
# MATRIX SOLVER - COULOMB MATRIX BUILDERS
# ==============================================================================

def build_coulomb_matrix_direct(positions, box, pairs, alpha):
    """Build Coulomb interaction matrix for Direct method
    
    Parameters
    ----------
    positions : array
        Atomic positions
    box : array
        Simulation box matrix
    pairs : array
        Neighbor pairs
    alpha : array
        Gaussian width parameters
        
    Returns
    -------
    J : array
        Coulomb interaction matrix (eV)
    """
    natoms = positions.shape[0]
    J = jnp.zeros((natoms, natoms))
    ds = ds_pairs(positions, box, pairs)
    
    # Direct Coulomb with Gaussian screening
    alpha_ij = jnp.sqrt(alpha[pairs[:,0]]**2 + alpha[pairs[:,1]]**2)
    screening = erfc(ds / (jnp.sqrt(2) * alpha_ij))
    interaction = screening * COULOMB_CONST / ds
    
    # Fill symmetric matrix
    i_indices, j_indices = pairs[:, 0], pairs[:, 1]
    J = J.at[i_indices, j_indices].set(interaction)
    J = J.at[j_indices, i_indices].set(interaction)
    
    return J


def build_coulomb_matrix_pme(positions, box, pairs, alpha):
    """Build Coulomb matrix for PME method
    
    WARNING: Approximates PME with real-space screened Coulomb only.
    Full PME would require grid operations unsuitable for matrix form.
    
    Returns
    -------
    J : array
        Approximated Coulomb matrix (eV)
    """
    return build_coulomb_matrix_direct(positions, box, pairs, alpha)

def build_coulomb_matrix_ewald(positions, box, pairs, alpha, ewald_alpha=0.3):
    """Build Coulomb matrix for Ewald method
    
    WARNING: Incomplete representation - includes screening but may not
    fully capture Ewald convergence acceleration.
    
    Parameters
    ----------
    ewald_alpha : float
        Ewald splitting parameter (1/Angstrom)
        
    Returns
    -------
    J : array
        Coulomb matrix with Ewald screening (eV)
    """
    natoms = positions.shape[0]
    J = jnp.zeros((natoms, natoms))
    ds = ds_pairs(positions, box, pairs)
    
    # Combined Gaussian and Ewald screening
    alpha_ij = jnp.sqrt(alpha[pairs[:,0]]**2 + alpha[pairs[:,1]]**2)
    gaussian_screening = erfc(ds / (jnp.sqrt(2) * alpha_ij))
    ewald_screening = erfc(ewald_alpha * ds)
    interaction = gaussian_screening * ewald_screening * COULOMB_CONST / ds
    
    # Fill symmetric matrix
    i_indices, j_indices = pairs[:, 0], pairs[:, 1]
    J = J.at[i_indices, j_indices].set(interaction)
    J = J.at[j_indices, i_indices].set(interaction)
    
    return J


# ==============================================================================
# MATRIX SOLVER - DIRECT INVERSION SOLVERS
# ==============================================================================

def solve_q_matrix_direct(positions, box, pairs, alpha, chi, hardness, total_charge=0.0):
    """Solve for equilibrium charges using direct matrix inversion
    
    Solves the augmented linear system:
        [J  1] [q]   [-χ]
        [1ᵀ 0] [λ] = [Q_tot]
    
    where J_ii = η_i + self-interaction and J_ij = Coulomb interaction
    
    Parameters
    ----------
    total_charge : float
        Total system charge constraint
        
    Returns
    -------
    charges : array
        Optimized atomic charges (e)
    """
    natoms = positions.shape[0]
    J_coulomb = build_coulomb_matrix_direct(positions, box, pairs, alpha)
    
    # Add hardness to diagonal: J_ii = η_i + self-interaction
    self_interaction = COULOMB_CONST / (jnp.sqrt(jnp.pi) * alpha)
    J = J_coulomb + jnp.diag(hardness * EV_PER_KJMOL + self_interaction)
    
    # Build augmented system with charge constraint
    augmented = jnp.zeros((natoms + 1, natoms + 1))
    augmented = augmented.at[:natoms, :natoms].set(J)
    augmented = augmented.at[:natoms, natoms].set(1.0)
    augmented = augmented.at[natoms, :natoms].set(1.0)
    
    # Right-hand side: [-χ, Q_tot]
    rhs = jnp.zeros(natoms + 1)
    rhs = rhs.at[:natoms].set(-chi * EV_PER_KJMOL)
    rhs = rhs.at[natoms].set(total_charge)
    
    # Solve and extract charges
    solution = jnp.linalg.solve(augmented, rhs)
    return solution[:natoms]


def solve_q_matrix_pme(positions, box, pairs, alpha, chi, hardness, total_charge=0.0):
    """Solve for equilibrium charges using direct matrix inversion (PME method)
    
    WARNING: The matrix-based solver approximates PME with real-space screened Coulomb only.
    It does NOT include the reciprocal space (k-space) contributions from PME.
    For accurate PME results, use solver='pg' instead.
    This approximation is only provided for comparison purposes.
    """
    # PME matrix approximation uses same as direct
    return solve_q_matrix_direct(positions, box, pairs, alpha, chi, hardness, total_charge)

def solve_q_matrix_ewald(positions, box, pairs, alpha, chi, hardness, total_charge=0.0):
    """Solve for equilibrium charges using direct matrix inversion (Ewald method)
    
    WARNING: The matrix-based solver uses an incomplete representation of Ewald summation.
    It includes Ewald screening but may not fully capture the convergence acceleration.
    For accurate Ewald results, use solver='pg' instead.
    This approximation is only provided for comparison purposes.
    """
    natoms = positions.shape[0]
    J_coulomb = build_coulomb_matrix_ewald(positions, box, pairs, alpha)
    
    # Add hardness to diagonal: J_ii = η_i + self-interaction
    # Self-interaction from Gaussian: q²/(2√π α)
    self_interaction = 1389.35455846 / (jnp.sqrt(jnp.pi) * alpha)
    J = J_coulomb + jnp.diag(hardness * 96.4869 + self_interaction)
    
    # Build augmented matrix for charge constraint
    # [J  1] 
    # [1ᵀ 0]
    augmented = jnp.zeros((natoms + 1, natoms + 1))
    augmented = augmented.at[:natoms, :natoms].set(J)
    augmented = augmented.at[:natoms, natoms].set(1.0)
    augmented = augmented.at[natoms, :natoms].set(1.0)
    
    # Build right-hand side: [-χ, Q_tot]
    rhs = jnp.zeros(natoms + 1)
    rhs = rhs.at[:natoms].set(-chi * 96.4869)
    rhs = rhs.at[natoms].set(total_charge)
    
    # Solve linear system
    solution = jnp.linalg.solve(augmented, rhs)
    
    # Extract charges (first N elements)
    charges = solution[:natoms]
    
    return charges

def solve_q_pg(charges, positions, box, pairs, alpha, chi, hardness, total_charge=0.0, method='pme'):
    """Solve for equilibrium charges using projected gradient method (L-BFGS)
    
    Parameters
    ----------
    charges : array
        Initial atomic charges (e)
    positions : array
        Atomic positions (Angstrom)
    box : array
        Simulation box matrix (Angstrom)
    pairs : array
        Neighbor pair indices
    alpha : array
        Gaussian width parameters (Angstrom)
    chi : array
        Electronegativity (eV)
    hardness : array
        Chemical hardness (eV)
    total_charge : float, optional
        Total system charge (default: 0.0)
    method : str, optional
        Electrostatics method: 'pme' (default), 'direct', or 'ewald'
        
    Returns
    -------
    charges : array
        Optimized atomic charges (e)
    """
    # Select energy function based on method
    if method.lower() == 'direct':
        energy_func = get_Energy_Qeq_Direct
    elif method.lower() == 'ewald':
        energy_func = get_Energy_Qeq_Ewald
    else:  # default to PME
        energy_func = get_Energy_Qeq_PME
    
    # Adjust initial charges to satisfy total charge constraint
    charge_deficit = total_charge - jnp.sum(charges)
    charges = charges + charge_deficit / len(charges)
    
    func = fn_value_and_proj_grad(energy_func, jnp.ones_like(charges).reshape(1, -1))
    solver = jaxopt.LBFGS(
        fun=func,
        value_and_grad=True,
        tol=1e-2,
        )
    res = solver.run(charges, positions, box, pairs, alpha, chi, hardness)
    x_opt = res.params
    
    # Re-apply total charge constraint after optimization
    charge_deficit = total_charge - jnp.sum(x_opt)
    x_opt = x_opt + charge_deficit / len(x_opt)
    
    return x_opt

def get_force(charges, positions, box, pairs, alpha, chi, hardness, method='pme'):
    """Calculate energy and forces
    
    Parameters
    ----------
    method : str
        Electrostatics method: 'pme' (default), 'direct', or 'ewald'
    """
    # Select energy function based on method
    if method.lower() == 'direct':
        energy_func = get_Energy_Qeq_Direct
    elif method.lower() == 'ewald':
        energy_func = get_Energy_Qeq_Ewald
    else:  # default to PME
        energy_func = get_Energy_Qeq_PME
    
    energy_force_fn = jit(value_and_grad(energy_func, argnums=(1)))
    energy, force = energy_force_fn(charges, positions, box, pairs, alpha, chi, hardness)
    return energy, -force

def get_qeq_energy_and_force_pg(charges, positions, box, pairs, alpha, chi, hardness, total_charge=0.0, method='pme', solver='pg'):
    """Calculate QEq energy, forces, and charges
    
    Parameters
    ----------
    charges : array
        Initial atomic charges (e)
    positions : array
        Atomic positions (Angstrom)
    box : array
        Simulation box matrix (Angstrom)
    pairs : array
        Neighbor pair indices
    alpha : array
        Gaussian width parameters (Angstrom)
    chi : array
        Electronegativity (eV)
    hardness : array
        Chemical hardness (eV)
    total_charge : float, optional
        Total system charge (default: 0.0)
    method : str, optional
        Electrostatics method: 'pme' (default), 'direct', or 'ewald'
    solver : str, optional
        Solver type: 'pg' for projected gradient (L-BFGS, default) or 'matrix' for direct matrix inversion
        
    Returns
    -------
    energy : float
        Total QEq energy (eV)
    force : array
        Forces on atoms (eV/Angstrom)
    q : array
        Optimized atomic charges (e)
    """
    method_lower = method.lower()
    solver_lower = solver.lower()
    
    # Select solver based on method and solver type
    if solver_lower == 'matrix':
        if method_lower == 'direct':
            q = solve_q_matrix_direct(positions, box, pairs, alpha, chi, hardness, total_charge)
        elif method_lower == 'ewald':
            q = solve_q_matrix_ewald(positions, box, pairs, alpha, chi, hardness, total_charge)
        else:  # pme
            q = solve_q_matrix_pme(positions, box, pairs, alpha, chi, hardness, total_charge)
    else:  # 'pg' solver
        q = solve_q_pg(charges, positions, box, pairs, alpha, chi, hardness, total_charge, method)
    
    energy, force = get_force(q, positions, box, pairs, alpha, chi, hardness, method)
    return energy, force, q


# ==============================================================================
# NEIGHBOR LIST INITIALIZATION
# ==============================================================================

def get_neighbor_list(box, rc, positions, natoms, padding=True, max_shape=0):
    """Create and populate neighbor list for given system
    
    Parameters
    ----------
    box : array
        Simulation box matrix (Angstrom)
    rc : float
        Cutoff radius (Angstrom)
    positions : array
        Atomic positions (Angstrom)
    natoms : int
        Number of atoms
    padding : bool, optional
        Whether to use padding (default: True)
    max_shape : int, optional
        Maximum neighbor list size (default: 0)
        
    Returns
    -------
    pairs : array
        Regularized neighbor pairs
    """
    nbl = NeighborListJAX(box, rc, jnp.zeros([natoms, natoms], dtype=jnp.int32), padding=padding, max_shape=max_shape)
    nbl.allocate(positions, box)
    pairs = nbl.pairs
    pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
    return pairs


# ==============================================================================
# UTILITY FUNCTIONS FOR UNIT CELL CONVERSION
# ==============================================================================

def cell_to_box(a, b, c, alpha, beta, gamma):
    """Convert cell parameters to box matrix
    
    Parameters
    ----------
    a, b, c : float
        Cell lengths (Angstrom)
    alpha, beta, gamma : float
        Cell angles (degrees)
        
    Returns
    -------
    box : array
        3x3 box matrix (Angstrom)
    """
    alpha = alpha / 180 * np.pi
    beta  = beta  / 180 * np.pi
    gamma = gamma / 180 * np.pi

    box = np.zeros((3,3), dtype=np.double) 
    box[0, 0] = a
    box[0, 1] = 0
    box[0, 2] = 0
    box[1, 0] = b * np.cos(gamma)
    box[1, 1] = b * np.sin(gamma)
    box[1, 2] = 0
    box[2, 0] = c * np.cos(beta)
    box[2, 1] = c * (np.cos(alpha)-np.cos(beta)*np.cos(gamma)) / np.sin(gamma)
    box[2, 2] = c * np.sqrt(1 - np.cos(beta)**2 - ((np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma))**2)
    return box


# ==============================================================================
# ASE CALCULATOR INTERFACE
# ==============================================================================

if TYPE_CHECKING:
    from ase import Atoms

__all__ = ["QEqCalculator"]


class QEqCalculator(Calculator):
    """
    ASE Calculator for charge equilibration (QEq) method.
    
    Calculates atomic charges, energy, and forces using the charge equilibration method
    with PME, direct Coulomb, or Ewald summation electrostatics and Gaussian charge distributions.
    
    Parameters
    ----------
    total_charge : float, optional
        Total charge of the system (default: 0.0 for neutral systems)
    rcut : float, optional
        Cutoff distance for neighbor list in Angstroms (default: 6.0)
    max_pairs : int, optional
        Maximum number of neighbor pairs for padding (default: 200000)
    initial_charges : np.ndarray, optional
        Initial guess for charges. If None, random charges are used.
    method : str, optional
        Electrostatics method: 'pme' for Particle Mesh Ewald (default), 
        'direct' for direct Coulomb, or 'ewald' for Ewald summation
    solver : str, optional
        Solver type: 'pg' for projected gradient/L-BFGS (default, iterative),
        or 'matrix' for direct matrix inversion (exact, single-step)
    params_file : str, optional
        Path to JSON file containing QEq parameters (chi and eta values). 
        If None, uses hardcoded default values (default: 'qeq_params.json')
    kappa : float, optional
        Ewald parameter (default: 4.3804348, only used for PME method)
    K1, K2, K3 : int, optional
        PME grid dimensions (default: 45, 123, 22, only used for PME method)
    ewald_alpha : float, optional
        Ewald splitting parameter in 1/Angstrom (default: 0.3, only used for Ewald method)
    **kwargs
        Additional arguments passed to Calculator
    """
    
    implemented_properties = ['energy', 'forces', 'charges']
    
    def __init__(self, 
                 total_charge: float = 0.0,
                 rcut: float = 6.0,
                 max_pairs: int = 200000,
                 initial_charges: Optional[np.ndarray] = None,
                 method: str = 'pme',
                 solver: str = 'pg',
                 params_file: Optional[str] = 'qeq_params.json',
                 kappa: float = 4.3804348,
                 K1: int = 45,
                 K2: int = 123,
                 K3: int = 22,
                 ewald_alpha: float = 0.3,
                 **kwargs):
        
        Calculator.__init__(self, **kwargs)
        
        self.total_charge = total_charge
        self.rcut = rcut
        self.max_pairs = max_pairs
        self.initial_charges = initial_charges
        self.method = method.lower()
        self.solver = solver.lower()
        self.params_file = params_file
        self.kappa = kappa
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.ewald_alpha = ewald_alpha
        self.last_charges = None
        
        # Load QEq parameters from file if provided
        if self.params_file is not None and os.path.exists(self.params_file):
            self.qeq_params = load_qeq_params(self.params_file)
        else:
            self.qeq_params = None
    
    def calculate(self, 
                  atoms: Optional[Atoms] = None,
                  properties: List[str] = ['energy', 'forces', 'charges'],
                  system_changes: List[str] = all_changes):
        """
        Calculate properties for the given atoms object.
        
        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object
        properties : list of str
            List of properties to calculate
        system_changes : list of str
            List of changes since last calculation
        """
        
        if atoms is not None:
            self.atoms = atoms.copy()
        
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Get atomic positions and cell
        positions = jnp.array(self.atoms.get_positions())
        cell = self.atoms.cell.cellpar()
        box = jnp.array(cell_to_box(cell[0], cell[1], cell[2], cell[3], cell[4], cell[5]))
        symbols = self.atoms.get_chemical_symbols()
        natoms = len(self.atoms)
        
        # Initialize charges
        if self.initial_charges is not None:
            charges = jnp.array(self.initial_charges)
        elif self.last_charges is not None and len(self.last_charges) == natoms:
            charges = self.last_charges
        else:
            charges = jnp.array(np.random.random(natoms))
        
        # Get element-specific parameters
        if self.qeq_params is not None:
            # Load from JSON file
            chi = jnp.array([self.qeq_params[tmp]['chi'] for tmp in symbols])
            hardness = jnp.array([self.qeq_params[tmp]['eta'] for tmp in symbols])
        else:
            # Use element_defaults dictionary
            chi = jnp.array([element_defaults[tmp]['chi'] for tmp in symbols])
            hardness = jnp.array([element_defaults[tmp]['eta'] for tmp in symbols])
        # Get covalent radii from ASE data (in Angstroms)
        alpha = jnp.array([covalent_radii[atomic_numbers[tmp]] for tmp in symbols])
        
        # Build neighbor list
        pairs = get_neighbor_list(box, self.rcut, positions, natoms, 
                                  padding=True, max_shape=self.max_pairs)
        
        # Solve for charges, energy, and forces
        energy, force, charges_opt = get_qeq_energy_and_force_pg(
            charges, positions, box, pairs, alpha, chi, hardness, self.total_charge, self.method, self.solver
        )
        
        # Store results
        self.results['energy'] = float(energy)
        self.results['forces'] = np.array(force).reshape(-1, 3)
        self.results['charges'] = np.array(charges_opt)
        
        # Save charges for next iteration
        self.last_charges = charges_opt
    
    def get_charges(self):
        """
        Get the calculated atomic charges.
        
        Returns
        -------
        charges : np.ndarray
            Atomic charges
        """
        if 'charges' not in self.results:
            raise RuntimeError("Charges have not been calculated yet. Run calculate() first.")
        return self.results['charges']


if __name__ == "__main__":
    # Example usage - comparing PME, Direct, and Ewald methods with different solvers
    print("=" * 70)
    print("QEq Calculator - Comparing Methods and Solvers")
    print("=" * 70)
    
    # Load structure
    # atoms = IO.read('water.xyz')
    atoms = IO.read('POSCAR')
    print(f"\nLoaded structure: {len(atoms)} atoms")
    print(f"Chemical formula: {atoms.get_chemical_formula()}")
    
    # Run calculations for all three methods and both solvers
    methods = ['pme', 'direct', 'ewald']
    solvers = ['pg', 'matrix']
    results = {}
    
    for method in methods:
        results[method] = {}
        for solver in solvers:
            key = f"{method}_{solver}"
            print(f"\n{'=' * 70}")
            print(f"Running {method.upper()} method with {solver.upper()} solver")
            print('=' * 70)
            
            # Create calculator with total charge = 0 (neutral system)
            calc = QEqCalculator(total_charge=0.0, rcut=6.0, max_pairs=200000, 
                                method=method, solver=solver)
            
            # Attach calculator to atoms
            atoms.calc = calc
            
            # Calculate properties
            print("\nCalculating charges, energy, and forces...")
            time1 = time.process_time()
            
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            charges = calc.get_charges()
            
            time2 = time.process_time()
            calc_time = time2 - time1
            
            # Store results
            results[method][solver] = {
                'energy': energy,
                'forces': forces,
                'charges': charges,
                'time': calc_time
            }
            
            # Print results
            print(f"\nResults ({method.upper()} method, {solver.upper()} solver):")
            print(f"  Energy: {energy:.6f} eV")
            print(f"  Total charge: {np.sum(charges):.6f} (Target: {calc.total_charge:.6f})")
            print(f"  Calculation time: {calc_time:.3f} s")
            print(f"\nCharge statistics:")
            print(f"  Min charge: {np.min(charges):.6f}")
            print(f"  Max charge: {np.max(charges):.6f}")
            print(f"  Mean charge: {np.mean(charges):.6f}")
            print(f"\nForce statistics:")
            print(f"  Max force: {np.max(np.linalg.norm(forces, axis=1)):.6f} eV/Å")
            print(f"  Mean force: {np.mean(np.linalg.norm(forces, axis=1)):.6f} eV/Å")
            
            # Save results
            np.savetxt(f'energy_qeq_{method}_{solver}.txt', [energy])
            np.savetxt(f'charges_qeq_{method}_{solver}.txt', charges)
            np.savetxt(f'forces_qeq_{method}_{solver}.txt', forces)
            
            print(f"\nResults saved to:")
            print(f"  - energy_qeq_{method}_{solver}.txt")
            print(f"  - charges_qeq_{method}_{solver}.txt")
            print(f"  - forces_qeq_{method}_{solver}.txt")
    
    # Summary comparison
    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print('=' * 70)
    print(f"\n{'Method-Solver':<20} {'Energy (eV)':<15} {'Time (s)':<12} {'Max Force (eV/Å)':<15}")
    print('-' * 70)
    for method in methods:
        for solver in solvers:
            r = results[method][solver]
            max_force = np.max(np.linalg.norm(r['forces'], axis=1))
            label = f"{method.upper()}-{solver.upper()}"
            print(f"{label:<20} {r['energy']:<15.6f} {r['time']:<12.3f} {max_force:<15.6f}")
    
    # Compare solvers for each method
    print(f"\n{'=' * 70}")
    print("SOLVER COMPARISON (for each method)")
    print('=' * 70)
    for method in methods:
        print(f"\n{method.upper()} method:")
        pg_result = results[method]['pg']
        mat_result = results[method]['matrix']
        
        energy_diff = abs(pg_result['energy'] - mat_result['energy'])
        charge_diff = np.max(np.abs(pg_result['charges'] - mat_result['charges']))
        force_diff = np.max(np.linalg.norm(pg_result['forces'] - mat_result['forces'], axis=1))
        
        print(f"  Energy difference (PG - Matrix):  {energy_diff:.6e} eV")
        print(f"  Max charge difference:             {charge_diff:.6e}")
        print(f"  Max force difference:              {force_diff:.6e} eV/Å")
        print(f"  Speedup (PG time / Matrix time):   {pg_result['time'] / mat_result['time']:.2f}x")
    
    print("\n" + "=" * 70)
