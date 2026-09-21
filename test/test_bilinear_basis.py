#!/usr/bin/env python3
"""Tests for the symmetry-closed bilinear basis.

    python3 test/test_bilinear_basis.py

Plain asserts, no pytest and no TRIQS. The Julia suite (test/runtests.jl) covers
the chi invariants; this covers the basis construction they depend on.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "Bare"))
from bilinear_basis import (canonical_frame, closure_of, density_seed,
                            hermitian_basis_closure, lattice_point_group_2d,
                            orbital_rep_from_bonds, same_site_seed)

PASS = []


def check(name, cond, detail=""):
    PASS.append(cond)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{'   ' + detail if detail else ''}")


def rot2(deg):
    th = np.deg2rad(deg)
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def c6v_2d():
    """The 12 operations of 6mm as 2x2 matrices, acting on an (x, y) doublet."""
    ops = []
    for k in range(6):
        ops.append(rot2(60 * k))
        ops.append(rot2(60 * k) @ np.diag([1.0, -1.0]))
    return ops


print("dimension of the closure of the densities")

# 1. Single orbital: nothing to mix.
d = hermitian_basis_closure([np.eye(1)], 1)
check("single orbital -> 1", len(d) == 1, f"got {len(d)}")

# 2. Two orbitals that PERMUTE: n_1 <-> n_2, closed already.
swap = np.array([[0.0, 1.0], [1.0, 0.0]])
d = hermitian_basis_closure([np.eye(2), swap], 2)
check("2 orbitals, D permutes -> 2", len(d) == 2, f"got {len(d)}")

# 3. An E doublet that ROTATES. 3, not 4: A2 (the orbital angular momentum) is
#    never generated from densities alone.
d = hermitian_basis_closure(c6v_2d(), 2)
check("E doublet, D rotates -> 3", len(d) == 3, f"got {len(d)}")

# 4. ZrNCl_4: two Zr, each an E doublet; operations rotate within a site and
#    swap the two sites.
blocks = []
for R in c6v_2d():
    blocks.append(np.kron(np.eye(2), R))
    blocks.append(np.kron(swap, R))
d4 = hermitian_basis_closure(blocks, 4)
check("ZrNCl_4 (2 Zr x E doublet) -> 6", len(d4) == 6, f"got {len(d4)}")
check("ZrNCl_4 closure is smaller than the full 16", len(d4) < 16, f"{len(d4)} < 16")

print("\nproperties of the returned basis")

# Hermitian.
check("basis is Hermitian",
      all(np.abs(M - M.conj().T).max() < 1e-10 for M in d4))

# Orthonormal in the real Hilbert-Schmidt product.
gram = np.array([[np.real(np.trace(A.conj().T @ B)) for B in d4] for A in d4])
check("basis is orthonormal", np.abs(gram - np.eye(len(d4))).max() < 1e-9,
      f"max|G - I| = {np.abs(gram - np.eye(len(d4))).max():.1e}")

# Actually closed: every U^dag M U must stay inside the span.
P = np.array([[np.real(np.trace(A.conj().T @ B)) for B in d4] for A in d4])
assert np.abs(P - np.eye(len(d4))).max() < 1e-9


def in_span(M, basis):
    coeffs = [np.real(np.trace(B.conj().T @ M)) for B in basis]
    recon = sum(c * B for c, B in zip(coeffs, basis))
    return np.abs(recon - M).max()


worst = max(in_span(U.conj().T @ M @ U, d4) for U in blocks for M in d4)
check("span is closed under every operation", worst < 1e-9, f"max residual {worst:.1e}")

# The densities must still be representable -- the new basis has to contain the old.
worst = max(in_span(M, d4) for M in density_seed(4))
check("contains the per-orbital densities", worst < 1e-9, f"max residual {worst:.1e}")

# Seeding with more must never shrink the basis.
extra = [np.eye(4, dtype=complex)]
check("extra seed cannot shrink the basis",
      len(hermitian_basis_closure(blocks, 4, extra_seed=extra)) >= len(d4))

print("\nthe Kanamori seed: densities alone cannot hold U'/J'")

# The on-site off-diagonal bilinears the abab/abba components couple through.
# ZrNCl_4: 6 -> 8, the two extra being the A2 orbital current on each Zr.
d8 = hermitian_basis_closure(blocks, 4, extra_seed=same_site_seed([1, 1, 2, 2]))
check("ZrNCl_4 with pair hopping -> 8", len(d8) == 8, f"got {len(d8)}")
worst = max(in_span(U.conj().T @ M @ U, d8) for U in blocks for M in d8)
check("the d = 8 span is still closed under the group", worst < 1e-9,
      f"max residual {worst:.1e}")
d4e = hermitian_basis_closure(c6v_2d(), 2, extra_seed=same_site_seed([1, 1]))
check("one E doublet with pair hopping -> 4", len(d4e) == 4, f"got {len(d4e)}")
# One orbital per site has no same-site pair, so nothing is added: every model
# that was already fine keeps exactly the basis it had.
check("no same-site pair -> no extra seed", same_site_seed([1, 2, 3]) == [])
check("honeycomb (1 orbital/site) stays at 2",
      len(hermitian_basis_closure([np.eye(2), swap], 2,
                                  extra_seed=same_site_seed([1, 2]))) == 2)

print("\ncanonical frame: the basis depends on the SPAN, not on the SVD")

# The closure is projected onto a frame fixed by the span alone, because the
# bubble is contracted in Python and the vertex projected in Julia -- two
# frames of one subspace would mis-match coefficient by coefficient.
rng = np.random.default_rng(0)
V0 = np.linalg.qr(rng.standard_normal((16, 5)))[0]
rot = np.linalg.qr(rng.standard_normal((5, 5)))[0]
f1, f2 = canonical_frame(V0), canonical_frame(V0 @ rot)
check("same span, rotated frame -> same canonical frame",
      np.abs(f1 - f2).max() < 1e-10, f"max difference {np.abs(f1 - f2).max():.1e}")
check("canonical frame is orthonormal",
      np.abs(f1.T @ f1 - np.eye(5)).max() < 1e-10)
check("canonical frame spans the same subspace",
      np.abs(f1 @ f1.T - V0 @ V0.T).max() < 1e-10)

# The whole point: a permutation of the SEED order must not change the basis.
perm = hermitian_basis_closure(list(reversed(blocks)), 4,
                               extra_seed=list(reversed(same_site_seed([1, 1, 2, 2]))))
worst = max(np.abs(A - B).max() for A, B in zip(d8, perm))
check("closure is invariant to operation/seed ordering", worst < 1e-9,
      f"max |difference| {worst:.1e}")

# Closure under complex conjugation, which the Cooper-channel projection needs
# (the partner electron at -k couples to conj(M)); see conjugation_matrix.
C = np.array([[np.real(np.trace(B.conj().T @ A.conj())) for B in d8] for A in d8])
worst = max(np.abs(A.conj() - sum(c * B for c, B in zip(row, d8))).max()
            for A, row in zip(d8, C))
check("span is closed under complex conjugation", worst < 1e-9,
      f"max residual {worst:.1e}")
check("conjugation matrix is a real orthogonal involution",
      np.abs(C @ C - np.eye(len(d8))).max() < 1e-9 and
      np.abs(C - C.T).max() < 1e-9)

print("\nlattice point group from the metric")
a1, a2 = np.array([1.0, 0.0]), np.array([-0.5, np.sqrt(3) / 2])
check("hexagonal lattice -> 12 operations",
      len(lattice_point_group_2d(a1, a2)) == 12,
      f"got {len(lattice_point_group_2d(a1, a2))}")
check("square lattice -> 8 operations",
      len(lattice_point_group_2d(np.array([1.0, 0.0]), np.array([0.0, 1.0]))) == 8,
      f"got {len(lattice_point_group_2d(np.array([1.0, 0.0]), np.array([0.0, 1.0])))}")

print("\nD(g) measured from bonds, on a model whose answer is known")
# Slater-Koster (p_x, p_y) on a triangular lattice: D(g) must come back as the
# 2x2 rotation itself, since p_x, p_y transform like (x, y).
I2 = np.eye(2)
bonds, Fs = [], []
for (n1, n2) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]:
    dv = n1 * a1 + n2 * a2
    dh = dv / np.linalg.norm(dv)
    Fs.append((-0.35 * np.outer(dh, dh) + 0.12 * (I2 - np.outer(dh, dh))).astype(complex))
    bonds.append(dv)
D, cond, resid = orbital_rep_from_bonds(bonds, Fs, rot2(120))
check("residual F(gd) = D F(d) D+ is machine precision", resid < 1e-10,
      f"{resid:.1e}")
check("null space is well defined", cond < 1e-8, f"s_min/s_max = {cond:.1e}")
# D is fixed only up to a phase; compare through a phase-blind invariant.
target = rot2(120)
phase = np.trace(D.conj().T @ target) / 2.0
check("D(C3) equals the coordinate rotation up to a phase",
      np.abs(D * phase.conjugate() / abs(phase) - target).max() < 1e-8)
# And it must be the ROTATING kind, not a permutation -- the whole point.
check("D(C3) is not a permutation",
      max(abs(D[0, 1]), abs(D[1, 0])) > 0.1,
      f"max off-diagonal {max(abs(D[0, 1]), abs(D[1, 0])):.3f}")

print("\nmodel-level entry point (npz layout: spinful, spin innermost)")
from bilinear_basis import closed_basis_for_model, site_positions, site_labels

I2 = np.eye(2)
a1c, a2c = np.array([1.0, 0.0]), np.array([-0.5, np.sqrt(3) / 2])
units3 = [(1.0, 0.0, 0.0), (-0.5, np.sqrt(3) / 2, 0.0)]

# (p_x, p_y) on a triangular lattice: one site, E doublet -> 3
hop = {}
for (n1, n2) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]:
    dv = n1 * a1c + n2 * a2c
    dh = dv / np.linalg.norm(dv)
    h = (-0.35 * np.outer(dh, dh) + 0.12 * (I2 - np.outer(dh, dh))).astype(complex)
    hop[(n1, n2)] = np.kron(h, I2)
pos = [(0.0, 0.0)] * 4
b = closed_basis_for_model(hop, units3, pos, 2, verbose=False, seed="density")
check("E doublet via closed_basis_for_model -> 3", len(b) == 3, f"got {len(b)}")
# The default seed is the Kanamori one, so the basis can hold U'/J' as well as
# make chi covariant: one site, two orbitals -> the full 2x2 Hermitian space.
bk = closed_basis_for_model(hop, units3, pos, 2, verbose=False)
check("E doublet, default (kanamori) seed -> 4", len(bk) == 4, f"got {len(bk)}")

# Single orbital triangular -> 1, i.e. no growth through the same entry point.
hop1 = {(n1, n2): np.kron(np.array([[-0.4]], complex), I2)
        for (n1, n2) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]}
b1 = closed_basis_for_model(hop1, units3, [(0.0, 0.0)] * 2, 1, verbose=False)
check("single orbital via closed_basis_for_model -> 1", len(b1) == 1, f"got {len(b1)}")

# Bookkeeping offsets of ~1e-4 must be absorbed, not treated as real separations.
noisy = [(0.0, 0.0), (0.0, 0.0), (1e-4, -1e-4), (1e-4, -1e-4)]
_, n_sites = site_positions([np.array(t) for t in noisy])
check("1e-4 per-orbital offsets collapse to ONE site", n_sites == 1, f"got {n_sites}")
_, n_sites2 = site_positions([np.array((0.0, 0.0)), np.array((0.5, 0.29))])
check("genuinely distinct positions stay separate", n_sites2 == 2, f"got {n_sites2}")

# site_labels is what decides which orbital pairs get a same-site seed here; it
# must reproduce the interaction builder's sublattice(a) = a <= N/2 ? 1 : 2 for
# honeycomb_uc's ordering (first half on A, second half on B).
zrncl_tau = [np.array((0.0, 0.0)), np.array((1e-4, 1e-4)),
             np.array((0.5, 0.2887)), np.array((0.5001, 0.2888))]
check("site_labels matches the builder's sublattice() on ZrNCl_4",
      site_labels(zrncl_tau) == [1, 1, 2, 2], f"got {site_labels(zrncl_tau)}")

# An incomplete bond orbit must fail loudly rather than return a wrong basis.
truncated = {k: v for k, v in hop.items() if k not in [(1, 1), (-1, -1)]}
try:
    closed_basis_for_model(truncated, units3, pos, 2, verbose=False)
    check("incomplete bond orbit raises", False, "it returned instead")
except RuntimeError as exc:
    check("incomplete bond orbit raises", "bond orbits" in str(exc).lower()
          or "incomplete" in str(exc).lower(), str(exc)[:60])

print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
sys.exit(0 if all(PASS) else 1)
