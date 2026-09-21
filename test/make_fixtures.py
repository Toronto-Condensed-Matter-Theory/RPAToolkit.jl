#!/usr/bin/env python3
"""Generate the small chi(q) fixtures the Julia test suite runs against.

Run once (needs TRIQS); the .npz outputs are committed so the tests themselves
need neither TRIQS nor cluster data:

    PYTHONPATH=/usr/lib/python3/dist-packages python3 test/make_fixtures.py

Four models, chosen to span the cases that behave differently under the
point group:

    square, triangular  single orbital        -> per-orbital basis is closed
    honeycomb           2 sites, D PERMUTES   -> closed
    pdoublet            E doublet, D ROTATES  -> NOT closed (the ZrNCl case)

Each fixture stores, besides chi: the bands (so the input's own symmetry can be
checked) and `perm3`, the C3 permutation of the q grid derived independently
from the TRIQS mesh indices. That permutation is ground truth for testing
build_q_grid_maps, which nothing in the pipeline currently validates.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "Bare"))
import bare_response as br
from bilinear_basis import (hermitian_basis_closure,
                             lattice_point_group_2d,
                             orbital_rep_from_bonds)
from triqs_tprf.tight_binding import TBLattice
from triqs.gf import Idx

BETA, NK, W_MAX, DLR = 50.0, 12, 20.0, 1e-12
I2 = np.eye(2)
HEX = [(1.0, 0.0, 0.0), (-0.5, np.sqrt(3) / 2, 0.0), (0.0, 0.0, 1.0)]
SQ = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]


def hop_square(t=1.0):
    return {(1, 0, 0): -t * I2, (-1, 0, 0): -t * I2,
            (0, 1, 0): -t * I2, (0, -1, 0): -t * I2}


def hop_triangular(t=0.4):
    offs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (1, 1, 0), (-1, -1, 0)]
    return {o: -t * I2 for o in offs}


def hop_honeycomb(t=0.3):
    AB = np.array([[0.0, 1.0], [0.0, 0.0]])
    # F[a, b] is the a -> b amplitude and the bond is d = R + tau_b - tau_a, so
    # the A->B element must sit at +R for the short bond: AB at (1,0), AB.T at
    # (-1,0). The transpose gives bonds of length 1.53 a instead of 0.577 a,
    # which band symmetry cannot see (it is blind to tau) but which makes the
    # bond orbits incomplete and D(g) unmeasurable.
    h = {(0, 0, 0): -t * (AB + AB.T),
         (1, 0, 0): -t * AB, (-1, 0, 0): -t * AB.T,
         (1, 1, 0): -t * AB, (-1, -1, 0): -t * AB.T}
    return {R: np.kron(m, I2) for R, m in h.items()}


def hop_pdoublet(t_sig=-0.35, t_pi=0.12):
    a1, a2 = np.array([1.0, 0.0]), np.array([-0.5, np.sqrt(3) / 2])
    out = {}
    for (n1, n2) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]:
        d = n1 * a1 + n2 * a2
        dh = d / np.linalg.norm(d)
        h = t_sig * np.outer(dh, dh) + t_pi * (I2 - np.outer(dh, dh))
        out[(n1, n2, 0)] = np.kron(h.astype(complex), I2)
    return out


# angle of the rotation whose permutation is stored: 90 deg for the square
# lattice, 120 deg for the hexagonal ones.
def measure_orbital_reps(hoppings, positions, a1, a2, n_orb_sp):
    """{D(g)} for the spinless orbital space, measured from the bond vectors.

    The stored hoppings are spinful, kron(h, I2) with spin innermost, so the
    orbital block is h[a, b] = M[2a, 2b]. Bond vectors are per orbital PAIR,
    d = R + tau_b - tau_a, which is what makes a single k-independent D(g)
    exist even for operations that exchange sublattices.
    """
    tau = [positions[2 * o][0] * a1 + positions[2 * o][1] * a2 for o in range(n_orb_sp)]
    acc = {}
    for R, M in hoppings.items():
        Rc = R[0] * a1 + R[1] * a2
        for a in range(n_orb_sp):
            for b in range(n_orb_sp):
                d = Rc + tau[b] - tau[a]
                key = (round(d[0], 6), round(d[1], 6))
                entry = acc.setdefault(key, [d, np.zeros((n_orb_sp, n_orb_sp), complex)])
                entry[1][a, b] += M[2 * a, 2 * b]
    bonds = [v[0] for v in acc.values()]
    Fs = [v[1] for v in acc.values()]
    Ds = []
    for g in lattice_point_group_2d(a1, a2):
        D, cond, resid = orbital_rep_from_bonds(bonds, Fs, g)
        assert resid < 1e-8, f"D(g) residual {resid:.2e} -- bond orbits incomplete?"
        Ds.append(D)
    return Ds


MODELS = {
    "square":     (SQ,  hop_square(),     [(0, 0, 0)] * 2, 1, 90.0),
    "triangular": (HEX, hop_triangular(), [(0, 0, 0)] * 2, 1, 120.0),
    "honeycomb":  (HEX, hop_honeycomb(),
                   # tau_B = -(2 a1 + a2)/3: the offset for which the three
                   # A->B bonds at offsets (0,0), (1,0), (1,1) form a C3 orbit.
                   [(0, 0, 0), (0, 0, 0), (-2 / 3, -1 / 3, 0), (-2 / 3, -1 / 3, 0)], 2,
                   120.0),
    "pdoublet":   (HEX, hop_pdoublet(), [(0, 0, 0)] * 4, 2, 120.0),
}


def mesh_rotation(bz_units, degrees):
    """The rotation as an integer matrix on MESH INDICES.

    Mesh indices are fractional reciprocal coordinates, so the cartesian
    rotation Rot conjugates as B^-1 Rot B with the reciprocal vectors in the
    COLUMNS of B -- not the real-space matrix.
    """
    B = np.array([list(u)[:2] for u in bz_units])[:2].T
    th = np.deg2rad(degrees)
    Rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    M = np.linalg.inv(B) @ Rot @ B
    Mi = np.rint(M).astype(np.int64)
    assert np.abs(M - Mi).max() < 1e-8, f"rotation is not integral on this mesh: {M}"
    return Mi

for name, (units, hops, pos, n_sites, degrees) in MODELS.items():
    n_orb = len(pos)
    tb = TBLattice(units=units, orbital_positions=pos,
                   orbital_names=[f"o{i}" for i in range(n_orb)], hoppings=hops)
    kmesh = tb.get_kmesh(n_k=(NK, NK, 1))
    e_k = tb.fourier(kmesh)

    nq = NK * NK
    ks = np.zeros((nq, 3))
    bands = np.zeros((nq, n_orb))
    for k in kmesh:
        ks[k.data_index] = np.array(k.value)
        bands[k.data_index] = np.linalg.eigvalsh(e_k[k])
    mu = float(bands.min() + 0.05 * bands.ptp())

    # Ground-truth rotation permutation, from integer mesh indices.
    M = mesh_rotation(kmesh.bz.units, degrees)
    idx = {tuple(np.array(k.index)[:2] % NK): k.data_index for k in kmesh}
    perm = np.empty(nq, dtype=np.int64)
    for m, lin in idx.items():
        perm[lin] = idx[tuple((M @ np.array(m)) % NK)]

    chi00 = br.bare_chi(BETA, W_MAX, DLR, mu, e_k)

    # The symmetry-closed bilinear basis, from D(g) measured on this model.
    a1v = np.array(units[0][:2]); a2v = np.array(units[1][:2])
    Ds = measure_orbital_reps(hops, pos, a1v, a2v, n_sites)
    basis = hermitian_basis_closure(Ds, n_sites)
    n_basis = len(basis)

    chis = {}
    for direction, label in ((0, "chi_NN"), (3, "chi_ZZ")):
        out = np.zeros((nq, n_sites, n_sites), dtype=complex)
        for i in range(n_sites):
            for j in range(n_sites):
                g = chi00[0, 0, 0, 0].copy()
                g.data[:] = np.einsum("wqabcd,ab,cd->wq", chi00.data,
                                      br.S(i, direction, n_sites),
                                      br.S(j, direction, n_sites))[:, :]
                out[:, i, j] = g[Idx(0), :].data[:]
        chis[label] = out
        # the same chi contracted onto the closed basis
        chis[f"chiC_{label[-2:]}"] = br.chi_matrix_on_basis(chi00, basis, direction)

    # Written as flat CSV, read by the tests with DelimitedFiles (stdlib only):
    # the suite then needs no NPZ/FileIO and cannot be broken by a depot issue.
    cols, names = [], []
    cols.append(perm.astype(float));                      names.append("perm3")
    for b in range(n_orb):
        cols.append(bands[:, b]);                         names.append(f"band{b+1}")
    for label in ("NN", "ZZ"):
        m = chis[f"chi_{label}"]
        for i in range(n_sites):
            for j in range(n_sites):
                cols.append(m[:, i, j].real);             names.append(f"re{label}_{i+1}{j+1}")
                cols.append(m[:, i, j].imag);             names.append(f"im{label}_{i+1}{j+1}")
        c = chis[f"chiC_{label}"]
        for i in range(n_basis):
            for j in range(n_basis):
                cols.append(c[:, i, j].real);             names.append(f"reC{label}_{i+1}{j+1}")
                cols.append(c[:, i, j].imag);             names.append(f"imC{label}_{i+1}{j+1}")
    for c in range(2):
        cols.append(ks[:, c]);                            names.append(f"k{c+1}")
    table = np.column_stack(cols)
    path = os.path.join(os.path.dirname(__file__), "fixtures", f"{name}.csv")
    header = ",".join(names)
    np.savetxt(path, table, delimiter=",", header=header, comments="", fmt="%.17g")
    meta = os.path.join(os.path.dirname(__file__), "fixtures", f"{name}.meta")
    with open(meta, "w") as fh:
        # tau_equal: all orbitals at the same position, so chi carries no
        # exp(i k.tau) gauge phases and sum_ij chi_ij IS the total density
        # response. With distinct positions that sum is gauge-dependent.
        tau_equal = all(tuple(t) == tuple(pos[0]) for t in pos)
        fh.write(f"n_orb={n_orb}\nn_sites={n_sites}\nn_k={NK}\nbeta={BETA}\nmu={mu!r}\n")
        fh.write(f"tau_equal={int(tau_equal)}\n")
        fh.write(f"n_basis={n_basis}\n")
        fh.write("reciprocal=" + ";".join(",".join(f"{v!r}" for v in list(u)[:2])
                                          for u in list(kmesh.bz.units)[:2]) + "\n")
    bviol = np.abs(bands[perm] - bands).max()
    print(f"{name:<12} n_orb={n_orb} n_sites={n_sites} n_basis={n_basis}  "
          f"band viol {bviol:.2e}  -> {os.path.basename(path)}")
