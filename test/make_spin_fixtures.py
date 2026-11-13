#!/usr/bin/env python3
"""Generate the spin-orbit-coupling chi(q) fixtures the Julia test suite runs against.

Run once in the TRIQS environment; the CSV outputs are committed, so the tests themselves need
no TRIQS:

    python3 test/make_spin_fixtures.py

Four single-site models on a 12 x 12 mesh, each contracted onto ALL SIXTEEN channel pairs
(N, X, Y, Z) x (N, X, Y, Z) of the spinful bubble, through the same functions run_bare.py uses:

    ising          triangular, Ising SOC lam sigma_z on the C3 bonds   U1z, time reversal
    ising_zeeman   + an on-site B_z sigma_z                            U1z, chi^{NZ} != 0
    ising_inplane  + an on-site B_x sigma_x                            none
    square         square lattice, no SOC                              SU2, spinful bubble forced

The hopping matrices are written to the .meta in the TRIQS layout parse_model.jl produces, so the
tests can rebuild H(k) independently of TRIQS and check every contraction -- and the orientation
combine_chis gives the off-diagonal pairs -- against a brute-force sum. The square model also
stores the factorized (spinless x 1/2) contraction, which its diagonal pairs must equal.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "triqs_reference"))
import bare_response as br
import model as mdl

BETA, NK = 50.0, 12
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
HEX = [[1.0, 0.0], [-0.5, np.sqrt(3) / 2]]
SQ = [[1.0, 0.0], [0.0, 1.0]]


def unitcell(units, hoppings):
    """The dictionary parse_model.jl writes, for one site at the origin."""
    offsets = list(hoppings)
    return {"units": np.array([u + [0.0] for u in units]).T,
            "orbital_positions": np.zeros((3, 2)),
            "hopping offsets": np.array(offsets).T,
            "hopping matrices": np.array([hoppings[o] for o in offsets])}


def ising(t=-0.4, lam=0.1, bz=0.0, bx=0.0):
    # t on every bond, lam sigma_z odd under bond reversal on a1, a2, -a1-a2 (models.jl ising_model)
    h = {(0, 0): bz * SZ + bx * SX}
    for o in [(1, 0), (0, 1), (-1, -1)]:
        M = t * I2 + (lam / 2j) * SZ
        h[o] = M
        h[(-o[0], -o[1])] = M.conj().T
    return h


def square(t=-1.0):
    return {o: t * I2 for o in [(1, 0), (-1, 0), (0, 1), (0, -1)]}


MODELS = {
    "ising":         (HEX, ising(),          "U1z"),
    "ising_zeeman":  (HEX, ising(bz=0.05),   "U1z"),
    "ising_inplane": (HEX, ising(bx=0.05),   "none"),
    "square":        (SQ,  square(),         "SU2"),
}

here = os.path.join(os.path.dirname(__file__), "fixtures")
for name, (units, hops, expected) in MODELS.items():
    uc = unitcell(units, hops)
    symmetry = br.spin_symmetry(uc)
    assert symmetry == expected, f"{name}: detected {symmetry}, expected {expected}"

    model = mdl.triqs_model(uc)
    ham = mdl.hamiltonian(model, NK)
    kmesh = model.get_kmesh(n_k=(NK, NK, 1))
    ks = np.array([k.value for k in kmesh])
    bands = mdl.bands(ham, kmesh)
    mu = float(np.min(bands) + 0.3 * np.ptp(bands))

    chi00 = br.bare_chi(BETA, 20.0, 1e-12, mu, ham, method="lindhard")
    cols, names = [ks[:, 0], ks[:, 1]], ["k1", "k2"]
    worst_cross = 0.0
    for (a, b) in br.CHANNEL_PAIRS["none"]:
        c = br.interpolate_chi_mat(chi00, a, 1, ks, False, direction2=b)[:, 0, 0]
        label = br.pair_label(a, b)[4:]
        cols += [c.real, c.imag]
        names += [f"re{label}", f"im{label}"]
        if (a, b) not in br.CHANNEL_PAIRS[symmetry]:
            worst_cross = max(worst_cross, np.abs(c).max())
    if symmetry == "SU2":
        spinless = mdl.hamiltonian(mdl.triqs_model_spinless(br.spinless_unitcell(uc)), NK)
        chi_s = br.bare_chi(BETA, 20.0, 1e-12, mu, spinless, method="lindhard")
        c = br.interpolate_chi_mat(chi_s, 0, 1, ks, True)[:, 0, 0]
        cols += [c.real, c.imag]
        names += ["refact", "imfact"]

    np.savetxt(os.path.join(here, f"spin_{name}.csv"), np.column_stack(cols), delimiter=",",
               header=",".join(names), comments="", fmt="%.17g")
    with open(os.path.join(here, f"spin_{name}.meta"), "w") as fh:
        fh.write(f"symmetry={symmetry}\nn_k={NK}\nbeta={BETA}\nmu={float(mu)!r}\n")
        fh.write("units=" + ";".join(",".join(repr(float(v)) for v in u) for u in units) + "\n")
        fh.write("hoppings=" + ";".join(
            f"{o[0]},{o[1]}:" + ",".join(f"{float(z.real)!r},{float(z.imag)!r}" for z in M.ravel())
            for o, M in hops.items()) + "\n")
    print(f"{name:<14} {symmetry:<5} mu={mu:+.4f}  max |chi| over pairs {symmetry} forbids: "
          f"{worst_cross:.1e}")
