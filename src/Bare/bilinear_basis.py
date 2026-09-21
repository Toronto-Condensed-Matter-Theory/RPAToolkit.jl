"""The symmetry-closed basis of orbital bilinears the RPA has to be built in.

WHY THIS EXISTS
    The RPA works in a basis of bilinear operators. Using one operator per
    orbital -- the densities n_i = c^dag_i c_i and the spins S^a_i -- is only
    legitimate when the orbital representation D(g) PERMUTES orbitals. Under a
    general D(g),

        n_i  ->  sum_jk conj(D_ij) D_ik c^dag_j c_k

    which contains off-diagonal bilinears c^dag_j c_k that a per-orbital basis
    has no slot for. chi_ij then cannot be point-group covariant however exact
    the bubble is: measured on a Slater-Koster (p_x, p_y) model the per-orbital
    chi_ij is 0.35 asymmetric while the total density response, a point-group
    scalar, is exact to 5e-15.

    That is the ZrNCl_4 case: the two orbitals per Zr are an E doublet, so C3
    rotates them into each other rather than permuting them. It is also why
    single-orbital (square, triangular) and permuting (honeycomb) models were
    always fine.

WHAT IT DOES
    Closes the span of the density operators under the measured D(g) and returns
    an orthonormal HERMITIAN basis {M_k} of that span. The closure is computed
    by acting on matrices directly, M -> U^dag M U, rather than through a
    vec/kron identity, so no column-major convention enters; and it is done in
    the REAL vector space of Hermitian matrices, so the result is Hermitian by
    construction rather than by a later projection.

    The dimension is whatever symmetry forces and no more:

        single orbital                  1   (no growth)
        2 orbitals, D permutes          2   (no growth)
        E doublet, D rotates            3   (not 4: the orbital angular
                                             momentum A2 is never generated
                                             from densities alone)
        ZrNCl_4, 2 Zr x E doublet       6   (of 16 possible)

    So every model that already worked keeps its current basis exactly, at zero
    cost, and only the models that need more get more.

    Seed with more than the densities when the interaction couples to more: the
    basis must contain the interaction's own support or the interaction is not
    representable in it. `closure_of` takes an arbitrary seed for that reason.
"""
import numpy as np

__all__ = ["hermitian_basis_closure", "density_seed", "same_site_seed", "closure_of",
           "orbital_rep_from_bonds", "lattice_point_group_2d", "canonical_frame",
           "site_positions", "site_labels", "closed_basis_for_model"]


# --------------------------------------------------------------------------- #
# Hermitian matrices as a real vector space
# --------------------------------------------------------------------------- #
def _herm_to_vec(M):
    """Coordinates of a Hermitian matrix in a real orthonormal frame."""
    n = M.shape[0]
    iu = np.triu_indices(n, 1)
    return np.concatenate([np.real(np.diag(M)),
                           np.sqrt(2.0) * np.real(M[iu]),
                           np.sqrt(2.0) * np.imag(M[iu])])


def _vec_to_herm(v, n):
    iu = np.triu_indices(n, 1)
    m = len(iu[0])
    M = np.zeros((n, n), complex)
    M[np.diag_indices(n)] = v[:n]
    upper = (v[n:n + m] + 1j * v[n + m:]) / np.sqrt(2.0)
    M[iu] = upper
    M += M.conj().T - np.diag(np.diag(M).real)
    return M


def density_seed(n_orb):
    """The per-orbital density operators, the basis used up to now."""
    out = []
    for i in range(n_orb):
        M = np.zeros((n_orb, n_orb), complex)
        M[i, i] = 1.0
        out.append(M)
    return out


def same_site_seed(site_of):
    """Both Hermitian components of every ON-SITE off-diagonal bilinear.

    That is the support of the Kanamori abab (U') and abba (J') terms. A
    closure seeded with the densities alone spans what the densities generate
    -- enough for chi to be covariant, but not enough to HOLD the interaction:
    U' and J' couple through c^dag_a c_b with a != b on the same atom, and a
    vertex whose support leaves the span has no representation in it
    (BilinearVertex.uncovered_bilinears measures that). For ZrNCl_4 this takes
    the closure from 6 to 8, the extra two being the A2 orbital-current
    (L_z-like) bilinear on each Zr.

    Nothing is added for a model with one orbital per site, so square,
    triangular and honeycomb keep exactly the basis they had.
    """
    site_of = list(site_of)
    n = len(site_of)
    out = []
    for b in range(1, n):
        for a in range(b):
            if site_of[a] != site_of[b]:
                continue
            X = np.zeros((n, n), complex)
            X[a, b] = X[b, a] = 1 / np.sqrt(2.0)
            Y = np.zeros((n, n), complex)
            Y[a, b], Y[b, a] = 1j / np.sqrt(2.0), -1j / np.sqrt(2.0)
            out += [X, Y]
    return out


def canonical_frame(V, tol=1e-7):
    """An orthonormal frame of `V`'s column span that depends on the SPAN ALONE.

    `V`'s own columns do not: they come out of an SVD, and within a degenerate
    singular value the choice is arbitrary, so the Julia port of this closure
    can return a different -- equally valid -- frame of the same subspace.
    Coefficients are frame-dependent, so the vertex (projected in Julia) would
    then be mis-matched against the bubble (contracted here).

    The projector P = V V^T is a function of the span only. Gram-Schmidt on its
    columns in index order, keeping each one whose residual survives `tol`, is
    therefore reproducible; the residual of column j is the j-th diagonal entry
    of the remaining projector, which is non-negative, so the sign is fixed too.
    """
    n, d = V.shape
    P = V @ V.T
    Q = np.zeros((n, d))
    filled = 0
    for j in range(n):
        if filled == d:
            break
        v = P[:, j].copy()
        for _ in range(2):          # twice: classical GS loses orthogonality once
            if filled:
                v -= Q[:, :filled] @ (Q[:, :filled].T @ v)
        nv = np.linalg.norm(v)
        if nv <= tol:
            continue
        Q[:, filled] = v / nv
        filled += 1
    if filled != d:
        raise RuntimeError(f"canonical_frame: recovered {filled} of {d} directions; "
                           f"the span is degenerate at the tolerance {tol}.")
    return Q


def closure_of(seed, Ds, tol=1e-9, max_iter=32):
    """Close `seed` under M -> U^dag M U for every U in Ds.

    Returns an orthonormal Hermitian basis of the closed span, as a list of
    matrices. Orthonormal in the real Hilbert-Schmidt inner product
    <A, B> = Re tr(A^dag B).
    """
    if not seed:
        return []
    n = seed[0].shape[0]
    V = np.array([_herm_to_vec(0.5 * (M + M.conj().T)) for M in seed]).T  # columns
    for _ in range(max_iter):
        mats = [_vec_to_herm(V[:, k], n) for k in range(V.shape[1])]
        cols = [V] + [np.array([_herm_to_vec(U.conj().T @ M @ U) for M in mats]).T
                      for U in Ds]
        stacked = np.hstack(cols)
        u, s, _ = np.linalg.svd(stacked, full_matrices=False)
        rank = int((s > tol * max(1.0, s[0])).sum())
        if rank == V.shape[1]:
            break
        V = u[:, :rank]
    else:
        raise RuntimeError("bilinear closure did not converge")
    # Canonicalised, so this agrees matrix-for-matrix with the Julia closure in
    # src/BilinearBasis.jl rather than only subspace-for-subspace.
    V = canonical_frame(V)
    return [_vec_to_herm(V[:, k], n) for k in range(V.shape[1])]


def hermitian_basis_closure(Ds, n_orb, extra_seed=None, tol=1e-9):
    """The basis the RPA should use: densities (plus `extra_seed`) closed under Ds."""
    seed = density_seed(n_orb)
    if extra_seed:
        seed = seed + list(extra_seed)
    return closure_of(seed, Ds, tol=tol)


# --------------------------------------------------------------------------- #
# Measuring D(g) from the hoppings, in bond-vector form
# --------------------------------------------------------------------------- #
def lattice_point_group_2d(a1, a2, tol=1e-9):
    """Cartesian 2x2 operations that map the lattice onto itself.

    Derived from the metric rather than assumed, so it needs no lattice label
    and no orientation convention.
    """
    A = np.column_stack([a1, a2]).astype(float)
    G = A.T @ A
    scale = max(1.0, np.abs(G).max())
    ops = []
    for a in range(-2, 3):
        for b in range(-2, 3):
            for c in range(-2, 3):
                for d in range(-2, 3):
                    M = np.array([[a, b], [c, d]], float)
                    if abs(round(np.linalg.det(M))) != 1:
                        continue
                    if np.abs(M.T @ G @ M - G).max() < tol * scale:
                        ops.append(A @ M @ np.linalg.inv(A))
    return ops


def orbital_rep_from_bonds(bond_vecs, F_mats, g, tol=1e-6):
    """D(g) from the null space of F(g d) = D F(d) D^dag.

    `bond_vecs[i]` is the Cartesian vector d = R + tau_b - tau_a of the bond
    whose orbital matrix is `F_mats[i]`. Working with bond vectors rather than
    lattice vectors R is what makes a single k-independent D(g) exist, including
    for operations that exchange sublattices.

    Returns (D, conditioning, residual): `conditioning` is the ratio of the
    smallest to the largest singular value (small = a well-defined null vector)
    and `residual` is max|F(gd) - D F(d) D^dag|.
    """
    bond_vecs = np.asarray(bond_vecs, float)
    n = F_mats[0].shape[0]
    I_n = np.eye(n)
    zero = np.zeros((n, n), complex)

    def index_of(dv):
        i = int(np.argmin(np.abs(bond_vecs - dv).max(1)))
        return i if np.abs(bond_vecs[i] - dv).max() < tol else -1

    rows = []
    for dv, F in zip(bond_vecs, F_mats):
        j = index_of(g @ dv)
        rows.append(np.kron(I_n, F_mats[j] if j >= 0 else zero) - np.kron(F.T, I_n))
    _, s, Vh = np.linalg.svd(np.vstack(rows))
    D = Vh[-1].conj().reshape(n, n).T
    u, _, vh = np.linalg.svd(D)          # nearest unitary
    D = u @ vh
    resid = 0.0
    for dv, F in zip(bond_vecs, F_mats):
        j = index_of(g @ dv)
        target = F_mats[j] if j >= 0 else zero
        resid = max(resid, np.abs(target - D @ F @ D.conj().T).max())
    return D, s[-1] / s[0], resid


# --------------------------------------------------------------------------- #
# Model-level entry point
# --------------------------------------------------------------------------- #
def site_positions(tau_all, tol=1e-2):
    """Collapse per-orbital positions onto site centroids.

    TightBindingToolkit records a small (~1e-4 a) per-orbital offset that is
    bookkeeping rather than a real separation, and that noise is enough to stop
    symmetry-related bond vectors matching. Orbitals within `tol` of each other
    are treated as one site and given the cluster's mean position.
    """
    tau_all = [np.asarray(t, float)[:2] for t in tau_all]
    centres = []
    for t in tau_all:
        for c in centres:
            if np.linalg.norm(t - c[0]) < tol:
                c[1].append(t)
                break
        else:
            centres.append([t, [t]])
    means = [np.mean(np.array(group), axis=0) for _, group in centres]

    def nearest(t):
        return means[int(np.argmin([np.linalg.norm(t - m) for m in means]))]

    return [nearest(t) for t in tau_all], len(means)


def site_labels(tau_all, tol=1e-2):
    """`site_of[a]`: which atom each orbital belongs to, 1-based.

    Same clustering as `site_positions`, returned as labels so it can be
    compared with the interaction builder's own `sublattice(a)` -- the two must
    agree, or the two sides seed the closure with different same-site pairs and
    end up in different spans.
    """
    tau, _ = site_positions(tau_all, tol=tol)
    labels, out = [], []
    for t in tau:
        for i, c in enumerate(labels):
            if np.linalg.norm(t - c) < tol:
                out.append(i + 1)
                break
        else:
            labels.append(t)
            out.append(len(labels))
    return out


def closed_basis_for_model(hoppings, units, orbital_positions, n_orb_sp,
                           bond_tol=1e-3, resid_tol=1e-6, verbose=True,
                           seed="kanamori"):
    """The symmetry-closed bilinear basis for a model as stored in the npz.

    `hoppings`   {(n1, n2): spinful matrix}, spin the INNER index so the orbital
                 block is M[2a, 2b] (parse_model.jl's convention).
    `units`      the two real-space primitive vectors.
    `orbital_positions`  one Cartesian position per SPINFUL orbital.
    `n_orb_sp`   number of orbitals before spin doubling.
    `seed`       "kanamori" (default) adds `same_site_seed` to the densities so
                 the basis can hold the on-site U'/J' terms as well as carry a
                 covariant chi; "density" seeds with the densities alone.
                 The interaction builder must use the same choice -- it is what
                 makes the two closures span the same space.

    Raises if D(g) cannot be measured to `resid_tol`: a silently wrong basis
    would be worse than no basis, and an incomplete bond orbit is the usual
    cause (the input's R range is a parallelogram, so near its edge a bond can
    be present while its rotated image is not).
    """
    if seed not in ("kanamori", "density"):
        raise ValueError(f"seed must be 'kanamori' or 'density', got {seed!r}")
    a1 = np.asarray(units[0], float)[:2]
    a2 = np.asarray(units[1], float)[:2]
    tau_sp = [np.asarray(orbital_positions[2 * o], float)[:2] for o in range(n_orb_sp)]
    tau, n_sites = site_positions(tau_sp)
    site_of = site_labels(tau_sp)

    acc = {}
    for R, M in hoppings.items():
        Rc = R[0] * a1 + R[1] * a2
        for a in range(n_orb_sp):
            for b in range(n_orb_sp):
                d = Rc + tau[b] - tau[a]
                key = tuple(np.round(d / bond_tol).astype(int))
                entry = acc.setdefault(key, [d, np.zeros((n_orb_sp, n_orb_sp), complex)])
                entry[1][a, b] += M[2 * a, 2 * b]
    bonds = [v[0] for v in acc.values()]
    Fs = [v[1] for v in acc.values()]

    ops = lattice_point_group_2d(a1, a2)
    Ds, worst = [], 0.0
    for g in ops:
        D, cond, resid = orbital_rep_from_bonds(bonds, Fs, g, tol=bond_tol)
        Ds.append(D)
        worst = max(worst, resid)
    if worst > resid_tol:
        raise RuntimeError(
            f"cannot measure the orbital representation: max|F(gd) - D F(d) D+| = "
            f"{worst:.3e} > {resid_tol:.1e}. The bond orbits are probably incomplete "
            f"in this hopping set -- truncate to the largest radius whose orbits are "
            f"complete (see symmetrize_wannier.py) and retry.")

    extra = same_site_seed(site_of) if seed == "kanamori" else None
    basis = hermitian_basis_closure(Ds, n_orb_sp, extra_seed=extra)
    if verbose:
        print(f"bilinear basis: {n_orb_sp} orbitals on {n_sites} site(s) "
              f"(site_of = {site_of}), {len(bonds)} bond vectors, "
              f"{len(ops)} point-group operations, "
              f"max|F(gd) - D F(d) D+| = {worst:.2e}")
        print(f"  symmetry-closed dimension {len(basis)} "
              f"(per-orbital was {n_orb_sp}; seed '{seed}'"
              f"{'; unchanged' if len(basis) == n_orb_sp else ''})")
    return basis
