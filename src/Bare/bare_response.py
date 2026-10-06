import numpy as np
#####* triqs utilities
from triqs_tprf.tight_binding import TBLattice
from triqs_tprf.lattice_utils import k_space_path
from triqs_tprf.lattice import lindhard_chi00
from triqs.gf import MeshDLRImFreq, Idx
from triqs_tprf.lattice import lattice_dyson_g0_wk
from triqs_tprf.lattice_utils import imtime_bubble_chi0_wk

#####* Pauli matrices ###########
s1 = np.matrix([[0,1],[1,0]])
s2 = np.matrix([[0,-1j],[1j,0]])
s3 = np.matrix([[1,0],[0,-1]])
s4 = np.eye(2, dtype=np.complex128)
sup = np.matrix([[1,0],[0,0]])
sdo = np.matrix([[0,0],[0,1]])
paulis = [s4, s1, s2, s3, s4]

#####* returns the bare response tensor chi0(k, iwn) for a given model
def bare_chi(beta: float, w_max: float, dlr_err: float, mu: float, ham, method: str = "lindhard"):
    """The static bare susceptibility chi0(Omega = 0, q), by one of two routes.

    "lindhard" (default) evaluates the Matsubara sum analytically from the band energies.
    Exact, no frequency-mesh error, and O(N_k * N_q): every q sums over every k. That
    quadratic scaling is the problem -- measured single-threaded on the 4-orbital ZrNCl
    model, 9.3 s at k_size 9 and 756 s at k_size 27, i.e. ~120 days per mu at k_size 297.

    "dlr" builds G0 on a Discrete Lehmann Representation mesh, Fourier transforms to
    (tau, r), forms the bubble as a product there and transforms back. Roughly LINEAR in
    N_k, so the same measurement gives 0.50 s and 4.32 s -- a 175x speedup at k_size 27,
    growing with the grid. Agreement with "lindhard" at k_size 27, as a fraction of
    max|chi|:

        dlr_err   1e-4      1e-6      1e-8      1e-10     1e-12
        n_dlr       22        32        42         50        58
        rel.err  6.0e-04   1.6e-05   1.3e-08    1.8e-09   3.2e-11

    It costs memory: ~3.7x lindhard's peak, because G0 exists simultaneously on the (w,k),
    (w,r) and (tau,r) meshes. Since it also needs far fewer MPI ranks to finish in time,
    and the per-rank footprint is what sets the rank count (tprf all-reduces, so every rank
    holds a full copy), that trade is strongly favourable -- but the launcher's rank count
    has to know which route is in use.

    `symmetrize=False` IS REQUIRED and is not a preference. On a symmetrized DLR mesh
    imtime_bubble_chi0_wk returns silently all-NaN: no exception, no warning, and G0 itself
    is finite, so the NaN is manufactured inside the bubble. (tprf's changelog note about
    raising on non-symmetrized DLR meshes applies to the Eliashberg solver, not here; we do
    the RPA ladder in Julia and never touch that solver.) The isfinite check below exists
    because of that failure mode -- a silent NaN would otherwise propagate through
    chi_RPA into the gap equation as numbers rather than as an error.
    """
    from triqs.gf import MeshImFreq

    if method == "lindhard":
        bmesh = MeshImFreq(beta=beta, statistic='Boson', n_iw=1)
        return lindhard_chi00(ham, bmesh, mu=mu)

    if method != "dlr":
        raise ValueError(f"bubble must be 'lindhard' or 'dlr', got {method!r}")

    wmesh = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=w_max, eps=dlr_err,
                          symmetrize=False)
    g0_wk = lattice_dyson_g0_wk(mu=mu, e_k=ham, mesh=wmesh)
    chi00_wk = imtime_bubble_chi0_wk(g0_wk, nw=1)
    if not np.isfinite(chi00_wk.data).all():
        n_bad = int((~np.isfinite(chi00_wk.data)).sum())
        raise RuntimeError(
            f"the DLR bubble returned {n_bad} non-finite entries of "
            f"{chi00_wk.data.size} (beta={beta}, w_max={w_max}, eps={dlr_err}, "
            f"{len(wmesh)} DLR nodes). G0 was finite, so this came out of "
            f"imtime_bubble_chi0_wk. Check that the DLR mesh is NOT symmetrized, then "
            f"loosen eps; set `bubble: lindhard` in the config to fall back to the "
            f"analytic Matsubara sum.")
    return chi00_wk

#####* ---------------------------------------------------------------------- #
#####* Spin factorization of the bubble.
#####*
#####* A model without spin-orbit coupling has H = h (x) 1_spin, spin the INNER index
#####* (see S_operator), so G0 = g0 (x) 1 and tprf's particle-hole bubble, which pairs
#####* the indices as -G_da G_bc, carries delta(s_d, s_a) delta(s_b, s_c). Contracted
#####* with (M_i (x) sigma^a/2) and (M_j (x) sigma^a/2) that is the spinless bubble
#####* contracted with M_i, M_j times Tr[(sigma^a/2)^2] = 1/2, the same for every
#####* direction a. Exact: on ZrNCl_4 (k_size 12, mu 0.9) the dlr routes agree to 5e-15
#####* for all four directions, per-orbital and closed basis. lindhard agrees to 2e-8:
#####* the spinful Hamiltonian's exactly degenerate spin pairs feed rounding-level
#####* splittings into its (f1 - f2)/(e1 - e2), which the spinless one never has. The spinless bubble holds
#####* 16x fewer chi0 entries (norb^4) and 4x fewer G0 entries (norb^2) per rank.
#####* ---------------------------------------------------------------------- #
SPIN_FACTOR = 0.5   # Tr[(sigma^a / 2)^2] for a = 0..3


def spin_factorizable(unitcell) -> bool:
    """True when the model is exactly h (x) 1_spin: no spin-flip hopping, identical up and down
    blocks, and spin partners at the same position. Exactly, not to a tolerance: the
    factorization is an identity only then."""
    H = np.asarray(unitcell["hopping matrices"])
    P = np.asarray(unitcell["orbital_positions"])
    if H.shape[-1] % 2 or P.shape[1] != H.shape[-1]:
        return False
    return (not np.any(H[:, 0::2, 1::2]) and not np.any(H[:, 1::2, 0::2])
            and np.array_equal(H[:, 0::2, 0::2], H[:, 1::2, 1::2])
            and np.array_equal(P[:, 0::2], P[:, 1::2]))


def spinless_unitcell(unitcell) -> dict:
    """The spin-up block of a spin_factorizable unit cell, in the same dictionary layout."""
    return {"units": np.asarray(unitcell["units"]),
            "orbital_positions": np.asarray(unitcell["orbital_positions"])[:, 0::2],
            "hopping offsets": np.asarray(unitcell["hopping offsets"]),
            "hopping matrices": np.asarray(unitcell["hopping matrices"])[:, 0::2, 0::2]}


#####* returns S^a at site i of total sites N where S^a = [rho, Sx, Sy, Sz, rho].
def S(i:int, direction:int, N: int) -> np.matrix:
    mat = np.zeros((2*N, 2*N), dtype=np.complex128)
    mat[2*i:2*i+2, 2*i:2*i+2] = paulis[direction]/2
    
    return mat

#####* contract the full rank-4 susceptibility tensor to return a matrix
def chi_contraction(chi, i:int, j:int, direction:int, N: int):
    Si = S(i, direction, N)
    Sj = S(j, direction, N)
    
    chi_contracted = chi[0, 0, 0, 0].copy()
    chi_contracted.data[:] = np.einsum('wqabcd,ab,cd->wq', chi.data, Si, Sj)[:, :]
    chi_contracted = chi_contracted[Idx(0), :]
    return chi_contracted

#####* interpolate the susceptibility along a high symmetry path in the Brillouin zone
def interpolate_chi(chi_contracted, ks):
    assert( ks.shape[1] == 3 )
    chi_interp = np.zeros(
        [ks.shape[0]] + list(chi_contracted.target_shape), dtype=complex)

    for kidx, (kx, ky, kz) in enumerate(ks):
        chi_interp[kidx] = chi_contracted((kx, ky, kz))

    return chi_interp

#####* interpolate the contracted susceptibility tensor along a high symmetry path in the Brillouin zone
#####* spin_factorized: `chi` is the spinless bubble of a spin_factorizable model; the result is the
#####* same contraction of the spinful one (direction drops out).
def interpolate_chi_mat(chi, direction:int, N: int, ks, spin_factorized: bool = False):

    chiMats = np.zeros([ks.shape[0], N, N], dtype=complex)
    
    for i in range(N):
        for j in range(N):
            
            if spin_factorized:
                Ei, Ej = np.diag(np.eye(N)[i]), np.diag(np.eye(N)[j])
                chi_contracted = chi_contraction_ops(chi, Ei, Ej)
                chiMats[:, i, j] += SPIN_FACTOR * interpolate_chi(chi_contracted, ks)
            else:
                chi_contracted = chi_contraction(chi, i, j, direction, N)
                chiMats[:, i, j] += interpolate_chi(chi_contracted, ks)
    
    return chiMats

#####* ---------------------------------------------------------------------- #
#####* Contraction on a symmetry-closed basis of orbital bilinears.
#####*
#####* S(i, ...) above puts a Pauli matrix in the spin block of ONE orbital, i.e.
#####* it contracts onto the per-orbital densities and spins. That operator set is
#####* closed under the point group only when D(g) permutes orbitals; when D(g)
#####* rotates them (an E doublet) n_i maps into off-diagonal bilinears the set
#####* does not contain, and the resulting chi_ij cannot be point-group covariant
#####* however exact the bubble is. Measured on a Slater-Koster (p_x, p_y) model:
#####* residual 3.4e-01 per-orbital against 3.2e-15 on the closed basis.
#####*
#####* bilinear_basis.hermitian_basis_closure builds the closed basis; these
#####* functions contract onto it. The index pairing is deliberately the same as
#####* chi_contraction's: it makes no difference to covariance once the basis is
#####* closed (measured 3.2e-15 vs 2.8e-15 either way), and keeping it leaves the
#####* physics of existing single-orbital results untouched.
#####* ---------------------------------------------------------------------- #
def S_operator(M, direction: int):
    """The operator M (x) sigma^a / 2 in the (orbital, spin) basis.

    Spin is the INNER index, matching parse_model.jl's
    positions = repeat(basis, inner = localDim), so orbital o and spin s sit at
    2*(o-1) + s. With M = diag(e_i) this reproduces S(i, direction, N).
    """
    return np.kron(np.asarray(M), paulis[direction] / 2)


def chi_contraction_ops(chi, Oi, Oj):
    """chi contracted with two arbitrary bilinear operators, on the mesh."""
    from triqs.gf import Idx
    out = chi[0, 0, 0, 0].copy()
    out.data[:] = np.einsum('wqabcd,ab,cd->wq', chi.data, Oi, Oj)[:, :]
    return out[Idx(0), :]


def _basis_ops(basis, direction: int, spin_factorized: bool):
    """The operators chi is contracted with, and the factor the contraction carries: M (x) sigma/2
    on a spinful bubble, M itself times SPIN_FACTOR on the spinless one."""
    if spin_factorized:
        return [np.asarray(M) for M in basis], SPIN_FACTOR
    return [S_operator(M, direction) for M in basis], 1.0


def chi_matrix_on_basis(chi, basis, direction: int, spin_factorized: bool = False):
    """chi_kl(q) on a basis of orbital bilinears, read straight off the mesh.

    Returns (N_q, d, d). No interpolation: use this when the q grid IS the
    bubble's own k mesh, which is the case whenever k_size matches.
    """
    ops, factor = _basis_ops(basis, direction, spin_factorized)
    d = len(ops)
    first = chi_contraction_ops(chi, ops[0], ops[0])
    n_q = first.data.shape[0]
    out = np.zeros((n_q, d, d), dtype=complex)
    out[:, 0, 0] = first.data[:]
    for i in range(d):
        for j in range(d):
            if i == 0 and j == 0:
                continue
            out[:, i, j] = chi_contraction_ops(chi, ops[i], ops[j]).data[:]
    return factor * out


def interpolate_chi_basis(chi, basis, direction: int, ks, spin_factorized: bool = False):
    """chi_kl interpolated onto `ks`, the basis-generalised interpolate_chi_mat."""
    ops, factor = _basis_ops(basis, direction, spin_factorized)
    d = len(ops)
    out = np.zeros((ks.shape[0], d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            out[:, i, j] = interpolate_chi(chi_contraction_ops(chi, ops[i], ops[j]), ks)
    return factor * out
