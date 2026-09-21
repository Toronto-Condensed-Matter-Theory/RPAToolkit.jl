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
def bare_chi(beta:float, w_max: float, dlr_err: float, mu: float, ham):
    from triqs.gf import MeshImFreq
    
    # We want the static bare susceptibility (nw=1, iw_n=0). 
    # lindhard_chi00 evaluates the Matsubara sum analytically directly from the band energies, 
    # bypassing DLR mesh errors entirely.
    # 
    # TODO: Eventually fix the DLR bindings and switch back to the bubble approach:
    # wmesh = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=w_max, eps=dlr_err, symmetrize=True)
    # g0_wk = lattice_dyson_g0_wk(mu=mu, e_k=ham, mesh=wmesh)
    # chi00_wk = imtime_bubble_chi0_wk(g0_wk, nw=1)
    
    bmesh = MeshImFreq(beta=beta, statistic='Boson', n_iw=1)
    chi00_wk = lindhard_chi00(ham, bmesh, mu=mu)
    return chi00_wk

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
def interpolate_chi_mat(chi, direction:int, N: int, ks):

    chiMats = np.zeros([ks.shape[0], N, N], dtype=complex)
    
    for i in range(N):
        for j in range(N):
            
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


def chi_matrix_on_basis(chi, basis, direction: int):
    """chi_kl(q) on a basis of orbital bilinears, read straight off the mesh.

    Returns (N_q, d, d). No interpolation: use this when the q grid IS the
    bubble's own k mesh, which is the case whenever k_size matches.
    """
    ops = [S_operator(M, direction) for M in basis]
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
    return out


def interpolate_chi_basis(chi, basis, direction: int, ks):
    """chi_kl interpolated onto `ks`, the basis-generalised interpolate_chi_mat."""
    ops = [S_operator(M, direction) for M in basis]
    d = len(ops)
    out = np.zeros((ks.shape[0], d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            out[:, i, j] = interpolate_chi(chi_contraction_ops(chi, ops[i], ops[j]), ks)
    return out
