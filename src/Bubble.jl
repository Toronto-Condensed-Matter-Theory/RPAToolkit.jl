module Bubble

#=
The bare particle-hole bubble, in Julia.

Everything here works on a `BandModel`: the hopping matrices t(R) of a TightBindingToolkit `UnitCell`
(spin the INNER index, every bond's Hermitian conjugate filled in, on-site fields included), with
the Bloch Hamiltonian in the periodic gauge,

    H(k) = sum_R t(R) e^{i k.R},      R a lattice vector (no basis offsets),

which is TightBindingToolkit's `FillHoppingHamiltonian` exactly and the one `interaction()` builds
V(q) in. No gauge phases are needed anywhere.

The kernel is the Lindhard sum over band pairs (m at k, n at k - q), for operators O_p,

    raw_ij(q, w) = (1/N) sum_k sum_mn  F_mn(w) <m k|O_j|n k-q> <n k-q|O_i|m k>,
    F_mn(0)      = (f_m - f_n) / (E_n - E_m)                      (static, -f' when degenerate)
    F_mn(w)      = (f_m - f_n) / (E_n - E_m - w - i eta)          (retarded)

[`static_bubble`](@ref) and [`dynamic_bubble`](@ref)'s internals compute exactly this. Statically
raw(-q) = transpose(raw(q)) (the isothermal susceptibility is symmetric in its two operators).

What the bare stage STORES, pair by pair, is the orbital bubble chi0_abcd contracted with O_i on
(a, b) and O_j on (c, d), which is

    stored(q) = transpose(raw(q) computed with the TRANSPOSED operators O_p^T)

([`stored_bubble`](@ref)). `combine_chis` then returns 2 transpose(stored) = 2 raw_{O^T}(q): the
matrix the static RPA has always multiplied with `interaction()`'s V(q). Single-site models with a
spin-flip symmetry cannot tell the transposes apart, which is why only a multi-site cell showed it.

[`dynamic_bubble`](@ref) returns the retarded function that reduces to that matrix at w = 0. In the
continuum it is chi(q, w) = 2 transpose(raw_{O^T}(-q, w)), a proper response matrix (response of
the a-th operator to a field on the b-th); written over the same (k, k - q) pairs as the stored
bubble, so that w = 0 is the same discrete sum even for q off the k grid,

    chi_ab(q, w) = (2/N) sum_k sum_mn (f_n(k-q) - f_m(k)) / (E_m(k) - E_n(k-q) - w - i eta)
                                      x conj(<m k|O_a^T|n k-q>) <m k|O_b^T|n k-q>,

with w Im chi_aa(q, w) >= 0, one 4d x 4d matrix per (q, w), channel innermost.
=#

using LinearAlgebra, TightBindingToolkit, FFTW
using ..RPAToolkit.DLR: dlr_grid, matsubara_weights
using ..RPAToolkit.SpinSectors: SPIN_SYMMETRIES
using ..RPAToolkit.BilinearBasis: bilinear_closure, same_site_seed, lattice_point_group_2d,
                                  orbital_rep_from_bonds

export BandModel, band_model, bloch_hamiltonian, spinless_model, band_spin_symmetry, spin_symmetry,
       reciprocal_vectors, k_grid, k_path, fermi, band_energies,
       channel_operators, closed_basis_for_model, static_bubble, static_bubble_dlr, stored_bubble,
       dynamic_bubble, lindhard_weight, band_eigen, particle_hole_edges, PAULI

const PAULI = (ComplexF64[1 0; 0 1], ComplexF64[0 1; 1 0], ComplexF64[0 -im; im 0], ComplexF64[1 0; 0 -1])

#####* ---------------------------------------------------------------------------------------
#####* band model
#####* ---------------------------------------------------------------------------------------

"""
    BandModel

A periodic-gauge hopping model: `matrices[i]` is t(R) for the lattice vector
`lattice_vectors[i] = offsets[i][1] a1 + offsets[i][2] a2`, every bond's Hermitian conjugate
included, and `positions` holds one position per (spinful) orbital. Build one with
[`band_model`](@ref).
"""
struct BandModel
    primitives::Vector{Vector{Float64}}
    positions::Vector{Vector{Float64}}
    offsets::Vector{Vector{Int}}
    lattice_vectors::Vector{Vector{Float64}}
    matrices::Vector{Matrix{ComplexF64}}
end

norbitals(bm::BandModel) = size(bm.matrices[begin], 1)

"""
    band_model(uc::UnitCell) -> BandModel
    band_model(primitives, hoppings::Dict; positions = nothing) -> BandModel

From a TightBindingToolkit unit cell, term by term as its `FillHoppingHamiltonian` builds H(k)
(the bonds with their Hermitian conjugates, on-site bonds symmetrized, minus the on-site `fields`),
or from a `Dict(offset => matrix)` that already holds every bond's Hermitian conjugate.
"""
function band_model(uc::UnitCell)
    length(uc.primitives) == 2 || error("band_model: needs a 2D unit cell, got $(length(uc.primitives)) primitives.")
    d = uc.localDim
    n = d * length(uc.basis)
    hoppings = Dict{Vector{Int}, Matrix{ComplexF64}}()
    t(offset) = get!(() -> zeros(ComplexF64, n, n), hoppings, Int.(offset))
    orbitals(site) = d * (site - 1) .+ (1:d)
    for bond in uc.bonds
        i, j = orbitals(bond.base), orbitals(bond.target)
        if bond.base == bond.target && iszero(bond.offset)
            t([0, 0])[i, i] .+= (bond.mat .+ bond.mat') ./ 2
        else
            t(bond.offset)[i, j] .+= bond.mat
            t(-bond.offset)[j, i] .+= bond.mat'
        end
    end
    for site in eachindex(uc.basis)
        field = sum(uc.fields[site] .* uc.OnSiteMats)
        any(!iszero, field) && (t([0, 0])[orbitals(site), orbitals(site)] .-= field)
    end
    isempty(hoppings) && t([0, 0])
    positions = [Float64.(uc.basis[site][1:2]) for site in eachindex(uc.basis) for _ in 1:d]
    return band_model([uc.primitives[1], uc.primitives[2]], hoppings; positions = positions)
end

function band_model(primitives, hoppings::AbstractDict; positions = nothing)
    a1, a2 = Float64.(primitives[1][1:2]), Float64.(primitives[2][1:2])
    offsets = [Int.(collect(o)) for o in keys(hoppings)]
    mats = [Matrix{ComplexF64}(hoppings[o]) for o in keys(hoppings)]
    n = size(mats[begin], 1)
    pos = positions === nothing ? [zeros(2) for _ in 1:n] : [Float64.(p[1:2]) for p in positions]
    return BandModel([a1, a2], pos, offsets, [o[1] .* a1 .+ o[2] .* a2 for o in offsets], mats)
end

"""
    bloch_hamiltonian(bm, k) -> Hermitian matrix

``H(k) = \\sum_R t(R) e^{i k \\cdot R}``. `k` is Cartesian; a third component is ignored.
"""
function bloch_hamiltonian(bm::BandModel, k::AbstractVector)
    n = norbitals(bm)
    H = zeros(ComplexF64, n, n)
    kx, ky = k[1], k[2]
    for (R, M) in zip(bm.lattice_vectors, bm.matrices)
        H .+= cis(kx * R[1] + ky * R[2]) .* M
    end
    return Hermitian(0.5 .* (H .+ H'))
end

"""
    spinless_model(bm) -> BandModel

The spin-up block. For a model with H = h (x) 1_spin (spin inner), this is h.
"""
function spinless_model(bm::BandModel)
    up = 1:2:norbitals(bm)
    return BandModel(bm.primitives, bm.positions[up], bm.offsets, bm.lattice_vectors,
                     [M[up, up] for M in bm.matrices])
end

"""
    band_spin_symmetry(bm) -> String

`"SU2"`, `"U1z"` or `"none"`: no spin-flip
entries and identical spin blocks (with spin partners at the same position) is SU2, no spin-flip
entries U1z.
"""
function band_spin_symmetry(bm::BandModel)::String
    n = norbitals(bm)
    iseven(n) || return "none"
    up, dn = 1:2:n, 2:2:n
    any(M -> any(!iszero, M[up, dn]) || any(!iszero, M[dn, up]), bm.matrices) && return "none"
    same = all(M -> M[up, up] == M[dn, dn], bm.matrices) && bm.positions[up] == bm.positions[dn]
    return same ? "SU2" : "U1z"
end

"""
    spin_symmetry(uc::UnitCell) -> String

[`band_spin_symmetry`](@ref) of the unit cell's [`band_model`](@ref), on-site fields included. Needs
`localDim = 2`.
"""
function spin_symmetry(uc::UnitCell)::String
    uc.localDim == 2 || error("spin_symmetry: localDim = $(uc.localDim); the spin index needs 2.")
    return band_spin_symmetry(band_model(uc))
end

#####* ---------------------------------------------------------------------------------------
#####* momenta
#####* ---------------------------------------------------------------------------------------

"""
    reciprocal_vectors(primitives) -> (b1, b2)

``b_i \\cdot a_j = 2\\pi \\delta_{ij}``.
"""
function reciprocal_vectors(primitives)
    B = 2pi * inv(transpose(hcat(Float64.(primitives[1][1:2]), Float64.(primitives[2][1:2]))))
    return B[:, 1], B[:, 2]
end

"""
    k_grid(primitives, n) -> Vector{Vector{Float64}}

The `n × n` Monkhorst-Pack-free grid ``k = (i b_1 + j b_2)/n``, ``i, j = 0 … n-1``, with `j`
running fastest, so grid index `n i + j + 1` is momentum `(i, j)`.
"""
function k_grid(primitives, n::Integer)
    b1, b2 = reciprocal_vectors(primitives)
    return [(i / n) .* b1 .+ (j / n) .* b2 for i in 0:n-1 for j in 0:n-1]
end

"""
    k_path(primitives, points; num = 100) -> (ks, distance, ticks)

The closed path through `points` (reduced coordinates, in units of the reciprocal vectors) and
back to the first, `num` points per segment with both ends included: Cartesian momenta, the
distance travelled at each, and the distance at each corner.
"""
function k_path(primitives, points; num::Integer = 100)
    b1, b2 = reciprocal_vectors(primitives)
    cart(p) = p[1] .* b1 .+ p[2] .* b2
    corners = [cart(Float64.(collect(p))) for p in points]
    length(corners) >= 2 || error("k_path: at least two k-points are needed.")
    segments = [(corners[i], corners[mod1(i + 1, length(corners))]) for i in eachindex(corners)]
    ks = Vector{Float64}[]
    dist = Float64[]
    ticks = [0.0]
    travelled = 0.0
    for (ki, kf) in segments
        for x in range(0.0, 1.0, length = num)
            k = ki .+ x .* (kf .- ki)
            push!(ks, k)
            push!(dist, travelled + norm(k .- ki))
        end
        travelled = dist[end]
        push!(ticks, travelled)
    end
    return ks, dist, ticks
end

#####* ---------------------------------------------------------------------------------------
#####* occupations
#####* ---------------------------------------------------------------------------------------

"""Fermi function, overflow-free."""
fermi(E::Real, beta::Real, mu::Real) = 0.5 * (1 - tanh(0.5 * beta * (E - mu)))

logcosh(x::Real) = abs(x) + log1p(exp(-2abs(x))) - log(2.0)
logsinhc(y::Real) = abs(y) < 1e-4 ? y^2 / 6 : abs(y) + log1p(-exp(-2abs(y))) - log(2abs(y))

"""
    lindhard_weight(E1, E2, beta, mu) -> Float64

``(f(E_1) - f(E_2)) / (E_2 - E_1)``, evaluated as
``\\frac{\\beta}{4}\\,\\frac{\\sinh y / y}{\\cosh x_1 \\cosh x_2}`` with ``x_i = \\beta(E_i-\\mu)/2``,
``y = x_2 - x_1``: no cancellation for nearly degenerate levels (it becomes ``-f'`` smoothly) and no
overflow at large ``\\beta``. Never negative.
"""
function lindhard_weight(E1::Real, E2::Real, beta::Real, mu::Real)
    x1 = 0.5 * beta * (E1 - mu)
    x2 = 0.5 * beta * (E2 - mu)
    return 0.25 * beta * exp(logsinhc(x2 - x1) - logcosh(x1) - logcosh(x2))
end

"""
    band_energies(bm, ks) -> Matrix{Float64}

The eigenvalues of H(k), one column per momentum.
"""
band_energies(bm::BandModel, ks) = reduce(hcat, [eigvals(bloch_hamiltonian(bm, k)) for k in ks])

#####* ---------------------------------------------------------------------------------------
#####* operators
#####* ---------------------------------------------------------------------------------------

"""
    channel_operators(n_orb; basis = nothing, directions = 0:3, spinful = true)

The operators the bubble is contracted with, ordered with the direction innermost:
entry `length(directions) (p - 1) + a` is ``M_p \\otimes \\sigma^{d_a}/2`` (spin inner, so orbital
`o` spin `s` is index `2(o - 1) + s`), with ``M_p`` the `basis` matrices or, by default, the
per-orbital densities ``E_{pp}``. With `spinful = false` the operators are the ``M_p`` themselves,
for the spinless half of an SU(2) model.
"""
function channel_operators(n_orb::Integer; basis = nothing, directions = 0:3, spinful::Bool = true)
    Ms = basis === nothing ? [(E = zeros(ComplexF64, n_orb, n_orb); E[p, p] = 1; E) for p in 1:n_orb] :
         [Matrix{ComplexF64}(M) for M in basis]
    spinful || return Ms
    return [kron(M, PAULI[a + 1] ./ 2) for M in Ms for a in directions]
end

#####* ---------------------------------------------------------------------------------------
#####* the symmetry-closed bilinear basis of a model
#####* ---------------------------------------------------------------------------------------

#####* per-orbital positions collapsed onto site centroids (TightBindingToolkit's ~1e-4 per-orbital
#####* bookkeeping offsets would otherwise stop symmetry-related bonds matching), and the site labels.
function site_clusters(tau_all; tol::Real = 1e-2)
    centres = Vector{Vector{Float64}}[]
    for t in tau_all
        c = findfirst(group -> norm(t .- group[1]) < tol, centres)
        c === nothing ? push!(centres, [t]) : push!(centres[c], t)
    end
    means = [sum(g) ./ length(g) for g in centres]
    nearest(t) = argmin([norm(t .- m) for m in means])
    labels = [nearest(t) for t in tau_all]
    return [means[l] for l in labels], labels
end

"""
    closed_basis_for_model(bm; seed = "kanamori", bond_tol = 1e-3, resid_tol = 1e-6)

The symmetry-closed bilinear basis of an SU(2) model: the per-orbital densities (plus, with the
default `"kanamori"` seed, every on-site off-diagonal bilinear) closed under the orbital
representation D(g) of the lattice point group, measured from the bond vectors
``d = R + \\tau_b - \\tau_a``. The Julia port of the bare stage's `closed_basis_for_model`.
"""
function closed_basis_for_model(bm::BandModel; seed::AbstractString = "kanamori",
                                bond_tol::Real = 1e-3, resid_tol::Real = 1e-6, verbose::Bool = false)
    seed in ("kanamori", "density") || error("closed_basis_for_model: seed must be \"kanamori\" or \"density\", got $(repr(seed)).")
    n_orb = norbitals(bm) ÷ 2
    tau, site_of = site_clusters([bm.positions[2o - 1] for o in 1:n_orb])
    acc = Dict{Vector{Int}, Tuple{Vector{Float64}, Matrix{ComplexF64}}}()
    for (R, M) in zip(bm.lattice_vectors, bm.matrices), a in 1:n_orb, b in 1:n_orb
        d = R .+ tau[b] .- tau[a]
        key = Int.(round.(d ./ bond_tol))
        entry = get!(acc, key) do
            (d, zeros(ComplexF64, n_orb, n_orb))
        end
        entry[2][a, b] += M[2a - 1, 2b - 1]
    end
    bonds = [v[1] for v in values(acc)]
    Fs = [v[2] for v in values(acc)]
    Ds = Matrix{ComplexF64}[]
    worst = 0.0
    for g in lattice_point_group_2d(bm.primitives[1], bm.primitives[2])
        D, _, resid = orbital_rep_from_bonds(bonds, Fs, g; tol = bond_tol)
        push!(Ds, D)
        worst = max(worst, resid)
    end
    worst <= resid_tol ||
        error("closed_basis_for_model: cannot measure the orbital representation, " *
              "max|F(gd) - D F(d) D'| = $(worst) > $(resid_tol). The bond orbits are probably " *
              "incomplete in this hopping set; truncate to the largest radius whose orbits are complete.")
    extra = seed == "kanamori" ? same_site_seed(site_of) : nothing
    basis = bilinear_closure(Ds, n_orb; extra_seed = extra)
    verbose && println("bilinear basis: $n_orb orbitals on $(maximum(site_of)) site(s), $(length(bonds)) bond " *
                       "vectors, max|F(gd) - D F(d) D'| = $(worst); closed dimension $(length(basis)).")
    return basis
end

#####* ---------------------------------------------------------------------------------------
#####* the kernel
#####* ---------------------------------------------------------------------------------------

#####* run f(chunk) over `n` items split into contiguous chunks, one task each
function chunked(f, n::Int)
    ntasks = max(1, min(n, 4 * Threads.nthreads()))
    chunks = collect(Iterators.partition(1:n, cld(n, ntasks)))
    tasks = [Threads.@spawn f(chunk) for chunk in chunks]
    foreach(wait, tasks)
    return nothing
end

#####* The operators, both dense and as their nonzero entries. Per-orbital operators (a site projector
#####* times a Pauli matrix) have one or two entries each, and contracting entry by entry beats any
#####* small-matrix BLAS call; a dense closed-basis operator goes through BLAS instead.
struct OpSet
    dense::Vector{Matrix{ComplexF64}}
    cat::Matrix{ComplexF64}                 # [O_1 O_2 ... O_d], n x (n d)
    rows::Vector{Vector{Int}}
    cols::Vector{Vector{Int}}
    vals::Vector{Vector{ComplexF64}}
    sparse::Bool
end
function OpSet(ops)
    dense = [Matrix{ComplexF64}(O) for O in ops]
    nz = [findall(!iszero, O) for O in dense]
    n = size(dense[1], 1)
    return OpSet(dense, reduce(hcat, dense), [[I[1] for I in z] for z in nz], [[I[2] for I in z] for z in nz],
                 [[dense[p][I] for I in nz[p]] for p in eachindex(dense)], sum(length, nz) <= n * length(dense))
end
Base.length(o::OpSet) = length(o.dense)

#####* Per-task buffers.
struct Workspace
    X::Matrix{ComplexF64}           # X[m + nb (n - 1), p] = <m k|O_p|n k-q>
    WX::Matrix{ComplexF64}          # w .* X
    w::Vector{ComplexF64}
    T::Matrix{ComplexF64}           # O_p U2, dense path only
    Y::Matrix{ComplexF64}           # U1' O_p U2, dense path only
end
Workspace(n::Int, d::Int) = Workspace(zeros(ComplexF64, n * n, d), zeros(ComplexF64, n * n, d), zeros(ComplexF64, n * n),
                                      zeros(ComplexF64, n, n), zeros(ComplexF64, n, n))

function matrix_elements!(ws::Workspace, ops::OpSet, U1::AbstractMatrix, U2::AbstractMatrix)
    X = ws.X
    nb = size(U1, 2)
    if ops.sparse
        fill!(X, 0)
        @inbounds for p in eachindex(ops.vals)
            rows, cols, vals = ops.rows[p], ops.cols[p], ops.vals[p]
            for t in eachindex(vals)
                a, b, v = rows[t], cols[t], vals[t]
                for nn in 1:nb
                    vb = v * U2[b, nn]
                    off = nb * (nn - 1)
                    for m in 1:nb
                        X[off + m, p] += conj(U1[a, m]) * vb
                    end
                end
            end
        end
    else
        @inbounds for p in eachindex(ops.dense)
            small_mul!(ws.T, ops.dense[p], U2)
            small_mul!(ws.Y, U1, ws.T; adjA = true)
            for idx in eachindex(ws.Y)
                X[idx, p] = ws.Y[idx]
            end
        end
    end
    return X
end

#####* raw += X' diag(w) X, i.e. raw_ij += sum_mn w_mn conj(X_mn,i) X_mn,j. Plain loops, here and in the
#####* other small products below: OpenBLAS serializes tiny gemm calls made from several Julia threads.
function accumulate!(raw::AbstractMatrix, ws::Workspace)
    X, WX, w = ws.X, ws.WX, ws.w
    @inbounds for p in axes(X, 2), idx in axes(X, 1)
        WX[idx, p] = w[idx] * X[idx, p]
    end
    @inbounds for j in axes(X, 2), i in axes(X, 2)
        acc = zero(ComplexF64)
        @simd for idx in axes(X, 1)
            acc += conj(X[idx, i]) * WX[idx, j]
        end
        raw[i, j] += acc
    end
    return raw
end

#####* C = A B (adjA = false) or A' B (adjA = true), or A B' (adjB = true), for small dense matrices
function small_mul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix; adjA::Bool = false, adjB::Bool = false)
    fill!(C, 0)
    @inbounds for j in axes(C, 2), l in (adjA ? axes(A, 1) : axes(A, 2))
        b = adjB ? conj(B[j, l]) : B[l, j]
        if adjA
            @simd for i in axes(C, 1)
                C[i, j] += conj(A[l, i]) * b
            end
        else
            @simd for i in axes(C, 1)
                C[i, j] += A[i, l] * b
            end
        end
    end
    return C
end

struct Eigen2
    E::Vector{Float64}
    U::Matrix{ComplexF64}
end

#####* H(k) into a preallocated matrix
function bloch_hamiltonian!(H::Matrix{ComplexF64}, bm::BandModel, k::AbstractVector)
    fill!(H, 0)
    kx, ky = k[1], k[2]
    @inbounds for (R, M) in zip(bm.lattice_vectors, bm.matrices)
        ph = cis(kx * R[1] + ky * R[2])
        for idx in eachindex(H)
            H[idx] += ph * M[idx]
        end
    end
    @inbounds for j in axes(H, 2), i in 1:j       # Hermitian part
        h = 0.5 * (H[i, j] + conj(H[j, i]))
        H[i, j] = h
        H[j, i] = conj(h)
    end
    return H
end

"""
    jacobi_eigen!(A, V, E) -> (E, V)

Eigenvalues (ascending, into `E`) and eigenvectors (columns of `V`) of the small Hermitian matrix
`A`, which is destroyed, by cyclic complex Jacobi rotations. Used instead of LAPACK for the many
tiny diagonalizations in the bubbles: LAPACK calls on small matrices do not run concurrently from
Julia threads, these do, and Jacobi is accurate to machine precision.
"""
function jacobi_eigen!(A::Matrix{ComplexF64}, V::Matrix{ComplexF64}, E::Vector{Float64}; max_sweeps::Int = 60)
    n = size(A, 1)
    fill!(V, 0)
    @inbounds for i in 1:n
        V[i, i] = 1
    end
    scale = 0.0
    @inbounds for idx in eachindex(A)
        scale = max(scale, abs(A[idx]))
    end
    tol = (1e-15 * max(scale, floatmin(Float64)))^2
    @inbounds for _ in 1:max_sweeps
        off = 0.0
        for q in 2:n, p in 1:q-1
            off += abs2(A[p, q])
        end
        off <= tol && break
        for q in 2:n, p in 1:q-1
            apq = A[p, q]
            mag = abs(apq)
            mag == 0 && continue
            phase = apq / mag                              # e^{i phi}
            theta = (real(A[q, q]) - real(A[p, p])) / (2mag)
            t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + sqrt(theta^2 + 1))
            c = 1 / sqrt(t^2 + 1)
            s = t * c
            #####* G = [c, s; -s conj(phase), c conj(phase)] on (p, q); A <- G' A G, V <- V G
            gpp, gpq, gqp, gqq = complex(c), complex(s), -s * conj(phase), c * conj(phase)
            for i in 1:n
                aip, aiq = A[i, p], A[i, q]
                A[i, p] = aip * gpp + aiq * gqp
                A[i, q] = aip * gpq + aiq * gqq
                vip, viq = V[i, p], V[i, q]
                V[i, p] = vip * gpp + viq * gqp
                V[i, q] = vip * gpq + viq * gqq
            end
            for j in 1:n
                apj, aqj = A[p, j], A[q, j]
                A[p, j] = conj(gpp) * apj + conj(gqp) * aqj
                A[q, j] = conj(gpq) * apj + conj(gqq) * aqj
            end
            A[p, q] = 0
            A[q, p] = 0
        end
    end
    @inbounds for i in 1:n
        E[i] = real(A[i, i])
    end
    #####* ascending order (insertion sort, n is small), columns of V alongside
    @inbounds for i in 2:n
        j = i
        while j > 1 && E[j - 1] > E[j]
            E[j - 1], E[j] = E[j], E[j - 1]
            for r in 1:n
                V[r, j - 1], V[r, j] = V[r, j], V[r, j - 1]
            end
            j -= 1
        end
    end
    return E, V
end

#####* a reusable diagonalization buffer
struct EigBuffer
    H::Matrix{ComplexF64}
    U::Matrix{ComplexF64}
    E::Vector{Float64}
end
EigBuffer(n::Int) = EigBuffer(zeros(ComplexF64, n, n), zeros(ComplexF64, n, n), zeros(n))

#####* the eigen-decomposition of H(k), into `buf` (overwritten on the next call), as an Eigen2 view of it
function eigen2!(buf::EigBuffer, bm::BandModel, k)
    bloch_hamiltonian!(buf.H, bm, k)
    if size(buf.H, 1) <= 16
        jacobi_eigen!(buf.H, buf.U, buf.E)
    else
        F = eigen!(Hermitian(buf.H))
        buf.E .= F.values
        buf.U .= F.vectors
    end
    return Eigen2(buf.E, buf.U)
end

eigen2(bm::BandModel, k) = (b = eigen2!(EigBuffer(norbitals(bm)), bm, k); Eigen2(copy(b.E), copy(b.U)))

"""
    band_eigen(bm, ks) -> Vector

The eigen-decompositions of H(k) at every `k`, threaded: pass them as `eig_k` to the bubbles to
diagonalize a grid once for a whole chemical-potential scan.
"""
function band_eigen(bm::BandModel, ks::AbstractVector)
    out = Vector{Eigen2}(undef, length(ks))
    chunked(length(ks)) do chunk
        eb = EigBuffer(norbitals(bm))
        for i in chunk
            e = eigen2!(eb, bm, ks[i])
            out[i] = Eigen2(copy(e.E), copy(e.U))
        end
    end
    return out
end

#####* where k - q lands on the grid, for q a grid point (periodic gauge: H(k + G) = H(k))
function shifted_index(n::Int, iq::Int, ik::Int)
    i_q, j_q = divrem(iq - 1, n)
    i, j = divrem(ik - 1, n)
    return mod(i - i_q, n) * n + mod(j - j_q, n) + 1
end

"""
    static_bubble(bm, qs, k_size; beta, mu, ops, on_grid = false, eig_k = nothing)
        -> Vector{Matrix{ComplexF64}}

The static bubble ``\\mathrm{raw}_{ij}(q)`` (see the module notes) at each `q`, summed over the
`k_size × k_size` grid [`k_grid`](@ref), for the operators `ops` ([`channel_operators`](@ref)).
`on_grid = true` declares that `qs` IS that grid, in its order, so k - q is looked up instead of
diagonalized. Threaded over q.
"""
function static_bubble(bm::BandModel, qs::AbstractVector, k_size::Integer;
                       beta::Real, mu::Real, ops, on_grid::Bool = false,
                       eig_k = nothing)
    ks = k_grid(bm.primitives, k_size)
    on_grid && length(qs) != length(ks) &&
        error("static_bubble: on_grid needs the q list to be the $(k_size)x$(k_size) grid itself.")
    eigs = eig_k === nothing ? band_eigen(bm, ks) : eig_k
    opset = OpSet(ops)
    n, d, nk = norbitals(bm), length(opset), length(ks)
    out = Vector{Matrix{ComplexF64}}(undef, length(qs))
    chunked(length(qs)) do chunk
        ws = Workspace(n, d)
        eb = EigBuffer(n)
        kq = zeros(2)
        for iq in chunk
            raw = zeros(ComplexF64, d, d)
            for ik in 1:nk
                e1 = eigs[ik]
                if on_grid
                    e2 = eigs[shifted_index(k_size, iq, ik)]
                else
                    kq[1] = ks[ik][1] - qs[iq][1]; kq[2] = ks[ik][2] - qs[iq][2]
                    e2 = eigen2!(eb, bm, kq)
                end
                matrix_elements!(ws, opset, e1.U, e2.U)
                @inbounds for nn in 1:n, m in 1:n
                    ws.w[m + (nn - 1) * n] = lindhard_weight(e1.E[m], e2.E[nn], beta, mu)
                end
                accumulate!(raw, ws)
            end
            out[iq] = raw ./ nk
        end
    end
    return out
end

#####* e^{xi tau} f(xi) and e^{-xi tau} (1 - f(xi)) for 0 < tau < beta, overflow-free, with tauc = beta - tau
occ_hole_side(xi, tau, tauc, beta) = xi >= 0 ? exp(-xi * tauc) / (1 + exp(-beta * xi)) : exp(xi * tau) / (1 + exp(beta * xi))
occ_particle_side(xi, tau, tauc, beta) = xi >= 0 ? exp(-xi * tau) / (1 + exp(-beta * xi)) : exp(xi * tauc) / (1 + exp(beta * xi))

#####* M = U diag(g) U', in place
function dressed!(M::AbstractMatrix, U::Matrix{ComplexF64}, g::AbstractVector, buf::Matrix{ComplexF64})
    @inbounds for c in axes(U, 2), r in axes(U, 1)
        buf[r, c] = U[r, c] * g[c]
    end
    small_mul!(M, buf, U; adjB = true)
    return M
end

"""
    static_bubble_dlr(bm, k_size; beta, mu, ops, euv = 20.0, rtol = 1e-12, matsubara = 0,
                      eig_k = nothing) -> Vector{Matrix{ComplexF64}}

The same bubble as [`static_bubble`](@ref) on its own `k_size × k_size` grid (in [`k_grid`](@ref)
order), by the imaginary-time route:

    raw_ij(q) = int_0^beta dtau sum_R e^{-iq.R} Tr[P(R, tau) O_j Q(-R, tau) O_i],
    P(k, tau) = U diag(e^{xi tau} f) U',   Q(k, tau) = U diag(e^{-xi tau} (1 - f)) U',

with P and Q carried to real space by FFT and the tau integral done on a discrete Lehmann
representation ([`dlr_grid`](@ref), cutoff `euv` which must exceed the band span, accuracy `rtol`).
O(N_k log N_k) per DLR node against the Lindhard sum's O(N_k^2). `matsubara = n` returns the
bosonic frequency ``\\Omega_n = 2\\pi n/\\beta`` instead of 0.
"""
function static_bubble_dlr(bm::BandModel, k_size::Integer; beta::Real, mu::Real, ops,
                           euv::Real = 20.0, rtol::Real = 1e-12, matsubara::Integer = 0,
                           eig_k = nothing)
    ks = k_grid(bm.primitives, k_size)
    eigs = eig_k === nothing ? band_eigen(bm, ks) : eig_k
    span = maximum(maximum(e.E) for e in eigs) - minimum(minimum(e.E) for e in eigs)
    euv > span ||
        @warn "static_bubble_dlr: euv = $euv is below the band span $(round(span, sigdigits = 4)); " *
              "the DLR cannot represent the bubble's particle-hole energies. Raise w_max."
    grid = dlr_grid(beta, euv; rtol = rtol)
    weights = matsubara_weights(grid, matsubara)
    opset = OpSet(ops)
    n, d, L, nk = norbitals(bm), length(opset), k_size, length(ks)

    P = zeros(ComplexF64, n, n, L, L)
    Q = zeros(ComplexF64, n, n, L, L)
    C = zeros(ComplexF64, d, d, L, L)
    total = zeros(ComplexF64, d, d, L, L)
    FFTW.set_num_threads(Threads.nthreads())
    planP = plan_bfft!(P, (3, 4))
    planQ = plan_bfft!(Q, (3, 4))
    planC = plan_fft!(C, (3, 4))
    norm2 = 1 / nk^2
    for (tau, tauc, wk) in zip(grid.tau, grid.tauc, weights)
        chunked(nk) do chunk
            buf = zeros(ComplexF64, n, n)
            g = zeros(n)
            for ik in chunk
                i, j = divrem(ik - 1, L)
                e = eigs[ik]
                @inbounds for m in 1:n
                    g[m] = occ_hole_side(e.E[m] - mu, tau, tauc, beta)
                end
                dressed!(view(P, :, :, j + 1, i + 1), e.U, g, buf)
                @inbounds for m in 1:n
                    g[m] = occ_particle_side(e.E[m] - mu, tau, tauc, beta)
                end
                dressed!(view(Q, :, :, j + 1, i + 1), e.U, g, buf)
            end
        end
        planP * P
        planQ * Q
        chunked(nk) do chunk
            B = zeros(ComplexF64, n, n * d)
            D = zeros(ComplexF64, n, n * d)
            for iR in chunk
                r1, r2 = divrem(iR - 1, L)
                PR = view(P, :, :, r2 + 1, r1 + 1)
                QmR = view(Q, :, :, mod(-r2, L) + 1, mod(-r1, L) + 1)
                if opset.sparse
                    #####* Tr[P O_j Q O_i] = sum over entries (a, b, v) of O_j and (c, x, u) of O_i of v u P[x, a] Q[b, c]
                    @inbounds for jj in 1:d, ii in 1:d
                        acc = zero(ComplexF64)
                        for t in eachindex(opset.vals[jj]), s in eachindex(opset.vals[ii])
                            acc += opset.vals[jj][t] * opset.vals[ii][s] *
                                   PR[opset.cols[ii][s], opset.rows[jj][t]] * QmR[opset.cols[jj][t], opset.rows[ii][s]]
                        end
                        C[ii, jj, r2 + 1, r1 + 1] = acc * norm2
                    end
                else
                    small_mul!(B, PR, opset.cat)
                    small_mul!(D, QmR, opset.cat)
                    @inbounds for jj in 1:d, ii in 1:d
                        acc = zero(ComplexF64)
                        for a in 1:n, b in 1:n
                            acc += B[a, (jj - 1) * n + b] * D[b, (ii - 1) * n + a]
                        end
                        C[ii, jj, r2 + 1, r1 + 1] = acc * norm2
                    end
                end
            end
        end
        planC * C
        total .+= wk .* C
    end
    return [total[:, :, j + 1, i + 1] for i in 0:L-1 for j in 0:L-1]
end

transposed_ops(ops) = [collect(transpose(O)) for O in ops]

"""
    stored_bubble(bm, qs, k_size; beta, mu, ops, method = :lindhard, on_grid = false,
                  euv = 20.0, rtol = 1e-12) -> Vector{Matrix{ComplexF64}}

The bubble in the convention the bare stage writes to disk (see the module notes):
`transpose(raw)` with the operators transposed. `method = :dlr` needs `on_grid = true`.
"""
function stored_bubble(bm::BandModel, qs::AbstractVector, k_size::Integer; beta::Real, mu::Real, ops,
                       method::Symbol = :lindhard, on_grid::Bool = false, euv::Real = 20.0,
                       rtol::Real = 1e-12, eig_k = nothing)
    opsT = transposed_ops(ops)
    raw = if method == :lindhard
        static_bubble(bm, qs, k_size; beta = beta, mu = mu, ops = opsT, on_grid = on_grid, eig_k = eig_k)
    elseif method == :dlr
        on_grid || error("stored_bubble: the DLR bubble is computed on its own k grid; pass on_grid = true.")
        static_bubble_dlr(bm, k_size; beta = beta, mu = mu, ops = opsT, euv = euv, rtol = rtol, eig_k = eig_k)
    else
        error("stored_bubble: method must be :lindhard or :dlr, got $(repr(method)).")
    end
    return [collect(transpose(r)) for r in raw]
end

#####* the SU(2) spinless response (d x d, contracted with M_p, NOT yet times 1/2) placed into the
#####* four diagonal channel blocks of a 4d x 4d matrix: each is 2 (raw / 2).
function su2_blocks(raw_spinless::AbstractMatrix)
    d = size(raw_spinless, 1)
    out = zeros(ComplexF64, 4d, 4d)
    t = raw_spinless
    for a in 1:4, j in 1:d, i in 1:d
        out[4(i - 1) + a, 4(j - 1) + a] = t[i, j]
    end
    return out
end

"""
    particle_hole_edges(bm, qs, k_size; beta, mu, weight_tol = 1e-8) -> Vector{Float64}

The upper edge of the particle-hole continuum at each `q`: the largest ``|E_m(k) - E_n(k-q)|`` over
the band pairs whose occupation difference exceeds `weight_tol`, on the same `k_size × k_size`
grid and (k, k - q) pairs as [`dynamic_bubble`](@ref). This is the support of
``\\mathrm{Im}\\,\\chi_0(q, \\omega)`` before the ``\\eta`` broadening, so a collective mode above it is
undamped; reading the edge off the broadened bubble instead is biased upward by the Lorentzian tails.
"""
function particle_hole_edges(bm::BandModel, qs::AbstractVector, k_size::Integer;
                             beta::Real, mu::Real, weight_tol::Real = 1e-8)
    ks = k_grid(bm.primitives, k_size)
    eigs = band_eigen(bm, ks)
    n = norbitals(bm)
    out = fill(NaN, length(qs))
    chunked(length(qs)) do chunk
        eb = EigBuffer(n)
        kq = zeros(2)
        for iq in chunk
            edge = -Inf
            for ik in eachindex(ks)
                e1 = eigs[ik]
                kq[1] = ks[ik][1] - qs[iq][1]; kq[2] = ks[ik][2] - qs[iq][2]
                e2 = eigen2!(eb, bm, kq)
                @inbounds for nn in 1:n, m in 1:n
                    abs(fermi(e2.E[nn], beta, mu) - fermi(e1.E[m], beta, mu)) > weight_tol || continue
                    edge = max(edge, abs(e1.E[m] - e2.E[nn]))
                end
            end
            isfinite(edge) && (out[iq] = edge)
        end
    end
    return out
end

"""
    dynamic_bubble(bm, qs, omegas, k_size; beta, mu, eta, basis = nothing, nbins = 4000,
                   spin = :auto) -> Matrix{Matrix{ComplexF64}}

The retarded bare bubble ``\\chi_0(q, \\omega + i\\eta)``, as `chi[iq, iw]`, in the convention
`combine_chis` gives the static bubble (it reduces to it at ω = 0; see the module notes),
each a `4d × 4d` matrix with the channel (N, X, Y, Z) innermost, `d` the number of basis operators
(the orbitals, or the `basis` matrices). The k-sum runs over the `k_size × k_size` grid; the `qs`
are arbitrary (a path), Cartesian.

`nbins = nothing` sums every pole exactly, ``\\sum (f_m - f_n) X X^\\dagger / (E_n - E_m - \\omega - i\\eta)``,
which costs `length(omegas)` per band pair. Otherwise the pole weights are first spread linearly
onto `nbins` energies spanning the particle-hole range and the frequency sum is one matrix product:
the error is O(bin width / eta), so keep the bin width a small fraction of `eta`.

`spin = :auto` uses the spinless bubble for an SU(2) model ([`band_spin_symmetry`](@ref)), whose
four channels are then identical; `spin = :full` always sums the spinful one.
"""
function dynamic_bubble(bm::BandModel, qs::AbstractVector, omegas::AbstractVector, k_size::Integer;
                        beta::Real, mu::Real, eta::Real, basis = nothing,
                        nbins::Union{Nothing, Integer} = 4000, spin::Symbol = :auto)
    spin in (:auto, :full) || error("dynamic_bubble: spin must be :auto or :full, got $(repr(spin)).")
    eta > 0 || error("dynamic_bubble: eta must be positive.")
    factorized = spin == :auto && band_spin_symmetry(bm) == "SU2"
    model = factorized ? spinless_model(bm) : bm
    n_orb = factorized ? norbitals(model) : norbitals(model) ÷ 2
    ops = OpSet(transposed_ops(channel_operators(n_orb; basis = basis, spinful = !factorized)))
    d = length(ops)
    n = norbitals(model)
    ks = k_grid(model.primitives, k_size)
    nk = length(ks)
    eigs = band_eigen(model, ks)
    Es = reduce(vcat, [e.E for e in eigs])
    span = maximum(Es) - minimum(Es)
    Omega_max = span + 10eta

    bins = nbins === nothing ? nothing : collect(range(-Omega_max, Omega_max, length = nbins))
    kernel = bins === nothing ? nothing :
             [1 / (W - w - im * eta) for w in omegas, W in bins]       # (nw, nbins)

    out = Matrix{Matrix{ComplexF64}}(undef, length(qs), length(omegas))
    dW = bins === nothing ? 0.0 : bins[2] - bins[1]
    kernelT = bins === nothing ? nothing : collect(transpose(kernel))           # (nbins, nw)
    chunked(length(qs)) do chunk
        ws = Workspace(n, d)
        eb = EigBuffer(n)
        kq = zeros(2)
        acc = bins === nothing ? zeros(ComplexF64, d * d, length(omegas)) : zeros(ComplexF64, d * d, length(bins))
        for iq in chunk
            fill!(acc, 0)
            q = qs[iq]
            for ik in 1:nk
                e1 = eigs[ik]
                kq[1] = ks[ik][1] - q[1]; kq[2] = ks[ik][2] - q[2]       # the same (k, k - q) pairs as the static bubble
                e2 = eigen2!(eb, model, kq)
                X = matrix_elements!(ws, ops, e1.U, e2.U)
                @inbounds for nn in 1:n, m in 1:n
                    idx = m + (nn - 1) * n
                    df = fermi(e2.E[nn], beta, mu) - fermi(e1.E[m], beta, mu)
                    df == 0 && continue
                    Δ = e1.E[m] - e2.E[nn]
                    if bins === nothing
                        for (iw, w) in enumerate(omegas)
                            wt = df / (Δ - w - im * eta)
                            for j in 1:d, i in 1:d
                                acc[i + (j - 1) * d, iw] += wt * conj(X[idx, i]) * X[idx, j]
                            end
                        end
                    else
                        x = (Δ + Omega_max) / dW
                        b = clamp(floor(Int, x), 0, nbins - 2)
                        frac = x - b
                        for j in 1:d, i in 1:d
                            v = df * conj(X[idx, i]) * X[idx, j]
                            acc[i + (j - 1) * d, b + 1] += (1 - frac) * v
                            acc[i + (j - 1) * d, b + 2] += frac * v
                        end
                    end
                end
            end
            R = bins === nothing ? acc : acc * kernelT                          # (d^2, nw)
            for iw in eachindex(omegas)
                raw = reshape(R[:, iw], d, d) ./ nk
                out[iq, iw] = factorized ? su2_blocks(raw) : 2 .* raw
            end
        end
    end
    return out
end

end # module
