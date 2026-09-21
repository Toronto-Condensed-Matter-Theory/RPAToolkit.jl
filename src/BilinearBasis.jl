module BilinearBasis

using LinearAlgebra

export bilinear_closure, density_seed, same_site_seed, lattice_point_group_2d,
       orbital_rep_from_bonds, canonical_frame, conjugation_matrix,
       herm_to_vec, vec_to_herm

# Julia port of src/Bare/bilinear_basis.py. Both exist because the basis is
# needed at two different points in the pipeline: the interaction builder runs
# before the _triqs.npz is written, so it cannot read a basis computed in
# Python, while run_bare needs one to contract the bubble with.
# Both suites assert the same canonical closure dimensions (1, 2, 3, 6 for a
# single orbital, a permuting pair, an E doublet and ZrNCl_4), so a divergence
# between them shows up as a test failure on one side.
#
# The two implementations must agree on the BASIS, not only on its dimension:
# the bubble is contracted onto the Python basis and the vertex is projected
# onto the Julia one, and coefficients in two different frames of the same span
# are not comparable. An SVD frame is not enough for that -- the span fixes it
# only up to an orthogonal rotation, and degenerate singular values make even a
# single implementation's choice arbitrary. canonical_frame below fixes the
# frame from the span alone, so both sides land on the same matrices.
# run_RPA.jl still measures the residual rotation between the two and applies
# it, so a future divergence is corrected rather than silently mis-projected.

"""Coordinates of a Hermitian matrix in a real orthonormal frame."""
function herm_to_vec(M::AbstractMatrix)
    n = size(M, 1)
    m = (n * (n - 1)) ÷ 2
    out = zeros(Float64, n * n)
    for i in 1:n
        out[i] = real(M[i, i])
    end
    k = n
    for j in 2:n, i in 1:(j - 1)
        k += 1
        out[k] = sqrt(2) * real(M[i, j])
        out[k + m] = sqrt(2) * imag(M[i, j])
    end
    return out
end

"""Inverse of `herm_to_vec`."""
function vec_to_herm(v::AbstractVector, n::Int)
    M = zeros(ComplexF64, n, n)
    for i in 1:n
        M[i, i] = v[i]
    end
    m = (n * (n - 1)) ÷ 2
    k = n
    for j in 2:n, i in 1:(j - 1)
        k += 1
        re = v[k] / sqrt(2)
        imag_part = v[k + m] / sqrt(2)
        M[i, j] = re + im * imag_part
        M[j, i] = re - im * imag_part
    end
    return M
end

"""The per-orbital density operators: the basis used before the closure."""
function density_seed(n_orb::Int)
    out = Matrix{ComplexF64}[]
    for i in 1:n_orb
        M = zeros(ComplexF64, n_orb, n_orb)
        M[i, i] = 1
        push!(out, M)
    end
    return out
end

"""
    same_site_seed(site_of) -> Vector{Matrix{ComplexF64}}

Both Hermitian components of every ON-SITE off-diagonal bilinear: the support
of the Kanamori abab (U') and abba (J') terms.

A closure seeded with the densities alone spans what the densities generate,
which for ZrNCl_4 is 6 of 16 bilinears -- enough for chi to be covariant, but
NOT enough to hold the interaction: U' and J' couple through c^dag_a c_b with
a != b on the same atom, and a vertex whose support leaves the span has no
representation in it (BilinearVertex.uncovered_bilinears measures exactly
that). Seeding with these takes ZrNCl_4 from 6 to 8, the extra two being the
A2 orbital-current (L_z-like) bilinear on each Zr.

Nothing is added for a model with one orbital per site -- there are no
same-site pairs -- so square, triangular and honeycomb keep their old basis.
"""
function same_site_seed(site_of::AbstractVector{Int})
    n = length(site_of)
    out = Matrix{ComplexF64}[]
    for b in 2:n, a in 1:(b - 1)
        site_of[a] == site_of[b] || continue
        X = zeros(ComplexF64, n, n); X[a, b] = 1 / sqrt(2); X[b, a] = 1 / sqrt(2)
        Y = zeros(ComplexF64, n, n); Y[a, b] = im / sqrt(2); Y[b, a] = -im / sqrt(2)
        push!(out, X); push!(out, Y)
    end
    return out
end

"""
    canonical_frame(V; tol) -> Matrix{Float64}

An orthonormal frame of the column span of `V` that depends on the SPAN ALONE.

`V`'s own columns do not: they come out of an SVD, and within a degenerate
singular value the choice is arbitrary, so two implementations of the same
closure can return different -- equally valid -- frames of the same subspace.
Coefficients are frame-dependent, so that would silently mis-project the
vertex against the bubble.

The projector P = V V' is a function of the span only. Gram-Schmidt on its
columns in index order, taking each one whose residual survives `tol`, is
therefore reproducible: the residual of column j is the j-th diagonal entry of
the remaining projector, which is non-negative, so even the sign is fixed with
no extra convention.
"""
function canonical_frame(V::AbstractMatrix{<:Real}; tol::Float64 = 1e-7)
    n, d = size(V)
    P = V * transpose(V)
    Q = zeros(Float64, n, d)
    filled = 0
    for j in 1:n
        filled == d && break
        v = P[:, j]
        for _ in 1:2, k in 1:filled          # twice: classical GS loses orthogonality once
            v .-= dot(view(Q, :, k), v) .* view(Q, :, k)
        end
        nv = norm(v)
        nv > tol || continue
        filled += 1
        Q[:, filled] = v ./ nv
    end
    filled == d || error("canonical_frame: recovered $filled of $d directions; the span is " *
                         "degenerate at the tolerance $tol.")
    return Q
end

"""
    conjugation_matrix(basis) -> Matrix{Float64}

`C[p, q]` with `conj(M_p) = sum_q C[p, q] M_q`: how complex conjugation acts on
the basis.

Needed by the pairing kernel. A Cooper pair is (k, -k), and without SOC the
orbital amplitudes at -k are the conjugates of those at k, so the partner
electron couples to `conj(M_q)` where the first couples to `M_p`
(Pairing_Theory.tex, "Band projection on a closed bilinear basis"). For the
per-orbital densities conj(M) = M and C is the identity, which is why the
per-orbital code never had to know about it; a closed basis contains
imaginary-antisymmetric (orbital-current) members for which conj(M) = -M.

Errors unless the span is closed under conjugation, which it is whenever D(g)
can be taken real -- the case for any Wannier model with real hoppings.
"""
function conjugation_matrix(basis::AbstractVector{<:AbstractMatrix}; tol::Float64 = 1e-8)
    d = length(basis)
    C = Matrix{Float64}(undef, d, d)
    for p in 1:d
        Mc = conj(basis[p])
        for q in 1:d
            C[p, q] = real(tr(basis[q]' * Mc))
        end
        resid = norm(Mc - sum(C[p, q] * basis[q] for q in 1:d))
        resid < tol || error("conjugation_matrix: conj(M_$p) leaves the span by $resid. The " *
                             "bilinear basis must be closed under complex conjugation for the " *
                             "Cooper-channel projection to be defined.")
    end
    return C
end

"""
    bilinear_closure(Ds, n_orb; extra_seed, tol) -> Vector{Matrix{ComplexF64}}

The span of the densities (plus `extra_seed`) closed under M -> U'MU for every
U in `Ds`, as an orthonormal Hermitian basis.

A per-orbital operator set is closed under the point group only when D(g)
PERMUTES orbitals; for an E doublet (D rotates) n_i maps into off-diagonal
bilinears that set does not contain, and no chi built on it can be covariant.
Worked on matrices rather than through a vec/kron identity so no column-major
convention enters, and in the real vector space of Hermitian matrices so the
result is Hermitian by construction.

Seed with more than the densities when the interaction couples to more: the
basis has to contain the interaction's support, which
BilinearVertex.uncovered_bilinears measures.
"""
function bilinear_closure(Ds::AbstractVector{<:AbstractMatrix}, n_orb::Int;
                          extra_seed = nothing, tol::Float64 = 1e-9,
                          max_iter::Int = 32)
    seed = density_seed(n_orb)
    extra_seed === nothing || (seed = vcat(seed, collect(extra_seed)))
    isempty(seed) && return Matrix{ComplexF64}[]

    V = reduce(hcat, [herm_to_vec(0.5 * (M + M')) for M in seed])
    for _ in 1:max_iter
        mats = [vec_to_herm(view(V, :, k), n_orb) for k in axes(V, 2)]
        cols = Matrix{Float64}[Matrix{Float64}(V)]
        for U in Ds
            push!(cols, reduce(hcat, [herm_to_vec(U' * M * U) for M in mats]))
        end
        F = svd(reduce(hcat, cols))
        rank = count(s -> s > tol * max(1.0, F.S[1]), F.S)
        rank == size(V, 2) && break
        V = F.U[:, 1:rank]
    end
    # Canonicalised, so this agrees matrix-for-matrix with the Python closure
    # rather than only subspace-for-subspace. See the note at the top.
    Vc = canonical_frame(Matrix{Float64}(V))
    return [vec_to_herm(view(Vc, :, k), n_orb) for k in axes(Vc, 2)]
end

"""
    lattice_point_group_2d(a1, a2; tol) -> Vector{Matrix{Float64}}

The Cartesian operations mapping the lattice onto itself, derived from the
metric rather than assumed, so no lattice label or orientation is needed.
"""
function lattice_point_group_2d(a1::AbstractVector, a2::AbstractVector; tol::Float64 = 1e-9)
    A = hcat(Float64.(a1[1:2]), Float64.(a2[1:2]))
    G = A' * A
    scale = max(1.0, maximum(abs, G))
    ops = Matrix{Float64}[]
    for a in -2:2, b in -2:2, c in -2:2, d in -2:2
        M = Float64[a b; c d]
        abs(round(det(M))) == 1 || continue
        maximum(abs.(M' * G * M - G)) < tol * scale || continue
        push!(ops, A * M * inv(A))
    end
    return ops
end

"""
    orbital_rep_from_bonds(bonds, Fs, g; tol) -> (D, conditioning, residual)

D(g) from the null space of F(g d) = D F(d) D'.

`bonds[i]` is the Cartesian bond vector d = R + tau_b - tau_a whose orbital
matrix is `Fs[i]`. Bond vectors rather than lattice vectors R are what make a
single k-independent D(g) exist, including for the operations that exchange
sublattices. There is no such D for t(R), which is also why D cannot be
measured from H(k): TBLattice's fourier works in the lattice gauge.
"""
function orbital_rep_from_bonds(bonds::AbstractVector, Fs::AbstractVector{<:AbstractMatrix},
                                g::AbstractMatrix; tol::Float64 = 1e-6)
    n = size(Fs[1], 1)
    I_n = Matrix{ComplexF64}(I, n, n)
    zero_n = zeros(ComplexF64, n, n)

    function index_of(dv)
        best, bi = Inf, -1
        for (i, b) in enumerate(bonds)
            dist = maximum(abs.(b .- dv))
            if dist < best
                best, bi = dist, i
            end
        end
        return best < tol ? bi : -1
    end

    rows = Matrix{ComplexF64}[]
    for (dv, F) in zip(bonds, Fs)
        j = index_of(g * dv)
        push!(rows, kron(I_n, j > 0 ? Fs[j] : zero_n) - kron(transpose(F), I_n))
    end
    dec = svd(reduce(vcat, rows))
    # No transpose here: numpy reshapes row-major and then transposes, Julia
    # reshapes column-major, so the two agree with the transpose dropped.
    D = Matrix(reshape(conj(dec.V[:, end]), n, n))
    Usvd, _, Vsvd = svd(D)                # nearest unitary: U * V'
    D = Usvd * Vsvd'
    resid = 0.0
    for (dv, F) in zip(bonds, Fs)
        j = index_of(g * dv)
        target = j > 0 ? Fs[j] : zero_n
        resid = max(resid, maximum(abs.(target - D * F * D')))
    end
    return D, dec.S[end] / dec.S[1], resid
end

end # module
