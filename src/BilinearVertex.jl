module BilinearVertex

using LinearAlgebra

export kanamori_vertex, project_vertex, channel_blocks, uncovered_bilinears,
       basis_rotation, rotate_channel_blocks

"""
    kanamori_vertex(n_orb, U, Up, J, Jp) -> (Us, Uc)

The on-site spin and charge vertices, as rank-4 orbital tensors, in the
conventions of Graser, Maier, Hirschfeld & Scalapino, NJP 11, 025016 (2009) --
the same reference gap_equation_utils.jl uses for Gamma_s/Gamma_t.

    component   U^(s)        U^(c)          bilinears
    aaaa        U            U              diagonal
    aabb        J            2U' - J        diagonal
    abab        U'           -U' + 2J       OFF-diagonal
    abba        J'           J'             OFF-diagonal

The last two need bilinears that are off-diagonal in the orbital index. They
have no slot in a per-orbital basis, which is why they were absent; on a
symmetry-closed basis they do, so the full Kanamori vertex can be built.

With U' = U - 2J and J' = J the interaction is orbital-rotation invariant. That
is the combination `interorbital: 0.5, hund: 0.25` already implies, so omitting
abab/abba left the vertex inconsistent with its own parameters.

`site_of` maps each orbital to its atom. U', J and J' are INTRA-ATOMIC, so they
apply only to pairs on the same atom; without the mask a multi-atom cell (ZrNCl_4
has two Zr, two orbitals each) would get a spurious U' between orbitals on
different atoms. The per-orbital builder restricts this with
`sublattice(a) == sublattice(b)`; this reproduces that.
"""
function kanamori_vertex(n_orb::Int, U::Real, Up::Real, J::Real, Jp::Real;
                         site_of::Union{AbstractVector{Int}, Nothing} = nothing)
    site_of === nothing || length(site_of) == n_orb ||
        error("kanamori_vertex: site_of has $(length(site_of)) entries for $n_orb orbitals.")
    same_site(a, b) = site_of === nothing ? true : site_of[a] == site_of[b]

    Us = zeros(Float64, n_orb, n_orb, n_orb, n_orb)
    Uc = zeros(Float64, n_orb, n_orb, n_orb, n_orb)
    for a in 1:n_orb, b in 1:n_orb
        if a == b
            Us[a, a, a, a] = U
            Uc[a, a, a, a] = U
        elseif same_site(a, b)
            Us[a, a, b, b] = J;   Uc[a, a, b, b] = 2 * Up - J
            Us[a, b, a, b] = Up;  Uc[a, b, a, b] = -Up + 2 * J
            Us[a, b, b, a] = Jp;  Uc[a, b, b, a] = Jp
        end
    end
    return Us, Uc
end

"""
    project_vertex(V4, basis) -> Matrix

Project a rank-4 orbital vertex onto a basis of Hermitian orbital bilinears:

    V[p, q] = sum_{l1 l2 l3 l4} conj(M_p[l1, l2]) V4[l1, l2, l3, l4] M_q[l3, l4]

The index pairing is (l1, l2) against the first operator and (l3, l4) against the
second. That is not a free choice: it is the pairing for which the diagonal basis
M_p = E_pp reproduces V[p, q] = V4[p, p, q, q], i.e. the aaaa and aabb entries of
the table above -- the matrices the per-orbital builder already uses. The other
pairing would return the abba component there instead.
"""
function project_vertex(V4::AbstractArray{<:Real,4}, basis::AbstractVector{<:AbstractMatrix})
    d = length(basis)
    n = size(V4, 1)
    all(size(M) == (n, n) for M in basis) ||
        error("project_vertex: basis matrices must be $(n)x$(n) to match the vertex.")
    out = zeros(ComplexF64, d, d)
    for q in 1:d, p in 1:d
        acc = zero(ComplexF64)
        Mp, Mq = basis[p], basis[q]
        for l4 in 1:n, l3 in 1:n, l2 in 1:n, l1 in 1:n
            v = V4[l1, l2, l3, l4]
            v == 0 && continue
            acc += conj(Mp[l1, l2]) * v * Mq[l3, l4]
        end
        out[p, q] = acc
    end
    return out
end

"""
    channel_blocks(Vs, Vc) -> Matrix

Assemble the spin and charge projections into the (N, X, Y, Z) bilinear layout
this code uses: one 4x4 channel block per basis pair, with the channel index
INNERMOST, so entry (p, q) sits at rows/cols 4*(p-1) + channel.

The slots follow the existing convention (see ZrNCl_4_interactions_rusty.jl):
NN carries U^(s) and ZZ carries -U^(c) -- the charge vertex is stored negated.
X and Y stay zero, as they are in the per-orbital matrices.
"""
function channel_blocks(Vs::AbstractMatrix, Vc::AbstractMatrix)
    d = size(Vs, 1)
    size(Vc) == (d, d) || error("channel_blocks: spin and charge blocks differ in size.")
    out = zeros(ComplexF64, 4 * d, 4 * d)
    for q in 1:d, p in 1:d
        out[4 * (p - 1) + 1, 4 * (q - 1) + 1] = Vs[p, q]
        out[4 * (p - 1) + 4, 4 * (q - 1) + 4] = -Vc[p, q]
    end
    return out
end

"""
    basis_rotation(target, source; tol) -> Matrix{Float64}

`O[p, q]` with `source[q] = sum_p O[p, q] target[p]`: the real orthogonal
change of frame between two orthonormal Hermitian bases of the SAME span.

The bubble is contracted onto a basis built in Python and the vertex is
projected onto one built in Julia. Both closures are canonicalised so they
agree, but "agree" has to be measured rather than assumed: coefficients in two
different frames of one subspace are not comparable, and nothing downstream
would notice. This returns the rotation that makes them comparable, and errors
when the two spans differ -- `O` is orthogonal exactly then.

The usual cause of a failure is the two sides seeding the closure differently
(densities against densities + pair hopping), which changes the span, not the
frame.
"""
function basis_rotation(target::AbstractVector{<:AbstractMatrix},
                        source::AbstractVector{<:AbstractMatrix}; tol::Float64 = 1e-8)
    d = length(target)
    length(source) == d ||
        error("basis_rotation: bases have $(length(target)) and $(length(source)) members; " *
              "they cannot span the same space. The two closures were seeded differently.")
    O = Matrix{Float64}(undef, d, d)
    for q in 1:d, p in 1:d
        O[p, q] = real(tr(target[p]' * source[q]))
    end
    resid = maximum(abs.(O * transpose(O) - I))
    resid < tol ||
        error("basis_rotation: the overlap matrix is not orthogonal (max|O O' - I| = $resid). " *
              "The two bilinear bases span DIFFERENT subspaces, so the vertex cannot be " *
              "expressed in the bubble's basis at all -- they were built from different seeds, " *
              "different hoppings or different site groupings.")
    return O
end

"""
    rotate_channel_blocks(M, O) -> Matrix

Carry a `4d x 4d` channel-block matrix from one bilinear frame to another.

`channel_blocks` puts the channel index INNERMOST, so the frame rotation acts
as `kron(O, I_4)`: channels do not mix, bilinears do.
"""
function rotate_channel_blocks(M::AbstractMatrix, O::AbstractMatrix)
    d = size(O, 1)
    size(M) == (4 * d, 4 * d) ||
        error("rotate_channel_blocks: matrix is $(size(M, 1))x$(size(M, 2)) but the rotation " *
              "is $(d)x$(d), which needs $(4 * d)x$(4 * d).")
    R = kron(O, Matrix{Float64}(I, 4, 4))
    return R * M * transpose(R)
end

"""
    uncovered_bilinears(V4, basis) -> Float64

How far the bilinears `V4` actually couples to sit from the basis's span,
relative to the span's own scale. Zero means the basis can hold the whole
vertex; nonzero means the closure was seeded too narrowly.

This is deliberately NOT a projection-and-reconstruct residual on the raw
coefficient array. A Hermitian basis spans only the Hermitian part of each
index slice, while individual Graser components (abab, say) have non-Hermitian
slices, so reconstructing the array is not the right question and does not even
decrease monotonically as the basis grows. The right question is the one asked
here: for every index pair (l1, l2) the vertex couples through, is that
bilinear's Hermitian part inside the span? That is monotonic in the basis by
construction, and it is exactly the condition for the interaction to have a
representation.
"""
function uncovered_bilinears(V4::AbstractArray{<:Real,4},
                             basis::AbstractVector{<:AbstractMatrix})
    n = size(V4, 1)
    isempty(basis) && return 1.0
    # Real Hilbert-Schmidt projector onto the span.
    proj(M) = sum(real(tr(B' * M)) * B for B in basis)
    worst = 0.0
    for l2 in 1:n, l1 in 1:n
        # does this bilinear appear in the vertex at all?
        coupled = any(V4[l1, l2, l3, l4] != 0 || V4[l3, l4, l1, l2] != 0
                      for l3 in 1:n, l4 in 1:n)
        coupled || continue
        # BOTH Hermitian components are required, not just the Hermitian part of
        # E. A term like pair hopping couples the same NON-Hermitian bilinear
        # twice, and with B = X - iY a product of Bs needs X and Y separately;
        # keeping only X leaves that term unrepresentable even though X alone
        # looks sufficient index by index.
        E = zeros(ComplexF64, n, n); E[l1, l2] = 1
        for part in (0.5 * (E + E'), 0.5im * (E - E'))
            norm(part) == 0 && continue
            worst = max(worst, norm(part - proj(part)) / norm(part))
        end
    end
    return worst
end

end # module
