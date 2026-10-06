module Preprocess

using NPZ, JLD2
using LinearAlgebra

export dress_primitives, dress_reciprocal, combine_chis, CHI_ORIENTATION
export get_reciprocal_ks

const labels = Dict(0 => "chi_NN", 1 => "chi_XX", 2 => "chi_YY", 3 => "chi_ZZ", 4 => "chi_NN")

"""
The convention `combine_chis` returns: chi_ij(q) = <O_i(-q) O_j(q)>, with O_p(q) = sum_k c^dag_{k+q} M_p c_k
in TightBindingToolkit's H(k) = sum_R e^{ik.R} t(R), normalised as the PER-SPIN bubble chi0 that the
Graser vertices U^(s), U^(c) multiply. Consumers store this string beside the chis they write, so a
reader can refuse chis assembled under an older convention: before 2026-09-25 they were
untransposed, and the first fixed files ("O(-q)O(q)") were still half the per-spin bubble.
"""
const CHI_ORIENTATION = "O(-q)O(q), per-spin chi0"

#####* Return a vector of the three 3-d real-space primitive vectors from input dictionary
function dress_primitives(data::Dict ; entry::String = "primitives")::Vector{Vector{Float64}}
    primitives = data[entry]
    primitives = Vector{eltype(data[entry])}[eachrow(data[entry])...]
    # primitives[1] = [primitives[1]; 0.0]
    # primitives[2] = [primitives[2]; 0.0]
    # push!(primitives, [0.0, 0.0, 1.0])

    return primitives
end

#####* Returns a vector of the three 3d reciprocal vectors from input dictionary
function dress_reciprocal(data::Dict ; entry::String = "reciprocal")::Vector{Vector{Float64}}
    reciprocal = data[entry]
    reciprocal = Vector{eltype(data[entry])}[eachrow(data[entry])...]
    return reciprocal
end

#####* returns a vector of vectors of all momenta in the BZ in units of the reciprocal lattice vectors.
function get_reciprocal_ks(data::Dict ; entry::String = "ks")::Vector{Vector{Float64}}
    primitives = dress_primitives(data)
    ks = Vector{eltype(data[entry])}[eachrow(data[entry])...]

    k1s = dot.(ks, Ref(primitives[1]))
    k2s = dot.(ks, Ref(primitives[2]))
    k3s = dot.(ks, Ref(primitives[3]))

    Ks = hcat(k1s, k2s, k3s)
    Ks = Vector{eltype(Ks)}[eachrow(Ks)...]

    return Ks
end

#####* returns a vector of (sublattice x spin) susceptibility matrices where the vector corresponds to different momentas.
"""
    combine_chis(data; directions, subs = nothing, suffix = "")

Assemble the per-direction susceptibilities into one block matrix per q.

The block size is read from the stored data rather than from `subs`, so this
works unchanged whether run_bare contracted onto the per-orbital operators
(block size = number of orbitals) or onto the symmetry-closed bilinear basis
(block size = that basis's dimension, which is larger exactly when the point
group mixes orbitals). `subs`, if given, is only checked against the data: a
mismatch is reported, not silently accepted, because it means the interaction
matrices -- built at 4 * subs -- will not match this chi.

Stays block diagonal in the direction index, which is correct without
spin-orbit coupling: the charge and the three spin channels do not mix.

Each block is TRANSPOSED on the way in, which is what makes a multi-site kernel covariant.
run_bare.py contracts TRIQS's chi0_abcd with O_i on (a,b) and O_j on (c,d), which gives
<O_i(q) O_j(-q)>. The RPA ladder of H = 1/2 sum V_pq O_p(q) O_q(-q), and the Cooper form
factors <u_i|M_p|u_j> that the gap stage pairs with the vertex at q = k_i - k_j, need
`CHI_ORIENTATION` = <O_i(-q) O_j(q)> = transpose(chi(q)) = chi(-q). Measured 2026-09-25 on
ZrNCl_4 against an independent Lindhard sum: the stored chi is that transpose to 7.8e-10.
Used as stored, the Fermi-surface kernel keeps only a 2mm subgroup of 6mm (C3 broken at
3.9e-2 from bare chi, so every E doublet splits); transposed, all twelve operations hold to
~6e-11 at k_size 33, 63 and 93. The transpose changes nothing where chi(q) is symmetric, e.g.
one site per cell with inversion through it, which is why earlier models never showed it.
Spectra, and hence critical scales, are unaffected either way. It is a plain transpose, not
the adjoint.

Each block is also DOUBLED. run_bare.py contracts with S = M (x) sigma/2, so what it stores is
<S S> = chi0 / 2 in every channel, where chi0 is the per-spin bubble. The vertices are Graser's
U^(s), U^(c), which multiply chi0 itself: the single-orbital Stoner point is U chi0 = 1, and a
density-density V is screened as V / (1 + 2 V chi0). Fed chi0 / 2, the RPA put the Stoner point
at U chi0 = 2 and made every vertex exactly Gamma(s) = 2 Gamma_Graser(s / 2), i.e. the
fluctuations of half the nominal interaction (verified 2026-09-25: the ZrNCl_4 critical strength
4.266 at k_size 33, mu = 0.15, reproduces from chi_NN and halves to 2.133 with chi0 = 2 chi_NN).
The bare terms in Gamma are unchanged, since they never touch chi.
"""
function combine_chis(data::Dict ; directions::Vector{Int},
                      subs::Union{Int64, Nothing} = nothing,
                      suffix::String = "")::Vector{Matrix{ComplexF64}}
    directions = sort(directions)
    data_labels = [labels[i] * suffix for i in directions]
    localDim = length(directions)

    first_chi = data[data_labels[begin]]
    dim = size(first_chi, 2)
    size(first_chi, 3) == dim ||
        error("combine_chis: $(data_labels[begin]) has blocks of size " *
              "$(size(first_chi, 2))x$(size(first_chi, 3)); expected square.")
    for label in data_labels
        size(data[label], 2) == dim && size(data[label], 3) == dim ||
            error("combine_chis: $label has block size $(size(data[label], 2)), " *
                  "but $(data_labels[begin]) has $dim -- the directions were not " *
                  "contracted onto the same basis.")
    end
    if subs !== nothing && subs != dim
        @info("combine_chis: stored chi has block size $dim while the model has " *
              "$subs orbitals -- run_bare used the symmetry-closed bilinear basis. " *
              "Interaction matrices must be built at 4 * $dim = $(4 * dim), not " *
              "4 * $subs.")
    end

    chi_combined = [zeros(ComplexF64, dim*localDim, dim*localDim)
                    for _ in 1:size(first_chi, 1)]

    for (ind, label) in enumerate(data_labels)
        chis = data[label]
        inds = [ind + localDim * (i - 1) for i in 1:dim]
        setindex!.(chi_combined, [2 .* transpose(c) for c in eachslice(chis, dims = 1)],
                   Ref(inds), Ref(inds))
    end

    return chi_combined
end




end
