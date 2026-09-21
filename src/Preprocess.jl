module Preprocess

using NPZ, JLD2
using LinearAlgebra

export dress_primitives, dress_reciprocal, combine_chis
export get_reciprocal_ks

const labels = Dict(0 => "chi_NN", 1 => "chi_XX", 2 => "chi_YY", 3 => "chi_ZZ", 4 => "chi_NN")

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
        setindex!.(chi_combined, eachslice(chis, dims = 1), Ref(inds), Ref(inds))
    end

    return chi_combined
end




end
