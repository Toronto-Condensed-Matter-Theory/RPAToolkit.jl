module Interactions
using LinearAlgebra

function interaction(strength::Float64, k::Vector{Float64};
        primitives::Vector{Vector{Float64}},
        subs::Int64, localDim::Integer,
        lookup::Dict)::Matrix{ComplexF64}

    mat = zeros(ComplexF64, subs*localDim, subs*localDim)

    for (key, value) in lookup
        i, j, offset = key
        δ = sum(offset .* primitives)

        b1  =   localDim * (i - 1) + 1
        b2  =   localDim * (j - 1) + 1

        mat[b1 : b1 + localDim - 1, b2 : b2 + localDim - 1] .+= value * exp(-1.0im * dot(k[1:length(δ)], δ))
        mat[b2 : b2 + localDim - 1, b1 : b1 + localDim - 1] .+= value' * exp(1.0im * dot(k[1:length(δ)], δ))
    end

    return strength * mat
end

function interaction(strength::Float64, ks::Vector{Vector{Float64}};
    primitives::Vector{Vector{Float64}},
    subs::Int64, localDim::Integer,
    lookup::Dict)::Vector{Matrix{ComplexF64}}

    return interaction.(Ref(strength), ks;
        primitives = primitives, subs = subs, localDim = localDim,
        lookup = lookup)
    end
end

ks = [[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]]
primitives = [[1.0, 0.0], [0.0, 1.0]]
subs = 2
localDim = 4
lookup = Dict{Any, Any}( (1, 1, [0,0]) => zeros(4,4) )

try
    Interactions.interaction(1.0, ks; primitives=primitives, subs=subs, localDim=localDim, lookup=lookup)
    println("Success")
catch e
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end
