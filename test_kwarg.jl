function interaction(strength::Float64, k::Vector{Float64};
        primitives::Vector{Vector{Float64}},
        subs::Int64, localDim::Integer,
        lookup::Dict)::Matrix{ComplexF64}
    return zeros(ComplexF64, 2, 2)
end

function interaction(strength::Float64, ks::Vector{Vector{Float64}};
    primitives::Vector{Vector{Float64}},
    subs::Int64, localDim::Integer,
    lookup::Dict)::Vector{Matrix{ComplexF64}}

    return interaction.(Ref(strength), ks;
        primitives = primitives, subs = subs, localDim = localDim,
        lookup = lookup)
end

ks = [[0.0, 0.0], [1.0, 1.0]]
primitives = [[1.0, 0.0], [0.0, 1.0]]
lookup = Dict{Any, Any}((1, 1, [0,0]) => 1.0)
interaction(1.0, ks; primitives=primitives, subs=1, localDim=2, lookup=lookup)
