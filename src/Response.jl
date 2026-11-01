module Response

using LinearAlgebra
using ..RPAToolkit.Interactions: interaction

export perform_RPA, minima, maxima, find_instability, effective_interaction

"""
    perform_RPA(chi, V; return_matrix = false)

The RPA susceptibility `chi_RPA = (1 - chi V)^-1 chi` at one momentum, returned as the matrix
(`return_matrix = true`) or as its `(eigenvalues, eigenvectors)`, sorted by real part.

With vectors of `chi` and `V` (one per momentum), returns a vector of those.
"""
function perform_RPA(chi::Matrix{ComplexF64}, interaction::Matrix{ComplexF64} ;
        return_matrix::Bool = false)
    mat = (I - chi * interaction) \ chi

    if return_matrix
        return mat
    else
        eigs = eigen(mat)

        values = eigs.values
        vectors = eigs.vectors

        return (values, vectors)
    end
end

function perform_RPA(chis::Vector{Matrix{ComplexF64}}, interactions::Vector{Matrix{ComplexF64}} ;
        return_matrix::Bool = false)
    return perform_RPA.(chis, interactions ; return_matrix = return_matrix)
end

"""
    effective_interaction(chi_bare, V) -> V + V chi_RPA V

The RPA-renormalized interaction at one momentum, or at each momentum for vectors of matrices.
"""
function effective_interaction(chi_bare::Matrix{ComplexF64}, interaction::Matrix{ComplexF64})
    chi_rpa = perform_RPA(chi_bare, interaction; return_matrix = true)
    return interaction + interaction * chi_rpa * interaction
end

function effective_interaction(chis_bare::Vector{Matrix{ComplexF64}}, interactions::Vector{Matrix{ComplexF64}})
    return effective_interaction.(chis_bare, interactions)
end

"""
    minima(eigenstates) -> Dict

The lowest eigenvalue over all momenta of a `perform_RPA` result, with its momentum index and
eigenvector, under the keys `"minimum eigenvalue"`, `"minimum index"`, `"minimum eigenvector"`.
"""
function minima(eigenstates::Vector{Tuple{Vector{ComplexF64}, Matrix{ComplexF64}}})::Dict{String, Any}

    eigenvalues = getindex.(eigenstates, 1)
    eigenvectors = getindex.(eigenstates, 2)

    minEigs = getindex.(eigenvalues, 1)
    value, index = findmin(real.(minEigs))
    return Dict("minimum index" => index,
                "minimum eigenvalue" => value,
                "minimum eigenvector" => eigenvectors[index][:, 1])
end

"""
    maxima(eigenstates) -> Dict

The highest eigenvalue over all momenta of a `perform_RPA` result, with its momentum index and
eigenvector, under the keys `"maximum eigenvalue"`, `"maximum index"`, `"maximum eigenvector"`.
"""
function maxima(eigenstates::Vector{Tuple{Vector{ComplexF64}, Matrix{ComplexF64}}})::Dict{String, Any}

    eigenvalues = getindex.(eigenstates, 1)
    eigenvectors = getindex.(eigenstates, 2)

    n = length(eigenvalues[begin])
    minEigs = getindex.(eigenvalues, n)
    value, index = findmax(real.(minEigs))
    return Dict("maximum index" => index,
                "maximum eigenvalue" => value,
                "maximum eigenvector" => eigenvectors[index][:, n])
end


"""
    find_instability(chis, ks, V_unit; steps = 32, lower = 0.0, upper = 10.0, primitives)
    find_instability(chis, ks; steps, lower, upper, primitives, subs, localDim, lookup)

The critical interaction strength: a bisection on `[lower, upper]` for the smallest `s` at which
`chi_RPA = (1 - chi s V)^-1 chi` acquires a negative eigenvalue at some momentum. `V_unit` are the
interaction matrices at unit strength on the same momenta `ks` as `chis`; the second form builds
them from a bond `lookup` with [`interaction`](@ref). If nothing goes unstable below `upper`, the
result is `upper` (to the bisection's resolution).

Returns a `Dict` with `"critical strength"` plus, from `chi_RPA` at that strength, the
[`minima`](@ref) and [`maxima`](@ref) entries and the momenta they sit at (`"minimum momentum"`,
`"maximum momentum"`, and their `"... reciprocal momentum"` in units of the reciprocal vectors).

Only one `chi_RPA` is diagonalized, at the end. For a positive semidefinite `chi`,
`chi^{1/2} (1 - s A)^-1 chi^{1/2}` with `A = chi^{1/2} V chi^{1/2}` has a negative eigenvalue
exactly when `s λ > 1` for an eigenvalue `λ` of `chi V` (Sylvester's law of inertia), so every
bisection step compares `s` against the largest such `λ` over all momenta, computed once.
"""
function find_instability(chis::Vector{Matrix{ComplexF64}}, ks::Vector{Vector{Float64}},
        V_unit::Vector{Matrix{ComplexF64}};
        steps::Int = 32, lower::Float64 = 0.0, upper::Float64 = 10.0,
        primitives = [[0.0, 0.0], [0.0, 0.0]])::Dict{String, Any}

    length(V_unit) == length(chis) ||
        error("find_instability: $(length(V_unit)) interaction matrices for $(length(chis)) " *
              "susceptibilities; they must live on the same q grid.")

    λmax = maximum(maximum(real, eigvals(chi * V)) for (chi, V) in zip(chis, V_unit))
    for _ in 1:steps
        mid = (upper + lower) / 2
        if mid * λmax > 1
            upper = mid
        else
            lower = mid
        end
    end

    d = length(primitives[begin])
    eigenstates = perform_RPA(chis, [lower .* V for V in V_unit])
    check = minima(eigenstates)
    peak = maxima(eigenstates)
    k_min = ks[check["minimum index"]]
    k_max = ks[peak["maximum index"]]

    return Dict("critical strength" => lower, check..., peak...,
                "minimum reciprocal momentum" => dot.(Ref(k_min[1:d]), primitives) ./ (2*pi),
                "minimum momentum" => k_min,
                "maximum reciprocal momentum" => dot.(Ref(k_max[1:d]), primitives) ./ (2*pi),
                "maximum momentum" => k_max)
end

function find_instability(chis::Vector{Matrix{ComplexF64}}, ks::Vector{Vector{Float64}};
        steps::Int = 32, lower::Float64 = 0.0, upper::Float64 = 10.0,
        kwargs...)::Dict{String, Any}

    primitives = get(kwargs, :primitives, [[0.0, 0.0], [0.0, 0.0]])
    return find_instability(chis, ks, interaction(1.0, ks; kwargs...);
        steps = steps, lower = lower, upper = upper, primitives = primitives)
end

end
