module Response

using LinearAlgebra
using ..RPAToolkit.Interactions: interaction
using ..RPAToolkit.SpinSectors: off_sector_norm, spin_sectors

export perform_RPA, minima, maxima, find_instability, effective_interaction, sector_plan

"""
    sector_plan(chis, Vs, sectors) -> Vector{NamedTuple}

How the RPA splits into independent blocks: one `(name, idx, rep)` per sector, where `rep` is
the index of an earlier sector whose blocks of every `chi` and `V` are identical to this one's
(so its results are reused, not recomputed), or 0.

`sectors` comes from [`spin_sectors`](@ref). Both `chi` and `V` must be block diagonal in them,
which a model's spin symmetry guarantees for `chi` and a symmetric interaction for `V`. If either
is not (an interaction that breaks the model's spin symmetry, say), the plan is the whole matrix
as one block, with a message: blocks are never dropped. `sectors = nothing` gives that plan too.
Under SU2 the three spin sectors of a spin-rotation-invariant interaction are identical, so only
one of them is computed.
"""
function sector_plan(chis::Vector{Matrix{ComplexF64}}, Vs::Vector{Matrix{ComplexF64}}, sectors)
    n = size(chis[begin], 1)
    whole = [(name = "all", idx = collect(1:n), rep = 0)]
    sectors === nothing && return whole
    sum(length(idx) for (_, idx) in sectors) == n ||
        error("sector_plan: the sectors cover $(sum(length(idx) for (_, idx) in sectors)) of $n rows.")

    scale = max(maximum(maximum(abs, c) for c in chis), maximum(maximum(abs, V) for V in Vs))
    outside = max(maximum(off_sector_norm(c, sectors) for c in chis),
                  maximum(off_sector_norm(V, sectors) for V in Vs))
    if outside > 1e-12 * scale
        @info("The susceptibility or the interaction is not block diagonal in the spin sectors " *
              "$(join(first.(sectors), ", ")) (off-sector entries up to $(round(outside / scale, sigdigits = 3)) " *
              "of the largest): solving the RPA on the whole matrix.")
        return whole
    end

    plan = NamedTuple{(:name, :idx, :rep), Tuple{String, Vector{Int}, Int}}[]
    for (name, idx) in sectors
        rep = findfirst(p -> p.rep == 0 && length(p.idx) == length(idx) &&
                             all(c[p.idx, p.idx] == c[idx, idx] for c in chis) &&
                             all(V[p.idx, p.idx] == V[idx, idx] for V in Vs), plan)
        push!(plan, (name = name, idx = idx, rep = something(rep, 0)))
    end
    return plan
end

#####* the name a sector is reported under: "spin z" stands for all three spin sectors when the
#####* other two are copies of it
function sector_name(plan, s::Int)
    any(p -> p.rep == s, plan) || return plan[s].name
    return replace(plan[s].name, r" [xyz]$" => "")
end

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
        return_matrix::Bool = false, sectors = nothing)
    sectors === nothing && return perform_RPA.(chis, interactions ; return_matrix = return_matrix)
    plan = sector_plan(chis, interactions, sectors)
    return [blocked_RPA(chi, V, plan; return_matrix = return_matrix) for (chi, V) in zip(chis, interactions)]
end

#####* perform_RPA one sector block at a time, reusing the blocks the plan marks as copies, and
#####* reassembled into the full matrix (or its eigen-decomposition, sorted as `eigen` sorts).
function blocked_RPA(chi::Matrix{ComplexF64}, V::Matrix{ComplexF64}, plan; return_matrix::Bool)
    n = size(chi, 1)
    parts = Vector{Any}(undef, length(plan))
    for (s, p) in enumerate(plan)
        p.rep == 0 || continue
        parts[s] = perform_RPA(chi[p.idx, p.idx], V[p.idx, p.idx]; return_matrix = return_matrix)
    end
    if return_matrix
        mat = zeros(ComplexF64, n, n)
        for (s, p) in enumerate(plan)
            mat[p.idx, p.idx] = parts[p.rep == 0 ? s : p.rep]
        end
        return mat
    end
    values = ComplexF64[]
    vectors = zeros(ComplexF64, n, n)
    for (s, p) in enumerate(plan)
        vals, vecs = parts[p.rep == 0 ? s : p.rep]
        cols = length(values) .+ (1:length(vals))
        append!(values, vals)
        vectors[p.idx, cols] = vecs
    end
    order = sortperm(values, by = v -> (real(v), imag(v)))
    return (values[order], vectors[:, order])
end

"""
    effective_interaction(chi_bare, V) -> V + V chi_RPA V
    effective_interaction(chis_bare, Vs; sectors = nothing)

The RPA-renormalized interaction at one momentum, or at each momentum for vectors of matrices.
With `sectors` ([`spin_sectors`](@ref)) the vectors are solved one sector block at a time
([`sector_plan`](@ref)); the result is the same full matrix.
"""
function effective_interaction(chi_bare::Matrix{ComplexF64}, interaction::Matrix{ComplexF64})
    chi_rpa = perform_RPA(chi_bare, interaction; return_matrix = true)
    return interaction + interaction * chi_rpa * interaction
end

function effective_interaction(chis_bare::Vector{Matrix{ComplexF64}}, interactions::Vector{Matrix{ComplexF64}};
        sectors = nothing)
    chi_rpas = perform_RPA(chis_bare, interactions; return_matrix = true, sectors = sectors)
    return [V + V * chi_rpa * V for (chi_rpa, V) in zip(chi_rpas, interactions)]
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

With `sectors` ([`spin_sectors`](@ref)) every eigenproblem is solved one spin sector at a time
([`sector_plan`](@ref)), and `"unstable sector"` names the one that goes unstable first:
`"charge"` or `"spin"` without spin-orbit coupling, `"longitudinal"` (N, Z) or `"transverse"`
(X, Y) with ``S_z`` conserved, `"all"` otherwise.
"""
function find_instability(chis::Vector{Matrix{ComplexF64}}, ks::Vector{Vector{Float64}},
        V_unit::Vector{Matrix{ComplexF64}};
        steps::Int = 32, lower::Float64 = 0.0, upper::Float64 = 10.0,
        primitives = [[0.0, 0.0], [0.0, 0.0]], sectors = nothing)::Dict{String, Any}

    length(V_unit) == length(chis) ||
        error("find_instability: $(length(V_unit)) interaction matrices for $(length(chis)) " *
              "susceptibilities; they must live on the same q grid.")

    plan = sector_plan(chis, V_unit, sectors)
    λs = [p.rep == 0 ? maximum(maximum(real, eigvals(chi[p.idx, p.idx] * V[p.idx, p.idx]))
                               for (chi, V) in zip(chis, V_unit)) : -Inf
          for p in plan]
    λmax, leading = findmax(λs)
    for _ in 1:steps
        mid = (upper + lower) / 2
        if mid * λmax > 1
            upper = mid
        else
            lower = mid
        end
    end

    d = length(primitives[begin])
    eigenstates = [blocked_RPA(chi, lower .* V, plan; return_matrix = false) for (chi, V) in zip(chis, V_unit)]
    check = minima(eigenstates)
    peak = maxima(eigenstates)
    k_min = ks[check["minimum index"]]
    k_max = ks[peak["maximum index"]]

    return Dict("critical strength" => lower, "unstable sector" => sector_name(plan, leading),
                check..., peak...,
                "minimum reciprocal momentum" => dot.(Ref(k_min[1:d]), primitives) ./ (2*pi),
                "minimum momentum" => k_min,
                "maximum reciprocal momentum" => dot.(Ref(k_max[1:d]), primitives) ./ (2*pi),
                "maximum momentum" => k_max)
end

function find_instability(chis::Vector{Matrix{ComplexF64}}, ks::Vector{Vector{Float64}};
        steps::Int = 32, lower::Float64 = 0.0, upper::Float64 = 10.0,
        kwargs...)::Dict{String, Any}

    primitives = get(kwargs, :primitives, [[0.0, 0.0], [0.0, 0.0]])
    sectors = get(kwargs, :sectors, nothing)
    return find_instability(chis, ks, interaction(1.0, ks; (k => v for (k, v) in kwargs if k != :sectors)...);
        steps = steps, lower = lower, upper = upper, primitives = primitives, sectors = sectors)
end

end
