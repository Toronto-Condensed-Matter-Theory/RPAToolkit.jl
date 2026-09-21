module Response

using LinearAlgebra
using ..RPAToolkit.Interactions: interaction

export perform_RPA, minima, maxima, find_instability, effective_interaction

#####* Returns the eigenvalues and eigenvectors of the RPA susceptibility matrix at some fixed momentum.
function perform_RPA(chi::Matrix{ComplexF64}, interaction::Matrix{ComplexF64} ;
        return_matrix::Bool = false)
    mat = inv(I - chi * interaction) * chi

    if return_matrix
        return mat
    else
        eigs = eigen(mat)

        values = eigs.values
        vectors = eigs.vectors

        return (values, vectors)
    end
end

#####* Returns the eigenvalues and eigenvectors of the RPA susceptibility matrix at multiple momenta ks.
function perform_RPA(chis::Vector{Matrix{ComplexF64}}, interactions::Vector{Matrix{ComplexF64}} ;
        return_matrix::Bool = false)
    return perform_RPA.(chis, interactions ; return_matrix = return_matrix)
end

#####* Returns the RPA renormalized interaction V_eff(q) = V(q) + V(q) chi_RPA(q) V(q)
function effective_interaction(chi_bare::Matrix{ComplexF64}, interaction::Matrix{ComplexF64})
    chi_rpa = perform_RPA(chi_bare, interaction; return_matrix = true)
    return interaction + interaction * chi_rpa * interaction
end

function effective_interaction(chis_bare::Vector{Matrix{ComplexF64}}, interactions::Vector{Matrix{ComplexF64}})
    return effective_interaction.(chis_bare, interactions)
end

#####* returns the eigenvalue and eigenvector corresponding to the minimum eigenvalue over all momenta.
function minima(eigenstates::Vector{Tuple{Vector{ComplexF64}, Matrix{ComplexF64}}})::Dict{String, Any}

    eigenvalues = getindex.(eigenstates, 1)
    eigenvectors = getindex.(eigenstates, 2)

    minEigs = getindex.(eigenvalues, 1)
    value, index = findmin(real.(minEigs))
    return Dict("minimum index" => index,
                "minimum eigenvalue" => value,
                "minimum eigenvector" => eigenvectors[index][:, 1])
end

#####* returns the eigenvalue and eigenvector corresponding to the maximum eigenvalue over all momenta.
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
    find_instability(chis, ks, V_unit; steps, lower, upper, primitives)

The same binary search, but against interaction matrices that are already
built. `interaction(strength, ks) == strength * interaction(1.0, ks)` exactly,
so the bisection only needs to rescale them.

This is the form the caller wants whenever the matrices do not come from a
`Lookup` at all -- a vertex projected onto a symmetry-closed bilinear basis is
one constant matrix per q, with no bonds to walk -- and it is also what keeps
the search from rebuilding the same matrices 32 times per case and mu. On the
ZrNCl Coulomb case (7419 bonds on a 297 x 297 grid) that rebuild is the whole
cost of the stage.
"""
function find_instability(chis::Vector{Matrix{ComplexF64}}, ks::Vector{Vector{Float64}},
        V_unit::Vector{Matrix{ComplexF64}};
        steps::Int = 32, lower::Float64 = 0.0, upper::Float64 = 10.0,
        primitives = [[0.0, 0.0], [0.0, 0.0]])::Dict{String, Any}

    length(V_unit) == length(chis) ||
        error("find_instability: $(length(V_unit)) interaction matrices for $(length(chis)) " *
              "susceptibilities; they must live on the same q grid.")
    scaled(s) = [s .* V for V in V_unit]

    check = nothing
    current = Float64[]
    for _ in 1:steps
        push!(current, (upper + lower) / 2)
        check = minima(perform_RPA(chis, scaled(current[end])))
        if check["minimum eigenvalue"] < -1e-6
            upper = current[end]
        else
            lower = current[end]
        end
    end

    d = length(primitives[begin])
    at = check["minimum eigenvalue"] < -1e-6 ? lower : current[end]
    eigenstates = perform_RPA(chis, scaled(at))
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

    current = Float64[]
    check = nothing

    #####* binary search for the critical interaction strength |J| at a given unit cell fixing ratios of interactions.
    for _ in 1:steps
        push!(current, (upper + lower) / 2)
        ##### determining the interaction matrices.
        interactions = interaction(current[end], ks ; kwargs...)
        ##### RPA calculation.
        eigenstates = perform_RPA(chis, interactions)

        check = minima(eigenstates)

        if check["minimum eigenvalue"] < -1e-6
            upper = current[end]
        else
            lower = current[end]
        end
    end

    primitives = get(kwargs, :primitives, [[0.0, 0.0], [0.0, 0.0]])
    d = length(primitives[begin])

    if check["minimum eigenvalue"] < -1e-6
        interactions = interaction(lower, ks ; kwargs...)
        eigenstates = perform_RPA(chis, interactions)

        check = minima(eigenstates)
        peak = maxima(eigenstates)

        k_min = ks[check["minimum index"]]
        k_max = ks[peak["maximum index"]]

        return Dict("critical strength" => lower, check..., peak...,
                    "minimum reciprocal momentum" => dot.(Ref(k_min[1:d]), primitives) ./ (2*pi),
                    "minimum momentum" => k_min,
                    "maximum reciprocal momentum" => dot.(Ref(k_max[1:d]), primitives) ./ (2*pi),
                    "maximum momentum" => k_max)
    else
        interactions = interaction(current[end], ks ; kwargs...)
        eigenstates = perform_RPA(chis, interactions)

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
end








































































end
