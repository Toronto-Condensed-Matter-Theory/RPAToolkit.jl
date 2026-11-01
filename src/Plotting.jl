module Plotting

using LinearAlgebra, Plots, LaTeXStrings
using ..RPAToolkit.Interactions: interaction
using ..RPAToolkit.Response: perform_RPA

export plot_chi, plot_strengthsVsN, plot_QsVsN


"""
    plot_chi(chis; path_plot, path_ticks, path_labels, kwargs...)
    plot_chi(chis, V; path_plot, path_ticks, path_labels, kwargs...)
    plot_chi(chis, strength, ks; primitives, subs, localDim, lookup, path_plot, path_ticks, path_labels, kwargs...)

Every eigenvalue branch of the susceptibility along a k-path, against `path_plot`, with the
high-symmetry points at `path_ticks` labelled `path_labels` (all three as stored by the bare
stage). With interaction matrices `V` -- or a `strength` and a bond `lookup` to build them on the
momenta `ks` -- plots `chi_RPA = (1 - chi V)^-1 chi` instead of `chi`. Other keywords go to `plot`.
"""
function plot_chi(chis::Vector{Matrix{ComplexF64}}; path_plot, path_ticks, path_labels, kwargs...)
    n = size(chis[begin])[1]
    egvals = eigvals.(chis)

    #####* the defaults, overridable by the caller (a title, say)
    p = plot(; merge((framestyle = :box, grid = false, title = L"\chi(Q, \Omega=0)"), kwargs)...)
    for i in 1:n
        plot!(path_plot, real.(getindex.(egvals, i)), label = "", lw=2.0)
    end
    xticks!(path_ticks, path_labels)
    vline!(path_ticks, color = :black, lw=1.0, ls=:dash, label = "")

    return p
end


function plot_chi(chis::Vector{Matrix{ComplexF64}}, strength::Float64, ks::Vector{Vector{Float64}};
                  primitives, subs, localDim, lookup, kwargs...)

    interaction_mats = interaction(strength, ks;
        primitives = primitives, subs = subs, localDim = localDim, lookup = lookup)

    return plot_chi(chis, interaction_mats; kwargs...)
end


function plot_chi(chis::Vector{Matrix{ComplexF64}}, interaction_mats::Vector{Matrix{ComplexF64}}; kwargs...)

    chi_RPA = perform_RPA.(chis, interaction_mats ; return_matrix = true)

    return plot_chi(chi_RPA; kwargs...)
end


"""
    plot_strengthsVsN(data, label)

The critical strength of interaction case `label` against filling, from a [`run_rpa`](@ref RPAToolkit.Pipeline.run_rpa) output.
"""
function plot_strengthsVsN(data::Dict, label::String; kwargs...)

    p = plot(framestyle = :box, grid=false, title=label)

    mus = Float64[]
    fillings = Float64[]
    strengths = Float64[]

    for (key, value) in data

        push!(mus, value["mu"])
        push!(fillings, value["filling"])
        push!(strengths, value[label]["critical strength"])
    end

    args = sortperm(mus)

    mus = mus[args]
    fillings = fillings[args]
    strengths = strengths[args]

    plot!(fillings, strengths, label = "", lw=2.0, m=:o)
    xlabel!(L"⟨n⟩")
    ylabel!(L"\mathcal{I}_c")

    return p

end


"""
    plot_QsVsN(data, label; plot_kwargs...)

The magnitude of the leading momentum (`"maximum momentum"`) of interaction case `label` against
filling, from a [`run_rpa`](@ref RPAToolkit.Pipeline.run_rpa) output.
"""
function plot_QsVsN(data::Dict, label::String; plot_kwargs...)

    p = plot(; framestyle = :box, grid=false, title=label, plot_kwargs...)

    mus = Float64[]
    fillings = Float64[]
    Qs = Vector{Float64}[]

    for (key, value) in data

        push!(mus, value["mu"])
        push!(fillings, value["filling"])
        push!(Qs, value[label]["maximum momentum"])
    end

    args = sortperm(mus)

    mus = mus[args]
    fillings = fillings[args]
    Qs = Qs[args]

    plot!(fillings, norm.(Qs), label = "", lw=2.0, m=:o)
    xlabel!(L"⟨n⟩")
    ylabel!(L"|Q_c|")

    return p

end

end
