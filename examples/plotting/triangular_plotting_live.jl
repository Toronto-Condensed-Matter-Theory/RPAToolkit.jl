using JLD2, Plots, LaTeXStrings, LinearAlgebra
using ArgParse, TightBindingToolkit, YAML, NPZ
include("../../src/RPA.jl")
using .RPA

"""
The lattice primitive vector for a triangular lattice : a1 and a2.
"""
const a1 = [ 1.0, 0.0 ]
const a2 = [ 1/2 , sqrt(3)/2 ]

const a1New = 2*a2
const a2New = a1
UC = UnitCell( [a1New , a2New] , 2) ##### localDim=2 since we are working with spin-1/2 particles now

"""
Unit cell has 2 sub-lattices.
"""
const b1 = [ 0.0 , 0.0 ]
const b2 = b1+a2
AddBasisSite!.( Ref(UC) , [b1, b2])

"""
Adding structure to the lattice now, through the bond objects.
"""
const SpinVec     =   SpinMats(1//2) ##### Working with spin-1/2
################ Flux pi-0 hoppings #################
const NNdistance  =   1.0
const secondNNDistance = sqrt(3)
########### triqs data ############
const triqs_data = npzread("./saves/data/triangular_Dirac_Bz=0.0_beta=20.0_mu=0.0.npz")
const primitives = RPA.dress_primitives(triqs_data)
const ks_contracted = triqs_data["contracted"]
const ks = Vector{eltype(ks_contracted)}[eachrow(ks_contracted)...]
const chis = RPA.combine_chis(triqs_data; directions = [1, 2, 3], subs = length(UC.basis))
############ Fixed tuning parameters ############
const δ = 0.7
const λ = 0.8
const χ = 0.2
############## Parameters #############
const J2s = collect(0.0:0.005:0.2)
const Deltas = collect(1.0:0.1:10.0)

function get_interaction(Δ1::Float64, Δ2::Float64, J2::Float64 ; strength::Float64 = 1.0)

    J1Param   =   Param(strength * cos(atan(J2)), 2)
    J2Param   =   Param(strength * sin(atan(J2)), 2)
    AddIsotropicBonds!(J1Param, UC , NNdistance, [1.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 Δ1] , "J1 XXZ")
    AddIsotropicBonds!(J2Param, UC , secondNNDistance, [1.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 Δ2] , "J2 XXZ")

    params = Param{2, Float64}[J1Param, J2Param]

    return Lookup(params)
end

function get_effective_params(α1::Float64,
                            δ::Float64,
                            λ::Float64,
                            Δ_physical::Float64,
                            J2_physical::Float64,
                            χ::Float64 = 0.2)
    ######## effective hopping #########
    t_eff = abs(χ * α1 * (0.5*δ + 0.25*Δ_physical))
    ########## effective exchange #########
    J1_eff_xy = (1-(α1 * δ))
    J1_eff_z = Δ_physical * (1 - α1)
    J2_eff_xy = J2_physical * (1-(α1 * λ * δ))
    J2_eff_z = J2_physical * Δ_physical * (1 - α1 * λ)

    return Dict("t_eff" => t_eff,
                "J1_eff_xy" => J1_eff_xy/t_eff, "J1_eff_z" => J1_eff_z/t_eff,
                "J2_eff_xy" => J2_eff_xy/t_eff, "J2_eff_z" => J2_eff_z/t_eff,
                "D1_eff" => J1_eff_z/J1_eff_xy, "D2_eff" =>  Δ_physical * (1 - α1 * λ)/(1-(α1 * λ * δ)),
                "J2/J1_eff" => J2_eff_xy/J1_eff_xy,
                "J_eff" => sqrt(J1_eff_xy^2 + J2_eff_xy^2)/t_eff)

end

function get_critical_params(chis::Vector{Matrix{ComplexF64}},
                    δ::Float64,
                    λ::Float64,
                    Δ_physical::Float64,
                    J2_physical::Float64,
                    χ::Float64 = 0.2 ;
                    steps::Int64 = 10+1)

    lower = 0.01
    upper = 1.00
    current = Float64[]
    check = nothing

    for _ in 1:steps
        push!(current, (upper + lower) / 2)
        params = get_effective_params(current[end], δ, λ, Δ_physical, J2_physical, χ)
        ##### determining the interaction matrices.
        interaction_lookup = get_interaction(params["D1_eff"], params["D2_eff"], params["J2/J1_eff"] ; strength = params["J_eff"])
        interaction_mats = RPA.interaction(1.0, ks;
                                        primitives,
                                        subs = 2, localDim = 3,
                                        lookup = interaction_lookup)
        ##### RPA calculation.
        eigenstates = RPA.perform_RPA(chis, interaction_mats)
        check = RPA.minima(eigenstates)

        if check["minimum eigenvalue"] < -1e-6
            lower = current[end]
        else
            upper = current[end]
        end
    end

    if check["minimum eigenvalue"] < -1e-6
        params = get_effective_params(upper, δ, λ, Δ_physical, J2_physical, χ)
        ##### determining the interaction matrices.
        interaction_lookup = get_interaction(params["D1_eff"], params["D2_eff"], params["J2/J1_eff"] ; strength = params["J_eff"])
        interaction_mats = RPA.interaction(1.0, ks;
                                        primitives,
                                        subs = 2, localDim = 3,
                                        lookup = interaction_lookup)
        eigenstates = RPA.perform_RPA(chis, interaction_mats)
        peak = RPA.maxima(eigenstates)

        k_max = ks[peak["maximum index"]]

        return Dict("critical alpha" => upper,
                    "maximum reciprocal momentum" => dot.(Ref(k_max[1:2]), primitives) ./ (2*pi),
                    "maximum momentum" => k_max,
                    "maximum eigenvector" => peak["maximum eigenvector"],
                    "colors" => (triqs_data["path_plot"][indexin([k_max], ks)] / triqs_data["path_plot"][end])...)
    else
        params = get_effective_params(current[end], δ, λ, Δ_physical, J2_physical, χ)
        ##### determining the interaction matrices.
        interaction_lookup = get_interaction(params["D1_eff"], params["D2_eff"], params["J2/J1_eff"] ; strength = params["J_eff"])
        interaction = RPA.interaction(1.0, ks;
                                        primitives,
                                        subs = 2, localDim = 3,
                                        lookup = interaction_lookup)
        eigenstates = RPA.perform_RPA(chis, interaction_mats)
        peak = RPA.maxima(eigenstates)

        k_max = ks[peak["maximum index"]]

        return Dict("critical alpha" => current[end],
                    "maximum reciprocal momentum" => dot.(Ref(k_max[1:2]), primitives) ./ (2*pi),
                    "maximum momentum" => k_max,
                    "maximum eigenvector" => peak["maximum eigenvector"],
                    "colors" => (triqs_data["path_plot"][indexin([k_max], ks)] / triqs_data["path_plot"][end])...)
    end

end

function get_critical_params(chis::Vector{Matrix{ComplexF64}},
                                δ::Float64,
                                λ::Float64,
                                Δ_physical::Vector{Float64},
                                J2_physical::Float64,
                                χ::Float64 = 0.2 ;
                                steps::Int64 = 10+1)
    return [get_critical_params(chis, δ, λ, Δ, J2_physical, χ ; steps = steps) for Δ in Δ_physical]
end

function get_critical_params(chis::Vector{Matrix{ComplexF64}},
                                δ::Float64,
                                λ::Float64,
                                Δ_physical::Float64,
                                J2_physical::Vector{Float64},
                                χ::Float64 = 0.2 ;
                                steps::Int64 = 10+1)
    return [get_critical_params(chis, δ, λ, Δ_physical, J2, χ ; steps = steps) for J2 in J2_physical]
end

function plot_J2_RPA_phasediagram(chis::Vector{Matrix{ComplexF64}},
                                    δ::Float64,
                                    λ::Float64,
                                    Δ_physical::Float64,
                                    J2_physical::Vector{Float64},
                                    χ::Float64 = 0.2 ;
                                    steps::Int64 = 10+1,
                                    critical_line::Float64 = -1.0)
    critical = get_critical_params(chis, δ, λ, Δ_physical, J2_physical, χ ; steps = steps)
    p = plot(framestyle=:box, grid=false,
            xlabel=L"$J_2/J_1$", ylabel=L"$\alpha_1$",
            title=L"\Delta/J_1=%$(round(Δ_physical, digits=1))",
            guidefont = font(14, "Computer Modern"), tickfont = font(12, "Computer Modern"),
            legendfont = font(12, "Computer Modern"), titlefont = font(14, "Computer Modern"))
    scatter!(p, J2_physical, getindex.(critical, Ref("critical alpha")),
            label=L"\delta=%$(round(δ, digits=1)), \lambda=%$(round(λ, digits=1))",
            legend_position=:bottomright,
            marker=:o, lw=2.0,
            markersize = 5, markerstrokealpha = 0.25,
            m=cgrad(:darktest, rev=true), zcolor = getindex.(critical, Ref("colors")), clims=(0, 1),)
    ylims!(-0.01, 1.01)
    if critical_line > 0
        hline!(p, [critical_line], lw=2.0, ls=:dash, color=:black, label=L"\alpha_1^c=%$(round(critical_line, digits=2))", linealpha=0.5)
    end
    return p
end

function plot_Delta_RPA_phasediagram(chis::Vector{Matrix{ComplexF64}},
                                    δ::Float64,
                                    λ::Float64,
                                    Δ_physical::Vector{Float64},
                                    J2_physical::Float64,
                                    χ::Float64 = 0.2 ;
                                    steps::Int64 = 10+1,
                                    critical_line::Float64 = -1.0)
    critical = get_critical_params(chis, δ, λ, Δ_physical, J2_physical, χ ; steps = steps)
    p = plot(framestyle=:box, grid=false,
            xlabel=L"$\Delta/J_1$", ylabel=L"$\alpha_1$",
            title=L"J_2/J_1=%$(round(J2_physical, digits=2))",
            guidefont = font(14, "Computer Modern"), tickfont = font(12, "Computer Modern"),
            legendfont = font(12, "Computer Modern"), titlefont = font(14, "Computer Modern"))
    scatter!(p, Δ_physical, getindex.(critical, Ref("critical alpha")),
            label=L"\delta=%$(round(δ, digits=1)), \lambda=%$(round(λ, digits=1))",
            legend_position=:bottomright,
            marker=:o, lw=2.0,
            markersize = 5, markerstrokealpha = 0.25,
            m=cgrad(:darktest, rev=true), zcolor = getindex.(critical, Ref("colors")), clims=(0, 1),)
    ylims!(-0.01, 1.01)
    if critical_line > 0
        hline!(p, [critical_line], lw=2.0, ls=:dash, color=:black, label=L"\alpha_1^c=%$(round(critical_line, digits=2))", linealpha=0.5)
    end
    return p
end

function get_dirac_boundary(chis::Vector{Matrix{ComplexF64}},
                            α1::Float64,
                            δ::Float64,
                            λ::Float64,
                            J2_physical::Float64,
                            χ::Float64 = 0.2 ;
                            )

    DeltaInv_Checks = collect(0.8:-0.01:0.1)

    for (d, Δ) in enumerate((1 ./ DeltaInv_Checks))
        # println("Working on Δ = $(Δ)...")
        params = get_effective_params(α1, δ, λ, Δ, J2_physical, χ)
        ##### determining the interaction matrices.
        interaction_lookup = get_interaction(params["D1_eff"], params["D2_eff"], params["J2/J1_eff"] ; strength = params["J_eff"])
        interaction_mats = RPA.interaction(1.0, ks;
                                        primitives,
                                        subs = 2, localDim = 3,
                                        lookup = interaction_lookup)
        ##### RPA calculation.
        eigenstates = RPA.perform_RPA(chis, interaction_mats)
        check = RPA.minima(eigenstates)
        if check["minimum eigenvalue"] < -1e-6 && Δ<=1/DeltaInv_Checks[1]
            return Dict("critical InvDelta" => DeltaInv_Checks[1],
                        "maximum reciprocal momentum" => Float64[],
                        "maximum momentum" => Float64[],
                        "maximum eigenvector" => Float64[],
                        "colors" => -1.0)
        elseif check["minimum eigenvalue"] < -1e-6 && Δ>1/DeltaInv_Checks[1]
            params = get_effective_params(α1, δ, λ, 1/DeltaInv_Checks[d-1], J2_physical, χ)
            ##### determining the interaction matrices.
            interaction_lookup = get_interaction(params["D1_eff"], params["D2_eff"], params["J2/J1_eff"] ; strength = params["J_eff"])
            interaction_mats = RPA.interaction(1.0, ks;
                                            primitives,
                                            subs = 2, localDim = 3,
                                            lookup = interaction_lookup)
            ##### RPA calculation.
            eigenstates = RPA.perform_RPA(chis, interaction_mats)
            peak = RPA.maxima(eigenstates)
            k_max = ks[peak["maximum index"]]

            return Dict("critical InvDelta" => 1/Δ,
                        "maximum reciprocal momentum" => dot.(Ref(k_max[1:2]), primitives) ./ (2*pi),
                        "maximum momentum" => k_max,
                        "maximum eigenvector" => peak["maximum eigenvector"],
                        "colors" => (triqs_data["path_plot"][indexin([k_max], ks)] / triqs_data["path_plot"][end])...)
        end
    end
end

function get_dirac_boundary(chis::Vector{Matrix{ComplexF64}},
    α1::Float64,
    δ::Float64,
    λ::Float64,
    J2_physical::Vector{Float64},
    χ::Float64 = 0.2 ;
    )

    return [get_dirac_boundary(chis, α1, δ, λ, J2, χ) for J2 in J2_physical]
end
