using ArgParse, JLD2, TightBindingToolkit, YAML, NPZ, LaTeXStrings, Plots

function parse_commandline()

    settings = ArgParseSettings()

    @add_arg_table settings begin
        "--input"
            help = "directory of the input yml file."
            arg_type = String
            required = true
    end

    return parse_args(settings)
end

"""
The lattice primitive vector for a triangular lattice : a1 and a2.
"""
const a1 = [ 1.0, 0.0 ]
const a2 = [ 1/2 , sqrt(3)/2 ]

const a1New = 2*a2
const a2New = a1
UC = UnitCell( [a1New , a2New] , 2) ##### localDim=2 since we are working with spin-1/2 particles now

"""
Unit cell has 6 sub-lattices.
"""
const b1 = [ 0.0 , 0.0 ]
const b2 = b1+a2
AddBasisSite!.( Ref(UC) , [b1, b2])

"""
Adding structure to the lattice now, through the bond objects.
"""
SpinVec     =   SpinMats(1//2) ##### Working with spin-1/2
################ Flux pi-0 hoppings #################
const NNdistance  =   1.0
const secondNNDistance = sqrt(3)

function interaction(D1::Float64, D2::Float64, J2s::Vector{Float64})

    J1Param   =   Param(1.0, 2)
    J2Param   =   Param(1.0, 2)
    AddIsotropicBonds!(J1Param, UC , NNdistance, [1.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 D1] , "J1 XXZ")
    AddIsotropicBonds!(J2Param, UC , secondNNDistance, [1.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 D2] , "J2 XXZ")

    params = [J1Param, J2Param]

    values = Dict()
    for J2 in J2s
        values["J2 = $(round(J2, digits=3))"] = [cos(atan(J2)), sin(atan(J2))]
    end
    file_name = "/home/anjishnubose/Research/Repos/RPA.jl/saves/interactions/triangular_J1J2.jld2"
    save(file_name, Dict("parameters" => params, "values" => values))

end


function readData(Parentdata::Dict, J2s::Vector{Float64})
    data = Parentdata["beta=20.0_mu=0.0"]
    labels = ["J2 = $(round(J2, digits=3))" for J2 in J2s]

    ks = data["triqs_data"]["contracted"]
    ks = Vector{Float64}[eachrow(ks)...]

    Js = Float64[]
    Qs = Vector{Float64}[]
    eigenvectors = Vector{ComplexF64}[]

    for label in labels
        push!(Js, data[label]["critical strength"])
        push!(Qs, data[label]["maximum momentum"])
        push!(eigenvectors, data[label]["maximum eigenvector"])
    end

    ks = data["triqs_data"]["contracted"]
    ks = Vector{Float64}[eachrow(ks)...]
    colors = data["triqs_data"]["path_plot"][indexin(Qs, ks)] / data["triqs_data"]["path_plot"][end]

    return Dict("Js" => Js, "Qs" => Qs, "eigenvectors" => eigenvectors, "colors" => colors, "ks" => ks)
end

if abspath(PROGRAM_FILE) == @__FILE__

    include("./RPA.jl")
    using .RPA

    parsed_args = parse_commandline()

    const nD = 21
    const nJ = 101
    const Δ1s = collect(LinRange(0.0, 1.0, nD))
    const Δ2s = collect(LinRange(0.0, 1.0, nD))
    const J2s = collect(LinRange(0.0, 1.0, nJ))

    Qs = Array{Vector{Float64}}(undef, nD, nD, nJ)
    Js = zeros(Float64, nD, nD, nJ)
    colors = zeros(Float64, nD, nD, nJ)
    eigenvectors = Array{Vector{ComplexF64}}(undef, nD, nD, nJ)
    global ks = Vector{Float64}[]

    #####* Iterating
    for (d1, D1) in enumerate(Δ1s)
        for (d2, D2) in enumerate(Δ2s)
            println("Working on D1 = $(D1), D2 = $(D2)...")
            ##### Saving the new interaction file
            interaction(D1, D2, J2s)
            ##### Running the RPA
            command = `julia --project=../Project.toml ./run_RPA.jl --input=$(parsed_args["input"])`
            run(command)
            ##### Saving the output
            output = load("/home/anjishnubose/Research/Repos/RPA.jl/saves/data/triangular_Dirac_Bz=0.0_combined_new.jld2")
            data = readData(output, J2s)
            Js[d1, d2, :] = data["Js"]
            Qs[d1, d2, :] = data["Qs"]
            eigenvectors[d1, d2, :] = data["eigenvectors"]
            colors[d1, d2, :] = data["colors"]
            global ks = data["ks"]
            GC.gc()
        end
        finalOutput = Dict("Js" => Js, "Qs" => Qs, "eigenvectors" => eigenvectors,
                    "colors" => colors, "ks"=>ks,
                    "J2s" => J2s, "Δ1s" => Δ1s, "Δ2s" => Δ2s)
        save("/home/anjishnubose/Research/Repos/RPA.jl/saves/data/triangular_Dirac_Bz=0.0_combined_allDs.jld2", finalOutput)
    end








end
