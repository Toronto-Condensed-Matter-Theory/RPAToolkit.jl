using ArgParse, JLD2, TightBindingToolkit, YAML, NPZ, LaTeXStrings, Plots

function discover_mus_from_outputs(input)
    output_prefix = String(input["output"])
    beta_label = string(input["beta"])

    output_dir = dirname(output_prefix)
    output_base = basename(output_prefix)
    prefix = output_base * "_beta=" * beta_label * "_mu="

    if !isdir(output_dir)
        return Float64[]
    end

    mus = Float64[]
    for file_name in readdir(output_dir)
        if !startswith(file_name, prefix) || !endswith(file_name, ".npz")
            continue
        end

        mu_str = file_name[length(prefix)+1:end-4]
        mu_val = tryparse(Float64, mu_str)
        if mu_val !== nothing
            push!(mus, mu_val)
        end
    end

    sort!(mus)
    return unique(mus)
end

function get_mu_values(input)
    if haskey(input, "mus") && haskey(input["mus"], "values")
        return input["mus"]["values"]
    end

    discovered = discover_mus_from_outputs(input)
    if !isempty(discovered)
        return discovered
    end

    if haskey(input, "fillings")
        error("Input defines fillings but no mus.values and no matching output files were found. Run bare stage first.")
    end

    error("Input must define mus.values, or output files must exist so mu values can be discovered.")
end

function default_path_labels(input)
    if haskey(input, "k_points_labels")
        labels = input["k_points_labels"]
    elseif haskey(input, "k_labels")
        labels = input["k_labels"]
    else
        return LaTeXString[]
    end

    path_labels = [L"%$(String(label))" for label in labels]
    if !isempty(path_labels)
        push!(path_labels, path_labels[1])
    end
    return path_labels
end

function npz_path_labels(triqs_data)
    if !haskey(triqs_data, "k_point_labels")
        return nothing
    end

    labels = vec(triqs_data["k_point_labels"])
    path_labels = [L"%$(String(label))" for label in labels]
    if !isempty(path_labels)
        push!(path_labels, path_labels[1])
    end

    return path_labels
end

function get_instability_range(input)
    instability_input = get(input, "instability", Dict{Any, Any}())
    lower = Float64(get(instability_input, "lower", 0.0))
    upper = Float64(get(instability_input, "upper", 10.0))
    return lower, upper
end

function get_strength_range(input)
    strengths_input = get(input, "strengths", Dict{Any, Any}())
    lower = Float64(get(strengths_input, "lower", 0.0))
    upper = Float64(get(strengths_input, "upper", 10.0))
    n = Int(get(strengths_input, "n", 6))
    return lower, upper, n
end

function parse_commandline()

    settings = ArgParseSettings()

    @add_arg_table settings begin
        "--input"
            help = "directory of the input yml file."
            arg_type = String
            required = true
        "--run_bare"
            help = "does the bare susceptibility calculation need to be run in TRIQS."
            arg_type = Bool
            default = false
        "--plot_RPA"
            help = "does the RPA calculation need to be plotted."
            arg_type = Bool
            default = false
        "--save_individual"
            help = "does the individual RPA calculations need to be saved."
            arg_type = Bool
            default = false
    end

    return parse_args(settings)
end


if abspath(PROGRAM_FILE) == @__FILE__

    include("./RPA.jl")
    using .RPA

    parsed_args = parse_commandline()

    #####* reading the parent input file
    input = YAML.load_file(parsed_args["input"])
    localDim = length(input["directions"])
    model = load(input["unitcell"]["julia"])
    println("Unit Cell loaded!")

    #####* Extracting the unit cell and parameters from the model
    unitcell = model["unit cell"]
    parameters = model["parameters"]
    subs = length(model["unit cell"].basis)

    #####* Reading interaction data from file : parameters and different values to run RPA on
    interaction_data = load(input["interactions"])
    interaction_params = interaction_data["parameters"]
    interaction_cases = interaction_data["values"]

    #####* running the bare susceptibility calculation in TRIQS if needed
    if parsed_args["run_bare"]
        println("Running the bare susceptibility calculation in TRIQS...")

        command = `julia --project=../Project.toml ./Bare/run_bare.jl --input=$(parsed_args["input"])`
        run(command)
        input = YAML.load_file(parsed_args["input"])
        command = `conda run -n $(input["triqs_environment"]) python ./Bare/plot_bare.py $(parsed_args["input"])`
        run(command)

        println("Bare susceptibility calculation complete!")
    end

    input = YAML.load_file(parsed_args["input"])
    #####* plot labels for k-space plots of bands and susceptibility
    input_path_labels = default_path_labels(input)
    instability_lower, instability_upper = get_instability_range(input)
    strengths_lower, strengths_upper, strengths_n = get_strength_range(input)

    CombinedOutput = Dict()

    for mu in get_mu_values(input)
        println("Working on mu = $(mu)...")
        parent_label = "beta=$(input["beta"])_mu=$(round(mu, digits=3))"
        triqs_data = npzread(input["output"] * "_$(parent_label).npz")
        path_labels = something(npz_path_labels(triqs_data), input_path_labels)

        CombinedOutput[parent_label] = Dict()
        CombinedOutput[parent_label]["mu"] = mu
        CombinedOutput[parent_label]["beta"] = input["beta"]
        CombinedOutput[parent_label]["triqs_data"] = triqs_data

        primitives = dress_primitives(triqs_data)

        ks_contracted = triqs_data["contracted"]
        ks = Vector{eltype(ks_contracted)}[eachrow(ks_contracted)...]
        #####* combining different chis to form the full susceptibility matrix
        chis = combine_chis(triqs_data; directions = input["directions"], subs = subs)
        CombinedOutput[parent_label]["combined_chis"] = chis

        println("Starting RPA...")
        for (label, value) in interaction_cases
            #####* setting the interaction values
            push!.(getproperty.(interaction_params, :value), value)
            lookup = Lookup(interaction_params)

            CombinedOutput[parent_label]["$(label)_interaction"] = interaction(1.0, ks; primitives = primitives,
                                                                                subs = subs, localDim=localDim,
                                                                                lookup = lookup)

            instability = find_instability(chis, ks; primitives = primitives,
                subs = subs, localDim=localDim,
                lookup = lookup,
                lower=instability_lower,
                upper=instability_upper,)

            critical = instability["critical strength"]
            println("Critical interaction strength for $(label) is approximately $(round(critical, digits=3)).")
            candidate_strengths = collect(LinRange(strengths_lower, strengths_upper, strengths_n))
            strengths = [strength for strength in candidate_strengths if strength <= critical]

            if parsed_args["plot_RPA"]
                println("Plotting RPA for $(label)...")

                if isempty(strengths)
                    println("No plotting strengths are <= critical for $(label). Check strengths range in input.")
                end

                for strength in strengths
                    p = plot_chi(chis, strength, ks; primitives = primitives,
                        subs = subs, localDim=localDim,
                        lookup = lookup, path_plot = triqs_data["path_plot"],
                        path_ticks = triqs_data["path_ticks"],
                        path_labels = path_labels)

                    savefig(input["plots"] * "_chi_beta=$(input["beta"])_mu=$(round(mu, digits=3))_interactionID=$(label)_strength=$(round(strength, digits=3)).png")
                end
            end

            CombinedOutput[parent_label][label] = instability

            if parsed_args["save_individual"]
                save(input["output"] * "_beta=$(input["beta"])_mu=$(round(mu, digits=3))_interactionID=$(label).jld2",
                    instability)
            end
        end
        println("RPA complete!")

    end

    save(input["output"] * "_combined_new.jld2", CombinedOutput)

end
