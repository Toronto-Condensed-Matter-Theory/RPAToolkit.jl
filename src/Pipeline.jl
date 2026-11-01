module Pipeline

using JLD2, NPZ, YAML, LaTeXStrings, LinearAlgebra, Plots, TightBindingToolkit
using ..RPAToolkit.InputParser: load_rpa_input
using ..RPAToolkit.parse_model: parse_unitcell
using ..RPAToolkit.Preprocess: dress_primitives, combine_chis, CHI_ORIENTATION
using ..RPAToolkit.Interactions: interaction
using ..RPAToolkit.Response: find_instability
using ..RPAToolkit.BilinearVertex: basis_rotation, rotate_channel_blocks
using ..RPAToolkit.Plotting: plot_chi

export run_bare, run_rpa, rpa_output_path

# Julia block-buffers stdout to a file, so without the flush nothing this stage prints reaches
# a batch log until it exits.
say(msg) = (println(msg); flush(stdout))

#####* ---------------------------------------------------------------------------------------
#####* bare stage
#####* ---------------------------------------------------------------------------------------

function normalize_kpoint_coordinates(point)
    values = collect(point)
    if length(values) == 2
        return [Float64(values[1]), Float64(values[2]), 0.0]
    elseif length(values) == 3
        return [Float64(values[1]), Float64(values[2]), Float64(values[3])]
    end

    error("Each k-point must have length 2 or 3, got length $(length(values)).")
end

function canonical_high_symmetry_label(label::AbstractString)
    aliases = Dict(
        "G" => "G", "Gamma" => "G", "\\Gamma" => "G", "Γ" => "G",
        "K1" => "K1", "K_1" => "K1",
        "M2" => "M2", "M_2" => "M2",
    )
    return get(aliases, String(label), String(label))
end

function reciprocal_basis_matrix(unitcell)
    a1 = Float64.(unitcell.primitives[1])
    a2 = Float64.(unitcell.primitives[2])
    return 2*pi * inv(transpose(hcat(a1, a2)))
end

function cartesian_to_reduced(point, reciprocal_basis)
    reduced = reciprocal_basis \ Float64.(collect(point))
    return [abs(value) < 1e-12 ? 0.0 : value for value in reduced]
end

#####* k_points given as TightBindingToolkit HighSymPoints labels -> reduced coordinates,
#####* which is the only form the TRIQS stage accepts.
function resolve_k_points!(input, unitcell)
    if !haskey(input, "k_points")
        return
    end

    bz = BZ([input["k_size"], input["k_size"]])
    FillBZ!(bz, unitcell)
    reciprocal_basis = reciprocal_basis_matrix(unitcell)

    resolved = Any[]
    for point in input["k_points"]
        if point isa AbstractString
            label = canonical_high_symmetry_label(point)
            if !haskey(bz.HighSymPoints, label)
                available = join(sort(collect(keys(bz.HighSymPoints))), ", ")
                error("Unknown HighSymPoint '$(point)'. Available points: $(available)")
            end
            reduced = cartesian_to_reduced(bz.HighSymPoints[label], reciprocal_basis)
            push!(resolved, normalize_kpoint_coordinates(reduced))
        else
            push!(resolved, normalize_kpoint_coordinates(point))
        end
    end

    input["k_points"] = resolved
end

#####* where run_bare writes the resolved config, next to the `output` prefix
function runtime_input_path(input)
    output_target = String(input["output"])
    output_dir = (endswith(output_target, "/") || isdir(output_target)) ? output_target : dirname(output_target)
    prefix = basename(output_target)
    return joinpath(output_dir, "$(prefix)_runtime_input.yml")
end

"""
    run_bare(config) -> String

The bare-susceptibility stage, in TRIQS.

Reads the model's `"unit cell"` from `unitcell.julia`, writes it in TRIQS form to `unitcell.triqs`,
writes the resolved config (k-points as coordinates, absolute paths) to
`<output>_runtime_input.yml`, and runs `src/Bare/run_bare.py` on it. That writes one
`<output>_beta=<beta>_mu=<mu>.npz` per chemical potential and records the resolved `mus` and
`fillings` back into the runtime file. Every mu is recomputed and overwritten.

The Python is launched with the config's `triqs_environment` if set, else the `TRIQS_ENVIRONMENT`
environment variable, else `python3` -- e.g. `mpirun -n 8 python3`. Returns the runtime file.
"""
function run_bare(config::AbstractString)
    input = load_rpa_input(config)
    runtime_input = deepcopy(input)

    model = load(input["unitcell"]["julia"])
    unitcell = model["unit cell"]

    resolve_k_points!(runtime_input, unitcell)

    triqs_input = input["unitcell"]["triqs"]
    mkpath(dirname(triqs_input))
    parse_unitcell(unitcell, triqs_input)

    runtime_input_file = runtime_input_path(input)
    mkpath(dirname(runtime_input_file))
    YAML.write_file(runtime_input_file, runtime_input)

    triqs_environment = String(get(input, "triqs_environment",
                                   get(ENV, "TRIQS_ENVIRONMENT", "python3")))
    say("TRIQS launcher: $triqs_environment")

    # One thread per python process: TRIQS parallelizes with MPI, and threaded BLAS in every rank
    # would oversubscribe the node.
    py_env = copy(ENV)
    py_env["OMP_NUM_THREADS"] = "1"
    py_env["MKL_NUM_THREADS"] = "1"
    py_env["OPENBLAS_NUM_THREADS"] = "1"

    triqs_env_cmd = String.(split(triqs_environment))
    command = setenv(Cmd(vcat(triqs_env_cmd, [joinpath(@__DIR__, "Bare", "run_bare.py"), runtime_input_file])), py_env)
    run(command)

    return runtime_input_file
end

#####* ---------------------------------------------------------------------------------------
#####* RPA stage
#####* ---------------------------------------------------------------------------------------

function get_mu_values(input)
    if haskey(input, "mus") && haskey(input["mus"], "values")
        return input["mus"]["values"]
    end

    runtime_file = runtime_input_path(input)
    if isfile(runtime_file)
        runtime_input = YAML.load_file(runtime_file)
        if haskey(runtime_input, "mus") && haskey(runtime_input["mus"], "values")
            return runtime_input["mus"]["values"]
        end
    end

    error("No mu values found in input mus.values or in runtime file $(runtime_file). " *
          "Run the bare stage first (run_bare, or run_rpa with run_bare = true).")
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

# NPZ.jl handles numeric and Bool dtypes only, and npzread parses every entry eagerly, so one
# string entry makes the whole archive unreadable. Read entry by entry and skip only the dtypes
# NPZ.jl cannot represent; rethrow anything else, so a truncated chi array is not quietly dropped.
function npzread_numeric(path)
    archive = NPZ.ZipFile.Reader(path)
    data = Dict{String, Any}()
    skipped = String[]
    try
        for entry in archive.files
            name = replace(entry.name, r"\.npy$" => "")
            try
                data[name] = NPZ.npzreadarray(entry)
            catch err
                occursin("unsupported type", err isa ErrorException ? err.msg : "") || rethrow()
                push!(skipped, name)
            end
        end
    finally
        close(archive)
    end
    isempty(skipped) || say("  $(basename(path)): skipped non-numeric entr$(length(skipped) == 1 ? "y" : "ies") " *
                            "$(join(skipped, ", ")) (NPZ.jl reads numeric dtypes only)")
    return data
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

"""
    rpa_output_path(input) -> String

Where [`run_rpa`](@ref) saves its combined output, for an `input` from
[`load_rpa_input`](@ref): `gap_equation_rpa.rpa_data` if the config has one, else
`<output>combined.jld2`.
"""
function rpa_output_path(input)
    if haskey(input, "gap_equation_rpa") && haskey(input["gap_equation_rpa"], "rpa_data")
        return String(input["gap_equation_rpa"]["rpa_data"])
    end
    return String(input["output"]) * "combined.jld2"
end

#####* one mu's bubble, assembled into the block matrices the RPA multiplies
function load_bubble(input, mu, localDim, subs)
    parent_label = "beta=$(input["beta"])_mu=$(round(mu, digits=3))"
    triqs_data = npzread_numeric(input["output"] * "_$(parent_label).npz")

    ks_contracted = triqs_data["contracted"]
    ks = Vector{eltype(ks_contracted)}[eachrow(ks_contracted)...]
    ks_path_matrix = triqs_data["path"]
    ks_path = Vector{eltype(ks_path_matrix)}[eachrow(ks_path_matrix)...]

    # The basis run_bare.py contracted this bubble onto. Absent on the per-orbital path, where
    # the block size is the orbital count.
    chi_basis = let stored = get(triqs_data, "bilinear_basis", nothing)
        (stored === nothing || size(stored, 1) == 0) ? nothing :
            [Matrix{ComplexF64}(stored[k, :, :]) for k in axes(stored, 1)]
    end
    chi_dim = chi_basis === nothing ? subs : length(chi_basis)

    chis = combine_chis(triqs_data; directions = input["directions"], subs = chi_dim)
    chis_path = combine_chis(triqs_data; directions = input["directions"], subs = chi_dim, suffix = "_path")
    size(chis[begin], 1) == localDim * chi_dim ||
        error("assembled chi at $parent_label is $(size(chis[begin], 1))x$(size(chis[begin], 1)) " *
              "but the stored basis has dimension $chi_dim over $localDim directions, which " *
              "requires $(localDim * chi_dim). The bubble file and its basis disagree.")

    return (mu = mu, parent_label = parent_label, triqs_data = triqs_data,
            primitives = dress_primitives(triqs_data), ks = ks, ks_path = ks_path,
            chi_basis = chi_basis, chi_dim = chi_dim, chis = chis, chis_path = chis_path)
end

"""
    run_rpa(config; run_bare = false, plot = false, save_individual = false) -> Dict

The RPA stage. For every chemical potential of the bare stage and every interaction case in the
`interactions` file, finds the critical strength with [`find_instability`](@ref). Saves everything
to [`rpa_output_path`](@ref) and returns it.

- `run_bare = true` runs [`run_bare`](@ref) first.
- `plot = true` saves `chi_RPA` along the k-path, at each of the config's `strengths` below the
  critical one, as `<plots>_chi_beta=..._mu=..._interactionID=..._strength=....png`.
- `save_individual = true` also saves each instability on its own, as
  `<output>_beta=..._mu=..._interactionID=<case>.jld2`.

The (mu, case) pairs run in parallel over Julia threads (`julia -t N`); each pair is serial over
momenta. Keep BLAS single-threaded (`BLAS.set_num_threads(1)`) so the two do not oversubscribe.

The output has one entry per mu, keyed `"beta=<beta>_mu=<mu>"`, holding `"mu"`, `"beta"`,
`"filling"`, `"triqs_data"` (the bubble file), `"combined_chis"` and `"combined_chis_path"` (from
[`combine_chis`](@ref)), `"chi_orientation"`, `"<case>_interaction"` and
`"<case>_interaction_path"` (the unit-strength `V(q)`), and `"<case>"` (the `find_instability`
result). On the closed bilinear basis it also holds `"bilinear_basis"` and `"basis_dim"`.
"""
function run_rpa(config::AbstractString; run_bare::Bool = false, plot::Bool = false,
                 save_individual::Bool = false)
    input = load_rpa_input(config)
    localDim = length(input["directions"])

    model = load(input["unitcell"]["julia"])
    unitcell = model["unit cell"]
    subs = length(unitcell.basis)
    say("Unit cell loaded: $subs orbitals.")

    interaction_data = load(input["interactions"])

    #####* The model and the interactions must describe the SAME orbital manifold. `subs` comes
    # from the model and sets every interaction matrix's dimension; the bonds filling them come
    # from the interactions file. A mismatch does not throw on its own -- the extra sites simply
    # end up with no interaction -- so it is checked here.
    haskey(interaction_data, "unit cell") ||
        error("Interactions file $(input["interactions"]) carries no 'unit cell', so the orbital " *
              "count it was built for cannot be checked against the model's $subs. Save the " *
              "unit cell the interactions were built on under the key \"unit cell\".")
    interaction_subs = length(interaction_data["unit cell"].basis)
    interaction_subs == subs ||
        error("Orbital mismatch between model and interactions.\n" *
              "  model        $(input["unitcell"]["julia"])\n" *
              "               -> $subs orbitals\n" *
              "  interactions $(input["interactions"])\n" *
              "               -> $interaction_subs orbitals\n" *
              "Rebuild one of them so both describe the same orbitals.")
    say("Orbital count check: model and interactions agree at $subs orbitals.")

    interaction_params = interaction_data["parameters"]
    interaction_cases = interaction_data["values"]

    #####* The closed bilinear basis, when the interaction builder used one. Then the file holds
    # one constant 4d x 4d matrix per case plus the basis it is expressed in, and the bubble must
    # be contracted onto the same basis (`bilinear_basis: closed` for both stages). The residual
    # rotation between the two frames is measured and applied below.
    closed_matrices = get(interaction_data, "closed_matrices", nothing)
    interaction_basis = get(interaction_data, "bilinear_basis", nothing)
    if closed_matrices !== nothing
        interaction_basis === nothing &&
            error("$(input["interactions"]) carries 'closed_matrices' but no 'bilinear_basis'. " *
                  "The matrices' coefficients are meaningless without the basis they are " *
                  "expressed in. Rebuild the interaction file.")
        say("Interaction file carries a closed bilinear basis: d = $(length(interaction_basis)), " *
            "cases $(join(sort(collect(keys(closed_matrices))), ", ")).")
    end

    if run_bare
        say("Running the bare susceptibility calculation in TRIQS (every mu, unconditionally)...")
        Pipeline.run_bare(config)
        say("Bare susceptibility calculation complete!")
    end

    input_path_labels = default_path_labels(input)
    instability_lower, instability_upper = get_instability_range(input)
    strengths_lower, strengths_upper, strengths_n = get_strength_range(input)

    #####* every mu's bubble. They share one q grid and one basis, which is what lets each case's
    # interaction be built once below.
    bubbles = [load_bubble(input, mu, localDim, subs) for mu in get_mu_values(input)]
    ref = bubbles[begin]
    for b in bubbles
        (b.ks == ref.ks && b.ks_path == ref.ks_path && b.primitives == ref.primitives) ||
            error("The bubbles at mu = $(ref.mu) and mu = $(b.mu) are on different q grids or " *
                  "lattices. Rerun the bare stage so every mu comes from the same config.")
        (b.chi_basis === nothing) == (ref.chi_basis === nothing) &&
            (b.chi_basis === nothing || b.chi_basis == ref.chi_basis) ||
            error("The bubbles at mu = $(ref.mu) and mu = $(b.mu) were contracted onto different " *
                  "bilinear bases. Rerun the bare stage so every mu comes from the same config.")
    end
    if (ref.chi_basis === nothing) != (closed_matrices === nothing)
        error("The bubble and the interaction were built in different bilinear bases: the bubble is " *
              (ref.chi_basis === nothing ? "per-orbital" : "on a closed basis of dimension $(ref.chi_dim)") *
              " while the interaction file is " *
              (closed_matrices === nothing ? "per-orbital" : "closed") * ".\n" *
              "  Set `bilinear_basis: closed` (or remove it) in the config and rebuild BOTH " *
              "the interactions and the bare bubbles.")
    end

    #####* The unit-strength interaction of every case, on the grid and on the path.
    # interaction(strength, ks) == strength * interaction(1.0, ks), so these serve every mu and
    # every bisection step. The lookups are built serially, because each one reads the Params'
    # current value; the matrices are then built in parallel over cases.
    labels = String[]
    lookups = Dict{String, Any}()
    for (label, value) in interaction_cases
        if closed_matrices !== nothing && !haskey(closed_matrices, label)
            # A q-dependent tail has no constant closed-basis matrix, and the per-orbital
            # interaction() would give a 4*subs matrix against a 4d chi, so skip the case rather
            # than mix frames.
            say("Skipping the $(label) case: the interaction file has no closed-basis matrix " *
                "for it, and the bubble is on the closed basis. Only " *
                "$(join(sort(collect(keys(closed_matrices))), ", ")) can be run this way.")
            continue
        end
        push!(labels, label)
        if closed_matrices === nothing
            push!.(getproperty.(interaction_params, :value), value)
            lookups[label] = Lookup(interaction_params)
            pop!.(getproperty.(interaction_params, :value))
        else
            lookups[label] = nothing
        end
    end
    sort!(labels)

    unit_interactions = Vector{Any}(undef, length(labels))
    Threads.@threads for c in eachindex(labels)
        label = labels[c]
        build_start = time()
        if closed_matrices === nothing
            lookup = lookups[label]
            V = interaction(1.0, ref.ks; primitives = ref.primitives, subs = subs, localDim = localDim, lookup = lookup)
            V_path = interaction(1.0, ref.ks_path; primitives = ref.primitives, subs = subs, localDim = localDim, lookup = lookup)
            say("Built the unit-strength $(label) interaction ($(length(lookup)) bonds) on " *
                "$(length(ref.ks)) + $(length(ref.ks_path)) k points in $(round(time() - build_start, digits=1)) s.")
        else
            # One constant matrix: the on-site vertex is local, so V(q) has no q dependence.
            # Carried into the BUBBLE's frame first.
            O = basis_rotation(ref.chi_basis, interaction_basis)
            M = Matrix{ComplexF64}(rotate_channel_blocks(closed_matrices[label], O))
            size(M, 1) == localDim * ref.chi_dim ||
                error("closed-basis $(label) matrix is $(size(M, 1))x$(size(M, 1)) but " *
                      "the assembled chi needs $(localDim * ref.chi_dim). The interaction " *
                      "was built at basis_dim $(size(M, 1) ÷ 4) and the bubble at $(ref.chi_dim).")
            V = [M for _ in ref.ks]
            V_path = [M for _ in ref.ks_path]
            say("Using the closed-basis $(label) matrix ($(size(M, 1))x$(size(M, 1)), " *
                "q-independent); frame rotation from the interaction's basis to the bubble's " *
                "differs from the identity by $(round(maximum(abs.(O - I)), sigdigits=3)).")
        end
        unit_interactions[c] = (V = V, V_path = V_path)
    end

    #####* the (mu, case) pairs, in parallel
    tasks = [(b, c) for b in eachindex(bubbles) for c in eachindex(labels)]
    instabilities = Vector{Dict{String, Any}}(undef, length(tasks))
    say("Running $(length(tasks)) (mu, case) pairs on $(Threads.nthreads()) threads...")
    Threads.@threads for t in eachindex(tasks)
        b, c = tasks[t]
        bubble = bubbles[b]
        instabilities[t] = find_instability(bubble.chis, bubble.ks, unit_interactions[c].V;
                                            primitives = bubble.primitives,
                                            lower = instability_lower, upper = instability_upper)
        say("mu = $(bubble.mu), $(labels[c]): critical strength ≈ " *
            "$(round(instabilities[t]["critical strength"], digits=3)).")
    end

    CombinedOutput = Dict()
    for bubble in bubbles
        entry = CombinedOutput[bubble.parent_label] = Dict()
        entry["mu"] = bubble.mu
        entry["beta"] = input["beta"]
        entry["filling"] = Float64(only(bubble.triqs_data["filling"]))
        entry["triqs_data"] = bubble.triqs_data
        # combine_chis transposes TRIQS's contraction into the ordering the RPA ladder and the
        # Cooper form factors need (see its docstring); recorded so a consumer can refuse output
        # assembled under an older convention.
        entry["chi_orientation"] = CHI_ORIENTATION
        entry["combined_chis"] = bubble.chis
        entry["combined_chis_path"] = bubble.chis_path
        if bubble.chi_basis !== nothing
            entry["bilinear_basis"] = bubble.chi_basis
            entry["basis_dim"] = bubble.chi_dim
        end
    end

    candidate_strengths = collect(LinRange(strengths_lower, strengths_upper, strengths_n))
    for (t, (b, c)) in enumerate(tasks)
        bubble, label, instability = bubbles[b], labels[c], instabilities[t]
        entry = CombinedOutput[bubble.parent_label]
        entry["$(label)_interaction"] = unit_interactions[c].V
        entry["$(label)_interaction_path"] = unit_interactions[c].V_path
        entry[label] = instability

        if save_individual
            save(input["output"] * "_$(bubble.parent_label)_interactionID=$(label).jld2", instability)
        end

        if plot
            critical = instability["critical strength"]
            strengths = [s for s in candidate_strengths if s <= critical]
            isempty(strengths) &&
                say("No plotting strengths are <= critical for $(label) at mu = $(bubble.mu). " *
                    "Check the strengths range in the config.")
            path_labels = something(npz_path_labels(bubble.triqs_data), input_path_labels)
            mkpath(dirname(input["plots"]))
            for strength in strengths
                p = plot_chi(bubble.chis_path, [strength .* V for V in unit_interactions[c].V_path];
                             path_plot = bubble.triqs_data["path_plot"],
                             path_ticks = bubble.triqs_data["path_ticks"],
                             path_labels = path_labels)
                savefig(p, input["plots"] * "_chi_$(bubble.parent_label)_interactionID=$(label)_strength=$(round(strength, digits=3)).png")
            end
        end
    end

    output_file = rpa_output_path(input)
    mkpath(dirname(output_file))
    save(output_file, CombinedOutput)
    say("RPA output saved to $output_file")

    return CombinedOutput
end

end
