module Pipeline

using JLD2, NPZ, YAML, LaTeXStrings, LinearAlgebra, Plots, TightBindingToolkit
using ..RPAToolkit.InputParser: load_rpa_input
using ..RPAToolkit.SpinSectors: SPIN_SYMMETRIES, channel_pairs, pair_label
using ..RPAToolkit.Bubble: band_model, band_spin_symmetry, spin_symmetry, spinless_model, channel_operators,
                           closed_basis_for_model, stored_bubble, k_grid, k_path, band_energies,
                           reciprocal_vectors, band_eigen
using ..RPAToolkit.Preprocess: dress_primitives, combine_chis, CHI_ORIENTATION
using ..RPAToolkit.SpinSectors: stored_spin_symmetry, spin_sectors, VERTEX_SLOTS
using ..RPAToolkit.Interactions: interaction, INTERACTION_CONVENTION
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

#####* k_points given as TightBindingToolkit HighSymPoints labels -> reduced coordinates
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

#####* one scan axis, from `values`, or `min`/`max`/`n`, or a bare `n` spanning `span`
function scan_axis(cfg, name::AbstractString; span = nothing)
    haskey(cfg, "values") && return Float64.(collect(cfg["values"]))
    all(k -> haskey(cfg, k), ("min", "max", "n")) &&
        return collect(range(Float64(cfg["min"]), Float64(cfg["max"]), length = Int(cfg["n"])))
    haskey(cfg, "n") && span !== nothing &&
        return collect(range(Float64(span[1]), Float64(span[2]), length = Int(cfg["n"])))
    error("$name must define `values`, or `min`+`max`+`n`" * (span === nothing ? "" : ", or a bare `n` spanning the bandwidth") *
          "; got keys $(sort(collect(keys(cfg)))).")
end

"""
    free_model(unitcell, k_size, beta) -> TightBindingToolkit.Model

The free model on TightBindingToolkit's `k_size × k_size` Brillouin-zone grid (the momenta of
[`k_grid`](@ref RPAToolkit.Bubble.k_grid), in another order), diagonalized, at `T = 1/beta`. Its bands
fix the scan's fillings and chemical potentials.
"""
function free_model(unitcell::UnitCell, k_size::Integer, beta::Real)
    bz = BZ(k_size)
    FillBZ!(bz, unitcell)
    H = Hamiltonian(unitcell, bz)
    DiagonalizeHamiltonian!(H; verbose = false)
    return Model(unitcell, bz, H; T = 1 / beta)
end

"""
    resolve_scan_values(params, model::Model) -> (mus, fillings, source)

The chemical potentials a run scans, the fillings they give on the bands of `model`
([`free_model`](@ref)), and which of the two the config specified (`"mus"` or `"fillings"`).

MU IS THE SCAN VARIABLE: the bubble is computed at a chemical potential and the filling is a label
counted from it. A config may ask for fillings instead; the mus are then found ONCE by inverting
filling(mu) (TightBindingToolkit's `GetMu!`), and from then on they are what the run is. The
resolved values are written back into the runtime YAML with `scan.source`, and a file that carries
`scan.source` is taken verbatim (only its fillings are recounted), so reading a resolved file again
is a no-op.
"""
function resolve_scan_values(params, model::Model)
    span = model.Ham.bandwidth
    fillings_of(mus) = [FindFilling(Float64(mu), model.Ham.bands, model.T, model.stat) for mu in mus]

    recorded = get(get(params, "scan", Dict()), "source", nothing)
    if recorded !== nothing
        mus = Float64.(collect(params["mus"]["values"]))
        return mus, fillings_of(mus), String(recorded)
    end
    has_mus, has_fillings = haskey(params, "mus"), haskey(params, "fillings")
    has_mus && has_fillings &&
        error("The config sets BOTH `mus` and `fillings`, two parameterisations of the same scan. Delete one.")
    has_mus || has_fillings || error("The config must define either `mus` or `fillings`.")
    if has_mus
        mus = scan_axis(params["mus"], "mus"; span = span)
        return mus, fillings_of(mus), "mus"
    end
    fillings = scan_axis(params["fillings"], "fillings")
    all(f -> 0 <= f <= 1, fillings) || error("fillings must lie in [0, 1], got $fillings.")
    mus = map(fillings) do f
        model.filling = f
        GetMu!(model; mu_tol = 1e-12, filling_tol = 1e-14)
        model.mu
    end
    return mus, fillings, "fillings"
end

"""
    mu_label(mu) -> String

The `mu` in the bubble file names and output keys, `"beta=<beta>_mu=<mu_label(mu)>"`: six significant
digits, so chemical potentials a fraction of a meV apart in a narrow band stay distinct. (Rounding to
three decimals, as before, merged the scan points of a 35 meV-wide band in eV into a handful of keys.)
"""
mu_label(mu) = string(round(Float64(mu), sigdigits = 6))

"""The label of bubble files written before [`mu_label`](@ref): `mu` rounded to three decimals."""
legacy_mu_label(mu) = string(round(Float64(mu), digits = 3))

"""
    run_bare(config) -> String

The bare-susceptibility stage.

Reads the model's `"unit cell"` from `unitcell.julia` and, for every chemical potential of the scan
(`mus`, or `fillings` inverted once on the bands), computes the static bubble on the
`k_size × k_size` grid and along the closed path through `k_points`. Writes one
`<output>_beta=<beta>_mu=<mu>.npz` per mu, in the stored layout and convention ([`stored_bubble`](@ref)), and the resolved config (k-points as coordinates, the resolved `mus`
and `fillings` under `scan.source`) to `<output>_runtime_input.yml`, which it returns.

Config keys: `bubble` (`lindhard`, the exact sum and the default, or `dlr`, the imaginary-time FFT
route that scales as N_k log N_k, with `w_max` its spectral cutoff and `dlr_err` its accuracy),
`bubble_spin` (`auto`, or `full` to skip the SU(2) factorization), `bilinear_basis`
(`per_orbital` or `closed`, with `bilinear_seed`) and `directions`. Threaded over momenta.
"""
function run_bare(config::AbstractString)
    input = load_rpa_input(config)
    runtime_input = deepcopy(input)

    unitcell = load(input["unitcell"]["julia"])["unit cell"]
    resolve_k_points!(runtime_input, unitcell)
    haskey(runtime_input, "k_points") || error("The config must give `k_points` for the path.")

    bm = band_model(unitcell)
    k_size = Int(input["k_size"])
    beta = Float64(input["beta"])
    directions = sort(Int.(input["directions"]))
    n_orb = length(unitcell.basis)

    #####* the operator basis
    basis_mode = lowercase(String(get(input, "bilinear_basis", "per_orbital")))
    basis_mode in ("per_orbital", "closed") ||
        error("bilinear_basis must be 'per_orbital' or 'closed', got $(repr(basis_mode)).")

    #####* which channel pairs, and whether the spinless bubble serves for all of them
    bubble_spin = lowercase(String(get(input, "bubble_spin", "auto")))
    bubble_spin in ("auto", "full") || error("bubble_spin must be 'auto' or 'full', got $(repr(bubble_spin)).")
    model_symmetry = band_spin_symmetry(bm)
    symmetry = bubble_spin == "auto" ? model_symmetry : "none"
    factorized = symmetry == "SU2"
    symmetry == "SU2" || directions == [0, 1, 2, 3] ||
        error("directions = $directions: a $symmetry bubble mixes the charge and spin channels, so it needs all four, [0, 1, 2, 3].")
    basis = nothing
    if basis_mode == "closed"
        model_symmetry == "SU2" ||
            error("bilinear_basis: closed needs a model without spin-orbit coupling, and this one is $model_symmetry.")
        basis = closed_basis_for_model(bm; seed = String(get(input, "bilinear_seed", "kanamori")), verbose = true)
    end
    d = basis === nothing ? n_orb : length(basis)
    pairs = [(a, b) for (a, b) in channel_pairs(symmetry) if a in directions && b in directions]

    method = Symbol(lowercase(String(get(input, "bubble", "lindhard"))))
    method in (:lindhard, :dlr) || error("bubble must be 'lindhard' or 'dlr', got $(repr(method)).")
    w_max = Float64(get(input, "w_max", 20.0))
    dlr_err = Float64(get(input, "dlr_err", 1e-12))

    model = factorized ? spinless_model(bm) : bm
    ops = factorized ? channel_operators(n_orb; basis = basis, spinful = false) :
                       channel_operators(n_orb; basis = basis, directions = 0:3)
    factor = factorized ? 0.5 : 1.0       # Tr[(sigma^a / 2)^2] on the spinless bubble

    ks = k_grid(bm.primitives, k_size)
    path, path_plot, path_ticks = k_path(bm.primitives, runtime_input["k_points"])
    free = free_model(unitcell, k_size, beta)
    bandwidth = collect(free.Ham.bandwidth)

    say("model spin symmetry: $model_symmetry; bare bubble: $method" *
        (method == :dlr ? " (w_max = $w_max, eps = $dlr_err)" : "") * ", " *
        (factorized ? "spin-factorized (spinless bubble x 1/2)" : "spinful, $(length(pairs)) channel pairs") *
        ", basis dimension $d, $(length(ks)) + $(length(path)) momenta on $(Threads.nthreads()) threads")

    mus, fillings, source = resolve_scan_values(runtime_input, free)
    labels = mu_label.(mus)
    allunique(labels) || error("Two scan points share the file label mu = " *
        "$(first(l for l in labels if count(==(l), labels) > 1)); their bubbles would overwrite each other. " *
        "Space the scan points further apart.")
    runtime_input["scan"] = Dict("source" => source,
                                 "requested" => get(get(runtime_input, "scan", Dict()), "requested",
                                                    get(runtime_input, source, Dict())))
    runtime_input["mus"] = Dict("values" => mus, "n" => length(mus))
    runtime_input["fillings"] = Dict("values" => fillings, "n" => length(fillings))
    runtime_input_file = runtime_input_path(input)
    mkpath(dirname(runtime_input_file))
    YAML.write_file(runtime_input_file, runtime_input)
    say("scan: $(length(mus)) point(s), specified as `$source`; mu $(minimum(mus)) .. $(maximum(mus)), " *
        "filling $(minimum(fillings)) .. $(maximum(fillings))")

    #####* the stored channel-pair blocks of a full (4d x 4d, channel innermost) or spinless (d x d) bubble
    block(S, a, b) = factorized ? factor .* S : S[[4(i - 1) + a + 1 for i in 1:d], [4(j - 1) + b + 1 for j in 1:d]]
    function stack(mats)
        out = Array{ComplexF64}(undef, length(mats), size(mats[1])...)
        for (i, M) in enumerate(mats)
            out[i, :, :] = M
        end
        return out
    end

    eig_k = band_eigen(model, ks)
    for (index, mu) in enumerate(mus)
        start = time()
        grid = stored_bubble(model, ks, k_size; beta = beta, mu = mu, ops = ops, method = method,
                             on_grid = true, euv = w_max, rtol = dlr_err, eig_k = eig_k)
        along = stored_bubble(model, path, k_size; beta = beta, mu = mu, ops = ops, eig_k = eig_k)

        #####* The pairs S_z conservation says are zero, measured once rather than assumed.
        if symmetry == "U1z" && index == 1
            scale = maximum(maximum(abs, S) for S in grid)
            skipped = [(a, b) for (a, b) in channel_pairs("none") if (a, b) ∉ channel_pairs("U1z")]
            worst = maximum(maximum(abs, block(S, a, b)) for S in grid for (a, b) in skipped)
            say("U1z sector check: max |chi| over the $(length(skipped)) skipped channel pairs is $worst against $scale")
            worst <= 1e-10 * scale ||
                error("The model has no spin-flip hopping, so the {N, Z} x {X, Y} channel pairs must vanish, but they " *
                      "reach $(worst / scale) of max |chi|. Set `bubble_spin: full` to keep every pair.")
        end

        output = Dict{String, Any}()
        for (a, b) in pairs
            output[pair_label(a, b)] = stack([block(S, a, b) for S in grid])
            output[pair_label(a, b) * "_path"] = stack([block(S, a, b) for S in along])
        end
        if basis !== nothing
            output["bilinear_dim"] = [d]
            output["bilinear_basis"] = stack(basis)
        end
        pad(v) = [v[1], v[2], 0.0]
        b1, b2 = reciprocal_vectors(bm.primitives)
        merge!(output, Dict{String, Any}(
            "beta" => [beta], "mu" => [mu], "filling" => [fillings[index]],
            "bubble_spin_factorized" => [Int(factorized)],
            "spin_symmetry" => [findfirst(==(symmetry), SPIN_SYMMETRIES) - 1],
            "primitives" => permutedims(hcat(pad(bm.primitives[1]), pad(bm.primitives[2]))),
            "reciprocal" => permutedims(hcat(pad(b1), pad(b2), [0.0, 0.0, 2pi])),
            "ks" => permutedims(reduce(hcat, pad.(ks))), "contracted" => permutedims(reduce(hcat, pad.(ks))),
            "path" => permutedims(reduce(hcat, pad.(path))), "path_plot" => path_plot, "path_ticks" => path_ticks,
            "bandwidth" => bandwidth, "bands" => permutedims(band_energies(bm, path))))
        file = String(input["output"]) * "_beta=$(input["beta"])_mu=$(mu_label(mu)).npz"
        mkpath(dirname(file))
        npzwrite(file, output)
        say("mu = $mu (filling $(round(fillings[index], digits = 6))): bubble written in $(round(time() - start, digits = 1)) s")
    end

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

function npz_path_labels(bare_data)
    if !haskey(bare_data, "k_point_labels")
        return nothing
    end

    labels = vec(bare_data["k_point_labels"])
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
function load_bubble(input, mu, localDim, subs, model_symmetry)
    parent_label = "beta=$(input["beta"])_mu=$(mu_label(mu))"
    file = input["output"] * "_$(parent_label).npz"
    if !isfile(file)        # a bubble from before mu_label: three decimals
        legacy = input["output"] * "_beta=$(input["beta"])_mu=$(legacy_mu_label(mu)).npz"
        isfile(legacy) && (file = legacy)
    end
    bare_data = npzread_numeric(file)

    # Which channel pairs the bubble holds. Contracted under less symmetry than the model has is
    # fine (bubble_spin: full); under more means it came from a different model.
    symmetry = stored_spin_symmetry(bare_data; model_symmetry = model_symmetry)
    level(sym) = findfirst(==(sym), ("SU2", "U1z", "none"))
    level(symmetry) >= level(model_symmetry) ||
        error("The bubble at $parent_label was contracted as $symmetry, but the model is " *
              "$model_symmetry: it holds too few channel pairs for this model. Rerun the bare stage.")

    ks_contracted = bare_data["contracted"]
    ks = Vector{eltype(ks_contracted)}[eachrow(ks_contracted)...]
    ks_path_matrix = bare_data["path"]
    ks_path = Vector{eltype(ks_path_matrix)}[eachrow(ks_path_matrix)...]

    # The basis the bare stage contracted this bubble onto. Absent on the per-orbital path, where
    # the block size is the orbital count.
    chi_basis = let stored = get(bare_data, "bilinear_basis", nothing)
        (stored === nothing || size(stored, 1) == 0) ? nothing :
            [Matrix{ComplexF64}(stored[k, :, :]) for k in axes(stored, 1)]
    end
    chi_dim = chi_basis === nothing ? subs : length(chi_basis)

    chis = combine_chis(bare_data; directions = input["directions"], subs = chi_dim,
                        model_symmetry = model_symmetry)
    chis_path = combine_chis(bare_data; directions = input["directions"], subs = chi_dim, suffix = "_path",
                             model_symmetry = model_symmetry)
    size(chis[begin], 1) == localDim * chi_dim ||
        error("assembled chi at $parent_label is $(size(chis[begin], 1))x$(size(chis[begin], 1)) " *
              "but the stored basis has dimension $chi_dim over $localDim directions, which " *
              "requires $(localDim * chi_dim). The bubble file and its basis disagree.")

    return (mu = mu, parent_label = parent_label, bare_data = bare_data,
            primitives = dress_primitives(bare_data), ks = ks, ks_path = ks_path,
            chi_basis = chi_basis, chi_dim = chi_dim, chis = chis, chis_path = chis_path,
            spin_symmetry = symmetry)
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
`"filling"`, `"bare_data"` (the bubble file), `"combined_chis"` and `"combined_chis_path"` (from
[`combine_chis`](@ref)), `"chi_orientation"`, `"spin_symmetry"` (the channel pairs chi holds, see
[`spin_sectors`](@ref)), `"vertex_slots"`, `"<case>_interaction"` and `"<case>_interaction_path"`
(the unit-strength `V(q)`), and `"<case>"` (the `find_instability` result, with the
`"unstable sector"`).

The interactions file must record `"vertex_slots" => VERTEX_SLOTS`. The RPA is solved one spin
sector at a time: the model's spin symmetry ([`spin_symmetry`](@ref)) makes `chi` block diagonal
in them, so a model without spin-orbit coupling solves its charge and one spin block, and one
with Ising spin-orbit coupling its longitudinal (N, Z) and transverse (X, Y) blocks. On the closed bilinear basis it also holds `"bilinear_basis"` and `"basis_dim"`.
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

    #####* The channel layout of the interaction. Files from before the physical slots carried the
    # spin vertex in N and the charge vertex in Z, which reads as the wrong physics now, and
    # silently so: refuse them.
    get(interaction_data, "vertex_slots", nothing) == VERTEX_SLOTS ||
        error("Interactions file $(input["interactions"]) does not record the channel layout " *
              "\"$(VERTEX_SLOTS)\" under \"vertex_slots\" (it has " *
              "$(repr(get(interaction_data, "vertex_slots", nothing)))). Files built before " *
              "2026-11-01 put the spin vertex in slot N and the charge vertex in slot Z. Rebuild it " *
              "with HUBBARD_CHANNELS/DENSITY_CHANNELS or channel_blocks, saving " *
              "\"vertex_slots\" => VERTEX_SLOTS.")

    model_symmetry = spin_symmetry(unitcell)
    say("Model spin symmetry: $model_symmetry.")

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
        say("Running the bare susceptibility calculation (every mu, unconditionally)...")
        Pipeline.run_bare(config)
        say("Bare susceptibility calculation complete!")
    end

    input_path_labels = default_path_labels(input)
    instability_lower, instability_upper = get_instability_range(input)
    strengths_lower, strengths_upper, strengths_n = get_strength_range(input)

    #####* every mu's bubble. They share one q grid and one basis, which is what lets each case's
    # interaction be built once below.
    bubbles = [load_bubble(input, mu, localDim, subs, model_symmetry) for mu in get_mu_values(input)]
    ref = bubbles[begin]
    for b in bubbles
        (b.ks == ref.ks && b.ks_path == ref.ks_path && b.primitives == ref.primitives) ||
            error("The bubbles at mu = $(ref.mu) and mu = $(b.mu) are on different q grids or " *
                  "lattices. Rerun the bare stage so every mu comes from the same config.")
        b.spin_symmetry == ref.spin_symmetry ||
            error("The bubbles at mu = $(ref.mu) and mu = $(b.mu) were contracted as " *
                  "$(ref.spin_symmetry) and $(b.spin_symmetry). Rerun the bare stage.")
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
                                            lower = instability_lower, upper = instability_upper,
                                            sectors = spin_sectors(bubble.spin_symmetry, bubble.chi_dim))
        say("mu = $(bubble.mu), $(labels[c]): critical strength ≈ " *
            "$(round(instabilities[t]["critical strength"], digits=3)) " *
            "($(instabilities[t]["unstable sector"])).")
    end

    CombinedOutput = Dict()
    for bubble in bubbles
        entry = CombinedOutput[bubble.parent_label] = Dict()
        entry["mu"] = bubble.mu
        entry["beta"] = input["beta"]
        entry["filling"] = Float64(only(bubble.bare_data["filling"]))
        entry["bare_data"] = bubble.bare_data
        # combine_chis transposes the stored contraction into the ordering the RPA ladder and the
        # Cooper form factors need (see its docstring); recorded so a consumer can refuse output
        # assembled under an older convention.
        entry["chi_orientation"] = CHI_ORIENTATION
        # Which channel pairs chi holds, and the layout of the interaction matrices beside it.
        entry["spin_symmetry"] = bubble.spin_symmetry
        entry["vertex_slots"] = VERTEX_SLOTS
        entry["interaction_convention"] = INTERACTION_CONVENTION
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
            path_labels = something(npz_path_labels(bubble.bare_data), input_path_labels)
            mkpath(dirname(input["plots"]))
            for strength in strengths
                p = plot_chi(bubble.chis_path, [strength .* V for V in unit_interactions[c].V_path];
                             path_plot = bubble.bare_data["path_plot"],
                             path_ticks = bubble.bare_data["path_ticks"],
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
