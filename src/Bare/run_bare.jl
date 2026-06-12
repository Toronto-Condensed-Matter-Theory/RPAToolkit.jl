using ArgParse, JLD2, TightBindingToolkit, YAML

function _normalize_kpoint_coordinates(point)
    values = collect(point)
    if length(values) == 2
        return [Float64(values[1]), Float64(values[2]), 0.0]
    elseif length(values) == 3
        return [Float64(values[1]), Float64(values[2]), Float64(values[3])]
    end

    error("Each k-point must have length 2 or 3, got length $(length(values)).")
end

function _canonical_high_symmetry_label(label::AbstractString)
    aliases = Dict(
        "G" => "G", "Gamma" => "G", "\\Gamma" => "G", "Γ" => "G",
        "K1" => "K1", "K_1" => "K1",
        "M2" => "M2", "M_2" => "M2",
    )
    return get(aliases, String(label), String(label))
end

function _reciprocal_basis_matrix(unitcell)
    a1 = Float64.(unitcell.primitives[1])
    a2 = Float64.(unitcell.primitives[2])
    return 2*pi * inv(transpose(hcat(a1, a2)))
end

function _cartesian_to_reduced(point, reciprocal_basis)
    reduced = reciprocal_basis \ Float64.(collect(point))
    return [abs(value) < 1e-12 ? 0.0 : value for value in reduced]
end

function resolve_k_points!(input, unitcell)
    if !haskey(input, "k_points")
        return
    end

    bz = BZ([input["k_size"], input["k_size"]])
    FillBZ!(bz, unitcell)
    reciprocal_basis = _reciprocal_basis_matrix(unitcell)

    resolved = Any[]
    for point in input["k_points"]
        if point isa AbstractString
            label = _canonical_high_symmetry_label(point)
            if !haskey(bz.HighSymPoints, label)
                available = join(sort(collect(keys(bz.HighSymPoints))), ", ")
                error("Unknown HighSymPoint '$(point)'. Available points: $(available)")
            end
            reduced = _cartesian_to_reduced(bz.HighSymPoints[label], reciprocal_basis)
            push!(resolved, _normalize_kpoint_coordinates(reduced))
        else
            push!(resolved, _normalize_kpoint_coordinates(point))
        end
    end

    input["k_points"] = resolved
end

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


if abspath(PROGRAM_FILE) == @__FILE__

    include("../RPAToolkit.jl")
    using .RPAToolkit

    parsed_args = parse_commandline()
    input = YAML.load_file(parsed_args["input"])
    runtime_input = deepcopy(input)

    model = load(input["unitcell"]["julia"])

    unitcell = model["unit cell"]
    parameters = haskey(model, "parameters") ? model["parameters"] : []
    triqs_input = input["unitcell"]["julia"][1:end-5] * ".npz"

    resolve_k_points!(runtime_input, unitcell)

    parse_unitcell(unitcell, triqs_input)

    runtime_input["unitcell"]["triqs"] = triqs_input

    output_target = String(get(runtime_input, "output", dirname(parsed_args["input"])))
    runtime_dir = (endswith(output_target, "/") || isdir(output_target)) ? output_target : dirname(output_target)
    mkpath(runtime_dir)
    prefix = basename(output_target)
    runtime_input_file = joinpath(runtime_dir, "$(prefix)_runtime_input.yml")
    YAML.write_file(runtime_input_file, runtime_input)

    env_args = split(input["triqs_environment"])
    
    # Prevent thread explosion (100 workers * 100 threads) by forcing 1 thread per python process
    py_env = copy(ENV)
    py_env["OMP_NUM_THREADS"] = "1"
    py_env["MKL_NUM_THREADS"] = "1"
    py_env["OPENBLAS_NUM_THREADS"] = "1"

    triqs_env_cmd = String.(split(input["triqs_environment"]))
    command = setenv(Cmd(vcat(triqs_env_cmd, [joinpath(@__DIR__, "run_bare.py"), runtime_input_file])), py_env)
    run(command)

    command_plot = setenv(Cmd(vcat(triqs_env_cmd, [joinpath(@__DIR__, "plot_bare.py"), runtime_input_file])), py_env)
    run(command_plot)
end
