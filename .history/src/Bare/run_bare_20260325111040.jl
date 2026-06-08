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

    include("../RPAToolbox.jl")
    using .RPAToolbox

    parsed_args = parse_commandline()
    input = YAML.load_file(parsed_args["input"])
    runtime_input = deepcopy(input)

    model = load(input["unitcell"]["julia"])

    unitcell = model["unit cell"]
    parameters = model["parameters"]
    triqs_input = input["unitcell"]["julia"][1:end-5] * ".npz"

    resolve_k_points!(runtime_input, unitcell)

    parse_unitcell(unitcell, triqs_input)

    runtime_input["unitcell"]["triqs"] = triqs_input

    output_target = String(get(runtime_input, "output", dirname(parsed_args["input"])))
    runtime_dir = (endswith(output_target, "/") || isdir(output_target)) ? output_target : dirname(output_target)
    mkpath(runtime_dir)

    (temp_input_file, temp_io) = mktemp(runtime_dir)
    close(temp_io)
    YAML.write_file(temp_input_file, runtime_input)

    try
        command = `$(input["triqs_environment"]) $(@__DIR__)/run_bare.py $(temp_input_file)`
        run(command)

        command = `$(input["triqs_environment"]) $(@__DIR__)/plot_bare.py $(temp_input_file)`
        run(command)
    finally
        rm(temp_input_file; force = true)
    end

end
