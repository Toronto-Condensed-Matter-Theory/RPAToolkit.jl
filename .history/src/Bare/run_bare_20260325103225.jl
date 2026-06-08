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

function resolve_k_points!(input, unitcell)
    if !haskey(input, "k_points")
        return
    end

    bz = BZ([input["k_size"], input["k_size"]])
    FillBZ!(bz, unitcell)

    resolved = Any[]
    for point in input["k_points"]
        if point isa AbstractString
            if !haskey(bz.HighSymPoints, point)
                available = join(sort(collect(keys(bz.HighSymPoints))), ", ")
                error("Unknown HighSymPoint '$(point)'. Available points: $(available)")
            end
            push!(resolved, _normalize_kpoint_coordinates(bz.HighSymPoints[point]))
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

    model = load(input["unitcell"]["julia"])

    unitcell = model["unit cell"]
    parameters = model["parameters"]
    triqs_input = input["unitcell"]["julia"][1:end-5] * ".npz"

    resolve_k_points!(input, unitcell)

    parse_unitcell(unitcell, triqs_input)

    input["unitcell"]["triqs"] = triqs_input
    YAML.write_file(parsed_args["input"], input)

    command = `$(input["triqs_environment"]) $(@__DIR__)/run_bare.py $(parsed_args["input"])`
    run(command)

    command = `$(input["triqs_environment"]) $(@__DIR__)/plot_bare.py $(parsed_args["input"])`
    run(command)

end
