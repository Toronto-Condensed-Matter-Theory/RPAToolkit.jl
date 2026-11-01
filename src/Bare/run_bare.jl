# Command-line entry point for the bare-susceptibility stage (see `run_bare`):
#
#     julia --project=<env with RPAToolkit> src/Bare/run_bare.jl --input config.yml

using ArgParse
using RPAToolkit

function parse_commandline()

    settings = ArgParseSettings()

    @add_arg_table settings begin
        "--input"
        help = "the input yml file."
        arg_type = String
        required = true
    end

    return parse_args(settings)
end


if abspath(PROGRAM_FILE) == @__FILE__
    run_bare(parse_commandline()["input"])
end
