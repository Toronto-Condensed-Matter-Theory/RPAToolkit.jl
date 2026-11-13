# Command-line entry point for the RPA stage (see `run_rpa`):
#
#     julia --project=<env with RPAToolkit> -t <threads> src/run_RPA.jl --input config.yml \
#           [--run_bare true] [--plot_RPA true] [--save_individual true]

using ArgParse, LinearAlgebra
using RPAToolkit

function parse_commandline()

    settings = ArgParseSettings()

    @add_arg_table settings begin
        "--input"
            help = "the input yml file."
            arg_type = String
            required = true
        "--run_bare"
            help = "run the bare susceptibility calculation first."
            arg_type = Bool
            default = false
        "--plot_RPA"
            help = "plot the RPA susceptibility along the k-path."
            arg_type = Bool
            default = false
        "--save_individual"
            help = "also save each instability to its own file."
            arg_type = Bool
            default = false
    end

    return parse_args(settings)
end


if abspath(PROGRAM_FILE) == @__FILE__
    parsed_args = parse_commandline()

    # The (mu, case) pairs run on Julia threads, and each is a stream of small dense solves that
    # multithreaded BLAS would only oversubscribe.
    BLAS.set_num_threads(1)

    run_rpa(parsed_args["input"]; run_bare = parsed_args["run_bare"],
            plot = parsed_args["plot_RPA"], save_individual = parsed_args["save_individual"])
end
