module RPAToolkit

# Write your package code here.
include("./Bare/parse_model.jl")
using .parse_model
export parse_unitcell

include("Preprocess.jl")
using .Preprocess
export dress_primitives, dress_reciprocal, combine_chis, get_reciprocal_ks

include("Interactions.jl")
using .Interactions
export interaction

    prefix = basename(output_target)
    return joinpath(runtime_dir, "$(prefix)_runtime_input.yml")

include("Response.jl")
using .Response
export perform_RPA, minima, maxima, find_instability, effective_interaction

include("Plotting.jl")
using .Plotting
export plot_chi



end
