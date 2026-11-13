# The bare and RPA susceptibilities of an example along its k-path, and the critical strength
# against filling, from the output of run_rpa.
#
#     julia --project=. examples/plot_susceptibilities.jl examples/Square/Square.yml
#
# For every mu, saves <plots>chi_bare_mu=<mu>.png and, for every interaction case that goes unstable,
# <plots>chi_RPA_<case>_mu=<mu>.png at 98% of its critical strength. With more than one mu, also
# saves <plots>critical_<case>.png.

using RPAToolkit, JLD2, Plots, LaTeXStrings
using RPAToolkit.Plotting: plot_strengthsVsN

length(ARGS) == 1 || error("usage: julia --project=. examples/plot_susceptibilities.jl <config.yml>")
input = load_rpa_input(ARGS[1])
output = load(rpa_output_path(input))
mkpath(dirname(input["plots"]))

labels = [L"%$(label)" for label in input["k_labels"]]
push!(labels, labels[1])                      # the path closes back to its first point
upper = Float64(get(get(input, "instability", Dict()), "upper", 10.0))

cases = String[]
for (key, data) in output
    mu = round(data["mu"], digits = 3)
    path = (path_plot = data["bare_data"]["path_plot"], path_ticks = data["bare_data"]["path_ticks"],
            path_labels = labels)

    p = plot_chi(data["combined_chis_path"]; path..., title = L"\chi_0(Q)" * ", μ = $mu")
    savefig(p, input["plots"] * "chi_bare_mu=$mu.png")

    for case in filter(k -> haskey(data, k * "_interaction_path"), collect(keys(data)))
        case in cases || push!(cases, case)
        critical = data[case]["critical strength"]
        critical < 0.999 * upper || continue  # never went unstable in the searched range
        strength = 0.98 * critical
        p = plot_chi(data["combined_chis_path"], [strength .* V for V in data["$(case)_interaction_path"]];
                     path..., title = "RPA, $case at $(round(strength, digits = 3)), μ = $mu")
        savefig(p, input["plots"] * "chi_RPA_$(case)_mu=$mu.png")
    end
end

if length(output) > 1
    for case in cases
        savefig(plot_strengthsVsN(output, case), input["plots"] * "critical_$case.png")
    end
end
println("Plots saved with prefix $(input["plots"])")
