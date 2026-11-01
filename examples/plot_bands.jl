# Band structure of an example's model along its k-path.
#
#     julia --project=. examples/plot_bands.jl examples/Square/Square.yml
#
# Saves <plots>bands.png.

using RPAToolkit, TightBindingToolkit, JLD2, Plots, LaTeXStrings

length(ARGS) == 1 || error("usage: julia --project=. examples/plot_bands.jl <config.yml>")
input = load_rpa_input(ARGS[1])

unitcell = load(input["unitcell"]["julia"])["unit cell"]
bz = BZ([input["k_size"], input["k_size"]])
FillBZ!(bz, unitcell)
H = Hamiltonian(unitcell, bz)
DiagonalizeHamiltonian!(H)

k_points = [bz.HighSymPoints[name] for name in input["k_points"]]
labels = [L"%$(label)" for label in input["k_labels"]]

p = Plot_Band_Structure(Model(unitcell, bz, H), k_points; labels = labels, closed = true,
                        nearest = true, plot_legend = false)

file = input["plots"] * "bands.png"
mkpath(dirname(file))
savefig(p, file)
println("Band structure saved to $file")
