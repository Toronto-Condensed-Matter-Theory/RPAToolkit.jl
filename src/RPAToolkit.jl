module RPAToolkit

# Write your package code here.
include("./Bare/parse_model.jl")
using .parse_model
export parse_unitcell

include("Preprocess.jl")
using .Preprocess
export dress_primitives, dress_reciprocal, combine_chis, get_reciprocal_ks

include("InputParser.jl")
using .InputParser
export load_rpa_input

include("Interactions.jl")
using .Interactions
export interaction

# The symmetry-closed bilinear basis, and the interaction vertex projected onto
# it. Exported from the package because both are needed OUTSIDE run_bare: an
# interaction builder has to project its vertex onto the same basis the bubble
# was contracted with, and run_RPA has to check that the two agree.
include("BilinearBasis.jl")
using .BilinearBasis
export bilinear_closure, density_seed, same_site_seed, lattice_point_group_2d,
       orbital_rep_from_bonds, canonical_frame, conjugation_matrix

include("BilinearVertex.jl")
using .BilinearVertex
export kanamori_vertex, project_vertex, channel_blocks, uncovered_bilinears,
       basis_rotation, rotate_channel_blocks


include("Response.jl")
using .Response
export perform_RPA, minima, maxima, find_instability, effective_interaction

include("Plotting.jl")
using .Plotting
export plot_chi



end
