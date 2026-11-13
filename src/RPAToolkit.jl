module RPAToolkit

include("./Bare/parse_model.jl")
using .parse_model
export parse_unitcell

# The spin symmetry of a model and the channel sectors it makes chi and V block diagonal in.
include("SpinSectors.jl")
using .SpinSectors
export SPIN_SYMMETRIES, VERTEX_SLOTS, channel_pairs, pair_label, spin_symmetry,
       stored_spin_symmetry, spin_sectors, off_sector_norm

include("Preprocess.jl")
using .Preprocess
export dress_primitives, dress_reciprocal, combine_chis, get_reciprocal_ks, CHI_ORIENTATION

include("InputParser.jl")
using .InputParser
export load_rpa_input

include("Interactions.jl")
using .Interactions
export interaction, INTERACTION_CONVENTION, HUBBARD_CHANNELS, DENSITY_CHANNELS, hubbard_coulomb_interactions,
       tdhf_channel_vertex, quartic_onsite_vertex, quartic_hartree_vertex

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

# The discrete Lehmann representation (adapted from Lehmann.jl), for the imaginary-time bubble.
include("DLR.jl")
using .DLR
export DLRGrid, dlr_grid, matsubara_weights

# The bare bubble, static and real-frequency, in Julia.
include("Bubble.jl")
using .Bubble
export BandModel, band_model, bloch_hamiltonian, spinless_model, band_spin_symmetry,
       reciprocal_vectors, k_grid, k_path, fermi, band_energies, filling_at, mu_from_filling,
       channel_operators, closed_basis_for_model, static_bubble, static_bubble_dlr, stored_bubble,
       dynamic_bubble, lindhard_weight, band_eigen, particle_hole_edges

include("Response.jl")
using .Response
export perform_RPA, minima, maxima, find_instability, effective_interaction, sector_plan

# The RPA at real frequency: collective modes (magnons, plasmons) on top of dynamic_bubble.
include("Dynamics.jl")
using .Dynamics
export dynamic_rpa, unfold_vector, spectral_weight, spectral_map, coulomb_2d, rpa_poles

# Mean-field theory -> RPA: methods live in the MeanFieldToolkit extension.
include("MeanFieldInterface.jl")
export rpa_model, rpa_vertex

include("Plotting.jl")
using .Plotting
export plot_chi

# The two stages of a run, driven by one YAML config.
include("Pipeline.jl")
using .Pipeline
export run_bare, run_rpa, rpa_output_path

end
