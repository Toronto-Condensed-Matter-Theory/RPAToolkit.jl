#=
The interface to MeanFieldToolkit, whose methods are in ext/RPAToolkitMeanFieldToolkitExt.jl and
load with `using MeanFieldToolkit`. With them, a converged Hartree-Fock run becomes the input of the
RPA, and the RPA on those bands with the vertex built from the same interaction is the linear
response of that Hartree-Fock state (time-dependent Hartree-Fock).
=#

"""
    rpa_model(mft; residual = nothing, tol = 1e-6) -> NamedTuple

The Hartree-Fock bands of a converged MeanFieldToolkit `TBMFTModel`, ready for
[`dynamic_bubble`](@ref RPAToolkit.Bubble.dynamic_bubble) / [`stored_bubble`](@ref): `(bands, mu, beta, filling, unitcell)`, with
`bands` the [`BandModel`](@ref) of `mft.model.uc` (the free hoppings plus the decoupled
interaction) and `mu`, `beta` its chemical potential and inverse temperature.

Pass the self-consistency `residual` (`maximum(abs, sc.VOuts[end] - sc.VIns[end])`) to have it
checked against `tol`: MeanFieldToolkit builds the final unit cell from the last INPUT order
parameters, so the bands, and with them the Goldstone mode of a broken continuous symmetry, are off
by O(residual).

Needs `using MeanFieldToolkit` (a package extension).
"""
function rpa_model end

"""
    rpa_vertex(mft; interactions = nothing) -> Function

`q -> V(q)`: the TDHF vertex of a MeanFieldToolkit `TBMFTModel`'s interactions, in the layout of
[`interaction`](@ref) (`4 × n_sites` square, channel innermost), so that
`dynamic_rpa(dynamic_bubble(...), V.(qs))` is the linear response of the Hartree-Fock state.

On-site interactions (`IntraQuarticToHopping`) enter with their full Hartree and Fock kernel
([`quartic_onsite_vertex`](@ref)): the same interaction tensor MeanFieldToolkit decoupled, so a
spontaneously broken continuous symmetry gives an exact Goldstone mode. Inter-site interactions
(`InterQuarticToHopping`) enter with their Hartree part ([`quartic_hartree_vertex`](@ref)); their
exchange part renormalizes the bands but is not in the vertex, and a warning says so.
`interactions` (indices into `mft.Interactions`) keeps only those terms, e.g. to take the on-site
vertex from here and a long-range tail from [`coulomb_2d`](@ref), summing the matrices.

Needs `using MeanFieldToolkit` (a package extension).
"""
function rpa_vertex end
