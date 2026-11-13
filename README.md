# RPAToolkit.jl

[![Build Status](https://github.com/Toronto-Condensed-Matter-Theory/RPAToolkit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Toronto-Condensed-Matter-Theory/RPAToolkit.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://Toronto-Condensed-Matter-Theory.github.io/RPAToolkit.jl/dev/)

RPAToolkit.jl is a Julia package for the random phase approximation (RPA) on lattice models. It
computes the bare susceptibility of a tight-binding model, statically or at real frequency. Then,
for any set of two-body interactions, it finds the strength at which the static RPA first goes
unstable and at which ordering wavevector, or follows the collective modes (magnons, plasmons) of
the dynamic RPA.

Currently supported:
* Lattices and hoppings are defined with [TightBindingToolkit.jl](https://github.com/Toronto-Condensed-Matter-Theory/TightBindingToolkit.jl). Any 2D unit cell with any number of sites and orbitals is supported.
* Interactions are TightBindingToolkit `Param`s in the (N, X, Y, Z) channels: on-site, inter-site and long-range terms alike. An on-site Hubbard plus `1/r` Coulomb builder is included.
* Several interaction cases run against the same bubbles, over a scan of fillings or chemical potentials, in parallel.
* For multi-orbital models whose orbitals rotate into each other under the point group, the susceptibility and the Kanamori vertex can be put on a symmetry-closed bilinear basis.
* Spin-orbit coupling: the bubble keeps every channel pair the model's spin symmetry allows (all sixteen in general, the {N, Z} and {X, Y} sectors with Ising coupling), and the RPA is solved one spin sector at a time. Without spin-orbit coupling the spinless bubble is used.
* Outputs include the critical strength, the leading instability and its momentum, the RPA susceptibility, and the screened interaction.
* The bare bubble is computed in Julia, threaded: the exact Lindhard sum, or an imaginary-time FFT route on a discrete Lehmann representation that scales as `N_k log N_k`, for large grids. Both reproduce the TRIQS/triqs_tprf numbers the package used to rely on.
* Real-frequency RPA: the retarded bubble `χ₀(q, ω + iη)`, the RPA at every frequency, spectral functions unfolded from supercells, and an Ewald-summed long-range Coulomb `V(q)` for two-dimensional plasmons.
* On top of [MeanFieldToolkit.jl](https://github.com/Toronto-Condensed-Matter-Theory/MeanFieldToolkit.jl) Hartree-Fock bands (a package extension): the RPA with the vertex built from the same interaction is time-dependent Hartree-Fock, with exact Goldstone modes in broken-symmetry states.

## Installation

`] add https://github.com/Toronto-Condensed-Matter-Theory/RPAToolkit.jl`

Everything is Julia. (Up to version 1.0 the bare bubble was computed with TRIQS; that code is kept in
`test/triqs_reference/` only to regenerate the test fixtures.) For the mean-field extension, also add
MeanFieldToolkit and `using MeanFieldToolkit` next to `using RPAToolkit`.

## Examples

`examples/` has a square, a triangular and a honeycomb lattice with nearest-neighbour hopping, an
on-site Hubbard and a `1/r` Coulomb tail. Each comes with a notebook that plots its results.
```
julia --project=. -t 4 examples/Square/Square.jl
```
`examples/MeanField/` runs MeanFieldToolkit Hartree-Fock and then the real-frequency RPA on top of it: the magnons of
the Hubbard model's Néel state and the plasmon of a dilute metal with `1/r` repulsion.
```
julia --project=examples/MeanField -t 8 examples/MeanField/Magnons.jl
```
See [examples/README.md](examples/README.md).

# Documentation

For further details, please refer to the [Documentation](https://Toronto-Condensed-Matter-Theory.github.io/RPAToolkit.jl/dev/).
