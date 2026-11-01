# RPAToolkit.jl

[![Build Status](https://github.com/Toronto-Condensed-Matter-Theory/RPAToolkit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Toronto-Condensed-Matter-Theory/RPAToolkit.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://Toronto-Condensed-Matter-Theory.github.io/RPAToolkit.jl/dev/)

RPAToolkit.jl is a Julia package for the random phase approximation (RPA) on lattice models. It
computes the static bare susceptibility of a tight-binding model with
[TRIQS](https://triqs.github.io/triqs/latest/)/[triqs_tprf](https://triqs.github.io/tprf/latest/).
Then, for any set of two-body interactions, it finds the strength at which the RPA first goes
unstable, and at which ordering wavevector.

Currently supported:
* Lattices and hoppings are defined with [TightBindingToolkit.jl](https://github.com/Toronto-Condensed-Matter-Theory/TightBindingToolkit.jl). Any 2D unit cell with any number of sites and orbitals is supported.
* Interactions are TightBindingToolkit `Param`s in the (N, X, Y, Z) channels: on-site, inter-site and long-range terms alike. An on-site Hubbard plus `1/r` Coulomb builder is included.
* Several interaction cases run against the same bubbles, over a scan of fillings or chemical potentials, in parallel.
* For multi-orbital models whose orbitals rotate into each other under the point group, the susceptibility and the Kanamori vertex can be put on a symmetry-closed bilinear basis.
* Spin-orbit coupling: the bubble keeps every channel pair the model's spin symmetry allows (all sixteen in general, the {N, Z} and {X, Y} sectors with Ising coupling), and the RPA is solved one spin sector at a time. Without spin-orbit coupling the spinless bubble is used.
* Outputs include the critical strength, the leading instability and its momentum, the RPA susceptibility, and the screened interaction.

## Installation

1. Install TRIQS and [triqs_tprf](https://triqs.github.io/tprf/latest/install.html), plus `numpy` and `pyyaml`, in one Python environment.
2. `] add https://github.com/Toronto-Condensed-Matter-Theory/RPAToolkit.jl`
3. Point RPAToolkit at that Python with `export TRIQS_ENVIRONMENT=python3` (or `"mpirun -n 8 python3"`), or with the `triqs_environment` config key.

## Examples

`examples/` has a square, a triangular and a honeycomb lattice with nearest-neighbour hopping, an
on-site Hubbard and a `1/r` Coulomb tail. Each comes with a notebook that plots its results.
```
julia --project=. -t 4 examples/Square/Square.jl
```
See [examples/README.md](examples/README.md).

# Documentation

For further details, please refer to the [Documentation](https://Toronto-Condensed-Matter-Theory.github.io/RPAToolkit.jl/dev/).
