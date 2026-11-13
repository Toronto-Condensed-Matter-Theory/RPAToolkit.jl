# RPAToolkit.jl

RPAToolkit.jl computes the bare susceptibility of a tight-binding model, statically,
``\chi_0(q, \Omega = 0)``, or at real frequency, ``\chi_0(q, \omega + i\eta)``. It then solves the
random phase approximation (RPA) for any set of two-body interactions: statically, to find which
interaction strength first drives a charge, spin or orbital instability and at which ordering
wavevector; dynamically, to follow collective modes such as magnons and plasmons
([Collective modes and mean-field theory](@ref)).

Features:
* Lattices and hoppings are defined with
  [TightBindingToolkit.jl](https://github.com/Toronto-Condensed-Matter-Theory/TightBindingToolkit.jl).
  Any 2D unit cell with any number of sites and orbitals is supported.
* Interactions are TightBindingToolkit `Param`s whose bonds carry matrices in the (N, X, Y, Z)
  channels, so on-site, inter-site and long-range terms all go in the same way. A Hubbard + ``1/r``
  Coulomb builder is included ([`hubbard_coulomb_interactions`](@ref)).
* Several interaction cases (e.g. "Hubbard" and "Coulomb") run against the same bubbles, over a scan
  of fillings or chemical potentials.
* For multi-orbital models whose orbitals rotate into each other under the point group (an E
  doublet), the susceptibility and the Kanamori vertex can be put on a symmetry-closed bilinear
  basis, so ``\chi`` stays covariant.
* The critical strength, the leading eigenvector and its momentum, the RPA susceptibility, and the
  screened interaction ``V + V \chi_\mathrm{RPA} V`` ([`effective_interaction`](@ref)).
* The bubble is computed in Julia, threaded: the exact Lindhard sum ([`static_bubble`](@ref)), or an
  imaginary-time route with FFTs and a discrete Lehmann representation ([`static_bubble_dlr`](@ref),
  [`dlr_grid`](@ref)) for large grids.
* The real-frequency bubble ([`dynamic_bubble`](@ref)) and RPA ([`dynamic_rpa`](@ref)), spectral
  functions unfolded from supercells, and an Ewald-summed ``1/r`` interaction ([`coulomb_2d`](@ref)).
* Hartree-Fock bands from MeanFieldToolkit.jl, with the TDHF vertex of the same interaction
  ([`rpa_model`](@ref), [`rpa_vertex`](@ref)).

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/Toronto-Condensed-Matter-Theory/RPAToolkit.jl")
```

## Quick start

```julia
using RPAToolkit
run_rpa("examples/Square/Square.yml"; run_bare = true)
```

This needs the model and interactions files the config points to. The scripts in `examples/`
write them and then call `run_rpa`, so the shortest path is
```
julia --project=. -t 4 examples/Square/Square.jl
```
from a checkout of this repository. See [Examples](@ref) and [Running a calculation](@ref).
