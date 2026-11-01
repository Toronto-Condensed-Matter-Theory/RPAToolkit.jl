# RPAToolkit.jl

RPAToolkit.jl computes the static bare susceptibility ``\chi_0(q, \Omega = 0)`` of a tight-binding
model with [TRIQS](https://triqs.github.io/triqs/latest/) /
[triqs_tprf](https://triqs.github.io/tprf/latest/). It then solves the random phase
approximation (RPA) for any set of two-body interactions, to find which interaction strength first
drives a charge, spin or orbital instability, and at which ordering wavevector.

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

## Installation

RPAToolkit has two halves: a Julia package, and the Python scripts that compute the bubble with
TRIQS.

1. Install [TRIQS](https://triqs.github.io/triqs/latest/install.html) and
   [triqs_tprf](https://triqs.github.io/tprf/latest/install.html), plus `numpy` and `pyyaml`, in
   one Python environment.
2. Add the Julia package:
   ```julia
   using Pkg
   Pkg.add(url = "https://github.com/Toronto-Condensed-Matter-Theory/RPAToolkit.jl")
   ```
3. Tell RPAToolkit how to launch that Python. Either set the `triqs_environment` key of your
   config, or export `TRIQS_ENVIRONMENT` (e.g. `python3`, or `mpirun -n 8 python3` to use MPI).
   With neither set, `python3` from the `PATH` is used.

The Julia half (RPA, interactions, plotting, tests) does not need TRIQS. Only the bare stage does.

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
from a checkout of this repository, with TRIQS available. See [Examples](@ref) and
[Running a calculation](@ref).
