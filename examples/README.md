# Examples

Three lattices with nearest-neighbour hopping `t = -1`, one orbital per site, an on-site Hubbard `U`
and a bare `1/r` Coulomb tail:

| folder | lattice | sites per cell | path |
|:-------|:--------|:---------------|:-----|
| `Square/` | square | 1 | Γ → X → M → Γ |
| `Triangle/` | triangular | 1 | Γ → M → K → Γ |
| `Honeycomb/` | honeycomb (graphene) | 2 | Γ → M → K → Γ |

Each folder has
- `<Lattice>.yml`: the config (β, k-grid, fillings, k-path, and the `coulomb:` block the script reads);
- `<Lattice>.jl`: builds the model and the interactions, then runs both stages with `run_rpa(config; run_bare = true)`;
- `<Lattice>.ipynb`: a Julia notebook that plots the results. It covers the bands, the Fermi surfaces and the
  nesting by the ordering wavevector `Q` (drawn with TightBindingToolkit's `Plot_FS!`), the bare bubble, the RPA spin
  susceptibility approaching the Stoner point, the charge channel with and without the `1/r` tail, the screened
  interaction `V + V χ_RPA V`, a map over the Brillouin zone, and the critical `U` against filling.

Everything a run writes goes to `<Lattice>/data/` (git-ignored).

## Running

From the repository root:
```
julia --project=. -t 4 examples/Square/Square.jl
```
Each example takes seconds on a laptop at `k_size: 48`; set `bubble: dlr` in the config for much larger grids. Then open `examples/Square/Square.ipynb`
with the Julia kernel (IJulia), or make the two quick plots from the shell:
```
julia --project=. examples/plot_bands.jl examples/Square/Square.yml
julia --project=. examples/plot_susceptibilities.jl examples/Square/Square.yml
```

## The interactions

`hubbard_coulomb_interactions(uc; onsite, range)` registers two cases:
- `Hubbard`: the on-site `U = onsite`;
- `Coulomb`: the same `U` plus `1/r` to every site within `range`, in units of `onsite`.

The tail is pure charge. It screens the charge channel and does nothing to the spin channel, so both cases share
the spin-driven critical `U`; only their charge responses differ. Make `range` larger or `onsite` smaller and the
charge channel can go attractive at large `q`, which shows up as a charge-ordering instability with a lower critical
strength than the Hubbard case.

## Hartree-Fock + real-frequency RPA (`MeanField/`)

[MeanFieldToolkit.jl](https://github.com/Toronto-Condensed-Matter-Theory/MeanFieldToolkit.jl) solves the
Hartree-Fock problem (Anderson mixing); `rpa_model(mft)` and `rpa_vertex(mft)` (a package extension, loaded by
`using MeanFieldToolkit`) hand its bands and the TDHF vertex of the same interaction to the RPA. The folder has its
own environment, which takes MeanFieldToolkit, FixedPointToolkit and TightBindingToolkit from sibling checkouts
(`../MeanFieldToolkit.jl` and so on, see `MeanField/Project.toml`):

- `Magnons.jl`: the Néel state of the square-lattice Hubbard model at `U = 4, 8, 12`. Checks the Goldstone mode
  (the largest transverse eigenvalue of `χ₀V` at the ordering vector is exactly 1) and plots the transverse spin
  spectral function along Γ → X → M → Γ against linear spin-wave theory, `ω = 2J√(1 − γ_q²)`, `J = 4t²/U`.
- `Plasmons.jl`: a dilute metal (`n = 0.1`) with on-site `U` and `e²/εr`. Hartree-Fock of `U` and the
  nearest-neighbour part (the Fock term widens the band), then the RPA with the full Ewald-summed `1/r` tail:
  the loss function, the plasmon against `√(2π n e² q / m*)`, and the particle-hole continuum.

```
julia --project=examples/MeanField -t 8 examples/MeanField/Magnons.jl
julia --project=examples/MeanField -t 8 examples/MeanField/Plasmons.jl
```
Output goes to `MeanField/data/`. The extension's tests run in the same environment:
`julia --project=examples/MeanField test/test_meanfield_ext.jl`.
