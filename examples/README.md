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

The bare stage needs TRIQS and triqs_tprf, plus `numpy` and `pyyaml`, in one Python environment. Make that
environment's `python3` the one on your `PATH`, or say how to launch it:
```
export TRIQS_ENVIRONMENT="python3"              # or "mpirun -n 8 python3"
```
Then, from the repository root:
```
julia --project=. -t 4 examples/Square/Square.jl
```
Each example takes under a minute on a laptop at `k_size: 48` (run serially; `mpirun` speeds up larger grids). Then open `examples/Square/Square.ipynb`
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
