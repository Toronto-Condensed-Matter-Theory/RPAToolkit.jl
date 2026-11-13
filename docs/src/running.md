# Running a calculation

A calculation is driven by one YAML config and has two stages:

```
model .jld2 ─┐
             ├─ run_bare ─────────> one bubble .npz per mu ─┐
config .yml ─┘                                              ├─ run_rpa ─> combined .jld2
interactions .jld2 ─────────────────────────────────────────┘
```

1. **You** write the model and the interactions, as JLD2 files (below).
2. [`run_bare`](@ref) computes the bare susceptibility for every chemical potential.
3. [`run_rpa`](@ref) finds the critical strength of every interaction case at every chemical
   potential and saves everything in one file.

`run_rpa(config; run_bare = true)` does stages 2 and 3 in one call.

## The input files

**Model** (`unitcell.julia`): a JLD2 file with key `"unit cell"`, a TightBindingToolkit `UnitCell`
whose bonds are the hoppings, with `localDim = 2` (spin). The hoppings may be spin-dependent; if
they are spin-diagonal and spin-independent, the bubble is computed spinless, which is 16x cheaper.

**Interactions** (`interactions`): a JLD2 file with keys
- `"parameters"`: a vector of TightBindingToolkit `Param`s. Each bond matrix is
  `length(directions) x length(directions)`, in the channel layout of [Conventions](@ref).
- `"values"`: a `Dict` from case name to a vector of weights, one per `Param`. Each case is one RPA
  calculation, with interaction `sum_p weight_p * Param_p`.
- `"unit cell"`: the unit cell the bonds were registered on. `run_rpa` refuses to run if its site
  count differs from the model's.

[`hubbard_coulomb_interactions`](@ref) returns `"parameters"` and `"values"` for an on-site Hubbard
plus a ``1/r`` tail. For a multi-orbital model on the closed bilinear basis, the file also carries
`"bilinear_basis"` and `"closed_matrices"` (see [Conventions](@ref)).

## Running

From Julia:
```julia
using RPAToolkit, LinearAlgebra
BLAS.set_num_threads(1)
output = run_rpa("config.yml"; run_bare = true)
```

From the shell, with the scripts in `src/`:
```
julia --project=<env> src/Bare/run_bare.jl --input config.yml
julia --project=<env> -t 16 src/run_RPA.jl --input config.yml [--run_bare true] [--plot_RPA true] [--save_individual true]
```

Both stages use Julia threads (`julia -t N`): the bare stage spreads each bubble's momenta over
them (and FFTW its transforms, with `bubble: dlr`); the RPA stage runs its (chemical potential,
case) pairs on them, each pair serial over momenta. Keep BLAS single-threaded
(`BLAS.set_num_threads(1)`), as the scripts do. For grids beyond `k_size` ≈ 100, `bubble: dlr`
is much faster than the exact Lindhard sum, at an accuracy set by `dlr_err`.

## Config reference

Paths are relative to the YAML file. Anything not listed here is ignored by RPAToolkit, so a config
can carry keys for other tools (the examples carry `coulomb:` for their own interaction builder).

| key | | meaning |
|:----|:-|:--------|
| `beta` | required | inverse temperature |
| `k_size` | required | the bubble is computed on a `k_size x k_size` momentum grid |
| `directions` | required | the channels: `0` = N, `1` = X, `2` = Y, `3` = Z. Usually `[0, 1, 2, 3]`. |
| `fillings` *or* `mus` | required | the scan. `values: [...]`, or `min`, `max`, `n`. A bare `mus: {n: ...}` spans the bandwidth. Fillings are the fraction of all spin-orbital states occupied (0.5 = half filling) and are inverted to chemical potentials once, at the start of the bare stage. |
| `k_points` | required | the k-path for the `_path` quantities: TightBindingToolkit high-symmetry labels (`G`, `M1`, `K2''`, ...) or reduced coordinates. The path closes back to the first point. |
| `k_labels` | optional | plot labels for `k_points` (`k_points_labels` also accepted) |
| `data_dir` | optional | where default file names go. Default: the YAML's directory. |
| `output` | optional | prefix of the bubble files and the combined output. Default `<data_dir>/<name>_`. |
| `plots` | optional | prefix of plots. Default `<data_dir>/<name>_`. |
| `interactions` | optional | the interactions file. Default `<data_dir>/<name>_interactions.jld2`. |
| `unitcell.julia` | optional | the model file. Default `<data_dir>/<name>_dos_scan_data.jld2`. |
| `instability.lower`, `.upper` | optional | the bisection bracket for the critical strength. Default `[0, 10]`. |
| `strengths.lower`, `.upper`, `.n` | optional | strengths at which `plot = true` draws ``\chi_\mathrm{RPA}``. Default `0, 10, 6`. |
| `bubble` | optional | `lindhard` (default; the exact Matsubara sum, ``O(N_k^2)``, [`static_bubble`](@ref)) or `dlr` (the imaginary-time FFT bubble on a discrete Lehmann representation, ``O(N_k \log N_k)``, [`static_bubble_dlr`](@ref); `w_max`, default 20, is its spectral cutoff and must exceed the band span; `dlr_err`, default 1e-12, its accuracy) |
| `bubble_spin` | optional | `auto` (default; the spinless bubble without spin-orbit coupling, and only the channel pairs the model's spin symmetry allows, see [Spin-orbit coupling and spin sectors](@ref)) or `full` (the spinful bubble and all sixteen channel pairs) |
| `bilinear_basis` | optional | `per_orbital` (default) or `closed`. Must match the interaction builder. |
| `bilinear_seed` | optional | for `closed`: `kanamori` (default; densities plus on-site off-diagonal bilinears) or `density` |
| `name` | optional | replaces the file name in the default interactions path |
| `gap_equation_rpa.rpa_data` | optional | if `gap_equation_rpa` is present, where `run_rpa` saves its output (for a downstream gap-equation code). Default `<data_dir>/<name>_combined_new.jld2`. |

## Files written

| file | written by | contents |
|:-----|:-----------|:---------|
| `<output>_runtime_input.yml` | `run_bare` | the resolved config: absolute paths, k-points as coordinates, and the scan as both `mus.values` and `fillings.values`. `run_rpa` reads the chemical potentials from here. |
| `<output>_beta=<beta>_mu=<mu>.npz` | `run_bare` | one bubble: `chi_NN`, `chi_XX`, `chi_YY`, `chi_ZZ` on the grid `ks` (shape `n_q x d x d`), and with spin-orbit coupling the off-diagonal pairs `chi_NZ`, `chi_XY`, ... ([`pair_label`](@ref)); the same with `_path` along the k-path (computed on the path itself); `spin_symmetry` (the index into [`SPIN_SYMMETRIES`](@ref)); plus `beta`, `mu`, `filling`, `primitives`, `reciprocal`, `path`, `path_plot`, `path_ticks`, `bands` (along the path) and `bandwidth`. On the closed basis also `bilinear_basis`. |
| [`rpa_output_path`](@ref) | `run_rpa` | the combined output, below |

## The combined output

A `Dict` with one entry per chemical potential, keyed `"beta=<beta>_mu=<mu>"`, each holding:

| key | contents |
|:----|:---------|
| `"mu"`, `"beta"`, `"filling"` | the point of the scan |
| `"bare_data"` | the bubble `.npz`, as read (`"triqs_data"` in files written before version 1.1) |
| `"combined_chis"`, `"combined_chis_path"` | ``\chi(q)`` on the grid and along the path, one `4d x 4d` matrix per momentum ([`combine_chis`](@ref)) |
| `"chi_orientation"` | the convention of those matrices ([`CHI_ORIENTATION`](@ref)) |
| `"spin_symmetry"` | `"SU2"`, `"U1z"` or `"none"`: which channel pairs ``\chi`` holds, and the sectors the RPA was solved in ([`spin_sectors`](@ref)) |
| `"vertex_slots"` | the layout of the interaction matrices ([`VERTEX_SLOTS`](@ref)) |
| `"<case>_interaction"`, `"<case>_interaction_path"` | ``V(q)`` of each case at unit strength, on the grid and along the path |
| `"<case>"` | the [`find_instability`](@ref) result: `"critical strength"`, the `"unstable sector"`, and at that strength the minimum/maximum eigenvalues of ``\chi_\mathrm{RPA}``, their eigenvectors, and the momenta they sit at |
| `"bilinear_basis"`, `"basis_dim"` | on the closed basis only |

The grid momenta are `output[key]["bare_data"]["contracted"]` (Cartesian, one row per momentum).
With these, anything else is a few lines. For example, the RPA susceptibility and the screened
interaction along the path at strength `s`:
```julia
entry = output["beta=20.0_mu=0.0"]
V = [s .* M for M in entry["Hubbard_interaction_path"]]
chi_rpa = perform_RPA(entry["combined_chis_path"], V; return_matrix = true)
V_eff = effective_interaction(entry["combined_chis_path"], V)
```
