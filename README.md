# RPA
Codebase to calculate bare susceptiblity of a lattice model (using TRIQS), and then perform general random phase approximation (RPA) on it to get instabilities of the model.

## Installation
- Install the TRIQS and [triqs_tprf](https://triqs.github.io/tprf/latest/install.html) package in a Python venv (with default name "triqs").
On Ubuntu there are precompiled packages. 
Otherwise this requires compilation described [here](https://triqs.github.io/triqs/3.3.x/install.html). 
- Add the julia repo locally in Package mode.

## Usage
- Create the free Hamiltonian first. This is done using the [TightBindingToolkit.jl](https://github.com/Toronto-Condensed-Matter-Theory/TightBindingToolkit.jl) interface. 
Some examples are given in [./examples/models](https://github.com/Toronto-Condensed-Matter-Theory/RPAToolkit.jl//main/examples/models).
- Define the interactions which are to be used for the RPA. This can also done in the same interface, with examples given in [./examples/interactions](https://github.com/Toronto-Condensed-Matter-Theory/RPAToolkit.jl/tree/main/examples/interactions).
- Write a parent input file with details of parameters required for the calculation such as the inverse temperature, number of k-points, and number of matsubara frequencies.
 Refer to [./Inputs](https://github.com/Toronto-Condensed-Matter-Theory/RPAToolkit.jl/tree/main/Inputs) for formatting.
- Run the following command
  ```julia
  julia --project=../Project.toml --heap-size-hint=4G run_RPA.jl --input="../Inputs/name_of_input.yml" --run_bare=true
  ```

## Input Notes
- Preferred scan input is `fillings`, with either explicit values:
  ```yaml
  fillings:
    values: [0.2, 0.4, 0.6]
  ```
  or a compact range:
  ```yaml
  fillings:
    min: 0.0
    max: 1.0
    n: 51
  ```
- Legacy `mus` inputs are still supported, and are converted to `fillings` during the bare stage.
- `k_points` can be either explicit coordinates or HighSymPoints labels understood by the model Brillouin zone.
  Example mixed format:
  ```yaml
  k_points:
    - G
    - [0.666666666667, 0.333333333333, 0.0]
    - M2
  k_points_labels: ["\\Gamma", "K_1", "M_2"]
  ```
- If `k_points_labels` is omitted, labels are inferred from `k_points`.
