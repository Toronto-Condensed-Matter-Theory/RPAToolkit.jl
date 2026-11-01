# Examples

The repository's `examples/` folder has three lattices with nearest-neighbour hopping `t = -1`, an
on-site Hubbard ``U`` and a bare ``1/r`` Coulomb tail:

| example | lattice | sites per cell | path |
|:--------|:--------|:---------------|:-----|
| `Square` | square | 1 | Γ → X → M → Γ |
| `Triangle` | triangular | 1 | Γ → M → K → Γ |
| `Honeycomb` | honeycomb (graphene) | 2 | Γ → M → K → Γ |

Each has a config, a script that builds the model and the interactions and runs both stages, and a
Julia notebook that plots the results. Run one from the repository root, with TRIQS available (see
[Installation](@ref)):
```
julia --project=. -t 4 examples/Square/Square.jl
```

## Writing your own

The square example, in full:
```julia
using RPAToolkit, TightBindingToolkit, JLD2, LinearAlgebra

function square_uc()
    uc = UnitCell([[1.0, 0.0], [0.0, 1.0]], 2)
    AddBasisSite!(uc, [0.0, 0.0])
    return uc
end

config = "examples/Square/Square.yml"
input = load_rpa_input(config)
mkpath(input["data_dir"])

# the free model: nearest-neighbour hopping, the same for both spins
uc = square_uc()
t1Param = Param(-1.0, 2)
AddIsotropicBonds!(t1Param, uc, 1.0, SpinMats(1 // 2)[4], "t1")
CreateUnitCell!(uc, [t1Param])
save(input["unitcell"]["julia"], Dict("unit cell" => uc, "parameters" => [t1Param]))

# the interactions: cases "Hubbard" and "Coulomb"
params, cases = hubbard_coulomb_interactions(square_uc(); onsite = 4.0, range = 3.0)
save(input["interactions"], Dict("parameters" => params, "values" => cases, "unit cell" => square_uc()))

BLAS.set_num_threads(1)
output = run_rpa(config; run_bare = true)
```
For another interaction, register your own `Param`s with 4x4 channel matrices (see
[Conventions](@ref)) and give each case a weight per `Param` in `"values"`.

## Reading the output

```julia
output = load(rpa_output_path(load_rpa_input("examples/Square/Square.yml")))
for (key, entry) in output
    println(entry["filling"], "  ", entry["Hubbard"]["critical strength"],
            "  ", entry["Hubbard"]["maximum reciprocal momentum"])
end
```
See [The combined output](@ref) for every key.
