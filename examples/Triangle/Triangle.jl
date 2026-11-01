# Triangular lattice: nearest-neighbour hopping, on-site Hubbard and a 1/r Coulomb tail.
#
#     julia --project=. -t 4 examples/Triangle/Triangle.jl
#
# Builds the model and the interactions, runs the bare bubble in TRIQS (see examples/README.md for
# the TRIQS environment) and then the RPA. Everything is written to examples/Triangle/data/.

using RPAToolkit, TightBindingToolkit, JLD2, LinearAlgebra

function triangle_uc()
    a1 = [ 1/2, sqrt(3)/2]
    a2 = [-1/2, sqrt(3)/2]
    uc = UnitCell([a1, a2], 2)
    AddBasisSite!(uc, [0.0, 0.0])
    return uc
end

config = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "Triangle.yml")
input = load_rpa_input(config)
mkpath(input["data_dir"])

#####* the free model: nearest-neighbour hopping t = -1, the same for both spins
SpinVec = SpinMats(1 // 2)
uc = triangle_uc()
t1Param = Param(-1.0, 2)
AddIsotropicBonds!(t1Param, uc, 1.0, SpinVec[4], "t1")
CreateUnitCell!(uc, [t1Param])
save(input["unitcell"]["julia"], Dict("unit cell" => uc, "parameters" => [t1Param]))

#####* the interactions: cases "Hubbard" (on-site) and "Coulomb" (on-site + 1/r tail)
coulomb = input["coulomb"]
params, cases = hubbard_coulomb_interactions(triangle_uc(); onsite = coulomb["onsite"], range = coulomb["range"])
save(input["interactions"], Dict("parameters" => params, "values" => cases, "unit cell" => triangle_uc()))

BLAS.set_num_threads(1)
run_rpa(config; run_bare = true)
