using Plots, LaTeXStrings, TightBindingToolkit, JLD2

"""
This script sets up a simple triangular lattice with the pi/0 flux state and a 3-sublattice spin order.
"""

"""
The lattice primitive vector for a triangular lattice : a1 and a2.
"""
const a1 = [ 1.0, 0.0 ]
const a2 = [ 1/2 , sqrt(3)/2 ]

const a1New = 2*a2
const a2New = a1
UC = UnitCell( [a1New , a2New] , 2) ##### localDim=2 since we are working with spin-1/2 particles now

"""
Unit cell has 6 sub-lattices.
"""
const b1 = [ 0.0 , 0.0 ]
const b2 = b1+a2
AddBasisSite!.( Ref(UC) , [b1, b2])

"""
Adding structure to the lattice now, through the bond objects.
"""
SpinVec     =   SpinMats(1//2) ##### Working with spin-1/2
################ Flux pi-0 hoppings #################
const NNdistance  =   1.0
const secondNNDistance = sqrt(3)
const Δ = 1.5   ##### Physical anisotropy. Assumed to be same for J1 and J2
const δ = 0.7   ##### Anisotropy in the αs such that δ = α_{||}/α_z for both NN and 2NN.
const λ = 1.0   ##### α_2/α_1 for 2NN.
const α1s = collect(LinRange(0.0, 1.0, 101))
const J2byJ1s = collect(LinRange(0.0, 1.0, 101))

const J1  =   +1.0
J1xxParam   =   Param(1.0, 2)
J2xxParam   =   Param(1.0, 2)

J1zParam   =   Param(1.0, 2)
J2zParam   =   Param(1.0, 2)

AddIsotropicBonds!(J1xxParam, UC , NNdistance, [1.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 0.0] , "J1 XX")
AddIsotropicBonds!(J1zParam, UC , NNdistance, [0.0 0.0 0.0 ; 0.0 0.0 0.0 ; 0.0 0.0 1.0] , "J1 Ising")
AddIsotropicBonds!(J2xxParam, UC , secondNNDistance, [1.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 0.0] , "J2 XX")
AddIsotropicBonds!(J2zParam, UC , secondNNDistance, [0.0 0.0 0.0 ; 0.0 0.0 0.0 ; 0.0 0.0 1.0] , "J2 Ising")


params = [J1xxParam, J1zParam, J2xxParam, J2zParam]

#####* multiple different interactions to run RPA on
values = Dict()

for a in α1s
    values["a = $(round(a, digits=3))"] = []
end


#####* Saving the unit cell in a JLD2 file
file_name = "../../saves/interactions/triangular_J1J2.jld2"
save(file_name, Dict("parameters" => params, "values" => values))
