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
const Δ = 1.5
const J1  =   +1.0
J1Param   =   Param(1.0, 2)
J2Param   =   Param(1.0, 2)

AddIsotropicBonds!(J1Param, UC , NNdistance, [1.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 Δ] , "J1 XXZ")
AddIsotropicBonds!(J2Param, UC , secondNNDistance, [1.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 Δ] , "J2 XXZ")


params = [J1Param, J2Param]

#####* multiple different interactions to run RPA on
const thetas = collect(LinRange(0.0, 0.25, 101))

values = Dict()

for theta in thetas
    values["theta = $(round(theta, digits=3))*2*pi"] = [cos(2*theta*pi), sin(2*theta*pi)]
end


#####* Saving the unit cell in a JLD2 file
file_name = "../../saves/interactions/triangular_J1J2.jld2"
save(file_name, Dict("parameters" => params, "values" => values))
