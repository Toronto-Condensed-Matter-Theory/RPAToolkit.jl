##### Small lattices shared by the tests (the geometries of the examples in examples/)

const SpinVec   =   SpinMats(1 // 2)

function square_uc()
    uc  =   UnitCell([[1.0, 0.0], [0.0, 1.0]], 2)
    AddBasisSite!(uc, [0.0, 0.0])
    return uc
end

function honeycomb_uc()
    uc  =   UnitCell([[1/2, sqrt(3)/2], [-1/2, sqrt(3)/2]], 2)
    AddBasisSite!(uc, [0.0, 0.0])
    AddBasisSite!(uc, [0.0, 1/sqrt(3)])
    return uc
end

##### square lattice with nearest-neighbour hopping t = -1
function square_model()
    uc  =   square_uc()
    t1  =   Param(-1.0, 2)
    AddIsotropicBonds!(t1, uc, 1.0, SpinVec[4], "t1")
    CreateUnitCell!(uc, [t1])
    return uc, [t1]
end

##### the bond Lookup of one interaction case, leaving the Params as they were
function case_lookup(params, weights)
    push!.(getproperty.(params, :value), weights)
    lookup  =   Lookup(params)
    pop!.(getproperty.(params, :value))
    return lookup
end
