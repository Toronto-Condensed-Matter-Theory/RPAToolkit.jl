module RPAToolkitMeanFieldToolkitExt

using RPAToolkit, MeanFieldToolkit, TightBindingToolkit, LinearAlgebra
import RPAToolkit: rpa_model, rpa_vertex

function rpa_model(mft::TBMFTModel; residual::Union{Nothing, Real} = nothing, tol::Real = 1e-6)
    model = mft.model
    model.uc.localDim == 2 ||
        error("rpa_model: the unit cell has localDim = $(model.uc.localDim); the RPA channels need one spin-1/2 orbital per basis site (localDim = 2).")
    model.T > 0 || error("rpa_model: the model is at T = $(model.T); the bubble needs a finite temperature.")
    residual === nothing || residual <= tol ||
        @warn "rpa_model: the Hartree-Fock run is converged only to $(residual) (> $(tol)). Its bands, and any Goldstone mode, are off by about that much."
    return (bands = band_model(model.uc), mu = model.mu, beta = 1 / model.T, filling = model.filling, unitcell = model.uc)
end

function rpa_vertex(mft::TBMFTModel; interactions = nothing)
    uc = mft.model.uc
    uc.localDim == 2 || error("rpa_vertex: localDim = $(uc.localDim); one spin-1/2 orbital per basis site is needed.")
    all(isone, values(mft.MFTScaling)) ||
        @warn "rpa_vertex: the mean-field channels are rescaled ($(mft.MFTScaling)); the vertex is built from the bare interaction, so the RPA is no longer the linear response of this mean-field state."

    terms = Tuple{Int, Int, Vector{Float64}, Matrix{ComplexF64}}[]
    exchange_dropped = false
    chosen = interactions === nothing ? eachindex(mft.Interactions) : interactions
    for (index, (param, decomposition)) in enumerate(zip(mft.Interactions, mft.MFTDecomposition))
        index in chosen || continue
        strength = param.value[end]
        for bond in param.unitBonds
            U = strength .* bond.mat
            onsite = bond.base == bond.target && iszero(bond.offset)
            if onsite
                decomposition === IntraQuarticToHopping ||
                    error("rpa_vertex: the on-site interaction \"$(param.label)\" is decoupled with $(decomposition); only IntraQuarticToHopping has a TDHF vertex here.")
                push!(terms, (bond.base, bond.base, zeros(2), quartic_onsite_vertex(U)))
            else
                decomposition === InterQuarticToHopping ||
                    error("rpa_vertex: the inter-site interaction \"$(param.label)\" is decoupled with $(decomposition); only InterQuarticToHopping has a vertex here.")
                R = Float64.(sum(bond.offset .* uc.primitives))[1:2]
                Vij, Vji = quartic_hartree_vertex(U)
                push!(terms, (bond.base, bond.target, R, Vij))
                push!(terms, (bond.target, bond.base, -R, Vji))
                exchange_dropped = true
            end
        end
    end
    exchange_dropped &&
        @warn "rpa_vertex: inter-site interactions enter the vertex with their Hartree part only. Their exchange part renormalizes the Hartree-Fock bands but has no representation on per-site channels, so this part of the RPA is not the linear response of the mean-field state."

    ns = length(uc.basis)
    return function (q)
        V = zeros(ComplexF64, 4ns, 4ns)
        for (i, j, R, M) in terms
            V[4(i - 1) .+ (1:4), 4(j - 1) .+ (1:4)] .+= M .* cis(q[1] * R[1] + q[2] * R[2])      # INTERACTION_CONVENTION
        end
        return V
    end
end

end # module
