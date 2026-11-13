#=
Shared pieces of the MeanFieldToolkit -> RPAToolkit examples (Magnons.jl, Plasmons.jl) and of
test/test_meanfield_ext.jl: the Hartree-Fock models, the self-consistency, and small helpers.

The RPA on top of a Hartree-Fock state, with the vertex built from the same interaction tensor the
mean-field theory decoupled, is the linear response of that state: time-dependent Hartree-Fock.
=#

using RPAToolkit, MeanFieldToolkit, TightBindingToolkit, FixedPointToolkit
using LinearAlgebra, Logging

const SPIN = SpinMats(1 // 2)                                            # [Sx, Sy, Sz, Id]
const HUBBARD_U = DensityToPartonCoupling([1.0 0.0; 0.0 0.0], [0.0 0.0; 0.0 1.0])   # n_up n_down
const DENSITY_DENSITY = DensityToPartonCoupling(Matrix(1.0I, 2, 2), Matrix(1.0I, 2, 2))   # n_i n_j
const SITE_CHANNELS = ["n" => SPIN[4], "Sx" => SPIN[1], "Sy" => SPIN[2], "Sz" => SPIN[3]]

"""Every spin channel on every site: unrestricted on-site Hartree-Fock."""
function onsite_orders(uc::UnitCell)
    orders = Param{2, ComplexF64}[]
    for site in eachindex(uc.basis), (name, mat) in SITE_CHANNELS
        p = Param(0.0 + 0.0im, 2)
        AddAnisotropicBond!(p, uc, site, site, [0, 0], mat, 0.0, "site $site : $name")
        push!(orders, p)
    end
    return orders
end

"""The charge channel on every nearest-neighbour bond (the Fock amplitude of a NN interaction)."""
function nn_bond_orders(uc::UnitCell)
    mother = Param(1.0, 2)
    AddIsotropicBonds!(mother, uc, 1.0, SPIN[4], "nearest neighbours")
    orders = Param{2, ComplexF64}[]
    for bond in mother.unitBonds
        p = Param(0.0 + 0.0im, 2)
        AddAnisotropicBond!(p, uc, bond.base, bond.target, bond.offset, SPIN[4], 1.0, "bond $(bond.base) → $(bond.target) $(bond.offset)")
        push!(orders, p)
    end
    return orders
end

"""
The square-lattice Hubbard model H = -t Σ c†c + U Σ n↑n↓ on the √2×√2 cell that holds Néel order,
unrestricted on-site Hartree-Fock.
"""
function neel_hubbard(; U::Float64, t::Float64 = 1.0, filling::Float64 = 0.5, T::Float64 = 0.01, kSize::Int = 40)
    uc = UnitCell([[1.0, 1.0], [1.0, -1.0]], 2, 2)
    AddBasisSite!(uc, [0.0, 0.0])
    AddBasisSite!(uc, [1.0, 0.0])
    hopping = Param(-t, 2)
    AddIsotropicBonds!(hopping, uc, 1.0, SPIN[4], "t")
    CreateUnitCell!(uc, [hopping])
    UParam = Param(U, 4)
    AddIsotropicBonds!(UParam, uc, 0.0, HUBBARD_U, "Hubbard U")
    bz = BZ([kSize, kSize]); FillBZ!(bz, uc)
    H = Hamiltonian(uc, bz); DiagonalizeHamiltonian!(H)
    model = Model(uc, bz, H; T = T, filling = filling, stat = -1)
    return TBMFTModel(model, onsite_orders(uc), [UParam], IntraQuarticToHopping)
end

"""
The square-lattice extended Hubbard model, one site per cell: on-site U and nearest-neighbour V
n_i n_j, decoupled in the on-site (Hartree + Fock) and nearest-neighbour (Hartree + Fock) channels.
"""
function extended_hubbard(; U::Float64, V::Float64, t::Float64 = 1.0, filling::Float64, T::Float64 = 0.01, kSize::Int = 64)
    uc = UnitCell([[1.0, 0.0], [0.0, 1.0]], 2, 2)
    AddBasisSite!(uc, [0.0, 0.0])
    hopping = Param(-t, 2)
    AddIsotropicBonds!(hopping, uc, 1.0, SPIN[4], "t")
    CreateUnitCell!(uc, [hopping])
    UParam = Param(U, 4)
    AddIsotropicBonds!(UParam, uc, 0.0, HUBBARD_U, "Hubbard U")
    VParam = Param(V, 4)
    AddIsotropicBonds!(VParam, uc, 1.0, DENSITY_DENSITY, "nearest-neighbour V")
    bz = BZ([kSize, kSize]); FillBZ!(bz, uc)
    H = Hamiltonian(uc, bz); DiagonalizeHamiltonian!(H)
    model = Model(uc, bz, H; T = T, filling = filling, stat = -1)
    return TBMFTModel(model, vcat(onsite_orders(uc), nn_bond_orders(uc)), [UParam, VParam],
                      Function[IntraQuarticToHopping, InterQuarticToHopping])
end

"""
Solve the mean-field theory from `initial` with Anderson mixing; returns the residual
max|V_out - V_in| of the last step, which `rpa_model` checks.
"""
function solve!(mft::TBMFTModel, initial; tol::Float64 = 1e-10, max_iter::Int = 500, alpha::Float64 = 0.5)
    sc = with_logger(NullLogger()) do
        SolveMFT!(mft, ComplexF64.(initial); Update = AndersonMixing,
                  Update_kwargs = Dict{Symbol, Any}(:alpha => alpha, :m => 5), max_iter = max_iter, tol = tol)
    end
    return maximum(abs, sc.VOuts[end] - sc.VIns[end]), length(sc.VOuts) - 1
end

"""The converged order parameters, by label."""
order_parameters(mft::TBMFTModel) = Dict(p.label => p.value[end] for p in mft.HoppingOrders)

"""The static bubble in the RPA's convention (what run_rpa assembles), at the momenta `qs`."""
function static_chi(bands::BandModel, qs, k_size; beta, mu)
    n_sites = length(bands.positions) ÷ 2
    stored = stored_bubble(bands, qs, k_size; beta = beta, mu = mu, ops = channel_operators(n_sites))
    return [2 .* transpose(S) for S in stored]
end

"""The original square lattice's path Γ → X → M → Γ, `num` points per segment, Cartesian."""
function square_path(num::Int)
    corners = [[0.0, 0.0], [pi, 0.0], [pi, pi]]
    ks, dist, ticks = Vector{Float64}[], Float64[], [0.0]
    for (a, b) in zip(corners, vcat(corners[2:end], [corners[1]]))
        for x in range(0.0, 1.0, length = num)[1:end-1]
            k = a .+ x .* (b .- a)
            push!(dist, isempty(ks) ? 0.0 : dist[end] + norm(k .- ks[end]))
            push!(ks, k)
        end
        push!(ticks, ticks[end] + norm(b .- a))
    end
    push!(dist, dist[end] + norm(corners[1] .- ks[end])); push!(ks, corners[1])
    return ks, dist, ticks
end
