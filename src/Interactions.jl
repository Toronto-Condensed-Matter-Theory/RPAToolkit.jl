module Interactions

using LinearAlgebra, TightBindingToolkit
using ..RPAToolkit.SpinSectors: VERTEX_SLOTS

export interaction, INTERACTION_CONVENTION, HUBBARD_CHANNELS, DENSITY_CHANNELS, hubbard_coulomb_interactions,
       tdhf_channel_vertex, quartic_onsite_vertex, quartic_hartree_vertex

"""
    INTERACTION_CONVENTION

The Fourier convention of every interaction matrix: a bond `(i, j, offset)` with matrix `M` enters
``V_{ij}(q)`` as ``M e^{+iq\\cdot\\delta}``, ``\\delta`` the lattice vector of `offset`, which is the Bloch
convention of TightBindingToolkit's ``H(k)`` for the same bond. It is the one for which
``\\chi_\\mathrm{RPA} = (1 - \\chi V)^{-1}\\chi`` is the RPA: [`combine_chis`](@ref RPAToolkit.Preprocess.combine_chis)'s ``\\chi_{ij}(q)`` is the
response of the ``e^{iq\\cdot R}`` component of ``O_i`` to a field ``e^{iq\\cdot R}`` on ``O_j``, and the
mean field of a pair interaction acting on that component is ``\\sum_R e^{+iq\\cdot R} v_{ij}(R)``.
Checked against an exact real-space Hartree RPA on a periodic cluster (test/test_rpa_realspace.jl).

Until 2026-11-12 `interaction` used ``e^{-iq\\cdot\\delta}``, i.e. ``V(-q) = V(q)^T``. Identical for
on-site interactions and for any lattice where every bond comes with its reverse (one site per cell),
but on a multi-site cell a non-local interaction entered transposed: the honeycomb example's
``1/r`` charge response was off by 30-40%. `run_rpa` records this string as `"interaction_convention"`.
"""
const INTERACTION_CONVENTION = "V_ij(q) = sum_bonds M e^{+iq.δ} (H(k)'s Bloch convention)"

"""
    HUBBARD_CHANNELS

The on-site Hubbard term `U n_up n_dn` in the (N, X, Y, Z) channel slots: `diag(-1, +1, +1, +1)`.

These are the physical channels ([`VERTEX_SLOTS`](@ref)). With `chi_RPA = (1 - chi V)^-1 chi` and a
positive bare `chi`, a positive entry is enhanced: slots X, Y and Z carry the Stoner (spin) vertex
`U^(s)` and slot N carries the charge vertex negated, `-U^(c)` -- the layout `channel_blocks`
uses. Written in the Pauli channels, the antisymmetrized Hubbard vertex is exactly this matrix,
so the layout stays right when spin-orbit coupling makes `chi` mix the channels.

`interaction` adds every registered bond AND its Hermitian conjugate, on-site bonds included, so
an on-site bond lands in its own block twice: register `U/2 .* HUBBARD_CHANNELS` to get `U`.
"""
const HUBBARD_CHANNELS = ComplexF64[-1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]

"""
    DENSITY_CHANNELS

A density-density term `V n_a n_b` in the (N, X, Y, Z) channel slots: `diag(-1, 0, 0, 0)`, i.e.
charge only (slot N holds `-U^(c)`, see [`HUBBARD_CHANNELS`](@ref)). A repulsive `V > 0` screens
the charge channel and never enhances the spin channels.

Register one direction per bond: `interaction` adds the Hermitian conjugate itself.
"""
const DENSITY_CHANNELS = ComplexF64[-1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0]

#####* the lookup table flattened into (block offsets, bond vector, matrix, adjoint) once, so a
#####* k-scan does not rebuild the bond vectors and adjoints at every momentum.
function bond_table(lookup::Dict, primitives::Vector{Vector{Float64}}, localDim::Integer)
    return [(b1 = localDim * (i - 1) + 1, b2 = localDim * (j - 1) + 1,
             δ = sum(offset .* primitives), value = value, value_adj = collect(value'))
            for ((i, j, offset), value) in lookup]
end

function assemble(table, k::Vector{Float64}, dim::Integer, localDim::Integer)::Matrix{ComplexF64}
    mat = zeros(ComplexF64, dim, dim)
    for bond in table
        phase = exp(1.0im * dot(k[1:length(bond.δ)], bond.δ))      # H(k)'s Bloch convention, see INTERACTION_CONVENTION
        r1 = bond.b1 : bond.b1 + localDim - 1
        r2 = bond.b2 : bond.b2 + localDim - 1
        mat[r1, r2] .+= bond.value .* phase
        mat[r2, r1] .+= bond.value_adj .* conj(phase)
    end
    return mat
end

"""
    interaction(strength, k; primitives, subs, localDim, lookup) -> Matrix{ComplexF64}
    interaction(strength, k, params::Vector{<:Param}; primitives, subs, localDim)
    interaction(strength, k, unitcell::UnitCell)

The interaction matrix `V(k)` of size `subs * localDim`, scaled by `strength`.

Each entry `(i, j, offset) => M` of the bond `lookup` (a `TightBindingToolkit.Lookup`) adds
`M e^{+i k.δ}` to block `(i, j)` and its Hermitian conjugate `M' e^{-i k.δ}` to block `(j, i)`,
with `δ = sum(offset .* primitives)`: the Bloch convention of TightBindingToolkit's `H(k)` for the
same bond ([`INTERACTION_CONVENTION`](@ref)). On-site bonds are no exception, so they count twice. The
channel index (`localDim`, 4 for N, X, Y, Z) is innermost.

`k` may also be a vector of momenta, in which case a vector of matrices is returned.
`interaction(s, k) == s * interaction(1.0, k)` exactly.
"""
function interaction(strength::Float64, k::Vector{Float64};
        primitives::Vector{Vector{Float64}},
        subs::Int64, localDim::Integer,
        lookup::Dict)::Matrix{ComplexF64}

    return strength * assemble(bond_table(lookup, primitives, localDim), k, subs * localDim, localDim)
end

function interaction(strength::Float64, k::Vector{Float64}, params::Vector{<:Param};
        primitives::Vector{Vector{Float64}},
        subs::Int64, localDim::Integer)::Matrix{ComplexF64}

    return interaction(strength, k;
        primitives = primitives, subs = subs, localDim = localDim,
        lookup = Lookup(params))
end

function interaction(strength::Float64, k::Vector{Float64},
    unitcell::UnitCell)::Matrix{ComplexF64}

    return interaction(strength, k;
        primitives = unitcell.primitives, subs = length(unitcell.basis), localDim = unitcell.localDim,
        lookup = Lookup(unitcell))
end

function interaction(strength::Float64, ks::Vector{Vector{Float64}};
    primitives::Vector{Vector{Float64}},
    subs::Int64, localDim::Integer,
    lookup::Dict)::Vector{Matrix{ComplexF64}}

    table = bond_table(lookup, primitives, localDim)
    return [strength * assemble(table, k, subs * localDim, localDim) for k in ks]
end

function interaction(strength::Float64, ks::Vector{Vector{Float64}}, params::Vector{<:Param};
    primitives::Vector{Vector{Float64}},
    subs::Int64, localDim::Integer,
    )::Vector{Matrix{ComplexF64}}

    return interaction(strength, ks;
        primitives = primitives, subs = subs, localDim = localDim,
        lookup = Lookup(params))
end

function interaction(strength::Float64, ks::Vector{Vector{Float64}}, unitcell::UnitCell)::Vector{Matrix{ComplexF64}}

    return interaction(strength, ks;
        primitives = unitcell.primitives, subs = length(unitcell.basis), localDim = unitcell.localDim,
        lookup = Lookup(unitcell))
end

#####* ---------------------------------------------------------------------------------------
#####* the TDHF kernel of a quartic interaction, in the channel slots
#####* ---------------------------------------------------------------------------------------

const PAULI_MATS = (ComplexF64[1 0; 0 1], ComplexF64[0 1; 1 0], ComplexF64[0 -im; im 0], ComplexF64[1 0; 0 -1])

"""
    tdhf_channel_vertex(K) -> Matrix{ComplexF64}

The 4×4 block, in the (N, X, Y, Z) slots of [`interaction`](@ref), of a mean-field kernel
`K[a, b, c, d]` on spin-1/2 sites: the linear change of the mean-field hopping
``\\delta t_{ab} = \\sum_{cd} K_{abcd}\\,\\delta\\langle c^\\dagger_c c_d\\rangle`` (a, b on the site the field acts on,
c, d on the site whose density changed). That is the time-dependent Hartree-Fock kernel
``\\delta\\Sigma/\\delta\\rho``, so the RPA built from it is the linear response of the same mean-field theory.

With channel operators ``O_p = c^\\dagger M_p c``, ``M_p = (\\sigma^p)^T/2`` (transposed, the convention
the stored bubble carries; see [`stored_bubble`](@ref RPAToolkit.Bubble.stored_bubble)), the kernel acts as
``\\varphi_p = \\sum_q W_{pq}\\,\\delta\\langle O_q\\rangle`` with
``W_{pq} = \\sum \\overline{M_p}_{ab} K_{abcd} \\overline{M_q}_{cd} / \\mathrm{Tr}(M^2)^2``, and the slot
matrix is ``-W/2``: minus, because chi is a positive response, and one half, because chi is the
per-spin bubble (twice the susceptibility of the sigma/2 operators). For ``U n_\\uparrow n_\\downarrow``
this is exactly [`HUBBARD_CHANNELS`](@ref)` .* U`.
"""
function tdhf_channel_vertex(K::AbstractArray{<:Number, 4})
    size(K) == (2, 2, 2, 2) || error("tdhf_channel_vertex: the kernel is $(size(K)); one spin-1/2 site needs 2x2x2x2.")
    M = [collect(transpose(P)) ./ 2 for P in PAULI_MATS]
    norm2 = real(sum(abs2, M[1]))              # Tr(M^2) = 1/2 for every channel
    W = zeros(ComplexF64, 4, 4)
    for q in 1:4, p in 1:4, d in 1:2, c in 1:2, b in 1:2, a in 1:2
        W[p, q] += conj(M[p][a, b]) * K[a, b, c, d] * conj(M[q][c, d])
    end
    return -(W ./ norm2^2) ./ 2
end

"""
    quartic_onsite_vertex(U) -> Matrix{ComplexF64}

The TDHF vertex of an on-site interaction ``\\sum_{abcd} U_{abcd}\\, c^\\dagger_a c_b c^\\dagger_c c_d`` (one
spin-1/2 site, MeanFieldToolkit's convention): Hartree and Fock together,
``K_{abcd} = U_{abcd} + U_{cdab} - U_{cbad} - U_{adcb}``, exactly the kernel of MeanFieldToolkit's
`IntraQuarticToHopping`. See [`tdhf_channel_vertex`](@ref).
"""
function quartic_onsite_vertex(U::AbstractArray{<:Number, 4})
    K = [U[a, b, c, d] + U[c, d, a, b] - U[c, b, a, d] - U[a, d, c, b] for a in 1:2, b in 1:2, c in 1:2, d in 1:2]
    return tdhf_channel_vertex(K)
end

"""
    quartic_hartree_vertex(U) -> (V_ij, V_ji)

The Hartree part of the TDHF vertex of a two-site interaction
``\\sum_{abcd} U_{abcd}\\, c^\\dagger_{ia} c_{ib} c^\\dagger_{jc} c_{jd}``: the field on site i from the density on
site j, and on j from i (MeanFieldToolkit's `InterQuarticToHopping` on-site terms). Its exchange
part renormalizes the bond hopping instead, which has no representation on per-site channels.
"""
function quartic_hartree_vertex(U::AbstractArray{<:Number, 4})
    Kij = [U[a, b, c, d] for a in 1:2, b in 1:2, c in 1:2, d in 1:2]
    Kji = [U[c, d, a, b] for a in 1:2, b in 1:2, c in 1:2, d in 1:2]
    return tdhf_channel_vertex(Kij), tdhf_channel_vertex(Kji)
end

"""
    hubbard_coulomb_interactions(uc; onsite, range) -> (params, cases)

On-site Hubbard plus a bare `1/r` density-density tail, for a 2D unit cell with one orbital per
basis site, in the form `run_rpa` reads from an interactions file:

- `params`: a `Vector{Param}`. The first holds the on-site Hubbard bonds, registered at `onsite / 2`
  (see [`HUBBARD_CHANNELS`](@ref)). The rest hold one `(1/r) .* DENSITY_CHANNELS` bond each, one
  direction per pair of sites with `0 < r <= range`.
- `cases`: `"Hubbard"` (the on-site term alone) and `"Coulomb"` (on-site plus the tail), each a
  vector of `Param` weights in the order of `params`.

Lengths are in the units of `uc.primitives`, and `onsite` in units of the tail's `1/r`. The RPA
strength then scales the whole interaction. At momentum `q` the assembled matrix has
`-U^(c)(q) = -(onsite + V(q))` in slot N and `U^(s) = onsite` in slots X, Y and Z on every site,
with `V(q) = sum_{0 < |δ| <= range} e^{+i q.δ} / |δ|`.

Save the returned values with the interaction unit cell and the slot layout under the keys
`"parameters"`, `"values"`, `"unit cell"` and `"vertex_slots" => VERTEX_SLOTS`; `run_rpa` checks
that unit cell against the model's and refuses a file without the layout.
"""
function hubbard_coulomb_interactions(uc::UnitCell; onsite::Real, range::Real)
    length(uc.primitives) == 2 ||
        error("hubbard_coulomb_interactions: needs a 2D unit cell, got $(length(uc.primitives)) primitives.")
    onsite > 0 || error("hubbard_coulomb_interactions: onsite = $onsite must be > 0.")
    range >= 0 || error("hubbard_coulomb_interactions: range = $range must be >= 0.")
    nsites = length(uc.basis)

    UParam = Param(1.0, 2)
    AddIsotropicBonds!(UParam, uc, 0.0, (onsite / 2) .* HUBBARD_CHANNELS, "Onsite Hubbard")
    length(UParam.unitBonds) == nsites ||
        error("hubbard_coulomb_interactions: expected $nsites on-site bonds, got $(length(UParam.unitBonds)).")

    params = [UParam]
    cases = Dict("Hubbard" => Float64[1.0], "Coulomb" => Float64[1.0])

    #####* enough lattice offsets to reach every site within `range`: the lattice lines are
    #####* |a1 x a2| / max|a| apart, and the basis can shift a bond by up to its own extent.
    A = reduce(hcat, uc.primitives)
    spacing = abs(det(A)) / maximum(norm.(uc.primitives))
    extent = maximum(norm(b2 .- b1) for b1 in uc.basis, b2 in uc.basis)
    M = ceil(Int, (range + extent) / spacing) + 1

    for i in 1:nsites, j in i:nsites, Rx in -M:M, Ry in -M:M
        (i == j && Rx == 0 && Ry == 0) && continue
        #####* (i, j, R) and (j, i, -R) are the same bond: keep one direction
        (i < j || Rx > 0 || (Rx == 0 && Ry > 0)) || continue
        r = norm(sum([Rx, Ry] .* uc.primitives) .+ (uc.basis[j] .- uc.basis[i]))
        r <= range || continue

        p = Param(1.0, 2)
        AddAnisotropicBond!(p, uc, i, j, [Rx, Ry], (1 / r) .* DENSITY_CHANNELS, r, "Coulomb_$(i)_$(j)_$(Rx)_$(Ry)")
        length(p.unitBonds) == 1 || error("hubbard_coulomb_interactions: bond ($i, $j, [$Rx, $Ry]) was not registered.")
        push!(params, p)
        push!(cases["Hubbard"], 0.0)
        push!(cases["Coulomb"], 1.0)
    end

    return params, cases
end

end
