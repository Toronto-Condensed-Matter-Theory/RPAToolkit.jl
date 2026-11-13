module Dynamics

#=
The RPA at real frequency, chi_RPA(q, w) = (1 - chi0(q, w) V(q))^-1 chi0(q, w), on top of
`dynamic_bubble`, and what is needed to read collective modes off it:

- `dynamic_rpa` solves the RPA at every (q, w);
- `unfold_vector` and `spectral_map` turn a supercell's response into the spectral function of the
  original lattice (a Néel cell's magnon at Q, rather than at Q and Q + (pi, pi) folded together);
- `coulomb_2d` is a long-range 1/r interaction, Ewald-summed, whose V(q) diverges as 1/|q|: what a
  two-dimensional plasmon needs, and what a real-space tail truncated at a finite range cannot give.

The chi0 here is `dynamic_bubble`'s: one 4d x 4d matrix per (q, w), channel (N, X, Y, Z)
innermost, normalized as the per-spin bubble the Graser vertices multiply (the static pipeline's
convention), so the same `interaction()` matrices apply.
=#

using LinearAlgebra, SpecialFunctions
using ..RPAToolkit.Response: perform_RPA

export dynamic_rpa, unfold_vector, spectral_weight, spectral_map, coulomb_2d, rpa_poles

"""
    dynamic_rpa(chi0, V) -> Matrix{Matrix{ComplexF64}}

``\\chi_{RPA}(q, \\omega) = (1 - \\chi_0(q, \\omega) V(q))^{-1} \\chi_0(q, \\omega)`` for `chi0[iq, iw]`
(from [`dynamic_bubble`](@ref RPAToolkit.Bubble.dynamic_bubble)) and `V[iq]`, the interaction at each momentum in the same layout
(`interaction()`, [`coulomb_2d`](@ref), or one constant matrix for an on-site vertex).
"""
function dynamic_rpa(chi0::AbstractMatrix{<:AbstractMatrix}, V::AbstractVector{<:AbstractMatrix})
    size(chi0, 1) == length(V) ||
        error("dynamic_rpa: $(size(chi0, 1)) momenta in chi0 but $(length(V)) interaction matrices.")
    out = similar(chi0, Matrix{ComplexF64})
    Threads.@threads for idx in CartesianIndices(chi0)
        out[idx] = perform_RPA(Matrix{ComplexF64}(chi0[idx]), Matrix{ComplexF64}(V[idx[1]]); return_matrix = true)
    end
    return out
end
dynamic_rpa(chi0::AbstractMatrix{<:AbstractMatrix}, V::AbstractMatrix) = dynamic_rpa(chi0, fill(V, size(chi0, 1)))

"""
    unfold_vector(positions, q, channels) -> Matrix{ComplexF64}

One column per channel in `channels` (0 = N, 1 = X, 2 = Y, 3 = Z): the operator of the ORIGINAL
lattice at momentum `q`, ``O^a(q) = \\sum_r e^{-iq\\cdot r} c^\\dagger_r \\sigma^a c_r / 2``, written in the
supercell's per-site channels with `positions` the sites of the cell,

    u_{(i, a)} = e^{+i q.b_i} / sqrt(n_sites),

so that `u' chi u` (chi in the static pipeline's convention, channel innermost) is the
original lattice's per-site susceptibility at `q`. Pinned numerically: a paramagnetic two-site
cell, unfolded this way, reproduces the one-site bubble at every q and frequency.
"""
function unfold_vector(positions::AbstractVector, q::AbstractVector, channels)
    ns = length(positions)
    U = zeros(ComplexF64, 4ns, length(channels))
    for (c, a) in enumerate(channels), i in 1:ns
        U[4(i - 1) + a + 1, c] = cis(q[1] * positions[i][1] + q[2] * positions[i][2]) / sqrt(ns)
    end
    return U
end

"""
    spectral_weight(chi, U) -> Float64

``\\sum_c \\mathrm{Im}(u_c^\\dagger \\chi u_c)/\\pi`` over the columns of `U`: the spectral function of
the operators `U` describes. Non-negative for w > 0 when `chi` is a retarded response.
"""
spectral_weight(chi::AbstractMatrix, U::AbstractVecOrMat) = sum(imag(dot(u, chi * u)) for u in eachcol(U)) / pi

"""
    spectral_map(chi, qs, positions, channels) -> Matrix{Float64}

[`spectral_weight`](@ref) of `chi[iq, iw]` with the unfolded operators of `channels` at each
`qs[iq]`, as an `(nq, nw)` array: e.g. `channels = (1, 2)` for the transverse spin response of a
magnet ordered along z, `(0,)` for the density response.
"""
function spectral_map(chi::AbstractMatrix{<:AbstractMatrix}, qs::AbstractVector, positions::AbstractVector, channels)
    out = zeros(size(chi))
    for iq in axes(chi, 1)
        U = unfold_vector(positions, qs[iq], channels)
        for iw in axes(chi, 2)
            out[iq, iw] = spectral_weight(chi[iq, iw], U)
        end
    end
    return out
end

"""
    rpa_poles(spectral, omegas; contrast = 5.0) -> Vector{Float64}

For each row of `spectral` (one momentum), the frequency of its largest peak, or `NaN` when that
peak is less than `contrast` times the row's median (no sharp mode there): a quick read-out of a
collective mode's dispersion. The peak is refined by a parabola through its three highest points.
"""
function rpa_poles(spectral::AbstractMatrix, omegas::AbstractVector; contrast::Real = 5.0)
    out = fill(NaN, size(spectral, 1))
    for iq in axes(spectral, 1)
        row = view(spectral, iq, :)
        i = argmax(row)
        row[i] < contrast * sort(collect(row))[cld(length(row), 2)] && continue
        if 1 < i < length(omegas)
            y1, y2, y3 = row[i - 1], row[i], row[i + 1]
            denom = y1 - 2y2 + y3
            shift = denom == 0 ? 0.0 : 0.5 * (y1 - y3) / denom
            out[iq] = omegas[i] + shift * (omegas[i + 1] - omegas[i])
        else
            out[iq] = omegas[i]
        end
    end
    return out
end

#####* the Ewald sum v_ij(q) = sum_R e^{-iq.R} / |R + tau|, tau = b_j - b_i, the R + tau = 0 term left out
function ewald_lattice_sum(q, tau, a1, a2, b1, b2, area, alpha)
    rcut = 6.0 / alpha
    gcut = 12.0 * alpha
    nR = ceil(Int, rcut / min(norm(a1), norm(a2), abs(a1[1] * a2[2] - a1[2] * a2[1]) / max(norm(a1), norm(a2)))) + 2
    nG = ceil(Int, gcut / min(norm(b1), norm(b2), abs(b1[1] * b2[2] - b1[2] * b2[1]) / max(norm(b1), norm(b2)))) + 2
    total = zero(ComplexF64)
    for m in -nR:nR, n in -nR:nR
        R = m .* a1 .+ n .* a2
        r = norm(R .+ tau)
        (r < 1e-12 || r > rcut + norm(tau)) && continue
        total += cis(-(q[1] * R[1] + q[2] * R[2])) * erfc(alpha * r) / r
    end
    for m in -nG:nG, n in -nG:nG
        k = q[1:2] .+ m .* b1 .+ n .* b2
        kk = norm(k)
        (kk < 1e-12 || kk > gcut + norm(q[1:2])) && continue        # the G = -q term is the neutralizing background
        total += (2pi / area) * cis(k[1] * tau[1] + k[2] * tau[2]) * erfc(kk / (2alpha)) / kk
    end
    norm(tau) < 1e-12 && (total -= 2alpha / sqrt(pi))
    return total
end

"""
    coulomb_2d(primitives, positions; e2 = 1.0, eps = 1.0, onsite = 0.0, alpha = nothing) -> Function

The long-range Coulomb interaction ``H = \\frac{1}{2}\\sum_{r \\ne r'} \\frac{e^2}{\\varepsilon |r - r'|} n_r n_{r'}``
plus an on-site Hubbard `onsite` ``\\sum_r n_{r\\uparrow} n_{r\\downarrow}``, as `q -> V(q)`, a `4n × 4n`
matrix (n = `length(positions)`) in `interaction()`'s layout, charge slot only:

    V_{(i,N),(j,N)}(q) = -( onsite delta_ij + 2 v_ij(q) ),
    v_ij(q) = (e^2 / eps) sum_R e^{+iq.R} / |R + b_j - b_i|      (Ewald-summed, 2 pi e^2 / (eps A |q|) as q -> 0)

The factor 2 is there because chi is the PER-SPIN bubble: the RPA then reads 1 + 2 v chi0 = 1 + v Π
with Π the full density bubble, so the plasmon of a 2D electron gas comes out at
``\\omega_p^2 = 2\\pi n e^2 q / (\\varepsilon m)``. The phase convention is `interaction()`'s
([`INTERACTION_CONVENTION`](@ref RPAToolkit.Interactions.INTERACTION_CONVENTION), ``e^{+iq\\cdot R}``), so the two can be added. At q = 0 the divergent
G = 0 term is dropped (a neutralizing background).
"""
function coulomb_2d(primitives, positions; e2::Real = 1.0, eps::Real = 1.0, onsite::Real = 0.0,
                    alpha::Union{Nothing, Real} = nothing)
    a1, a2 = Float64.(primitives[1][1:2]), Float64.(primitives[2][1:2])
    area = abs(a1[1] * a2[2] - a1[2] * a2[1])
    B = 2pi * inv(transpose(hcat(a1, a2)))
    b1, b2 = B[:, 1], B[:, 2]
    al = alpha === nothing ? sqrt(pi / area) : Float64(alpha)
    pos = [Float64.(p[1:2]) for p in positions]
    ns = length(pos)
    return function (q)
        V = zeros(ComplexF64, 4ns, 4ns)
        for i in 1:ns, j in 1:ns
            #####* the sum below is written with e^{-iq.R}: evaluated at -q it is INTERACTION_CONVENTION's e^{+iq.R}
            v = (e2 / eps) * ewald_lattice_sum(-q[1:2], pos[j] .- pos[i], a1, a2, b1, b2, area, al)
            V[4(i - 1) + 1, 4(j - 1) + 1] = -((i == j ? onsite : 0.0) + 2v)
        end
        return V
    end
end

end # module
