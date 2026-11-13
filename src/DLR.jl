module DLR

#=
The discrete Lehmann representation (DLR) of imaginary-time functions, as far as the bare bubble
needs it: a set of imaginary-time nodes tau_k and weights w_k such that

    F(i Omega_n) = int_0^beta e^{i Omega_n tau} F(tau) dtau  ≈  sum_k w_k(n) F(tau_k)

to relative accuracy `rtol` for every F with a spectral representation supported in
[-E_uv, E_uv] (Kaye, Chen & Parcollet, Phys. Rev. B 105, 235115 (2022)).

The grid construction is adapted from Lehmann.jl, src/discrete/builder.jl and src/discrete/kernel.jl
(MIT License, Copyright (c) 2021 Kun Chen, Tao Wang, Xiansheng Cai):

1. Discretize the dimensionless kernel K(t, w) = e^{-w t} / (1 + e^{-w}), t in (0, 1),
   w in [-Lambda, Lambda], Lambda = beta E_uv, on composite Chebyshev grids whose panels are
   dyadically refined towards w = 0 and t = 0, 1.
2. A pivoted QR of that matrix picks the DLR real frequencies w_l (its epsilon-rank columns).
3. A pivoted QR of the transpose, restricted to those columns, picks the imaginary-time nodes.

The fermionic kernel is used for bosonic functions too, as in Lehmann.jl and libdlr: the DLR is a
basis for the span of e^{-w tau}, whatever the statistics. Points past t = 1/2 are built as mirror
images, and their distance 1 - t is kept exactly (`tauc`), because that is what the kernel and the
occupation factors near tau = beta depend on.
=#

using LinearAlgebra

export DLRGrid, dlr_grid, dlr_kernel, matsubara_weights

"""
    DLRGrid

`omega` (real frequencies), `tau` (imaginary times in (0, beta)) and `tauc = beta - tau` (kept
exactly) of a discrete Lehmann representation at inverse temperature `beta`, cutoff `euv` and
accuracy `rtol`. Build one with [`dlr_grid`](@ref).
"""
struct DLRGrid
    beta::Float64
    euv::Float64
    rtol::Float64
    omega::Vector{Float64}
    tau::Vector{Float64}
    tauc::Vector{Float64}
end

Base.length(g::DLRGrid) = length(g.omega)

#####* Chebyshev nodes of the first kind on [-1, 1], ascending
chebyshev_nodes(degree::Int) = [-cos((2j - 1) * pi / (2degree)) for j in 1:degree]

#####* `degree` Chebyshev nodes in every panel between consecutive break points
function composite_grid(breaks::Vector{Float64}, degree::Int)
    x = chebyshev_nodes(degree)
    return reduce(vcat, [a .+ (b - a) .* (x .+ 1) ./ 2 for (a, b) in zip(breaks[1:end-1], breaks[2:end])])
end

#####* panels on [-Lambda, Lambda], dyadically dense towards 0 from both sides (Lehmann's ωChebyGrid)
function omega_breaks(Lambda::Float64)
    npo = max(1, ceil(Int, log2(Lambda)))
    pos = [Lambda / 2^(npo - i) for i in 1:npo]
    return vcat(-reverse(pos), 0.0, pos)
end

#####* panel break points on [0, 1/2], dyadically dense towards 0 (half of Lehmann's τChebyGrid)
function tau_half_breaks(Lambda::Float64)
    npt = max(1, ceil(Int, log2(Lambda)) - 2)
    return vcat(0.0, [1.0 / 2^(npt - i + 1) for i in 1:npt])
end

"""
    dlr_kernel(t, tc, w) -> Float64

The dimensionless kernel ``K(t, w) = e^{-wt}/(1 + e^{-w})`` with `tc = 1 - t` supplied exactly,
evaluated without overflow for either sign of `w`.
"""
dlr_kernel(t::Real, tc::Real, w::Real) = w >= 0 ? exp(-w * t) / (1 + exp(-w)) : exp(w * tc) / (1 + exp(w))

"""
    dlr_grid(beta, euv; rtol = 1e-12, degree = 24) -> DLRGrid

The DLR for functions with spectral support in `[-euv, euv]` at inverse temperature `beta`.
"""
function dlr_grid(beta::Real, euv::Real; rtol::Real = 1e-12, degree::Int = 24)
    beta > 0 && euv > 0 || error("dlr_grid: beta and euv must be positive.")
    Lambda = max(Float64(beta * euv), 4.0)

    w_fine = composite_grid(omega_breaks(Lambda), degree)
    half = composite_grid(tau_half_breaks(Lambda), degree)
    t_fine = vcat(half, reverse(1 .- half))         # mirror: t = 1 - half
    tc_fine = vcat(1 .- half, reverse(half))         # 1 - t, exact on the mirrored half

    K = [dlr_kernel(t, tc, w) for (t, tc) in zip(t_fine, tc_fine), w in w_fine]

    #####* the epsilon-rank, from the singular values, then the smallest k whose trailing R block of
    #####* the pivoted QR is below rtol (Golub & Van Loan, 4th ed., p. 487), as Lehmann's ωQR
    s = svdvals(K)
    rank = something(findfirst(x -> x / s[1] < rtol, s), length(s) + 1) - 1
    F = qr(K, ColumnNorm())
    R = F.R
    for idx in rank:min(size(R)...)
        if opnorm(R[idx:end, idx:end]) / s[1] < rtol
            rank = idx
            break
        end
    end
    w_index = F.p[1:rank]

    #####* imaginary-time nodes: pivoted QR of the transpose, on the DLR frequencies only
    t_index = qr(transpose(K[:, w_index]), ColumnNorm()).p[1:rank]
    order = sortperm(t_fine[t_index])
    t_index = t_index[order]

    return DLRGrid(Float64(beta), Float64(euv), Float64(rtol),
                   sort(w_fine[w_index]) ./ beta,
                   t_fine[t_index] .* beta, tc_fine[t_index] .* beta)
end

#####* tanh(x / 2) / x, regular at x = 0
tanh_half_over(x::Real) = abs(x) < 1e-6 ? 0.5 - x^2 / 24 : tanh(x / 2) / x

"""
    matsubara_weights(g::DLRGrid, n = 0) -> Vector{ComplexF64}

Weights ``w_k`` with ``\\int_0^\\beta e^{i\\Omega_n\\tau} F(\\tau)\\,d\\tau \\approx \\sum_k w_k F(\\tau_k)``,
``\\Omega_n = 2\\pi n/\\beta`` (bosonic), for any `F` the grid represents.

Each DLR basis function ``K(\\tau, \\omega_l) = e^{-\\omega_l\\tau}/(1 + e^{-\\beta\\omega_l})`` integrates to
``\\tanh(\\beta\\omega_l/2)/(\\omega_l - i\\Omega_n)``; fitting F at the nodes and integrating the fit
gives ``w = K^{-T} I``.
"""
function matsubara_weights(g::DLRGrid, n::Integer = 0)
    beta = g.beta
    Omega = 2pi * n / beta
    K = [dlr_kernel(t / beta, tc / beta, beta * w) for (t, tc) in zip(g.tau, g.tauc), w in g.omega]
    integral = ComplexF64[n == 0 ? beta * tanh_half_over(beta * w) : tanh(beta * w / 2) / (w - im * Omega)
                          for w in g.omega]
    return transpose(K) \ integral
end

end # module
