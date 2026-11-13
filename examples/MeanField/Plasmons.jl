#=
Plasmons of a dilute two-dimensional metal with long-range Coulomb repulsion, from the RPA on top of
its Hartree-Fock bands.

    julia --project=examples/MeanField -t 8 examples/MeanField/Plasmons.jl

The model is the square lattice at low filling (near the band bottom the band is parabolic, mass
m = 1/(2t)) with e^2/(eps r) between electrons. MeanFieldToolkit decouples its two shortest-range
parts, the on-site U and the nearest-neighbour V = e^2/(eps a), in the Hartree and Fock channels: the
Fock term renormalizes the hopping, so the Hartree-Fock band is wider and its mass lighter than the
bare one. The RPA then screens the full 1/r tail (`coulomb_2d`, Ewald-summed) on top of those bands,
with the on-site TDHF vertex from the same mean-field theory (`rpa_vertex(mft; interactions = [1])`).

The loss function -Im chi_nn^RPA(q, w) shows the plasmon above the particle-hole continuum (whose
edge is read off Im chi0), with the two-dimensional dispersion w_p^2 = 2 pi n e^2 q / (eps m*) at small q,
m* the Hartree-Fock mass and n the electron density.

Writes examples/MeanField/data/plasmons.jld2 and .png.
=#

include(joinpath(@__DIR__, "common.jl"))
using JLD2, Plots, Printf

BLAS.set_num_threads(1)
const DATA = joinpath(@__DIR__, "data")
mkpath(DATA)

const t = 1.0
const E2 = 0.5            # e^2 / (eps a): the nearest-neighbour V and the 1/r tail
const U = 2.0
const FILLING = 0.05      # per spin-orbital: n = 0.1 electrons per site
const T = 0.01
const K_RPA = 320
const ETA = 0.01

#####* 1. Hartree-Fock of the on-site U and the nearest-neighbour V
mft = extended_hubbard(; U, V = E2, t, filling = FILLING, T, kSize = 96)
residual, iterations = solve!(mft, [FILLING, 0, 0, 0, 0.04, 0.04])
op = order_parameters(mft)
r = rpa_model(mft; residual = residual)
bond = real(op[first(k for k in keys(op) if startswith(k, "bond"))])
@printf("Hartree-Fock: %d iterations, residual %.1e, moment %.1e, bond amplitude %.4f\n",
        iterations, residual, abs(op["site 1 : Sz"]), bond)

#####* the Hartree-Fock band near its bottom: mass from the curvature at Γ
energy(k) = minimum(eigvals(bloch_hamiltonian(r.bands, k)))
dk = 1e-3
m_star = dk^2 / (energy([dk, 0.0]) + energy([-dk, 0.0]) - 2energy([0.0, 0.0]))
n = 2FILLING                              # electrons per unit area (a = 1)
k_F = sqrt(2pi * n)
v_F = k_F / m_star
@printf("band mass: bare %.4f, Hartree-Fock %.4f (the Fock term widens the band by %.1f%%)\n",
        1 / (2t), m_star, 100 * (1 / (2t) / m_star - 1))

#####* 2. the RPA: on-site TDHF vertex from the mean-field theory, plus the full 1/r tail
V_onsite = rpa_vertex(mft; interactions = [1])
V_coulomb = coulomb_2d(r.bands.primitives, r.bands.positions[1:2:end]; e2 = E2)
qs = [[q, 0.0] for q in range(0.01, 1.2, length = 120)]
omegas = collect(range(0.002, 1.6, length = 500))
start = time()
chi0 = dynamic_bubble(r.bands, qs, omegas, K_RPA; beta = r.beta, mu = r.mu, eta = ETA)
chi = dynamic_rpa(chi0, [V_onsite(q) + V_coulomb(q) for q in qs])
loss = spectral_map(chi, qs, r.bands.positions[1:2:end], (0,))
@printf("dynamic RPA: %d momenta x %d frequencies on a %dx%d grid in %.1f s\n", length(qs), length(omegas), K_RPA, K_RPA, time() - start)

plasmon = rpa_poles(loss, omegas)
qx = [q[1] for q in qs]
w_p(q) = sqrt(2pi * n * E2 * q / m_star)
#####* the particle-hole continuum's upper edge, the support of Im chi0 before broadening (on the
#####* lattice the band flattens, so this lies below the parabolic estimate v_F q + q^2/2m*)
edges = particle_hole_edges(r.bands, qs, K_RPA; beta = r.beta, mu = r.mu)
edge(q) = edges[argmin(abs.(qx .- q))]
println("\n q       ω_plasmon (RPA)   ω_p = sqrt(2π n e² q / m*)   continuum edge")
for q in (0.02, 0.05, 0.1, 0.2, 0.4)
    iq = argmin(abs.(qx .- q))
    @printf(" %.3f   %.4f            %.4f                       %.4f\n", qx[iq], plasmon[iq], w_p(qx[iq]), edge(qx[iq]))
end

jldsave(joinpath(DATA, "plasmons.jld2"); U, e2 = E2, filling = FILLING, m_star, v_F, qx, omegas,
        loss, plasmon, w_p = w_p.(qx), continuum_edge = edges)

p = heatmap(qx, omegas, log10.(max.(loss', 1e-4)); color = :inferno, xlabel = "q_x a", ylabel = "ω / t",
            title = @sprintf("charge response, HF + RPA with e²/εr, n = %.2f", n),
            colorbar_title = "log₁₀ (−Im χ_nn / π)", size = (720, 460))
plot!(p, qx, w_p.(qx); color = :cyan, lw = 1.5, ls = :dash, label = "√(2π n e² q / m*)")
plot!(p, qx, edges; color = :white, lw = 1, ls = :dot, label = "particle-hole continuum edge (from χ₀)")
plot!(p; legend = :bottomright, ylims = (0, omegas[end]), xlims = (0, qx[end]))
savefig(p, joinpath(DATA, "plasmons.png"))
