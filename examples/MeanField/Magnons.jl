#=
Magnons of the Hartree-Fock Néel state of the square-lattice Hubbard model, from the RPA on top of it.

    julia --project=. -t 8 examples/MeanField/Magnons.jl

For each U:
1. MeanFieldToolkit solves unrestricted on-site Hartree-Fock on the √2×√2 cell (Anderson mixing),
   converging to Néel order along z.
2. `rpa_model` / `rpa_vertex` hand its bands and the TDHF vertex of the same Hubbard tensor to the
   RPA. Because the two are consistent, the transverse static RPA has an exact zero mode at the
   ordering vector (the Goldstone mode); the script checks that the largest transverse eigenvalue
   of chi0 V at q = 0 is 1.
3. The transverse spin spectral function A(q, w) = -Im chi^{+-}_RPA / pi, unfolded to the original
   square lattice, is computed along Γ → X → M → Γ. Its sharp low-energy branch is the magnon, below
   the Stoner continuum that starts at the Hartree-Fock gap; at large U it approaches linear spin-wave
   theory, w = 2J sqrt(1 - γ_q^2), J = 4t^2/U, γ_q = (cos qx + cos qy)/2.

Writes examples/MeanField/data/magnons_U=<U>.jld2 and .png.
=#

include(joinpath(@__DIR__, "common.jl"))
using JLD2, Plots, Printf

BLAS.set_num_threads(1)
const DATA = joinpath(@__DIR__, "data")
mkpath(DATA)

const t = 1.0
const Us = length(ARGS) > 0 ? parse.(Float64, ARGS) : [4.0, 8.0, 12.0]
const K_HF = 40           # Hartree-Fock k grid (magnetic cells)
const K_RPA = 48          # bubble k grid
const ETA = 0.02

lswt(q, U) = 2 * (4t^2 / U) * sqrt(max(0.0, 1 - ((cos(q[1]) + cos(q[2])) / 2)^2))

qs, dist, ticks = square_path(31)
results = Dict{Float64, Any}()

for U in Us
    @printf("\n=== U = %.1f t ===\n", U)
    mft = neel_hubbard(; U, t, kSize = K_HF)
    residual, iterations = solve!(mft, [0.5, 0, 0, 0.9, 0.5, 0, 0, -0.9])
    m = real(order_parameters(mft)["site 1 : Sz"])
    r = rpa_model(mft; residual = residual)
    V = rpa_vertex(mft)
    E = band_energies(r.bands, k_grid(r.bands.primitives, K_RPA))
    gap = minimum(E[3, :]) - maximum(E[2, :])
    @printf("Hartree-Fock: %d iterations, residual %.1e, staggered moment m = %.4f, gap %.4f t\n", iterations, residual, m, gap)

    #####* the Goldstone check: transverse chi0 V at q = 0 (the ordering vector, folded)
    chi_static = static_chi(r.bands, [[0.0, 0.0]], K_RPA; beta = r.beta, mu = r.mu)[1]
    leading = sort(real(eigvals(chi_static * V([0.0, 0.0]))), rev = true)[1:2]
    @printf("Goldstone check: largest eigenvalues of chi0 V at q = 0: %.8f, %.8f (exactly 1 for TDHF)\n", leading...)

    #####* the dynamic transverse response along the path
    #####* up to the continuum edge or three times the LSWT bandwidth, whichever is lower, resolving eta
    omega_max = min(1.2 * gap, 3 * lswt([pi, 0.0], U))
    omegas = collect(range(0.005, omega_max, length = 400))
    start = time()
    chi0 = dynamic_bubble(r.bands, qs, omegas, K_RPA; beta = r.beta, mu = r.mu, eta = ETA)
    chi = dynamic_rpa(chi0, [V(q) for q in qs])
    A = spectral_map(chi, qs, r.bands.positions[1:2:end], (1, 2))
    @printf("dynamic RPA: %d momenta x %d frequencies in %.1f s\n", length(qs), length(omegas), time() - start)

    magnon = rpa_poles(A, omegas)
    for (label, q) in (("X  (pi, 0)", [pi, 0.0]), ("(pi/2, pi/2)", [pi / 2, pi / 2]))
        iq = argmin([norm(k .- q) for k in qs])
        @printf("  magnon at %-13s %.4f t   (LSWT %.4f t)\n", label, magnon[iq], lswt(q, U))
    end

    results[U] = (; U, m, gap, residual, leading, qs, dist, ticks, omegas, A, magnon,
                  lswt = [lswt(q, U) for q in qs])
    jldsave(joinpath(DATA, "magnons_U=$(U).jld2"); U, m, gap, residual, goldstone = leading, qs, dist, ticks,
            omegas, spectral = A, magnon, lswt = [lswt(q, U) for q in qs])

    p = heatmap(dist, omegas, log10.(max.(A', 1e-4)); color = :inferno, xlabel = "", ylabel = "ω / t",
                title = @sprintf("transverse spin response, HF Néel + RPA, U = %.0f t", U),
                colorbar_title = "log₁₀ A(q, ω)", xticks = (ticks, ["Γ", "X", "M", "Γ"]), size = (720, 460))
    plot!(p, dist, [lswt(q, U) for q in qs]; color = :cyan, lw = 1.5, ls = :dash, label = "LSWT, J = 4t²/U")
    gap < omega_max && hline!(p, [gap]; color = :white, lw = 1, ls = :dot, label = "HF gap (Stoner continuum)")
    plot!(p; legend = :topright, ylims = (0, omega_max))
    savefig(p, joinpath(DATA, "magnons_U=$(U).png"))
end

#####* the zone-boundary magnon against LSWT, as U grows
println("\nU      m        gap      ω_X (RPA)   ω_X (LSWT)   ratio")
for U in Us
    res = results[U]
    iq = argmin([norm(k .- [pi, 0.0]) for k in res.qs])
    @printf("%-6.1f %.4f   %.4f   %.4f      %.4f       %.3f\n", U, res.m, res.gap, res.magnon[iq], res.lswt[iq], res.magnon[iq] / res.lswt[iq])
end
