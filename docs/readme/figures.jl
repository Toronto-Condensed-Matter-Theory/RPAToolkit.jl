#=
The README figures.

    julia --project=docs/readme -t 8 docs/readme/figures.jl              # all of them
    julia --project=docs/readme -t 8 docs/readme/figures.jl magnons      # or a subset:
                                         # chi0_map, critical_U, magnons, plasmons

Each figure is a real RPAToolkit calculation, on grids kept small enough that the whole script runs
in a few minutes on 8 threads. The static figures go through the YAML pipeline (`run_rpa`), in a
temporary directory; the dynamic ones reuse the Hartree-Fock setups of examples/MeanField/.
PNGs are written to docs/src/assets/readme/.
=#

include(joinpath(@__DIR__, "style.jl"))
include(joinpath(@__DIR__, "..", "..", "examples", "MeanField", "common.jl"))
using JLD2, Printf

BLAS.set_num_threads(1)
const WHICH = isempty(ARGS) ? ["chi0_map", "critical_U", "magnons", "plasmons"] : ARGS

#####* ------------------------------------------------------------------------------------------
#####* the static pipeline: a nearest-neighbour model, on-site Hubbard U, run_rpa over a filling scan
#####* ------------------------------------------------------------------------------------------

square_uc() = (uc = UnitCell([[1.0, 0.0], [0.0, 1.0]], 2); AddBasisSite!(uc, [0.0, 0.0]); uc)
triangle_uc() = (uc = UnitCell([[1/2, sqrt(3)/2], [-1/2, sqrt(3)/2]], 2); AddBasisSite!(uc, [0.0, 0.0]); uc)

"""
Run the bare bubble and the static RPA for `make_uc()` with hopping t = -1 and an on-site Hubbard
interaction of unit strength, so the critical strength is U_c / t. Returns the run_rpa output.
"""
function static_run(make_uc, k_points; k_size, fillings, beta = 20.0)
    dir = mktempdir()
    cfg = joinpath(dir, "model.yml")
    write(cfg, """
    beta: $beta
    k_size: $k_size
    directions: [0, 1, 2, 3]
    fillings:
      min: $(first(fillings))
      max: $(last(fillings))
      n: $(length(fillings))
    k_points: [$(join(repr.(k_points), ", "))]
    instability:
      lower: 0.0
      upper: 20.0
    """)
    input = load_rpa_input(cfg)

    uc = make_uc()
    t1 = Param(-1.0, 2)
    AddIsotropicBonds!(t1, uc, 1.0, SpinMats(1 // 2)[4], "t1")
    CreateUnitCell!(uc, [t1])
    save(input["unitcell"]["julia"], Dict("unit cell" => uc, "parameters" => [t1]))

    #####* the Hubbard case alone: hubbard_coulomb_interactions with a tail too short to reach a neighbour
    params, cases = hubbard_coulomb_interactions(make_uc(); onsite = 1.0, range = 0.5)
    save(input["interactions"], Dict("parameters" => params, "values" => Dict("Hubbard" => cases["Hubbard"]),
                                     "unit cell" => make_uc(), "vertex_slots" => VERTEX_SLOTS))
    run_rpa(cfg; run_bare = true)
    return load(rpa_output_path(input))
end

"""Is the Cartesian `q` inside the first Brillouin zone of the reciprocal vectors `b1, b2`?"""
function in_first_bz(q, b1, b2)
    Gs = [b1, b2, b1 .+ b2, b1 .- b2]
    return all(abs(dot(q, G)) <= dot(G, G) / 2 + 1e-12 for G in Gs)
end

"""The corners of the first Brillouin zone, in order, closed: the Wigner-Seitz cell of b1, b2."""
function bz_corners(b1, b2)
    Gs = [s .* G for G in (b1, b2, b1 .+ b2, b1 .- b2) for s in (1, -1)]
    corners = Vector{Float64}[]
    for i in eachindex(Gs), j in i+1:length(Gs)
        A = [Gs[i]'; Gs[j]']
        abs(det(A)) < 1e-9 && continue
        p = A \ [dot(Gs[i], Gs[i]) / 2, dot(Gs[j], Gs[j]) / 2]
        all(dot(p, G) <= dot(G, G) / 2 + 1e-9 for G in Gs) || continue
        any(norm(p .- c) < 1e-9 for c in corners) || push!(corners, p)
    end
    sort!(corners; by = p -> atan(p[2], p[1]))
    return push!(corners, corners[1])
end

"""Bilinear, periodic interpolation of `values` on the `n × n` grid k = (i b1 + j b2)/n (j fastest)."""
function periodic_interpolate(values, n, q, Binv)
    r = Binv * q .* n
    i0, j0 = floor(Int, r[1]), floor(Int, r[2])
    x, y = r[1] - i0, r[2] - j0
    at(i, j) = values[mod(i, n) * n + mod(j, n) + 1]
    return (1 - x) * (1 - y) * at(i0, j0) + x * (1 - y) * at(i0 + 1, j0) +
           (1 - x) * y * at(i0, j0 + 1) + x * y * at(i0 + 1, j0 + 1)
end


#####* ------------------------------------------------------------------------------------------
#####* 1. the bare spin susceptibility over the Brillouin zone, triangular lattice
#####* ------------------------------------------------------------------------------------------

if "chi0_map" in WHICH
    println("\n=== chi0_map ===")
    k_size, filling = 96, 0.375
    out = static_run(triangle_uc, ["G", "M1", "K2''"]; k_size, fillings = [filling])
    entry = only(values(out))
    chi_zz = [real(c[4, 4]) for c in entry["combined_chis"]]
    b1, b2 = reciprocal_vectors(triangle_uc().primitives)
    Binv = inv(hcat(b1, b2))
    corners = bz_corners(b1, b2)
    L = 1.08 * maximum(norm.(corners))
    qx = range(-L, L, length = 401)
    qy = range(-L, L, length = 401)
    Z = [in_first_bz([x, y], b1, b2) ? periodic_interpolate(chi_zz, k_size, [x, y], Binv) : NaN for y in qy, x in qx]
    @printf("triangular, filling %.3f per spin, mu = %.4f: chi0_zz from %.4f to %.4f\n", filling, entry["mu"], extrema(filter(!isnan, Z))...)

    clims = extrema(filter(!isnan, Z))
    p = heatmap(qx, qy, Z; color = SEQUENTIAL, clims, colorbar = false,
                xlims = (-L, L), ylims = (-L, L), xlabel = L"q_x a", ylabel = L"q_y a")
    plot!(p, first.(corners), last.(corners); color = COLORS.ink, lw = 1.2, label = "")
    K, M = corners[findmin(c -> abs(atan(c[2], c[1])), corners[1:end-1])[2]], (corners[1] .+ corners[2]) ./ 2
    for (label, at, align) in ((L"\Gamma", [0.0, 0.0], :left), (L"K", K, :right), (L"M", M .+ [0.0, 0.2], :left))
        annotate!(p, at[1] + (align == :left ? 0.15 : -0.2), at[2] + 0.15, Plots.text(label, 12, COLORS.ink, align))
    end
    savefig_readme(with_colorbar(p, clims, L"\chi_0^{zz}(\mathbf{q})\ \ [1/t]"; size = (550, 416), widths = [0.895, 0.105]), "chi0_triangular")
end

#####* ------------------------------------------------------------------------------------------
#####* 2. the critical Hubbard U against density, square lattice, with the ordering vector
#####* ------------------------------------------------------------------------------------------

if "critical_U" in WHICH
    println("\n=== critical_U ===")
    fillings = collect(range(0.3, 0.5, length = 21))
    out = static_run(square_uc, ["G", "M1", "M2"]; k_size = 64, fillings)
    rows = sort([(n = 2e["filling"], U = e["Hubbard"]["critical strength"], Q = e["Hubbard"]["maximum reciprocal momentum"])
                 for e in values(out)]; by = r -> r.n)
    commensurate(r) = all(abs.(mod.(r.Q, 1.0) .- 0.5) .< 1e-6)
    for r in rows
        @printf("n = %.3f   U_c = %.4f t   Q = (%.4f, %.4f) 2π   %s\n", r.n, r.U, r.Q..., commensurate(r) ? "(π, π)" : "incommensurate")
    end
    n, U = [r.n for r in rows], [r.U for r in rows]
    c = commensurate.(rows)
    p = plot(n, U; color = COLORS.muted, lw = 1.2, label = "", xlabel = L"\textrm{electron\ density}\ n",
             ylabel = L"U_c / t", legend = :topright, xlims = (0.585, 1.015), ylims = (0, 1.1 * maximum(U)))
    scatter!(p, n[c], U[c]; color = COLORS.blue, markersize = 6, markerstrokecolor = :white, markerstrokewidth = 1.5,
             label = L"\mathbf{Q} = (\pi, \pi)")
    scatter!(p, n[.!c], U[.!c]; color = COLORS.orange, markersize = 6, markerstrokecolor = :white, markerstrokewidth = 1.5,
             label = L"\textrm{incommensurate}\ \mathbf{Q}")
    savefig_readme(p, "critical_U_square")
end

#####* ------------------------------------------------------------------------------------------
#####* 3. magnons of the Hartree-Fock Néel state, U = 8t (examples/MeanField/Magnons.jl)
#####* ------------------------------------------------------------------------------------------

if "magnons" in WHICH
    println("\n=== magnons ===")
    t, U = 1.0, 8.0
    lswt(q) = 2 * (4t^2 / U) * sqrt(max(0.0, 1 - ((cos(q[1]) + cos(q[2])) / 2)^2))
    mft = neel_hubbard(; U, t, kSize = 40)
    residual, iterations = solve!(mft, [0.5, 0, 0, 0.9, 0.5, 0, 0, -0.9])
    r = rpa_model(mft; residual = residual)
    V = rpa_vertex(mft)
    qs, dist, ticks = square_path(81)
    omegas = collect(range(0.005, 2.0, length = 300))
    start = time()
    chi0 = dynamic_bubble(r.bands, qs, omegas, 48; beta = r.beta, mu = r.mu, eta = 0.02)
    chi = dynamic_rpa(chi0, [V(q) for q in qs])
    A = spectral_map(chi, qs, r.bands.positions[1:2:end], (1, 2))
    @printf("HF: %d iterations, m = %.4f; dynamic RPA %d x %d in %.1f s\n", iterations,
            real(order_parameters(mft)["site 1 : Sz"]), length(qs), length(omegas), time() - start)

    Z = log10.(max.(A', 1e-3))
    clims = extrema(Z)
    p = heatmap(dist, omegas, Z; color = SEQUENTIAL, clims, colorbar = false,
                xticks = (ticks, [L"\Gamma", L"X", L"M", L"\Gamma"]), xlims = (0, dist[end]), ylims = (0, omegas[end]),
                ylabel = L"\omega / t", minorticks = false)
    plot!(p, dist, lswt.(qs); color = COLORS.orange, lw = 1.8, ls = :dash, label = L"\textrm{spin\ waves},\ J = 4t^2/U",
          legend = :topright, foreground_color_legend = nothing, background_color_legend = RGBA(1, 1, 1, 0.85))
    savefig_readme(with_colorbar(p, clims, L"\log_{10} A(\mathbf{q}, \omega)"), "magnons_square")
end

#####* ------------------------------------------------------------------------------------------
#####* 4. the plasmon of a dilute 2D metal with 1/r repulsion (examples/MeanField/Plasmons.jl)
#####* ------------------------------------------------------------------------------------------

if "plasmons" in WHICH
    println("\n=== plasmons ===")
    t, E2, U, FILLING, K_RPA = 1.0, 0.5, 2.0, 0.05, 320
    mft = extended_hubbard(; U, V = E2, t, filling = FILLING, T = 0.01, kSize = 96)
    residual, _ = solve!(mft, [FILLING, 0, 0, 0, 0.04, 0.04])
    r = rpa_model(mft; residual = residual)
    energy(k) = minimum(eigvals(bloch_hamiltonian(r.bands, k)))
    dk = 1e-3
    m_star = dk^2 / (energy([dk, 0.0]) + energy([-dk, 0.0]) - 2energy([0.0, 0.0]))
    n = 2FILLING
    V_onsite = rpa_vertex(mft; interactions = [1])
    V_coulomb = coulomb_2d(r.bands.primitives, r.bands.positions[1:2:end]; e2 = E2)
    qs = [[q, 0.0] for q in range(0.01, 1.2, length = 120)]
    omegas = collect(range(0.002, 1.6, length = 500))
    start = time()
    chi0 = dynamic_bubble(r.bands, qs, omegas, K_RPA; beta = r.beta, mu = r.mu, eta = 0.01)
    chi = dynamic_rpa(chi0, [V_onsite(q) + V_coulomb(q) for q in qs])
    loss = spectral_map(chi, qs, r.bands.positions[1:2:end], (0,))
    edges = particle_hole_edges(r.bands, qs, K_RPA; beta = r.beta, mu = r.mu)
    qx = first.(qs)
    @printf("m* = %.4f; dynamic RPA %d x %d on %d^2 k in %.1f s\n", m_star, length(qs), length(omegas), K_RPA, time() - start)

    w_p(q) = sqrt(2pi * n * E2 * q / m_star)
    Z = log10.(max.(loss', 1e-3))
    clims = extrema(Z)
    p = heatmap(qx, omegas, Z; color = SEQUENTIAL, clims, colorbar = false,
                xlims = (0, qx[end]), ylims = (0, omegas[end]), xlabel = L"q_x a", ylabel = L"\omega / t")
    small = qx .<= 0.4
    plot!(p, qx[small], w_p.(qx[small]); color = COLORS.orange, lw = 1.8, ls = :dash, label = L"\omega_p = \sqrt{2\pi n e^2 q / m^*}")
    plot!(p, qx, edges; color = COLORS.ink, lw = 1.2, ls = :dot, label = "particle-hole continuum edge",
          legend = :topleft)
    savefig_readme(with_colorbar(p, clims, L"\log_{10}\,[-\mathrm{Im}\,\chi_{nn}(\mathbf{q}, \omega)/\pi]"), "plasmons_square")
end
