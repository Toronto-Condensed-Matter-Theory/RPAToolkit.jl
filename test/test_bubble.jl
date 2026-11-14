##### The bare bubble (src/Bubble.jl, src/DLR.jl) against the committed fixtures, and the real-frequency
##### bubble against the static one.
#####
##### The fixtures are reference bubbles from an independent implementation (the bare stage of RPAToolkit up
##### to 1.0, generator in the git history), in the stored convention.

const I2 = Matrix{ComplexF64}(I, 2, 2)
const HEX2 = [[1.0, 0.0], [-0.5, sqrt(3) / 2]]
const SQ2 = [[1.0, 0.0], [0.0, 1.0]]

##### the four fixture models (spin inner, every bond's conjugate)
fixture_hop_square(t = 1.0) = Dict([1, 0] => -t * I2, [-1, 0] => -t * I2, [0, 1] => -t * I2, [0, -1] => -t * I2)
fixture_hop_triangular(t = 0.4) = Dict(o => -t * I2 for o in ([1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]))
function fixture_hop_honeycomb(t = 0.3)
    AB = ComplexF64[0 1; 0 0]
    h = Dict([0, 0] => -t * (AB + AB'), [1, 0] => -t * AB, [-1, 0] => -t * collect(AB'),
             [1, 1] => -t * AB, [-1, -1] => -t * collect(AB'))
    return Dict(R => kron(m, I2) for (R, m) in h)
end
function fixture_hop_pdoublet(ts = -0.35, tp = 0.12)
    out = Dict{Vector{Int}, Matrix{ComplexF64}}()
    for (n1, n2) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
        d = n1 * HEX2[1] + n2 * HEX2[2]
        dh = d / norm(d)
        out[[n1, n2]] = kron(ComplexF64.(ts * dh * dh' + tp * (I - dh * dh')), I2)
    end
    return out
end
##### (units, hoppings, positions in reduced coordinates, sites)
const FIXTURE_MODELS = Dict(
    "square"     => (SQ2, fixture_hop_square(), [[0.0, 0.0]], 1),
    "triangular" => (HEX2, fixture_hop_triangular(), [[0.0, 0.0]], 1),
    "honeycomb"  => (HEX2, fixture_hop_honeycomb(), [[0.0, 0.0], [-2 / 3, -1 / 3]], 2),
    "pdoublet"   => (HEX2, fixture_hop_pdoublet(), [[0.0, 0.0], [0.0, 0.0]], 2))

function fixture_band_model(name)
    units, hop, posr, _ = FIXTURE_MODELS[name]
    positions = [p[1] .* units[1] .+ p[2] .* units[2] for p in posr for _ in 1:2]
    return band_model(units, hop; positions = positions)
end

sorted_spectrum(M) = sort(real(eigvals(Hermitian(0.5 .* (M + M')))))

@testset "Julia bare bubble" begin

    @testset "grid and path match the fixtures' mesh and the closed-path layout" begin
        f = load_spin_fixture("ising")
        bm = band_model(f.units, f.hoppings)
        n = isqrt(length(f.ks))
        @test maximum(maximum(abs.(k .- kf)) for (k, kf) in zip(k_grid(bm.primitives, n), f.ks)) < 1e-12
        ks, dist, ticks = k_path(SQ2, [[0, 0], [0.5, 0], [0.5, 0.5]]; num = 5)
        @test length(ks) == 15 && ks[1] ≈ [0.0, 0.0] && ks[5] ≈ [pi, 0.0] && ks[end] ≈ [0.0, 0.0]
        @test ticks ≈ [0.0, pi, 2pi, 2pi + sqrt(2) * pi]
        @test dist[end] ≈ ticks[end]
    end

    @testset "Lindhard and DLR reproduce every channel pair of the spin fixtures ($name)" for name in SPIN_FIXTURES
        f = load_spin_fixture(name)
        bm = band_model(f.units, f.hoppings)
        n = isqrt(length(f.ks))
        @test band_spin_symmetry(bm) == f.symmetry
        ops = channel_operators(1)
        scale = maximum(maximum(abs, v) for v in values(f.chis))
        for method in (:lindhard, :dlr)
            stored = stored_bubble(bm, f.ks, n; beta = f.beta, mu = f.mu, ops = ops, method = method, on_grid = true)
            worst = maximum(abs(stored[iq][a + 1, b + 1] - f.chis[(a, b)][iq]) for iq in eachindex(stored), a in 0:3, b in 0:3)
            @test worst / scale < (method == :lindhard ? 1e-12 : 1e-10)
        end
        ##### the path route (k - q diagonalized, not looked up) is the same sum
        off = stored_bubble(bm, f.ks, n; beta = f.beta, mu = f.mu, ops = ops)
        on = stored_bubble(bm, f.ks, n; beta = f.beta, mu = f.mu, ops = ops, on_grid = true)
        @test maximum(maximum(abs, a - b) for (a, b) in zip(off, on)) / scale < 1e-12
        if f.fact !== nothing          # the SU(2) factorization: spinless bubble x 1/2
            fact = stored_bubble(spinless_model(bm), f.ks, n; beta = f.beta, mu = f.mu,
                                 ops = channel_operators(1; spinful = false), on_grid = true)
            @test maximum(abs.(0.5 .* getindex.(fact, 1, 1) .- f.fact)) / scale < 1e-12
        end
    end

    @testset "per-orbital and closed-basis bubbles match the multi-site fixtures ($name)" for name in CASES
        _, _, _, ns = FIXTURE_MODELS[name]
        bm = fixture_band_model(name)
        meta = read_meta(name)
        beta, mu, n = parse(Float64, meta["beta"]), parse(Float64, meta["mu"]), parse(Int, meta["n_k"])
        table, header = readdlm(joinpath(FIXTURES, "$name.csv"), ',', Float64, '\n'; header = true)
        col = Dict(strip(String(h)) => i for (i, h) in enumerate(vec(header)))
        qs = k_grid(bm.primitives, n)
        basis = closed_basis_for_model(bm; seed = "density")
        @test length(basis) == parse(Int, meta["n_basis"])
        for (a, label) in ((0, "NN"), (3, "ZZ"))
            stored = stored_bubble(bm, qs, n; beta, mu, ops = channel_operators(ns; directions = [a]), on_grid = true)
            ref = [ComplexF64[table[q, col["re$(label)_$(i)$(j)"]] + im * table[q, col["im$(label)_$(i)$(j)"]] for i in 1:ns, j in 1:ns]
                   for q in axes(table, 1)]
            scale = maximum(maximum(abs, r) for r in ref)
            @test maximum(maximum(abs, s - r) for (s, r) in zip(stored, ref)) / scale < 1e-12
            ##### the closed basis: the fixture does not store its frame, so compare frame-invariantly
            nb = length(basis)
            storedC = stored_bubble(bm, qs, n; beta, mu, ops = channel_operators(ns; basis, directions = [a]), on_grid = true)
            refC = [ComplexF64[table[q, col["reC$(label)_$(i)$(j)"]] + im * table[q, col["imC$(label)_$(i)$(j)"]] for i in 1:nb, j in 1:nb]
                    for q in axes(table, 1)]
            @test maximum(maximum(abs.(sorted_spectrum(s) - sorted_spectrum(r))) for (s, r) in zip(storedC, refC)) / scale < 1e-12
            dlr = stored_bubble(bm, qs, n; beta, mu, ops = channel_operators(ns; directions = [a]), method = :dlr, on_grid = true)
            @test maximum(maximum(abs, s - r) for (s, r) in zip(dlr, ref)) / scale < 1e-10
        end
    end

    @testset "DLR quadrature" begin
        beta = 25.0
        g = dlr_grid(beta, 8.0; rtol = 1e-12)
        @test all(0 .< g.tau .< beta) && g.tau .+ g.tauc ≈ fill(beta, length(g))
        w0 = matsubara_weights(g, 0)
        w2 = matsubara_weights(g, 2)
        Omega2 = 4pi / beta
        worst = 0.0
        for E in range(-8.0, 8.0, length = 41)
            F(tau, tauc) = E >= 0 ? exp(-E * tau) : exp(E * tauc) * exp(-beta * E)   # e^{-E tau}, overflow-free
            Fs = [F(t, tc) for (t, tc) in zip(g.tau, g.tauc)]
            exact0 = abs(E) < 1e-12 ? beta : -expm1(-beta * E) / E
            exact2 = -expm1(-beta * E) / (E - im * Omega2)
            scale = max(abs(exact0), 1.0)
            worst = max(worst, abs(sum(w0 .* Fs) - exact0) / scale, abs(sum(w2 .* Fs) - exact2) / scale)
        end
        @test worst < 1e-9
    end

    @testset "Jacobi diagonalization" begin
        worst = 0.0
        rng = Xoshiro(7)
        for n in (1, 2, 3, 4, 6, 8, 12, 16), rep in 1:20
            A = randn(rng, ComplexF64, n, n)
            A = A + A'
            rep % 4 == 0 && n >= 2 && (A = kron(I2, A[1:n÷2, 1:n÷2]))          # exactly degenerate pairs
            m = size(A, 1)
            E, V = zeros(m), zeros(ComplexF64, m, m)
            RPAToolkit.Bubble.jacobi_eigen!(copy(A), V, E)
            worst = max(worst, maximum(abs.(A * V - V * Diagonal(E))), maximum(abs.(V' * V - I)),
                        maximum(abs.(E - eigvals(Hermitian(A)))))
        end
        @test worst < 1e-12
    end

    @testset "real-frequency bubble" begin
        ##### a two-site cell without inversion through a site and with an Ising-like spin splitting:
        ##### chi(q) != chi(-q) and chi^{XY} != 0, so every convention is visible
        function honeycomb_cell(ising)
            uc = UnitCell([[sqrt(3) / 2, 1.5], [-sqrt(3) / 2, 1.5]], 2)
            AddBasisSite!(uc, [0.0, 0.0]); AddBasisSite!(uc, [0.0, 1.0])
            S = SpinMats(1 // 2)
            t1 = Param(-1.0, 2); AddIsotropicBonds!(t1, uc, 1.0, S[4], "t1")
            t3 = Param(0.3, 2); AddIsotropicBonds!(t3, uc, 2.0, S[4] + ising * S[3], "t3")
            CreateUnitCell!(uc, [t1, t3])
            return uc
        end
        n, beta, mu = 10, 10.0, -0.4
        qs = [[0.37, 0.11], [1.1, -0.4], [0.0, 0.9]]           # off the k grid on purpose
        ws = collect(range(-3.0, 3.0, length = 25))
        for (ising, symmetry) in ((0.2, "U1z"), (0.0, "SU2"))
            bm = band_model(honeycomb_cell(ising))
            @test band_spin_symmetry(bm) == symmetry
            ##### w -> 0 is exactly what run_rpa assembles from the stored static bubble
            stored = stored_bubble(bm, qs, n; beta, mu, ops = channel_operators(2))
            data = Dict{String, Any}("spin_symmetry" => [findfirst(==("none"), SPIN_SYMMETRIES) - 1])
            for a in 0:3, b in 0:3
                blocks = [S[[4(i - 1) + a + 1 for i in 1:2], [4(j - 1) + b + 1 for j in 1:2]] for S in stored]
                data[pair_label(a, b)] = permutedims(cat(blocks...; dims = 3), (3, 1, 2))
            end
            combined = combine_chis(data; directions = [0, 1, 2, 3])
            dyn0 = dynamic_bubble(bm, qs, [0.0], n; beta, mu, eta = 1e-8, nbins = nothing)
            scale = maximum(maximum(abs, c) for c in combined)
            @test maximum(maximum(abs, dyn0[iq, 1] - combined[iq]) for iq in eachindex(qs)) / scale < 1e-6

            exact = dynamic_bubble(bm, qs, ws, n; beta, mu, eta = 0.1, nbins = nothing)
            binned = dynamic_bubble(bm, qs, ws, n; beta, mu, eta = 0.1, nbins = 6000)
            full = dynamic_bubble(bm, qs, ws, n; beta, mu, eta = 0.1, nbins = nothing, spin = :full)
            escale = maximum(maximum(abs, e) for e in exact)
            @test maximum(maximum(abs, a - b) for (a, b) in zip(exact, binned)) / escale < 1e-4
            @test maximum(maximum(abs, a - b) for (a, b) in zip(exact, full)) / escale < 1e-12
            ##### retarded: the absorptive part (chi - chi')/2i is sign(w) times a positive semidefinite matrix
            @test minimum(sign(w) * minimum(eigvals(Hermitian((exact[iq, iw] - exact[iq, iw]') / 2im)))
                          for iq in eachindex(qs), (iw, w) in enumerate(ws) if w != 0) > -1e-12
        end
    end

    @testset "run_bare end to end" begin
        mktempdir() do dir
            config = joinpath(dir, "square.yml")
            write(config, """
                beta: 10.0
                k_size: 8
                directions: [0, 1, 2, 3]
                fillings: {values: [0.3, 0.45]}
                k_points: [G, M1, M2]
                """)
            input = load_rpa_input(config)
            uc, hopping = square_model()
            save(input["unitcell"]["julia"], Dict("unit cell" => uc, "parameters" => hopping))
            runtime = run_bare(config)
            resolved = YAML.load_file(runtime)
            @test resolved["scan"]["source"] == "fillings"
            mus = resolved["mus"]["values"]
            @test resolved["fillings"]["values"] ≈ [0.3, 0.45] atol = 1e-9
            ##### a resolved runtime file is taken verbatim
            again = YAML.load_file(run_bare(runtime))
            @test again["mus"]["values"] == mus
            for mu in mus
                data = npzread(input["output"] * "_beta=10.0_mu=$(RPAToolkit.mu_label(mu)).npz")
                @test Int(only(data["spin_symmetry"])) == 0 && Int(only(data["bubble_spin_factorized"])) == 1
                @test size(data["chi_NN"]) == (64, 1, 1) && size(data["chi_ZZ_path"], 1) == size(data["path"], 1)
                chis = combine_chis(data; directions = [0, 1, 2, 3])
                ##### the same numbers straight from the bubble: SU(2), so 2 x (spinless / 2) in every channel
                bm = band_model(uc)
                direct = stored_bubble(spinless_model(bm), k_grid(bm.primitives, 8), 8; beta = 10.0, mu = mu,
                                       ops = channel_operators(1; spinful = false), on_grid = true)
                @test maximum(abs(chis[iq][a, a] - direct[iq][1, 1]) for iq in eachindex(chis), a in 1:4) < 1e-12
            end
        end
    end
end
