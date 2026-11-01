##### perform_RPA, effective_interaction and find_instability

##### find_instability as it was before the eigenvalue shortcut: every bisection step diagonalizes
##### chi_RPA at every q. Kept as the reference the fast version must reproduce.
function reference_critical_strength(chis, V_unit; steps = 32, lower = 0.0, upper = 10.0)
    rpa(s)  =   [eigen(inv(I - chi * (s .* V)) * chi) for (chi, V) in zip(chis, V_unit)]
    for _ in 1:steps
        mid     =   (upper + lower) / 2
        lowest  =   minimum(real(first(e.values)) for e in rpa(mid))
        lowest < -1e-6 ? (upper = mid) : (lower = mid)
    end
    return lower
end

random_psd(n, scale)        =   (A = randn(ComplexF64, n, n); scale .* (A * A') ./ n)
random_hermitian(n)         =   (B = randn(ComplexF64, n, n); (B + B') ./ 2)

@testset "response" begin
    resolution  =   10.0 / 2^32

    @testset "single channel: Stoner" begin
        chi, U  =   0.3, 2.0
        @test perform_RPA(fill(chi + 0im, 1, 1), fill(U + 0im, 1, 1); return_matrix = true)[1] ≈ chi / (1 - chi * U)
        @test effective_interaction(fill(chi + 0im, 1, 1), fill(U + 0im, 1, 1))[1] ≈ U / (1 - chi * U)

        qs      =   range(0, 2pi, length = 41)[1:end-1]
        chis    =   [fill(0.2 + 0.1 * cos(q) + 0im, 1, 1) for q in qs]
        ks      =   [[q, 0.0] for q in qs]
        result  =   find_instability(chis, ks, [fill(1.0 + 0im, 1, 1) for _ in qs])
        @test abs(result["critical strength"] - 1 / 0.3) <= resolution
        @test result["maximum momentum"] ≈ [0.0, 0.0]
        ##### a repulsive charge vertex only screens: nothing goes unstable below `upper`
        @test find_instability(chis, ks, [fill(-1.0 + 0im, 1, 1) for _ in qs])["critical strength"] ≈ 10.0
    end

    @testset "matches the full bisection" begin
        Random.seed!(20261031)
        for (n, nq) in ((4, 30), (8, 20))
            chis    =   [random_psd(n, 0.2) for _ in 1:nq]
            V_unit  =   [random_hermitian(n) for _ in 1:nq]
            ks      =   [[0.1 * i, 0.0] for i in 1:nq]
            fast    =   find_instability(chis, ks, V_unit)["critical strength"]
            @test abs(fast - reference_critical_strength(chis, V_unit)) <= resolution
        end
    end

    @testset "Lookup and matrix forms agree" begin
        uc          =   square_uc()
        params, cases   =   hubbard_coulomb_interactions(square_uc(); onsite = 2.0, range = 1.5)
        lookup      =   case_lookup(params, cases["Coulomb"])
        ks          =   [[2pi * i / 6, 2pi * j / 6] for i in 0:5 for j in 0:5]
        chis        =   [Matrix{ComplexF64}(Diagonal(fill(0.1 + 0.05 * cos(k[1]) * cos(k[2]), 4))) for k in ks]
        kw          =   (primitives = uc.primitives, subs = 1, localDim = 4)
        from_lookup =   find_instability(chis, ks; kw..., lookup = lookup)
        from_matrix =   find_instability(chis, ks, interaction(1.0, ks; kw..., lookup = lookup); primitives = uc.primitives)
        @test from_lookup["critical strength"] == from_matrix["critical strength"]
        @test from_lookup["maximum momentum"] == from_matrix["maximum momentum"]
        ##### Stoner on the spin slots: U chi = 1 with U = 2 at the chi maximum 0.15
        @test abs(from_lookup["critical strength"] - 1 / (2.0 * 0.15)) <= resolution
        @test from_lookup["unstable sector"] == "all"
        sectored    =   find_instability(chis, ks; kw..., lookup = lookup, sectors = spin_sectors("SU2", 1))
        @test sectored["critical strength"] == from_lookup["critical strength"]
        @test sectored["unstable sector"] == "spin"
    end

    ##### Solving one spin sector at a time must change nothing but the cost.
    @testset "spin sectors" begin
        Random.seed!(20261101)
        d, nq   =   2, 12
        ks      =   [[0.1 * i, 0.0] for i in 1:nq]
        function embed(blocks, sectors)
            M = zeros(ComplexF64, 4d, 4d)
            for (B, (_, idx)) in zip(blocks, sectors)
                M[idx, idx] = B
            end
            return M
        end

        ##### U1z: independent longitudinal (N, Z) and transverse (X, Y) blocks
        sectors =   spin_sectors("U1z", d)
        chis    =   [embed([random_psd(2d, 0.2), random_psd(2d, 0.2)], sectors) for _ in 1:nq]
        Vs      =   [embed([random_hermitian(2d), random_hermitian(2d)], sectors) for _ in 1:nq]
        whole   =   find_instability(chis, ks, Vs)
        blocked =   find_instability(chis, ks, Vs; sectors = sectors)
        @test abs(whole["critical strength"] - blocked["critical strength"]) <= resolution
        λ(idx)  =   maximum(maximum(real, eigvals(c[idx, idx] * V[idx, idx])) for (c, V) in zip(chis, Vs))
        @test blocked["unstable sector"] == (λ(sectors[1][2]) > λ(sectors[2][2]) ? "longitudinal" : "transverse")
        @test maximum(maximum(abs, a - b) for (a, b) in zip(effective_interaction(chis, Vs),
                                                             effective_interaction(chis, Vs; sectors = sectors))) < 1e-10
        for (e1, e2) in zip(perform_RPA(chis, Vs), perform_RPA(chis, Vs; sectors = sectors))
            @test maximum(abs.(e1[1] .- e2[1])) < 1e-10
        end

        ##### SU2: chi the same in every channel and a spin-rotation-invariant V -- the three spin
        ##### sectors are one computation, reported as "spin"
        su2     =   spin_sectors("SU2", d)
        chis    =   [kron(random_psd(d, 0.2), Matrix{ComplexF64}(I, 4, 4)) for _ in 1:nq]
        Vs      =   [kron(random_hermitian(d), Matrix{ComplexF64}(HUBBARD_CHANNELS)) for _ in 1:nq]
        plan    =   sector_plan(chis, Vs, su2)
        @test [p.rep for p in plan] == [0, 0, 2, 2]
        blocked =   find_instability(chis, ks, Vs; sectors = su2)
        @test abs(find_instability(chis, ks, Vs)["critical strength"] - blocked["critical strength"]) <= resolution
        @test blocked["unstable sector"] in ("charge", "spin")

        ##### an interaction that breaks the sectors falls back to the whole matrix, never drops blocks
        Vmix    =   [V .+ 0.1 .* random_hermitian(4d) for V in Vs]
        @test [p.name for p in sector_plan(chis, Vmix, su2)] == ["all"]
        @test abs(find_instability(chis, ks, Vmix)["critical strength"] -
                  find_instability(chis, ks, Vmix; sectors = su2)["critical strength"]) <= resolution
    end
end
