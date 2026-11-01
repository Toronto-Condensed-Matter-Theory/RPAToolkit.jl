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
        ##### Stoner on the spin slot: U chi = 1 with U = 2 at the chi maximum 0.15
        @test abs(from_lookup["critical strength"] - 1 / (2.0 * 0.15)) <= resolution
    end
end
