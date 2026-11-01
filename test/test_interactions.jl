##### interaction(): the Fourier transform of a bond Lookup, and the Hubbard + Coulomb builder

@testset "interaction" begin
    uc          =   square_uc()
    kw          =   (primitives = uc.primitives, subs = 1, localDim = 4)
    onsite      =   4.0
    params, cases   =   hubbard_coulomb_interactions(square_uc(); onsite = onsite, range = 2.5)
    coulomb     =   case_lookup(params, cases["Coulomb"])
    hubbard     =   case_lookup(params, cases["Hubbard"])

    @testset "Hermitian, and linear in strength" begin
        for k in ([0.0, 0.0], [0.37, -1.21], [pi, pi / 2])
            V   =   interaction(1.0, k; kw..., lookup = coulomb)
            @test maximum(abs.(V - V')) < 1e-12
            @test interaction(2.5, k; kw..., lookup = coulomb) ≈ 2.5 .* V
        end
        ks  =   [[0.1, 0.2], [1.0, -0.5], [0.0, 3.0]]
        @test interaction(1.7, ks; kw..., lookup = coulomb) == [interaction(1.7, k; kw..., lookup = coulomb) for k in ks]
        @test interaction(1.7, ks, params; kw...) == interaction(1.7, ks; kw..., lookup = coulomb)
    end

    ##### the Hermitian conjugate is added for EVERY bond, so an on-site matrix M gives 2M
    @testset "on-site bonds count twice" begin
        p   =   Param(1.0, 2)
        AddIsotropicBonds!(p, uc, 0.0, HUBBARD_CHANNELS, "U")
        @test interaction(1.0, [0.3, 0.4], [p]; kw...) ≈ 2 .* HUBBARD_CHANNELS
    end

    ##### distances <= 2.5 on the square lattice: 1 (x4), sqrt(2) (x4), 2 (x4), sqrt(5) (x8)
    @testset "Hubbard and Coulomb at q = 0" begin
        tail    =   4 / 1 + 4 / sqrt(2) + 4 / 2 + 8 / sqrt(5)
        @test length(params) == 1 + 10        # one direction per bond
        @test interaction(1.0, [0.0, 0.0]; kw..., lookup = hubbard) ≈ Diagonal([-onsite, onsite, onsite, onsite])
        @test interaction(1.0, [0.0, 0.0]; kw..., lookup = coulomb) ≈ Diagonal([-(onsite + tail), onsite, onsite, onsite])
    end

    @testset "V(q) is the lattice sum of 1/r" begin
        q   =   [0.37, -1.21]
        Vq  =   sum(exp(-1im * dot(q, [x, y])) / hypot(x, y) for x in -3:3, y in -3:3 if 0 < hypot(x, y) <= 2.5)
        V   =   interaction(1.0, q; kw..., lookup = coulomb)
        ##### physical slots: the tail is charge only, N; the three spin slots hold the same U
        @test V[1, 1] ≈ -(onsite + Vq)
        @test V[2, 2] == V[3, 3] == V[4, 4] ≈ onsite
        @test maximum(abs.(V - Diagonal(diag(V)))) == 0
    end

    ##### two sites: the A-B tail lands off the diagonal, the A-A tail on it
    @testset "honeycomb sublattice blocks" begin
        huc         =   honeycomb_uc()
        hp, hcases  =   hubbard_coulomb_interactions(honeycomb_uc(); onsite = onsite, range = 1.01)
        V0          =   interaction(1.0, [0.0, 0.0]; primitives = huc.primitives, subs = 2, localDim = 4,
                                    lookup = case_lookup(hp, hcases["Coulomb"]))
        ##### within 1.01: three A-B neighbours at 1/sqrt(3) (the next are at 2/sqrt(3)), six A-A at 1
        @test real(V0[1, 1]) ≈ -(onsite + 6)
        @test real(V0[5, 5]) ≈ -(onsite + 6)
        @test real(V0[1, 5]) ≈ -3 * sqrt(3)
        @test real(V0[4, 4]) ≈ onsite && V0[4, 8] == 0
    end

    @testset "argument checks" begin
        @test_throws ErrorException hubbard_coulomb_interactions(square_uc(); onsite = 0.0, range = 1.0)
        @test_throws ErrorException hubbard_coulomb_interactions(square_uc(); onsite = 1.0, range = -1.0)
    end
end
