##### The MeanFieldToolkit extension: Hartree-Fock bands + the TDHF vertex of the same interaction.
#####
#####     julia --project=examples/MeanField test/test_meanfield_ext.jl
#####
##### (also run by runtests.jl when MeanFieldToolkit is in the environment)

if !isdefined(Main, :neel_hubbard)
    using Test
    include(joinpath(@__DIR__, "..", "examples", "MeanField", "common.jl"))
end

@testset "MeanFieldToolkit extension" begin

    @testset "the Néel state has an exact Goldstone mode (U = $U)" for U in (4.0, 8.0)
        mft = neel_hubbard(; U, kSize = 24)
        residual, _ = solve!(mft, [0.5, 0, 0, 0.9, 0.5, 0, 0, -0.9])
        @test residual < 1e-9
        @test abs(real(order_parameters(mft)["site 1 : Sz"])) > 0.5                      # ordered
        r = rpa_model(mft; residual = residual)
        V = rpa_vertex(mft)
        chi = static_chi(r.bands, [[0.0, 0.0]], 24; beta = r.beta, mu = r.mu)[1]
        leading = sort(real(eigvals(chi * V([0.0, 0.0]))), rev = true)
        @test leading[1] ≈ 1 atol = 1e-6          # the two transverse rotations of the ordered moment
        @test leading[2] ≈ 1 atol = 1e-6
        @test leading[3] < 1 - 1e-2               # nothing else is critical
    end

    @testset "a paramagnetic state is the free bubble at the Hartree-Fock chemical potential" begin
        ##### away from half filling (at half filling the square lattice is nested and orders at any U)
        filling = 0.35
        mft = neel_hubbard(; U = 1.0, filling, kSize = 24)
        residual, _ = solve!(mft, [filling, 0, 0, 0.1, filling, 0, 0, -0.1])
        r = rpa_model(mft; residual = residual)
        @test abs(order_parameters(mft)["site 1 : Sz"]) < 1e-6
        ##### the bands are the free ones shifted by the uniform Hartree term U <n_opposite spin> = U filling
        free = band_model(let uc = UnitCell([[1.0, 1.0], [1.0, -1.0]], 2, 2)
            AddBasisSite!(uc, [0.0, 0.0]); AddBasisSite!(uc, [1.0, 0.0])
            p = Param(-1.0, 2); AddIsotropicBonds!(p, uc, 1.0, SPIN[4], "t"); CreateUnitCell!(uc, [p]); uc
        end)
        shift = 1.0 * filling
        qs = [[0.3, 0.1], [1.0, 2.0]]
        a = static_chi(r.bands, qs, 24; beta = r.beta, mu = r.mu)
        b = static_chi(free, qs, 24; beta = r.beta, mu = r.mu - shift)
        @test maximum(maximum(abs, x - y) for (x, y) in zip(a, b)) < 1e-8
        ##### and the vertex is the plain Hubbard one on every site
        @test rpa_vertex(mft)([0.2, 0.0]) ≈ kron(I(2), 1.0 .* HUBBARD_CHANNELS)
    end
end
