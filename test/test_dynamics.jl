##### The real-frequency RPA helpers (src/Dynamics.jl) and the TDHF vertex builders (src/Interactions.jl)

@testset "dynamics" begin

    @testset "TDHF vertices" begin
        ##### U n_up n_down, as MeanFieldToolkit's DensityToPartonCoupling writes it: U_abcd = Ui_ab Uj_cd
        nup, ndn = [1.0 0.0; 0.0 0.0], [0.0 0.0; 0.0 1.0]
        hubbard = [nup[a, b] * ndn[c, d] for a in 1:2, b in 1:2, c in 1:2, d in 1:2]
        @test quartic_onsite_vertex(3.0 .* hubbard) ≈ 3.0 .* HUBBARD_CHANNELS atol = 1e-14
        ##### V n_i n_j: Hartree only, charge slot, -2V (chi is the per-spin bubble)
        dd = [Float64(a == b) * Float64(c == d) for a in 1:2, b in 1:2, c in 1:2, d in 1:2]
        vij, vji = quartic_hartree_vertex(0.7 .* dd)
        @test vij ≈ vji && vij[1, 1] ≈ -1.4 && count(!iszero, round.(vij, digits = 14)) == 1
        ##### J S_i.S_j: -J/2 in each spin slot
        S = [ComplexF64[0 1; 1 0] / 2, ComplexF64[0 -im; im 0] / 2, ComplexF64[1 0; 0 -1] / 2]
        heis = [sum(S[i][a, b] * S[i][c, d] for i in 1:3) for a in 1:2, b in 1:2, c in 1:2, d in 1:2]
        hij, _ = quartic_hartree_vertex(heis)
        @test diag(hij) ≈ [0.0, -0.5, -0.5, -0.5] atol = 1e-14
    end

    @testset "Ewald-summed Coulomb" begin
        prims = [[1.0, 0.0], [-0.5, sqrt(3) / 2]]
        pos = [[0.0, 0.0], [0.0, 1 / sqrt(3)]]
        area = sqrt(3) / 2
        for q in ([0.3, 0.1], [1.7, -0.4])
            Vs = [coulomb_2d(prims, pos; alpha = a)(q) for a in (0.8, 1.5, 3.0)]
            @test maximum(maximum(abs, V - Vs[1]) for V in Vs) < 1e-10           # independent of the Ewald split
            @test Vs[1] ≈ Vs[1]'                                                  # Hermitian
            @test count(!iszero, Vs[1]) == 4                                      # charge slots only
        end
        q = [1e-3, 0.0]
        v = -coulomb_2d(prims, pos)(q)[1, 1] / 2
        @test abs(v * norm(q) - 2pi / area) / (2pi / area) < 1e-2               # 2 pi / (A |q|) as q -> 0
        @test coulomb_2d(prims, pos; onsite = 2.0)(q)[1, 1] ≈ coulomb_2d(prims, pos)(q)[1, 1] - 2.0
    end

    @testset "supercell unfolding" begin
        S = SpinMats(1 // 2)
        one = UnitCell([[1.0, 0.0], [0.0, 1.0]], 2); AddBasisSite!(one, [0.0, 0.0])
        t1 = Param(-1.0, 2); AddIsotropicBonds!(t1, one, 1.0, S[4], "t"); CreateUnitCell!(one, [t1])
        two = UnitCell([[1.0, 1.0], [1.0, -1.0]], 2); AddBasisSite!(two, [0.0, 0.0]); AddBasisSite!(two, [1.0, 0.0])
        t2 = Param(-1.0, 2); AddIsotropicBonds!(t2, two, 1.0, S[4], "t"); CreateUnitCell!(two, [t2])
        ##### the two cells sample the Brillouin zone differently, so compare on converged grids with a wide eta
        qs = [[0.4, 0.13], [1.2, -0.7]]
        ws = [0.0, 0.7]
        c1 = dynamic_bubble(band_model(one), qs, ws, 64; beta = 10.0, mu = -0.3, eta = 0.3, nbins = nothing)
        c2 = dynamic_bubble(band_model(two), qs, ws, 46; beta = 10.0, mu = -0.3, eta = 0.3, nbins = nothing)
        for iq in eachindex(qs), iw in eachindex(ws)
            U = unfold_vector(two.basis, qs[iq], (0,))
            @test abs(dot(U[:, 1], c2[iq, iw] * U[:, 1]) - c1[iq, iw][1, 1]) < 2e-3 * abs(c1[iq, iw][1, 1])
        end
    end

    @testset "particle-hole edge and pole read-out" begin
        S = SpinMats(1 // 2)
        uc = UnitCell([[1.0, 0.0], [0.0, 1.0]], 2); AddBasisSite!(uc, [0.0, 0.0])
        t1 = Param(-1.0, 2); AddIsotropicBonds!(t1, uc, 1.0, S[4], "t"); CreateUnitCell!(uc, [t1])
        bm = band_model(uc)
        ##### a nearly empty band: E = -4 + k^2, so the edge is v_F q + q^2 with v_F = 2 k_F
        mu = -4 + 0.3^2
        qs = [[q, 0.0] for q in (0.02, 0.05)]
        edges = particle_hole_edges(bm, qs, 400; beta = 2000.0, mu = mu)
        @test all(abs.(edges .- [2 * 0.3 * q[1] + q[1]^2 for q in qs]) .< 0.15 .* edges)
        ##### the pole read-out finds a Lorentzian's centre between grid points
        ws = collect(range(0.0, 2.0, length = 201))
        spectral = reduce(vcat, [permutedims(0.05 ./ ((ws .- c) .^ 2 .+ 0.05^2)) for c in (0.503, 1.2071)])
        @test rpa_poles(spectral, ws) ≈ [0.503, 1.2071] atol = 2e-3
        @test isnan(only(rpa_poles(ones(1, 201), ws)))                            # no peak, no pole
    end

    @testset "dynamic_rpa is perform_RPA at every (q, w)" begin
        chi0 = [ComplexF64[0.2 + 0.01im 0.05; 0.05 0.3 + 0.02im] for _ in 1:2, _ in 1:3]
        V = [ComplexF64[1.0 0.2; 0.2 0.5], ComplexF64[0.4 0.0; 0.0 0.9]]
        out = dynamic_rpa(chi0, V)
        @test size(out) == size(chi0)
        @test out[2, 3] ≈ (I - chi0[2, 3] * V[2]) \ chi0[2, 3]
        @test dynamic_rpa(chi0, V[1])[1, 1] ≈ out[1, 1]
    end
end
