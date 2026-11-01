# The interaction vertex on the bilinear basis.
@testset "Kanamori vertex" begin
    n = 2
    U, Up, J = 6.0, 3.0, 1.5        # U' = U - 2J, as the ZrNCl config sets
    Jp = J                           # Kanamori-consistent
    Us, Uc = kanamori_vertex(n, U, Up, J, Jp)
    diagb = [ComplexF64[i == p && j == p for i in 1:n, j in 1:n] for p in 1:n]

    # REGRESSION: in the diagonal limit the projection must reproduce the
    # per-orbital matrices the existing builder uses, which is also what pins the
    # index pairing -- the other pairing would return the abba component here.
    @testset "reproduces the per-orbital matrices exactly" begin
        Vs, Vc = project_vertex(Us, diagb), project_vertex(Uc, diagb)
        @test real(Vs[1, 1]) ≈ U            # U^(s)_aaaa
        @test real(Vs[1, 2]) ≈ J            # U^(s)_aabb
        @test real(Vc[1, 1]) ≈ U            # U^(c)_aaaa
        @test real(Vc[1, 2]) ≈ 2Up - J      # U^(c)_aabb
        blocks = channel_blocks(Vs, Vc)
        # HUBBARD and CHARGE/HUND as ZrNCl_4_interactions_rusty.jl defines them, in the
        # physical slots: charge (negated) in N, spin in X, Y and Z.
        onsite_ref = (U / 2) .* Matrix(Diagonal([-1.0, 1, 1, 1]))
        interorb_ref = Up .* Matrix(Diagonal([-2.0, 0, 0, 0])) .+
                       J .* Matrix(Diagonal([1.0, 1, 1, 1]))
        @test onsite_ref == (U / 2) .* HUBBARD_CHANNELS
        @test maximum(abs.(blocks[1:4, 1:4] ./ 2 .- onsite_ref)) < 1e-12
        @test maximum(abs.(blocks[1:4, 5:8] .- interorb_ref)) < 1e-12
        # The three spin slots carry the same vertex, and no channel mixes with another.
        spin(c) = blocks[4 .* (0:1) .+ c, 4 .* (0:1) .+ c]
        @test spin(2) == spin(3) == spin(4)
        @test off_sector_norm(blocks, spin_sectors("SU2", 2)) == 0
    end

    @testset "off-diagonal components need off-diagonal bilinears" begin
        offd(a, b, kind) = begin
            E = zeros(ComplexF64, n, n)
            kind === :re ? (E[a, b] = 1; E[b, a] = 1) : (E[a, b] = im; E[b, a] = -im)
            E / sqrt(2)
        end
        re12, im12 = offd(1, 2, :re), offd(1, 2, :im)
        # The per-orbital basis cannot hold the full vertex: abab (U') and abba
        # (J') couple through off-diagonal bilinears.
        @test uncovered_bilinears(Us, diagb) > 0.5
        # Both Hermitian components are needed, not just the Hermitian part: pair
        # hopping couples the same non-Hermitian bilinear twice.
        @test uncovered_bilinears(Us, vcat(diagb, [re12])) > 0.5
        @test uncovered_bilinears(Us, vcat(diagb, [re12, im12])) < 1e-12
        @test uncovered_bilinears(Uc, vcat(diagb, [re12, im12])) < 1e-12
        # Monotonic in the basis, which the earlier reconstruct-the-array metric
        # was not.
        dims = [uncovered_bilinears(Us, b) for b in
                (diagb, vcat(diagb, [re12]), vcat(diagb, [re12, im12]))]
        @test issorted(dims, rev = true)
    end

    # U', J and J' are INTRA-ATOMIC. ZrNCl_4 has two Zr with two orbitals each,
    # so without the site mask the vertex would put a U' between orbitals on
    # DIFFERENT Zr, which the per-orbital builder correctly excludes via
    # sublattice(a) == sublattice(b).
    @testset "U', J, J' are intra-atomic" begin
        site = [1, 1, 2, 2]
        Us4, Uc4 = kanamori_vertex(4, U, Up, J, Jp; site_of = site)
        db4 = [ComplexF64[i == p && j == p for i in 1:4, j in 1:4] for p in 1:4]
        Vs4, Vc4 = project_vertex(Us4, db4), project_vertex(Uc4, db4)
        for a in 1:4, b in 1:4
            if a == b
                @test real(Vs4[a, b]) ≈ U
            elseif site[a] == site[b]
                @test real(Vs4[a, b]) ≈ J
                @test real(Vc4[a, b]) ≈ 2Up - J
            else
                @test real(Vs4[a, b]) == 0      # no inter-atomic U'/J
                @test real(Vc4[a, b]) == 0
            end
        end
        # Without the mask the inter-atomic entry is spuriously J.
        Us_nomask, _ = kanamori_vertex(4, U, Up, J, Jp)
        @test real(project_vertex(Us_nomask, db4)[1, 3]) ≈ J
    end

    @testset "single orbital has no off-diagonal content" begin
        Us1, Uc1 = kanamori_vertex(1, U, Up, J, Jp)
        b1 = [ones(ComplexF64, 1, 1)]
        @test uncovered_bilinears(Us1, b1) < 1e-12
        @test uncovered_bilinears(Uc1, b1) < 1e-12
        @test real(project_vertex(Us1, b1)[1, 1]) ≈ U
    end
end
