# The Julia closure must agree with the Python reference in test/triqs_reference/bilinear_basis.py.
# These are the same canonical dimensions test_bilinear_basis.py asserts, so the
# two implementations cannot drift apart without one suite failing.
@testset "bilinear closure (Julia port)" begin
    rot2(d) = (t = deg2rad(d); [cos(t) -sin(t); sin(t) cos(t)])
    c6v = vcat([[rot2(60k), rot2(60k) * Diagonal([1.0, -1.0])] for k in 0:5]...)
    swap = ComplexF64[0 1; 1 0]

    @testset "dimensions match the python implementation" begin
        @test length(bilinear_closure([Matrix{ComplexF64}(I, 1, 1)], 1)) == 1
        @test length(bilinear_closure(Matrix{ComplexF64}[I(2), swap], 2)) == 2
        @test length(bilinear_closure(Matrix{ComplexF64}.(c6v), 2)) == 3
        blocks = Matrix{ComplexF64}[]
        for R in c6v
            push!(blocks, kron(Matrix(1.0I, 2, 2), R))
            push!(blocks, kron(swap, R))
        end
        @test length(bilinear_closure(blocks, 4)) == 6
    end

    @testset "basis is orthonormal and Hermitian" begin
        b = bilinear_closure(Matrix{ComplexF64}.(c6v), 2)
        gram = [real(tr(A' * B)) for A in b, B in b]
        @test maximum(abs.(gram - I)) < 1e-9
        @test all(maximum(abs.(M - M')) < 1e-12 for M in b)
    end

    @testset "point group from the metric" begin
        @test length(lattice_point_group_2d([1.0, 0.0], [-0.5, sqrt(3) / 2])) == 12
        @test length(lattice_point_group_2d([1.0, 0.0], [0.0, 1.0])) == 8
    end

    # D(g) on a model whose answer is known: p_x, p_y transform like (x, y), so
    # D must come back as the coordinate rotation and must NOT be a permutation.
    @testset "D(g) measured from bonds" begin
        a1, a2 = [1.0, 0.0], [-0.5, sqrt(3) / 2]
        bonds, Fs = Vector{Float64}[], Matrix{ComplexF64}[]
        for (n1, n2) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
            d = n1 .* a1 .+ n2 .* a2
            dh = d ./ norm(d)
            push!(Fs, ComplexF64.(-0.35 * (dh * dh') + 0.12 * (I(2) - dh * dh')))
            push!(bonds, d)
        end
        ops = lattice_point_group_2d(a1, a2)
        worst = maximum(orbital_rep_from_bonds(bonds, Fs, g)[3] for g in ops)
        @test worst < 1e-10
        D, cond, _ = orbital_rep_from_bonds(bonds, Fs, rot2(120))
        @test cond < 1e-8
        @test max(abs(D[1, 2]), abs(D[2, 1])) > 0.1     # a rotation, not a permutation
        # and the closure from the MEASURED reps is still 3
        Ds = [orbital_rep_from_bonds(bonds, Fs, g)[1] for g in ops]
        @test length(bilinear_closure(Ds, 2)) == 3
    end

    # The seed the interaction forces. Densities alone span enough for chi to be
    # covariant but not enough to HOLD the Kanamori vertex: U' and J' couple
    # through on-site off-diagonal bilinears.
    @testset "Kanamori seed" begin
        blocks = Matrix{ComplexF64}[]
        for R in c6v
            push!(blocks, kron(Matrix(1.0I, 2, 2), R))
            push!(blocks, kron(swap, R))
        end
        b8 = bilinear_closure(blocks, 4; extra_seed = same_site_seed([1, 1, 2, 2]))
        @test length(b8) == 8
        @test length(bilinear_closure(Matrix{ComplexF64}.(c6v), 2;
                                      extra_seed = same_site_seed([1, 1]))) == 4
        # One orbital per site adds nothing, so every model that already worked
        # keeps exactly the basis it had.
        @test isempty(same_site_seed([1, 2, 3]))
        @test length(bilinear_closure(Matrix{ComplexF64}[I(2), swap], 2;
                                      extra_seed = same_site_seed([1, 2]))) == 2
        # THE POINT of the wider seed: the vertex is now representable.
        Us, Uc = kanamori_vertex(4, 6.0, 3.0, 1.5, 1.5; site_of = [1, 1, 2, 2])
        @test uncovered_bilinears(Us, b8) < 1e-9
        @test uncovered_bilinears(Uc, b8) < 1e-9
        @test uncovered_bilinears(Us, bilinear_closure(blocks, 4)) > 0.5
    end

    # The two implementations must agree on the BASIS, not only on its
    # dimension: chi is contracted in Python and the vertex projected here, and
    # coefficients in two frames of one span are not comparable. An SVD frame
    # does not give that -- within a degenerate singular value the choice is
    # arbitrary -- so the closure is canonicalised.
    @testset "canonical frame" begin
        blocks = Matrix{ComplexF64}[]
        for R in c6v
            push!(blocks, kron(Matrix(1.0I, 2, 2), R))
            push!(blocks, kron(swap, R))
        end
        seed = same_site_seed([1, 1, 2, 2])
        b = bilinear_closure(blocks, 4; extra_seed = seed)
        # Reordering the operations and the seed reorders the SVD's input
        # columns, which is exactly what used to move the frame.
        b_perm = bilinear_closure(reverse(blocks), 4; extra_seed = reverse(seed))
        @test maximum(maximum(abs.(A .- B)) for (A, B) in zip(b, b_perm)) < 1e-9
        # Frame-only property: a rotated frame of the same span canonicalises
        # to the same thing.
        V = Matrix(qr(randn(16, 5)).Q)[:, 1:5]
        R = Matrix(qr(randn(5, 5)).Q)
        @test maximum(abs.(canonical_frame(V) .- canonical_frame(V * R))) < 1e-10
        @test maximum(abs.(canonical_frame(V) * transpose(canonical_frame(V)) -
                           V * transpose(V))) < 1e-10

        # Complex conjugation, which the Cooper-channel projection needs: the
        # partner electron at -k couples to conj(M). For ZrNCl_4 six members are
        # real-symmetric (C = +1) and two are the imaginary-antisymmetric
        # orbital currents (C = -1).
        C = conjugation_matrix(b)
        @test maximum(abs.(C * C - I)) < 1e-9
        @test maximum(abs.(C - transpose(C))) < 1e-9
        @test sort(round.(Int, diag(C))) == [-1, -1, 1, 1, 1, 1, 1, 1]
        @test maximum(abs.(C - Diagonal(diag(C)))) < 1e-9
        # The per-orbital densities are real, which is why the old code never
        # had to know about any of this.
        @test maximum(abs.(conjugation_matrix(density_seed(3)) - I)) < 1e-12
    end

    # Two frames of one span must be reconciled, not assumed equal.
    @testset "basis rotation between frames" begin
        b = bilinear_closure(Matrix{ComplexF64}.(c6v), 2;
                             extra_seed = same_site_seed([1, 1]))
        d = length(b)
        R = Matrix(qr(randn(d, d)).Q)
        rotated = [sum(R[p, q] * b[p] for p in 1:d) for q in 1:d]
        O = basis_rotation(b, rotated)
        @test maximum(abs.(O - R)) < 1e-10
        # Same span, different frame: a vertex carried across must be unchanged
        # as an operator, i.e. the quadratic form on matching coefficients agrees.
        Us, Uc = kanamori_vertex(2, 6.0, 3.0, 1.5, 1.5)
        M = channel_blocks(project_vertex(Us, b), project_vertex(Uc, b))
        M_rot = channel_blocks(project_vertex(Us, rotated), project_vertex(Uc, rotated))
        @test maximum(abs.(rotate_channel_blocks(M_rot, O) - M)) < 1e-9
        # Different SPAN is an error, not a rotation.
        @test_throws ErrorException basis_rotation(b, b[1:(d - 1)])
        @test_throws ErrorException basis_rotation(b, vcat(b[1:(d - 1)], [b[1]]))
    end
end
