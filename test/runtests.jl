# Symmetry invariants of the bare bubble and its per-orbital contraction.
#
#     julia --project=. test/runtests.jl
#
# Runs against the small stored fixtures in test/fixtures. Regenerate them with
# test/make_fixtures.py (needs TRIQS); the tests themselves need neither TRIQS
# nor any stored run data.
#
# The fixtures span the cases that behave differently under the point group:
#
#     square, triangular  single orbital        per-orbital basis is closed
#     honeycomb           2 sites, D PERMUTES   closed
#     pdoublet            E doublet, D ROTATES  NOT closed
#
# The last case is the one that matters: chi_contraction projects the rank-4
# Lindhard tensor onto per-orbital densities and spins, and that operator set is
# closed under the point group only when D(g) permutes orbitals. When D(g)
# rotates them -- an E doublet, as in ZrNCl's two orbitals per Zr -- n_i maps
# into off-diagonal bilinears the basis cannot hold, so chi_ij is not covariant
# however exact the bubble is.

using Test
using LinearAlgebra
using DelimitedFiles

include(joinpath(@__DIR__, "..", "src", "BilinearVertex.jl"))
include(joinpath(@__DIR__, "..", "src", "BilinearBasis.jl"))
using .BilinearVertex
using .BilinearBasis

const FIXTURES = joinpath(@__DIR__, "fixtures")
const TOL = 1e-12
const CASES = ["square", "triangular", "honeycomb", "pdoublet"]

"""Sorted-spectrum residual: invariant under the unknown orbital rotation D."""
function spectrum_residual(chi::Array{ComplexF64,3}, perm::Vector{Int})
    ord(x) = (real(x), imag(x))
    worst, scale = 0.0, 0.0
    for iq in axes(chi, 1)
        e1 = sort(eigvals(chi[iq, :, :]), by = ord)
        e2 = sort(eigvals(chi[perm[iq], :, :]), by = ord)
        worst = max(worst, maximum(abs.(e1 .- e2)))
        scale = max(scale, maximum(abs.(e1)))
    end
    return scale > 0 ? worst / scale : 0.0
end

scalar_residual(f, perm) = maximum(abs.(f[perm] .- f)) / maximum(abs.(f))

function read_meta(name)
    meta = Dict{String,String}()
    for line in eachline(joinpath(FIXTURES, "$name.meta"))
        k, v = split(line, "=", limit = 2)
        meta[k] = v
    end
    return meta
end

function load_fixture(name)
    table, header = readdlm(joinpath(FIXTURES, "$name.csv"), ',', Float64, '\n';
                            header = true)
    col = Dict(strip(String(h)) => i for (i, h) in enumerate(vec(header)))
    meta = read_meta(name)
    ns = parse(Int, meta["n_sites"])
    n_orb = parse(Int, meta["n_orb"])
    n_q = size(table, 1)

    function chi(label)
        out = Array{ComplexF64,3}(undef, n_q, ns, ns)
        for i in 1:ns, j in 1:ns
            out[:, i, j] = table[:, col["re$(label)_$(i)$(j)"]] .+
                           im .* table[:, col["im$(label)_$(i)$(j)"]]
        end
        return out
    end

    bands = hcat((table[:, col["band$b"]] for b in 1:n_orb)...)
    nb = parse(Int, meta["n_basis"])
    function chiC(label)
        out = Array{ComplexF64,3}(undef, n_q, nb, nb)
        for i in 1:nb, j in 1:nb
            out[:, i, j] = table[:, col["reC$(label)_$(i)$(j)"]] .+
                           im .* table[:, col["imC$(label)_$(i)$(j)"]]
        end
        return out
    end
    return (chi_NN = chi("NN"), chi_ZZ = chi("ZZ"),
            chiC_NN = chiC("NN"), chiC_ZZ = chiC("ZZ"), n_basis = nb,
            perm = Int.(table[:, col["perm3"]]) .+ 1,   # python 0-based -> julia
            bands = bands, n_q = n_q,
            tau_equal = get(meta, "tau_equal", "1") == "1")
end

# Models whose orbital representation D(g) PERMUTES orbitals (or which have a
# single orbital): the per-orbital operator set is closed, so chi_ij transforms
# by a unitary and both its trace and its spectrum are invariant.
const CLOSED = ["square", "triangular", "honeycomb"]
# D(g) ROTATES the orbitals -- an E doublet, as in ZrNCl's two orbitals per Zr.
# n_i then maps into off-diagonal bilinears the basis cannot hold.
const NOT_CLOSED = ["pdoublet"]

@testset "bare bubble symmetry" begin

    @testset "fixtures come from symmetric models" begin
        for name in CASES
            f = load_fixture(name)
            @test maximum(abs.(f.bands[f.perm, :] .- f.bands)) < TOL
        end
    end

    # THE BUBBLE IS EXACT. sum_ij chi_ij is the response of the TOTAL density,
    # a point-group scalar -- but only in a gauge with no exp(i k.tau) phases,
    # i.e. when every orbital sits at the same position. Where positions differ
    # (honeycomb) the operation acts as a permutation TIMES a diagonal phase and
    # the raw sum is gauge-dependent, so the check does not apply there.
    @testset "total density response is invariant" begin
        for name in CASES
            f = load_fixture(name)
            f.tau_equal || continue
            for chi in (f.chi_NN, f.chi_ZZ)
                tot = ComplexF64[sum(chi[iq, :, :]) for iq in 1:f.n_q]
                @test scalar_residual(tot, f.perm) < TOL
            end
        end
    end

    # Trace and spectrum are invariant under ANY unitary conjugation, so they
    # hold exactly when the per-orbital basis is closed -- independent of gauge.
    @testset "per-orbital chi_ij covariance (closed cases)" begin
        for name in CLOSED
            f = load_fixture(name)
            ns = size(f.chi_NN, 2)
            for chi in (f.chi_NN, f.chi_ZZ)
                tr = ComplexF64[sum(chi[iq, i, i] for i in 1:ns) for iq in 1:f.n_q]
                @test scalar_residual(tr, f.perm) < TOL
                @test spectrum_residual(chi, f.perm) < TOL
            end
        end
    end

    # Marked broken so the suite reports an Unexpected Pass -- i.e. tells us --
    # the moment a symmetry-closed bilinear basis replaces this projection.
    @testset "E doublet is not covariant in the per-orbital basis" begin
        for name in NOT_CLOSED
            f = load_fixture(name)
            ns = size(f.chi_NN, 2)
            tr = ComplexF64[sum(f.chi_NN[iq, i, i] for i in 1:ns) for iq in 1:f.n_q]
            @test_broken scalar_residual(tr, f.perm) < TOL
            @test_broken spectrum_residual(f.chi_NN, f.perm) < TOL
            # Pin the order of magnitude so a partial change cannot slip through.
            @test spectrum_residual(f.chi_NN, f.perm) > 1e-2
            # The bubble underneath is still exact.
            tot = ComplexF64[sum(f.chi_NN[iq, :, :]) for iq in 1:f.n_q]
            @test scalar_residual(tot, f.perm) < TOL
        end
    end

    # THE FIX. Contracting onto the symmetry-CLOSED bilinear basis -- the span of
    # the densities closed under the measured D(g) -- makes chi covariant for
    # every model, the E doublet included. The basis transforms by a real
    # orthogonal matrix, so trace and spectrum are both invariant.
    @testset "closed basis restores covariance" begin
        for name in CASES
            f = load_fixture(name)
            for chi in (f.chiC_NN, f.chiC_ZZ)
                tr = ComplexF64[sum(chi[iq, i, i] for i in 1:f.n_basis) for iq in 1:f.n_q]
                @test scalar_residual(tr, f.perm) < TOL
                @test spectrum_residual(chi, f.perm) < TOL
            end
        end
    end

    # The basis must not grow for models that were already fine -- otherwise the
    # fix would cost every existing calculation something for nothing.
    @testset "closed basis does not grow where D(g) permutes" begin
        for name in CLOSED
            f = load_fixture(name)
            @test f.n_basis == size(f.chi_NN, 2)
        end
        # The E doublet is the one case that must grow: 2 -> 3.
        f = load_fixture("pdoublet")
        @test f.n_basis == size(f.chi_NN, 2) + 1
    end
end

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
        # HUBBARD and CHARGE/HUND as ZrNCl_4_interactions_rusty.jl defines them.
        onsite_ref = (U / 2) .* Matrix(Diagonal([1.0, 0, 0, -1.0]))
        interorb_ref = Up .* Matrix(Diagonal([0.0, 0, 0, -2.0])) .+
                       J .* Matrix(Diagonal([1.0, 0, 0, 1.0]))
        @test maximum(abs.(blocks[1:4, 1:4] ./ 2 .- onsite_ref)) < 1e-12
        @test maximum(abs.(blocks[1:4, 5:8] .- interorb_ref)) < 1e-12
        # X and Y slots stay empty, as in the per-orbital matrices.
        @test all(blocks[4 .* (0:1) .+ 2, :] .== 0)
        @test all(blocks[4 .* (0:1) .+ 3, :] .== 0)
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

# The Julia closure must agree with the Python one in src/Bare/bilinear_basis.py.
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
