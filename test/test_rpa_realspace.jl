##### The RPA's momentum conventions against a calculation that has none: the exact real-space Hartree
##### RPA of a finite periodic cluster.
#####
##### On an L x L cluster of a two-site honeycomb cell (no inversion through a site: a sublattice
##### potential) with a non-local density-density interaction (nearest-neighbour A-B, so V(q) != V(q)^T),
##### the static density response is computed site by site, chi0_xy, and the Hartree RPA is the matrix
##### equation chi = (1 + Pi v)^-1 Pi. Fourier transformed (the e^{iq.R} component's response to a field
##### e^{iq.R}), it must equal what run_rpa computes on the L x L grid: combine_chis's chi(q), interaction()'s
##### V(q), and (I - chi V)^-1 chi. Placing hoppings and interactions in real space by the same bond rule
##### makes the comparison independent of any Fourier convention; pairing chi(q) with V(-q) fails it by
##### ~60% of the response.

@testset "RPA against the real-space Hartree RPA" begin
    S = SpinMats(1 // 2)
    a1, a2 = [sqrt(3) / 2, 1.5], [-sqrt(3) / 2, 1.5]
    cell() = (uc = UnitCell([a1, a2], 2); AddBasisSite!(uc, [0.0, 0.0]); AddBasisSite!(uc, [0.0, 1.0]); uc)
    uc = cell()
    t1 = Param(-1.0, 2); AddIsotropicBonds!(t1, uc, 1.0, S[4], "t1")
    t2 = Param(0.15, 2); AddIsotropicBonds!(t2, uc, sqrt(3), S[4], "t2")
    D = Param(0.4, 2); AddAnisotropicBond!(D, uc, 1, 1, [0, 0], S[4], 0.0, "sublattice potential")
    CreateUnitCell!(uc, [t1, t2, D])
    bm = band_model(uc)
    Vp = Param(1.0, 2); AddIsotropicBonds!(Vp, cell(), 1.0, 0.6 .* DENSITY_CHANNELS, "V nn")
    lookup = Lookup([Vp])

    L, beta, mu = 6, 8.0, -0.3
    qs = k_grid(bm.primitives, L)

    ##### the pipeline
    stored = stored_bubble(bm, qs, L; beta, mu, ops = channel_operators(2), on_grid = true)
    P = [2 .* transpose(s) for s in stored]
    Vq = interaction(1.0, qs; primitives = bm.primitives, subs = 2, localDim = 4, lookup = lookup)
    N = [1, 5]
    @test maximum(maximum(abs, v - transpose(v)) for v in Vq) > 0.5          # the test can tell V(q) from V(-q)

    ##### the cluster, spinless (SU(2)), site x = (R, i), hoppings t_ij(R) = <0,i|H|R,j>
    idx(r1, r2, i) = 2 * (mod(r1, L) * L + mod(r2, L)) + i
    n = 2L^2
    H = zeros(ComplexF64, n, n)
    for (o, M) in zip(bm.offsets, bm.matrices), r1 in 0:L-1, r2 in 0:L-1, i in 1:2, j in 1:2
        H[idx(r1, r2, i), idx(r1 + o[1], r2 + o[2], j)] += M[2i - 1, 2j - 1]
    end
    E, Ψ = eigen(Hermitian(H))
    @test sort(E) ≈ sort(vec(band_energies(spinless_model(bm), qs)))
    chi0 = zeros(ComplexF64, n, n)
    for b in 1:n, a in 1:n
        w = lindhard_weight(E[a], E[b], beta, mu)
        w < 1e-14 && continue
        u = conj.(Ψ[:, a]) .* Ψ[:, b]
        chi0 .+= w .* (u * u')
    end
    ##### the same bonds as pair couplings (H_int = 1/2 sum v_xy n_x n_y); the slot entry -w is Graser's 2v
    v = zeros(n, n)
    for ((i, j, o), M) in lookup, r1 in 0:L-1, r2 in 0:L-1
        x, y = idx(r1, r2, i), idx(r1 + o[1], r2 + o[2], j)
        v[x, y] += -real(M[1, 1]) / 2
        v[y, x] += -real(M[1, 1]) / 2
    end
    Pi = 2 .* chi0
    chi = (I + Pi * v) \ Pi
    X(c, q) = [sum(c[idx(0, 0, i), idx(r1, r2, j)] * cis(dot(q, r1 .* a1 .+ r2 .* a2)) for r1 in 0:L-1, r2 in 0:L-1)
               for i in 1:2, j in 1:2] ./ 2                                  # per spin

    scale = maximum(maximum(abs, p[N, N]) for p in P)
    @test maximum(maximum(abs, P[iq][N, N] - X(Pi, qs[iq])) for iq in eachindex(qs)) < 1e-12 * scale
    rpa = [((I - p * V) \ p)[N, N] for (p, V) in zip(P, Vq)]
    @test maximum(maximum(abs, rpa[iq] - X(chi, qs[iq])) for iq in eachindex(qs)) < 1e-12 * scale
    wrong = [((I - p * transpose(V)) \ p)[N, N] for (p, V) in zip(P, Vq)]
    @test maximum(maximum(abs, wrong[iq] - X(chi, qs[iq])) for iq in eachindex(qs)) > 0.1 * scale

    ##### coulomb_2d is in the same convention: its A-B element against the direct lattice sum
    ##### sum_R e^{+iq.R} e^{-kappa r} / r, r = |R + b_B - b_A|, which converges to it linearly in kappa
    ##### (3.0931, 3.0904, 3.0890 at kappa = 4, 2, 1 x 10^-3 against 3.0875); the other sign flips Im.
    prims, pos = [[1.0, 0.0], [-0.5, sqrt(3) / 2]], [[0.0, 0.0], [0.0, 1 / sqrt(3)]]
    q = [1.3, 0.7]
    vAB = -coulomb_2d(prims, pos)(q)[1, 5] / 2
    kappa, M = 4e-3, 600
    direct(sign) = sum(cis(sign * dot(q, m .* prims[1] .+ l .* prims[2])) *
                       exp(-kappa * norm(m .* prims[1] .+ l .* prims[2] .+ pos[2])) / norm(m .* prims[1] .+ l .* prims[2] .+ pos[2])
                       for m in -M:M, l in -M:M)
    @test abs(vAB - direct(+1)) < 5e-3 * abs(vAB)
    @test abs(vAB - direct(-1)) > 0.2 * abs(vAB)
end
