# The bubble with spin-orbit coupling: every channel pair, against a brute-force sum.
#
# Runs against the spin fixtures in test/fixtures (test/make_spin_fixtures.py, needs TRIQS once):
# single-site models whose spinful bubble was contracted onto all sixteen channel pairs with the
# functions run_bare.py uses. Here H(k) is rebuilt from the stored hoppings, in TightBindingToolkit's
# convention H(k) = sum_R e^{ik.R} t(R), and
#
#     chi_{mu nu}(q) = 2 <O_mu(-q) O_nu(q)>,   O_mu(q) = sum_r e^{-iq.r} c^dag_r (sigma^mu / 2) c_r
#                                                       = sum_k c^dag_k (sigma^mu / 2) c_{k+q},
#
# is summed directly, which is CHI_ORIENTATION (O carries the e^{-iq.r} of interaction()'s V(q)).
# Agreement in every block, the off-diagonal ones included, pins the orientation combine_chis gives
# the pairs that only exist with SOC: transposing each block in place instead of the whole matrix
# would swap chi^{XY} and chi^{YX}, which differ here.

const SPIN_FIXTURES = ["ising", "ising_zeeman", "ising_inplane", "square"]
const PAULI = [ComplexF64[1 0; 0 1], ComplexF64[0 1; 1 0], ComplexF64[0 -im; im 0], ComplexF64[1 0; 0 -1]]

function load_spin_fixture(name)
    meta = Dict{String, String}()
    for line in eachline(joinpath(FIXTURES, "spin_$name.meta"))
        k, v = split(line, "=", limit = 2)
        meta[k] = v
    end
    table, header = readdlm(joinpath(FIXTURES, "spin_$name.csv"), ',', Float64, '\n'; header = true)
    col = Dict(strip(String(h)) => i for (i, h) in enumerate(vec(header)))
    pair(label) = table[:, col["re$label"]] .+ im .* table[:, col["im$label"]]

    units = [parse.(Float64, split(u, ",")) for u in split(meta["units"], ";")]
    hoppings = Dict{Vector{Int}, Matrix{ComplexF64}}()
    for entry in split(meta["hoppings"], ";")
        R, vals = split(entry, ":")
        v = parse.(Float64, split(vals, ","))
        hoppings[parse.(Int, split(R, ","))] = permutedims(reshape(v[1:2:end] .+ im .* v[2:2:end], 2, 2))
    end
    names = "NXYZ"
    chis = Dict((a, b) => pair(names[a + 1] * names[b + 1]) for a in 0:3 for b in 0:3)
    return (symmetry = String(meta["symmetry"]), beta = parse(Float64, meta["beta"]),
            mu = parse(Float64, meta["mu"]), units = units, hoppings = hoppings,
            ks = [[table[i, col["k1"]], table[i, col["k2"]]] for i in axes(table, 1)],
            chis = chis, fact = haskey(col, "refact") ? pair("fact") : nothing)
end

hamiltonian(f, k) = sum(M .* exp(im * dot(k, R[1] .* f.units[1] .+ R[2] .* f.units[2])) for (R, M) in f.hoppings)

##### sum over m at k, n at k - q of <m k|A_mu|n k-q><n k-q|A_nu|m k> (f_m - f_n) / (E_n - E_m):
##### O_nu(q) moves an electron from k to k - q, and O_mu(-q) moves it back.
function brute_force_chi(f, q)
    fermi(E) = 1 / (exp(f.beta * (E - f.mu)) + 1)
    chi = zeros(ComplexF64, 4, 4)
    for k in f.ks
        E1, U1 = eigen(Hermitian(hamiltonian(f, k)))
        E2, U2 = eigen(Hermitian(hamiltonian(f, k .- q)))
        for m in 1:2, n in 1:2
            dE = E2[n] - E1[m]
            F = abs(dE) < 1e-10 ? f.beta * fermi(E1[m]) * (1 - fermi(E1[m])) : (fermi(E1[m]) - fermi(E2[n])) / dE
            for mu in 1:4, nu in 1:4
                chi[mu, nu] += F * (U1[:, m]' * (PAULI[mu] / 2) * U2[:, n]) * (U2[:, n]' * (PAULI[nu] / 2) * U1[:, m])
            end
        end
    end
    return 2 .* chi ./ length(f.ks)
end

function as_npz(f, pairs, symmetry)
    data = Dict{String, Any}("spin_symmetry" => [findfirst(==(symmetry), SPIN_SYMMETRIES) - 1])
    for (a, b) in pairs
        data[pair_label(a, b)] = reshape(f.chis[(a, b)], :, 1, 1)
    end
    return data
end

@testset "bare bubble with spin-orbit coupling" begin

    @testset "every channel pair matches the brute-force sum ($name)" for name in SPIN_FIXTURES
        f = load_spin_fixture(name)
        full = combine_chis(as_npz(f, channel_pairs("none"), "none"); directions = [0, 1, 2, 3])
        reduced = combine_chis(as_npz(f, channel_pairs(f.symmetry), f.symmetry); directions = [0, 1, 2, 3])
        scale = maximum(maximum(abs, c) for c in full)
        worst = maximum(maximum(abs, full[iq] - brute_force_chi(f, f.ks[iq])) for iq in eachindex(f.ks))
        @test worst < 1e-10 * scale
        ##### the pairs the symmetry drops are exactly zero, so the reduced route loses nothing
        @test reduced == full
        @test maximum(off_sector_norm(c, spin_sectors(f.symmetry, 1)) for c in full) == 0
    end

    @testset "which cross terms Ising SOC makes" begin
        nz(name) = maximum(abs, load_spin_fixture(name).chis[(0, 3)])
        xy(name) = maximum(abs, load_spin_fixture(name).chis[(1, 2)])
        scale = maximum(abs, load_spin_fixture("ising").chis[(0, 0)])
        ##### time reversal maps chi_upup(q) onto chi_dndn(q) for one band: no charge-S_z mixing,
        ##### until a Zeeman field breaks it
        @test nz("ising") < 1e-12 * scale
        @test nz("ising_zeeman") > 1e-3 * scale
        ##### the Ising splitting separates chi^{+-} from chi^{-+}
        @test xy("ising") > 1e-3 * scale
        @test xy("square") == 0
    end

    @testset "spin factorization: the spinless bubble x 1/2 is every diagonal pair" begin
        f = load_spin_fixture("square")
        for a in 0:3
            @test maximum(abs.(f.chis[(a, a)] .- f.fact)) < 1e-12 * maximum(abs, f.fact)
        end
    end

    @testset "the fixture's Ising model is models.jl's" begin
        f = load_spin_fixture("ising")
        out = parse_unitcell(ising_model()[1])
        mats = Dict(collect(Int, c) => out["hopping matrices"][i, :, :] for (i, c) in enumerate(eachcol(out["hopping offsets"])))
        zero2 = zeros(ComplexF64, 2, 2)
        @test all(get(mats, R, zero2) ≈ get(f.hoppings, R, zero2) for R in keys(mats) ∪ keys(f.hoppings))
    end
end
