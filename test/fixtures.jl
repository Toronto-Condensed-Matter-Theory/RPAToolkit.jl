##### Loading the stored chi(q) fixtures in test/fixtures, shared by the symmetry tests.

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
            perm = Int.(table[:, col["perm3"]]) .+ 1,   # stored 0-based
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
