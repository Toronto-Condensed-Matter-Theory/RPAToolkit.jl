module SpinSectors

using LinearAlgebra, TightBindingToolkit
using ..RPAToolkit.parse_model: parse_unitcell

export SPIN_SYMMETRIES, VERTEX_SLOTS, channel_pairs, pair_label, spin_symmetry,
       stored_spin_symmetry, spin_sectors, off_sector_norm

"""
    SPIN_SYMMETRIES

The spin symmetries a model can have, from most to least symmetric: `"SU2"` (no spin-orbit
coupling, `H = h ⊗ 1`), `"U1z"` (``S_z`` conserved: no spin-flip hopping, but up and down differ,
as with Ising spin-orbit coupling) and `"none"`. The bare stage stores the 0-based index of the
one it used as `spin_symmetry` in each `.npz`.
"""
const SPIN_SYMMETRIES = ("SU2", "U1z", "none")

const CHANNEL_NAMES = ('N', 'X', 'Y', 'Z')

"""
    VERTEX_SLOTS

The channel layout of every interaction matrix: slot N holds minus the charge vertex,
``-U^{(c)}``, and slots X, Y and Z each hold the spin vertex ``U^{(s)}``. These are the physical
channels, so the layout is valid with spin-orbit coupling, when ``\\chi`` mixes them.
Interaction files and `run_rpa` output record this string under `"vertex_slots"`, and `run_rpa`
refuses an interaction file without it: before 2026-11-01 slot N held ``U^{(s)}`` and slot Z
held ``-U^{(c)}``, which is only equivalent without spin-orbit coupling.
"""
const VERTEX_SLOTS = "physical: N = -U^(c), X = Y = Z = U^(s)"

const CHANNEL_PAIRS = Dict(
    "SU2"  => [(0, 0), (1, 1), (2, 2), (3, 3)],
    "U1z"  => [(0, 0), (0, 3), (3, 0), (3, 3), (1, 1), (1, 2), (2, 1), (2, 2)],
    "none" => [(a, b) for a in 0:3 for b in 0:3],
)

"""
    channel_pairs(symmetry) -> Vector{Tuple{Int, Int}}

The channel pairs `(a, b)` (directions, 0 = N to 3 = Z) for which ``\\chi^{ab}`` can be nonzero
under `symmetry`: the four diagonal ones for `"SU2"`; NN, NZ, ZN, ZZ and XX, XY, YX, YY for
`"U1z"`, where ``S_z`` conservation only connects operators of equal ``S_z`` charge; all sixteen
for `"none"`.
"""
channel_pairs(symmetry::AbstractString) = CHANNEL_PAIRS[symmetry]

"""
    pair_label(a, b) -> String

The key of the `(a, b)` channel pair in a bare-stage `.npz`: `"chi_NN"`, `"chi_NZ"`, `"chi_XY"`, ...
"""
pair_label(a::Integer, b::Integer) = "chi_" * CHANNEL_NAMES[a + 1] * CHANNEL_NAMES[b + 1]

"""
    spin_symmetry(uc::UnitCell) -> String

The spin symmetry of the hopping model, by the same exact tests the bare stage applies to the
same serialized hopping matrices ([`parse_unitcell`](@ref)): `"SU2"` when every spin block is the
identity times one number, `"U1z"` when no hopping flips spin, else `"none"`.
"""
function spin_symmetry(uc::UnitCell)::String
    uc.localDim == 2 || error("spin_symmetry: localDim = $(uc.localDim); the spin index needs 2.")
    H = parse_unitcell(uc)["hopping matrices"]
    up, dn = 1:2:size(H, 2), 2:2:size(H, 2)
    (any(!iszero, H[:, up, dn]) || any(!iszero, H[:, dn, up])) && return "none"
    return H[:, up, up] == H[:, dn, dn] ? "SU2" : "U1z"
end

"""
    stored_spin_symmetry(data; model_symmetry) -> String

The spin symmetry a bare-stage `.npz` was contracted under, i.e. which channel pairs it holds.

Files written before 2026-11-01 do not record it. Those with `bubble_spin_factorized = 1` are
`"SU2"`. The rest hold the diagonal pairs of a spinful bubble, which is the whole of ``\\chi``
only when the model is `"SU2"`: accepted when `model_symmetry` says so, refused otherwise.
"""
function stored_spin_symmetry(data::Dict; model_symmetry::Union{Nothing, AbstractString} = nothing)::String
    if haskey(data, "spin_symmetry")
        return SPIN_SYMMETRIES[Int(only(data["spin_symmetry"])) + 1]
    end
    if Int(only(get(data, "bubble_spin_factorized", [0]))) == 1 || model_symmetry == "SU2"
        return "SU2"
    end
    error("This bubble predates spin-orbit coupling support: it holds only the four diagonal " *
          "channel pairs, which is all of chi only without spin-orbit coupling, and the model " *
          "here is $(something(model_symmetry, "of unknown spin symmetry")). Rerun the bare stage.")
end

"""
    spin_sectors(symmetry, d) -> Vector{Pair{String, Vector{Int}}}

The channel sectors a `4d × 4d` matrix (channel innermost, entry `(p, mu)` at `4 (p - 1) + mu`)
is block diagonal in when the model has `symmetry`, as `name => indices`:

- `"SU2"`: `"charge"` (N), `"spin z"`, `"spin x"`, `"spin y"`.
- `"U1z"`: `"longitudinal"` (N and Z, which ``S_z`` conservation leaves coupled) and
  `"transverse"` (X and Y).
- `"none"`: `"all"`.
"""
function spin_sectors(symmetry::AbstractString, d::Integer)::Vector{Pair{String, Vector{Int}}}
    idx(channels) = [4 * (p - 1) + c for p in 1:d for c in channels]
    symmetry == "SU2" &&
        return ["charge" => idx(1), "spin z" => idx(4), "spin x" => idx(2), "spin y" => idx(3)]
    symmetry == "U1z" && return ["longitudinal" => idx((1, 4)), "transverse" => idx((2, 3))]
    symmetry == "none" && return ["all" => collect(1:4d)]
    error("spin_sectors: unknown spin symmetry $(repr(symmetry)); expected one of $(SPIN_SYMMETRIES).")
end

"""
    off_sector_norm(M, sectors) -> Float64

The largest `|M[i, j]|` with `i` and `j` in different `sectors`: zero exactly when `M` is block
diagonal in them.
"""
function off_sector_norm(M::AbstractMatrix, sectors)::Float64
    owner = zeros(Int, size(M, 1))
    for (s, (_, idx)) in enumerate(sectors)
        owner[idx] .= s
    end
    worst = 0.0
    for j in axes(M, 2), i in axes(M, 1)
        owner[i] == owner[j] || (worst = max(worst, abs(M[i, j])))
    end
    return worst
end

end
