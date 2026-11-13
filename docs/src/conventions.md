# Conventions

## Channels and matrix layout

Each orbital ``a`` carries four bilinears, one per entry of `directions`: N (`0`), X (`1`), Y
(`2`), Z (`3`), from the spin matrices ``\sigma^0, \sigma^x, \sigma^y, \sigma^z``. Every matrix
RPAToolkit works with, ``\chi(q)`` and ``V(q)`` alike, is `d * length(directions)` square, with
the channel index **innermost**: orbital ``a``, channel ``\mu`` sits at row
`length(directions) * (a - 1) + μ`. Without spin-orbit coupling ``\chi`` is block diagonal in the
channel; with it, the channels mix (see [Spin-orbit coupling and spin sectors](@ref)).

## The bare susceptibility

[`combine_chis`](@ref) assembles the stored bubbles into ``\chi_{ij}(q)``, the response of the
``e^{iq\cdot R}`` component of ``O_i`` to a field ``e^{iq\cdot R}`` on ``O_j`` (``R`` the lattice
vectors, in TightBindingToolkit's ``H(k) = \sum_R e^{ik\cdot R} t(R)``). The interaction is built in
the same Bloch convention, ``V_{ij}(q) = \sum M e^{+iq\cdot\delta}`` per bond
([`INTERACTION_CONVENTION`](@ref)), which is what makes ``(1 - \chi V)^{-1}\chi`` the RPA. Both are
checked against an exact real-space Hartree RPA on a periodic cluster.
That is a plain transpose, of the whole matrix, of what the bare stage stores (TPRF's contraction, which the
Julia bubble reproduces: [`stored_bubble`](@ref) is the transpose of the Lindhard sum with the operators
transposed), normalized as
the per-spin bubble ``\chi_0`` of Graser et al., NJP **11**, 025016 (2009) (twice what is stored). [`CHI_ORIENTATION`](@ref) names this convention, and `run_rpa` records it
in its output. The static bubble is positive semidefinite.

When the model is spin-independent the four channels of ``\chi`` are identical. The channel slots
then only acquire a meaning through the interaction matrix.

## RPA and the instability

```math
\chi_\mathrm{RPA}(q) = \left[1 - \chi(q)\, s V(q)\right]^{-1} \chi(q)
```
for an interaction ``V`` at strength ``s``. Because ``\chi \ge 0``, a **positive** eigen-direction
of ``\chi V`` is enhanced and a negative one is screened. The critical strength is the smallest
``s`` at which ``\chi_\mathrm{RPA}`` acquires a negative eigenvalue at some ``q``, i.e.
``s \lambda_\mathrm{max} = 1`` for the largest eigenvalue ``\lambda_\mathrm{max}`` of
``\chi(q) V(q)`` over all ``q``. [`find_instability`](@ref) brackets it by bisection on
`instability.lower`/`upper` (resolution ``(\text{upper} - \text{lower}) / 2^{32}``). It reports
``\chi_\mathrm{RPA}`` just below it, whose largest eigenvalue sits at the ordering wavevector
(`"maximum momentum"`). A case that never goes unstable reports `upper`.

The screened interaction is ``V + V \chi_\mathrm{RPA} V`` ([`effective_interaction`](@ref)).

## Interaction slots

With that sign convention, the slots are the physical channels ([`VERTEX_SLOTS`](@ref)):

- **slot N holds minus the charge vertex, ``-U^{(c)}``.** A repulsive charge interaction is
  negative there and screened.
- **slots X, Y and Z each hold the spin vertex ``U^{(s)}``.** A positive entry is Stoner-enhanced.

Written in the Pauli channels, the antisymmetrized Hubbard vertex is exactly
``U\,\mathrm{diag}(-1, 1, 1, 1)``, so this layout is the full spinful RPA, and stays right when
spin-orbit coupling makes ``\chi`` mix the channels.

The two building blocks:

| term | channels | constant |
|:-----|:---------|:---------|
| on-site Hubbard ``U n_\uparrow n_\downarrow`` | ``\mathrm{diag}(-1, +1, +1, +1)``: ``U^{(s)} = U^{(c)} = U`` | [`HUBBARD_CHANNELS`](@ref) |
| density-density ``V n_a n_b`` | ``\mathrm{diag}(-1, 0, 0, 0)``: charge only | [`DENSITY_CHANNELS`](@ref) |

With `DENSITY_CHANNELS`, an on-site ``U`` plus a tail ``V(q) = \sum_\delta e^{+iq\cdot\delta} V_\delta``
gives ``U^{(c)}(q) = U + V(q)``. Graser's density-density vertex is ``2V``; the multi-orbital
ZrNCl builder in the TMD project registers `diag(-2, 0, 0, 0)` for that reason. The two differ
only in what is called ``V``.

Interaction files record the layout as `"vertex_slots" => VERTEX_SLOTS`, and `run_rpa` refuses a
file without it.

Until 2026-11-12 [`interaction`](@ref) built ``V(q)`` with ``e^{-iq\cdot\delta}``, which is ``V(-q) = V(q)^T``.
That is the same thing for on-site interactions and on lattices with one site per cell, but on a
multi-site cell a non-local interaction entered the RPA transposed; the honeycomb example's ``1/r``
charge response was off by 30-40% (its critical strengths, set by the local spin vertex, were not).
`run_rpa` now records `"interaction_convention" => INTERACTION_CONVENTION` in its output. Until 2026-11-01 slot N held ``U^{(s)}`` and slot Z held ``-U^{(c)}``, with X and
Y empty: the same physics without spin-orbit coupling (critical strengths are identical, and the
screened interaction's N and Z blocks swap), but not with it.

## Spin-orbit coupling and spin sectors

How many entries of the ``4 \times 4`` channel matrix ``\chi^{\mu\nu}`` can be nonzero is set by
the spin rotations the hopping model commutes with. [`spin_symmetry`](@ref) and the bare stage
([`band_spin_symmetry`](@ref)) apply the same exact tests to the serialized hoppings:

| symmetry | the model | bubble | channel pairs | sectors |
|:---------|:----------|:-------|:--------------|:--------|
| `SU2` | ``H = h \otimes 1``, no spin-orbit coupling | spinless ``\times \tfrac12`` | NN = XX = YY = ZZ | charge (N); spin (Z, with X and Y identical copies) |
| `U1z` | no spin-flip hopping, up ``\ne`` down (Ising) | spinful | NN, NZ, ZN, ZZ; XX, XY, YX, YY | longitudinal (N, Z); transverse (X, Y) |
| `none` | anything else | spinful | all sixteen | one |

``S_z`` conservation only connects operators of equal ``S_z`` charge, which is why N and Z
(charge 0) and X and Y (charges ``\pm 1``) form separate sectors. The cross terms are real
physics: a Zeeman field gives ``\chi^{NZ}``, and Ising spin-orbit coupling separates
``\chi^{+-}`` from ``\chi^{-+}``, i.e. gives ``\chi^{XY}``. With time reversal and one band,
``\chi^{NZ}`` vanishes.

The bare stage contracts only the pairs the symmetry allows, and stores which one it used as
`spin_symmetry` in each `.npz`. A `U1z` run also contracts the pairs it skips once, at the first
chemical potential, and stops if they are not zero. `bubble_spin: full` forces the spinful bubble
and all sixteen pairs, which checks the reduced routes. [`combine_chis`](@ref) places every stored
pair, and refuses a file missing one its symmetry needs. The RPA is then solved one sector at a
time ([`spin_sectors`](@ref), [`sector_plan`](@ref)). An interaction that is not block diagonal in
the model's sectors falls back to the whole matrix. [`find_instability`](@ref) reports which
sector goes unstable.

`bilinear_basis: closed` needs an `SU2` model: the closure measures the orbital representation
from one spin block, which is not the symmetry of a spin-orbit-coupled model.

## Hermitian conjugates

[`interaction`](@ref) adds every registered bond **and** its Hermitian conjugate. Two consequences:

- Register **one direction** per inter-site bond. TightBindingToolkit's `Lookup` merges
  ``(i, j, R)`` with ``(j, i, -R)``, so registering both counts the bond twice.
- An **on-site** bond lands in its own block twice. Register `U/2 .* HUBBARD_CHANNELS` to get ``U``.
  [`hubbard_coulomb_interactions`](@ref) does this.

The interaction strength ``s`` multiplies the whole interaction file. A case's critical ``U`` is
``s_c`` times the ``U`` the file was built with.

## The bilinear basis

By default (`bilinear_basis: per_orbital`) the operators are the per-orbital densities and spins,
one per orbital. That set is closed under the point group only when the symmetry operations
**permute** orbitals. That holds for one orbital per site (square, triangular, honeycomb), but not
for an E doublet, where ``\chi_{ij}`` cannot be covariant.

`bilinear_basis: closed` replaces it with the span of the densities closed under the measured
orbital representation ([`bilinear_closure`](@ref)). That span is larger only where symmetry forces
it. Both stages must use it:

1. The interaction builder projects its vertex onto the same span: [`kanamori_vertex`](@ref) →
   [`project_vertex`](@ref) → [`channel_blocks`](@ref). It saves `"bilinear_basis"` and one
   constant matrix per case in `"closed_matrices"`.
2. `run_bare` contracts the bubble onto it and stores the basis in each `.npz`.
3. `run_rpa` measures the residual rotation between the two frames ([`basis_rotation`](@ref)) and
   applies it. It refuses mixed per-orbital/closed inputs, and skips cases with no closed-basis
   matrix (a ``q``-dependent tail has none).
