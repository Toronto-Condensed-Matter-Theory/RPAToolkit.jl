# Conventions

## Channels and matrix layout

Each orbital ``a`` carries four bilinears, one per entry of `directions`: N (`0`), X (`1`), Y
(`2`), Z (`3`), from the spin matrices ``\sigma^0, \sigma^x, \sigma^y, \sigma^z``. Every matrix
RPAToolkit works with, ``\chi(q)`` and ``V(q)`` alike, is `d * length(directions)` square, with
the channel index **innermost**: orbital ``a``, channel ``\mu`` sits at row
`length(directions) * (a - 1) + μ`. Without spin-orbit coupling ``\chi`` is block diagonal in the
channel.

## The bare susceptibility

[`combine_chis`](@ref) assembles the stored bubbles into
``\chi_{ij}(q) = \langle O_i(-q) O_j(q) \rangle`` (a plain transpose of what TRIQS's contraction
gives), normalized as the per-spin bubble ``\chi_0`` of Graser et al., NJP **11**, 025016 (2009)
(twice what is stored). [`CHI_ORIENTATION`](@ref) names this convention, and `run_rpa` records it
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

With that sign convention:

- **slot 1 (N) holds the spin vertex ``U^{(s)}``.** A positive entry is Stoner-enhanced.
- **slot 4 (Z) holds minus the charge vertex, ``-U^{(c)}``.** A repulsive charge interaction is
  negative there and screened.
- X and Y stay empty without spin-orbit coupling.

The two building blocks:

| term | channels | constant |
|:-----|:---------|:---------|
| on-site Hubbard ``U n_\uparrow n_\downarrow`` | ``\mathrm{diag}(+1, 0, 0, -1)``: ``U^{(s)} = U^{(c)} = U`` | [`HUBBARD_CHANNELS`](@ref) |
| density-density ``V n_a n_b`` | ``\mathrm{diag}(0, 0, 0, -1)``: charge only | [`DENSITY_CHANNELS`](@ref) |

With `DENSITY_CHANNELS`, an on-site ``U`` plus a tail ``V(q) = \sum_\delta e^{-iq\cdot\delta} V_\delta``
gives ``U^{(c)}(q) = U + V(q)``. Graser's density-density vertex is ``2V``; the multi-orbital
ZrNCl builder in the TMD project registers `diag(0, 0, 0, -2)` for that reason. The two differ
only in what is called ``V``.

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
