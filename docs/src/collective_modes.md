# Collective modes and mean-field theory

The static RPA finds where an instability sets in. At real frequency the same RPA describes the
collective modes of the state it is built on: magnons of a magnet, plasmons of a metal. Built on a
Hartree-Fock state with the vertex of the same interaction, it is the linear response of that
state, time-dependent Hartree-Fock (TDHF).

## The real-frequency bubble

[`dynamic_bubble`](@ref) returns the retarded bubble ``\chi_0(q, \omega + i\eta)``, one
`4d × 4d` matrix per momentum and frequency, in the static pipeline's layout (channel innermost,
per-spin normalization, [Conventions](@ref)). At ``\omega = 0`` it is exactly what `run_rpa`
assembles from the stored static bubble, the same discrete sum even for momenta off the k grid,
so every interaction matrix built for the static RPA applies unchanged. It is a proper response
matrix: the absorptive part ``(\chi - \chi^\dagger)/2i`` is ``\mathrm{sign}(\omega)`` times a positive
semidefinite matrix.

Two ways of doing the frequency sum:

- `nbins = nothing`: every pole exactly. Costs one term per band pair and frequency.
- `nbins = N` (default 4000): the pole weights are spread linearly onto ``N`` energies spanning the
  particle-hole range, then one matrix product gives all frequencies. The error is
  O(bin width / ``\eta``).

The momenta are arbitrary (a path), the k sum runs over a `k_size × k_size` grid, and the work is
threaded over momenta. The broadening ``\eta`` has to exceed the level spacing the grid resolves
near the momenta of interest, roughly ``v_F \cdot 2\pi/(k_\mathrm{size}\, a)``.

[`dynamic_rpa`](@ref) solves ``\chi_\mathrm{RPA} = (1 - \chi_0 V)^{-1} \chi_0`` at every point;
[`particle_hole_edges`](@ref) gives the upper edge of the particle-hole continuum (the support of
``\mathrm{Im}\,\chi_0`` before broadening), above which a collective mode is undamped.

## Spectral functions and supercells

A magnetically ordered state lives on a supercell, whose Brillouin zone folds the original one.
[`unfold_vector`](@ref) writes the original lattice's operator at momentum ``q``,
``\sum_r e^{-iq\cdot r} c^\dagger_r \sigma^a c_r/2``, in the supercell's per-site channels,
``u_{(i,a)} = e^{+iq\cdot b_i}/\sqrt{n}``, so that ``u^\dagger \chi u`` is the original lattice's
susceptibility at ``q`` (pinned numerically: a paramagnetic two-site cell unfolds onto the one-site
bubble). [`spectral_map`](@ref) does this along a path, and [`rpa_poles`](@ref) reads off the peak
at each momentum.

## Long-range Coulomb

A two-dimensional plasmon needs ``V(q) \propto 1/|q|`` as ``q \to 0``, which no real-space tail
truncated at a finite range has. [`coulomb_2d`](@ref) Ewald-sums
``\frac{1}{2}\sum_{r \ne r'} e^2/(\varepsilon|r - r'|)\, n_r n_{r'}`` exactly (its value does not
depend on the Ewald split, to ``10^{-13}``). It returns ``-2 v(q)`` in the charge slot: the factor 2
because ``\chi`` is the per-spin bubble, so that the RPA reads ``1 + 2 v \chi_0 = 1 + v \Pi`` with
``\Pi`` the full density bubble, and the 2D electron gas gives
``\omega_p^2 = 2\pi n e^2 q/(\varepsilon m)``.

## Hartree-Fock bands: the MeanFieldToolkit interface

A converged MeanFieldToolkit `TBMFTModel` hands its state to the RPA:

```julia
using RPAToolkit, MeanFieldToolkit
# ... build `mft`, then sc = SolveMFT!(mft, initial; Update = AndersonMixing, ...) ...
r = rpa_model(mft; residual = maximum(abs, sc.VOuts[end] - sc.VIns[end]))
V = rpa_vertex(mft)                         # q -> V(q)
chi0 = dynamic_bubble(r.bands, qs, omegas, k_size; beta = r.beta, mu = r.mu, eta = 0.02)
chi = dynamic_rpa(chi0, V.(qs))
```

[`rpa_model`](@ref) returns the [`BandModel`](@ref) of the Hartree-Fock unit cell (the free
hoppings plus the decoupled interaction), with its chemical potential and temperature.
[`rpa_vertex`](@ref) builds the vertex from the same interaction tensor MeanFieldToolkit
decoupled. For an on-site term ``\sum U_{abcd}\, c^\dagger_a c_b c^\dagger_c c_d``, that is the
derivative of its Hartree-Fock self-energy,
``K_{abcd} = U_{abcd} + U_{cdab} - U_{cbad} - U_{adcb}`` (MeanFieldToolkit's
`IntraQuarticToHopping` kernel), projected on the channels ([`quartic_onsite_vertex`](@ref),
[`tdhf_channel_vertex`](@ref)). For ``U n_\uparrow n_\downarrow`` it is exactly
[`HUBBARD_CHANNELS`](@ref).

Because the bands and the vertex come from one mean-field functional, a spontaneously broken
continuous symmetry gives an exact Goldstone mode. For the Néel state of the Hubbard model, the
largest transverse eigenvalue of ``\chi_0 V`` at the ordering vector is 1 to all printed digits, at
every ``U`` (`examples/MeanField/Magnons.jl`, `test/test_meanfield_ext.jl`). The offset scales with
the self-consistency residual, which is why `rpa_model` checks it: MeanFieldToolkit builds the final
unit cell from the last input order parameters.

Inter-site interactions (`InterQuarticToHopping`) enter with their Hartree part
([`quartic_hartree_vertex`](@ref)). Their exchange part renormalizes the Hartree-Fock bands (a
nearest-neighbour ``V`` widens the band). That part is a bond term with no representation on
per-site channels, so it is left out of the RPA, and that part of the RPA is not the linear response
of the mean-field state; `rpa_vertex` warns. `rpa_vertex(mft; interactions = [...])` keeps only
some terms, e.g. to take the on-site vertex from the mean-field theory and the long-range tail from
[`coulomb_2d`](@ref).

## Worked examples

- `examples/MeanField/Magnons.jl`: square-lattice Hubbard, Néel order. The zone-boundary magnon is
  0.71, 0.91 and 0.96 of linear spin-wave theory's ``2J = 8t^2/U`` at ``U = 4, 8, 12``, and the
  zone-boundary dispersion has the sign of the higher-order (ring-exchange) correction,
  ``\omega(\pi/2, \pi/2) < \omega(\pi, 0)``.
- `examples/MeanField/Plasmons.jl`: a dilute metal (``n = 0.1``) with ``e^2/\varepsilon r``. The plasmon
  follows ``\sqrt{2\pi n e^2 q/m^*}`` at small ``q`` (``m^*`` the Hartree-Fock mass, 2% lighter than the
  bare one through the Fock term), disperses faster at larger ``q``, and is Landau-damped once it
  meets the particle-hole continuum.
