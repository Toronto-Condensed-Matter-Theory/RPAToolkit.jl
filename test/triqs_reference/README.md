# TRIQS reference implementation

The Python modules here are the bare stage RPAToolkit used before the bubble moved to Julia
(`src/Bubble.jl`, `src/DLR.jl`). They are kept for two reasons only:

- `test/make_fixtures.py` and `test/make_spin_fixtures.py` import them to regenerate the committed
  fixtures in `test/fixtures/`, which the Julia bubble is tested against. Regenerating needs TRIQS
  and triqs_tprf; running the test suite does not.
- `bilinear_basis.py` is the reference the Julia symmetry closure (`src/BilinearBasis.jl`) is
  checked against by `test/test_bilinear_basis.py` (numpy only).

Nothing in the package imports them.

The convention the Julia stage reproduces, measured against these modules to 5e-14 on every
fixture and on a two-site cell run through the old pipeline: the stored channel-pair blocks are
`transpose(raw)` of the Lindhard sum computed with the transposed operators (see the notes at the
top of `src/Bubble.jl`).
