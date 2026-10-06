import numpy as np
import argparse
import yaml 
#####* importing other modules
import model as mdl
import bare_response as br
import bilinear_basis as bb

try:
    import triqs.utility.mpi as triqs_mpi
    if hasattr(triqs_mpi, 'size'):
        mpi_size = triqs_mpi.size
        mpi_rank = triqs_mpi.rank
    else:
        mpi_size = triqs_mpi.mpi.size
        mpi_rank = triqs_mpi.mpi.rank
except ImportError:
    mpi_size = 1
    mpi_rank = 0

labels = {0 : "chi_NN", 1 : "chi_XX", 2 : "chi_YY", 3 : "chi_ZZ", 4 : "chi_NN"}


def _scan_axis(cfg: dict, name: str, span=None) -> np.ndarray:
    """One scan axis, from `values`, or `min`/`max`/`n`, or a bare `n` spanning `span`."""
    if "values" in cfg:
        return np.asarray(cfg["values"], dtype=float)
    if all(k in cfg for k in ("min", "max", "n")):
        return np.linspace(float(cfg["min"]), float(cfg["max"]), int(cfg["n"]))
    if "n" in cfg and span is not None:
        return np.linspace(float(span[0]), float(span[1]), int(cfg["n"]))
    allowed = "`values`, or `min`+`max`+`n`" + (", or a bare `n` spanning the bandwidth"
                                                if span is not None else "")
    raise ValueError(f"{name} must define {allowed}; got keys {sorted(cfg)}.")


def resolve_scan_values(params: dict, beta: float, hamiltonian, kmesh, bwidth: tuple):
    """The (mus, fillings) this run scans over, and which of the two the config asked for.

    MU IS THE SCAN VARIABLE. chi0 is computed at a chemical potential; the filling is a
    label derived from it by counting states. A config may instead ask for fillings, and
    then the mus are obtained ONCE by numerically inverting filling(mu) -- but from that
    point on the mus are what the run IS, and the fillings are a record of what was asked
    for.

    That "once" is the whole reason this function returns a `source` and the caller writes
    it back. run_bare.py resolves the scan and then dumps the resolved values into the
    runtime YAML, which means the file it reads and the file it writes are the same file.
    Before this was explicit, the write-back put BOTH `mus.values` and `fillings.values`
    into that file while `fillings` silently took precedence on read -- so re-running the
    same runtime YAML took the inversion branch instead of the values branch and landed on
    a slightly different mu (measured: 9.3e-08, which moves chi by 4.5e-07, ten thousand
    times the 1.9e-11 that separates the two bubble methods). The pipeline regenerates the
    runtime file from the config each time so it never hit this, but anyone re-running a
    runtime YAML by hand did.

    With `scan.source` recorded, a resolved file is resolved: its mus are taken verbatim
    and only the fillings are recomputed, so reading it again is a no-op.
    """
    band = mdl.bands(hamiltonian, kmesh)
    fillings_of = lambda mus: np.array([mdl.filling(band, beta, float(mu)) for mu in mus],
                                       dtype=float)

    recorded = params.get("scan", {}).get("source")
    if recorded is not None:
        mus = np.asarray(params["mus"]["values"], dtype=float)
        return mus, fillings_of(mus), recorded

    has_mus, has_fillings = "mus" in params, "fillings" in params
    if has_mus and has_fillings:
        raise ValueError(
            "the config sets BOTH `mus` and `fillings`, and there is no way to tell which "
            "one is meant: they are two parameterisations of the same scan. Delete one. "
            "(If this is a runtime YAML a previous run wrote, it should also carry "
            "`scan.source`; regenerate it from the config rather than editing it.)")
    if not (has_mus or has_fillings):
        raise ValueError("the config must define either `mus` or `fillings`.")

    if has_mus:
        mus = _scan_axis(params["mus"], "mus", span=bwidth)
        return mus, fillings_of(mus), "mus"

    fillings = _scan_axis(params["fillings"], "fillings")
    return mdl.mus_from_fillings(fillings, beta, hamiltonian, kmesh), fillings, "fillings"

if __name__=="__main__":
    
    #####* defining the command line arguments to parse
    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help')
    
    parser.add_argument('input', help='Input file location', type=str, default="")
    args = parser.parse_args()
    #####* loading the input file
    fobj = open(args.input, "r")
    params = yaml.load(fobj, Loader=yaml.CLoader)
    #####* loading the unit cell
    unitcell = np.load(params["unitcell"]["triqs"])
    print("Unit cell loaded")
    
    #####* building the triqs model
    model = mdl.triqs_model(unitcell)
    print("Model built")

    N = int(len(model.orbital_names)/2)

    #####* the bilinear basis the susceptibility is contracted onto.
    #####* "per_orbital" (default) keeps the historical one operator per orbital.
    #####* That set is closed under the point group only when D(g) PERMUTES
    #####* orbitals; for an E doublet (D rotates) chi_ij cannot be covariant, so
    #####* "closed" replaces it with the span of the densities closed under the
    #####* measured D(g) -- identical for every model where the old basis was
    #####* already adequate, larger only where symmetry forces it.
    #####*
    #####* The seed is the Kanamori one (densities PLUS the on-site off-diagonal
    #####* bilinears), not the densities alone: the interaction builder has to
    #####* project its vertex onto the SAME span or that vertex is not
    #####* representable here, and run_RPA.jl refuses the pair when the two
    #####* disagree. Seeding wider costs nothing for a model with one orbital
    #####* per site, which is every model that already worked.
    basis_mode = str(params.get("bilinear_basis", "per_orbital")).lower()
    if basis_mode not in ("per_orbital", "closed"):
        raise ValueError(f"bilinear_basis must be 'per_orbital' or 'closed', got {basis_mode!r}")
    bilinear = None
    if basis_mode == "closed":
        hoppings_sp = {tuple(int(x) for x in unitcell["hopping offsets"][:, i]):
                       np.array(unitcell["hopping matrices"][i, :, :])
                       for i in range(unitcell["hopping offsets"].shape[1])}
        units_rs = [np.asarray(u, float) for u in np.transpose(unitcell["units"])]
        positions_rs = [np.asarray(p, float) for p in np.transpose(unitcell["orbital_positions"])]
        bilinear = bb.closed_basis_for_model(hoppings_sp, units_rs, positions_rs, N,
                                             seed=str(params.get("bilinear_seed", "kanamori")))
    basis_dim = N if bilinear is None else len(bilinear)
    #####* building the Brillouin zone and a high symmetry path
    ksize = params["k_size"]
    kmesh = model.get_kmesh(n_k=(ksize, ksize, 1))
    ks = np.array([k.value for k in kmesh])
    resolved_k_points, inferred_k_labels = mdl.normalize_k_points(model, params["k_points"])
    if "k_points_labels" in params:
        k_point_labels = params["k_points_labels"]
    elif "k_labels" in params:
        k_point_labels = params["k_labels"]
    else:
        k_point_labels = inferred_k_labels

    if len(k_point_labels) != len(resolved_k_points):
        raise ValueError("k_points_labels length must match number of k_points.")

    k_point_labels = [str(label) for label in k_point_labels]
    path_vecs, path_plot, path_ticks = mdl.k_path(model, resolved_k_points)
    
    #####* building the hamiltonian
    hamiltonian = mdl.hamiltonian(model, ksize)
    bandwidth = mdl.bandwidth(kmesh, hamiltonian)
    print("Hamiltonian built")

    #####* The bubble's own Hamiltonian. With no spin-orbit coupling (H = h x 1_spin exactly) the
    #####* spinless bubble times 1/2 IS every contraction of the spinful one (bare_response,
    #####* "Spin factorization"): 16x less chi0 and 4x less G0 per rank, the same numbers.
    #####* `bubble_spin: full` forces the spinful bubble, e.g. to check the two against each other.
    bubble_spin = str(params.get("bubble_spin", "auto")).lower()
    if bubble_spin not in ("auto", "full"):
        raise ValueError(f"bubble_spin must be 'auto' or 'full', got {bubble_spin!r}")
    spin_factorized = bubble_spin == "auto" and br.spin_factorizable(unitcell)
    if spin_factorized:
        ham_bubble = mdl.hamiltonian(mdl.triqs_model_spinless(br.spinless_unitcell(unitcell)), ksize)
    else:
        ham_bubble = hamiltonian
    print("bare bubble spin: " + ("factorized (H = h x 1_spin exactly; spinless bubble x 1/2, one "
                                  "contraction for every direction)" if spin_factorized else
                                  f"full ({'forced by bubble_spin: full' if bubble_spin == 'full' else 'the model is not h x 1_spin'})"))
    
    
    #####* fillings vs chemical potential
    beta = params["beta"]
    w_max = float(params.get("w_max", 20.0))
    dlr_err = float(params.get("dlr_err", 1e-12))
    #####* how chi0(Omega = 0, q) is evaluated. "lindhard" (default) is the analytic
    #####* Matsubara sum: exact, but O(N_k * N_q), which is ~120 days per mu at k_size 297.
    #####* "dlr" is the DLR/imaginary-time FFT bubble: ~linear in N_k, measured 175x faster
    #####* at k_size 27 and agreeing to 3e-11 at dlr_err 1e-12, at ~3.7x the peak memory.
    #####* w_max and dlr_err were dead config keys until this existed. See bare_chi.
    bubble = str(params.get("bubble", "lindhard")).lower()
    if bubble not in ("lindhard", "dlr"):
        raise ValueError(f"bubble must be 'lindhard' or 'dlr', got {bubble!r}")
    print(f"bare bubble: {bubble}" + (f" (w_max={w_max}, eps={dlr_err})" if bubble == "dlr" else ""))
    mus, fillings, scan_source = resolve_scan_values(params, beta, hamiltonian, kmesh, bandwidth)

    # Persist the resolved scan into the runtime YAML this run was handed. run_bare.jl
    # regenerates that file from the config, so the user's own input is never modified --
    # but this function both READS and WRITES it, so it has to say which of mus/fillings
    # was the input. `scan.source` is what makes a second read a no-op (see
    # resolve_scan_values); `scan.requested` keeps the original block, because the
    # resolved arrays overwrite it below. Overwriting rather than updating in place is
    # deliberate: leaving a stale `min`/`max` next to a resolved `values` invites exactly
    # the "which one is real?" confusion this is meant to remove.
    params["scan"] = {"source": scan_source,
                      "requested": params.get("scan", {}).get("requested",
                                                              params.get(scan_source, {}))}
    params["mus"] = {"values": [float(mu) for mu in mus], "n": int(len(mus))}
    params["fillings"] = {"values": [float(v) for v in fillings], "n": int(len(fillings))}
    print(f"scan: {len(mus)} point(s), specified as `{scan_source}`"
          + (" (mus inverted from them numerically, once)" if scan_source == "fillings" else "")
          + f"\n      mu      {np.min(mus):.6g} .. {np.max(mus):.6g}"
          + f"\n      filling {np.min(fillings):.6g} .. {np.max(fillings):.6g}")

    if mpi_rank == 0:
        with open(args.input, 'w') as file:
            yaml.dump(params, file)
    
    # Barrier to ensure all processes have read before rank 0 potentially overwrites it,
    # though they should have read it on line 50.
    if mpi_size > 1:
        triqs_mpi.barrier()
        
    print(f"Rank {mpi_rank} starting TRIQS calculations...")

    def compute_for_filling(args_tuple):
        index, mu, filling = args_tuple
        print(f"calculating bare bubble for mu = {mu} => filling = {filling}...")

        chi00 = br.bare_chi(beta, w_max, dlr_err, mu, ham_bubble, method=bubble)
        
        output = {}
        contracted = {}
        for direction in params["directions"]:
            # Factorized, every direction is the same contraction: do it once.
            key = "any" if spin_factorized else direction
            if key not in contracted:
                if bilinear is None:
                    contracted[key] = (br.interpolate_chi_mat(chi00, direction, N, ks, spin_factorized),
                                       br.interpolate_chi_mat(chi00, direction, N, path_vecs, spin_factorized))
                else:
                    contracted[key] = (br.interpolate_chi_basis(chi00, bilinear, direction, ks, spin_factorized),
                                       br.interpolate_chi_basis(chi00, bilinear, direction, path_vecs, spin_factorized))
            chi_grid, chi_path = contracted[key]
            output[labels[direction]] = chi_grid
            output[labels[direction] + "_path"] = chi_path

        print(f"contraction completed for mu = {mu}")

        fileName = params["output"] + f"_beta={beta}_mu={np.round(mu, 3)}.npz"
        
        # The basis travels with the bubble, because a set of coefficients means nothing
        # without the frame it is expressed in: run_RPA.jl rotates the vertex into THIS basis.
        # Written only on the closed path -- a zero-length placeholder array is not a neutral
        # way to say "absent": ZipFile.jl raises EOFError reading a zero-length entry, and
        # npzread_numeric re-raises anything that is not an unsupported dtype, so one such
        # entry makes the whole handoff file unreadable from Julia.
        basis_out = ({} if bilinear is None else
                     {"bilinear_dim": np.array([basis_dim]),
                      "bilinear_basis": np.array(bilinear)})

        # Only rank 0 saves the file in MPI runs
        if mpi_rank == 0:
            np.savez(fileName, **output, **basis_out,
                        beta = beta, mu = float(mu), filling=float(filling),
                        bubble_spin_factorized = np.array([int(spin_factorized)]),
                        primitives=model.units, reciprocal = kmesh.bz.units,
                        ks = ks, path = path_vecs, path_plot = path_plot, path_ticks = path_ticks,
                        contracted = ks,
                        bandwidth = np.array(bandwidth), bands = np.array([mdl.energies(k, hamiltonian) for k in path_vecs]))

    tasks = [(index, mu, fillings[index]) for index, mu in enumerate(mus)]

    if mpi_size > 1:
        # TRIQS parallelizes internally over k-points via MPI
        for task in tasks:
            compute_for_filling(task)
    else:
        # No MPI: parallelize over fillings using multiprocessing
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        print(f"No MPI detected. Using multiprocessing over fillings with {num_cores} cores.")
        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.map(compute_for_filling, tasks)

    


