import numpy as np
import argparse
import yaml 
#####* importing other modules
import model as mdl
import bare_response as br

labels = {0 : "chi_NN", 1 : "chi_XX", 2 : "chi_YY", 3 : "chi_ZZ", 4 : "chi_NN"}


def resolve_scan_values(params: dict, beta: float, hamiltonian, kmesh, bwidth: tuple):
    if "fillings" in params:
        filling_config = params["fillings"]
        if "values" in filling_config:
            fillings = np.asarray(filling_config["values"], dtype=float)
        elif all(key in filling_config for key in ["min", "max", "n"]):
            fillings = np.linspace(float(filling_config["min"]), float(filling_config["max"]), int(filling_config["n"]))
        else:
            raise ValueError("fillings must define either values or min/max/n.")

        mus = mdl.mus_from_fillings(fillings, beta, hamiltonian, kmesh)
        return mus, fillings

    if "mus" not in params:
        raise ValueError("Input must define either fillings or mus.")

    if "values" in params["mus"]:
        mus = np.asarray(params["mus"]["values"], dtype=float)
    elif "n" in params["mus"]:
        mus = np.linspace(*bwidth, int(params["mus"]["n"]))
    else:
        raise ValueError("mus must define either values or n.")

    band = mdl.bands(hamiltonian, kmesh)
    fillings = np.array([mdl.filling(band, beta, float(mu)) for mu in mus], dtype=float)

    return mus, fillings

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
    
    
    #####* fillings vs chemical potential
    beta = params["beta"]
    w_max = float(params.get("w_max", 20.0))
    dlr_err = float(params.get("dlr_err", 1e-12))
    mus, fillings = resolve_scan_values(params, beta, hamiltonian, kmesh, bandwidth)

    # Persist resolved scan values to the runtime input file used in this run.
    # run_bare.jl passes a temporary YAML, so the user input file is not modified.
    params.setdefault("mus", {})
    params["mus"]["values"] = [float(mu) for mu in mus]
    params["mus"]["n"] = int(len(mus))
    params.setdefault("fillings", {})
    params["fillings"]["values"] = [float(val) for val in fillings]
    params["fillings"]["n"] = int(len(fillings))

    with open(args.input, 'w') as file:
        yaml.dump(params, file)
    
    print("Starting TRIQS calculations...")
    
    try:
        from triqs.utility.mpi import mpi
        mpi_size = mpi.size
        mpi_rank = mpi.rank
    except ImportError:
        mpi_size = 1
        mpi_rank = 0

    def compute_for_filling(args_tuple):
        index, mu, filling = args_tuple
        print(f"calculating bare bubble for mu = {mu} => filling = {filling}...")

        chi00 = br.bare_chi(beta, w_max, dlr_err, mu, hamiltonian)
        
        if params["contract"]=="path":
            ks_contract = path_vecs
        else:
            ks_contract = ks
        
        output = {}
        for direction in params["directions"]:
            chi = br.interpolate_chi_mat(chi00, direction, N, ks_contract)
            output[labels[direction]] = chi

        print(f"contraction completed for mu = {mu}")

        fileName = params["output"] + f"_beta={beta}_mu={np.round(mu, 3)}.npz"
        
        # Only rank 0 saves the file in MPI runs
        if mpi_rank == 0:
            np.savez(fileName, **output,
                        beta = beta, mu = float(mu), filling=float(filling), 
                        primitives=model.units, reciprocal = kmesh.bz.units, 
                        ks = ks, path = path_vecs, path_plot = path_plot, path_ticks = path_ticks,
                        contracted = ks_contract, 
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

    


