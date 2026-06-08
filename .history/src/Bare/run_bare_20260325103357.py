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
        params["fillings"]["values"] = fillings.tolist()
        params.setdefault("mus", {})
        params["mus"]["values"] = mus.tolist()
        params["mus"]["n"] = int(len(mus))
        return mus, fillings

    if "mus" not in params:
        raise ValueError("Input must define either fillings or mus.")

    if "values" in params["mus"]:
        mus = np.asarray(params["mus"]["values"], dtype=float)
    elif "n" in params["mus"]:
        mus = np.linspace(*bwidth, int(params["mus"]["n"]))
        params["mus"]["values"] = mus.tolist()
    else:
        raise ValueError("mus must define either values or n.")

    band = mdl.bands(hamiltonian, kmesh)
    fillings = np.array([mdl.filling(band, beta, float(mu)) for mu in mus], dtype=float)
    params["fillings"] = {
        "values": fillings.tolist(),
        "n": int(len(fillings)),
    }

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

    params["k_points_labels"] = [str(label) for label in k_point_labels]
    params["k_labels"] = [str(label) for label in k_point_labels]
    path_vecs, path_plot, path_ticks = mdl.k_path(model, resolved_k_points)
    
    #####* building the hamiltonian
    hamiltonian = mdl.hamiltonian(model, ksize)
    bandwidth = mdl.bandwidth(kmesh, hamiltonian)
    print("Hamiltonian built")
    
    
    #####* fillings vs chemical potential
    beta = params["beta"]
    w_max = float(params.get("w_max", 20.0))
    dlr_err = float(params.get("dlr_err", 1e-12))
    params["w_max"] = w_max
    params["dlr_err"] = dlr_err
    mus, fillings = resolve_scan_values(params, beta, hamiltonian, kmesh, bandwidth)

    with open(args.input, 'w') as file:
        yaml.dump(params, file)
    
    print("Starting TRIQS calculations...")
    
    for index, mu in enumerate(mus):
        print(f"calculating bare bubble for mu = {mu} => filling = {fillings[index]}...")

        chi00 = br.bare_chi(beta, w_max, dlr_err, mu, hamiltonian)
        
        if params["contract"]=="path":
            ks_contract = path_vecs
        else:
            ks_contract = ks
        
        output = {}
        for direction in params["directions"]:
            chi = br.interpolate_chi_mat(chi00, direction, N, ks_contract)
            output[labels[direction]] = chi

        print("contraction completed")


        fileName = params["output"] + f"_beta={beta}_mu={np.round(mu, 3)}.npz"
        np.savez(fileName, **output,
                    beta = beta, mu = float(mu), filling=float(fillings[index]), 
                    primitives=model.units, reciprocal = kmesh.bz.units, 
                    ks = ks, path = path_vecs, path_plot = path_plot, path_ticks = path_ticks,
                    contracted = ks_contract, 
                    bandwidth = np.array(bandwidth), bands = np.array([mdl.energies(k, hamiltonian) for k in path_vecs]))
    


