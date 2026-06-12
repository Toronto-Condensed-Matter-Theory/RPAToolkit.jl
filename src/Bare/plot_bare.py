import numpy as np
import yaml
from matplotlib import pyplot as plt
import argparse

labels = {0 : "chi_NN", 1 : "chi_XX", 2 : "chi_YY", 3 : "chi_ZZ", 4 : "chi_NN"}
titles = {0 : r'$\chi_{dd}(\mathbf{Q}, \Omega=0)$', 
            1 : r'$\chi_{xx}(\mathbf{Q}, \Omega=0)$', 
            2 : r'$\chi_{yy}(\mathbf{Q}, \Omega=0)$', 
            3 : r'$\chi_{zz}(\mathbf{Q}, \Omega=0)$', 
            4 : r'$\chi_{dd}(\mathbf{Q}, \Omega=0)$'}

def plot_chi(data: dict, direction: int, path_labels: list, title: str, saveFile: str):
    chi = data[labels[direction]]
    eigs = np.linalg.eigvals(chi)
    
    plt.clf()
    plt.plot(data["path_plot"], np.real(eigs))
    plt.grid()
    plt.xticks(ticks=data["path_ticks"], labels=path_labels)
    plt.title(title)
    plt.savefig(saveFile)


def format_path_labels(raw_labels):
    labels = [str(label) for label in raw_labels]
    if len(labels) == 0:
        return labels

    formatted = [r'${label}$'.format(label=label) for label in labels]
    formatted.append(formatted[0])
    return formatted


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
    
    try:
        from triqs.utility.mpi import mpi
        mpi_rank = mpi.rank
    except ImportError:
        mpi_rank = 0

    if mpi_rank == 0:
        #####* fillings vs chemical potential
        beta = params["beta"]
        outDirectory = params["output"]
        if "mus" not in params or "values" not in params["mus"]:
            raise ValueError("plot_bare.py requires mus.values in the provided input file.")
        mus = params["mus"]["values"]

        plotDirectory = params["plots"]
        
        for mu in mus:
            output = outDirectory + f"_beta={beta}_mu={np.round(mu, 3)}.npz"
            data = np.load(output, allow_pickle=True)

            if "k_point_labels" in data.files:
                path_labels = format_path_labels(data["k_point_labels"])
            elif "k_points_labels" in params:
                path_labels = format_path_labels(params["k_points_labels"])
            else:
                path_labels = format_path_labels(params["k_labels"])
            
            #####* plotting the bands
            plt.clf()
            plt.plot(data["path_plot"], data["bands"])
            plt.xticks(data["path_ticks"], path_labels)
            plt.ylabel(r'$\epsilon(\mathbf{k})$')
            plt.grid(True)
            plt.savefig(f"{plotDirectory}_bands.png") 
            
            #####* plotting the path
            plt.clf()
            k_vecs = data["path"]
            plt.scatter(k_vecs[:,0], k_vecs[:,1]) 
            plt.xlabel(r'$k_x$')
            plt.ylabel(r'$k_y$')
            plt.grid(True)
            plt.savefig(f"{plotDirectory}_path.png")
    


