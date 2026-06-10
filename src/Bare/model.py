import numpy as np
import ast
from triqs_tprf.tight_binding import TBLattice
from triqs_tprf.lattice_utils import k_space_path


def _as_float_triplet(point) -> list:
    coords = np.asarray(point, dtype=float).reshape(-1)
    if len(coords) != 3:
        raise ValueError(f"k-point {point} must contain exactly 3 coordinates.")
    return coords.tolist()


def normalize_k_points(model, k_points: list):
    resolved_points = []
    inferred_labels = []

    for point in k_points:
        if isinstance(point, str):
            raise ValueError(
                "String k_points must be resolved before Python execution. "
                "Run src/Bare/run_bare.jl so TightBindingToolkit resolves HighSymPoints."
            )

        resolved = _as_float_triplet(point)
        resolved_points.append(resolved)
        inferred_labels.append(str([float(value) for value in resolved]))

    if len(resolved_points) < 2:
        raise ValueError("At least two k-points are required to build a path.")

    return resolved_points, inferred_labels

#####* convert a unitcell dictionary to a triqs model
#####* unitcell dictionary can be made from TightBindingToolkit or from a saved file
def triqs_model(unitcell: dict):
    
    units = [tuple(unit) for unit in np.transpose(unitcell["units"])]
    positions = [tuple(pos) for pos in np.transpose(unitcell["orbital_positions"])]
    
    subs = len(positions)/2
    names = [f"{i+1}:{spin}" for spin in ["up", "dn"] for i in range(int(subs))]
    hoppings = {tuple(unitcell["hopping offsets"][:, i]) : np.array(unitcell["hopping matrices"][i, :, :]) 
                    for i in range(unitcell["hopping offsets"].shape[1])}
    
    return TBLattice(
        units = units,
        orbital_positions = positions,
        orbital_names = names,
        hoppings = hoppings
    )

#####* return the hamiltonian in the Brillouin zone for the corresponding model
def hamiltonian(model, ksize: int):
    kmesh = model.get_kmesh(n_k=(ksize, ksize, 1))
    ham = model.fourier(kmesh)
    return ham

#####* return a high symmetry path in the Brillouin zone
def k_path(model, k_points: list):
    if len(k_points) < 2:
        raise ValueError("At least two k-points are required to build a path.")

    paths = []
    for i in range(len(k_points)-1):
        paths.append((k_points[i], k_points[i+1]))
    
    paths.append((k_points[-1], k_points[0]))
        
    k_vecs, k_plot, k_ticks = k_space_path(paths, bz=model.bz)
    return k_vecs, k_plot, k_ticks

#####* return bands at a momentum k ##########
def energies(k, ham):
    return np.linalg.eigvalsh(ham(k))

#####* returns band energies at each k-value
def bands(ham, kmesh):
    band = [np.linalg.eigvalsh(ham(k.value)) for k in kmesh]
    band = np.concatenate(band, axis = 0)
    
    return band

#####* return total bandwidth #########
def bandwidth(ks, ham):
    bands = np.array([energies(k, ham) for k in ks])
    return (np.min(bands), np.max(bands))

#####* fermi distribution function
def fermi(e: float, beta: float, mu: float) -> float:
    return 1.0 / (np.exp(beta * (e-mu)) + 1.0)

#####* filling at fixed temperature and chemical potential
def filling(band, beta:float, mu: float)-> float:
    return np.sum(fermi(band, beta, mu))/len(band)

def get_filling(mu: float, beta: float, ham, kmesh) -> float:
    return filling(bands(ham, kmesh), beta, mu)


def _mu_from_filling(target_filling: float, beta: float, band: np.ndarray,
                     mu_min: float, mu_max: float, tol: float = 1e-8,
                     max_iter: int = 200) -> float:
    if not (0.0 <= target_filling <= 1.0):
        raise ValueError(f"Target filling {target_filling} must be in [0, 1].")

    def objective(mu_value: float) -> float:
        return filling(band, beta, mu_value) - target_filling

    f_min = objective(mu_min)
    f_max = objective(mu_max)

    if f_min > 0.0 and abs(f_min) > tol:
        raise ValueError(f"Target filling {target_filling} is below reachable range at mu_min={mu_min}.")
    if f_max < 0.0 and abs(f_max) > tol:
        raise ValueError(f"Target filling {target_filling} is above reachable range at mu_max={mu_max}.")

    if abs(f_min) <= tol:
        return float(mu_min)
    if abs(f_max) <= tol:
        return float(mu_max)

    low = float(mu_min)
    high = float(mu_max)

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = objective(mid)

        if abs(f_mid) <= tol:
            return float(mid)

        if f_mid > 0.0:
            high = mid
        else:
            low = mid

    return float(0.5 * (low + high))


def mus_from_fillings(fillings: np.ndarray, beta: float, ham, kmesh,
                      tol: float = 1e-8, max_iter: int = 200) -> np.ndarray:
    band = bands(ham, kmesh)
    mu_min = float(np.min(band))
    mu_max = float(np.max(band))

    return np.array([
        _mu_from_filling(float(target), beta, band, mu_min, mu_max, tol=tol, max_iter=max_iter)
        for target in np.asarray(fillings, dtype=float)
    ])

#####* finding filling vs mu ##################
def filling_vs_mu(beta: float, n: int, ham, kmesh):

    bwidth = bandwidth(kmesh, ham)
    mus = np.linspace(*bwidth, n)
    fillings = np.zeros(len(mus))
    band = bands(ham, kmesh)
    
    print("calculating chemical potential vs filling...")
    for (i, mu) in enumerate(mus):
        fillings[i] = filling(band, beta, mu)
    
    return mus, fillings