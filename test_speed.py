import numpy as np
import time
from triqs_tprf.tight_binding import TBLattice
from triqs.gf import MeshImFreq
from triqs_tprf.lattice import lindhard_chi00

lat = TBLattice(
    units = [(1,0,0),(0,1,0),(0,0,1)],
    orbital_positions = [(0,0,0), (0.5, 0.5, 0.5)],
    orbital_names = ['1', '2'],
    hoppings = {(0,0,0): np.zeros((2,2))}
)
kmesh = lat.get_kmesh((96,96,1))
ham = lat.fourier(kmesh)
bmesh = MeshImFreq(beta=50.0, S='Boson', n_iw=1)
mu = 0.0

start = time.time()
chi00 = lindhard_chi00(ham, bmesh, mu=mu)
print("Time taken:", time.time() - start)
