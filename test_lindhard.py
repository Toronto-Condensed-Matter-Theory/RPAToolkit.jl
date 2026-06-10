import numpy as np
from triqs_tprf.tight_binding import TBLattice
from triqs_tprf.lattice import lindhard_chi00
from triqs.gf import MeshImFreq

lat = TBLattice(
    units = [(1,0,0),(0,1,0),(0,0,1)],
    orbital_positions = [(0,0,0)],
    orbital_names = ['1'],
    hoppings = {(0,0,0): np.zeros((1,1))}
)
kmesh = lat.get_kmesh((10,10,1))
ham = lat.fourier(kmesh)
beta = 10.0
mu = 0.0

bmesh = MeshImFreq(beta=beta, S='Boson', n_iw=1)
chi00 = lindhard_chi00(ham, bmesh, mu=mu)
print("chi00 shape:", chi00.data.shape)
