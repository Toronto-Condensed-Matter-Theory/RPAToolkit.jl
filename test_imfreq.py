import numpy as np
from triqs_tprf.tight_binding import TBLattice
from triqs.gf import MeshImFreq
from triqs_tprf.lattice import lattice_dyson_g0_wk
from triqs_tprf.lattice_utils import imtime_bubble_chi0_wk

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

wmesh = MeshImFreq(beta=beta, S='Fermion', n_iw=100)
g0_wk = lattice_dyson_g0_wk(mu=mu, e_k=ham, mesh=wmesh)
chi00_wk = imtime_bubble_chi0_wk(g0_wk, nw=1)
print("chi00_wk shape:", chi00_wk.data.shape)
