import numpy as np
from triqs_tprf.tight_binding import TBLattice

lat = TBLattice(
    units = [(1,0,0),(0,1,0),(0,0,1)],
    orbital_positions = [(0,0,0)],
    orbital_names = ['1'],
    hoppings = {(0,0,0): np.zeros((1,1))}
)
kmesh = lat.get_kmesh((10,10,1))
ham = lat.fourier(kmesh)
band1 = [np.linalg.eigvalsh(ham(kmesh[i].value)) for i in range(len(kmesh))]
band2 = [np.linalg.eigvalsh(ham(k.value)) for k in kmesh]
print("band1 len:", len(band1))
print("band2 len:", len(band2))
