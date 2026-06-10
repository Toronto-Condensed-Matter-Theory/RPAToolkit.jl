import numpy as np
from triqs_tprf.tight_binding import TBLattice

lat = TBLattice(
    units = [(1,0,0),(0,1,0),(0,0,1)],
    orbital_positions = [(0,0,0)],
    orbital_names = ['1'],
    hoppings = {(0,0,0): np.zeros((1,1))}
)
kmesh = lat.get_kmesh((10,10,1))
N = kmesh.dims
print("type N:", type(N))
print("N:", N)
print("np.prod(N):", np.prod(N), type(np.prod(N)))
try:
    print("range(np.prod(N)):", range(np.prod(N)))
except Exception as e:
    print("Error:", e)
print("len(kmesh):", len(kmesh))
print("int(np.prod(N)):", int(np.prod(N)) if not isinstance(np.prod(N), type(N)) else "can't")
