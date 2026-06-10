import numpy as np

def fermi(e, beta, mu):
    # to avoid overflow
    x = beta * (e - mu)
    return np.where(x > 100, 0.0, np.where(x < -100, 1.0, 1.0 / (np.exp(x) + 1.0)))

def filling(band, beta, mu):
    return np.sum(fermi(band, beta, mu)) / len(band)

band = np.linspace(-0.7, 0.7, 1000)
beta = 10.0 # moderate temp
mu_min = np.min(band)
mu_max = np.max(band)
target = 0.01
print("Filling at mu_min:", filling(band, beta, mu_min))

# how to find good bounds
def find_bounds(target, beta, band):
    mu_low = np.min(band) - 1.0
    mu_high = np.max(band) + 1.0
    while filling(band, beta, mu_low) > target:
        mu_low -= 1.0
    while filling(band, beta, mu_high) < target:
        mu_high += 1.0
    return mu_low, mu_high

mu_low, mu_high = find_bounds(target, beta, band)
print("Bounds:", mu_low, mu_high)
print("Filling at mu_low:", filling(band, beta, mu_low))
print("Filling at mu_high:", filling(band, beta, mu_high))

