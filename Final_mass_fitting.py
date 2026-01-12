# Imports
import urllib.request
import pandas as pd
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import awkward as ak
import vector

import atlasopenmagic as atom

from iminuit import Minuit
from scipy.special import wofz
from numpy import real

# Load ATLAS Open Data
atom.set_release('2025e-13tev-beta')
skim = '2muons'
files_list = atom.get_urls('data', skim, protocol='https', cache=True)

def calc_mll(lep_pt, lep_eta, lep_phi, lep_e):
    p4 = vector.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "e": lep_e})
    return (p4[:, 0] + p4[:, 1]).M

mass_list = []

for afile in files_list:
    print(f'working on file {afile}')
    tree = uproot.open(afile + ":analysis")
    numevents = tree.num_entries

    for data in tree.iterate(['lep_pt','lep_eta','lep_phi','lep_e'],
                             entry_stop=int(numevents * 0.5),
                             library="ak"):
        data['mll'] = calc_mll(data.lep_pt, data.lep_eta, data.lep_phi, data.lep_e)
        mass_list.append(data['mll'])
    break

masses = ak.to_numpy(ak.flatten(mass_list))

# Histogram
bins = np.linspace(0, 200, 200)
hist, bin_edges = np.histogram(masses, bins=bins)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

plt.figure(figsize=(10,6))
plt.step(bin_centers, hist, where='mid', color='black')
plt.xlabel("m$_{\ell\ell}$ [GeV]")
plt.ylabel("events / bin")
plt.title("dilepton invariant mass spectrum")

ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# -----------------------------
# Z-ONLY PRECISION FIT SECTION
# -----------------------------

# Voigt profile (Breit-Wigner ⊗ Gaussian)
def voigt(x, m0, gamma, sigma):
    z = ((x - m0) + 1j * gamma/2) / (sigma * np.sqrt(2))
    return real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

# Restrict to Z peak region
fit_mask = (bin_centers >= 80) & (bin_centers <= 100)
x_fit = bin_centers[fit_mask]
y_fit = hist[fit_mask]

# Z-only model
def model_Zonly(x, A, m0, gamma, sigma, p1, p2):
    V = voigt(x, m0, gamma, sigma)
    poly = 1 + p1*(x - m0) + p2*(x - m0)**2
    return A * V * poly


# Poisson NLL
def nll_Zonly(A, m0, gamma, sigma, p1, p2):
    mu = model_Zonly(x_fit, A, m0, gamma, sigma, p1, p2)
    mu = np.clip(mu, 1e-12, None)
    return 2 * np.sum(mu - y_fit * np.log(mu))


# Minuit fit
mZ = Minuit(
    nll_Zonly,
    A=50000,
    m0=91.0,
    gamma=2.5,
    sigma=2.0,
    p1=0.0,
    p2=0.0
)

mZ.limits["gamma"] = (1.0, 5.0)
mZ.limits["sigma"] = (0.5, 5.0)
mZ.limits["p1"] = (-0.1, 0.1)
mZ.limits["p2"] = (-0.01, 0.01)


mZ.limits["gamma"] = (1.0, 5.0)
mZ.limits["sigma"] = (0.5, 5.0)

mZ.migrad()
mZ.hesse()

print("\n=== Z-Only Fit Results ===")
print(mZ.values)

# Goodness of fit
mu_fit = model_Zonly(x_fit, *mZ.values)
mu_fit = np.clip(mu_fit, 1e-12, None)

gof = 0.0
for yi, mui in zip(y_fit, mu_fit):
    if yi > 0:
        gof += 2 * (yi * np.log(yi / mui) - (yi - mui))
    else:
        gof += 2 * mui

ndof = len(y_fit) - mZ.nfit
print(f"GOF: {gof:.2f}")
print(f"ndof: {ndof}")
print(f"GOF/ndof: {gof/ndof:.2f}")

# Plot the Z-only fit
x_dense = np.linspace(80, 100, 2000)
y_dense = model_Zonly(x_dense, *mZ.values)

plt.figure(figsize=(10,6))
plt.step(bin_centers, hist, where='mid', color='black')
plt.plot(x_dense, y_dense, color='red', label='Z-only fit')
plt.xlabel("m$_{\ell\ell}$ [GeV]")
plt.ylabel("events / bin")
plt.title("Z-only Voigt Fit (80–100 GeV)")
plt.legend()
plt.show()

# Extract Z boson properties
mZ_val = mZ.values["m0"]
mZ_err = mZ.errors["m0"]

gammaZ = mZ.values["gamma"]
gammaZ_err = mZ.errors["gamma"]

sigma_det = mZ.values["sigma"]
sigma_det_err = mZ.errors["sigma"]

print(f"Z mass: {mZ_val:.3f} ± {mZ_err:.3f} GeV")
print(f"Z width: {gammaZ:.3f} ± {gammaZ_err:.3f} GeV")
print(f"Detector resolution σ: {sigma_det:.3f} ± {sigma_det_err:.3f} GeV")

# Signal yield in 80–100 GeV
signal_dense = mZ.values["A"] * voigt(x_dense, mZ_val, gammaZ, sigma_det)
sig_mask = (x_dense >= 80) & (x_dense <= 100)

N_signal = np.trapz(signal_dense[sig_mask], x_dense[sig_mask])
print(f"Z signal yield (80–100 GeV): {N_signal:.1f} events")

print("\n--- Z Boson Extraction Summary ---")
print(f"Z mass m0: {mZ_val:.3f} GeV")
print(f"Z width gamma: {gammaZ:.3f} GeV")
print(f"Detector resolution sigma: {sigma_det:.3f} GeV")
print(f"Signal yield (80–100 GeV): {N_signal:.1f}")
