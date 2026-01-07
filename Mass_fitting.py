# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import urllib.request
import pandas as pd
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import awkward as ak
import vector

import atlasopenmagic as atom

# iminuit for proper physics fitting

from iminuit import Minuit
from iminuit.cost import LeastSquares

# ------------------------------------------------------------
# Load ATLAS Open Data
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Histogram
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Physics Model: Voigt (BW ⊗ Gaussian) + exponential background
# ------------------------------------------------------------
from numpy import sqrt, pi, exp
from numpy import real
from scipy.special import wofz

def voigt(x, m0, gamma, sigma):
    z = ((x - m0) + 1j * gamma/2) / (sigma * np.sqrt(2))
    return real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

def model(x, A, m0, gamma, sigma, B, C):
    signal = A * voigt(x, m0, gamma, sigma)
    background = np.exp(B + C*x)
    return signal + background

# ------------------------------------------------------------
# Fit using iminuit
# ------------------------------------------------------------
x = bin_centers
y = hist
yerr = np.sqrt(hist + 1)

least_squares = LeastSquares(x, y, yerr, model)

m = Minuit(least_squares,
           A=50000,
           m0=91.0,
           gamma=2.5,
           sigma=2.0,
           B=5.0,
           C=-0.02)

m.limits["gamma"] = (1.0, 5.0)
m.limits["sigma"] = (0.5, 5.0)

m.migrad()
m.hesse()

print("Fit results:")
print(m.values)
print("chi2 =", m.fval)
print("ndof =", len(x) - m.nfit)
print("chi2/ndof =", m.fval / (len(x) - m.nfit))

# ------------------------------------------------------------
# Plot final fit
# ------------------------------------------------------------
x_dense = np.linspace(0, 200, 2000)
y_fit = model(x_dense, *m.values)

plt.figure(figsize=(10,6))
plt.step(bin_centers, hist, where='mid', color='black')
plt.plot(x_dense, y_fit, color='red', label='iminuit fit')
plt.xlabel("m$_{\ell\ell}$ [GeV]")
plt.ylabel("events / bin")
plt.title("Dilepton invariant mass spectrum")
plt.legend()
plt.show()
