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
from iminuit.cost import LeastSquares  # not used now, but fine to leave
from numpy import sqrt, pi, exp, real
from scipy.special import wofz


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

# Physics Model: Voigt (Breit-Wigner / Gaussian) + exponential background

def voigt(x, m0, gamma, sigma):
    z = ((x - m0) + 1j * gamma/2) / (sigma * np.sqrt(2))
    return real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

def model(x, A, m0, gamma, sigma, B, C, D):
    signal = A * voigt(x, m0, gamma, sigma)
    background = np.exp(B + C * x + D * (x**2))
    return signal + background


# Fit using iminuit — Poisson log-likelihood
x = bin_centers
y = hist


# Define per-bin weights to emphasize the Drell–Yan tail
weights = np.ones_like(x, dtype=float)
#weights = 1.0 + 4.0 * np.exp(-0.5 * ((x - 50)/15)**2)


# Example: upweight bins above 110 GeV by a factor of 5
tail_threshold = 110.0
tail_weight = 5.0

weights[x >= tail_threshold] = tail_weight
tail_weight = 8.0
weights[x <= tail_threshold] = tail_weight



def nll(A, m0, gamma, sigma, B, C, D):
    mu = model(x, A, m0, gamma, sigma, B, D, C)
    mu = np.clip(mu, 1e-12, None)

    # 2*NLL with per-bin weights to emphasize the tail
    return 2 * np.sum(weights * (mu - y * np.log(mu)))


m = Minuit(
    nll,
    A=50000,
    m0=91.0,
    gamma=2.5,
    sigma=2.0,
    B=5.0,
    C=-0.02,
    D=0.0
)

m.limits["gamma"] = (1.0, 5.0)
m.limits["sigma"] = (0.5, 5.0)

m.migrad()
m.hesse()

print("Fit results:")
print(m.values)
print("NLL =", m.fval)
print("ndof =", len(x) - m.nfit)
print("Number of datapoints used:", len(y))
print("Non‑empty bins:", np.sum(y > 0))
print("Number of events:", len(masses))


# Likelihood-ratio goodness-of-fit test
mu = model(x, *m.values)
mu = np.clip(mu, 1e-12, None)

y = hist.astype(float)

gof = 0.0
for yi, mui in zip(y, mu):
    if yi > 0:
        gof += 2 * (yi * np.log(yi / mui) - (yi - mui))
    else:
        gof += 2 * mui

ndof = len(y) - m.nfit
print(f"GOF statistic: {gof:.2f}")
print(f"ndof: {ndof}")
print(f"GOF/ndof: {gof/ndof:.2f}")


# Plot final fit
x_dense_plot = np.linspace(0, 200, 2000)
y_fit = model(x_dense_plot, *m.values)

plt.figure(figsize=(10,6))
plt.step(bin_centers, hist, where='mid', color='black')
plt.plot(x_dense_plot, y_fit, color='red', label='iminuit fit')
plt.xlabel("m$_{\ell\ell}$ [GeV]")
plt.ylabel("events / bin")
plt.title("Dilepton invariant mass spectrum")
plt.legend()
plt.savefig("plot.png")
plt.show()


# Extract Z boson properties
mZ = m.values["m0"]
mZ_err = m.errors["m0"]

gammaZ = m.values["gamma"]
gammaZ_err = m.errors["gamma"]

sigma_det = m.values["sigma"]
sigma_det_err = m.errors["sigma"]

print(f"Z mass: {mZ:.3f} ± {mZ_err:.3f} GeV")
print(f"Z width: {gammaZ:.3f} ± {gammaZ_err:.3f} GeV")
print(f"Detector resolution σ: {sigma_det:.3f} ± {sigma_det_err:.3f} GeV")


# Number of Z events and background (80–100 GeV)
x_dense = np.linspace(60, 120, 2000)

A = m.values["A"]
m0 = m.values["m0"]
gamma = m.values["gamma"]
sigma = m.values["sigma"]
B = m.values["B"]
C = m.values["C"]

signal_dense = A * voigt(x_dense, m0, gamma, sigma)
background_dense = np.exp(B + C * x_dense)

sig_mask = (x_dense >= 80) & (x_dense <= 100)

N_signal = np.trapz(signal_dense[sig_mask], x_dense[sig_mask])
print(f"Z signal yield (80–100 GeV): {N_signal:.1f} events")

N_background = np.trapz(background_dense[sig_mask], x_dense[sig_mask])
print(f"Background yield (80–100 GeV): {N_background:.1f} events")

# Purity and significance
purity = N_signal / (N_signal + N_background)
significance = N_signal / np.sqrt(N_background)

print(f"Purity S/(S+B): {purity:.4f}")
print(f"Significance S/sqrt(B): {significance:.2f}")

print("\n--- Z Boson Extraction Summary ---")
print(f"Z mass m0: {m0:.3f} GeV")
print(f"Z width gamma: {gamma:.3f} GeV")
print(f"Detector resolution sigma: {sigma:.3f} GeV")
print(f"Signal yield (80–100 GeV): {N_signal:.1f}")
print(f"Background yield (80–100 GeV): {N_background:.1f}")
print(f"Purity: {purity:.4f}")
print(f"Significance: {significance:.2f}")
