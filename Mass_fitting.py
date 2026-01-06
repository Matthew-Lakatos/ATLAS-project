import urllib.request
import pandas as pd
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import awkward as ak
import vector

import atlasopenmagic as atom

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


atom.set_release('2025e-13tev-beta')
skim = '2muons'
files_list = atom.get_urls('data', skim, protocol='https', cache=True)


def calc_mll(lep_pt, lep_eta, lep_phi, lep_e):
    p4 = vector.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "e": lep_e})
    return (p4[:, 0] + p4[:, 1]).M

mass_list = []

for afile in files_list:
    print(f'Working on file {afile} ({files_list.index(afile)}/{len(files_list)})')

    tree = uproot.open(afile + ":analysis")
    numevents = tree.num_entries

    for data in tree.iterate(['lep_pt','lep_eta','lep_phi','lep_e'],
                             entry_stop=int(numevents * 0.5),
                             library="ak"):
        data['mll'] = calc_mll(data.lep_pt, data.lep_eta, data.lep_phi, data.lep_e)
        mass_list.append(data['mll'])

masses = ak.to_numpy(ak.flatten(mass_list))

bins = np.linspace(0, 200, 200)
hist, bin_edges = np.histogram(masses, bins=bins)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

plt.figure(figsize=(10,6))
plt.step(bin_centers, hist, where='mid', color='black')
plt.xlabel("m$_{\ell\ell}$ [GeV]")
plt.ylabel("Events / bin")
plt.title("Dilepton Invariant Mass Spectrum")

ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())


#  REGION SELECTION

def select_region(x, y, xmin, xmax):
    mask = (x >= xmin) & (x <= xmax)
    return x[mask], y[mask]

# Fit region: Z peak
x_fit, y_fit = select_region(bin_centers, hist, 80, 100)

# Validation region: high-mass tail
x_test, y_test = select_region(bin_centers, hist, 110, 150)


#  MACHINE LEARNING FIT

ml_model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(hidden_layer_sizes=(20,20),
                         activation='relu',
                         alpha=1e-3,
                         max_iter=5000,
                         random_state=42))
])

ml_model.fit(x_fit.reshape(-1,1), y_fit)


# VALIDATING

y_pred = ml_model.predict(x_test.reshape(-1,1))
chi2 = np.sum((y_test - y_pred)**2 / (y_pred + 1e-6))
print("Validation χ² =", chi2)# CHI SQUARED TEST FOR MODEL ACCURACY


#  OVERLAY FIT ON ORIGINAL PLOT


x_dense = np.linspace(0, 200, 500)
y_dense = ml_model.predict(x_dense.reshape(-1,1))

plt.plot(x_dense, y_dense, color='red', label="ML Fit")
plt.legend()

plt.show()
