# imports
import urllib.request
import pandas as pd
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import awkward as ak
import vector

import atlasopenmagic as atom

# scikit-learn imports
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression



# set up atlas open data
atom.set_release('2025e-13tev-beta')
skim = '2muons'
files_list = atom.get_urls('data', skim, protocol='https', cache=True)


# dilepton invariant mass function
def calc_mll(lep_pt, lep_eta, lep_phi, lep_e):
    p4 = vector.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "e": lep_e})
    return (p4[:, 0] + p4[:, 1]).M


# load masses from data
mass_list = []

for afile in files_list:
    print(f'working on file {afile} ({files_list.index(afile)}/{len(files_list)})')

    tree = uproot.open(afile + ":analysis")
    numevents = tree.num_entries

    for data in tree.iterate(['lep_pt','lep_eta','lep_phi','lep_e'],
                             entry_stop=int(numevents * 0.5),
                             library="ak"):
        data['mll'] = calc_mll(data.lep_pt, data.lep_eta, data.lep_phi, data.lep_e)
        mass_list.append(data['mll'])
    break


# flatten mass array
masses = ak.to_numpy(ak.flatten(mass_list))


# histogram
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


# region selection helper
def select_region(x, y, xmin, xmax):
    mask = (x >= xmin) & (x <= xmax)
    return x[mask], y[mask]


# choose fit and validation regions
x_fit, y_fit = select_region(bin_centers, hist, 80, 100)   # z peak
x_test, y_test = select_region(bin_centers, hist, 110, 150)  # high-mass tail

# define compound fit function
def compound_model(x, A, m0, gamma, B, C):
    bw = A * (gamma**2 / ((x - m0)**2 + (gamma**2)/4))
    bkg = np.exp(B + C * x)
    return bw + bkg


# build background model
bkg_model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(hidden_layer_sizes=(10,10),
                         activation='tanh',
                         alpha=1e-2,
                         learning_rate_init=1e-3,
                         max_iter=8000,
                         random_state=1))
])

# smooth resonance model
res_model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(hidden_layer_sizes=(15,),
                         activation='tanh',
                         alpha=1e-2,
                         learning_rate_init=1e-3,
                         max_iter=8000,
                         random_state=2))
])

# --- Fit Breit-Wigner amplitude using scikit-learn LinearRegression ---

def breit_wigner_shape(x, m0=91.2, gamma=2.5):
    return gamma**2 / ((x - m0)**2 + (gamma**2)/4)

# Prepare training data for amplitude fit (use the peak region)
X_bw = breit_wigner_shape(x_peak).reshape(-1, 1)
y_bw = y_peak

# Fit amplitude A using linear regression WITHOUT intercept
bw_regressor = LinearRegression(fit_intercept=False)
bw_regressor.fit(X_bw, y_bw)

A_fit = bw_regressor.coef_[0]

def breit_wigner(x, A=A_fit, m0=91.2, gamma=2.5):
    return A * (gamma**2 / ((x - m0)**2 + (gamma**2)/4))


# training regions
x_low, y_low = select_region(bin_centers, hist, 40, 70)
x_peak, y_peak = select_region(bin_centers, hist, 80, 100)
x_high, y_high = select_region(bin_centers, hist, 110, 150)

# train background on sidebands
x_bkg = np.concatenate([x_low, x_high])
y_bkg = np.concatenate([y_low, y_high])
bkg_model.fit(x_bkg.reshape(-1,1), y_bkg)

# train resonance on peak
res_model.fit(x_peak.reshape(-1,1), np.log1p(y_peak))

# combined prediction
def combined_predict(x):
    b = bkg_model.predict(x.reshape(-1,1))   # no expm1 here anymore
    r = breit_wigner(x)
    y = b + r
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)



# validation
x_val, y_val = select_region(bin_centers, hist, 150, 180)
y_val_pred = combined_predict(x_val)
chi2 = np.sum((y_val - y_val_pred)**2 / (y_val_pred + 1e-6))
print("validation x^2 =", chi2)

# plot
x_dense = np.linspace(0, 200, 500)
y_dense = combined_predict(x_dense)
plt.plot(x_dense, y_dense, color='red', label='compound fit')
plt.legend()
plt.savefig("plot.png")
