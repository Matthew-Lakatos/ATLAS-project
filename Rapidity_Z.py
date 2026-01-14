import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import atlasopenmagic as atom

atom.set_release('2025e-13tev-beta')
skim = '2muons'   # Z boson skim
files_list = atom.get_urls('data', skim, protocol='https', cache=True)

# Use only the first 3 files
files_list = files_list[:3]

#rapidity function
def rapidity(E, pz):
    return 0.5 * np.log((E + pz) / (E - pz))

#event loop
y_list = []

for afile in files_list:
    print(f"Processing {afile}")
    tree = uproot.open(afile + ":analysis")
    
    for data in tree.iterate(
        ['lep_pt', 'lep_eta', 'lep_phi', 'lep_E'],
        library="ak"
    ):
        # Two leptons per event
        pt = data['lep_pt']
        eta = data['lep_eta']
        phi = data['lep_phi']
        E   = data['lep_E']

        # Compute px, py, pz for each lepton
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)

        # Dilepton system
        px_Z = px[:,0] + px[:,1]
        py_Z = py[:,0] + py[:,1]
        pz_Z = pz[:,0] + pz[:,1]
        E_Z  = E[:,0]  + E[:,1]

        # Rapidity of the Z boson
        y_Z = rapidity(E_Z, pz_Z)
        y_list.append(y_Z)

# Flatten
y_all = ak.to_numpy(ak.flatten(y_list))

#plot
plt.figure(figsize=(10,6))

bins = np.linspace(-3, 3, 60)
hist, bin_edges = np.histogram(y_all, bins=bins)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

plt.step(bin_centers, hist, where='mid', color='black')
plt.xlabel("Rapidity $y_Z$")
plt.ylabel("Events / bin")
plt.title("Rapidity Distribution of the Z Boson (ATLAS Open Data)")

# Grid lines behind plot
ax = plt.gca()
ax.set_axisbelow(True)
ax.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.8)
ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

ax.xaxis.set_major_locator(MaxNLocator(integer=False))
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

plt.show()

