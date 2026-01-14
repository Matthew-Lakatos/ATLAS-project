import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import atlasopenmagic as atom

# Load ATLAS Open Data
atom.set_release('2025e-13tev-beta')
skim = '1LMET30'
files_list = atom.get_urls('data', skim, protocol='https', cache=True)

# Transverse mass function
def transverse_mass(pt_lep, met, dphi):
    return np.sqrt(2 * pt_lep * met * (1 - np.cos(dphi)))


# Event loop
mT_list = []

for afile in files_list:
    print(f'working on file {afile}')
    tree = uproot.open(afile + ":analysis")
    numevents = tree.num_entries

    for data in tree.iterate(
        ['lep_pt','lep_phi','met','met_phi'],
        entry_stop=int(numevents * 0.5),
        library="ak"
    ):
        lep_pt  = data['lep_pt'][:,0]
        lep_phi = data['lep_phi'][:,0]

        met     = data['met']
        met_phi = data['met_phi']

        dphi = lep_phi - met_phi

        mT = transverse_mass(lep_pt, met, dphi)
        mT_list.append(mT)


mT_all = ak.to_numpy(ak.flatten(mT_list))


# Histogram
bins = np.linspace(0, 200, 200)
hist, bin_edges = np.histogram(mT_all, bins=bins)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

plt.figure(figsize=(10,6))
plt.step(bin_centers, hist, where='mid', color='black')
plt.xlabel("Transverse Mass $m_T$ [GeV]")
plt.ylabel("Events / bin")
plt.title("W Boson Transverse Mass Spectrum (1LMET30 Skim)")

ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

plt.show()
plt.savefig("plot.png")
