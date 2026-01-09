import numpy as np
import awkward as ak
import uproot
import vector
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

import atlasopenmagic as atom

vector.register_awkward()  # enable vector with awkward

# Setup and load 1to4lep skim
atom.set_release('2025e-13tev-beta')
skim = '2to4lep'
files_list = atom.get_urls('data', skim='2to4lep', protocol='https', cache=True)

# Helper: compute Collins–Soper cos(theta*) for a dilepton pair
def cos_theta_cs(lep1, lep2):
    """
    lep1, lep2: vector Lorentz objects (can be awkward arrays) for l- and l+ (order we'll enforce)
    Uses the standard CS formula in terms of lab-frame four-vectors.
    """

    # Total dilepton momentum
    q = lep1 + lep2

    # Invariant mass and transverse momentum of the pair
    M = q.mass
    qT = q.pt

    # Light-cone components of leptons
    # p^+ = (E + pz) / sqrt(2),  p^- = (E - pz) / sqrt(2)
    sqrt2 = np.sqrt(2.0)
    p1_plus  = (lep1.E + lep1.pz) / sqrt2
    p1_minus = (lep1.E - lep1.pz) / sqrt2
    p2_plus  = (lep2.E + lep2.pz) / sqrt2
    p2_minus = (lep2.E - lep2.pz) / sqrt2

    numerator = 2.0 * (p1_plus * p2_minus - p1_minus * p2_plus)
    denominator = M * np.sqrt(M**2 + qT**2)

    cos_theta = numerator / denominator

    # Sign convention: use sign of dilepton rapidity to approximate quark direction
    sign_y = np.sign(q.rapidity)
    # where rapidity is exactly zero, leave sign as +1
    sign_y = ak.where(sign_y == 0, 1.0, sign_y)

    return sign_y * cos_theta


# Loop over files and build OSSF pairs
mll_all = []
costh_all = []

for afile in files_list:
    print(f"Working on file {afile}")
    tree = uproot.open(afile + ":analysis")
    numevents = tree.num_entries

    # Branch names may need adjustment if different in this skim
    branches = [
        "lep_pt", "lep_eta", "lep_phi", "lep_e",
        "lep_charge",  # needed for OS pairs
        # "lep_type",  # if you want to restrict to e/μ; optional
    ]

    for data in tree.iterate(branches, entry_stop=numevents, library="ak"):
        lep_pt     = data["lep_pt"]
        lep_eta    = data["lep_eta"]
        lep_phi    = data["lep_phi"]
        lep_e      = data["lep_e"]
        lep_charge = data["lep_charge"]

        # Optional: if you want only electrons/muons, restrict here using lep_type, if available.
        # Example
        # lep_type = data["lep_type"]  # say 11 for e, 13 for mew
        # mask_em = (abs(lep_type) == 11) | (abs(lep_type) == 13)
        # lep_pt, lep_eta, lep_phi, lep_e, lep_charge = lep_pt[mask_em], ...

        # Build four-vectors for all leptons in the event
        leps = vector.zip({
            "pt": lep_pt,
            "eta": lep_eta,
            "phi": lep_phi,
            "E": lep_e,
        })

        # Also keep charges aligned
        charges = lep_charge

        # Build all lepton pairs per event
        pairs = ak.combinations(ak.local_index(leps), 2, fields=["i", "j"])

        if len(pairs) == 0:
            continue

        i = pairs["i"]
        j = pairs["j"]

        lep1 = leps[i]
        lep2 = leps[j]
        q1   = charges[i]
        q2   = charges[j]

        # Opposite-sign requirement
        os_mask = (q1 * q2) < 0

        lep1 = lep1[os_mask]
        lep2 = lep2[os_mask]
        q1   = q1[os_mask]
        q2   = q2[os_mask]

        # If you want only same-flavour (ee, μμ), and have lep_type, enforce that here:
        # t1 = lep_type[i][os_mask]
        # t2 = lep_type[j][os_mask]
        # sf_mask = (t1 == t2)
        # lep1, lep2, q1, q2 = lep1[sf_mask], lep2[sf_mask], q1[sf_mask], q2[sf_mask]

        if ak.count(lep1) == 0:
            continue

        # Ensure lep1 is the negatively charged lepton (convention for CS)
        swap = (q1 > 0)
        lep1_cs = ak.where(swap, lep2, lep1)
        lep2_cs = ak.where(swap, lep1, lep2)

        # Compute m_ll
        dilepton = lep1_cs + lep2_cs
        mll = dilepton.mass

        # Compute cos(theta*_CS)
        costh_cs = cos_theta_cs(lep1_cs, lep2_cs)

        # Flatten and store
        mll_all.append(ak.flatten(mll))
        costh_all.append(ak.flatten(costh_cs))

# Concatenate all arrays
mll_all = ak.to_numpy(ak.flatten(mll_all))
costh_all = ak.to_numpy(ak.flatten(costh_all))

print("Total OSSF pairs:", len(mll_all))


# Compute A_FB(m_ll)

# Choose mass range and binning
m_min, m_max = 40.0, 200.0

# You can tune bins: finer near Z, coarser in tails
# For now, uniform bins for simplicity
n_bins = 24
m_bins = np.linspace(m_min, m_max, n_bins + 1)

# Indices of forward/backward events
forward = costh_all > 0
backward = costh_all < 0

m_forward = mll_all[forward]
m_backward = mll_all[backward]

# Histograms of forward and backward in the same mass bins
N_F, _ = np.histogram(m_forward, bins=m_bins)
N_B, _ = np.histogram(m_backward, bins=m_bins)

# Total and A_FB
N_tot = N_F + N_B
A_FB = np.zeros_like(N_tot, dtype=float)
A_FB_err = np.zeros_like(N_tot, dtype=float)

# Avoid division by zero
mask = N_tot > 0
A_FB[mask] = (N_F[mask] - N_B[mask]) / N_tot[mask]

# Binomial error estimate: var(A_FB) ≈ (1 - A_FB^2) / N_tot
A_FB_err[mask] = np.sqrt((1.0 - A_FB[mask]**2) / N_tot[mask])

# Bin centers for plotting
m_centers = 0.5 * (m_bins[:-1] + m_bins[1:])

# Plot A_FB(m_ll)
plt.figure(figsize=(10,6))
plt.errorbar(m_centers, A_FB, yerr=A_FB_err, fmt='o', color='black', capsize=3)
plt.axhline(0.0, color='gray', linestyle='--')

plt.xlabel(r"$m_{\ell\ell}$ [GeV]")
plt.ylabel(r"$A_{\mathrm{FB}}$")
plt.title(r"Forward–Backward Asymmetry $A_{\mathrm{FB}}(m_{\ell\ell})$ (OSSF leptons)")

ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
