HELPFUL LINKS:
https://hub.gesis.mybinder.org/user/atlas-outreach--ection-opendata-iyzqocta/doc/tree/13-TeV-examples/uproot_python/Find_the_Z.ipynb
https://opendata.atlas.cern/docs/videotutorials/overview
https://opendata.atlas.cern/docs/atlasopenmagic

CODE SNIPPETS:

To set up the environment:

#install required packages
import sys
%pip install atlasopenmagic
from atlasopenmagic import install_from_environment
install_from_environment

Import dependancies:

import urllib.request # for downloading files
import pandas as pd # to store data as dataframes
import numpy as np # for numerical calculations such as histogramming
import uproot # to read .root files as dataframes
import matplotlib.pyplot as plt # for plotting
from matplotlib.ticker import MaxNLocator,AutoMinorLocator # for minor ticks
import awkward as ak # for handling complex and nested data structures efficiently
import vector # For convenient 4-vector manipulation

To retrieve data from the project:

# Import the atlasopenmagic package so that we can use it
import atlasopenmagic as atom
# Make sure that we are using the newest release of Education and Outreach open data
atom.set_release('2025e-13tev-beta')
# We will use a skim (pre-selection) of the data so that we start from events
# with exactly two muons
skim = '2muons'
# And now we can get the files we want; note that we use "cache=True" to copy
# the files locally rather than streaming them
files_list = atom.get_urls('data', skim, protocol='https', cache=True)

To calculate invariant dilepton mass:

# calculate dilepton invariant mass
def calc_mll(lep_pt,lep_eta,lep_phi,lep_e):
    p4 = vector.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "e": lep_e})
    invariant_mass = (p4[:, 0] + p4[:, 1]).M # .M calculates the invariant mass
    return invariant_mass

Processing mass data:

mass_list = [] # This list will hold all our output masses

# Loop through all the files that we have
for afile in files_list:
    print(f'Working on file {afile} ({files_list.index(afile)}/{len(files_list)})')

    tree = uproot.open(afile+":analysis") # open the file, and the tree called mini
    numevents = tree.num_entries # number of events

    for data in tree.iterate(['lep_pt','lep_eta','lep_phi','lep_e'],
                             entry_stop=numevents*0.5, # stop after fraction of events we want to process
                             library="ak"):
        # calculation of 2-lepton invariant mass
        data['mll'] = calc_mll(data.lep_pt, data.lep_eta, data.lep_phi, data.lep_e)
        # Keep the mll as a flat array
        mass_list.append( data['mll'] )
    # For simplicity and speed, we'll just work with the first file to start
    # Stop here!
    break

Plotting data:

# In case we ran over multiple files / batches, merge them together
full_mass_list = ak.concatenate(mass_list)

bin_edges = np.arange(start=70, # The interval includes this value
                      stop=110, # The interval doesn't include this value
                      step=1 ) # Spacing between values
bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2 # central values of each bin

# histogram the data
data_x,_ = np.histogram(full_mass_list, bins=bin_edges )

# statistical error on the data
data_x_errors = np.sqrt(data_x)

# get current axes
main_axes = plt.gca()

# plot the data points
main_axes.errorbar(x=bin_centres,
                   y=data_x,
                   yerr=data_x_errors,
                   fmt='ko',  # 'k' means black and 'o' is for circles
                   label = 'Data')

# set the axis tick parameters for the main axes
main_axes.tick_params(which='both', # ticks on both x and y axes
                        direction='in', # Put ticks inside and outside the axes
                        top=True, # draw ticks on the top axis
                        right=True ) # draw ticks on right axis

# x-axis label
main_axes.set_xlabel('Di-lepton Mass [GeV]', fontsize=13, x=1, horizontalalignment='right')

# y-axis label
main_axes.set_ylabel('Events', fontsize=13, y=1, horizontalalignment='right')

# make the y-axis log scale
main_axes.set_yscale('log')

# add minor ticks on x-axis for main axes
main_axes.xaxis.set_minor_locator( AutoMinorLocator() )

# draw the legend
main_axes.legend( frameon=False ); # no box around the l
