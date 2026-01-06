Go into the source of this page to copy code

HELPFUL LINKS:
https://hub.gesis.mybinder.org/user/atlas-outreach--ection-opendata-iyzqocta/doc/tree/13-TeV-examples/uproot_python/Find_the_Z.ipynb
https://opendata.atlas.cern/docs/videotutorials/overview
https://opendata.atlas.cern/docs/atlasopenmagic
https://opendata.atlas.cern/docs/documentation/data_format/ntuple

CODE SNIPPETS:

To set up the environment:


#install required packages


pip install atlasopenmagic;
pip install uproot;
pip install vector;
pip install requests;
pip install aiohttp;
pip install pandas;
pip install matplotlib;
pip install scikit-learn;



Import dependancies:


import urllib.request # for downloading files
import pandas as pd # to store data as dataframes
import numpy as np # for numerical calculations such as histogramming
import uproot # to read .root files as dataframes
import matplotlib.pyplot as plt # for plotting
from matplotlib.ticker import MaxNLocator,AutoMinorLocator # for minor ticks
import awkward as ak # for handling complex and nested data structures efficiently
import vector # For convenient 4-vector manipulatio
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
