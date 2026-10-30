import os

# using the plain image backend so tests never try to open a window
os.environ["MPLBACKEND"] = "Agg"
