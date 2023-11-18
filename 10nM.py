import numpy as np
import matplotlib.pyplot as plt
from analysis import Txt_Reading
import pandas as pd
import os
from draw import Draw

folder = "10nM"
# f_MAX = [10, 15, 20, 25, 30, 35]
f_max = 10
reading = Txt_Reading(folder, f_max)

# file = reading.readTxt(number=7, N=18, ty='u', forced_reshaped=0, graph=True) # 10 reshape works best
if reading.finish:
    molecules, all_molecules_f, all_molecules_u = reading.seq_analysis_post_meta()


overall_images = True
single_molecules_images = True

plots = Draw(folder, f_max)

if overall_images:
    plots.final_plots(molecules, all_molecules_f, all_molecules_u)

if single_molecules_images:
    plots.plots_per_single_molecule(molecules, all_molecules_f, all_molecules_u)
        