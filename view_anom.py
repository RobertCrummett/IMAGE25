# https://docs.simpeg.xyz/latest/content/tutorials/04-magnetics/plot_2a_magnetics_induced.html#sphx-glr-content-tutorials-04-magnetics-plot-2a-magnetics-induced-py
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from simpeg.utils import plot2Ddata

from forward import load_survey_results

from pathlib import Path

declination = 0
inclination = 90

receiver_locations, survey_results, params = \
        load_survey_results(declination, inclination)

fig = plt.figure(figsize=(6, 5))
v_max = np.max(np.abs(survey_results))

ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.85])
plot2Ddata(
    receiver_locations,
    survey_results,
    ax=ax1,
    ncontour=10,
    clim=(-v_max, v_max),
    contourOpts={"cmap": "bwr"},
)
ax1.set_title(f"TMI: Inclination {inclination}, Declination {declination}")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")

ax2 = fig.add_axes([0.87, 0.1, 0.03, 0.85])
norm = mpl.colors.Normalize(vmin=-np.max(np.abs(survey_results)), vmax=np.max(np.abs(survey_results)))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr
)
cbar.set_label("$nT$", rotation=270, labelpad=15, size=12)

plt.show()
