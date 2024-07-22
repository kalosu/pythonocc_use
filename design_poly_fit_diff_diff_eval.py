###We want to use this script to evaluate the difference in the "polynomial fit - design points" difference.
##We know that between printed structures, there are variations in the printed structures.
##This can give us an estimate on the variations between prints for nominally the same printing parameters???

import numpy as np
from common_functions import confocal_data_read
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

ref_x, ref_y, ref_z = confocal_data_read("design_poly_fit_repro_1_z_shifted.txt")
repro_x, repro_y, repro_z = confocal_data_read("design_poly_fit_repro_2_z_shifted.txt")

##We calculate the pointwise difference between both difference distributions

z_diff = ref_z - repro_z

cmap = 'inferno'

fig = plt.figure()
ax1 = fig.add_subplot(121)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax1.tricontourf(ref_x.flatten(), ref_y.flatten(), z_diff.flatten(), levels=50, cmap=cmap)
plt.colorbar(im, cax=cax, orientation='vertical')
plt.title("Difference between polynomial fits - reproducibility eval")
#
# fig = plt.figure()
ax2 = fig.add_subplot(122)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
im2 = ax2.tricontour(ref_x.flatten(), ref_y.flatten(), z_diff.flatten(), levels=50, cmap=cmap)
plt.colorbar(im2, cax=cax2, orientation='vertical')
plt.title("Difference between polynomial fits - reproducibility eval")
plt.show()


