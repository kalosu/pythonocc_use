import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

filepath = "D:\\Confocal_measurements\\beam_shaping_project\\q_surface_40x_ip_visio_20240628\\structure_3_LP_84_83_ss_70k\\offset_calculation_from_Joshuas_ICP_reversed_order\\"



filename_design = "q_surface_ref_minus q_surf_w_subs_str_3_cropped_adapted_area_raw0.xyz"
filename_measu = "q_surface_ref_minus q_surf_w_subs_str_3_cropped_adapted_area_raw1.xyz"

filename_reference = "q_surface_ref_minus q_surf_w_subs_str_3_cropped_adapted_area.xyz"




pcd_design = o3d.io.read_point_cloud(filepath+filename_design)
pcd_measu = o3d.io.read_point_cloud(filepath+filename_measu)
pcd_compensation = o3d.io.read_point_cloud(filepath+filename_reference)


design_points = np.asarray(pcd_design.points)

compensation_points = np.asarray(pcd_compensation.points)


measu_points = np.asarray(pcd_measu.points)

fig = plt.figure()
ax1 = fig.add_subplot(121)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax1.tricontourf(measu_points[:,0], measu_points[:,1], measu_points[:,2], levels=50, cmap='hsv')
plt.colorbar(im, cax=cax, orientation='vertical')
plt.title("Measured surface profile")

# fig = plt.figure()
ax2 = fig.add_subplot(122)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
im2 = ax2.tricontourf(design_points[:,0], design_points[:,1], design_points[:,2], levels=50, cmap='hsv')
plt.colorbar(im2, cax=cax2, orientation='vertical')
plt.title("Surface original points")
# # ax.scatter(p_tt_deg*np.cos(p_pp),p_tt_deg*np.sin(p_pp),color='blue')
# ax.set_aspect('equal')
# ax.axis('off')
# ax_polar = fig.add_axes(ax.get_position(),polar=True)
# ax_polar.set_facecolor('none')
# ax_polar.set_ylim(0,np.max(p_tt_deg))
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot()
# im = ax.tricontourf(compensation_points[:,0],compensation_points[:,1],compensation_points[:,2],levels=50,cmap='hsv')
# plt.colorbar(im)
# # ax.scatter(p_tt_deg*np.cos(p_pp),p_tt_deg*np.sin(p_pp),color='blue')
# ax.set_aspect('equal')
# ax.axis('off')
# ax_polar = fig.add_axes(ax.get_position(),polar=True)
# ax_polar.set_facecolor('none')
# ax_polar.set_ylim(0,np.max(p_tt_deg))
# plt.show()

compensated_points = design_points[:,2] + compensation_points[:,2]


##We create a new point cloud for the compensated surface
pcd_compensated = o3d.geometry.PointCloud()
points_comp = np.stack((design_points[:,0].flatten(), design_points[:,1].flatten(), compensated_points.flatten()), -1)
pcd_compensated.points = o3d.utility.Vector3dVector(points_comp)

pcd_design.paint_uniform_color([1, 0, 0])
pcd_measu.paint_uniform_color([0.8, 0.8, 0.8])

o3d.visualization.draw_geometries([pcd_design,pcd_measu])

fig = plt.figure()
ax = fig.add_subplot()
im = ax.tricontourf(design_points[:,0],design_points[:,1],compensated_points,levels=50,cmap='hsv')
plt.colorbar(im)
# # ax.scatter(p_tt_deg*np.cos(p_pp),p_tt_deg*np.sin(p_pp),color='blue')
# ax.set_aspect('equal')
# ax.axis('off')
# ax_polar = fig.add_axes(ax.get_position(),polar=True)
# ax_polar.set_facecolor('none')
# ax_polar.set_ylim(0,np.max(p_tt_deg))
plt.show()


