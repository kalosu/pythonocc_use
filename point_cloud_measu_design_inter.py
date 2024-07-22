##In this script, we want to use the point clouds found from the ICP algorithm in order
##to find the required compensation terms.
##As we mentioned before, we want to restrict the evaluation points to those defined by the design surface
##so that we avoid introducing weird artifacts towards the edges of the surface.
##This would basically mean that we would compensate for the regions that are valid within the measurement data.
import open3d as o3d
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline,GeomAPI_PointsToBSplineSurface
from OCC.Core.Approx import  Approx_Centripetal,Approx_IsoParametric,Approx_ChordLength
from OCC.Core.TColgp import TColgp_Array1OfPnt,TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.GeomAbs import GeomAbs_C2,GeomAbs_C3
from OCC.Display.SimpleGui import init_display
from OCC.Core.STEPControl import STEPControl_Writer,STEPControl_AsIs,STEPControl_Controller
from OCC.Core.Interface import Interface_Static
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface,shapeanalysis
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf



def points_to_surf(p,name):
    array = TColgp_Array2OfPnt(1,p.shape[1],1,p.shape[2])
    for i in range(0,p.shape[1],1):
        for j in range(0,p.shape[2],1):
            # point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j]-p_center) ##Original - We shift the points so that the apex is at (0,0,0)
            point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j])
            array.SetValue(i+1,j+1,point_to_add)
    print ("Surface creation")
##Least-square based
    bspl_surface = GeomAPI_PointsToBSplineSurface(array,Approx_IsoParametric,3,8,GeomAbs_C2,1e-1)#.Interpolate(array) ###For fitting

##Interpolation based
    # bspl_surface = GeomAPI_PointsToBSplineSurface()#.Interpolate(array)
    # bspl_surface.Interpolate(array)


    face_builder = BRepBuilderAPI_MakeFace(bspl_surface.Surface(),1e-6).Shape()
    display.DisplayShape(bspl_surface.Surface(),update=True)
    # for i in range(0,p.shape[1],1):
    #     for j in range(0,p.shape[2],1):
            # point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j]-p_center)
            # point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j])
            # display.DisplayShape(point_to_add,update=False)
    # display.Repaint()
    return bspl_surface.Surface()

filepath ="D:\\Confocal_measurements\\beam_shaping_project\\q_surface_40x_ip_visio_20240628\\structure_3_LP_84_83_ss_70k\\adjusted_area_adjusted_orientation_confocal\\"


# source_pc = "source_pc.xyz"
source_pc = "q_surf_measurement_cloud_compare_xy_trans_z_rot.xyz"
# target_pc = "target_pc.xyz"
target_pc = "q_surf_design_cloud_compare_xy_trans_z_rot.xyz"

pcd_design = o3d.io.read_point_cloud(filepath+target_pc)
pcd_measu = o3d.io.read_point_cloud(filepath+source_pc)


# o3d.visualization.draw_geometries_with_editing([pcd_design])

design_points = np.asarray(pcd_design.points)
measu_points = np.asarray(pcd_measu.points)
print ("Shape of the point cloud arrays")
print (design_points.shape)
print (measu_points.shape)

pcd_design.paint_uniform_color([1,0,0])
pcd_measu.paint_uniform_color([0.8,0.8,0.8])

print ("We display first the point cloud from the design")
o3d.visualization.draw_geometries([pcd_design])

print ("We display first the point cloud from the measurement")
o3d.visualization.draw_geometries([pcd_measu])



print ("We will now try to do the interpolation on the measurement surface")

measu_points_inter = griddata((measu_points[:,0],measu_points[:,1]),measu_points[:,2],(design_points[:,0],design_points[:,1]),method='cubic')

measu_points_inter_filtered = np.where(np.isnan(measu_points_inter),design_points[:,2],measu_points_inter)

fig = plt.figure()
ax1 = fig.add_subplot(121)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax1.tricontourf(design_points[:,0].flatten(), design_points[:,1].flatten(), measu_points_inter_filtered.flatten(), levels=50, cmap='hsv')
plt.colorbar(im, cax=cax, orientation='vertical')
plt.title("Measured surface (interpolation on design xy coordinates)")

# fig = plt.figure()
ax2 = fig.add_subplot(122)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
im2 = ax2.tricontourf(design_points[:,0].flatten(), design_points[:,1].flatten(), design_points[:,2].flatten(), levels=50, cmap='hsv')
plt.colorbar(im2, cax=cax2, orientation='vertical')
plt.title("Surface original points")
# # ax.scatter(p_tt_deg*np.cos(p_pp),p_tt_deg*np.sin(p_pp),color='blue')
# ax.set_aspect('equal')
# ax.axis('off')
# ax_polar = fig.add_axes(ax.get_position(),polar=True)
# ax_polar.set_facecolor('none')
# ax_polar.set_ylim(0,np.max(p_tt_deg))
plt.show()

print("We now estimate the point-wise difference between the design and the interpolated measurement data")

surf_diff = design_points[:,2] - measu_points_inter_filtered

print ("We now try to apply a filter to the surface differences to separate the macro deviations from the local deviations")

surf_diff_filtered = gaussian_filter(surf_diff,sigma=5.0)

high_frequency_surf_diff = surf_diff - surf_diff_filtered

fig = plt.figure()
ax1 = fig.add_subplot(121)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax1.tricontourf(design_points[:,0].flatten(), design_points[:,1].flatten(), surf_diff_filtered.flatten(), levels=50, cmap='hsv')
plt.colorbar(im, cax=cax, orientation='vertical')
plt.title("Surface deviations - Low frequency contributions")

# fig = plt.figure()
ax2 = fig.add_subplot(122)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
im2 = ax2.tricontourf(design_points[:,0].flatten(), design_points[:,1].flatten(), high_frequency_surf_diff.flatten(), levels=50, cmap='hsv')
plt.colorbar(im2, cax=cax2, orientation='vertical')
plt.title("Surface deviations - High frequency contributions")


fig = plt.figure()
ax1 = fig.add_subplot()
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax1.tricontourf(design_points[:,0], design_points[:,1], surf_diff, levels=50, cmap='hsv')
plt.colorbar(im, cax=cax, orientation='vertical')
plt.title("Design surface points - Measurement (interpolation) surface points")
plt.show()

surf_compensated = design_points[:,2] + surf_diff_filtered ##We add the computed difference to the original surface points

fig = plt.figure()
ax1 = fig.add_subplot()
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax1.tricontourf(design_points[:,0], design_points[:,1], surf_compensated, levels=50, cmap='hsv')
plt.colorbar(im, cax=cax, orientation='vertical')
plt.title("Compensated surface")
plt.show()

pcd_compensated = o3d.geometry.PointCloud()


points_compensated = np.stack((design_points[:,0].flatten(), design_points[:,1].flatten(), surf_compensated.flatten()), -1)
pcd_compensated.points = o3d.utility.Vector3dVector(points_compensated)

o3d.visualization.draw_geometries_with_editing([pcd_compensated])

Nx = int(np.sqrt(points_compensated.shape[0]))

display, start_display, add_menu, add_function_to_menu = init_display()

print ("shape of points_compensated")
print (points_compensated.shape)
points_compensated = points_compensated.swapaxes(0,1)
print (points_compensated.shape)

q_surf = points_to_surf(points_compensated.reshape(3, Nx, Nx), 'name')

start_display()








