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

from sklearn.preprocessing import PolynomialFeatures,SplineTransformer
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel


def points_to_surf(p,name):
    array = TColgp_Array2OfPnt(1,p.shape[1],1,p.shape[2])
    for i in range(0,p.shape[1],1):
        for j in range(0,p.shape[2],1):
            # point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j]-p_center) ##Original - We shift the points so that the apex is at (0,0,0)
            point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j])
            array.SetValue(i+1,j+1,point_to_add)
    print ("Surface creation")
##Least-square based
    # bspl_surface = GeomAPI_PointsToBSplineSurface(array,Approx_IsoParametric,3,8,GeomAbs_C2,1e-1)#.Interpolate(array) ###For fitting

##Interpolation based
    bspl_surface = GeomAPI_PointsToBSplineSurface()#.Interpolate(array)
    bspl_surface.Interpolate(array,Approx_ChordLength)

    face_builder = BRepBuilderAPI_MakeFace(bspl_surface.Surface(),1e-6).Shape()

    c= STEPControl_Controller()
    c.Init()
    step_writer = STEPControl_Writer()
    Interface_Static("write.step.schema","AP214")
    step_writer.Transfer(face_builder,STEPControl_AsIs)
    filename="compen1_q_res201_dip_meep_to_gauss_ip_visio.step" ##For compensated surface
    # filename="q_displaced_res201_dip_meep_to_gauss_ip_visio.step" ##For design surface
    step_writer.Write(filename)

    display.DisplayShape(bspl_surface.Surface(),update=True)
    # for i in range(0,p.shape[1],1):
    #     for j in range(0,p.shape[2],1):
            # point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j]-p_center)
            # point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j])
            # display.DisplayShape(point_to_add,update=False)
    # display.Repaint()
    return bspl_surface.Surface()

filepath ="D:\\Confocal_measurements\\beam_shaping_project\\q_surface_40x_ip_visio_20240628\\structure_3_LP_84_83_ss_70k\\adjusted_area_adjusted_orientation_confocal\\"

cmap = 'inferno'


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

# pcd_design.paint_uniform_color([1,0,0])
# pcd_measu.paint_uniform_color([0.8,0.8,0.8])

print ("We display first the point cloud from the design")
o3d.visualization.draw_geometries([pcd_design])

print ("We display first the point cloud from the measurement")
o3d.visualization.draw_geometries([pcd_measu])

print ("We will now try to use ridge regression to fit a polynomial to the measurement data")

# m = int(np.ceil(np.sqrt(measu_points[:,0].shape[0])))
m = int(np.ceil(np.sqrt(measu_points[:,0].shape[0])))

xy_data = np.vstack([measu_points[:,0].flatten(),measu_points[:,1].flatten()]).T ##Original for polynomial fitting
xy_data = xy_data.astype(np.float32)
z_data = measu_points[:,2].astype(np.float32)
print ("Shape of xy_data")
print (xy_data.shape)
print ("Shape of z_data")
print (z_data.shape)


measu_points_z_low_pass_filter = gaussian_filter(measu_points[:,2],sigma=2.0)

measu_points_z_high_frequ = measu_points[:,2] - measu_points_z_low_pass_filter

# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# divider = make_axes_locatable(ax1)
# cax = divider.append_axes('right', size='5%', pad=0.05)
# im = ax1.tricontour(measu_points[:,0].flatten(), measu_points[:,1].flatten(), measu_points_z_low_pass_filter.flatten(), levels=50, cmap=cmap)
# plt.colorbar(im, cax=cax, orientation='vertical')
# plt.title("Measured surface - Low frequency contributions")
#
# # fig = plt.figure()
# ax2 = fig.add_subplot(122)
# divider2 = make_axes_locatable(ax2)
# cax2 = divider2.append_axes('right', size='5%', pad=0.05)
# im2 = ax2.tricontour(measu_points[:,0].flatten(), measu_points[:,1].flatten(), measu_points_z_high_frequ.flatten(), levels=50, cmap=cmap)
# plt.colorbar(im2, cax=cax2, orientation='vertical')
# plt.title("Measured surface - High frequency contributions")
#
#
# fig = plt.figure()
# ax1 = fig.add_subplot()
# divider = make_axes_locatable(ax1)
# cax = divider.append_axes('right', size='5%', pad=0.05)
# im = ax1.tricontour(measu_points[:,0], measu_points[:,1], measu_points[:,2], levels=50, cmap=cmap)
# plt.colorbar(im, cax=cax, orientation='vertical')
# plt.title("Measured surface - Full contributions")
# plt.show()

kernel = RBF(length_scale=0.5) + WhiteKernel(noise_level=0.1)
gp_model = GaussianProcessRegressor(kernel=kernel,alpha=0.1)
gp_model.fit(xy_data,z_data)

# poly = SplineTransformer(n_knots=m,degree=3) ##Spline based basis function regression
# poly = PolynomialFeatures(9) ##Polynomial based basis function regression
# X_poly = poly.fit_transform(xy_data)
# X_poly = rbf_feature.fit_transform(xy_data)



# ridge = Ridge(alpha=2)

# ridge.fit(X_poly,measu_points[:,2].flatten()) ##This is a polynomial regression using the original measurement data
# ridge.fit(X_poly,measu_points_z_low_pass_filter.flatten()) ##This is a polynomial regression using the low pass filter from the Gaussian filter


# z_fitted = ridge.predict(X_poly)

# r2 = r2_score(measu_points[:,2].flatten(),z_fitted)
# print ("The R^2 parameter value is equal to: ")
# print (r2)


pcd_poly_fit= o3d.geometry.PointCloud()

points_poly_fit = np.stack((measu_points[:,0].flatten(), measu_points[:,1].flatten(), z_fitted.flatten()), -1)
pcd_poly_fit.points = o3d.utility.Vector3dVector(points_poly_fit)


o3d.visualization.draw_geometries_with_editing([pcd_poly_fit])

print ("We want now to calculate the deviations between the generated polynomial fit and the original measurement data")
poly_measu_diff = measu_points[:,2] - z_fitted ##For initial data without gaussian low pass filter
# poly_measu_diff = measu_points_z_low_pass_filter - z_fitted ##For data from low pass filter (gaussian filter)

# fig = plt.figure()
# ax1 = fig.add_subplot()
# divider = make_axes_locatable(ax1)
# cax = divider.append_axes('right', size='5%', pad=0.05)
# im = ax1.tricontourf(measu_points[:,0], measu_points[:,1], poly_measu_diff, levels=50, cmap='hsv')
# plt.colorbar(im, cax=cax, orientation='vertical')
# ax1.title.set_text("Deviation between fitted polynomial and original measurement data")
# plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(121)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax1.tricontourf(measu_points[:,0].flatten(), measu_points[:,1].flatten(), poly_measu_diff.flatten(), levels=100, cmap=cmap)
plt.colorbar(im, cax=cax, orientation='vertical')
# plt.title("Surface deviations - Low frequency contributions")
ax1.title.set_text("Deviation between fitted polynomial and original measurement data")

# fig = plt.figure()
ax2 = fig.add_subplot(122)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
im2 = ax2.tricontour(measu_points[:,0].flatten(), measu_points[:,1].flatten(), poly_measu_diff.flatten(), levels=100, cmap=cmap)
plt.colorbar(im2, cax=cax2, orientation='vertical')
ax2.title.set_text("Deviation between fitted polynomial and original measurement data")
plt.show()

print ("We want now to try evaluating the polynomial expressions but at the coordinates defined by my design surface")
xy_design = np.vstack([design_points[:,0].flatten(),design_points[:,1].flatten()]).T

X_new_poly = poly.transform(xy_design)

z_fitted_design = ridge.predict(X_new_poly)

pcd_poly_fit_design= o3d.geometry.PointCloud()

points_poly_fit_design = np.stack((design_points[:,0].flatten(), design_points[:,1].flatten(), z_fitted_design.flatten()), -1)
pcd_poly_fit_design.points = o3d.utility.Vector3dVector(points_poly_fit_design)


o3d.visualization.draw_geometries_with_editing([pcd_poly_fit_design])

print ("We now try to calculate the deviations between the design points and the fitted polynomial points")

poly_fit_design_diff = design_points[:,2] - z_fitted_design

fig = plt.figure()
ax1 = fig.add_subplot(121)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax1.tricontourf(design_points[:,0].flatten(), design_points[:,1].flatten(), poly_fit_design_diff.flatten(), levels=50, cmap=cmap)
plt.colorbar(im, cax=cax, orientation='vertical')
# plt.title("Surface deviations - Low frequency contributions")
ax1.title.set_text("Deviation between fitted polynomial and original design data")

# fig = plt.figure()
ax2 = fig.add_subplot(122)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
im2 = ax2.tricontour(design_points[:,0].flatten(), design_points[:,1].flatten(), poly_fit_design_diff.flatten(), levels=50, cmap=cmap)
plt.colorbar(im2, cax=cax2, orientation='vertical')
ax2.title.set_text("Deviation between fitted polynomial and original design data")
plt.show()

poly_fit_design_compensation = design_points[:,2] + poly_fit_design_diff

print ("We now try to generate a spline surface from the compensated points")


pcd_poly_fit_design_compensation = o3d.geometry.PointCloud()

points_poly_fit_compensation = np.stack((design_points[:,0].flatten(), design_points[:,1].flatten(), poly_fit_design_compensation.flatten()), -1)

pcd_poly_fit_design_compensation.points = o3d.utility.Vector3dVector(points_poly_fit_compensation)


pcd_design.paint_uniform_color([1,0,0])
pcd_poly_fit_design_compensation.paint_uniform_color([0.8,0.8,0.8])


o3d.visualization.draw_geometries([pcd_poly_fit_design_compensation,pcd_design])

print ("Finally,we try to generate a spline based continuous surface from the obtained points")

points_poly_fit_compensation = points_poly_fit_compensation.swapaxes(0,1)

design_points = design_points.swapaxes(0,1)
print ("Shape of design points after swapaxes")
print (design_points.shape)
# print ("shape of entry array for step surface")
# print (points_poly_fit_compensation.shape)
#
#
display, start_display, add_menu, add_function_to_menu = init_display()
#
Nx = int(np.sqrt(design_points.shape[1]))
#
#
q_surf = points_to_surf(points_poly_fit_compensation.reshape(3,Nx,Nx),'name')
#
start_display()




