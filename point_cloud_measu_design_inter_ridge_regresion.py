##In this script, we want to use the point clouds found from the ICP algorithm in order
##to find the required compensation terms.
##As we mentioned before, we want to restrict the evaluation points to those defined by the design surface
##so that we avoid introducing weird artifacts towards the edges of the surface.
##This would basically mean that we would compensate for the regions that are valid within the measurement data.
import open3d as o3d
from matplotlib import path
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from common_functions import confocal_data_read,xyz_surf_save

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
from sklearn.linear_model import Ridge,LinearRegression,Lasso
from sklearn.metrics import r2_score



def points_to_surf(p,name):
    array = TColgp_Array2OfPnt(1,p.shape[1],1,p.shape[2])
    for i in range(0,p.shape[1],1):
        for j in range(0,p.shape[2],1):
            # point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j]-p_center) ##Original - We shift the points so that the apex is at (0,0,0)
            point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j])
            array.SetValue(i+1,j+1,point_to_add)
    print ("Surface creation")
##Least-square based
    # bspl_surface = GeomAPI_PointsToBSplineSurface(array,Approx_ChordLength,3,8,GeomAbs_C2,1e-4)#.Interpolate(array) ###For fitting

##Interpolation based
    bspl_surface = GeomAPI_PointsToBSplineSurface()#.Interpolate(array)
    bspl_surface.Interpolate(array,Approx_ChordLength)
    # bspl_surface.Interpolate(array)
    print ("Went through here already")

    face_builder = BRepBuilderAPI_MakeFace(bspl_surface.Surface(),1e-8).Shape()

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

##Taken from stackoverflow
####https://stackoverflow.com/questions/31542843/inpolygon-examples-of-matplotlib-path-path-contains-points-method
def inpolygon(xq, yq, xv, yv):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)

# filepath ="D:\\Confocal_measurements\\beam_shaping_project\\q_surface_40x_ip_visio_20240628\\structure_3_LP_84_83_ss_70k\\adjusted_area_adjusted_orientation_confocal\\" ##From confocal measurements

# filepath = "D:\\white_light_interfer\\rq_surface_extended_comp_q_exposed_20240715\\rq_surf_LP5\\" ##Reference first structure
filepath = "D:\\white_light_interfer\\rq_surface_extended_offset_hatching_30_deg\\"
# filepath ="D:\\white_light_interfer\\LP4\\"

cmap = 'inferno'


# design_pc= "q_surface_ref_z_right_position_cc.txt" ##Original surface profile -> oscillatory boundary
# design_pc= "surf_q_surface_smooth_c_401.txt" ##Surface profile with smoothed boundary
design_pc_2 = "surf_ff3_dip_meep_z50nm_to_gaus_res251_n_IP_Visio_v_real_further.txt"
design_pc = "q_surface_smooth_c_251.txt"
measu_pc= "str_3_LP78_upper_surf_normal_cc_xy_trans_xyz_rot.txt" ##reproducibility test structure
# measu_pc= "LP3_2_upper_surface_avg_1_cc_xy_trans_xyz_rot_z_trans.txt" ##reproducibility test structure ##2
# measu_pc= "rq_surf_LP5_upper_surface_reference_original_cc_z_trans_only.txt" ##Reference structure after applying z translation only to match the heights

cf_x_measu, cf_y_measu, cf_z_measu = confocal_data_read(filepath + measu_pc)
pcd_measu = o3d.geometry.PointCloud()
points_measu = np.stack((cf_x_measu.flatten(), cf_y_measu.flatten(), cf_z_measu.flatten()), -1)
pcd_measu.points = o3d.utility.Vector3dVector(points_measu)

cf_x_design, cf_y_design, cf_z_design= confocal_data_read(filepath + design_pc)
cf_x_design2, cf_y_design2, cf_z_design2= confocal_data_read(filepath + design_pc_2)
pcd_design= o3d.geometry.PointCloud()
points_design= np.stack((cf_x_design.flatten(), cf_y_design.flatten(), cf_z_design.flatten()), -1)
pcd_design.points = o3d.utility.Vector3dVector(points_design)

plt.figure()
plt.scatter(cf_x_design,cf_y_design,label='Points from point by point construction')
plt.scatter(cf_x_design2,cf_y_design2,label='Points from spline evaluation')
plt.legend()

# pcd_design = o3d.io.read_point_cloud(filepath+target_pc)
# pcd_measu = o3d.io.read_point_cloud(filepath+source_pc)


# o3d.visualization.draw_geometries_with_editing([pcd_design])

design_points = np.asarray(pcd_design.points)
# measu_points = np.asarray(pcd_measu.points)
print ("Shape of the point cloud arrays")
print (design_points.shape)
# print (measu_points.shape)
##We extract the boundary points for the design data to define the ROI for the polynomial fit
d_x = design_points[:,0].reshape(int(np.sqrt(design_points.shape[0])),-1)
d_y = design_points[:,1].reshape(int(np.sqrt(design_points.shape[0])),-1)
d_z = design_points[:,2].reshape(int(np.sqrt(design_points.shape[0])),-1)
d_x_b = np.hstack((d_x[0,:],d_x[1:,-1],d_x[-1,:][::-1][1:],d_x[:,0][::-1][1:]))*(1)
d_y_b = np.hstack((d_y[0,:],d_y[1:,-1],d_y[-1,:][::-1][1:],d_y[:,0][::-1][1:]))*(1)
d_z_b = np.hstack((d_z[0,:],d_z[1:,-1],d_z[-1,:][::-1][1:],d_z[:,0][::-1][1:]))*(1)


fig = plt.figure()
ax = plt.axes(projection='3d')
surf = ax.scatter(d_x_b, d_y_b, d_z_b)
ax.scatter(d_x[0,0],d_y[0,0],d_z[0,0],color='red')
plt.title("Points on boundary - Design surface")

plt.figure()
plt.plot(d_x_b,d_y_b,color='blue')
plt.scatter(d_x_b,d_y_b,color='red')
plt.title("XY coordinates - surface contour")
plt.show()

###We want to find the points from the measurement data that are found within the definition of the optical surface
points_measu_x = points_measu[:,0]
points_measu_y = points_measu[:,1]
points_measu_z = points_measu[:,2]
inside_boundary = inpolygon(points_measu_x,points_measu_y,d_x_b,d_y_b)
print ("This is the output of inpolygon")
print (inside_boundary.shape)
print (points_measu_x.shape)

point_measu_x_inside = points_measu_x[inside_boundary]
point_measu_y_inside = points_measu_y[inside_boundary]
point_measu_z_inside = points_measu_z[inside_boundary]

pcd_measu_inside = o3d.geometry.PointCloud()
points_measu_inside = np.stack((point_measu_x_inside.flatten(), point_measu_y_inside.flatten(), point_measu_z_inside.flatten()), -1)
pcd_measu_inside.points = o3d.utility.Vector3dVector(points_measu_inside)


# pcd_design.paint_uniform_color([1,0,0])
# pcd_measu.paint_uniform_color([0.8,0.8,0.8])

pcd_measu_inside.paint_uniform_color([0.8,0.8,0.8])

print ("We display first the point cloud from the design")
o3d.visualization.draw_geometries([pcd_design,pcd_measu_inside])

print ("We display first the point cloud from the measurement")
pcd_measu.paint_uniform_color([1,0,0])

o3d.visualization.draw_geometries([pcd_measu_inside,pcd_measu])
# o3d.visualization.draw_geometries([pcd_measu_inside])


measu_points = np.asarray(pcd_measu_inside.points) ##For fitting the polynomial, we just use the points found within the defined contour



print ("We will now try to use ridge regression to fit a polynomial to the measurement data")

# m = int(np.ceil(np.sqrt(measu_points[:,0].shape[0])))
m = int(np.ceil(np.sqrt(measu_points[:,0].shape[0]))) ##Original m value -> Number of knot points for the spline based interpolation
# m = int(np.ceil((measu_points[:,0].shape[0])*0.5))

xy_data = np.vstack([measu_points[:,0].flatten(),measu_points[:,1].flatten()]).T ##Original for polynomial fitting


measu_points_z_low_pass_filter = gaussian_filter(measu_points[:,2],sigma=3.0)

measu_points_z_high_frequ = measu_points[:,2] - measu_points_z_low_pass_filter

print ("We display the high frequency components of the surface")

pcd_high_freq_comps= o3d.geometry.PointCloud()
points_high_freq = np.stack((measu_points[:,0].flatten(), measu_points[:,1].flatten(), measu_points_z_high_frequ.flatten()), -1)
pcd_high_freq_comps.points = o3d.utility.Vector3dVector(points_high_freq)
o3d.visualization.draw_geometries_with_editing([pcd_high_freq_comps])


print ("We display the low frequency components of the surface")

pcd_low_freq_comps= o3d.geometry.PointCloud()
points_low_freq = np.stack((measu_points[:,0].flatten(), measu_points[:,1].flatten(), measu_points_z_low_pass_filter.flatten()), -1)
pcd_low_freq_comps.points = o3d.utility.Vector3dVector(points_low_freq)
o3d.visualization.draw_geometries_with_editing([pcd_low_freq_comps])

plot_freq_comps = False

if plot_freq_comps:

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax1.tricontour(measu_points[:,0].flatten(), measu_points[:,1].flatten(), measu_points_z_low_pass_filter.flatten(), levels=50, cmap=cmap)
    plt.colorbar(im, cax=cax, orientation='vertical')
    plt.title("Measured surface - Low frequency contributions")
    #
    # fig = plt.figure()
    ax2 = fig.add_subplot(122)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    im2 = ax2.tricontour(measu_points[:,0].flatten(), measu_points[:,1].flatten(), measu_points_z_high_frequ.flatten(), levels=50, cmap=cmap)
    plt.colorbar(im2, cax=cax2, orientation='vertical')
    plt.title("Measured surface - High frequency contributions")
    #
    #
    fig = plt.figure()
    ax1 = fig.add_subplot()
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax1.tricontour(measu_points[:,0], measu_points[:,1], measu_points[:,2], levels=50, cmap=cmap)
    plt.colorbar(im, cax=cax, orientation='vertical')
    plt.title("Measured surface - Full contributions")
    plt.show()

# poly = SplineTransformer(n_knots=m,degree=5) ##Spline based basis function regression
# poly= RBFSampler(gamma=1e-3)
poly = PolynomialFeatures(6) ##Polynomial based basis function regression
# X_poly = poly.fit_transform(xy_data)
X_poly = poly.fit_transform(xy_data)

# ridge = LinearRegression()
# ridge = Lasso(alpha=3)

ridge = Ridge(alpha=4)
ridge.fit(X_poly,measu_points[:,2].flatten())


# ridge.fit(X_poly,measu_points[:,2].flatten()) ##This is a polynomial regression using the original measurement data


# ridge.fit(X_poly,measu_points_z_low_pass_filter.flatten()) ##This is a polynomial regression using the low pass filter from the Gaussian filter


z_fitted = ridge.predict(X_poly)

r2 = r2_score(measu_points[:,2].flatten(),z_fitted)
print ("The R^2 parameter value is equal to: ")
print (r2)


pcd_poly_fit= o3d.geometry.PointCloud()

points_poly_fit = np.stack((measu_points[:,0].flatten(), measu_points[:,1].flatten(), z_fitted.flatten()), -1)
pcd_poly_fit.points = o3d.utility.Vector3dVector(points_poly_fit)


o3d.visualization.draw_geometries_with_editing([pcd_poly_fit],window_name='Polynomial fit')

print ("We want now to calculate the deviations between the generated polynomial fit and the original measurement data")
poly_measu_diff = measu_points[:,2] - z_fitted ##For initial data without gaussian low pass filter
# poly_measu_diff = measu_points_z_low_pass_filter - z_fitted ##For data from low pass filter (gaussian filter)

pcd_poly_measu_diff = o3d.geometry.PointCloud()

points_poly_measu_diff = np.stack((measu_points[:,0].flatten(), measu_points[:,1].flatten(), poly_measu_diff.flatten()), -1)
pcd_poly_measu_diff.points = o3d.utility.Vector3dVector(points_poly_measu_diff)


o3d.visualization.draw_geometries_with_editing([pcd_poly_measu_diff],window_name='Difference - Polynomial and measurement')

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
ax1.scatter(d_x_b,d_y_b,color='red')
plt.colorbar(im, cax=cax, orientation='vertical')
# plt.title("Surface deviations - Low frequency contributions")
ax1.title.set_text("Deviation between fitted polynomial and measurement data")

# fig = plt.figure()
ax2 = fig.add_subplot(122)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
im2 = ax2.tricontour(measu_points[:,0].flatten(), measu_points[:,1].flatten(), poly_measu_diff.flatten(), levels=100, cmap=cmap)
ax2.scatter(d_x_b,d_y_b,color='red')
plt.colorbar(im2, cax=cax2, orientation='vertical')
ax2.title.set_text("Deviation between fitted polynomial and measurement data")
plt.show()

print ("We want now to try evaluating the polynomial expressions but at the coordinates defined by the design surface")
xy_design = np.vstack([design_points[:,0].flatten(),design_points[:,1].flatten()]).T

X_new_poly = poly.transform(xy_design)

z_fitted_design = ridge.predict(X_new_poly)

pcd_poly_fit_design= o3d.geometry.PointCloud()

points_poly_fit_design = np.stack((design_points[:,0].flatten(), design_points[:,1].flatten(), z_fitted_design.flatten()), -1)
pcd_poly_fit_design.points = o3d.utility.Vector3dVector(points_poly_fit_design)


o3d.visualization.draw_geometries_with_editing([pcd_poly_fit_design])

print ("We now try to calculate the deviations between the design points and the fitted polynomial points")

print ("We restrict the compensation to the boundary contour defined above.")
print ("For design points found outside of the defined boundary contour, no compensation is added")


inside_boundary_design = inpolygon(design_points[:,0],design_points[:,1],d_x_b,d_y_b)

##We now use the results from above mask to apply the compensation

poly_fit_design_diff = design_points[:,2] - z_fitted_design

# poly_fit_design_diff = np.where(inside_boundary_design,design_points[:,2]- z_fitted_design,0)

# xyz_surf_save(design_points[:,0].flatten(),design_points[:,1].flatten(),poly_fit_design_diff.flatten(),"design_poly_fit_repro_2_z_shifted_r_0p8.txt")

fig = plt.figure()
ax1 = fig.add_subplot(121)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax1.tricontourf(design_points[:,0].flatten(), design_points[:,1].flatten(), poly_fit_design_diff.flatten(), levels=50, cmap=cmap)
ax1.scatter(d_x_b,d_y_b,color='red')
plt.colorbar(im, cax=cax, orientation='vertical')
# plt.title("Surface deviations - Low frequency contributions")
ax1.title.set_text("Deviation between fitted polynomial and original design data")

# fig = plt.figure()
ax2 = fig.add_subplot(122)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
im2 = ax2.tricontour(design_points[:,0].flatten(), design_points[:,1].flatten(), poly_fit_design_diff.flatten(), levels=50, cmap=cmap)
ax2.scatter(d_x_b,d_y_b,color='red')
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


o3d.visualization.draw_geometries([pcd_poly_fit_design_compensation])


o3d.visualization.draw_geometries([pcd_design])

print ("Finally,we try to generate a spline based continuous surface from the obtained points")

points_poly_fit_compensation = points_poly_fit_compensation.swapaxes(0,1)

design_points = design_points.swapaxes(0,1)
print ("Shape of design points after swapaxes")
print (design_points.shape)
# print ("shape of entry array for step surface")
print (points_poly_fit_compensation.shape)

design_comp_z_low_pass_filter= gaussian_filter(poly_fit_design_compensation,sigma=8.0)
print ("shape of design_comp_z_low_pass_filter")
print (design_comp_z_low_pass_filter.shape)

design_comp_z_high_comps = poly_fit_design_compensation - design_comp_z_low_pass_filter

fig = plt.figure()
ax1 = fig.add_subplot(121)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax1.tricontour(design_points[0,:].flatten(), design_points[1,:].flatten(),
                    design_comp_z_low_pass_filter.flatten(), levels=50, cmap=cmap)
plt.colorbar(im, cax=cax, orientation='vertical')
plt.title("Compensated surface - Low frequency contributions")
#
# fig = plt.figure()
ax2 = fig.add_subplot(122)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
im2 = ax2.tricontour(design_points[0,:].flatten(), design_points[1,:].flatten(), design_comp_z_high_comps.flatten(),
                     levels=50, cmap=cmap)
plt.colorbar(im2, cax=cax2, orientation='vertical')
plt.title("Compensated surface - High frequency contributions")
#
#
fig = plt.figure()
ax1 = fig.add_subplot()
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax1.tricontour(design_points[0,:], design_points[1,:], poly_fit_design_compensation, levels=50, cmap=cmap)
plt.colorbar(im, cax=cax, orientation='vertical')
plt.title("Compensated surface - Full contributions")
plt.show()


points_poly_fit_compensation_after_g_f = np.stack((design_points[0,:].flatten(), design_points[1,:].flatten(), poly_fit_design_compensation.flatten()), -1)

points_poly_fit_compensation_after_g_f= points_poly_fit_compensation_after_g_f.swapaxes(0,1)
#
#
display, start_display, add_menu, add_function_to_menu = init_display()
#
Nx = int(np.sqrt(design_points.shape[1]))
#
#
q_surf = points_to_surf(points_poly_fit_compensation_after_g_f.reshape(3,Nx,Nx),'name')
#
start_display()




