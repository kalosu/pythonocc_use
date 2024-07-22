##Is it possible to generate a spline object from the confocal measurement data??
##If yes, can we use this object and the original STEP object (or spline object)
##To estimate the deviations in the surface??
import numpy as np
from common_functions import surf_param_read,surf_params_sN_read,xy_target_coords_read,uv_plane_point,stereo_proj_dir,circ_boundary_delim,surf_N_eval,output_dir_eval,stereo_proj_dir_to_plane,xy_target_coords_save,confocal_data_read,xyz_surf_save,icp_pcd
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

import open3d as o3d
from scipy.spatial import Delaunay
from scipy.interpolate import RBFInterpolator,RectBivariateSpline,griddata
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def points_to_surf(p,name):
    array = TColgp_Array2OfPnt(1,p.shape[1],1,p.shape[2])
    for i in range(0,p.shape[1],1):
        for j in range(0,p.shape[2],1):
            # point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j]-p_center) ##Original - We shift the points so that the apex is at (0,0,0)
            point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j])
            array.SetValue(i+1,j+1,point_to_add)
    print ("Surface creation")
    bspl_surface = GeomAPI_PointsToBSplineSurface()#.Interpolate(array)
    bspl_surface.Interpolate(array,Approx_ChordLength)
    face_builder = BRepBuilderAPI_MakeFace(bspl_surface.Surface(),1e-6).Shape()
    display.DisplayShape(bspl_surface.Surface(),update=True)
    # for i in range(0,p.shape[1],1):
    #     for j in range(0,p.shape[2],1):
            # point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j]-p_center)
            # point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j])
            # display.DisplayShape(point_to_add,update=False)
    # display.Repaint()
    return bspl_surface.Surface()


def points_from_surface(bspl_surf,Nx,Ny):

    face = BRepBuilderAPI_MakeFace(bspl_surf,1e-6).Face()
    ##get face uv bounds
    umin, umax, vmin, vmax = shapeanalysis.GetFaceUVBounds(face)
    print (umin,umax,vmin,vmax)
    u_eval = np.linspace(umin,umax,Nx)
    v_eval = np.linspace(vmin,vmax,Ny)
    uu_eval, vv_eval = np.meshgrid(u_eval,v_eval)
    uu_eval = uu_eval.flatten()
    vv_eval = vv_eval.flatten()
    pnts_x = []
    pnts_y = []
    pnts_z = []
    sas = ShapeAnalysis_Surface(bspl_surf)
    print ("We start evaluating the points on the spline surface")
    for i in range(0,uu_eval.shape[0],1):
        p = sas.Value(uu_eval[i],vv_eval[i])
        pnts_x.append(p.X())
        pnts_y.append(p.Y())
        pnts_z.append(p.Z())
    print ("finished")
    return np.asarray(pnts_x),np.asarray(pnts_y),np.asarray(pnts_z)


if __name__== "__main__":

    # filepath ="D:\\Confocal_measurements\\beam_shaping_project\\q_surface_40x_ip_visio_20240628\\structure_3_LP_84_83_ss_70k\\adjusted_area_adjusted_orientation_confocal\\"

    # filepath ="D:\\white_light_interfer\\LP4\\"
    filepath = "D:\\white_light_interfer\\rq_surface_extended_comp_q_exposed_20240715\\"

    cropped_flag = False

    if cropped_flag:
        filename = "top_surf_measurement_ls_40mu.ply"
        pcd = o3d.io.read_point_cloud(filepath + filename)
        points = np.asarray(pcd.points)
        cf_x = points[:,0]
        cf_y = points[:,1]
        cf_z = points[:,2]
    else:
        # filename ="structure_3_adjusted_area_orientation.txt"
        filename ="rq_surf_extended_LP1_increased_LP.txt"

        cf_x, cf_y, cf_z = confocal_data_read(filepath+filename)

    ##Can we also try to move the points in the xy plane so that the maximum is at xy=0?
    ##This is definitely not the best way to do this but let's see:
    z_max_index = np.abs(cf_z - np.max(cf_z)).argmin()


    ##Can we extract the y and x coordinate for the point of maximum height value?

    x_max = np.max(cf_x)
    x_min = np.min(cf_x)

    y_max = np.max(cf_y)
    y_min = np.min(cf_y)

    # cf_z -= np.min(cf_z)

    fig = plt.figure()
    ax1 = fig.add_subplot()
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right',size='5%',pad=0.05)
    im = ax1.tricontourf(cf_y, cf_x, cf_z, levels=50, cmap='hsv')
    plt.colorbar(im,cax=cax,orientation='vertical')
    plt.title("Measured surface profile")
    plt.show()

    print ("We want to first have an estimate on the range of x and y coordinates")
    print (x_max)
    print (x_min)
    print (y_max)
    print (y_min)
    print ("We calculate then the displacement as the center between the maxima and the minima")
    x_delta = x_max - x_min
    print ("Extension along x")
    print (x_delta)
    y_delta = y_max - y_min
    print ("Extension along y")
    print (y_delta)


    x_z_max = cf_x[int(z_max_index)]

    y_z_max = cf_y[int(z_max_index)]
    print ("These are the x and y coordinates for the maximum")
    print (x_z_max)
    print (y_z_max)
    # cf_x -= x_z_max ##Originally used in combination with the z of heighest value
    # cf_y -= y_z_max

    # cf_x -= x_delta*0.5 ##This is applied when the domains are not aligned, i.e, right after measurement
    # cf_y -= y_delta*0.5 ##This is applied when the domains are not aligned, i.e, right after measurement


    ##Can we try to move the points so that the maximum is at z=0??
    # cf_z -=np.max(cf_z)

    # u = (cf_x - cf_x.min()) / (cf_x.max() - cf_x.min())
    # v = (cf_y - cf_y.min()) / (cf_y.max() - cf_y.min())
    #
    # u2 = np.linspace(0,1,101)
    # v2 = np.linspace(0,1,101)
    # uu, vv = np.meshgrid(u2,v2)
    #
    # grid_x = griddata((u,v),cf_x,(uu,vv),method='linear')
    # grid_y = griddata((u,v),cf_y,(uu,vv),method='linear')
    # grid_z = griddata((u,v),cf_z,(uu,vv),method='linear')
    #
    # ##Do we need to filter the nan values first??
    #
    # print ("shape of the interpolated objects")
    # print (grid_x.shape)
    # print (grid_y.shape)
    # print (grid_z.shape)
    #
    # print (grid_x)
    #
    # fig = plt.figure(figsize=(12, 6))
    #
    # ax1 = fig.add_subplot(131)
    # ax1.imshow(grid_x, extent=(0, 1, 0, 1), origin='lower')
    # # ax1.scatter(u, v, c='red', edgecolor='k')
    # ax1.set_title('Interpolated x(u, v)')
    #
    # ax2 = fig.add_subplot(132)
    # ax2.imshow(grid_y, extent=(0, 1, 0, 1), origin='lower')
    # # ax2.scatter(u, v, c='red', edgecolor='k')
    # ax2.set_title('Interpolated y(u, v)')
    #
    # ax3 = fig.add_subplot(133)
    # ax3.imshow(grid_z, extent=(0, 1, 0, 1), origin='lower')
    # # ax3.scatter(u, v, c=cf_z, edgecolor='k', cmap='viridis')
    # ax3.set_title('Interpolated z(u, v)')
    #
    # plt.show()





    pcd = o3d.geometry.PointCloud()

    # points = np.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), -1)
    points = np.stack((cf_y.flatten(), cf_x.flatten(), cf_z.flatten()), -1)
    pcd.points = o3d.utility.Vector3dVector(points)

    print ("We first plot the measurement data points")

    o3d.visualization.draw_geometries_with_editing([pcd])

    xy = np.stack((cf_x,cf_y))
    print ("shape of xy")
    print (xy.shape)

    # rbf_inter = RBFInterpolator(xy.transpose(),cf_z,neighbors=10,kernel='cubic')


    filename3 = "ff3_dip_meep_z50nm_to_gaus_res201_n_IP_Visio_real_further2"

    Nx =201
    Ny =201
    s3, N3 = surf_params_sN_read(filename3)
    # s3[2,:] -= np.max(s3[2,:])
    ###What happens if we transpose all the points from our surface??
    # s3_x = s3[0,:].reshape(Nx,Ny).transpose()
    # s3_y = s3[1,:].reshape(Nx,Ny).transpose()
    # s3_z = s3[2,:].reshape(Nx,Ny).transpose()

    # s3_transpose = np.stack((s3_x.flatten(),s3_y.flatten(),s3_z.flatten()),axis=-1)

    ##We generate a new grid of points but for this we will first use the spline surface object
    ##We will then sample this surface to obtain the grid that we want
    display, start_display, add_menu, add_function_to_menu = init_display()
    q_surf = points_to_surf(s3.reshape(3,Nx,Ny),'name')

    start_display()

    shape_desired = int(np.ceil(np.sqrt(points.shape[0])))
    print ("The closest number of points along one axis to have approximately the same number of points would be:")
    print (shape_desired)


    surf_x, surf_y, surf_z = points_from_surface(q_surf,(shape_desired),(shape_desired))

    surf_x = surf_x.reshape(shape_desired,shape_desired).transpose()
    surf_y = surf_y.reshape(shape_desired,shape_desired).transpose()
    surf_z = surf_z.reshape(shape_desired,shape_desired).transpose()

    # surf_z -= 40 ##We substract the offset applied to the solidworks model. Remember that what we printed included the surface but at a different
    max_surf_z = np.max(surf_z)
    # offset_surf_z = max_surf_z - 98.403
    # surf_z -= offset_surf_z ##This should be used only if we want to apply some offset along z to all coordinate points
    ##location

    x_max2 = np.max(surf_x)
    x_min2 = np.min(surf_x)

    y_max2 = np.max(surf_y)
    y_min2 = np.min(surf_y)

    print ("We want to first have an estimate on the range of x and y coordinates (reference surface)")
    print (x_max2)
    print (x_min2)
    print (y_max2)
    print (y_min2)
    print ("We calculate then the displacement as the center between the maxima and the minima (reference surface)")
    x_delta2 = x_max2 - x_min2
    print ("Extension along x (reference surface)")
    print (x_delta2)
    y_delta2 = y_max2 - y_min2
    print ("Extension along y (reference surface)")
    print (y_delta2)

    print ("min and max along x")
    print (np.min(surf_x))
    print (np.max(surf_x))
    print ("Approximate spacing along x")
    print ((np.max(surf_x)-np.min(surf_x))/401)

    print ("min and max along y")
    print (np.min(surf_y))
    print (np.max(surf_y))
    print ("Approximate spacing along y")
    print ((np.max(surf_y)-np.min(surf_y))/401)

    print ("Maximum z value for design surface")
    print (np.max(surf_z))
    print ("Maximum z value for measured data after height correction")
    print (np.max(cf_z))

    # points_original = np.stack((s3[0,:].flatten(), s3[1,:].flatten(), s3[2,:].flatten()), -1)
    points_original = np.stack((surf_x.flatten(), surf_y.flatten(), surf_z.flatten()), -1)



    print ("shapes of both point clouds")
    print (points.shape)
    print (points_original.shape)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right',size='5%',pad=0.05)
    im = ax1.tricontour(cf_y, cf_x, cf_z, levels=50, cmap='hsv')
    plt.colorbar(im,cax=cax,orientation='vertical')
    plt.title("Measured surface profile")

    # fig = plt.figure()
    ax2 = fig.add_subplot(122)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right',size='5%',pad=0.05)
    im2 = ax2.tricontour(surf_x.flatten(), surf_y.flatten(), surf_z.flatten(), levels=50, cmap='hsv')
    plt.colorbar(im2,cax=cax2,orientation='vertical')
    plt.title("Surface original points")
    # # ax.scatter(p_tt_deg*np.cos(p_pp),p_tt_deg*np.sin(p_pp),color='blue')
    # ax.set_aspect('equal')
    # ax.axis('off')
    # ax_polar = fig.add_axes(ax.get_position(),polar=True)
    # ax_polar.set_facecolor('none')
    # ax_polar.set_ylim(0,np.max(p_tt_deg))
    plt.show()

    ##We save the data points from the surface profile from pythonocc
    filepath="D:\\Confocal_measurements\\beam_shaping_project\\"
    filename = "q_surface_ref_z_right_position.txt"
    xyz_surf_save(surf_x.flatten(),surf_y.flatten(),surf_z.flatten(),filepath+filename)

    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(points_original)


    o3d.visualization.draw_geometries_with_editing([pcd_original])

    # pcd.paint_uniform_color([1,0,0])
    # pcd_original.paint_uniform_color([0.8,0.8,0.8])

    # print ("These are the number of points for both arrays")
    # print (points_original.shape)
    # print (points.shape)

    # o3d.visualization.draw_geometries([pcd,pcd_original])

    # print ("We would then use apply the ICP here using the measurement data and the data from the surface")
    # icp_pcd(pcd,pcd_original)

