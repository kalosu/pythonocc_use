# from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from common_functions import surf_param_read,surf_params_sN_read,xy_target_coords_read,uv_plane_point,stereo_proj_dir,circ_boundary_delim,surf_N_eval,output_dir_eval,stereo_proj_dir_to_plane,xy_target_coords_save,xyz_surf_save
from scipy.optimize import root,minimize,fsolve
from scipy.interpolate import CubicSpline,RectBivariateSpline


# from __future__ import print_function

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
from OCC.Core.Tesselator import ShapeTesselator
from OCC.Extend.DataExchange import write_stl_file

def surf_ray_inter(x,idir,sas):
    pnt = sas.Value(x[0],x[1])
    pnt2 = np.asarray((pnt.X(),pnt.Y(),pnt.Z()))
    d = np.linalg.norm(pnt2)
    pnt_from_ray = idir*x[2]
    # return [np.linalg.norm(pnt.X()-pnt_from_ray[0])**2, np.linalg.norm(pnt.Y()-pnt_from_ray[1])**2, np.linalg.norm(pnt.Z()-pnt_from_ray[2])**2]
    return [pnt.X()-pnt_from_ray[0], pnt.Y()-pnt_from_ray[1], pnt.Z()-pnt_from_ray[2]]
    # return np.linalg.norm(pnt2 - pnt_from_ray)

def surf_ray_inter_norm(x,idir,sas):
    pnt = sas.Value(x[0],x[1])
    pnt2 = np.asarray((pnt.X(),pnt.Y(),pnt.Z()))
    d = np.linalg.norm(pnt2)
    pnt_from_ray = idir*d
    return np.linalg.norm(pnt2-pnt_from_ray)




def uv_chordlength(p):
    ###By default, the spacing in the parametric U-V space is not uniform.
    ##Therefore, if we want to extract the surface points at the "correct" locations, we need to query the points at the correct UV coordinates.
    ##Following, the GeomAPI_PointsToBSSplineSurface class documentation, we construct arrays of U and V coordinates accordingly.
    u_min = 0.0
    u_max = 1.0
    v_min = 0.0
    v_max = 1.0
    u = []
    v = []
    u.append(u_min)
    v.append(v_min)
    for j in range(0,1,1):
        for i in range(0,p.shape[1]-1,1):
            u.append(u[-1] + gp_Pnt(p[0,i+1,j+1],p[1,i+1,j+1],p[2,i+1,j+1]).Distance(gp_Pnt(p[0,i+1-1,j+1],p[1,i+1-1,j+1],p[2,i+1-1,j+1])))
            v.append(v[-1])
    # for i in range(0,p.shape[1]-1,1):
    #     for j in range(0,p.shape[2]-1,1):
    #         v.append(v[-1] + gp_Pnt(p[0,i+1,j+1],p[1,i+1,j+1],p[2,i+1,j+1]).Distance(gp_Pnt(p[0,i+1,j+1-1],p[1,i+1,j+1-1],p[2,i+1,j+1-1])))
    return np.asarray(u),np.asarray(v)

def distance_point_to_surf(p,surface):
    projector = GeomAPI_ProjectPointOnSurf()
    distance_array = [ ]
    for i in range(0,p.shape[1],1):
        for j in range(0,p.shape[2],1):
            point = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j])
            projector.Init(point,surface)
            if projector.NbPoints()>0:
                nearest_point = projector.NearestPoint()
                distance = point.Distance(nearest_point)
                distance_array.append(distance)
            else:
                distance_array.append(1000.0)


def evaluate_surface_at_uv(surface,u,v):
    pnt = gp_Pnt()
    surface.D0(u,v,pnt)
    return pnt


def chord_length_parameterization(points_array):
    num_rows, num_cols = points_array.shape[1], points_array.shape[2]
    u_params = np.zeros(num_cols)
    v_params = np.zeros(num_rows)

    # Parameterize in the u direction (columns)
    for j in range(1, num_cols):
        dist = np.linalg.norm(points_array[:, :, j] - points_array[:, :, j - 1], axis=0)
        u_params[j] = u_params[j - 1] + dist.mean()
    u_params /= u_params[-1]  # Normalize to [0, 1]

    # Parameterize in the v direction (rows)
    for i in range(1, num_rows):
        dist = np.linalg.norm(points_array[:, i, :] - points_array[:, i - 1, :], axis=0)
        v_params[i] = v_params[i - 1] + dist.mean()
    v_params /= v_params[-1]  # Normalize to [0, 1]

    return u_params, v_params

def centripetal_parameterization(points_array, i, j):
    num_rows, num_cols = points_array.shape[1], points_array.shape[2]
    u = np.sqrt(j) / np.sqrt(num_cols - 1)
    v = np.sqrt(i) / np.sqrt(num_rows - 1)
    return u, v


def points_surf_append(p,name,N_extra_p):
    z_min = np.min(p[2,:,:]) ##We first extract the lowest height point from the surface profile.
    s1_x = p[0]
    s1_y = p[1]
    s1_z = p[2]

    middle_loc = int(s1_z.shape[1] * 0.5)
    z_boundary = np.hstack((s1_z[0,:], s1_z[1:, -1], s1_z[-1, :][::-1][1:], s1_z[:, 0][::-1][1:]))
    x_boundary = np.hstack((s1_x[0,:], s1_x[1:, -1], s1_x[-1, :][::-1][1:], s1_x[:, 0][::-1][1:]))
    y_boundary = np.hstack((s1_y[0,:], s1_y[1:, -1], s1_y[-1, :][::-1][1:], s1_y[:, 0][::-1][1:]))
    b = np.linspace(0,1.0,x_boundary.shape[0])
    x_b_spline = CubicSpline(b,x_boundary)
    y_b_spline = CubicSpline(b,y_boundary)
    z_b_spline = CubicSpline(b,z_boundary)

    ##We now create arrays in 2D which are going to be used to store the coordinates of the extention points
    # N_extra_p = 10
    x_ext = np.zeros((s1_x.shape[0]+(N_extra_p*2),s1_x.shape[0]+(N_extra_p*2)))
    y_ext = np.zeros((s1_x.shape[0]+(N_extra_p*2),s1_x.shape[0]+(N_extra_p*2)))
    z_ext = np.zeros((s1_x.shape[0]+(N_extra_p*2),s1_x.shape[0]+(N_extra_p*2)))

    # z_ext= np.pad(s1_z, (N_extra_p, N_extra_p), mode='constant')
    # y_ext= np.pad(s1_y, (N_extra_p, N_extra_p), mode='constant')
    # x_ext= np.pad(s1_x, (N_extra_p, N_extra_p), mode='constant')
    ##Since we want that the height array is somehow continuous we will start by adding points from the boundary
    ##We then move through the array inwards until we reach the original size of our array of points
    N_layers = N_extra_p
    max_s = 0.99
    min_s = 0.95
    scaling_array = np.linspace(min_s,max_s,N_layers)
    scaling_array = np.sqrt(scaling_array)
    z_base_array = np.linspace(z_min*0.8,z_min*0.99,N_layers)
    print ("These are the height values that will be used")
    print (z_base_array)
    print ("These are the scaling values that will be used ")
    print (scaling_array)

    for i in range(0,N_extra_p,1):
        N_eval = (x_ext.shape[0]-2*i) * 1 + ((x_ext.shape[0]-2*i) - 1) * 2 + ((x_ext.shape[0]-2*i)-2)*1  ##Oneside with full size and the other 3 with one element less than the size per side.
        a = (x_ext.shape[0]-2*i) *1
        b = a + ((x_ext.shape[0]-2*i)-1)*1
        c = b + ((x_ext.shape[0]-2*i)-1)*1
        b_eval = np.linspace(0,1,N_eval)
        x_layer = x_b_spline(b_eval)*scaling_array[i]
        y_layer = y_b_spline(b_eval)*scaling_array[i]
        z_layer = z_b_spline(b_eval)*scaling_array[i]
        # z_layer = np.ones(N_eval)*z_base_array[i]
        z_ext[i,i:x_ext.shape[1]-i:1] = z_layer[0:a]
        x_ext[i,i:x_ext.shape[1]-i:1] = x_layer[0:a]
        y_ext[i,i:x_ext.shape[1]-i:1] = y_layer[0:a]
        # z_boundary = np.hstack((s1_z[0, :], s1_z[1:, -1], s1_z[-1, :][::-1][1:], s1_z[:, 0][::-1][1:]))
        z_ext[i+1:x_ext.shape[1]-(i):1,-i-1] =z_layer[a:b]
        x_ext[i+1:x_ext.shape[1]-(i):1,-i-1] =x_layer[a:b]
        y_ext[i+1:x_ext.shape[1]-(i):1,-i-1] =y_layer[a:b]

        z_ext[-i-1,i:x_ext.shape[1]-i-1:1][::-1][:] = z_layer[b:c]
        x_ext[-i-1,i:x_ext.shape[1]-i-1:1][::-1][:] = x_layer[b:c]
        y_ext[-i-1,i:x_ext.shape[1]-i-1:1][::-1][:] = y_layer[b:c]

        z_ext[i+1:x_ext.shape[1]-i-1:1,i][::-1][:] = z_layer[c:]
        x_ext[i+1:x_ext.shape[1]-i-1:1,i][::-1][:] = x_layer[c:]
        y_ext[i+1:x_ext.shape[1]-i-1:1,i][::-1][:] = y_layer[c:]

    # plt.figure()
    # plt.imshow(z_ext.reshape(s1_x.shape[0]+(N_extra_p*2),s1_x.shape[0]+(N_extra_p*2)))
    # plt.title("Z coordinates after modification")

    # plt.figure()
    # plt.imshow(x_ext.reshape(s1_x.shape[0]+(N_extra_p*2),s1_x.shape[0]+(N_extra_p*2)))
    # plt.title("X coordinates after modification")

    # plt.figure()
    # plt.imshow(y_ext.reshape(s1_x.shape[0]+(N_extra_p*2),s1_x.shape[0]+(N_extra_p*2)))
    # plt.title("Y coordinates after modification")

    ##We now use the new arrays to generate the step file
    points_extended = np.stack((x_ext,y_ext,z_ext),axis=-1)

    # plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(x_ext,y_ext,z_ext,color='red',alpha=0.3)
    # ax.contour3D(s1[0].reshape(Nx,Ny),s1[1].reshape(Nx,Ny),s1[2].reshape(Nx,Ny),50,cmap='viridis')
    # plt.show()
    # points_to_surf(points_extended.reshape(3,s1_x.shape[0]+N_extra_p,s1_x.shape[1]+N_extra_p),"name")
    return points_extended



def points_to_surf(p,name,approx_type):

    p_center = p[2][int(p[2].shape[0]*0.5),int(p[2].shape[1]*0.5)]
    array = TColgp_Array2OfPnt(1,p.shape[1],1,p.shape[2])
    print ("shape of array")
    print (type(array))
    for i in range(0,p.shape[1],1):
        for j in range(0,p.shape[2],1):
            # point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j]-p_center) ##Original - We shift the points so that the apex is at (0,0,0)
            point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j])
            array.SetValue(i+1,j+1,point_to_add)
    # for i in range(0,p.shape[1],1):
    #     array.SetValue(i+1,gp_Pnt(*p[:,i]))

    print ("Surface creation")
    bspl_surface = GeomAPI_PointsToBSplineSurface(array,Approx_ChordLength,3,5,GeomAbs_C2,1e-4)#.Interpolate(array) ###For fitting
    # bspl_surface = GeomAPI_PointsToBSplineSurface(array,1.0,1.0,1.0,8,GeomAbs_C2,1e-3)#.Interpolate(array)
    # bspl_surface = GeomAPI_PointsToBSplineSurface(array,3,8,GeomAbs_C2,1e-1)#.Interpolate(array)
    # bspl = bspl_surface.Surface()
    # print ("number of knots and poles")
    # print (bspl_surface.Surface().FirstUKnotIndex())
    # print (bspl_surface.Surface().LastUKnotIndex())
    # print (bspl_surface.Surface().NbVPoles())
    # bspl_surface = GeomAPI_PointsToBSplineSurface()#.Interpolate(array)
    # bspl_surface.Interpolate(array,approx_type) ##This worked for the q surface
    # bspl_surface.Interpolate(array,Approx_IsoParametric) ##This worked for the q surface
    # bspl_surface.Interpolate(array,Approx_Centripetal)
    # bspl_surface2 = GeomAPI_PointsToBSplineSurface(array,3,8,GeomAbs_C2,1e-4)
    face_builder = BRepBuilderAPI_MakeFace(bspl_surface.Surface(),1e-6).Shape()

    ##Can we also generate the STL file directly here using pythonocc??
    # write_stl_file(face_builder,"p_surface_pythonocc.STL",linear_deflection=0.01,angular_deflection=0.01)

    c= STEPControl_Controller()
    c.Init()
    step_writer = STEPControl_Writer()
    Interface_Static("write.step.schema","AP214")
    step_writer.Transfer(face_builder,STEPControl_AsIs)
    filename="dip_meep_to_gauss_ip_visio_res251/"+"ptg_" + name + "_res251_dip_meep_to_gauss_ip_visio.step"
    # step_writer.Write(filename)
    display.DisplayShape(bspl_surface.Surface(),update=True)
    # for i in range(0,p.shape[1],1):
    #     for j in range(0,p.shape[2],1):
    #         point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j]-p_center)
            # point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j])
            # display.DisplayShape(point_to_add,update=False)
    # display.Repaint()
    return bspl_surface.Surface()

def points_to_surf_inter(p,name):

    p_center = p[2][int(p[2].shape[0]*0.5),int(p[2].shape[1]*0.5)]
    array = TColgp_Array2OfPnt(1,p.shape[1],1,p.shape[2])
    print ("shape of array")
    print (type(array))
    for i in range(0,p.shape[1],1):
        for j in range(0,p.shape[2],1):
            # point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j]-p_center) ##Original - We shift the points so that the apex is at (0,0,0)
            point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j])
            array.SetValue(i+1,j+1,point_to_add)
    # for i in range(0,p.shape[1],1):
    #     array.SetValue(i+1,gp_Pnt(*p[:,i]))

    print ("Surface creation")
    bspl_surface = GeomAPI_PointsToBSplineSurface(array,Approx_IsoParametric,3,5,GeomAbs_C3,1e-5)#.Interpolate(array)
    # bspl_surface = GeomAPI_PointsToBSplineSurface()#.Interpolate(array)
    # bspl_surface.Interpolate(array,Approx_ChordLength)
    # bspl_surface2 = GeomAPI_PointsToBSplineSurface(array,3,8,GeomAbs_C2,1e-4)
    face_builder = BRepBuilderAPI_MakeFace(bspl_surface.Surface(),1e-6).Shape()



    c= STEPControl_Controller()
    c.Init()
    step_writer = STEPControl_Writer()
    Interface_Static("write.step.schema","AP214")
    step_writer.Transfer(face_builder,STEPControl_AsIs)
    filename="ptg_" + name + "_res251.step"
    step_writer.Write(filename)
    display.DisplayShape(bspl_surface.Surface(),update=True)
    # for i in range(0,p.shape[1],1):
    #     for j in range(0,p.shape[2],1):
    #         point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j])
    #         display.DisplayShape(point_to_add,update=False)
    display.Repaint()
    return bspl_surface.Surface()

def build_points_network(bspl_srf):
    face = BRepBuilderAPI_MakeFace(bspl_srf,1e-6).Face()
    ##get face uv bounds
    umin, umax, vmin, vmax = shapeanalysis.GetFaceUVBounds(face)
    print (umin,umax,vmin,vmax)
    pnts =[]
    pnts_x = []
    pnts_y = []
    pnts_z = []
    sas = ShapeAnalysis_Surface(bspl_srf)
    u = umin
    while u <=umax:
        v = vmin
        while v<= vmax:
            p = sas.Value(u,v)
            pnts_x.append(p.X())
            pnts_y.append(p.Y())
            pnts_z.append(p.Z())
            v += 0.1
        u += 0.1
    print ("finished")
    return np.asarray(pnts_x),np.asarray(pnts_y),np.asarray(pnts_z)

def build_points_network2(bspl_surf,Nx,Ny,u,v):
    face = BRepBuilderAPI_MakeFace(bspl_surf, 1e-6).Face()
    ##get face uv bounds
    umin, umax, vmin, vmax = shapeanalysis.GetFaceUVBounds(face)
    print(umin, umax, vmin, vmax)
    # u = np.linspace(umin,umax,Nx)
    # v = np.linspace(vmin,vmax,Ny)
    sas = ShapeAnalysis_Surface(bspl_surf)
    pnts_x = []
    pnts_y = []
    pnts_z = []
    # for i in range(0,u.shape[0],1):
    #     p = sas.Value(u[i], v[i])
    #     pnts_x.append(p.X())
    #     pnts_y.append(p.Y())
    #     pnts_z.append(p.Z())

    for ui in u:
        for vj in v:
            p = sas.Value(ui,vj)
            pnts_x.append(p.X())
            pnts_y.append(p.Y())
            pnts_z.append(p.Z())
    print ("finished")
    return np.asarray(pnts_x),np.asarray(pnts_y),np.asarray(pnts_z),sas,u,v

def prism():
    # the bspline profile
    array = TColgp_Array1OfPnt(1, 5)
    array.SetValue(1, gp_Pnt(0, 0, 0))
    array.SetValue(2, gp_Pnt(1, 2, 0))
    array.SetValue(3, gp_Pnt(2, 3, 0))
    array.SetValue(4, gp_Pnt(4, 3, 0))
    array.SetValue(5, gp_Pnt(5, 5, 0))
    bspline = GeomAPI_PointsToBSpline(array).Curve()
    profile = BRepBuilderAPI_MakeEdge(bspline).Edge()

    # the linear path
    starting_point = gp_Pnt(0.0, 0.0, 0.0)
    end_point = gp_Pnt(0.0, 0.0, 6.0)
    vec = gp_Vec(starting_point, end_point)
    path = BRepBuilderAPI_MakeEdge(starting_point, end_point).Edge()

    # extrusion
    prism = BRepPrimAPI_MakePrism(profile, vec).Shape()

    display.DisplayShape(profile, update=False)
    display.DisplayShape(starting_point, update=False)
    display.DisplayShape(end_point, update=False)
    display.DisplayShape(path, update=False)
    display.DisplayShape(prism, update=True)

def points_from_surface(bspl_surf,Nx,Ny):

    face = BRepBuilderAPI_MakeFace(bspl_surf,1e-8).Face()
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
    # u = umin
    # while u <= umax:
    #     v = vmin
    #     while v <= vmax:
    #         p = sas.Value(u, v)
    #         pnts_x.append(p.X())
    #         pnts_y.append(p.Y())
    #         pnts_z.append(p.Z())
    #         v += 0.01
    #     u += 0.01
    print("finished")
    return np.asarray(pnts_x),np.asarray(pnts_y),np.asarray(pnts_z)


if __name__ == "__main__":

    filename = "ff1_dip_meep_z50nm_to_gaus_res251_n_IP_Visio_v_real"
    filename2 = "ff2_dip_meep_z50nm_to_gaus_res251_n_IP_Visio_v_real_further"
    # filename3 = "ff3_dip_meep_z50nm_to_gaus_res251_n_IP_Visio_v_real_further" ##Original
    filename3 = "ff3_dip_meep_z50nm_to_gaus_res251_n_IP_Visio_smooth_boundary" ##With smoothed contour

    Nx =251
    Ny =251


    s1, N1, out_dir1 = surf_param_read(filename)

    s2, N2 = surf_params_sN_read(filename2)
    s3, N3 = surf_params_sN_read(filename3)

    s_single, N_single = surf_params_sN_read("rect_to_circ_L2_OMT_surf")
    print ("shape of s3")
    print (s3.shape)
    Nx2 = 594
    Ny2 = 594

    display, start_display, add_menu, add_function_to_menu = init_display()


    spline_surf = points_to_surf_inter((s_single).reshape(3,Nx,Ny),"s_single_OMT")

    shape_desired = 251


    surf_x, surf_y, surf_z = points_from_surface(spline_surf,shape_desired,shape_desired)

    start_display()

    # shape_desired = int(np.sqrt(surf_x.shape[0]))

    ##We try to use the RectBivariateSpline to upsample the points using the original distribution of points
    # u = np.linspace(-1.0,1.0,Nx)
    # v = np.linspace(-1.0,1.0,Nx)
    # x_sp = RectBivariateSpline(u,v,s3[0,:].reshape(Nx,Ny),s=0)
    # y_sp = RectBivariateSpline(u,v,s3[1,:].reshape(Nx,Ny),s=0)
    # z_sp = RectBivariateSpline(u,v,s3[2,:].reshape(Nx,Ny),s=0)
    # u2 = np.linspace(-1.0,1.0,shape_desired)
    # v2 = np.linspace(-1.0,1.0,shape_desired)
    #
    # x_sp_ev = x_sp(u2,v2,grid=True)
    # y_sp_ev = y_sp(u2,v2,grid=True)
    # z_sp_ev = z_sp(u2,v2,grid=True)


    # surf_x2 = x_sp_ev.reshape(shape_desired,shape_desired)#.transpose()
    # surf_y2 = y_sp_ev.reshape(shape_desired,shape_desired)#.transpose()
    # surf_z2 = z_sp_ev.reshape(shape_desired,shape_desired)#.transpose()
    # print ("shape of surf_x2")
    # print (surf_x2.shape)

    # s3_2 = np.stack((surf_x2.flatten(),surf_y2.flatten(),surf_z2.flatten()))


    # display, start_display, add_menu, add_function_to_menu = init_display()

    # spline_surf_2 = points_to_surf((s3_2).reshape(3,shape_desired,shape_desired),"q_smooth",Approx_IsoParametric)

    # start_display()


    # filepath="C:\\Users\\itojimenez\\PycharmProjects\\pythonocc\\dip_meep_to_gauss_ip_visio_res251\\"


    # xyz_surf_save(surf_x2.flatten(),surf_y2.flatten(),surf_z2.flatten(),filepath+"q_surface_smooth_c_251.txt")



