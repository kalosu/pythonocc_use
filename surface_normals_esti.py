##Following the examples we have seen, we will try to use pythonOCC in order to calculate the surface normals of the generated approximation
##surface funtions that are used to generate the STEP files.
##We have noticed that if we simply try to interpolate the original points, the obtained surface is not necessarily smooth at some points close to the boundary.
##We therefore generate a LSQ approximation of the surface and then use the built numerical routines in order to calculate the intersections between
##the incident rays and the surface.
##For this we calculate the intersects using the spline functions and no discrete data points. In theory, this should give us better estimates on where are the
#rays being hitted on the surface. Following we can calculate the associated normals and the corresponding ray directions.
##All of this is done with the hope that we can improve the integrability of the second surface which as we have observed, suffers from convergence.
##This might be related to the poor integrability of the output wavefront which we have always calculated from the raw data points (p surface)
##However, as already observed, these surface data points do not result neither in a really smooth surface and at such might cause problems.
import numpy as np
import matplotlib.pyplot as plt
from common_functions import surf_param_read,surf_params_sN_read,xy_target_coords_read,uv_plane_point,stereo_proj_dir,circ_boundary_delim,surf_N_eval,output_dir_eval
from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt, gp_Ax1, gp_Ax2, gp_Ax3, gp_Vec, gp_Dir
from OCC.Core.gp import gp_Pln, gp_Lin, gp_Translation, gp_Trsf
from OCC.Core.Geom import Geom_Plane, Geom_Surface, Geom_BSplineSurface
from OCC.Core.Geom import Geom_Curve, Geom_Line
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_IntCS
from OCC.Core.GeomAbs import GeomAbs_C0, GeomAbs_C1, GeomAbs_G1, GeomAbs_G2
from OCC.Core.GeomLProp import GeomLProp_SurfaceTool
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Vertex, TopoDS_Builder, TopoDS_Face
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.BRep import BRep_Tool
from OCCUtils.Construct import make_plane, make_line, make_edge, make_vertex
from OCCUtils.Construct import dir_to_vec, vec_to_dir
from OCCUtils.Common import project_point_on_plane
from OCC.Core.Approx import Approx_IsoParametric,Approx_Centripetal
from OCC.Core.GeomAbs import GeomAbs_C3,GeomAbs_C2
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface,shapeanalysis
from common_functions import output_dir_eval

def build_points_network2(bspl_surf,Nx,Ny):
    face = BRepBuilderAPI_MakeFace(bspl_surf, 1e-6).Face()
    ##get face uv bounds
    umin, umax, vmin, vmax = shapeanalysis.GetFaceUVBounds(face)
    print(umin, umax, vmin, vmax)
    u = np.linspace(umin,umax,Nx)
    v = np.linspace(vmin,vmax,Ny)
    sas = ShapeAnalysis_Surface(bspl_surf)
    pnts_x = []
    pnts_y = []
    pnts_z = []
    for ui in u:
        for vj in v:
            p = sas.Value(ui,vj)
            pnts_x.append(p.X())
            pnts_y.append(p.Y())
            pnts_z.append(p.Z())
    print ("finished")
    return np.asarray(pnts_x),np.asarray(pnts_y),np.asarray(pnts_z),sas,u,v


def points_to_surf(p,name):

    p_center = p[2][int(p[2].shape[0]*0.5),int(p[2].shape[1]*0.5)]
    array = TColgp_Array2OfPnt(1,p.shape[1],1,p.shape[2])
    print ("shape of array")
    print (type(array))
    for i in range(0,p.shape[1],1):
        for j in range(0,p.shape[2],1):
            # point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j]-p_center) ##Original - We shift the points so that the apex is at (0,0,0)
            point_to_add = gp_Pnt(p[0,i,j],p[1,i,j],p[2,i,j])
            array.SetValue(i+1,j+1,point_to_add)

    print ("Surface creation")
    # bspl_surface = GeomAPI_PointsToBSplineSurface(array,Approx_Centripetal,3,8,GeomAbs_C3,1e-3)#.Interpolate(array)

    bspl_surface = GeomAPI_PointsToBSplineSurface()#.Interpolate(array)
    bspl_surface.Interpolate(array)
    # bspl_surface2 = GeomAPI_PointsToBSplineSurface(array,3,8,GeomAbs_C2,1e-4)
    face_builder = BRepBuilderAPI_MakeFace(bspl_surface.Surface(),1e-6).Shape()

    # c= STEPControl_Controller()
    # c.Init()
    # step_writer = STEPControl_Writer()
    # Interface_Static("write.step.schema","AP214")
    # step_writer.Transfer(face_builder,STEPControl_AsIs)
    # filename="ptg_" + name + "_res151.step"
    # step_writer.Write(filename)

    return bspl_surface

def surf_norm(surf,pnt,vec):
    ray = Geom_Line(gp_Lin(pnt,vec_to_dir(vec.Normalized())))
    u,v,w= GeomAPI_IntCS(ray,surf).Parameters(1)
    p,vx,vy = gp_Pnt(), gp_Vec(),gp_Vec()
    GeomLProp_SurfaceTool.D1(surf,u,v,p,vx,vy)
    vz = vx.Crossed(vy)
    vx.Normalize()
    vy.Normalize()
    vz.Normalize()
    return p, vz

def surf_norm2(surf,u,v):
    p_x= []
    p_y= []
    p_z= []
    Nx = []
    Ny = []
    Nz = []
    for ui in u:
        for vj in v:
            p,vx,vy = gp_Pnt(),gp_Vec(),gp_Vec()
            GeomLProp_SurfaceTool.D1(surf,ui,vj,p,vx,vy)
            vz = vx.Crossed(vy)
            vx.Normalize()
            vy.Normalize()
            vz.Normalize()
            p_x.append(p.X())
            p_y.append(p.Y())
            p_z.append(p.Z())

            Nx.append(vz.X())
            Ny.append(vz.Y())
            Nz.append(vz.Z())
    p_total = np.stack((np.asarray(p_x),np.asarray(p_y),np.asarray(p_z)),axis=0)
    N= np.stack((np.asarray(Nx),np.asarray(Ny),np.asarray(Nz)),axis=0)
    return p_total,N



if __name__=='__main__':
    pathname = "C:\\Users\\itojimenez\\PycharmProjects\\beam_shaping_3D_freeform\\"
    filename = "ff1_point65_to_gaus_ts_r60_res91_I_b_equal"

    Nx = 91
    Ny = 91
    nlens = 1.5
    n2 = 1

    s1, N1, out_dir1 = surf_param_read(filename)

    s1 = s1.reshape(3,Nx,Ny)

    filepath = "C:\\Users\\itojimenez\\PycharmProjects\\beam_shaping_3D_freeform\\mapping_files\\"

    uuc, vvc = xy_target_coords_read(filepath+"omt_point_source_equi_flux_angle_65_res91_alpha001.txt")
    angle = 65.0
    r_max_val = uv_plane_point(angle)

    uuc2, vvc2 = circ_boundary_delim(
        np.copy(uuc.reshape(Nx, Ny)), np.copy(vvc.reshape(Nx, Ny)), r_max_val
    )

    u = np.linspace(-1.0, 1.0, Nx)
    v = np.linspace(-1.0, 1.0, Ny)
    uu, vv = np.meshgrid(u, v)
    uu2 = uu * np.sqrt(1 - 0.5 * (vv ** 2)) * r_max_val
    vv2 = vv * np.sqrt(1 - 0.5 * (uu ** 2)) * r_max_val


    I_dir = stereo_proj_dir(uuc2.reshape(Nx, Ny), vvc2.reshape(Nx, Ny)).reshape(3,Nx,Ny)


    spline_surf = points_to_surf(s1.reshape(3,Nx,Ny),"p")

    step_x, step_y, step_z,sas,u,v = build_points_network2(spline_surf.Surface(),Nx,Ny)
    s1_step = np.stack((step_x,step_y,step_z),axis=0)


    display, start_display, add_menu, add_function_to_menu = init_display()

    display.DisplayShape(spline_surf.Surface(),update=True)
    # for i in range(0,s1.shape[1],1):
    #     for j in range(0,s1.shape[2],1):
    #         point_to_add = gp_Pnt(s1[0,i,j],s1[1,i,j],s1[2,i,j])
    #         # point_to_add = gp_Pnt(I_dir[0,i,j],I_dir[1,i,j],I_dir[2,i,j])
    #         display.DisplayShape(point_to_add,update=False)
    # display.Repaint()
    display.FitAll()
    start_display()

    # ###We repeat the same thing but for the first surface
    #
    # spline_surf_p = points_to_surf(s1.reshape(3,Nx,Ny),"p")
    #
    # display, start_display, add_menu, add_function_to_menu = init_display()
    #
    # display.DisplayShape(spline_surf_p.Surface(),update=True)
    # for i in range(0,s1.shape[1],1):
    #     for j in range(0,s1.shape[2],1):
    #         point_to_add = gp_Pnt(s1[0,i,j],s1[1,i,j],s1[2,i,j])
    #         display.DisplayShape(point_to_add,update=False)
    # display.Repaint()
    # display.FitAll()
    # start_display()




    ##We now try to calculate the intersection of the incident rays with the surface
    spline_face = BRepBuilderAPI_MakeFace(spline_surf.Surface(),1e-6).Face()
    spline_surf2 = BRep_Tool.Surface(spline_face)

    ##We set the origin point and the ray direction
    # p0, v0 = gp_Pnt(), gp_Vec(I_dir[0,0],I_dir[1,0],I_dir[2,0]).Normalized()
    p0 = gp_Pnt()
    ##we calculate now the surface normal and the point of intersection with the ray
    px = []
    py = []
    pz = []
    Nx1 = []
    Ny1 = []
    Nz1 = []
    I_dir = I_dir.reshape(3,-1)

    p12, N12 = surf_norm2(spline_surf2,u,v)

    r_max = np.max(np.sqrt(s1[0]**2+s1[1]**2))

    circle = plt.Circle((0, 0), r_max, color="r", fill=False)

    fix, ax = plt.subplots()
    ax.scatter(s1[0],s1[1],label='Original points')
    ax.add_patch(circle)
    plt.legend()
    plt.title("XY components - s1")



    plt.figure()
    plt.scatter(N12[0],N12[1],label='Computed output directions')
    plt.scatter(I_dir[0],I_dir[1],label='Original directions')
    plt.legend()

    plt.figure()
    plt.imshow(N12[0].reshape(Nx, Ny))
    plt.title("N_x")

    plt.figure()
    plt.imshow(N12[1].reshape(Nx, Ny))
    plt.title("N_y")
    plt.show()


    for i in range(0,Nx*Ny,1):
        # print ("Point number:{}".format(i))
        v0 = gp_Vec(I_dir[0,i],I_dir[1,i],I_dir[2,i]).Normalized()
        p1, N1 = surf_norm(spline_surf2,p0,v0)
        px.append(p1.X())
        py.append(p1.Y())
        pz.append(p1.Z())
        Nx1.append(N1.X())
        Ny1.append(N1.Y())
        Nz1.append(N1.Z())

    px = np.asarray(px)
    py = np.asarray(py)
    pz = np.asarray(pz)
    Nx1 = np.asarray(Nx1)
    Ny1 = np.asarray(Ny1)
    Nz1 = np.asarray(Nz1)
    p = np.stack((px,py,py),axis=0)
    N = np.stack((Nx1,Ny1,Nz1),axis=0)
    out_dir,_ = output_dir_eval(N,I_dir,nlens,n2)

    plt.figure()
    plt.scatter(out_dir[0],out_dir[1],label='Computed output directions')
    plt.scatter(I_dir[0],I_dir[1],label='Original directions')
    plt.legend()

    plt.figure()
    plt.imshow(out_dir[0].reshape(Nx,Ny))
    plt.title("out_dir_x")

    plt.figure()
    plt.imshow(out_dir[1].reshape(Nx,Ny))
    plt.title("out_dir_y")

    plt.figure()
    plt.imshow(N[0].reshape(Nx, Ny))
    plt.title("N_x")

    plt.figure()
    plt.imshow(N[1].reshape(Nx, Ny))
    plt.title("N_y")
    plt.show()

    # p2 = gp_Pnt((gp_Vec(p1.XYZ())+N1).XYZ())
    # ray01 = make_edge(p0,p1)
    # ray1N = make_edge(p1,p2)

    # display, start_display, add_menu, add_function_to_menu = init_display()
    # display.DisplayShape(spline_surf2)
    # display.DisplayVector(v0*2,p0)
    # display.DisplayShape(ray01)
    # display.DisplayShape(ray1N)
    # display.FitAll()
    # start_display()

