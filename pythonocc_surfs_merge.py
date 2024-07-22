import numpy as np
import matplotlib.pyplot as plt
from common_functions import surf_param_read,surf_params_sN_read,xy_target_coords_read,uv_plane_point,stereo_proj_dir,circ_boundary_delim,surf_N_eval,output_dir_eval
from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt, gp_Ax1, gp_Ax2, gp_Ax3, gp_Vec, gp_Dir,gp_Circ
from OCC.Core.gp import gp_Pln, gp_Lin, gp_Translation, gp_Trsf
from OCC.Core.Geom import Geom_Plane, Geom_Surface, Geom_BSplineSurface,Geom_Circle
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
    # for i in range(0,p.shape[1],1):
    #     array.SetValue(i+1,gp_Pnt(*p[:,i]))

    print ("Surface creation")
    # bspl_surface = GeomAPI_PointsToBSplineSurface(array,Approx_IsoParametric,3,8,GeomAbs_C2,1e-2)#.Interpolate(array)
    # bspl_surface = GeomAPI_PointsToBSplineSurface(array,1.0,1.0,1.0,8,GeomAbs_C2,1e-3)#.Interpolate(array)
    bspl_surface = GeomAPI_PointsToBSplineSurface(array,3,8,GeomAbs_C2,1e-4)#.Interpolate(array)
    return bspl_surface

def create_base_surface(radius,center,height):
    circ = gp_Circ(gp_Ax2(center, gp_Dir(0, 0, 1)), radius)
    circ_geom = Geom_Circle(circ)
    base_face = BRepBuilderAPI_MakeFace(circ_geom).Face()
    return base_face

if __name__ == "__main__":
    filename = "ff1_p65_to_gaus_res201_n_IP_S"
    Nx =201
    Ny =201

    s1, N1, out_dir1 = surf_param_read(filename)


    spline_surf = points_to_surf((s1).reshape(3,Nx,Ny),"p")
    radius = 4.0
    base_center = gp_Pnt(0,0,0)
    base_surface = create_base_surface(radius,base_center,0)


