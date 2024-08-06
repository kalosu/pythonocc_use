##We use this script to generate a set of points (i.e, a point cloud) from a STEP file.
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline,GeomAPI_PointsToBSplineSurface
from OCC.Core.Approx import  Approx_Centripetal,Approx_IsoParametric,Approx_ChordLength
from OCC.Core.TColgp import TColgp_Array1OfPnt,TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.GeomAbs import GeomAbs_C2,GeomAbs_C3
from OCC.Display.SimpleGui import init_display
from OCC.Core.STEPControl import STEPControl_Writer,STEPControl_AsIs,STEPControl_Controller,STEPControl_Reader
from OCC.Core.Interface import Interface_Static
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.GeomConvert import geomconvert
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface,shapeanalysis
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.Tesselator import ShapeTesselator
from OCC.Extend.DataExchange import write_stl_file,read_step_file
from OCC.Core.STEPCAFControl import  STEPCAFControl_Reader
from OCC.Core.BRep import BRep_Tool
import math
import numpy as np
import matplotlib.pyplot as plt

def read_step_file(filename):
    step_reader = STEPControl_Reader()
    step_reader.ReadFile(filename)
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    surfaces =[]
    topExp = TopExp_Explorer(shape,TopAbs_FACE)
    while topExp.More():
        face = topExp.Current()
        surf = BRep_Tool.Surface(face)
        surfaces.append(surf)
        topExp.Next()
    return surfaces,shape

def get_bspline_surface(surf):
    if not surf.DynamicType().ToCString()=="Geom_BSplineSurface":
        surf = geomconvert

def sample_surface(surface,num_u,num_v):
    face = BRepBuilderAPI_MakeFace(surface,1e-8).Face()
    # u_min, u_max, v_min,v_max = surface.Bounds()
    u_min, u_max, v_min, v_max = shapeanalysis.GetFaceUVBounds(face)
    u_values = np.linspace(u_min,u_max,num_u)
    v_values = np.linspace(v_min,v_max,num_v)
    p_x =[]
    p_y = []
    p_z = []
    sas = ShapeAnalysis_Surface(surface)
    for u in u_values:
        for v in v_values:
            pnt = sas.Value(u,v)
            p_x.append(pnt.X())
            p_y.append(pnt.Y())
            p_z.append(pnt.Z())
    return np.asarray(p_x),np.asarray(p_y),np.asarray(p_z)

import matplotlib
matplotlib.use('Qt5Agg')

pathname = "dip_meep_to_gauss_ip_visio_res251/"

filename = pathname + "half_sphere_surface_smaller.STEP"

display, start_display, add_menu, add_function_to_menu = init_display()

surfaces, myshape = read_step_file(filename)
print ("content of surfaces")
print (type(surfaces[0]))

print (surfaces[0].Bounds())

Nx = 80
Ny = 80

p_x, p_y, p_z= sample_surface(surfaces[0],Nx,Ny)
print ("shape of points")
print ((p_x.shape))


# while faceExplorer.More():
#     faceExplorer.Current()

display.EraseAll()
display.DisplayShape(myshape,update=True)

start_display()

fig = plt.figure()
ax = plt.axes(projection='3d')
surf = ax.scatter(p_x.reshape(Nx,Ny), p_y.reshape(Nx,Ny), p_z.reshape(Nx,Ny))
plt.show()


