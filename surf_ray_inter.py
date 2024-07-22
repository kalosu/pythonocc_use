import numpy as np
import sys, time, os
import random

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


def make_face(px, py, pz):
    nx, ny = px.shape
    print(px.shape, pz.shape)
    pnt_2d = TColgp_Array2OfPnt(1, nx, 1, ny)
    print(pnt_2d.LowerRow(), pnt_2d.UpperRow())
    print(pnt_2d.LowerCol(), pnt_2d.UpperCol())
    for row in range(pnt_2d.LowerRow(), pnt_2d.UpperRow() + 1):
        for col in range(pnt_2d.LowerCol(), pnt_2d.UpperCol() + 1):
            i, j = row - 1, col - 1
            pnt = gp_Pnt(px[i, j], py[i, j], pz[i, j])
            pnt_2d.SetValue(row, col, pnt)
            # print (row, col, i, j, px[i, j], py[i, j], pz[i, j])
    curve = GeomAPI_PointsToBSplineSurface(pnt_2d, 3, 8, GeomAbs_G2, 0.001).Surface()
    return curve


def reflection(h_surf, pnt, vec):
    ray = Geom_Line(gp_Lin(pnt, vec_to_dir(vec.Normalized())))
    # uvw = GeomAPI_IntCS(ray.GetHandle(), h_surf).Parameters(1)
    uvw = GeomAPI_IntCS(ray, h_surf).Parameters(1)
    u, v, w = uvw
    p, vx, vy = gp_Pnt(), gp_Vec(), gp_Vec()
    GeomLProp_SurfaceTool.D1(h_surf, u, v, p, vx, vy)
    vz = vx.Crossed(vy)
    vx.Normalize()
    vy.Normalize()
    vz.Normalize()
    v = v0.Mirrored(gp_Ax2(pnt, gp_Dir(vz)))
    return p, v,vz


if __name__ == '__main__':
    lxy = [-100, -100, 100, 100]
    nxy = [300, 400]
    px = np.linspace(lxy[0], lxy[2], nxy[0])
    py = np.linspace(lxy[1], lxy[3], nxy[1])
    print(px[0], px[-1], px.shape)
    print(py[0], py[-1], py.shape)
    mesh = np.meshgrid(px, py)
    surf = mesh[0] ** 2 / 500 + mesh[1] ** 2 / 600 + mesh[0] * mesh[1] / 20000  # arbitrary function or data
    spl_geom = make_face(*mesh, surf)
    spl_objt = spl_geom#.GetObject()
    spl_face = BRepBuilderAPI_MakeFace(spl_geom, 0, 1, 0, 1, 0.001).Face()

    ax0 = gp_Ax3()
    ax1 = gp_Ax3(gp_Pnt(0, 0, 100), gp_Dir(0, 0, 1))
    trf_face = gp_Trsf()
    trf_face.SetTransformation(ax1, ax0)
    loc_face = TopLoc_Location(trf_face)
    spl_face.Location(loc_face)
    spl_surf = BRep_Tool.Surface(spl_face)
    print(spl_geom, spl_objt)
    print(spl_face, spl_surf)

    p0, v0 = gp_Pnt(), gp_Vec(0, -0.1, 1.0).Normalized()
    p1, v1,Normal = reflection(spl_surf, p0, v0)
    p2 = gp_Pnt((gp_Vec(p1.XYZ()) + v1 *1).XYZ())
    p3 = gp_Pnt((gp_Vec(p1.XYZ()) + Normal*1).XYZ())
    ray01 = make_edge(p0, p1)
    ray12 = make_edge(p1, p2)
    ray1N = make_edge(p1, p3)

    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(spl_face)
    display.DisplayVector(v0 * 10, p0)
    display.DisplayVector(v1 * 10, p1)
    display.DisplayShape(ray01)
    display.DisplayShape(ray12)
    display.DisplayShape(ray1N)

    display.FitAll()
    start_display()
