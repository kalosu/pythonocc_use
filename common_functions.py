###We will add all of the general auxiliary functions to this python file
import copy

import numpy as np
from scipy.integrate import dblquad
from scipy.interpolate import RectBivariateSpline,CloughTocher2DInterpolator,RBFInterpolator
# import open3d as o3d
import matplotlib.pyplot as plt
from scipy.optimize import fsolve,newton
import open3d as o3d



def stereo_proj_dir(u,v):
    I = np.stack((2*u,2*v,1-u**2-v**2)) / (1 + u**2+v**2)
    I_dir = np.stack((I[0, :, :].flatten(), I[1, :, :].flatten(), I[2, :, :].flatten()))
    return I_dir

def p_vec_eval(l0,dir,d):
    return l0 + dir*d

def uv_plane_z_0(d,x,y,z):
    dir = np.stack((x,y,z))
    l0 = np.stack((0,0,-1))
    p = l0 + d*dir
    return p[2]#**2

def uv_plane_point(angle):
    ###We first find the X and Z components and then we find the intersection of the line that originates at the south pole
##of the circle used for the stereographic projection
    Z = np.cos(np.deg2rad(angle))
    X = np.sin(np.deg2rad(angle))
    Y = 0
    P = np.stack((X,Y,Z))
    S = np.stack((0,0,-1))
    dir_vec = np.asarray((X,Y,Z+1))
    dir_vec /= np.linalg.norm(dir_vec)
    d = newton(uv_plane_z_0,x0=-1.0,args=(dir_vec[0],dir_vec[1],dir_vec[2]))
    p = S + dir_vec*d
    print ("Coordinates for UV plane")
    print (p[0])
    print (p[1])
    print (p[2])

    return p[0]

def stereo_proj_lamb_const(angle):
    r = np.linspace(0,1,500)
    phi = np.linspace(0,np.pi*2,500)
    rr, pphi = np.meshgrid(r,phi)
    u = rr*np.cos(pphi)
    v = rr*np.sin(pphi)
    E_uv = (4 * (1-u**2-v**2)) /(1+u**2+v**2)**3
    # r_max = np.sin(np.deg2rad(angle))
    # print ("Position based on sine")
    # print(r_max)
    r_max_uv_plane =uv_plane_point(angle)
    print ("Position based on fsolve")
    print (r_max_uv_plane)
    E_uv_sp = RectBivariateSpline(phi,r,E_uv*rr)
    return E_uv_sp.integral(0,np.pi*2,0,r_max_uv_plane),r_max_uv_plane

def stereo_proj_lamb_func():
    uc = np.linspace(-1.0,1.0,150)
    vc = np.linspace(-1.0,1.0,150)
    uuc, vvc = np.meshgrid(uc,vc)
    ##Transforming to circular shape
    u = uuc*np.sqrt(1-0.5*vvc**2)
    v = vvc*np.sqrt(1-0.5*uuc**2)
    E_uv = (4 * (1-u**2-v**2)) /(1+u**2+v**2)**3
    return E_uv,u,v

def target_irr_inter(target_irr,x,y):
    return CloughTocher2DInterpolator(list(zip(x.flatten(),y.flatten())),target_irr.flatten())

def xy_target_coords_save(x,y,filename):
    with open(filename,"w") as f:
        for i in range(0,x.shape[0],1):
            f.write(str(x[i]) + " " + str(y[i]))
            f.write("\n")
    f.close()

def xyz_surf_save(x,y,z,filename):
    with open(filename,"w") as f:
        for i in range(0,x.shape[0],1):
            f.write(str(x[i]) + " " + str(y[i]) + " " + str(z[i]))
            f.write("\n")
    f.close()

def surf_param_save(s,N,out_dir,filename):
    fn_surface = 'surf_' + filename + '.txt'
    fn_N= 'N_' + filename + '.txt'
    fn_out_dir= 'out_dir_' + filename + '.txt'
    out_dir2 = np.reshape(out_dir,((3,-1)))
    with open(fn_surface,"w") as f:
        for i in range(0,s.shape[1],1):
            txt = str(s[0,i]) + " " + str(s[1,i]) + " " + str(s[2,i])
            f.write(txt)
            f.write("\n")
    f.close()
    with open(fn_N,"w") as f:
        for i in range(0,N.shape[1],1):
            txt = str(N[0,i]) + " " + str(N[1,i]) + " " + str(N[2,i])
            f.write(txt)
            f.write("\n")
    f.close()
    with open(fn_out_dir,"w") as f:
        for i in range(0,out_dir2.shape[1],1):
            txt = str(out_dir2[0,i]) + " " + str(out_dir2[1,i]) + " " + str(out_dir2[2,i])
            f.write(txt)
            f.write("\n")
    f.close()

def surf_param_sN_save(s, N,filename):
    fn_surface = 'surf_' + filename + '.txt'
    fn_N = 'N_' + filename + '.txt'
    with open(fn_surface, "w") as f:
        for i in range(0, s.shape[1], 1):
            txt = str(s[0, i]) + " " + str(s[1, i]) + " " + str(s[2, i])
            f.write(txt)
            f.write("\n")
    f.close()

    with open(fn_N,"w") as f:
        for i in range(0,N.shape[1],1):
            txt = str(N[0,i]) + " " + str(N[1,i]) + " " + str(N[2,i])
            f.write(txt)
            f.write("\n")
    f.close()

def surf_params_sN_read(filename):

    pathname = "C:\\Users\\itojimenez\\PycharmProjects\\beam_shaping_3D_freeform\\surf_files_paper\\"
    # fn_surface = 'surface_files/'+'surf_' + filename + '.txt'
    fn_surface = pathname+'surf_' + filename + '.txt'
    # fn_N= 'surface_files/'+ 'N_' + filename + '.txt'
    fn_N= pathname+ 'N_' + filename + '.txt'
    sx = []
    sy = []
    sz = []
    with open(fn_surface, "r") as f:
        for line in f:
            line_p = line.split()
            sx.append(float(line_p[0]))
            sy.append(float(line_p[1]))
            sz.append(float(line_p[2]))
    s = np.stack((np.asarray(sx),np.asarray(sy),np.asarray(sz)))
    Nx = []
    Ny = []
    Nz = []
    with open(fn_N, "r") as f:
        for line in f:
            line_p = line.split()
            Nx.append(float(line_p[0]))
            Ny.append(float(line_p[1]))
            Nz.append(float(line_p[2]))
    N = np.stack((np.asarray(Nx),np.asarray(Ny),np.asarray(Nz)))
    return s, N

def surf_param_read(filename):
    pathname = "C:\\Users\\itojimenez\\PycharmProjects\\beam_shaping_3D_freeform\\surf_files_paper\\"
    # fn_surface = 'surface_files/'+'surf_' + filename + '.txt'
    fn_surface = pathname +'surf_'+filename + '.txt'
    # fn_N= 'surface_files/'+'N_' + filename + '.txt'
    fn_N= pathname+'N_' + filename + '.txt'
    # fn_out_dir= 'surface_files/' + 'out_dir_' + filename + '.txt'
    fn_out_dir= pathname+ 'out_dir_' + filename + '.txt'
    sx = []
    sy = []
    sz = []
    with open(fn_surface, "r") as f:
        for line in f:
            line_p = line.split()
            sx.append(float(line_p[0]))
            sy.append(float(line_p[1]))
            sz.append(float(line_p[2]))
    s = np.stack((np.asarray(sx),np.asarray(sy),np.asarray(sz)))
    Nx = []
    Ny = []
    Nz = []
    with open(fn_N, "r") as f:
        for line in f:
            line_p = line.split()
            Nx.append(float(line_p[0]))
            Ny.append(float(line_p[1]))
            Nz.append(float(line_p[2]))
    N = np.stack((np.asarray(Nx),np.asarray(Ny),np.asarray(Nz)))
    out_dir_x = []
    out_dir_y = []
    out_dir_z = []
    with open(fn_out_dir, "r") as f:
        for line in f:
            line_p = line.split()
            out_dir_x.append(float(line_p[0]))
            out_dir_y.append(float(line_p[1]))
            out_dir_z.append(float(line_p[2]))
    out_dir = np.stack((np.asarray(out_dir_x),np.asarray(out_dir_y),np.asarray(out_dir_z)))

    return s, N, out_dir


def xy_target_coords_read(filename):
    x = []
    y = []
    with open(filename,"r") as f:
        for line in f:
            line_p =line.split()
            x.append(float(line_p[0]))
            y.append(float(line_p[1]))
    return np.asarray(x),np.asarray(y)

def circ_boundary_delim(x,y,rmax):
    x_b = np.hstack((x[0,:],x[1:,-1],x[-1,:][::-1][1:],x[:,0][::-1][1:]))
    y_b = np.hstack((y[0,:],y[1:,-1],y[-1,:][::-1][1:],y[:,0][::-1][1:]))
    r_b = np.sqrt(x_b**2+y_b**2)
    # r_b_i = np.where(np.abs(rmax-r_b)>0.0001) #Original
    r_b_i = np.where(np.abs(rmax-r_b)>0.00000001) ##Original
    # r_b_i = np.where(np.abs(rmax-r_b)>0.01) #Original
    ##We create a line going from the origin (0,0) to the x and y coordinates that were found.
    ##We then displace both x and y in such a way that the distance from the origin is equal to the desired one
    angles = np.arctan2(y_b,x_b)
    angles = np.where(angles<0,angles+2*np.pi,angles)
    x_b[r_b_i] = rmax*np.cos(angles[r_b_i])
    y_b[r_b_i] = rmax*np.sin(angles[r_b_i])
    a = int(x[0,:].shape[0])
    b = a + int(x[1:,-1].shape[0])
    c = b + int(x[-1,:][::-1][1:].shape[0])
    d = c + int(x[:,0][::-1][1:].shape[0])
    x[0,:] = x_b[0:a]
    y[0,:] = y_b[0:a]
    x[1:,-1] = x_b[a:b]
    y[1:,-1] = y_b[a:b]
    x[-1,:][::-1][1:] = x_b[b:c]
    y[-1,:][::-1][1:] = y_b[b:c]
    x[:,0][::-1][1:] = x_b[c:d]
    y[:,0][::-1][1:] = y_b[c:d]
    return x,y


def zmx_grid_sag_write(dx,dy,z,res,filename):


    with open(filename,"w") as f:
        header = str(res) + " " + str(res) + " " + str(dx) + " " + str(dy) + " " + str(0) + " " + str(0) + " " + str(0)
        f.write(header)
        f.write("\n")
        for i in range(0,z.shape[0],1):
            line = str(z[i]) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(0)
            f.write(line)
            f.write("\n")
    f.close()

def single_surf_save(p_points,N_points,p_name,res,Nx,Ny,s,stl_flag):
    stl_dir = "C:\\Users\\itojimenez\\Documents\\Zemax\\Objects\\CAD Files\\"
    px = p_points[0, :].reshape(Nx, Ny)
    py = p_points[1, :].reshape(Nx, Ny)
    pz = p_points[2, :].reshape(Nx, Ny)
    N_x = N_points[0, :].reshape(Nx, Ny)
    N_y = N_points[1, :].reshape(Nx, Ny)
    N_z = N_points[2, :].reshape(Nx, Ny)
    p_center = pz[int(pz.shape[0]*0.5),int(pz.shape[1]*0.5)]
    print("Distance between the center of the first freeform surface and the origin")
    print(p_center)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(px, py, pz)
    plt.figure()
    plt.plot(px[int(px.shape[0] * 0.5), :], pz[int(pz.shape[0] * 0.5), :], label='P surface -X')
    plt.plot(py[:,int(py.shape[1] * 0.5)], pz[:,int(pz.shape[0] * 0.5)], label='P surface -Y',linestyle='--')
    plt.legend()
    plt.show()

    if stl_flag:
        x_d = np.linspace(-1.0, 1.0, Nx)
        y_d = np.linspace(-1.0, 1.0, Ny)
        px_sp = RectBivariateSpline(y_d, x_d, px, s=0)
        py_sp = RectBivariateSpline(y_d, x_d, py, s=0)
        pz_sp = RectBivariateSpline(y_d, x_d, pz - p_center, s=0)

        Nx_sp = RectBivariateSpline(y_d, x_d, N_x, s=0)
        Ny_sp = RectBivariateSpline(y_d, x_d, N_y, s=0)
        Nz_sp = RectBivariateSpline(y_d, x_d, N_z, s=0)

        x_d_eval = np.linspace(-1.0, 1.0, Nx * s)
        y_d_eval = np.linspace(-1.0, 1.0, Ny * s)

        px_sp_eval = px_sp(y_d_eval, x_d_eval)
        py_sp_eval = py_sp(y_d_eval, x_d_eval)
        pz_sp_eval = pz_sp(y_d_eval, x_d_eval)

        Nx_sp_eval = Nx_sp(y_d_eval, x_d_eval)
        Ny_sp_eval = Ny_sp(y_d_eval, x_d_eval)
        Nz_sp_eval = Nz_sp(y_d_eval, x_d_eval)

        ##We first generate the STL file for the first freeform surface
        print("We now create the STL file for the first freeform")
        pcd = o3d.geometry.PointCloud()
        points = np.stack((px_sp_eval.flatten(), py_sp_eval.flatten(), pz_sp_eval.flatten()), -1)
        Normals = np.stack((Nx_sp_eval.flatten(), Ny_sp_eval.flatten(), Nz_sp_eval.flatten()), -1)
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(Normals)
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 7 * avg_dist
        bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(
            [radius, radius]))
        o3d.visualization.draw_geometries([bpa_mesh])
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        name_p_full = stl_dir + "p_surf_" + p_name + ".stl"
        o3d.io.write_triangle_mesh(name_p_full, bpa_mesh)

def triple_surf_save(p_points,r_points,q_points,N_points,N_points_r,N_points_q,p_name,res,Nx,Ny,s,stl_flag):
    stl_dir = "C:\\Users\\itojimenez\\Documents\\Zemax\\Objects\\CAD Files\\"
    grid_dir = "C:\\Users\\itojimenez\\Documents\\Zemax\\Objects\\Grid Files\\"
    px = p_points[0, :].reshape(Nx, Ny)
    py = p_points[1, :].reshape(Nx, Ny)
    pz = p_points[2, :].reshape(Nx, Ny)
    rx = r_points[0, :].reshape(Nx, Ny)
    ry = r_points[1, :].reshape(Nx, Ny)
    rz = r_points[2, :].reshape(Nx, Ny)

    qx = q_points[0, :].reshape(Nx, Ny)
    qy = q_points[1, :].reshape(Nx, Ny)
    qz = q_points[2, :].reshape(Nx, Ny)

    N_x = N_points[0, :].reshape(Nx, Ny)
    N_y = N_points[1, :].reshape(Nx, Ny)
    N_z = N_points[2, :].reshape(Nx, Ny)

    N_x_r = N_points_r[0, :].reshape(Nx, Ny)
    N_y_r = N_points_r[1, :].reshape(Nx, Ny)
    N_z_r = N_points_r[2, :].reshape(Nx, Ny)

    N_x_q = N_points_q[0, :].reshape(Nx, Ny)
    N_y_q = N_points_q[1, :].reshape(Nx, Ny)
    N_z_q = N_points_q[2, :].reshape(Nx, Ny)

    p_center = pz[int(pz.shape[0]*0.5),int(pz.shape[1]*0.5)]
    q_center = qz[int(qz.shape[0]*0.5),int(qz.shape[1]*0.5)]
    r_center = rz[int(rz.shape[0]*0.5),int(rz.shape[1]*0.5)]
    print ("Distance between the center of the first freeform surface and the origin")
    print (p_center)
    print ("Distance between the center of the second and the first freeform surfaces")
    print (r_center - p_center)
    print ("Distance between center of second freeform surface and the origin")
    print (r_center)
    print ("Distance between the center of the third freeform and the second")
    rq_offset = q_center - r_center
    print (q_center - r_center)
    print ("Distance between center of third freeform and origin")
    print (q_center)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(px, py, pz)
    surf = ax.plot_surface(rx, ry, rz)
    surf = ax.plot_surface(qx, qy, qz)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(px, py, pz)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(qx, qy, qz)
    # ax.scatter(qx[0,0],qy[0,0],qz[0,0],color='red')
    # ax.scatter(qx[0,1],qy[0,1],qz[0,1],color='green')
    # ax.scatter(qx[0,2],qy[0,2],qz[0,2],color='red')

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(rx, ry, rz)
    # ax.scatter(rx[0,0],ry[0,0],rz[0,0],color='red')
    # ax.scatter(rx[0,1],ry[0,1],rz[0,1],color='green')
    # ax.scatter(rx[0,2],ry[0,2],rz[0,2],color='red')

    plt.figure()
    plt.plot(px[int(px.shape[0] * 0.5), :], pz[int(pz.shape[0] * 0.5), :], label='P surface -X')
    plt.plot(rx[int(rx.shape[0] * 0.5), :], rz[int(rz.shape[0] * 0.5), :], label='R surface -X')
    plt.plot(qx[int(qx.shape[0] * 0.5), :], qz[int(qz.shape[0] * 0.5), :], label='Q surface -X')
    # plt.legend()

    # plt.figure()
    plt.plot(py[:,int(py.shape[1] * 0.5)], pz[:,int(pz.shape[0] * 0.5)], label='P surface -Y',linestyle='--')
    plt.plot(ry[:,int(ry.shape[1] * 0.5)], rz[:,int(rz.shape[0] * 0.5)], label='R surface -Y',linestyle='--')
    plt.plot(qy[:,int(qy.shape[1] * 0.5)], qz[:,int(qz.shape[0] * 0.5)], label='Q surface -Y',linestyle='--')
    plt.legend()

    plt.figure()
    plt.plot(px[int(px.shape[0] * 0.5), :], pz[int(pz.shape[0] * 0.5), :], label='P surface -X')
    plt.plot(py[:,int(py.shape[1] * 0.5)], pz[:,int(pz.shape[0] * 0.5)], label='P surface -Y',linestyle='--')
    plt.legend()

    plt.figure()
    plt.plot(rx[int(rx.shape[0] * 0.5), :], rz[int(rz.shape[0] * 0.5), :], label='R surface -X')
    plt.plot(ry[:,int(ry.shape[1] * 0.5)], rz[:,int(rz.shape[0] * 0.5)], label='R surface -Y',linestyle='--')
    plt.legend()
    plt.show()

    if stl_flag:
        ##For the second freeform surface we use the RBF interpolation

        # print ("We now create the GRD files for the second freeform")
        # r_xy_points = np.stack((rx.flatten(),ry.flatten()),axis=-1)
        # r_z_rbf = RBFInterpolator(r_xy_points,(rz-r_center).flatten(),smoothing=0,kernel='cubic')
        # ##Remember that in order to store the Zmx file we need to save the date on an equidistant grid
        # x_rbf_eval = np.linspace(np.min(rx),np.max(rx),res)
        # dx = np.abs(x_rbf_eval[1]-x_rbf_eval[0])
        # y_rbf_eval = np.linspace(np.min(ry),np.max(ry),res)
        # dy = np.abs(y_rbf_eval[1]-y_rbf_eval[0])
        # x_rbf_eval,y_rbf_eval = np.meshgrid(x_rbf_eval,y_rbf_eval)
        # r_xy_points_eval = np.stack((x_rbf_eval.flatten(),y_rbf_eval.flatten()),-1)
        # r_z_rbf_eval = r_z_rbf(r_xy_points_eval)
        # name_q_1_full = grid_dir + "r_surf_" + p_name + ".GRD"
        # zmx_grid_sag_write(dx,dy,np.transpose(r_z_rbf_eval).flatten(),res,name_q_1_full)
        #
        # print ("We now create the GRD files for the third freeform")
        # q_xy_points = np.stack((qx.flatten(),qy.flatten()),axis=-1)
        # q_z_rbf = RBFInterpolator(q_xy_points,(qz-q_center).flatten(),smoothing=0,kernel='cubic')
        # ##Remember that in order to store the Zmx file we need to save the date on an equidistant grid
        # x_rbf_eval = np.linspace(np.min(qx),np.max(qx),res)
        # dx = np.abs(x_rbf_eval[1]-x_rbf_eval[0])
        # y_rbf_eval = np.linspace(np.min(qy),np.max(qy),res)
        # dy = np.abs(y_rbf_eval[1]-y_rbf_eval[0])
        # x_rbf_eval,y_rbf_eval = np.meshgrid(x_rbf_eval,y_rbf_eval)
        # q_xy_points_eval = np.stack((x_rbf_eval.flatten(),y_rbf_eval.flatten()),-1)
        # q_z_rbf_eval = q_z_rbf(q_xy_points_eval)
        # name_q_1_full = grid_dir + "q_surf_" + p_name + ".GRD"
        # zmx_grid_sag_write(dx,dy,np.transpose(q_z_rbf_eval).flatten(),res,name_q_1_full)

        ###We represent the first freeform surface in terms of a RectBivariate for the purpose of upsampling
        x_d = np.linspace(-1.0,1.0,Nx)
        y_d = np.linspace(-1.0,1.0,Ny)
        px_sp = RectBivariateSpline(y_d,x_d,px,s=0)
        py_sp = RectBivariateSpline(y_d,x_d,py,s=0)
        pz_sp = RectBivariateSpline(y_d,x_d,pz-p_center,s=0)

        Nx_sp = RectBivariateSpline(y_d,x_d,N_x,s=0)
        Ny_sp = RectBivariateSpline(y_d,x_d,N_y,s=0)
        Nz_sp = RectBivariateSpline(y_d,x_d,N_z,s=0)

        x_d_eval = np.linspace(-1.0,1.0,Nx*s)
        y_d_eval = np.linspace(-1.0,1.0,Ny*s)

        px_sp_eval = px_sp(y_d_eval,x_d_eval)
        py_sp_eval = py_sp(y_d_eval,x_d_eval)
        pz_sp_eval = pz_sp(y_d_eval,x_d_eval)

        Nx_sp_eval = Nx_sp(y_d_eval,x_d_eval)
        Ny_sp_eval = Ny_sp(y_d_eval,x_d_eval)
        Nz_sp_eval = Nz_sp(y_d_eval,x_d_eval)
        ##We represent the second freeform surface in terms of a RectBivariate for the purpose of upsampling
        rx_sp = RectBivariateSpline(y_d,x_d,rx,s=0)
        ry_sp = RectBivariateSpline(y_d,x_d,ry,s=0)
        rz_sp = RectBivariateSpline(y_d,x_d,rz-r_center,s=0)

        Nx_sp_r = RectBivariateSpline(y_d,x_d,N_x_r,s=0)
        Ny_sp_r = RectBivariateSpline(y_d,x_d,N_y_r,s=0)
        Nz_sp_r = RectBivariateSpline(y_d,x_d,N_z_r,s=0)

        rx_sp_eval = rx_sp(y_d_eval,x_d_eval)
        ry_sp_eval = ry_sp(y_d_eval,x_d_eval)
        rz_sp_eval = rz_sp(y_d_eval,x_d_eval)

        Nx_sp_r_eval = Nx_sp_r(y_d_eval,x_d_eval)
        Ny_sp_r_eval = Ny_sp_r(y_d_eval,x_d_eval)
        Nz_sp_r_eval = Nz_sp_r(y_d_eval,x_d_eval)
        ###We represent the third freeform surface in terms of a RectBivariate for the purpose of upsampling
        qx_sp = RectBivariateSpline(y_d, x_d, qx, s=0)
        qy_sp = RectBivariateSpline(y_d, x_d, qy, s=0)
        qz_sp = RectBivariateSpline(y_d, x_d, qz - q_center, s=0)

        Nx_sp_q = RectBivariateSpline(y_d, x_d, -N_x_q, s=0)
        Ny_sp_q = RectBivariateSpline(y_d, x_d, -N_y_q, s=0)
        Nz_sp_q = RectBivariateSpline(y_d, x_d, -N_z_q, s=0)

        qx_sp_eval = qx_sp(y_d_eval, x_d_eval)
        qy_sp_eval = qy_sp(y_d_eval, x_d_eval)
        qz_sp_eval = qz_sp(y_d_eval, x_d_eval)

        Nx_sp_q_eval = Nx_sp_q(y_d_eval, x_d_eval)
        Ny_sp_q_eval = Ny_sp_q(y_d_eval, x_d_eval)
        Nz_sp_q_eval = Nz_sp_q(y_d_eval, x_d_eval)

        ##We first generate the STL file for the first freeform surface
        print ("We now create the STL file for the first freeform")
        pcd = o3d.geometry.PointCloud()
        points = np.stack((px_sp_eval.flatten(),py_sp_eval.flatten(),pz_sp_eval.flatten()),-1)
        Normals= np.stack((Nx_sp_eval.flatten(),Ny_sp_eval.flatten(),Nz_sp_eval.flatten()),-1)
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(Normals)
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist =np.mean(distances)
        radius = 12*avg_dist
        bpa_mesh =o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius,radius]))
        o3d.visualization.draw_geometries([bpa_mesh])
        o3d.visualization.draw_geometries([pcd],point_show_normal=True)
        name_p_full = stl_dir + "p_surf_" + p_name + ".stl"
        o3d.io.write_triangle_mesh(name_p_full,bpa_mesh)

        ##For the second and the third freeforms we try to combine them into a single STL
        print ("We try to generate a single STL from the second freeform")
        pcd_2 = o3d.geometry.PointCloud()
        points2 = np.stack((rx_sp_eval.flatten(),ry_sp_eval.flatten(),rz_sp_eval.flatten()),-1)
        Normals2= np.stack((Nx_sp_r_eval.flatten(),Ny_sp_r_eval.flatten(),Nz_sp_r_eval.flatten()),-1)
        points23 = points2
        Normals23 = Normals2
        pcd_2.points = o3d.utility.Vector3dVector(points23)
        pcd_2.normals = o3d.utility.Vector3dVector(Normals23)
        distances2 = pcd_2.compute_nearest_neighbor_distance()
        avg_dist2 = np.mean(distances2)
        radius2 = 7*avg_dist2
        bpa_mesh2 = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_2,o3d.utility.DoubleVector([radius2,radius2]))
        o3d.visualization.draw_geometries([bpa_mesh2])
        o3d.visualization.draw_geometries([pcd_2],point_show_normal=True)
        name_rq_full = stl_dir + "r_surf_" + p_name + ".stl"
        o3d.io.write_triangle_mesh(name_rq_full,bpa_mesh2)

        print ("We try to generate a single STL from the third freeform")
        pcd_2 = o3d.geometry.PointCloud()
        points3 = np.stack((qx_sp_eval.flatten(),qy_sp_eval.flatten(),qz_sp_eval.flatten()),-1)
        Normals3= np.stack((Nx_sp_q_eval.flatten(),Ny_sp_q_eval.flatten(),Nz_sp_q_eval.flatten()),-1)
        points23 = points3
        Normals23 = Normals3
        pcd_2.points = o3d.utility.Vector3dVector(points23)
        pcd_2.normals = o3d.utility.Vector3dVector(Normals23)
        distances2 = pcd_2.compute_nearest_neighbor_distance()
        avg_dist2 = np.mean(distances2)
        radius2 = 7*avg_dist2
        bpa_mesh2 = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_2,o3d.utility.DoubleVector([radius2,radius2]))
        o3d.visualization.draw_geometries([bpa_mesh2])
        o3d.visualization.draw_geometries([pcd_2],point_show_normal=True)
        name_rq_full = stl_dir + "q_surf_" + p_name + ".stl"
        o3d.io.write_triangle_mesh(name_rq_full,bpa_mesh2)



def double_surf_save(p_points,q_points,N_points,N_points_q,p_name,res,Nx,Ny,s,flag):
    stl_dir = "C:\\Users\\itojimenez\\Documents\\Zemax\\Objects\\CAD Files\\"
    grid_dir = "C:\\Users\\itojimenez\\Documents\\Zemax\\Objects\\Grid Files\\"
    px = p_points[0, :].reshape(Nx, Ny)
    py = p_points[1, :].reshape(Nx, Ny)
    pz = p_points[2, :].reshape(Nx, Ny)
    qx = q_points[0, :].reshape(Nx, Ny)
    qy = q_points[1, :].reshape(Nx, Ny)
    qz = q_points[2, :].reshape(Nx, Ny)
    N_x = N_points[0, :].reshape(Nx, Ny)
    N_y = N_points[1, :].reshape(Nx, Ny)
    N_z = N_points[2, :].reshape(Nx, Ny)
    N_x_q = N_points_q[0, :].reshape(Nx, Ny)
    N_y_q = N_points_q[1, :].reshape(Nx, Ny)
    N_z_q = N_points_q[2, :].reshape(Nx, Ny)

    p_center = pz[int(pz.shape[0]*0.5),int(pz.shape[1]*0.5)]
    q_center = qz[int(qz.shape[0]*0.5),int(qz.shape[1]*0.5)]
    print ("Distance between the center of the first freeform surface and the origin")
    print (p_center)
    print ("Distance between the center of the second and the first freeform surfaces")
    print (q_center - p_center)
    print ("Distance between center of second freeform surface and the origin")
    print (q_center)



    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(px, py, pz)
    surf = ax.plot_surface(qx, qy, qz)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(px, py, pz)

    plt.figure()
    plt.plot(px[int(px.shape[0] * 0.5), :], pz[int(pz.shape[0] * 0.5), :], label='P surface -X')
    plt.plot(qx[int(qx.shape[0] * 0.5), :], qz[int(qz.shape[0] * 0.5), :], label='Q surface -X')
    plt.legend()

    plt.figure()
    plt.plot(px[int(px.shape[0] * 0.5), :], pz[int(pz.shape[0] * 0.5), :], label='P surface -X')
    plt.legend()
    plt.show()


    # ##For the first freeform surface we use the RBF interpolation
    # print ("We now create the GRD files for the first freeform")
    # p_xy_points = np.stack((px.flatten(),py.flatten()),axis=-1)
    # p_z_rbf = RBFInterpolator(p_xy_points,(pz-p_center).flatten(),smoothing=0,kernel='cubic')
    # ##Remember that in order to store the Zmx file we need to save the date on an equidistant grid
    # x_rbf_eval = np.linspace(np.min(px),np.max(px),res)
    # dx = np.abs(x_rbf_eval[1]-x_rbf_eval[0])
    # y_rbf_eval = np.linspace(np.min(py),np.max(py),res)
    # dy = np.abs(y_rbf_eval[1]-y_rbf_eval[0])
    # x_rbf_eval,y_rbf_eval = np.meshgrid(x_rbf_eval,y_rbf_eval)
    # q_xy_points_eval = np.stack((x_rbf_eval.flatten(),y_rbf_eval.flatten()),-1)
    # q_z_rbf_eval = p_z_rbf(q_xy_points_eval)
    # name_p_1_full = grid_dir + "p_surf_" + p_name + ".GRD"
    # zmx_grid_sag_write(dx,dy,np.transpose(q_z_rbf_eval).flatten(),res,name_p_1_full)
    #
    #
    # ##For the second freeform surface we use the RBF interpolation
    # print ("We now create the GRD files for the second freeform")
    # q_xy_points = np.stack((qx.flatten(),qy.flatten()),axis=-1)
    # q_z_rbf = RBFInterpolator(q_xy_points,(qz-q_center).flatten(),smoothing=0,kernel='cubic')
    # ##Remember that in order to store the Zmx file we need to save the date on an equidistant grid
    # x_rbf_eval = np.linspace(np.min(qx),np.max(qx),res)
    # dx = np.abs(x_rbf_eval[1]-x_rbf_eval[0])
    # y_rbf_eval = np.linspace(np.min(qy),np.max(qy),res)
    # dy = np.abs(y_rbf_eval[1]-y_rbf_eval[0])
    # x_rbf_eval,y_rbf_eval = np.meshgrid(x_rbf_eval,y_rbf_eval)
    # q_xy_points_eval = np.stack((x_rbf_eval.flatten(),y_rbf_eval.flatten()),-1)
    # q_z_rbf_eval = q_z_rbf(q_xy_points_eval)
    # name_q_1_full = grid_dir + "q_surf_" + p_name + ".GRD"
    # name_q_flat_full = grid_dir + "flat_q_surf_" + p_name + ".GRD"
    # zmx_grid_sag_write(dx,dy,np.transpose(q_z_rbf_eval).flatten(),res,name_q_1_full)
    # zmx_grid_sag_write(dx,dy,np.transpose(np.zeros(x_rbf_eval.shape)).flatten(),res,name_q_flat_full)

    if flag:

        ###We represent the first freeform surface in terms of a RectBivariate for the purpose of upsampling
        x_d = np.linspace(-1.0,1.0,Nx)
        y_d = np.linspace(-1.0,1.0,Ny)
        px_sp = RectBivariateSpline(y_d,x_d,px,s=0)
        py_sp = RectBivariateSpline(y_d,x_d,py,s=0)
        pz_sp = RectBivariateSpline(y_d,x_d,pz-p_center,s=0)

        Nx_sp = RectBivariateSpline(y_d,x_d,N_x,s=0)
        Ny_sp = RectBivariateSpline(y_d,x_d,N_y,s=0)
        Nz_sp = RectBivariateSpline(y_d,x_d,N_z,s=0)

        x_d_eval = np.linspace(-1.0,1.0,Nx*s)
        y_d_eval = np.linspace(-1.0,1.0,Ny*s)

        px_sp_eval = px_sp(y_d_eval,x_d_eval)
        py_sp_eval = py_sp(y_d_eval,x_d_eval)
        pz_sp_eval = pz_sp(y_d_eval,x_d_eval)

        Nx_sp_eval = Nx_sp(y_d_eval,x_d_eval)
        Ny_sp_eval = Ny_sp(y_d_eval,x_d_eval)
        Nz_sp_eval = Nz_sp(y_d_eval,x_d_eval)

        ###We represent the second freeform surface in terms of a RectBivariate for the purpose of upsampling
        # x_d = np.linspace(-1.0, 1.0, Nx)
        # y_d = np.linspace(-1.0, 1.0, Ny)
        qx_sp = RectBivariateSpline(y_d, x_d, qx, s=0)
        qy_sp = RectBivariateSpline(y_d, x_d, qy, s=0)
        qz_sp = RectBivariateSpline(y_d, x_d, qz - q_center, s=0)

        Nx_sp_q = RectBivariateSpline(y_d, x_d, N_x_q, s=0)
        Ny_sp_q = RectBivariateSpline(y_d, x_d, N_y_q, s=0)
        Nz_sp_q = RectBivariateSpline(y_d, x_d, N_z_q, s=0)

        # x_d_eval = np.linspace(-1.0, 1.0, Nx * s)
        # y_d_eval = np.linspace(-1.0, 1.0, Ny * s)

        qx_sp_eval = qx_sp(y_d_eval, x_d_eval)
        qy_sp_eval = qy_sp(y_d_eval, x_d_eval)
        qz_sp_eval = qz_sp(y_d_eval, x_d_eval)

        Nx_sp_q_eval = Nx_sp_q(y_d_eval, x_d_eval)
        Ny_sp_q_eval = Ny_sp_q(y_d_eval, x_d_eval)
        Nz_sp_q_eval = Nz_sp_q(y_d_eval, x_d_eval)


        ##We first generate the STL file for the first freeform surface
        print ("We now create the STL file for the first freeform")
        pcd = o3d.geometry.PointCloud()
        points = np.stack((px_sp_eval.flatten(),py_sp_eval.flatten(),pz_sp_eval.flatten()),-1)
        Normals= np.stack((Nx_sp_eval.flatten(),Ny_sp_eval.flatten(),Nz_sp_eval.flatten()),-1)
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(Normals)
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist =np.mean(distances)
        radius = 12*avg_dist
        bpa_mesh =o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius,radius]))
        o3d.visualization.draw_geometries([bpa_mesh])
        o3d.visualization.draw_geometries([pcd],point_show_normal=True)
        name_p_full = stl_dir + "p_surf_" + p_name + ".stl"
        o3d.io.write_triangle_mesh(name_p_full,bpa_mesh)

        ##We first generate the STL file for the first freeform surface
        print ("We now create the STL file for the second freeform")
        pcd_q = o3d.geometry.PointCloud()
        points_q = np.stack((qx_sp_eval.flatten(),qy_sp_eval.flatten(),qz_sp_eval.flatten()),-1)
        Normals_q= np.stack((Nx_sp_q_eval.flatten(),Ny_sp_q_eval.flatten(),Nz_sp_q_eval.flatten()),-1)
        pcd_q.points = o3d.utility.Vector3dVector(points_q)
        pcd_q.normals = o3d.utility.Vector3dVector(Normals_q)
        distances_q = pcd_q.compute_nearest_neighbor_distance()
        avg_dist_q =np.mean(distances_q)
        radius_q = 12*avg_dist_q
        bpa_mesh_q =o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_q,o3d.utility.DoubleVector([radius_q,radius_q]))
        o3d.visualization.draw_geometries([bpa_mesh_q])
        o3d.visualization.draw_geometries([pcd_q],point_show_normal=True)
        name_q_full = stl_dir + "r_surf_" + p_name + ".stl"
        o3d.io.write_triangle_mesh(name_q_full,bpa_mesh_q)

def xy_plane_inter_z(d, p, dir, z_tar):
    p_tar = p + d * dir
    return (z_tar - p_tar[2]) ** 2
def intermediate_xy_coords(p_points,r_points,pr_out_dir):
    ##We calculate the outgoing ray directions from the initial normals
    ##We use this method to calculate an intermediate plane coordinate which will be used to optimize the first freeform surface
    px = p_points[0,:,:]
    py = p_points[1,:,:]
    pz = p_points[2,:,:]

    rx = r_points[0,:,:]
    ry = r_points[1,:,:]
    rz = r_points[2,:,:]

    p_center = pz[int(pz.shape[0]*0.5),int(pz.shape[1]*0.5)]
    r_center = rz[int(rz.shape[0]*0.5),int(rz.shape[1]*0.5)]
    print("Distance between the center of the second and the first freeform surfaces")
    print(r_center - p_center)
    print ("We estimate the intermediate plane location as the middle point between these two surfaces")
    z_inter = p_center + (r_center-p_center)*0.5
    print (z_inter)
    z_inter_ini = np.ones(pz.shape)*z_inter
    d = newton(xy_plane_inter_z,x0=z_inter_ini,args=(p_points,pr_out_dir,z_inter_ini))
    P_inter = p_points + d * pr_out_dir
    P_inter = np.stack((P_inter[0,:,:].flatten(),P_inter[1,:,:].flatten(),P_inter[2,:,:].flatten()))
    return P_inter,z_inter

def surf_N_eval(p,Nx,Ny):
    ##We try to estimate the surface normals.
##For this we will first parametrize the surface points over a rectangular grid similarly to what we do in the symplectic
##algorithm
##We can then compute the normals by taking the partial derivatives with respect to these coordinates
##Similarly to what we do in the symplectic algorithm implementation we can use RectBivariate splines to calculate the derivatives
    xi_1d = np.linspace(0.0,1.0,Ny)
    eta_1d = np.linspace(0.0,1.0,Nx)
    xi, eta = np.meshgrid(xi_1d,eta_1d)
    pz_grad = RectBivariateSpline(eta_1d,xi_1d,p[2,:,:],s=0,kx=3,ky=3)
    pz_grad_eval = pz_grad.ev(eta,xi)
    pz_grad_eta = pz_grad.partial_derivative(dx=1,dy=0)
    pz_grad_xi= pz_grad.partial_derivative(dx=0,dy=1)
    px_grad = RectBivariateSpline(eta_1d,xi_1d,p[0,:,:],s=0,kx=3,ky=3)
    px_grad_eval = px_grad.ev(eta,xi)
    px_grad_eta = px_grad.partial_derivative(dx=1,dy=0)
    px_grad_xi= px_grad.partial_derivative(dx=0,dy=1)
    py_grad = RectBivariateSpline(eta_1d,xi_1d,p[1,:,:],s=0,kx=3,ky=3)
    py_grad_eval = py_grad.ev(eta,xi)
    py_grad_eta = py_grad.partial_derivative(dx=1,dy=0)
    py_grad_xi= py_grad.partial_derivative(dx=0,dy=1)
    dpz_deta = pz_grad_eta(eta_1d,xi_1d)
    dpz_dxi = pz_grad_xi(eta_1d,xi_1d)
    dpx_deta = px_grad_eta(eta_1d,xi_1d)
    dpx_dxi = px_grad_xi(eta_1d,xi_1d)
    dpy_deta = py_grad_eta(eta_1d,xi_1d)
    dpy_dxi = py_grad_xi(eta_1d,xi_1d)
    dp_deta = np.stack((dpx_deta.flatten(),dpy_deta.flatten(),dpz_deta.flatten()))
    dp_dxi= np.stack((dpx_dxi.flatten(),dpy_dxi.flatten(),dpz_dxi.flatten()))
    cross = np.cross(dp_deta,dp_dxi,axis=0)

    cross2 =cross/np.linalg.norm(cross,axis=0)
    cross2 = cross2.reshape(3,Nx,Ny)

    # cross2[0,0,0] = (cross2[0,1,0] + cross2[0,0,1] + cross2[0,1,1])/3
    # cross2[1,0,0] = (cross2[1,1,0] + cross2[1,0,1] + cross2[1,1,1])/3
    # cross2[2,0,0] = (cross2[2,1,0] + cross2[2,0,1] + cross2[2,1,1])/3
    #
    # cross2[0,-1,0] = (cross2[0,-2,0] + cross2[0,-1,1] + cross2[0,-2,1])/3
    # cross2[1,-1,0] = (cross2[1,-2,0] + cross2[1,-1,1] + cross2[1,-2,1])/3
    # cross2[2,-1,0] = (cross2[2,-2,0] + cross2[2,-1,1] + cross2[2,-2,1])/3
    #
    # cross2[0,-1,-1] = (cross2[0,-2,-1] + cross2[0,-1,-2] + cross2[0,-2,-2])/3
    # cross2[1,-1,-1] = (cross2[1,-2,-1] + cross2[1,-1,-2] + cross2[1,-2,-2])/3
    # cross2[2,-1,-1] = (cross2[2,-2,-1] + cross2[2,-1,-2] + cross2[2,-2,-2])/3
    #
    # cross2[0,0,-1] = (cross2[0,0,-2] + cross2[0,1,-1] + cross2[0,1,-2])/3
    # cross2[1,0,-1] = (cross2[1,0,-2] + cross2[1,1,-1] + cross2[1,1,-2])/3
    # cross2[2,0,-1] = (cross2[2,0,-2] + cross2[2,1,-1] + cross2[2,1,-2])/3
    return cross2

def surf_N_eval_direct(p,Nx,Ny):
    dpz_deta= np.gradient(p[2,:,:],axis=0,edge_order=2)
    dpz_dxi= np.gradient(p[2,:,:],axis=1,edge_order=2)

    dpx_deta = np.gradient(p[0,:,:],axis=0,edge_order=2)
    dpx_dxi= np.gradient(p[0,:,:],axis=1,edge_order=2)

    dpy_deta= np.gradient(p[1,:,:],axis=0,edge_order=2)
    dpy_dxi= np.gradient(p[1,:,:],axis=1,edge_order=2)

    dp_deta = np.stack((dpx_deta.flatten(),dpy_deta.flatten(),dpz_deta.flatten()))
    dp_dxi= np.stack((dpx_dxi.flatten(),dpy_dxi.flatten(),dpz_dxi.flatten()))
    cross = -np.cross(dp_deta,dp_dxi,axis=0)
    cross /= np.linalg.norm(cross,axis=0)
    cross2 = cross.reshape(3,Nx,Ny)
    return cross2

def curl_cost_eval(p,N_orig,Nx,Ny):
    xi_1d = np.linspace(-1.0,1.0,Ny)
    eta_1d = np.linspace(-1.0,1.0,Nx)
    xi, eta = np.meshgrid(xi_1d,eta_1d)
    pz_grad = RectBivariateSpline(eta_1d,xi_1d,p[2,:,:],s=0,kx=3,ky=3)
    pz_grad_eta = pz_grad.partial_derivative(dx=1,dy=0)
    pz_grad_xi= pz_grad.partial_derivative(dx=0,dy=1)
    px_grad = RectBivariateSpline(eta_1d,xi_1d,p[0,:,:],s=0,kx=3,ky=3)
    px_grad_eta = px_grad.partial_derivative(dx=1,dy=0)
    px_grad_xi= px_grad.partial_derivative(dx=0,dy=1)
    py_grad = RectBivariateSpline(eta_1d,xi_1d,p[1,:,:],s=0,kx=3,ky=3)
    py_grad_eta = py_grad.partial_derivative(dx=1,dy=0)
    py_grad_xi= py_grad.partial_derivative(dx=0,dy=1)

    dpz_deta = pz_grad_eta(eta_1d,xi_1d)
    dpz_dxi = pz_grad_xi(eta_1d,xi_1d)
    dpx_deta = px_grad_eta(eta_1d,xi_1d)
    dpx_dxi = px_grad_xi(eta_1d,xi_1d)
    dpy_deta = py_grad_eta(eta_1d,xi_1d)
    dpy_dxi = py_grad_xi(eta_1d,xi_1d)
    dp_deta = np.stack((dpx_deta.flatten(),dpy_deta.flatten(),dpz_deta.flatten()))
    dp_dxi = np.stack((dpx_dxi.flatten(),dpy_dxi.flatten(),dpz_dxi.flatten()))
    cross = np.cross(dp_deta,dp_dxi,axis=0)
    # cross = np.cross(dp_dxi,dp_deta,axis=0)
    cross /= np.linalg.norm(cross,axis=0)
    cross2 = cross.reshape(3,Nx,Ny)

    cross2[0,0,0] = (cross2[0,1,0] + cross2[0,0,1] + cross2[0,1,1])/3
    cross2[1,0,0] = (cross2[1,1,0] + cross2[1,0,1] + cross2[1,1,1])/3
    cross2[2,0,0] = (cross2[2,1,0] + cross2[2,0,1] + cross2[2,1,1])/3

    cross2[0,-1,0] = (cross2[0,-2,0] + cross2[0,-1,1] + cross2[0,-2,1])/3
    cross2[1,-1,0] = (cross2[1,-2,0] + cross2[1,-1,1] + cross2[1,-2,1])/3
    cross2[2,-1,0] = (cross2[2,-2,0] + cross2[2,-1,1] + cross2[2,-2,1])/3
    #
    cross2[0,-1,-1] = (cross2[0,-2,-1] + cross2[0,-1,-2] + cross2[0,-2,-2])/3
    cross2[1,-1,-1] = (cross2[1,-2,-1] + cross2[1,-1,-2] + cross2[1,-2,-2])/3
    cross2[2,-1,-1] = (cross2[2,-2,-1] + cross2[2,-1,-2] + cross2[2,-2,-2])/3
    #
    cross2[0,0,-1] = (cross2[0,0,-2] + cross2[0,1,-1] + cross2[0,1,-2])/3
    cross2[1,0,-1] = (cross2[1,0,-2] + cross2[1,1,-1] + cross2[1,1,-2])/3
    cross2[2,0,-1] = (cross2[2,0,-2] + cross2[2,1,-1] + cross2[2,1,-2])/3
    ##Similarly, we need to take the derivatives of the "wanted" normal vector field.
    ##We then use the gradients of the surface and this normal field to evaluate the cost
    ##Since we have access to the "real" normals, we could also try computing the dot product
    ##between the "real" and the "wanted" normals to estimate any deviation between them.
    Nz_grad = RectBivariateSpline(eta_1d,xi_1d,N_orig[2,:,:],s=0,kx=3,ky=3)
    Nz_grad_eta = Nz_grad.partial_derivative(dx=1, dy=0)
    Nz_grad_xi = Nz_grad.partial_derivative(dx=0, dy=1)
    Nx_grad = RectBivariateSpline(eta_1d,xi_1d,N_orig[0,:,:],s=0,kx=3,ky=3)
    Nx_grad_eta = Nx_grad.partial_derivative(dx=1, dy=0)
    Nx_grad_xi = Nx_grad.partial_derivative(dx=0, dy=1)
    Ny_grad = RectBivariateSpline(eta_1d,xi_1d,N_orig[1,:,:],s=0,kx=3,ky=3)
    Ny_grad_eta = Ny_grad.partial_derivative(dx=1, dy=0)
    Ny_grad_xi = Ny_grad.partial_derivative(dx=0, dy=1)

    Npz_deta = Nz_grad_eta(eta_1d,xi_1d)
    Npz_dxi = Nz_grad_xi(eta_1d,xi_1d)
    Npx_deta = Nx_grad_eta(eta_1d,xi_1d)
    Npx_dxi = Nx_grad_xi(eta_1d,xi_1d)
    Npy_deta = Ny_grad_eta(eta_1d,xi_1d)
    Npy_dxi = Ny_grad_xi(eta_1d,xi_1d)
    dN_deta = np.stack((Npx_deta.flatten(), Npy_deta.flatten(), Npz_deta.flatten()))
    dN_dxi = np.stack((Npx_dxi.flatten(), Npy_dxi.flatten(), Npz_dxi.flatten()))

    C = np.sum(dN_deta * dp_dxi, axis=0) - np.sum(dN_dxi * dp_deta, axis=0)

    dot_Ns = np.sum(N_orig.reshape(3,Nx,Ny)*cross2,axis=0)
    return C.reshape(Nx,Ny),dot_Ns


###Using the surface normal information we can compute additional things such as the Gaussian curvature
def gaussian_curv(p,Nx,Ny):
    xi_1d = np.linspace(-1.0, 1.0, Ny)
    eta_1d = np.linspace(-1.0, 1.0, Nx)
    xi, eta = np.meshgrid(xi_1d, eta_1d)
    pz_grad = RectBivariateSpline(eta_1d,xi_1d,p[2,:,:],s=0)
    pz_grad_eta = pz_grad.partial_derivative(dx=1,dy=0)
    pz_grad_xi= pz_grad.partial_derivative(dx=0,dy=1)
    px_grad = RectBivariateSpline(eta_1d,xi_1d,p[0,:,:],s=0)
    px_grad_eta = px_grad.partial_derivative(dx=1,dy=0)
    px_grad_xi= px_grad.partial_derivative(dx=0,dy=1)
    py_grad = RectBivariateSpline(eta_1d,xi_1d,p[1,:,:],s=0)
    py_grad_eta = py_grad.partial_derivative(dx=1,dy=0)
    py_grad_xi= py_grad.partial_derivative(dx=0,dy=1)
    dpz_deta = pz_grad_eta(eta_1d,xi_1d)
    dpz_dxi = pz_grad_xi(eta_1d,xi_1d)
    dpx_deta = px_grad_eta(eta_1d,xi_1d)
    dpx_dxi = px_grad_xi(eta_1d,xi_1d)
    dpy_deta = py_grad_eta(eta_1d,xi_1d)
    dpy_dxi = py_grad_xi(eta_1d,xi_1d)
    dp_deta = np.stack((dpx_deta.flatten(),dpy_deta.flatten(),dpz_deta.flatten()))
    dp_dxi= np.stack((dpx_dxi.flatten(),dpy_dxi.flatten(),dpz_dxi.flatten()))
    cross = np.cross(dp_deta,dp_dxi,axis=0)
    cross /= np.linalg.norm(cross,axis=0)
    cross2 = cross.reshape(3,Nx,Ny)
    ###We use the expressions given in the thesis:"Intelligent Freeform deformation for LED illumination Optics"
    ##We compute the quantities associated to the first fundamental form
    ##u -> eta
    ##v -> xi
    E = np.sum(dp_deta*dp_deta,axis=0)
    G = np.sum(dp_dxi*dp_dxi,axis=0)
    F = np.sum(dp_deta*dp_dxi,axis=0)
    ##We compute the quantities associated to the second fundamental form
    ##For these we need to compute the second order deritivatives with respect to u (eta) and v (xi)

    pz_grad2_eta = pz_grad.partial_derivative(dx=2,dy=0)
    pz_grad2_xi= pz_grad.partial_derivative(dx=0,dy=2)
    pz_grad_eta_xi= pz_grad.partial_derivative(dx=1,dy=1)
    px_grad2_eta = px_grad.partial_derivative(dx=2,dy=0)
    px_grad2_xi= px_grad.partial_derivative(dx=0,dy=2)
    px_grad_eta_xi= px_grad.partial_derivative(dx=1,dy=1)
    py_grad2_eta = py_grad.partial_derivative(dx=2,dy=0)
    py_grad2_xi= py_grad.partial_derivative(dx=0,dy=2)
    py_grad_eta_xi= py_grad.partial_derivative(dx=1,dy=1)
    d2pz_d2eta = pz_grad2_eta(eta_1d,xi_1d)
    d2pz_d2xi = pz_grad2_xi(eta_1d,xi_1d)
    dpz_dxideta = pz_grad_eta_xi(eta_1d,xi_1d)
    d2px_d2eta = px_grad2_eta(eta_1d,xi_1d)
    d2px_d2xi = px_grad2_xi(eta_1d,xi_1d)
    dpx_dxideta = px_grad_eta_xi(eta_1d,xi_1d)
    d2py_d2eta = py_grad2_eta(eta_1d,xi_1d)
    d2py_d2xi = py_grad2_xi(eta_1d,xi_1d)
    dpy_dxideta = py_grad_eta_xi(eta_1d,xi_1d)
    d2p_d2eta = np.stack((d2px_d2eta.flatten(),d2py_d2eta.flatten(),d2pz_d2eta.flatten()))
    d2p_d2xi= np.stack((d2px_d2xi.flatten(),d2py_d2xi.flatten(),d2pz_d2xi.flatten()))
    dp_dxideta = np.stack((dpx_dxideta.flatten(),dpy_dxideta.flatten(),dpz_dxideta.flatten()))
    L = np.sum(d2p_d2eta*cross,axis=0)
    N = np.sum(d2p_d2xi*cross,axis=0)
    M = np.sum(dp_dxideta*cross,axis=0)
    kappa = (L*N - (M**2))/(E*G-(F**2))
    return kappa.reshape(Nx,Ny)

def output_dir_eval(cross2,i_dir,n1,n2):
    ###Using the extracted normal vectors, we can use Snell's law in vector form to extract the output ray directions
    r = n1 / n2
    c = np.sum(-cross2* i_dir, axis=0)
    # plt.figure()
    # plt.imshow(c)
    # plt.show()
    pr_out_dir = r * i_dir + (r * c - np.sqrt(1 - (r ** 2) * (1 - c ** 2))) * cross2
    partial = 1 - (r**2)*(1-c**2)
    return pr_out_dir,partial

def output_xy_pos_eval(s,out_dir,z_out):
    z_out2d = np.ones(s.shape)*z_out
    d = (z_out2d - s[2])/(out_dir[2])
    s_out = s + out_dir*d
    return s_out

##General expression for the shoelace formula extracted from wikipedia.
##This function is used as the unit block needed to compute the area of a single general polygon
##made of 4 points. ##We approximate the area of each "cell" by the area formed by 4 different neighboring points
def shoelace_form(p1,p2):
    return p1[0]*p2[1] - p1[1]*p2[0]

def local_area_eval(p1,p2,p3,p4):
    return shoelace_form(p1,p2) + shoelace_form(p2,p3) + shoelace_form(p3,p4) + shoelace_form(p4,p1)


def global_area_eval(p1,p2):
    p = np.stack((p1,p2))
    A = np.zeros((p1.shape[0]-1,p1.shape[1]-1))
    for i in range(0,p1.shape[0]-1,1):
        for j in range(0,p1.shape[1]-1,1):
            A[i,j] = np.round(local_area_eval(p[:,i,j],p[:,i,j+1],p[:,i+1,j+1],p[:,i+1,j]),5)
    return A


def flux_eval(f,p1,p2):
    p = np.stack((p1,p2))
    flux = np.zeros((p1.shape[0]-1,p1.shape[1]-1))
    for i in range(0,p1.shape[0]-1,1):
        for j in range(0,p1.shape[1]-1,1):
            flux[i,j] = (local_area_eval(p[:,i,j],p[:,i,j+1],p[:,i+1,j+1],p[:,i+1,j]))*(f[i,j]+f[i,j+1]+f[i+1,j+1]+f[i+1,j])*(1/4)

    return flux
def OPL_ref_eval(p_points,r_points,q_points,x_tar,y_tar,z_out,n1,n2):
    opl_input = np.linalg.norm(p_points,axis=0)*n1
    opl_inter = np.linalg.norm(r_points-p_points,axis=0)*n2
    opl_inter2 = np.linalg.norm(q_points-r_points,axis=0)*n1
    output_vec = np.stack((x_tar,y_tar,np.ones(x_tar.shape)*z_out),axis=0)
    opl_output = np.linalg.norm(output_vec - q_points,axis=0)*n2 ##OPL from the q11 point to the output
    opl_val = opl_input + opl_inter + opl_inter2+opl_output
    opl_val2 = opl_input + opl_inter #+ opl_inter2#+opl_output
    return opl_val,opl_val2,opl_inter2

def stereo_proj_dir_to_plane(dir):
    plane_vec = np.stack((dir[0],dir[1]))/(1+dir[2])
    return plane_vec

def confocal_data_read(filename):
    x = []
    y= []
    z = []
    with open(filename,"r",encoding='utf-8') as f:
        for line in f:
            line_p = line.split()
            try:
                z.append(float(line_p[2]))
                x.append(float(line_p[0]))
                y.append(float(line_p[1]))
            except:
                pass
    return np.asarray(x),np.asarray(y),np.asarray(z)


class TransformationEstimationTranslationOnly(o3d.pipelines.registration.TransformationEstimation):
    def __init__(self):
            super(TransformationEstimationTranslationOnly,self).__init__()

    def compute_rms(self,source,target,corres):
        return np.sqrt(np.sum(np.asarray(corres)[:,2]))

    def compute_transformation(self,source,target,corres):
        if not corres.shape[0]==0:
            source_centroid = np.mean(np.asarray(source.points)[corres[:,0],:],axis=0)
            target_centroid = np.mean(np.asarray(target.points)[corres[:,0],:],axis=0)

            ##Compute the translation
            translation = target_centroid - source_centroid
            transformation = np.identity(4)
            transformation[:3,3] = translation

            return transformation
        else:
            return np.identity(4)


def icp_pcd(source,target):
    source.paint_uniform_color([1,0,0])
    target.paint_uniform_color([0,0,1])

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60,origin=[0,0,0])
    o3d.visualization.draw_geometries([source,target,mesh_frame])

    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_down = source.voxel_down_sample(voxel_size=1)
    print ("We start with the coarse ICP calculation")
    threshold = 50.0
    trans_init = np.asarray([[1, 0.0, 0.0, 0.0],
                             [0.0, 1, 0.0, 0.0],
                             [0.0, 0.0, 1, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])

    reg_p2p = o3d.pipelines.registration.registration_icp(source_down, target, threshold, trans_init,
                                                          o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                          o3d.pipelines.registration.ICPConvergenceCriteria(relative_rmse=1e-6,
                                                              max_iteration=10 ^ 9))  # max(max_iterationbs=10^9)


    print ("We apply the transformation to the source")
    source.transform(reg_p2p.transformation)
    trans_init = reg_p2p.transformation

    print ("We now plot the point clouds with the transformation applied to the source")

    o3d.visualization.draw_geometries([source,target,mesh_frame])

    filename1_coarser = "source_pc_coarser.xyz"
    filename2_coarser = "target_pc_coarser.xyz"

    o3d.io.write_point_cloud(filename1_coarser,source)
    o3d.io.write_point_cloud(filename2_coarser,target)

    print ("We now proceed to the finer fitting or ICP algorithm")


    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=int(1e9))


    reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,o3d.pipelines.registration.TransformationEstimationPointToPoint(),criteria=criteria)


    source.transform(reg_p2p.transformation)

    print ("We now plot the point clouds with the transformation applied to the source (finer)")

    o3d.visualization.draw_geometries([source, target, mesh_frame])

    filename1 = "source_pc.xyz"
    filename2 = "target_pc.xyz"

    o3d.io.write_point_cloud(filename1,source)
    o3d.io.write_point_cloud(filename2,target)


























