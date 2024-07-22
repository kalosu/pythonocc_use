import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from common_functions import surf_param_read

def points_surf_append(p,name):
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

    ##We now create arrays in 2D which are going to be used to store the coordinates of the extention points
    N_extra_p = 10
    x_ext_ref = np.zeros((s1_x.shape[0]+N_extra_p,s1_x.shape[0]+N_extra_p))

    pad_width = ((x_ext_ref.shape[0]- s1_x.shape[0]) // 2, (x_ext_ref.shape[0]- s1_x.shape[0]) // 2)
    z_ext= np.pad(s1_z, (pad_width, pad_width), mode='constant')
    y_ext= np.pad(s1_y, (pad_width, pad_width), mode='constant')
    x_ext= np.pad(s1_x, (pad_width, pad_width), mode='constant')
    ##Since we want that the height array is somehow continuous we will start by adding points from the boundary
    ##We then move through the array inwards until we reach the original size of our array of points
    N_layers = N_extra_p
    max_s = 0.98
    min_s = 0.7
    scaling_array = np.linspace(min_s,max_s,N_layers)
    z_base_array = np.linspace(0,z_min*0.5,N_layers)

    for i in range(0,N_extra_p-1,1):
        N_eval = (x_ext.shape[0]-2*i) * 1 + ((x_ext.shape[0]-2*i) - 1) * 2 + ((x_ext.shape[0]-2*i)-2)*1  ##Oneside with full size and the other 3 with one element less than the size per side.
        a = (x_ext.shape[0]-2*i) *1
        b = a + ((x_ext.shape[0]-2*i)-1)*1
        c = b + ((x_ext.shape[0]-2*i)-1)*1
        b_eval = np.linspace(0,1,N_eval)
        x_layer = x_b_spline(b_eval)*scaling_array[i]
        y_layer = y_b_spline(b_eval)*scaling_array[i]


        z_layer = np.ones(N_eval)*z_base_array[i]
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
    # plt.imshow(z_ext.reshape(s1_x.shape[0]+10,s1_x.shape[0]+10))
    # plt.title("Z coordinates after modification")
    #
    # plt.figure()
    # plt.imshow(x_ext.reshape(s1_x.shape[0]+10,s1_x.shape[0]+10))
    # plt.title("X coordinates after modification")
    #
    # plt.figure()
    # plt.imshow(y_ext.reshape(s1_x.shape[0]+10,s1_x.shape[0]+10))
    # plt.title("Y coordinates after modification")
    # plt.show()
    ##We now use the new arrays to generate the step file
    points_extended = np.stack((x_ext,y_ext,z_ext),axis=-1)

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_ext,y_ext,z_ext,color='red',alpha=0.3)
    # ax.contour3D(s1[0].reshape(Nx,Ny),s1[1].reshape(Nx,Ny),s1[2].reshape(Nx,Ny),50,cmap='viridis')
    plt.show()
    # points_to_surf(points_extended.reshape(3,s1_x.shape[0]+N_extra_p,s1_x.shape[1]+N_extra_p),"name")
    return points_extended


filename = "ff1_p65_to_gaus_res201_n_IP_S"

Nx = 201
Ny = 201
# nlens = 1.5
# n2 = 1
# prism()

s1, N1, out_dir1 = surf_param_read(filename)

points = points_surf_append(s1.reshape(3, Nx, Ny), "p")
