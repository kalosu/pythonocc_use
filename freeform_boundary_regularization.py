###We have the problem that the boundary contour from the generated freeforms has some oscillations
##exactly around the corner points.
##This is somehow probably related to the symplectic algorithm that we are using since at those points
##we move the dummy coordinates in a rather not appropriate manner.
##Question: Can we use ridge regression to "smooth" out the boundary points??
##Ideally I want to have a trajectory in 3D which follows the originally trajectory as close as possible
##to the original trajectory but only modifies those problematic regions so that the trajectory is smooth and continous.
##Can we use ridge regression to smooth out those oscillatory points around the corners??

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from common_functions import confocal_data_read,surf_param_sN_save,surf_params_sN_read
import open3d as o3d
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline,splrep,splev
from csaps import csaps


# filepath = "D:\\white_light_interfer\\rq_surface_extended_offset_hatching_30_deg\\"

filepath = "C:\\Users\\itojimenez\\PycharmProjects\\beam_shaping_3D_freeform\\surf_files_paper\\"

design_pc= "q_surface_ref_z_right_position_cc.txt"
# filename3 = "ff3_dip_meep_z50nm_to_gaus_res251_n_IP_Visio_v_real_further"  ##Original
filename3 = "q_surface_smooth_c_401"  ##Original

s3, N3 = surf_params_sN_read(filename3)

# cf_x_design, cf_y_design, cf_z_design= confocal_data_read(filepath + design_pc)
cf_x_design = s3[0,:]
print ("shape of cf_x_design")
print (np.sqrt(cf_x_design.shape))
cf_y_design = s3[1,:]
cf_z_design = s3[2,:]

pcd_design= o3d.geometry.PointCloud()
points_design= np.stack((cf_x_design.flatten(), cf_y_design.flatten(), cf_z_design.flatten()), -1)
pcd_design.points = o3d.utility.Vector3dVector(points_design)


design_points = np.asarray(pcd_design.points)

d_x = (design_points[:,0].reshape(int(np.sqrt(design_points.shape[0])),-1))
d_x_c = (design_points[:,0].reshape(int(np.sqrt(design_points.shape[0])),-1)).copy()
d_y = (design_points[:,1].reshape(int(np.sqrt(design_points.shape[0])),-1))
d_y_c = (design_points[:,1].reshape(int(np.sqrt(design_points.shape[0])),-1)).copy()
d_z = (design_points[:,2].reshape(int(np.sqrt(design_points.shape[0])),-1))
d_z_c = (design_points[:,2].reshape(int(np.sqrt(design_points.shape[0])),-1)).copy()

middle_loc = int(d_x.shape[1] * 0.5)


d_x_b = np.hstack((d_x[0,middle_loc:],d_x[1:,-1],d_x[-1,:][::-1][1:],d_x[:,0][::-1][1:],d_x[0,1:middle_loc]))#*(0.95)
d_y_b = np.hstack((d_y[0,middle_loc:],d_y[1:,-1],d_y[-1,:][::-1][1:],d_y[:,0][::-1][1:],d_y[0,1:middle_loc]))#*(0.95)
d_z_b = np.hstack((d_z[0,middle_loc:],d_z[1:,-1],d_z[-1,:][::-1][1:],d_z[:,0][::-1][1:],d_z[0,1:middle_loc]))#*(0.95)

r_xy = np.sqrt(d_x_b**2+d_y_b**2)

###We want to extract only those portions of the trajectories that are problematic plus some small regions that are smooth
n_points = 10
###First corner
x_b_1 = np.hstack((d_x[0:n_points,0][::-1][:],d_x[0,1:n_points]))
y_b_1 = np.hstack((d_y[0:n_points,0][::-1][:],d_y[0,1:n_points]))
z_b_1 = np.hstack((d_z[0:n_points,0][::-1][:],d_z[0,1:n_points]))
###Second corner
x_b_2 = np.hstack((d_x[0,-n_points:],d_x[1:n_points,-1]))
y_b_2 = np.hstack((d_y[0,-n_points:],d_y[1:n_points,-1]))
z_b_2 = np.hstack((d_z[0,-n_points:],d_z[1:n_points,-1]))
##Third corner
x_b_3 = np.hstack((d_x[-n_points:,-1],d_x[-1,-n_points:][::-1][1:]))
y_b_3 = np.hstack((d_y[-n_points:,-1],d_y[-1,-n_points:][::-1][1:]))
z_b_3 = np.hstack((d_z[-n_points:,-1],d_z[-1,-n_points:][::-1][1:]))
##Fourth corner
x_b_4 = np.hstack((d_x[-1,0:n_points][::-1][:],d_x[-n_points:,0][::-1][1:]))
y_b_4 = np.hstack((d_y[-1,0:n_points][::-1][:],d_y[-n_points:,0][::-1][1:]))
z_b_4 = np.hstack((d_z[-1,0:n_points][::-1][:],d_z[-n_points:,0][::-1][1:]))

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# surf = ax.scatter(d_x, d_y, d_z)
# plt.title("Original surface points design")

fig = plt.figure()
ax = plt.axes(projection='3d')
surf = ax.scatter(d_x_b, d_y_b, d_z_b)
ax.scatter(d_x[0,0],d_y[0,0],d_z[0,0],color='red')
plt.title("Points on boundary - Design surface")

plt.figure()
plt.plot(d_x_b,d_y_b,color='blue')
plt.scatter(d_x_b,d_y_b,color='red')
plt.scatter(d_x[0,0],d_y[0,0],color='green')
plt.title("XY coordinates - surface contour")

plt.figure()
plt.plot(d_z_b)
plt.title("height of boundary")

n = len(x_b_1)
t = np.linspace(0,1,n)
print ("shape of n")
print (n)

# plt.figure()
# plt.plot(t,r_xy)
# plt.title("radius for xy coordinates")
#
# plt.figure()
# plt.plot(t,d_z_b)
# plt.title("z values for boundary points")

# X = np.vander(t,degree+1)

poly = PolynomialFeatures(4)
X = poly.fit_transform(t.reshape(-1,1))

ridge_x = Ridge(alpha=1e-7)
ridge_y = Ridge(alpha=1e-7)
ridge_z = Ridge(alpha=1e-7)

ridge_x.fit(X,x_b_1)
ridge_y.fit(X,y_b_1)
ridge_z.fit(X,z_b_1)
x_boundary_smooth = ridge_x.predict(X)
y_boundary_smooth = ridge_y.predict(X)
z_boundary_smooth = ridge_z.predict(X)

ridge_x.fit(X,x_b_2)
ridge_y.fit(X,y_b_2)
ridge_z.fit(X,z_b_2)
x_boundary_smooth_2 = ridge_x.predict(X)
y_boundary_smooth_2 = ridge_y.predict(X)
z_boundary_smooth_2 = ridge_z.predict(X)

ridge_x.fit(X,x_b_3)
ridge_y.fit(X,y_b_3)
ridge_z.fit(X,z_b_3)
x_boundary_smooth_3 = ridge_x.predict(X)
y_boundary_smooth_3 = ridge_y.predict(X)
z_boundary_smooth_3 = ridge_z.predict(X)


ridge_x.fit(X,x_b_4)
ridge_y.fit(X,y_b_4)
ridge_z.fit(X,z_b_4)
x_boundary_smooth_4 = ridge_x.predict(X)
y_boundary_smooth_4 = ridge_y.predict(X)
z_boundary_smooth_4 = ridge_z.predict(X)





fig = plt.figure()
ax = plt.axes(projection='3d')
surf = ax.scatter(x_b_2, y_b_2, z_b_2)
surf = ax.scatter(x_b_1, y_b_1, z_b_1)
surf = ax.scatter(x_b_3, y_b_3, z_b_3)
surf = ax.scatter(x_b_4, y_b_4, z_b_4)
ax.plot(d_x_b,d_y_b,d_z_b)
surf = ax.scatter(x_boundary_smooth,y_boundary_smooth,z_boundary_smooth)
surf = ax.scatter(x_boundary_smooth_2,y_boundary_smooth_2,z_boundary_smooth_2)
surf = ax.scatter(x_boundary_smooth_3,y_boundary_smooth_3,z_boundary_smooth_3)
surf = ax.scatter(x_boundary_smooth_4,y_boundary_smooth_4,z_boundary_smooth_4)
plt.title("Problematic region 1")


###We then replace the points back into the original boundary points...
###First corner
d_x_c[0:n_points,0][::-1][:] = x_boundary_smooth[0:n_points]
d_x_c[0,1:n_points] = x_boundary_smooth[n_points:]
d_y_c[0:n_points,0][::-1][:] = y_boundary_smooth[0:n_points]
d_y_c[0,1:n_points] = y_boundary_smooth[n_points:]
d_z_c[0:n_points,0][::-1][:] = z_boundary_smooth[0:n_points]
d_z_c[0,1:n_points] = z_boundary_smooth[n_points:]
##Second corner
d_x_c[0,-n_points:] = x_boundary_smooth_2[0:n_points]
d_x_c[1:n_points,-1] = x_boundary_smooth_2[n_points:]
d_y_c[0,-n_points:] = y_boundary_smooth_2[0:n_points]
d_y_c[1:n_points,-1] = y_boundary_smooth_2[n_points:]
d_z_c[0,-n_points:] = z_boundary_smooth_2[0:n_points]
d_z_c[1:n_points,-1] = z_boundary_smooth_2[n_points:]
##Third corner
d_x_c[-n_points:,-1] = x_boundary_smooth_3[0:n_points]
d_x_c[-1,-n_points:][::-1][1:] = x_boundary_smooth_3[n_points:]
d_y_c[-n_points:,-1] = y_boundary_smooth_3[0:n_points]
d_y_c[-1,-n_points:][::-1][1:] = y_boundary_smooth_3[n_points:]
d_z_c[-n_points:,-1] = z_boundary_smooth_3[0:n_points]
d_z_c[-1,-n_points:][::-1][1:] = z_boundary_smooth_3[n_points:]
##Fourth corner
d_x_c[-1,0:n_points][::-1][:] = x_boundary_smooth_4[0:n_points]
d_x_c[-n_points:,0][::-1][1:] = x_boundary_smooth_4[n_points:]
d_y_c[-1,0:n_points][::-1][:] = y_boundary_smooth_4[0:n_points]
d_y_c[-n_points:,0][::-1][1:] = y_boundary_smooth_4[n_points:]
d_z_c[-1,0:n_points][::-1][:] = z_boundary_smooth_4[0:n_points]
d_z_c[-n_points:,0][::-1][1:] = z_boundary_smooth_4[n_points:]

d_x_b_c = np.hstack((d_x_c[0,middle_loc:],d_x_c[1:,-1],d_x_c[-1,:][::-1][1:],d_x_c[:,0][::-1][1:],d_x_c[0,1:middle_loc]))#*(0.95)
d_y_b_c = np.hstack((d_y_c[0,middle_loc:],d_y_c[1:,-1],d_y_c[-1,:][::-1][1:],d_y_c[:,0][::-1][1:],d_y_c[0,1:middle_loc]))#*(0.95)
d_z_b_c = np.hstack((d_z_c[0,middle_loc:],d_z_c[1:,-1],d_z_c[-1,:][::-1][1:],d_z_c[:,0][::-1][1:],d_z_c[0,1:middle_loc]))#*(0.95)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(d_x_b, d_y_b, d_z_b)
ax.scatter(d_x_b_c,d_y_b_c,d_z_b_c,color='red')
plt.title("Points on boundary - Design surface vs smoothed contour")

surf_points = np.stack((d_x_c.flatten(),d_y_c.flatten(),d_z_c.flatten()))

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# surf = ax.scatter(d_x,d_y,d_z)
# plt.title("Original surface points")
##Finally, we save the coordinates of the modified surface
surf_param_sN_save(surf_points,surf_points,"surf_q_smooth_c_401")





plt.show()



