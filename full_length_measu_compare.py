###Since we observe deviations on the position of the upper surfaces with respect to the substrates,
##can we see if these deviations are really there on the structures or are we parsing something wrong??
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from common_functions import confocal_data_read

filepath = "D:\\white_light_interfer\\rq_surface_extended_comp_q_exposed_reproducibility_eval_20240718\\"

filename_full_1 = "LP3_1_full_length.txt"

filename_full_2 = "LP3_2_full_length.txt"

cf_x1, cf_y1, cf_z1 = confocal_data_read(filepath + filename_full_1)

cf_z1 -= cf_z1[0]

print ("Maximum z first structure")
print (np.max(cf_z1))



cf_x2, cf_y2, cf_z2 = confocal_data_read(filepath + filename_full_2)



cf_z2 -= cf_z2[0]

print ("Maximum z second structure")
print (np.max(cf_z2))

pcd_1 = o3d.geometry.PointCloud()
points_1 = np.stack((cf_x1.flatten(), cf_y1.flatten(), cf_z1.flatten()), -1)
pcd_1.points = o3d.utility.Vector3dVector(points_1)

pcd_2 = o3d.geometry.PointCloud()
points_2 = np.stack((cf_x2.flatten(), cf_y2.flatten(), cf_z2.flatten()), -1)
pcd_2.points = o3d.utility.Vector3dVector(points_2)


pcd_1.paint_uniform_color([1,0,0])
pcd_2.paint_uniform_color([0.8,0.8,0.8])


o3d.visualization.draw_geometries([pcd_1, pcd_2])






