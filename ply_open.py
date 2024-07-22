import numpy as np
import open3d as o3d
filepath ="D:\\white_light_interfer\\LP5\\"

measu_1= "top_surf_measurement_ls_40mu_tilt_removed_cloudcompare.xyz"
# target_pc = "target_pc.xyz"
measu_2= "top_surf_measurement_ls_65mu_tilt_removed_cloudcompare.xyz"

pcd_measu1= o3d.io.read_point_cloud(filepath+measu_1)
pcd_measu2 = o3d.io.read_point_cloud(filepath+measu_2)

points1 = np.asarray(pcd_measu1.points)
points2 = np.asarray(pcd_measu2.points)

print ("Max z values for both point clouds")
print (np.max(points1[:,2]))
print (np.max(points2[:,2]))

pcd_measu1.paint_uniform_color([1,0,0])
pcd_measu2.paint_uniform_color([0.8,0.8,0.8])


o3d.visualization.draw_geometries([pcd_measu1,pcd_measu2])
# o3d.visualization.draw_geometries([pcd_measu1])
