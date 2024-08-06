from common_functions import confocal_data_read
import numpy as np
import open3d as o3d

if __name__=="__main__":

    # filepath = "D:\\white_light_interfer\\rq_surface_extended_offset_hatching_30_deg\\" ###White light interferometer based measurement data
    # filepath ="D:\\white_light_interfer\\rq_surf_offset_hatching_30_deg_iter_2\\no_icp\\"
    filepath ="D:\\Confocal_measurements\\beam_shaping_project\\rq_surface_iterations\\rq_lens_hatchin_offset_30_deg_iter_2\\with_icp\\"

    cropped_flag = True
    not_corrected= True
    if cropped_flag:
        filename_2 = "with_icp_upper_surface_measu.ply" ###ply file after editing
        pcd = o3d.io.read_point_cloud(filepath + filename_2)
        points = np.asarray(pcd.points)
        cf_x2 = points[:,0]
        cf_y2 = points[:,1]
        cf_z2 = points[:,2]
    else:
        filename_2 = "with_icp_upper_surface_measu.txt"
        cf_x2, cf_y2, cf_z2 = confocal_data_read(filepath+filename_2)

    filename = "with_icp_full_surface.txt"
    # filename_2 = "rq_surf_LP5_upper_surface.txt"

    cf_x1, cf_y1, cf_z1 = confocal_data_read(filepath+filename)
    # cf_x2, cf_y2, cf_z2 = confocal_data_read(filepath+filename_2)

    point_reference = np.stack((cf_x1[0],cf_y1[0],15),-1)[np.newaxis,:]

    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(point_reference)
    pcd= o3d.io.read_point_cloud(filepath+filename)


    print ("The first entry of z")
    print (cf_z1[0])

    pcd = o3d.geometry.PointCloud()
    points = np.stack((cf_x1.flatten(), cf_y1.flatten(), cf_z1.flatten()), -1)

    points = np.concatenate((points,point_reference))
    pcd.points = o3d.utility.Vector3dVector(points)

    # z_min_ref = np.min(cf_z1)
    z_min_ref = cf_z1[0]
    z_max_ref = np.max(cf_z1)
    z_height_ref = z_max_ref - z_min_ref
    print ("For the reference full length structure, these are the parameters")
    print ("Min z: {}".format(z_min_ref))
    print ("Max z: {}".format(z_max_ref))
    print ("Z Height: {}".format(z_height_ref))

    pcd_2 = o3d.geometry.PointCloud()
    points_2 = np.stack((cf_x2.flatten(), cf_y2.flatten(), cf_z2.flatten()), -1)
    pcd_2.points = o3d.utility.Vector3dVector(points_2)

    print ("We first plot the measurement data points")

    # pcd_ref.paint_uniform_color([1,0,0])
    # pcd.paint_uniform_color([0.8,0.8,0.8])



    o3d.visualization.draw_geometries([pcd,pcd_ref])
    # o3d.visualization.draw_geometries([pcd])

    z_min_upper = np.min(cf_z2)
    z_max_upper = np.max(cf_z2)
    z_height_upper = z_max_upper - z_min_upper

    print ("For the upper surface measurement data, these are the parameters")
    print ("Min z: {}".format(z_min_upper))
    print ("Max z: {}".format(z_max_upper))
    print ("Z Height: {}".format(z_height_upper))

    if not_corrected:

        print ("We need to apply a global offset to all points from the upper surface so that the surface is at the correct height")
        z_global_offset = z_height_ref -z_max_upper

        z_upper_corrected = cf_z2 + z_global_offset



        pcd_2 = o3d.geometry.PointCloud()
        points_2 = np.stack((cf_x2.flatten(), cf_y2.flatten(), z_upper_corrected.flatten()), -1)
        pcd_2.points = o3d.utility.Vector3dVector(points_2)

        z_min_upper = np.min(z_upper_corrected)
        z_max_upper = np.max(z_upper_corrected)
        z_height_upper = z_max_upper - z_min_upper

    print ("For the upper surface measurement data, these are the parameters (after correction)")
    print ("Min z: {}".format(z_min_upper))
    print ("Max z: {}".format(z_max_upper))
    print ("Z Height: {}".format(z_height_upper))


    o3d.visualization.draw_geometries_with_editing([pcd_2])


