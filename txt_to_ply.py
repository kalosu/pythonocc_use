import numpy as np
import open3d as o3d
from common_functions import confocal_data_read
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



if __name__=="__main__":


    cmap = 'inferno'

    filepath ="D:\\white_light_interfer\\LP4\\"
    # cropped_flag = False
    # if cropped_flag:
    #     pass
    # else:
        # filename ="top_surf_measurement_ls_65mu.txt"
    print ("We open first the full length measurement data to extract information on the length")
    filename ="full_length_measurement.txt"
    cf_x, cf_y, cf_z = confocal_data_read(filepath+filename)


    pcd = o3d.geometry.PointCloud()

    points = np.stack((cf_y.flatten(), cf_x.flatten(), cf_z.flatten()), -1)

    print ("We want to calculate the location of the minimum")
    print (np.min(cf_z))
    print ("We also need to know the location of the maximum")
    print (np.max(cf_z))

    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.visualization.draw_geometries_with_editing([pcd])

    max_height_reference = np.min(cf_z) + np.max(cf_z)
    print ("The relative position of the maximum with respect to the substrate is:")
    print (max_height_reference)
    print ("We now open and process the data for the measurements containing only the optical surface")


    filename2 ="top_surf_measurement_ls_65mu_tilt_removed_cloudcompare.xyz"
    cf2_x, cf2_y, cf2_z = confocal_data_read(filepath+filename2)

    filename3 ="top_surf_measurement_ls_40mu_tilt_removed_cloudcompare.xyz"
    cf3_x, cf3_y, cf3_z = confocal_data_read(filepath+filename3)







    print ("We now determine the height from the maximum point on the surface measurement data")
    print (np.max(cf2_z))
    print (np.max(cf3_z))
    global_z_offset = max_height_reference - np.max(cf2_z)
    global_z_offset2 = max_height_reference - np.max(cf3_z)
    print ("The global offset that needs to be applied is")
    print (global_z_offset)
    print (global_z_offset2)

    print ("Z value of highest point on surface after offset")
    print (np.max(cf2_z) + global_z_offset)
    print (np.max(cf3_z) + global_z_offset2)



    points2 = np.stack((cf2_y.flatten(), cf2_x.flatten(), cf2_z.flatten()), -1)
    points3 = np.stack((cf3_y.flatten(), cf3_x.flatten(), cf3_z.flatten()), -1)

    fig = plt.figure()
    ax1 = fig.add_subplot()
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax1.tricontour(points2[:, 0], points2[:, 1], points2[:, 2], levels=50, cmap=cmap)
    plt.colorbar(im, cax=cax, orientation='vertical')
    plt.title("Measured surface - After global Z offset")
    plt.show()

    pcd2 = o3d.geometry.PointCloud()

    pcd2.points = o3d.utility.Vector3dVector(points2)

    pcd3 = o3d.geometry.PointCloud()

    pcd3.points = o3d.utility.Vector3dVector(points3)

    pcd2.paint_uniform_color([1, 0, 0])
    pcd3.paint_uniform_color([0.8, 0.8, 0.8])

    # o3d.visualization.draw_geometries_with_editing([pcd2])
    o3d.visualization.draw_geometries([pcd2,pcd3])








