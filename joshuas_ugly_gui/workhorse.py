import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import time
import copy
import pandas as pd
from sklearn.cluster import SpectralClustering
from scipy.optimize import curve_fit
import csv
import tkinter as tk
from scipy.optimize import minimize
import cv2
from scipy.interpolate import griddata
from scipy.optimize import minimize_scalar
import datetime
import os

def untilt_profile_byedges (input):
    m=(input)
    mlx =[]
    mrx =[]             #split profiles in left and right ml, mr
    mly =[]
    mry =[]             #split profiles in left and right ml, mr

    i=0
    while i<len(m[0]):          #split array in two halfs (so i can calculate min of both halfs)
        if i<(len(m[0])/2):
            mlx.append(m[0][i])
            mly.append(m[1][i])
        if i>=(len(m[0])/2):
            mrx.append([m[0][i]])
            mry.append([m[1][i]])
        i=i+1

    i=0
    while i<len(mly):           #for tilt angle clac
        if mly[i]==min(mly):
            pl=i
        i=i+1
    i=0
    while i<len(mry):           #for tilt angle clac
        if mry[i]==min(mry):
            pr=i
        i=i+1

    m[0]=m[0]-(mlx[pl])
    m[1]=m[1]-mly[pl]
    mrx[pr]=mrx[pr]-mlx[pl]
    mry[pr]=mry[pr]-mly[pl]
    mlx[pl]=0
    mly[pl]=0

    angl=-(np.arctan((mry[pr]-mly[pl])/(mrx[pr]-mlx[pl])))
    m_old=np.copy(m)
    i=0
    while i<len(m[0]):
        ad=np.sqrt(((m[0][i]-mlx[pl])**2)+((m[1][i]-mly[pl])**2))
        m[0][i]=m[0][i]*np.cos(angl)-m[1][i]*np.sin(angl)
        m[1][i]=m[0][i]*np.sin(angl)+m[1][i]*np.cos(angl)
        i=i+1


    plt.scatter(m[0],m[1],label='rotated')
    plt.scatter(m_old[0],m_old[1],label='og shit')
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()
    einarray=np.r_[[m[0]],[m[1]]]
    return einarray

def extr_prof_by_height (path):
    m1=pd.read_csv(path,delimiter='\t')
    m2=np.array(m1)
    plt.scatter(sort_data(m2)[0],sort_data(m2)[1],label='determine ylim')
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.show()
    ylim=input('ylim:')
    ylim=float(ylim)
    xarr=[]
    yarr=[]
    i=0
    while i<len(m2):
        if m2[i][1] > ylim:
            xarr.append(m2[i][0])
            yarr.append(m2[i][1])
        i=i+1

    einarray=np.r_[[xarr],[yarr]]
    plt.scatter(einarray[0],einarray[1],size=1)
    plt.gca().set_aspect('equal')
    plt.show()
    return einarray


def plot_ply(input_ply,COSVAR):

    pcd = o3d.io.read_point_cloud(input_ply)
    grd = o3d.io.read_point_cloud('nanogrid.ply')
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(100,[0,0,0])

    if COSVAR==True:
        o3d.visualization.draw_geometries([pcd,mesh_frame])
    else:
        o3d.visualization.draw_geometries_with_editing([pcd])


def twd_comp (c,path,it):          #0-> input data path for measurements, 1 input array with measurement data or methode that outouts (processed) measurement data

    if c==0:
        m1 = pd.read_csv(path, delimiter='\t')
        m1=np.array(m1)
    if c==1:
        m1=np.transpose(path)
    else:
        print('data path error')
    xl=-70
    i=xl
    soll=[]
    x=[]
    while i<abs(xl):
        x.append(i)
        soll.append(f_sphere(i,100))
        i=i+0.01



    a0=np.r_[[x],[soll]]

    np.savetxt('sollx.txt',a0[0])
    np.savetxt('solly.txt',a0[1])

    a1=np.r_[[sort_data(m1)[0]],[sort_data(m1)[1]]]

    i=xl
    yc1=[]
    xc1=[]
    if it>0:
        while i < abs(xl):
            xc1.append(i)
            yc1.append(fc1(i))
            i = i + (abs(xl)*2)/len(a1[0])

    ac1=np.r_[[xc1],[yc1]]

    a1[0]=(a1[0]-min(a1[0])-((max(a1[0])-min(a1[0])))/2)+0.35        #center surface over x=0
    a1[1]=a1[1] + min(a0[1])-0.35                         #matching in y


    a1_old=1*a1

    np.savetxt('istx.txt',a1_old[0])
    np.savetxt('isty.txt',a1_old[1])

    plt.scatter(a0[0],a0[1],s=3)
    plt.scatter(a1_old[0],a1_old[1],s=3)
    plt.gca().set_aspect('equal')

    plt.show()


    i=0
    pos1=[]
    p1=0
    p0=0
    pos0=[]
    start_time = time.time()



    while i<(len(a1[0])-1):            #calculating the shortest distance
        k=0
        temp=10**10
        while k<(len(a0[0])-1):
            temp_o = temp
            temp=np.sqrt(((float(a1[0][i])-float(a0[0][k]))**2)+((float(a1[1][i])-float(a0[1][k]))**2))
            if ((temp_o)>(temp)):
                p1=k
                p0=i
            k=k+1
        pos1.append(p0)
        pos0.append(p1)
        i=i+1

    end_time = time.time()

    print("Time taken: ", end_time - start_time, "seconds")

    i=0
    fd =2 #factor for scaling 1- just fit, 2 corrects by 1x error
    ax=[]
    ay=[]

    a11=1*a1

    rm=[]
    while i<len(pos0):
        a1[1][i] = (a1[1][i] - fd*((a1[1][pos1[i]]) - (a0[1][pos0[i]])))
        a1[0][i] = (a1[0][i] - fd*((a1[0][pos1[i]]) - (a0[0][pos0[i]])))
        ax.append(float(a1[1][i]))
        ay.append(float(a1[0][i]))

        rmy=((a1_old[1][pos1[i]]) - (a0[1][pos0[i]]))
        rmx=((a1_old[0][pos1[i]]) - (a0[0][pos0[i]]))
        plt.plot((a1[0][pos1[i]],a1[0][pos1[i]]-rmx),(a1[1][pos1[i]],a1[1][pos1[i]]-rmy))
        rm.append(np.sqrt((rmx**2)+(rmy**2)))
        i=i+1

    if it>0:
        while i < abs(xl):
            xc1.append(i)
            yc1.append(fc1(i))
            i = i + (abs(xl)*2)/len(a1[0])

    axc1=[]
    ayc1=[]
    i=0
    fd2=1.25
    if it>0:
        while i < len(pos0):
            ac1[1][i] = (ac1[1][i] - (fd2)*((a11[1][pos1[i]]) - (a0[1][pos0[i]])))
            ac1[0][i] = (ac1[0][i] - (fd2)*((a11[0][pos1[i]]) - (a0[0][pos0[i]])))
            axc1.append(float(ac1[1][i]))
            ayc1.append(float(ac1[0][i]))
            i=i+1

    print('___dem seine RMS tut vielleicht sein:eins millionen meter, lol nein spaß:')
    print(sum(rm)/len(rm))
    spreg, pcov = curve_fit(f_p1, ay, ax,maxfev=2*10**9)
    spreg2, pcov = curve_fit(f_p2, ay, ax,maxfev=2*10**9)
    if it>0:
        it2, pcov = curve_fit(f_p2,ac1[0],ac1[1])
    arr=[]
    arr2=[]
    print('_____________Resurlts of profile comp_______________________________')
    print('for comp of', (fd-1)*100,'%of error to:')
    print('curve fit for ax4,bx3,cx2,dx,e')
    print(spreg)
    print('(', spreg2[0], '*((x^4)/1000000)+', spreg2[1], '*((x^3)/1000000)+', spreg2[2],'*((x^2)/1000000)+',spreg2[3], '*(x/1000000)+', spreg2[4],')')
    if it>0:
        print('_____________Resurlts second iteration_______________________________')
        print('for comp of', ((fd2))*100,'%of error to first fit:')
        print('curve fit for ax4,bx3,cx2,dx,e')
        print('(', it2[0], '*((x^4)/1000000)+', it2[1], '*((x^3)/1000000)+', it2[2],'*((x^2)/1000000)+',it2[3], '*(x/1000000)+', it2[4],')')
    i=-70
    itarr=[]
    if it>0:
        while i<70:
            arr.append(f_p1(i, spreg[0], spreg[1],spreg[2],spreg[3],spreg[4]))
            arr2.append(f_p2(i,spreg2[0],spreg2[1],spreg2[2],spreg2[3],spreg2[4]))
            itarr.append(f_p2(i,it2[0],it2[1],it2[2],it2[3],it2[4]))
            i=i+1

    plt.scatter(a0[0],a0[1],s=2,label='soll')
    plt.scatter(a1[0],a1[1],s=2,label='compensated')
    plt.scatter(a1_old[0],a1_old[1],s=2,label='ist')
    plt.plot(np.arange(-70,70,1),arr2,label='fit')
    plt.plot(np.arange(-70,70,1),itarr,label='fit2')
    plt.gca().set_aspect('equal')
    if it>0:
        plt.scatter(ac1[0],ac1[1],s=15,label='i1 minus i1 delta')
        plt.scatter(xc1,yc1,label='solve of i1 function')
    plt.legend()
    plt.show()

    plt.plot(abs(np.arange(-70,70,1)),arr,label='check of symetrie')
    plt.plot(abs(np.arange(-70,70,1)),arr2,label='check of symetrie')
    plt.legend()
    plt.show()


def txt_to_ply_simple(input,output):
    print('performing a data conversation txt to ply...')
    pcd = o3d.io.read_point_cloud(input, format="xyz")
    o3d.io.write_point_cloud(output, pcd)
    print('...done: look for ply file:',output)
    o3d.visualization.draw_geometries([pcd])
    return pcd


def base_extraction_a(input):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the subfolder
    subfolder_name = 'temp'
    subfolder_path = os.path.join(current_dir, subfolder_name)

    filename = "tempforplot.ply"

    file_path = os.path.join(subfolder_path, filename)

    txt_to_ply_simple(input, file_path)
    print(".....................")
    print("loading initial data")
    plot_ply(file_path,False)


    def read_pointcloud(file_path):
        # Create an empty list to store the point cloud
        point_cloud = []

        # Open the file and read in the point cloud
        with open(file_path, 'r') as file:
            # Use the csv module to read in the file
            reader = csv.reader(file, delimiter=' ')
            # Append each point to the point cloud list
            for row in reader:
                try:
                    point_cloud.append([float(row[0]), float(row[1]), float(row[2])])
                except ValueError:
                    continue

        return point_cloud

    def write_lowest_z_points(point_cloud, file_path):
        # Find the minimum z value
        min_z = min(point[2] for point in point_cloud)
        min_z=abs(min_z)
        # Find the minimum z value +/- 10%
        min_z_range = min_z * 0.9, min_z * 1.2

        # Create an empty list to store the points with the lowest z values
        lowest_z_points = []
        # Iterate through the point cloud and append points that fall within the min_z_range
        for point in point_cloud:
            if point[2] <= min_z_range[1]:  #min_z_range[0] <=
                lowest_z_points.append(point)


        # Open a new file and write the lowest z points to it
        with open(file_path, 'w') as file:
            writer = csv.writer(file, delimiter=' ')
            for point in lowest_z_points:
                writer.writerow(point)

    # Read in the point cloud file
    point_cloud = read_pointcloud(input)
    # Write the lowest z

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the subfolder
    subfolder_name = 'temp'
    subfolder_path = os.path.join(current_dir, subfolder_name)

    filename = "base.txt"

    file_path = os.path.join(subfolder_path, filename)

    write_lowest_z_points(point_cloud, file_path)
    print('..................')
    print('extracting points of base:')
    print('... done -> look for  ..temp/base.txt file in project folder')

def dd_to_2dimg (input):
    #base_extraction_a(input)
    print('...............')
    print('converting 3d base points in 2d jpg')
    data=np.loadtxt(input)
    x=[]
    y=[]
    for line in data:
        x.append(line[0])
        y.append(line[1])


    fig = plt.figure(frameon=False,figsize=(9,9))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.scatter(x,y)
    plt.gca().set_aspect('equal')
    plt.xlim(np.min(x),np.max(x))
    plt.ylim(np.min(y),np.max(y))

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the subfolder
    subfolder_name = 'temp'
    subfolder_path = os.path.join(current_dir, subfolder_name)

    filename = "foo.jpg"

    file_path = os.path.join(subfolder_path, filename)

    fig.savefig(file_path,dpi=400)

    print('... done -> look for  ..temp/foo.jpg file in project folder')
    plt.show()
    return (np.max(x)-np.min(x)), (np.max(y)-np.min(y))

def load_txt_return_array (input):
    arr=np.loadtxt(input)
    return arr

def icp_2xpcd(in1,in2):
    pcd = o3d.io.read_point_cloud(in1,format='xyz')
    pcd2 = o3d.io.read_point_cloud(in2,format='xyz')
    pcd.paint_uniform_color([1, 0, 0])
    pcd2.paint_uniform_color([0, 0, 1])

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([pcd, pcd2, mesh_frame])

    source = copy.deepcopy(pcd2)
    target = copy.deepcopy(pcd)
    source_down = source.voxel_down_sample(voxel_size=1)

    rotate = False
    if rotate == True:
        print("me do rotate")
        # Define the rotation angle in radians (180 degrees)
        angle_rad = np.pi

        # Define the rotation axis (Z-axis)
        rotation_vector = [0, 0, angle_rad]

        # Get rotation matrix from angle and axis
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)

        source.rotate(R)

    print('calc ipc:')
    threshold = 200  # Maximum correspondence points-pair distance
    trans_init = np.asarray([[1, 0.0, 0.0, 0.0],
                             [0.0, 1, 0.0, 0.0],
                             [0.0, 0.0, 1, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
    # coarse aligh with down pcd_should help to avoid loacl minima
    reg_p2p = o3d.pipelines.registration.registration_icp(source_down, target, threshold, trans_init,
                                                          o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                          o3d.pipelines.registration.ICPConvergenceCriteria(relative_rmse=1e-6,
                                                              max_iteration=10 ^ 9))  # max(max_iterationbs=10^9)
    source.transform(reg_p2p.transformation)
    trans_init = reg_p2p.transformation

    # fine fit
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=int(1e9))
    #reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())  # max(max_iterationbs=10^9),,o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1e9),

    reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,o3d.pipelines.registration.TransformationEstimationPointToPoint(),criteria=criteria)

    print(reg_p2p)
    print('_________________')
    print(reg_p2p.transformation)
    source.transform(reg_p2p.transformation)
    o3d.visualization.draw_geometries([source, target, mesh_frame])

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the subfolder
    subfolder_name = 'temp'
    subfolder_path = os.path.join(current_dir, subfolder_name)

    filename1 = "p1.xyz"
    filename2 = "p2.xyz"

    file_path1 = os.path.join(subfolder_path, filename1)
    file_path2 = os.path.join(subfolder_path, filename2)

    o3d.io.write_point_cloud(file_path1, source)
    o3d.io.write_point_cloud(file_path2, target)
    return reg_p2p.transformation

def here_iHough_again_a (input):            #play with smoothin loop might help inproving results, or sacrifice a virgin to the dark lord of the underworld for what i know
    xl, yl = dd_to_2dimg(input)
    data = np.loadtxt(input)
    print('performing hough transform......')
    xl=int(xl)
    yl=int(yl)

    # Load the image

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the subfolder
    subfolder_name = 'temp'
    subfolder_path = os.path.join(current_dir, subfolder_name)

    filename = "foo.jpg"

    file_path = os.path.join(subfolder_path, filename)

    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.imshow(img)
    plt.show()

    print('smooting jpg......')

    i=0
    thr = 4                                       #smoothing loop
    while i<thr:
        img = cv2.GaussianBlur(img, (3, 3), 0)
        i=i+1


    print('done')

    plt.imshow(img)
    plt.show()

    img = cv2.resize(img, (xl, yl))


    # Apply the Hough Circle Transform to detect circles in the image
    circles = cv2.HoughCircles(img,  cv2.HOUGH_GRADIENT, dp=1, minDist=30000, param1=1, param2=10, minRadius=10,maxRadius=1000)  # (input_array, output (ia circles), methode, dp,mindist, param1 (low:smal circles, large:big circles?),param2 (distance between circles?),minrad,maxrad)

    print('ah shit here i hough agian:')
    print(circles)

    # Convert the circles array to a more usable format
    circles = np.uint16(circles)

    # Plot the circles on the original image
    for i in circles[0, :]:
        cx=i[0]
        cy=i[1]
        cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 0), 1)
        cv2.circle(img, (i[0], i[1]), 2, (0, 255, 0), 1)

    # Show the image with the circles plotted on it
    print('...done')
    plt.imshow(img)
    plt.show()

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the subfolder
    subfolder_name = 'temp'
    subfolder_path = os.path.join(current_dir, subfolder_name)

    filename = "topofsurf.ply"

    file_path = os.path.join(subfolder_path, filename)

    pcd = o3d.io.read_point_cloud(file_path)


    data=np.transpose(data)

    pcd = copy.deepcopy(pcd).translate((-1*np.min(data[0]),-1*np.min(data[1]),0))

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(100,[0,0,0])
    cyl=o3d.geometry.TriangleMesh.create_cylinder(2,100,1000)
    cyl = copy.deepcopy(cyl).translate((cx, (yl-cy), 0))
    o3d.visualization.draw_geometries([pcd,mesh_frame,cyl])
    print('_______________')
    print('array circle canter by hough transform is:')
    print(cx,((yl-cy)))
    return cx, (yl-cy)

def plot_with_mag(input,mx,my,mz,COSVAR,save=None):
    data=load_txt_return_array(input)
    data=np.transpose(data)
    mx=float(mx)
    my=float(my)
    mz=float(mz)
    print(data)
    data[0]=mx*data[0]
    data[1]=my*data[1]
    data[2]=mz*data[2]
    data=np.transpose(data)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the subfolder
    subfolder_name = 'temp'
    subfolder_path = os.path.join(current_dir, subfolder_name)

    filename = "scaleplottemp.xyz"

    file_path = os.path.join(subfolder_path, filename)

    np.savetxt(file_path,data)

    pcd = o3d.io.read_point_cloud(file_path)

    grd = o3d.io.read_point_cloud('nanogrid.ply')
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(100,[0,0,0])

    if COSVAR==True:
        o3d.visualization.draw_geometries([pcd,mesh_frame])
    else:
        o3d.visualization.draw_geometries([pcd])

    if save:
        directory = os.path.dirname(input)
        filename = os.path.basename(input)
        base_name = os.path.splitext(filename)[0]
        print(base_name)
        loc= os.path.join("\\" + base_name + "_scaled.xyz")
        loc_name = str(directory+loc)
        #o3d.io.write_point_cloud(loc_name, pcd)
        np.savetxt(loc_name, data)
        print('mag pcd saved at'+loc_name)

def cut(input,output):
    pcd = o3d.io.read_point_cloud(input)
    cutter_cy = o3d.geometry.TriangleMesh.create_cylinder(100,300,resolution=500)
    cutter_cy = copy.deepcopy(cutter_cy).translate((150, 150, 0))
    obb = cutter_cy.get_oriented_bounding_box()
    cut = pcd.crop(obb)
    print(obb)
    pcd_c=o3d.visualization.draw_geometries_with_editing([pcd])
    o3d.io.write_point_cloud(output, pcd_c)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(100,[0,0,0])
    o3d.visualization.draw_geometries([pcd,mesh_frame])
    o3d.visualization.draw_geometries_with_editing([pcd]) #press k and contr click for selection, then press c to crop. NOW TAKE CARE TO GIVE THE FILE A.ply ENDING OTHERWISE YOU JUST CET THE CROPPING MASK !!!!!!!

def ply_to_txt (input,name):                #output with .xyz fileextension
    print('converting ply to txt....')
    pcd=o3d.io.read_point_cloud(input)
    o3d.io.write_point_cloud(name, pcd)
    print('...done look for :'+name)

def redgrid(input1,input2,fit):
    from scipy.interpolate import griddata

    if not fit:
        print('NO fitting will be applyed')
        data = np.loadtxt(input1)
        data2 = np.loadtxt(input2)


    if fit:
        print('fitting will be applyed')
        icp_2xpcd(input1,input2)

        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the relative path to the subfolder
        subfolder_name = 'temp'
        subfolder_path = os.path.join(current_dir, subfolder_name)

        filename1 = "p1.xyz"
        filename2 = "p2.xyz"

        file_path1 = os.path.join(subfolder_path, filename1)
        file_path2 = os.path.join(subfolder_path, filename2)

        data = np.loadtxt(file_path1)
        data2 = np.loadtxt(file_path2)



    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    x1,y1,z1 =data2[:, 0], data2[:, 1], data2[:, 2]


    gridstart_x,gridend_x=min(x),max(x)
    #gridstart_x, gridend_x =(-60),(15)

    gridstart_y,gridend_y=min(y),max(y)
    #gridstart_y, gridend_y =(-15),(15)

    gridscale=0.1

    # Generate a 2D grid with color representing the third dimension
    XI,YI = np.meshgrid(np.arange(gridstart_x,gridend_x,gridscale),np.arange(gridstart_y,gridend_y,gridscale))
    print('gridsize')
    print(len(XI),'x',len(YI))


    # Interpolate the data on the grid
    grid_z = griddata((x, y), z, (XI, YI), method='linear')
    grid_z1 = griddata((x1,y1), z1, (XI, YI), method='linear')

    compgrid_raw=grid_z-grid_z1
    compgrid=grid_z-grid_z1

    batman=False

    if batman:
        print('I am Batman')
        xmin0=54           #masks to mask out the alignment markers
        xmax0=300
        ymin0=170
        ymax0=200

        xmin1=193
        xmax1=250
        ymin1=100
        ymax1=200

        xmin2=0
        xmax2=300
        ymin2=100
        ymax2=130

        xmin3=0
        xmax3=100
        ymin3=0
        ymax3=200


        mask0 = (XI >= xmin0) & (XI <= xmax0) & (YI >= ymin0) & (YI <= ymax0)
        mask1 = (XI >= xmin1) & (XI <= xmax1) & (YI >= ymin1) & (YI <= ymax1)
        mask2 = (XI >= xmin2) & (XI <= xmax2) & (YI >= ymin2) & (YI <= ymax2)
        mask3 = (XI >= xmin3) & (XI <= xmax3) & (YI >= ymin3) & (YI <= ymax3)

        compgrid[mask0] = np.nan
        compgrid[mask1] = np.nan
        compgrid[mask2] = np.nan
        compgrid[mask3] = np.nan

    print(np.nanmean(abs(compgrid)),'µm')

    colorbar_by_max=True

    if colorbar_by_max:
        maxcolor=np.nanmax(compgrid)
        maxcolor=0.6*maxcolor
        mincolor=np.nanmin(compgrid)
        mincolor=0.6*mincolor
        plt.scatter(XI,YI,c=compgrid,cmap="viridis",s=7,vmin=mincolor, vmax=maxcolor,marker='s')

    if not colorbar_by_max:
        maxcolor=np.nanmax(compgrid)
        maxcolor=0.6*maxcolor
        mincolor=np.nanmin(compgrid)
        mincolor=0.6*mincolor
        plt.scatter(XI,YI,c=compgrid,cmap="viridis",s=7,vmin=-0.1, vmax=0.1,marker='s')

    #plt.colorbar()
    colorbar = plt.colorbar()
    colorbar.set_label('µm', labelpad=25, rotation=90)


    plt.gca().set_aspect('equal')
    filename = os.path.basename(input1)
    filename2 = os.path.basename(input2)
    thename=str(filename + ' minus ' + filename2)
    thename = thename.replace('.xyz', '')
    plt.title(thename)

    plt.show()
    #np.savetxt(r"cut_fitted.xyz", new_array)

    savesavesavedisave_yeahyeah=True

    if savesavesavedisave_yeahyeah:
        print('saving results...')
        x = XI.flatten()
        y = YI.flatten()
        z = compgrid.flatten()

        z0 = grid_z.flatten()
        z1 = grid_z1.flatten()

        a = np.c_[x, y, z]
        a0 = np.c_[x, y, z0]
        a1 = np.c_[x, y, z1]

        # Find rows that don't contain NaN values
        valid_rows = np.logical_not(np.isnan(a).any(axis=1))

        # Create a new array by selecting valid rows
        new_array = a[valid_rows]
        new_array0 = a0[valid_rows]
        new_array1 = a1[valid_rows]

        #new_array=a #so it has the same size (in theory)

        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the relative path to the subfolder
        subfolder_name = 'gridres'
        subfolder_path = os.path.join(current_dir, subfolder_name)

        filename1 = thename + "_raw0.xyz"
        filename2 = thename + "_raw1.xyz"
        filename3 = thename + ".xyz"

        file_path = os.path.join(subfolder_path, filename)

        np.savetxt(filename1,new_array0)
        np.savetxt(filename2,new_array1)
        np.savetxt(filename3,new_array)
        print('done.. look for : '+r"\gridres\_"+thename+".xyz")

    comp=True

    if comp:
        x = XI.flatten()
        y = YI.flatten()
        z = grid_z1-compgrid
        z = z.flatten()
        a = np.c_[x, y, z]
        valid_rows = np.logical_not(np.isnan(a).any(axis=1))
        new_array = a[valid_rows]
        np.savetxt(r"compensated.xyz",new_array)
        print('done.. look for : compensated.xyz')


def layervis_single(inputname,profspc):   #profspc step distance of profiles
    # !/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """
    Created on Wed May 24 16:28:27 2023

    @author: anton
    modifyed by joshua

    Program to load the data and plot the different YZ-layers of the lense

    save your data set in the folder 'resources' and change the variable 'file_name'

    in case there is more the one layer/line: adjust the variabel 'step'

    Things to be done:
        + write a sorting function which cuts out the filtered data
        + change file checking to pathlib https://stackoverflow.com/questions/82831/how-do-i-check-whether-a-file-exists-without-exceptions
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib.widgets import Slider
    import time
    from pathlib import Path

    file_name = inputname  # enter your file_name
    # file_name = 'data02_t55_PM_04_11_2023.xyz'
    project_dir = Path().parent.absolute().parent.absolute()
    file = project_dir / 'resources' / file_name

    # functions

    def load_data(file_path):
        if Path(file_path).exists():
            t1 = time.time()
            # raw_data = np.genfromtxt(file_path)  # np.loadtxt() is lighter if it does the job use this one
            raw_data = np.loadtxt(file_path)
            t2 = time.time()
            print('loading time: ', (t2 - t1))
            return raw_data
        else:
            print('File not exists')

    def gen_test_data(data_set, step):
        return data_set[::step].copy()

    def scatter_3D(data_set):
        if type(data_set) == np.ndarray:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x, y, z = axes3d.get_test_data(0.05)
            ax.plot(data_set[:, 0], data_set[:, 1], data_set[:, 2], 'o')
            plt.show()
        else:
            print('wrong data type')

    def update(val):
        layer_scatter.set_xdata(layer_set[val][:, 1])
        layer_scatter.set_ydata(layer_set[val][:, 2])
        fig.canvas.draw_idle()

    # %% ##########################################################################

    data = load_data(file)

    step = profspc
    layer_first = int(np.min(data[:, 0])) - step
    index_first = int((layer_first / step))
    layer_last = int(np.max(data[:, 0])) + 1
    index_last = int(layer_last / step)

    y_min = int(np.min(data[:, 1])) - 5
    y_max = int(np.max(data[:, 1])) + 5
    z_min = int(np.min(data[:, 2])) - 5
    z_max = int(np.max(data[:, 2])) + 5

    t1 = time.time()
    layer_set = []
    for i in range(index_first, index_last, 1):
        real_range = i * step
        layer = data[np.where(
            (data[:, 0] >= real_range) &
            (data[:, 0] < real_range + step))]
        layer_set.append(layer)
    t2 = time.time()
    print('sorting: ', (t2 - t1))

    fig = plt.figure()  # create figure
    ax = fig.add_subplot(111, xlim=(y_min, y_max),
                         ylim=(z_min, z_max),
                         xlabel='y-axis',
                         ylabel='z-axis',
                         title='YZ-Layers of the Lense')  # create axes
    layer_scatter, = ax.plot(layer_set[0][:, 1], layer_set[0][:, 2], '.')

    fig.subplots_adjust(bottom=0.25)

    axlayer = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    layer_slider = Slider(
        ax=axlayer,
        label='Layer',
        valmin=0,
        valmax=len(layer_set) - 1,
        valstep=1,
        valinit=0)

    layer_slider.on_changed(update)
    plt.show()

def onegrid(path,gridscale):
    from scipy.interpolate import griddata
    arr = np.loadtxt(path)

    #symetric grid so startx = starty =start
    grid_start= np.min(arr)
    grid_end = np.max(arr)

    #gridscale=0.1
    print('gridscale:' + gridscale)
    gridscale=float(gridscale)
    #print('gridscale:'+gridscale)

    XI,YI = np.meshgrid(np.arange(grid_start,grid_end,gridscale),np.arange(grid_start,grid_end,gridscale))


    grid_z = griddata((arr[:, 0], arr[:, 1]), arr[:, 2], (XI, YI), method='linear') #interpaolate the data
    grid_z = np.nan_to_num(grid_z, nan=0)

    np.savetxt('onegrid_grid.xyz',grid_z)
    print('grid store in project folder: onegrid_grid.xyz')

    #concert meshgrid into a three coulmn array:
    x = XI.flatten()
    y = YI.flatten()
    grid_z=grid_z.flatten()
    grid_z=grid_z-np.min(grid_z)

    a=np.column_stack((x.ravel(),y.ravel(),grid_z))

    non_zero_indices = np.nonzero(a[:, 2] != 0)[0]  # remove 0 rows
    a = a[non_zero_indices]

    np.savetxt('onegrid.xyz',a)
    print('grid store in project folder: onegrid.xyz')

def icp_2forone(in1,in2):       #two pcds are fitted the transfomation is applyed on the third (one for fitting, two is transformed)
    pcd = o3d.io.read_point_cloud(in1)
    pcd2 = o3d.io.read_point_cloud(in2)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the subfolder
    subfolder_name = 'tools'
    subfolder_path = os.path.join(current_dir, subfolder_name)

    filename = "merged_cloud.pcd"

    file_path = os.path.join(subfolder_path, filename)

    pcd3 = o3d.io.read_point_cloud(file_path) # ideal plane
    pcd.paint_uniform_color([1, 0, 0])
    pcd3.paint_uniform_color([0, 0, 1])

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([pcd, pcd3])
    #, mesh_frame
    source = copy.deepcopy(pcd)
    target = copy.deepcopy(pcd3)

    print('calc ipc:')
    threshold = 200  # Maximum correspondence points-pair distance
    trans_init = np.asarray([[1, 0.0, 0.0, 0.0],
                             [0.0, 1, 0.0, 0.0],
                             [0.0, 0.0, 1, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])

    reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
                                                          o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                          o3d.pipelines.registration.ICPConvergenceCriteria(relative_rmse=1e-6,
                                                              max_iteration=10 ^ 9))  # max(max_iterationbs=10^9)
    pcd2.transform(reg_p2p.transformation)


    print(reg_p2p)
    print('_________________')
    print(reg_p2p.transformation)
    source.transform(reg_p2p.transformation)
    o3d.visualization.draw_geometries([source, target, mesh_frame])

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the subfolder
    subfolder_name = 'temp'
    subfolder_path = os.path.join(current_dir, subfolder_name)

    filename = "fitonplane.xyz"

    file_path = os.path.join(subfolder_path, filename)

    o3d.io.write_point_cloud(file_path, pcd2)
    print(r"stored:\temp\fitonplane.xyz")
    return reg_p2p.transformation

def plotheat(input):
    import mplcursors  # Import the mplcursors library

    data=np.loadtxt(input)
    data=np.transpose(data)

    di=data[2]

    sorted_arr = np.sort(di)[::-1]
    max1 = []
    for i in range(0, int(len(sorted_arr) * 0.01)):
        max1.append(sorted_arr[i])

    cap = (sum(max1) / len(max1))  # a limit for some exeptionally high values (to one percent)

    di_cap = np.where(di > cap, np.mean(di), di)

    #plt.scatter(data[0], data[1], c=(di_cap), cmap="plasma", s=0.1)
    #plt.gca().set_aspect('equal')
    #plt.colorbar()

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the subfolder
    subfolder_name = 'tools'
    subfolder_path = os.path.join(current_dir, subfolder_name)

    filename = "that-heatmap-so.jpg"

    file_path = os.path.join(subfolder_path, filename)

    pcd3 = o3d.io.read_point_cloud(file_path)  # ideal plane

    image_path = file_path
    image = plt.imread(image_path)
    plt.imshow(image)
    plt.axis('off')  # Turn off axis labels
    plt.draw()

    # Pause to display the image for a few seconds
    plt.pause(1)  # duration is at least waht is in brakets plus loading of image itself
    plt.clf()

    scatter = plt.scatter(data[0], data[1], c=di,  s=0.1, cmap='viridis')

    plt.xlabel('x in µm')
    plt.ylabel('y in µm')
    plt.gca().set_aspect('equal')
    #plt.axis('off')
    #plt.savefig('foo.png', transparent=True)
    #plt.colorbar()
    cbar = plt.colorbar(scatter)


    # Set the label for the colorbar
    cbar.set_label('z in µm')



    cursor = mplcursors.cursor(scatter,multiple=True, hover=False, bindings={"toggle_visible": "h", "toggle_enabled": "e"})  # Enable highlighting
    cursor.connect("add", lambda sel: sel.annotation.set_text(f"Z=: {sel.artist.get_array().round(3)[sel.index]}"))

    plt.show()

def cyrcle_mask(path,x_pos,y_pos,r):

    print('input: x,y,r:')
    print(x_pos,y_pos,)


    def create_circle_mask(x, y, x_center, y_center, radius):
        # Calculate distances from each point to the center
        distances = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)

        # Create a boolean mask for points within the circle
        mask = distances <= radius

        return mask

    # Example data
    data=load_txt_return_array(path)
    data=np.transpose(data)
    x,y,z=data[0],data[1],data[2]

    # Define circle parameters
    x_center = float(x_pos)
    y_center = float(y_pos)
    radius = float(r)

    # Create circular mask
    mask = create_circle_mask(x, y, x_center, y_center, radius)

    # Select only the points inside the circle
    x_in = x[mask]
    y_in = y[mask]
    z_in = z[mask]

    plt.scatter(x, y, z)
    plt.scatter(x_in,y_in,y_in, color='red')
    plt.gca().set_aspect('equal')
    plt.show()

    # Plot or use the selected points as needed
    plot_heat(x_in, y_in, z_in)

    q = input('Save on desktop? (y/n): ')
    if q.lower() == 'y':
        a = np.column_stack((x_in, y_in, z_in))
        np.savetxt(r"C:\Users\itotrapp\Desktop\masked.txt", a)
        print("ok, saved as masked.txt")

    return x_in, y_in, z_in
