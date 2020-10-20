import numpy as np
import open3d as o3d
import os
def process(dirName, fileName):
    bunch3d_array = np.zeros((2, 1, 2))
    obstacle_3d = np.zeros((2, 1, 2))
    pt_cloud_file = os.path.join(dirName, fileName)
#read .ply file
    pcd_load = o3d.io.read_point_cloud(pt_cloud_file)
#save it as a numpy array
    xyz_load = np.asarray(pcd_load.points) #shape is (226995, 3)
#are all points useful? if not, segregate them, how?

    #sample and get more numpy arrays, output will be A bunch of 2D arrays
    # each array wil have cols as x,y,z and each 2d array is a configuration, a bunch of such arrays will be bunch of configurations
    # output of this function is a 3D array of n-size where n is the number of samples of 2d array, which is bunch3d_array.. here is formed, it has to be modified to appropiate size
    # segregate the obstacle 3D too..the structure will be same
    return bunch3d_array, obstacle_3d










