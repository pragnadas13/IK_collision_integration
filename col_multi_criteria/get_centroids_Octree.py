import os

def get_centroids_Octree(pcd):

    cmd = '/home/pragna/pcl_region_growing/project_octree '+ pcd+' -dump'
    os.system(cmd)

    base = pcd
    suffix = '.dat'
    centroid_file = os.path.join(base + '.' + 'centroid_octree' + suffix)
     # = '/home/pragna/segment_pcd/recorded_pcd/510090000.pcd.centroid_octree.dat'
    return centroid_file

#if __name__ == "__main__":
    #get_centroids_Octree('/home/pragna/segment_pcd/recorded_pcd/510090000.pcd')