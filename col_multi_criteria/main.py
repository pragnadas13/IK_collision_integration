import sys
sys.path.insert(0, '/home/pragna/Documents/Documents/collision/collision_model/')
sys.path.insert(0, '/home/pragna/Documents/Documents/collision/collision_model/main/')
# sys.path.insert(0, '/home/pragna/Documents/Documents/collision/collision_model/main/')
import glob

from input_jt_pos import find_pos_collision
# from segmentation.region_grow import region_growing
from main_nn import main_nn
#if __name__ == "__main__":
def collision_gradient(q_input, centroid, robot):
    ### get_centroids_Octree is a function to generate the octree and produce the object centroids.
    # It takes the current scene as pcd and produce centroid in .dat files
    # A folder is involved: the pcd files are in segmented_pcd/recorded_pcd_fg inside home
    # A .dat file is created for each pcd inside segmented_pcd/recorded_pcd_fg, since we do not need to train here, so a single consolidated centroid.dat file is not required.
    # This individual .dat files contains all the centroids of the individual pcds
    # read all pcd files one by one

    # find_pos_collision(centroid_file=centroid_file, input_config=q_input)
    # print("centroid_main.py",centroid)
    jac_input = main_nn(centroid=centroid, input_config=q_input, robot=robot)

    return jac_input
