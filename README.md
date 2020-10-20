This repository has 3 folders 
1) neural_network_train is for training the network,
The dataset is available at: 
2) find_col_centroid is for finding the centroids which have possibility of collision. 
Collision possibility is checked with the 4th joint and 5th joint, the mid-point of 3rd, 4th, 5th link
3) col_multi_criteria is for generating the trajectories with 
      a) collision as an objective function 
      b) collision & singularity avoidance as 2 objective functions
 
Inside col_multi_criteria, spatial_mechanism.py has a function def read_centroid_from_gazebo(): which is commented now. 
This function is to be defined to get the centroid from gazebo. Once the centroid can be fed to this program from gazebo, lines 376-388 have to be omitted
(including the for loop) and obs_centroid = < centroid obtained from gazebo>. The centroid should be ndarray. 


      
 
