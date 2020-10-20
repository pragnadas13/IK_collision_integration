########### Main module which holds and executes all the sub-modules####################
#######################################################################################
import pickle
import numpy as np
from scipy.spatial.distance import cdist

def mid_point(x,y):
    x_mid = 0.5 * (x[0]+y[0])
    y_mid = 0.5 * (x[1] + y[1])
    z_mid = 0.5 * (x[2] + y[2])
    return [x_mid, y_mid, z_mid]



# if __name__ == "__main__":
def prepare_nn():
    TB7_dict = []
    infile1 = open('/home/pragna/Documents/Documents/collision/collision_model/TB7_np_Data.pkl', 'rb')
    while (1):
        try:
            TB7_dict.append(pickle.load(infile1))
        except EOFError:
            break

    infile1.close()
    TB7_dict = np.asarray(TB7_dict, dtype=np.float)
    infile2 = open('/home/pragna/Documents/Documents/collision/collision_model/TranslatedCentroid_Data.pkl', 'rb')
    Centroid_dict = pickle.load(infile2)
    infile2.close()
    Centroid_dict = np.asarray(Centroid_dict, dtype=np.float)
    Centroid_dict = Centroid_dict/100.0  # scaling the positions as if objects has shrinked in size, why?
    for cnfg in TB7_dict:
        midpoints = np.zeros((len(cnfg) - 1, 3))
        for i in range(len(cnfg)-1):
            midpoints[i] = np.array(mid_point(cnfg[i], cnfg[i+1]))
        midpoints = midpoints/100.0  # scaling the positions as if glovebox has shrinked in size, why?
        eucleadeanDist = cdist(midpoints, Centroid_dict)
        eucleadeanDist_min = np.min(eucleadeanDist)
        eucleadeanDist_max = np.max(eucleadeanDist)

        sumDist = np.sum(np.exp(eucleadeanDist), axis=0)
        # Centroid_dict = Centroid_dict[:, np.newaxis, :]
        x = np.repeat(cnfg.reshape(-1, 21), 103770, axis=0)
        input_feats = np.concatenate((Centroid_dict, x), axis=1) #would not it be sumDist?


