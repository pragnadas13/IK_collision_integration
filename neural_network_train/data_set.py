import torch
import pandas as pd
from torch.utils.data import Dataset
import pickle
import numpy as np

from scipy.spatial.distance import cdist

def mid_point(x,y):
    x_mid = 0.5 * (x[0]+y[0])
    y_mid = 0.5 * (x[1] + y[1])
    z_mid = 0.5 * (x[2] + y[2])
    return [x_mid, y_mid, z_mid]


class Collision_Dataset(Dataset):

    def __init__(self, cnfg_file='/home/pragna/Documents/Documents/collision/collision_model/main/TB7_np_Data.pkl',
                 centroid_file='/home/pragna/Documents/Documents/collision/TranslatedCentroid_Data.pkl'):
        self.cnfg_file = cnfg_file
        df = pd.read_csv(
            '/home/pragna/Documents/Documents/collision/collision_model/Fk_ale/KinovaWorkSpaceInGloveBox.csv')
        print(len(df))
        df = np.array(df[['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7']])
        print(df.shape)
        self.df = df

        self.centroid_file = centroid_file

        self.TB7_dict = []
        infile1 = open(self.cnfg_file, 'rb')
        while (1):
            try:
                # print("Batuuuuuuuuuuuuuuuuuuuuuuu")collision_model/main/
                self.TB7_dict.append(pickle.load(infile1))
            except EOFError:
                break

        infile1.close()
        self.TB7_dict = np.asarray(self.TB7_dict, dtype=np.float)
        infile2 = open(self.centroid_file, 'rb')
        Centroids = pickle.load(infile2)
        infile2.close()
        Centroids = np.asarray(Centroids, dtype=np.float)
        self.Centroid_dict = Centroids / 100.0  # scaling the positions as if objects has shrinked in size
        self.config_sz = len(self.df)
        self.train = {}
        self.val = {}
        self.dset = 'train' # default dset

    def set_dset(self, dset='train'):
        self.dset = dset
        # self.input_feats = []
        #
        # self.create_dist(cnfg_set)

    def create_dist(self, cnfg_set_index):

        cnfg = self.TB7_dict[cnfg_set_index]
        cnfg_angle = self.df[cnfg_set_index]
        midpoints = np.zeros((len(cnfg) - 1, 3))
        for i in range(len(cnfg) - 1):
            midpoints[i] = np.array(mid_point(cnfg[i], cnfg[i + 1]))
        midpoints = midpoints / 100.0
        print(midpoints.shape)
        print(self.Centroid_dict.shape)
        # scaling the positions as if glovebox has shrinked in size
        eucleadeanDist = cdist(midpoints, self.Centroid_dict)
        print(midpoints.shape)
        print(self.Centroid_dict.shape)
        # eucleadeanDist_min = np.min(eucleadeanDist)
        # eucleadeanDist_max = np.max(eucleadeanDist)

        sumDist = np.sum(np.exp(-1 * eucleadeanDist), axis=0)
        # Centroid_dict = Centroid_dict[:, np.newaxis, :]
        x = np.repeat(cnfg_angle.reshape(-1, 7), self.Centroid_dict.shape[0], axis=0)
        #input_feats = np.concatenate((self.Centroid_dict, x), axis=1)
        # create random partition of train/validation
        assert self.Centroid_dict.shape[0] == sumDist.shape[0]
        len_dset = self.Centroid_dict.shape[0]
        train_samples = int(len_dset*0.7)
        indices = range(len_dset)
        indices = np.random.permutation(indices)
        indices_train = indices[:train_samples]
        indices_val = indices[train_samples:]
        input_feats_train_1 = self.Centroid_dict[indices_train]
        input_feats_train_2 = x[indices_train]
        input_feats_val_1 = self.Centroid_dict[indices_val]
        input_feats_val_2 = x[indices_val]
        target_train = sumDist[indices_train]
        target_val = sumDist[indices_val]
        self.train = {'input': (input_feats_train_1, input_feats_train_2), 'target': target_train}
        self.val = {'input': (input_feats_val_1, input_feats_val_2), 'target': target_val}

        # return sumDist, input_feats

    def __len__(self):
        if self.dset =='train':
            return len(self.train['input'])
        else:
            return len(self.val['input'])


    def __getitem__(self, index):
        if self.dset=='train':
            input1 = self.train['input'][0][index]
            input2 = self.train['input'][1][index]
            target = self.train['target'][index]
        if self.dset=='val':
            input1 = self.val['input'][0][index]
            input2 = self.val['input'][1][index]
            target = self.val['target'][index]
        # print(input.shape)
        # print(target.shape)
        input1 = input1.astype(np.float32)
        input2 = input2.astype(np.float32)
        target = target.astype(np.float32)
        # input = from_numpy(input)
        # target = torch.from_numpy(target)
        return input1,input2, target
