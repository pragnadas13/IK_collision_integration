import torch
import pickle
import numpy as np
from torch.autograd.functional import jacobian

def main_nn(centroid, input_config, robot):
    ### Each line in centroid file stores
    # a) the x, y, z co-ordinates of the centroid
    # the span of x, y, z
    # maximum of (x-span, y-span, z-span).. So there are 7 values
    ##############################################3
    #centroid_file = '/home/pragna/Documents/Documents/collision/collision_model/main/octree_TranslatedCentroid_Data_rr.pkl'

    batch_size = 8
    cost_list = []
    cost_collision = 0
    gt_cost_list = []
    elapsed_time_list_nn = []
    elapsed_time_list_man = []
    if robot=='RR':
        model_NN = torch.load('octree_Best_model_rr.th')
    elif robot=='LR':
        model_NN = torch.load('octree_Best_model_lr.th')
    sumSamples = 0

    #cnfg_file = '/home/pragna/Documents/Documents/collision/collision_model/main/TB7_np_Data.pkl'
    #TB7_dict = input_config
    # centroid_f = open(centroid_file, 'r')
    # text_lines = centroid_f.readlines()
    # centroid_f.close()
    # xyz =[]
    # for line in text_lines:
    #     lsplit = line.split(',')
    #     xyz_line = [float(item) for item in lsplit[1:4]]
    #     xyz.append(xyz_line)
    # xyz = np.array(xyz)/100.0
    # print(xyz.shape)
    print("Input config", input_config)
    # input_config = np.tile(input_config.reshape(-1, 1), (1, 10))
    # print("Input config shape", input_config.shape)
    print("centroid", centroid)
    # centroid = np.asarray([float(c) for c in centroid[2:-2].split()])
    input_feat2 = torch.from_numpy(centroid)
    input_feat1 = torch.from_numpy(input_config)
    # input_feat1 = input_feat1.unsqueeze(0).repeat(10, 1)
    # print("Input feat1", input_feat1)
    # print("Input feat shape", input_feat.shape)
    input_feat2 = torch.autograd.Variable(input_feat2.float())
    input_feat1 = torch.autograd.Variable(input_feat1.float())
    print("Input feat 1", input_feat1.shape)
    print("Input feat 2", input_feat2.shape)

    # inference = model_NN(input_feat1[0], input_feat2[0])
    #cost_collision = inference.item()
    jac_input = jacobian(model_NN, (input_feat1.unsqueeze(0), input_feat2.unsqueeze(0)))
    print("jac_input[0].shape", jac_input[0].shape)
    print("jac_input[1].shape", jac_input[1].shape)



    #cost_list.append(inference.item())

    return jac_input[0].squeeze()