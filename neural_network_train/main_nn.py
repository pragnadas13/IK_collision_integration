import sys
sys.path.insert(0, '/home/pragna/Documents/Documents/collision/collision_model/')
sys.path.insert(0, '/home/pragna/Documents/Documents/collision/collision_model/main/')
from neural_network_train.data_set import Collision_Dataset
from neural_network_train.neural_ntw_model import DetectCost
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter

def train(centroid_file, model_name):
    logging.basicConfig(filename='myapp.log', level=logging.INFO)
    logger = logging.getLogger('Collision-detection-Experiment::train')

    logger.info('--- Running Collision Detection Training ---')
    writer = SummaryWriter()
    batch_size = 8
    myNN = DetectCost()
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(params=myNN.parameters(), lr=0.0001)
    colData = Collision_Dataset(centroid_file=centroid_file)
    iteration =0
    for x in range(colData.config_sz):
        colData.create_dist(cnfg_set_index=x)

        train_loader = DataLoader(colData,
                                  batch_size=batch_size, shuffle=True,
                                  num_workers=8)### eta torch.util
        # val_loader = DataLoader(colData.)

        optimizer.zero_grad()
        for batch_id, samples in enumerate(train_loader):
            input_feat1, input_feat2, target = samples
            # print(input_feat[0])
            # print(target[0])

            input_feat1 = torch.autograd.Variable(input_feat1)
            input_feat2 = torch.autograd.Variable(input_feat2)
            target = torch.autograd.Variable(target.unsqueeze(1))
            inference = myNN(input_feat1, input_feat2)
            # print(inference.shape)
            loss = criterion(inference, target)
            writer.add_scalar('Loss/train', loss.item(), iteration)
            iteration = iteration+1
            # print(loss)
            logger.info('batch Number:%d Training Loss = %f', batch_id, loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_id%100==0:
                colData.set_dset('val')
                val_loader = DataLoader(colData, batch_size=batch_size, num_workers=8)
                total_val_loss = []
                min_val_loss = 1.0
                for test_batch_id, test_samples in enumerate(val_loader):
                    input_feat1,input_feat2, target = test_samples
                    input_feat1 = torch.autograd.Variable(input_feat1)
                    input_feat2 = torch.autograd.Variable(input_feat2)
                    target = torch.autograd.Variable(target.unsqueeze(1))
                    inference = myNN(input_feat1,input_feat2)
                    val_loss = criterion(inference, target)
                    if val_loss < min_val_loss:
                        torch.save(myNN, model_name)
                    if val_loss <0.0001:
                        sys.exit()
                    total_val_loss.append(val_loss.item())
                total_val_loss = np.mean(np.array(total_val_loss))
                writer.add_scalar('Loss/test', total_val_loss, iteration)

                logger.info('batch Number:%d validation Loss = %f',batch_id,total_val_loss)
                colData.set_dset('train')

# if __name__ == "__main__":
def main_nn(robot):
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    if robot=='rr':
        centroid_file='/home/pragna/Documents/Documents/collision/collision_model/main/Region_TranslatedCentroid_Data_rr.pkl'
        model_name = 'octree_Best_model_rr.th'
    else:
        centroid_file = '/home/pragna/Documents/Documents/collision/collision_model/main/Region_TranslatedCentroid_Data_lr.pkl'
        model_name = 'octree_Best_model_lr.th'

    train(centroid_file=centroid_file, model_name=model_name)
    # trainingNumber = 100
    # colData = Collision_Dataset()
    # myNN = DetectCost()
    # #for i in range(trainingNumber):
    # colData.sumDist, colData.input_feats = colData.create_dist(trainingNumber)
    #
    # for i in range(trainingNumber):
    #     input, target = colData.__getitem__(i)
    #     myNN(input)


