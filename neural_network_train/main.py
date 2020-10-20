########### Main module which holds and executes all the sub-modules####################
#######################################################################################

from process_vision_data import process
from functions import *

from model import sdm_lstm, regression, lagrange
from algorithm import optimise
from neural_ntw_model import Perceptron
from input_data import model

if __name__ == "__main__":
    dirName = '/home/pragna/Documents/PathPlanning/code/'
    fileName = 'sample_3d.ply'

    bunch_3d_pts = process(dirName, fileName)
    dist = distFunction(bunch_3d_pts)
    cost, samples = costFunc(dist) #develop a correspondence, like a dictionary or something, to associate the cost with each sample


