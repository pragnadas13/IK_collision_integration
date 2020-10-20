import pandas as pd
import numpy as np
import math
import pickle
from sympy import *
import sys
sys.path.insert(0, "/home/pragna/kinpy")
import kinpy as kp

from scipy.spatial.distance import cdist, euclidean

def mid_point(x,y):
    x_mid = 0.5 * (x[0]+y[0])
    y_mid = 0.5 * (x[1] + y[1])
    z_mid = 0.5 * (x[2] + y[2])
    return [x_mid, y_mid, z_mid]
def RotX(angle):
    cti =cos(angle)
    sti =sin(angle)
    res = Matrix([[1, 0, 0, 0], [0 , cti,  -sti,   0],[ 0, sti, cti, 0],[0, 0, 0, 1]])
    return res

def RotY(angle):
    cti= cos(angle)
    sti=sin(angle)
    res = Matrix([[  cti, 0, sti, 0],[ 0, 1, 0,0],[ -sti, 0, cti, 0],[0, 0, 0, 1]])
    return res

def RotZ(angle):
    cti = cos(angle);
    sti = sin(angle);
    res = Matrix([[cti, - sti, 0, 0],[ sti, cti, 0, 0],[ 0, 0, 1, 0],[ 0, 0, 0, 1]])
    return res

def TranX(t):
    res = Matrix([[ 1, 0, 0, t], [ 0, 1, 0, 0],[ 0, 0, 1, 0],[0,0,0,1]])
    return res

def TranY(t):
    res = Matrix([[ 1, 0, 0, 0], [ 0, 1, 0, t],[ 0, 0, 1, 0],[0,0,0,1]])
    return res

def TranZ(t):
    res = Matrix([[ 1, 0, 0, 0], [ 0, 1, 0, 0],[ 0, 0, 1, t],[0,0,0,1]])
    return res


def Skew(v):
    res = Matrix([[0, - v[2], v[1]], [v[2], 0, - v[0]], [- v[1], v[0], 0]])
    return res

def Jacobian(endEffector, t0, t1, t2, t3, t4, t5, t6, z0, z1, z2, z3, z4, z5, z6):

    res = zeros(6, 7)

    res[0: 3, 0] = (Transpose(Skew(endEffector - t0))) * z0
    res[0: 3, 1] = (Transpose(Skew(endEffector - t1))) * z1
    res[0: 3, 2] = (Transpose(Skew(endEffector - t2))) * z2
    res[0: 3, 3] = (Transpose(Skew(endEffector - t3))) * z3
    res[0: 3, 4] = (Transpose(Skew(endEffector - t4))) * z4
    res[0: 3, 5] = (Transpose(Skew(endEffector - t5))) * z5
    res[0: 3, 6] = (Transpose(Skew(endEffector - t6))) * z6
#
    res[3: 6, 0] = z0;
    res[3: 6, 1] = z1;
    res[3: 6, 2] = z2;
    res[3: 6, 3] = z3;
    res[3: 6, 4] = z4;
    res[3: 6, 5] = z5;
    res[3: 6, 6] = z6;
    return res


def Gen3DirKin(q1, q2, q3, q4, q5, q6, q7):

    d1 = -128.4
    theta1 = 0
    a1 = 0
    alpha1 = pi / 2
    b1 = -5.4
    beta1 = 0

    d2 = -6.4;
    theta2 = 0
    a2 = 0
    alpha2 = -pi / 2
    b2 = -210.4
    beta2 = 0

    d3 = -210.4
    theta3 = 0
    a3 = 0
    alpha3 = pi / 2
    b3 = -6.4
    beta3 = 0

    d4 = -6.4
    theta4 = 0
    a4 = 0
    alpha4 = -pi / 2
    b4 = -208.4
    beta4 = 0

    d5 = -105.9
    theta5 = 0
    a5 = 0
    alpha5 = pi / 2
    b5 = -0
    beta5 = 0

    d6 = 0
    theta6 = 0
    a6 = 0
    alpha6 = -pi / 2
    b6 = -105.9
    beta6 = 0

    d7 = -61.5
    theta7 = 0
    a7 = 0
    alpha7 = pi
    b7 = 0
    beta7 = 0

    m_LTR = TranZ(156.4) * RotZ(0) * TranX(0) * RotX(pi) * TranZ(0) * RotZ(0)
    m_0T1 = TranZ(d1) * RotZ(q1) * TranX(a1) * RotX(alpha1) * TranZ(b1) * RotZ(beta1)
    m_1T2 = TranZ(d2) * RotZ(q2) * TranX(a2) * RotX(alpha2) * TranZ(b2) * RotZ(beta2)
    m_2T3 = TranZ(d3) * RotZ(q3) * TranX(a3) * RotX(alpha3) * TranZ(b3) * RotZ(beta3)
    m_3T4 = TranZ(d4) * RotZ(q4) * TranX(a4) * RotX(alpha4) * TranZ(b4) * RotZ(beta4)
    m_4T5 = TranZ(d5) * RotZ(q5) * TranX(a5) * RotX(alpha5) * TranZ(b5) * RotZ(beta5)
    m_5T6 = TranZ(d6) * RotZ(q6) * TranX(a6) * RotX(alpha6) * TranZ(b6) * RotZ(beta6)
    m_6T7 = TranZ(d7) * RotZ(q7) * TranX(a7) * RotX(alpha7) * TranZ(b7) * RotZ(beta7)

    m_0T7Now = m_LTR * m_0T1 * m_1T2 * m_2T3 * m_3T4 * m_4T5 * m_5T6 * m_6T7

    m_0T1Now = m_LTR * m_0T1
    rot_0R1Now = m_0T1Now[0:3, 0: 3]
    v_0p1Now = m_0T1Now[0:3, 3]
    v_0z1Now = m_0T1Now[0:3, 2]

    m_0T2Now = m_0T1Now * m_1T2
    rot_0R2Now = m_0T2Now[0:3, 0: 3]
    v_0p2Now = m_0T2Now[0:3, 3]
    v_0z2Now = m_0T2Now[0:3, 2]

    m_0T3Now = m_0T2Now * m_2T3
    rot_0R3Now = m_0T3Now[0:3, 0: 3]
    v_0p3Now = m_0T3Now[0:3, 3]
    v_0z3Now = m_0T3Now[0:3, 2]

    m_0T4Now = m_0T3Now * m_3T4
    rot_0R4Now = m_0T4Now[0:3, 0: 3]
    v_0p4Now = m_0T4Now[0:3, 3]
    v_0z4Now = m_0T4Now[0:3, 2]

    m_0T5Now = m_0T4Now * m_4T5
    rot_0R5Now = m_0T5Now[0:3, 0: 3]
    v_0p5Now = m_0T5Now[0:3, 3]
    v_0z5Now = m_0T5Now[0:3, 2]

    m_0T6Now = m_0T5Now * m_5T6
    rot_0R6Now = m_0T6Now[0:3, 0: 3]
    v_0p6Now = m_0T6Now[0:3, 3]
    v_0z6Now = m_0T6Now[0:3, 2]

    rot_0R7Now = m_0T7Now[0:3, 0: 3]
    v_0p7Now = m_0T7Now[0:3, 3]
    v_0z7Now = m_0T7Now[0:3, 2]

    rot_BaseR0Now = m_LTR[0:3, 0: 3]
    v_Basep0Now = m_LTR[0:3, 3]
    v_Basez0Now = m_LTR[0:3, 2]

    # Jaco67 = Jacobian(v_0p7Now, v_Basep0Now, v_0p1Now, v_0p2Now, v_0p3Now, v_0p4Now, v_0p5Now, v_0p6Now, v_Basez0Now,
    #                   v_0z1Now, v_0z2Now, v_0z3Now, v_0z4Now, v_0z5Now, v_0z6Now)


    #return m_0T7Now,Jaco67

    return v_0p1Now, v_0p2Now, v_0p3Now, v_0p4Now, v_0p5Now, v_0p6Now, v_0p7Now

def get_input_joint_pos(q_sa):
    q1 = q_sa[0]
    q2 = q_sa[1]
    q3 = q_sa[2]
    q4 = q_sa[3]
    q5 = q_sa[4]
    q6 = q_sa[5]
    q7 = q_sa[6]
    joint_pos = Gen3DirKin(q1, q2, q3, q4, q5, q6, q7)
    joint_pos_np = np.zeros((len(joint_pos), 3))
    for i, x in enumerate(joint_pos):
        joint_pos_np[i] = np.array(x.tolist()).astype(np.float64).squeeze()

    return joint_pos_np

def cal_collision(centroid_file, joint_pos_np):
    lineNo = 0
    print("centroid filename in cal collision", centroid_file)
    file = centroid_file #centroids after region_growing
    fw = open(file, 'r')
    lines = fw.readlines()
    centroids = []
    span = []
    radius = []
    fw1 = open("/home/pragna/Documents/Documents/collision_final/obstacleavoidance_IK/Selected_Pcd_for_LR.txt", 'a')
    fw2 = open("/home/pragna/Documents/Documents/collision_final/obstacleavoidance_IK/Selected_Pcd_for_RR.txt", 'a')
    for l in lines:
        lineNo = lineNo + 1
        split_lines = l.split(',')
        filename = split_lines[0]

        centroids =[]
        centroids.append(float(split_lines[1]))
        centroids.append(float(split_lines[2]))
        centroids.append(float(split_lines[3]))
        centroids = np.asarray(centroids).reshape((-1, 3))
        translation = np.asarray([-0.80, 1.25, 1.80])
        translation = translation[np.newaxis, :]
        centroids_world = centroids.astype(np.float) - translation.repeat(centroids.shape[0], axis=0)
        translation_lr = np.asarray([-0.8777, -0.1488, 1.191])
        translation_rr = np.asarray([-0.42882, -0.1488, 1.191])
        centroids_lr = centroids_world + translation_lr
        centroids_rr = centroids_world + translation_rr

        span.append(float(split_lines[4]))
        span.append(float(split_lines[5]))
        span.append(float(split_lines[6]))
        radius.append(float(split_lines[7]) / 2.0)
        raduisObj = float(split_lines[7]) / 2.0

        mu, sigma = 0, 1  # mean and standard deviation
        t_x = np.random.normal(mu, sigma, 10)

        # mu, sigma = 0, 1  # mean and standard deviation
        t_y = np.random.normal(mu, sigma, 10)

        # mu, sigma = 0, 1  # mean and standard deviation
        t_z = np.random.normal(mu, sigma, 10)
        translation_random = [t_x, t_y, t_z]
        translated_centroids_rr = []
        translated_centroids_lr = []
        for i, x in enumerate(centroids_rr):  # why centroids_lr is not having the same translation?
            translated_centroids_rr.extend(np.array(translation_random).T + x)
        translated_centroids_rr = np.array(translated_centroids_rr)
        for i, x in enumerate(centroids_lr):  # why centroids_lr is not having the same translation?
            translated_centroids_lr.extend(np.array(translation_random).T + x)
        translated_centroids_lr = np.array(translated_centroids_lr)

        midpoints = np.zeros((len(joint_pos_np) - 1, 3))
        for i in range(len(joint_pos_np) - 1):
            midpoints[i] = np.array(mid_point(joint_pos_np[i], joint_pos_np[i + 1]))
        midpoints = midpoints / 1000.0
        print(midpoints.shape)
        # print(self.Centroid_dict.shape)
        # scaling the positions as if glovebox has shrinked in size
        print(np.array(centroids).shape)
        # print(np.array(midpoints).shape)
        row_no_tc_lr = -1
        for tc_lr in translated_centroids_lr:
            row_no_tc_lr = row_no_tc_lr + 1
            # print(tc_lr)
            eucleadeanDist_lr_midPts = cdist(midpoints, np.array(tc_lr).reshape(1, 3))
            eucleadeanDist_lr = cdist(joint_pos_np/1000.0, np.array(tc_lr).reshape(1, 3))
            print(eucleadeanDist_lr_midPts)
            print(eucleadeanDist_lr)
            if eucleadeanDist_lr[4] > raduisObj and eucleadeanDist_lr_midPts[3] > raduisObj and eucleadeanDist_lr_midPts[2] > raduisObj:
                collisionFlag_lr = "F"
            elif eucleadeanDist_lr[4] <= raduisObj:
                collisionFlag_lr = "T"
            elif eucleadeanDist_lr_midPts[3] <= raduisObj:
                collisionFlag_lr = "T"
            elif eucleadeanDist_lr_midPts[2] <= raduisObj:
                collisionFlag_lr = "T"
                print("collisionFlag", collisionFlag_lr)
            if collisionFlag_lr == "T":

                fw1.writelines(file + " "+filename + " "+ str(lineNo) + str(row_no_tc_lr) + " " + str(tc_lr)+"\n")

        row_no_tc_rr = -1
        for tc_rr in translated_centroids_rr:
            row_no_tc_rr = row_no_tc_rr +1
            eucleadeanDist_rr_midPts = cdist(midpoints, np.array(tc_rr).reshape(1, 3))
            eucleadeanDist_rr = cdist(joint_pos_np/1000.0, np.array(tc_rr).reshape(1, 3))
            print(eucleadeanDist_rr)
            print(eucleadeanDist_rr_midPts)
            if eucleadeanDist_rr[4] > raduisObj and eucleadeanDist_rr_midPts[2] > raduisObj and eucleadeanDist_rr_midPts[3] > raduisObj:
                collisionFlag_rr = "F"
            elif eucleadeanDist_rr[4] <= raduisObj:
                collisionFlag_rr = "T"
            elif eucleadeanDist_rr_midPts[2] <= raduisObj:
                collisionFlag_rr = "T"
            elif eucleadeanDist_rr_midPts[3] <= raduisObj:
                collisionFlag_rr = "T"
            print("collisionFlag", collisionFlag_rr)
            if collisionFlag_rr == "T":

                fw2.writelines(file + " "+filename + " "+ str(lineNo) + str(row_no_tc_rr) + " " + str(tc_rr) + "\n")
                # fw2.close()

    fw1.close()
    fw2.close()
def find_pos_collision(centroid_file, input_config):
    joint_pos_np = get_input_joint_pos(input_config)
    cal_collision(centroid_file=centroid_file, joint_pos_np=joint_pos_np)


