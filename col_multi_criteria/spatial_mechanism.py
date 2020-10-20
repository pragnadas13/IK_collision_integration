#!/usr/bin/env python
# coding: utf-8

# # Redundancy Resolution with Multiple Objective Functions &ndash; Spatial Mechanism

# In[1]:
import sys
# sys.path.insert(0, '/home/pragna/Documents/Documents/collision_final/obstacleavoidance_IK/')
# sys.path.insert(0, '/home/pragna/Documents/Documents/collision_final/obstacleavoidance_IK/main/')
from mechanism_library.Serial.Kinova import kinematics as spatial_mechanism
from main import collision_gradient as collision
from get_centroids_Octree import get_centroids_Octree
import numpy as np
import glob
import time
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm
from scipy.optimize import approx_fprime
import seaborn as sns
import os


# In[2]:


sns.set(style="whitegrid")
sns.set_palette('tab10')



# ## Simulation Parameters


# Simulation time step
h = 0.001

# Simulation time
t_initial = 0.0
t_final = 8.0
t_vec = np.arange(t_initial, t_final + h, h)

# Time scaling
A = np.array([[1.0, t_initial, t_initial**2, t_initial**3],
              [0.0, 1.0, 2.0 * t_initial, 3.0 * t_initial**2],
              [1.0, t_final, t_final**2, t_final**3],
              [0.0, 1.0, 2.0 * t_final, 3.0 * t_final**2]])
b = np.array([0.0, 0.0, 1.0, 0.0])

# coefficients of the cubic time scaling polynomial (s(t) = a[0] * t**3+ a[1] * t**2 + a[2] * t + a[3] * t)
a = (la.inv(A) @ b)[::-1]

# coefficients of the first derivative of the cubic time scaling polynomial
ap = np.polyder(a)

# coefficients of thfrom get_centroids_Octree import get_centroids_Octreee first derivative of the cubic time scaling polynomial
app = np.polyder(ap)

# Initial and final configuration of the manipulator
q_initial = np.array([0.0, -np.pi / 3, 0.0, -np.pi / 3, 0.0, -np.pi / 3, 0.0])

p_initial, R_initial = spatial_mechanism.forward_kinematics(q_initial)
p_final, R_final = spatial_mechanism.forward_kinematics(np.array([np.pi / 6, np.pi / 3, np.pi / 6, np.pi / 3, np.pi / 6, np.pi / 3, 0.0]))

# In homogeneous coordinates
T_initial = np.eye(4)
T_final = np.eye(4)

T_initial[0:3, 3] = p_initial
T_final[0:3, 3] = p_final

T_initial[0:3, 0:3] = R_initial.as_matrix()
T_final[0:3, 0:3] = R_final.as_matrix()

# Joint limits of the manipulator
q_min = np.array([np.finfo(np.float64).min / 10,
                  -126 * np.pi / 180.0,
                  np.finfo(np.float64).min / 10,
                  -147 * np.pi / 180.0,
                  np.finfo(np.float64).min / 10,
                  -117 * np.pi / 180.0,
                  np.finfo(np.float64).min / 10])
q_max = np.array([np.finfo(np.float64).max / 10,
                  126 * np.pi / 180.0,
                  np.finfo(np.float64).max / 10,
                  147 * np.pi / 180.0,
                  np.finfo(np.float64).max / 10,
                  117 * np.pi / 180.0,
                  np.finfo(np.float64).max / 10])


# ## Functions

# Smooth straight trajectory

# In[5]:


def trajectory_generation(T_start, T_end, coeffs, t):
    s_values = np.polyval(coeffs, t)
    sp_values = np.polyval(np.polyder(coeffs), t)
    return np.array([T_start @ expm(logm(la.inv(T_start) @ T_end) * s) for s in s_values]),            np.array([T_start @ logm(la.inv(T_start) @ T_end) @ expm(logm(la.inv(T_start) @ T_end) * s) * sp for (s, sp) in zip(s_values, sp_values)])


# Manipulability metric and its gradient with respect to joint angles

# In[6]:


def manipulability(joint_angles, jacobian_function):
    return np.sqrt(la.det(jacobian_function(joint_angles) @ jacobian_function(joint_angles).transpose()))


# In[7]:


def manipulability_gradient(joint_angles, jacobian_function):
    return approx_fprime(joint_angles, manipulability, np.sqrt(np.finfo(float).eps), jacobian_function)


# Joint limit metric and its gradient with respect to joint angles

# In[8]:


def joint_limits(joint_angles, q_min, q_max):
    q_bar = (q_min + q_max) / 2
    
    return -1 / 6 * np.sum(((joint_angles - q_bar) / (q_max - q_min))**2)


# In[9]:


def joint_limits_gradient(joint_angles, q_min, q_max):
    return approx_fprime(joint_angles, joint_limits, np.sqrt(np.finfo(float).eps), q_min, q_max)


# Mechanism kinematics, Jacobian of the manipulator

# In[10]:


def jacobian_spatial(joint_angles):
#     return spatial_mechanism.jacobian(joint_angles)[0:3, :]
    return spatial_mechanism.jacobian(joint_angles)


# ### Trajectory generation

# In[11]:


# Desired trajectory
desired_trajectory_pos_tf, desired_trajectory_vel_tf = trajectory_generation(T_initial, T_final, a, t_vec)

desired_trajectory_pos = np.array([T[0:3, 3] for T in desired_trajectory_pos_tf])
desired_trajectory_rot = [T[0:3, 0:3] for T in desired_trajectory_pos_tf]

desired_trajectory_lin_vel = np.array([T[0:3, 3] for T in desired_trajectory_vel_tf])
desired_trajectory_ang_vel = np.array([[T[2, 1], T[0, 2], T[1, 0]] for T in desired_trajectory_vel_tf])

desired_trajectory_vel = np.concatenate((desired_trajectory_lin_vel, desired_trajectory_ang_vel), axis=1)


# Numerical calculation of the first derivative

# In[12]:


# T_dot = [(T_now - T_prev) / h for (T_now, T_prev) in zip(desired_trajectory_pos_tf[1:], desired_trajectory_pos_tf[:-1])]

# linear_velocity = np.zeros((len(t_vec), 3))
# angular_velocity = np.zeros((len(t_vec), 3))

# linear_velocity[1:] = np.array([Tp[0:3, 3] for Tp in T_dot])
# angular_velocity[1:] = np.array([[Tp[2, 1], Tp[0, 2], Tp[1, 0]] for Tp in T_dot])



# ### Singularity analysis

# ### Inverse kinematics with least-squares

# In[14]:
def plain_IK(rootDic, robot):
    q_ls = np.zeros((len(t_vec), 7))
    q_ls[0, :] = q_initial
    start_IK = time.time()
    for i, t in enumerate(t_vec[1:]):
        J = jacobian_spatial(q_ls[i, :])
        qp_ls = la.pinv(J) @ desired_trajectory_vel[i, :]
        q_ls[i+1, :] = q_ls[i, :] + qp_ls * h
    end_IK = time.time()
    diff_time_IK = end_IK - start_IK
    fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
    ax.plot(t_vec, q_ls[:, 0], label=r'$q_1$')
    ax.plot(t_vec, q_ls[:, 1], label=r'$q_2$')
    ax.plot(t_vec, q_ls[:, 2], label=r'$q_3$')
    ax.plot(t_vec, q_ls[:, 3], label=r'$q_4$')
    ax.plot(t_vec, q_ls[:, 4], label=r'$q_5$')
    ax.plot(t_vec, q_ls[:, 5], label=r'$q_6$')
    ax.plot(t_vec, q_ls[:, 6], label=r'$q_7$')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [rad]')
    ax.set_title('Inverse kinematics (least squares)')
    ax.legend(loc='ceq_inputnter right', bbox_to_anchor=(1.2, 0.5))
    fig.tight_layout()
    rootDic = rootDic
    base_file_name = 'Figure_q_ls'
    suffix1 = '.csv'
    csv_fileHandle = os.path.join(rootDic, base_file_name + '_' + robot + suffix1)
    np.savetxt(csv_fileHandle, q_ls, delimiter=",")
    suffix2 = '.jpg'
    figureHandle = os.path.join(rootDic, base_file_name + '_' + robot + suffix2)
    fig.savefig(figureHandle)

    return q_ls, diff_time_IK

def collision_avoidance_IK(lineNo, rootDic, obs_centroid, __iteration_IK__, robot):
    q_sa = np.zeros((len(t_vec), 7))
    q_sa[0, :] = q_initial
    k0 = 5
    start_IK = time.time()
    for i, t in enumerate(t_vec[1:]):
        J = jacobian_spatial(q_sa[i, :])
        print("obs_centroid.shape", obs_centroid.shape)
        w_der = collision(q_input=q_sa[i, :], centroid=obs_centroid, robot=robot)
        w_der = w_der.numpy()
        print("W_der", w_der)
        print("W_der.shape", w_der.shape)
        q0 = k0 * w_der
        qp_sa = la.pinv(J) @ desired_trajectory_vel[i, :] + (np.eye(7) - la.pinv(J) @ J) @ q0
        q_sa[i+1, :] = q_sa[i, :] + qp_sa * h

    end_IK = time.time()
    diff_time_IK = end_IK - start_IK
    rootDic = rootDic
    base_file_name = 'Figure_collision_'
    suffix1 = '.csv'
    csv_fileHandle = os.path.join(rootDic, base_file_name + '_' + str(lineNo) + '_' + str(__iteration_IK__) + '_' + robot + suffix1)
    np.savetxt(csv_fileHandle, q_sa, delimiter=",")
    fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
    ax.plot(t_vec, q_sa[:, 0], label=r'$q_1$')
    ax.plot(t_vec, q_sa[:, 1], label=r'$q_2$')
    ax.plot(t_vec, q_sa[:, 2], label=r'$q_3$')
    ax.plot(t_vec, q_sa[:, 3], label=r'$q_4$')
    ax.plot(t_vec, q_sa[:, 4], label=r'$q_5$')
    ax.plot(t_vec, q_sa[:, 5], label=r'$q_6$')
    ax.plot(t_vec, q_sa[:, 6], label=r'$q_7$')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [rad]')
    ax.set_title('Inverse kinematics with collision avoidance')
    ax.legend(loc='ceq_inputnter right', bbox_to_anchor=(1.2, 0.5))
    fig.tight_layout()
    suffix2 = '.jpg'
    figureHandle = os.path.join(rootDic, base_file_name + '_' + str(lineNo) + '_' + str(__iteration_IK__) + '_' + robot + suffix2)
    fig.savefig(figureHandle)

    return q_sa, diff_time_IK


def multi_criteria(lineNo, rootDic, obs_centroid, __iteration_IK__, robot):#
    k0 = 1
    alpha_vec = np.arange(0, 1.1, 0.1)
    q_pareto = []
    for alpha in alpha_vec:
        q_mc = np.zeros((len(t_vec), 7))
        q_mc[0, :] = q_initial
        start_IK = time.time()
        for i, t in enumerate(t_vec[1:]):
            J = jacobian_spatial(q_mc[i, :])
            w1_der = manipulability_gradient(q_mc[i, :], jacobian_spatial)# singularity avoidance
            # w2_der = joint_limits_gradient(q_mc[i, :], q_min, q_max)
            w3_der = collision(q_input=q_sa[i, :], centroid=obs_centroid, robot=robot)
            w3_der = w3_der.numpy()
            q0 = k0 * (alpha * w1_der + (1-alpha)*w3_der)
            qp_mc = la.pinv(J) @ desired_trajectory_vel[i, :] + (np.eye(7) - la.pinv(J) @ J) @ q0
            q_mc[i + 1, :] = q_mc[i, :] + qp_mc * h
        end_IK = time.time()
        diff_time_IK = end_IK - start_IK
        fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
        ax.plot(t_vec, q_mc[:, 0], label=r'$q_1$')
        ax.plot(t_vec, q_mc[:, 1], label=r'$q_2$')
        ax.plot(t_vec, q_mc[:, 2], label=r'$q_3$')
        ax.plot(t_vec, q_mc[:, 3], label=r'$q_4$')
        ax.plot(t_vec, q_mc[:, 4], label=r'$q_5$')
        ax.plot(t_vec, q_mc[:, 5], label=r'$q_6$')
        ax.plot(t_vec, q_mc[:, 6], label=r'$q_7$')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position [rad]')
        ax.set_title('Inverse kinematics with multi-criteria')
        ax.legend(loc='ceq_inputnter right', bbox_to_anchor=(1.2, 0.5))
        fig.tight_layout()
        base_file_name1 = 'Figure_q_mc_multi_criteria'
        suffix1 = '.csv'
        csv_fileHandle = os.path.join(rootDic, base_file_name1 + '_' + str(lineNo) + '_' + str(__iteration_IK__) + '_' + robot + ' ' + str(alpha) + suffix1)
        np.savetxt(csv_fileHandle, q_ls, delimiter=",")
        suffix2 = '.jpg'
        figureHandle = os.path.join(rootDic, base_file_name1 + '_' + str(lineNo) + '_' + str(__iteration_IK__) + '_' + robot + ' ' + str(alpha) + suffix2)
        fig.savefig(figureHandle)

        q_pareto.append(q_mc)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * 6.472135955, 4), dpi=96)
    for q, alpha in zip(q_pareto, alpha_vec):
        manipulability_cost = np.array([manipulability(qi, jacobian_spatial) for qi in q])
        joint_limit_cost = np.array([joint_limits(qi, q_min, q_max) for qi in q])
        color = np.random.rand(3, )
        if alpha == 0.0 or alpha == 1.0:
            lw = 3
        else:
            lw = 1
        ax1.plot(t_vec, manipulability_cost, label=(r'$\alpha=$' + '{0:.1f}'.format(alpha)), color=color, linewidth=lw)
        ax2.plot(t_vec, joint_limit_cost, label=(r'$\alpha=$' + '{0:.1f}'.format(alpha)), color=color, linewidth=lw)

    ax1.legend(ncol=2, prop={'size': 7})
    ax1.set_xlabel('Time [s]')
    ax2.set_xlabel('Time [s]')
    ax1.set_ylabel('Manipulability')
    ax2.set_ylabel('Joint Limits')
    fig.tight_layout()
    base_file_name2 = 'spatial_pareto'
    figureHandle = os.path.join(rootDic, base_file_name2 + str(__iteration_IK__) + '_' + robot + suffix2)
    fig.savefig(figureHandle)

    return q_pareto, diff_time_IK
def translate_centroid(robot, centroid):
    translation = np.asarray([-0.80, 1.25, 1.80])
    translation = translation[np.newaxis, :]

    centroids_world = centroid.astype(np.float) - translation
    print(centroids_world.shape)
    # .repeat(centroid.shape[0], axis=0)
    translation_lr = np.asarray([-0.8777, -0.1488, 1.191])
    translation_rr = np.asarray([-0.42882, -0.1488, 1.191])
    if robot == 'LR':
        centroid_lr = centroids_world + translation_lr

        return centroid_lr
    elif robot == 'RR':
        centroid_rr = centroids_world + translation_rr

        return centroid_rr
# def read_centroid_from_gazebo():


# read_centroid_from_gazebo function inputs the centroid of the obstacle from gazebo.
# This function has to be defined. It returns
# 2 output parameters:
# centroid = object centroid, 3X1 numpy array
# __which__robot__ = a string value, 'LR' for left robot, 'RR' for right robot

if __name__ == "__main__":
    rootDic = sys.argv[1]
    rep = sys.argv[2]
    __which__robot__ = 'LR'
    # centroid, __which__robot__, __at_time__, lineNo = read_centroid_from_gazebo()
    # print("centroid", centroid)
    # obs_centroid = translate_centroid(robot=__which__robot__, centroid=centroid)
    # print("obs_centroid", obs_centroid)

    time_data_plain = list()
    time_data_col = list()
    time_data_multi = list()
    x_time_data = list()
    q_ls, diff_time_IK_plain = plain_IK(rootDic, robot=__which__robot__)
    __iteration_IK__ = 0
    while __iteration_IK__ < int(rep):

        __iteration_IK__ = __iteration_IK__ +1
        x_time_data.append(__iteration_IK__)
        time_data_plain.append(diff_time_IK_plain)

        if __which__robot__ == 'RR':
            fw = open("Selected_Centroid_for_RR.txt", 'r')
        elif __which__robot__ == 'LR':
            fw = open("Selected_Centroid_for_LR.txt", 'r')

        lines = fw.readlines()
        lineNo = 0
        for l in lines:
            lineNo = lineNo + 1
            split_lines = l.split(':')
            which_link_jt = split_lines[1]
            __at_time__ = split_lines[2]
            str_centroid = split_lines[3]
            obs_centroid = np.array([float(x) for x in str_centroid[2:-2].split()])
            q_sa, diff_time_IK_col = collision_avoidance_IK(lineNo=lineNo, rootDic=rootDic, obs_centroid=obs_centroid, __iteration_IK__=__iteration_IK__, robot=__which__robot__)
            time_data_col.append(diff_time_IK_col)
            q_pareto, diff_time_IK_multi = multi_criteria(lineNo=lineNo, rootDic=rootDic, obs_centroid=obs_centroid,__iteration_IK__=__iteration_IK__, robot=__which__robot__)
            time_data_multi.append(diff_time_IK_multi)

        diff_ls_sa = q_sa - q_ls

        fig_diff_col, ax_diff_col = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
        ax_diff_col.plot(t_vec, diff_ls_sa[:, 0], label=r'$q_1$')
        ax_diff_col.plot(t_vec, diff_ls_sa[:, 1], label=r'$q_2$')
        ax_diff_col.plot(t_vec, diff_ls_sa[:, 2], label=r'$q_3$')
        ax_diff_col.plot(t_vec, diff_ls_sa[:, 3], label=r'$q_4$')
        ax_diff_col.plot(t_vec, diff_ls_sa[:, 4], label=r'$q_5$')
        ax_diff_col.plot(t_vec, diff_ls_sa[:, 5], label=r'$q_6$')
        ax_diff_col.plot(t_vec, diff_ls_sa[:, 6], label=r'$q_7$')

        ax_diff_col.set_xlabel('Time [s]')
        ax_diff_col.set_ylabel('Position [rad]')
        ax_diff_col.set_title('Difference with q_sa and q_ls')
        ax_diff_col.legend(loc='ceq_inputnter right', bbox_to_anchor=(1.2, 0.5))
        fig_diff_col.tight_layout()
        base_file_diff1 = "Diff_curve_q_sa_q_ls"
        suffix2 = '.jpg'
        fig_diff1 = os.path.join(rootDic, base_file_diff1 + str(__iteration_IK__) + '_' + __which__robot__ + suffix2)
        fig_diff_col.savefig(fig_diff1)

        for j in len(q_pareto):
            diff_ls_mc = q_pareto[j:] - q_ls
            diff_sa_mc = q_pareto[j:] - q_sa

            fig_diff_multi, ax_diff_multi = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
            ax_diff_multi.plot(t_vec, diff_ls_mc[:, 0], label=r'$q_1$')
            ax_diff_multi.plot(t_vec, diff_ls_mc[:, 1], label=r'$q_2$')
            ax_diff_multi.plot(t_vec, diff_ls_mc[:, 2], label=r'$q_3$')
            ax_diff_multi.plot(t_vec, diff_ls_mc[:, 3], label=r'$q_4$')
            ax_diff_multi.plot(t_vec, diff_ls_mc[:, 4], label=r'$q_5$')
            ax_diff_multi.plot(t_vec, diff_ls_mc[:, 5], label=r'$q_6$')
            ax_diff_multi.plot(t_vec, diff_ls_mc[:, 6], label=r'$q_7$')

            ax_diff_multi.set_xlabel('Time [s]')
            ax_diff_multi.set_ylabel('Position [rad]')
            ax_diff_multi.set_title('Difference with q_sa and q_ls')
            ax_diff_multi.legend(loc='ceq_inputnter right', bbox_to_anchor=(1.2, 0.5))
            fig_diff_multi.tight_layout()
            base_file_diff2 = "Diff_curve_q_mc_q_ls"
            suffix2 = '.jpg'
            fig_diff2 = os.path.join(rootDic, base_file_diff2 + str(__iteration_IK__) + '_' + str(j) + '_' + __which__robot__ + suffix2)
            fig_diff_col.savefig(fig_diff2)

            fig_diff_multi_col, ax_diff_multi_col = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
            ax_diff_multi_col.plot(t_vec, diff_sa_mc[:, 0], label=r'$q_1$')
            ax_diff_multi_col.plot(t_vec, diff_sa_mc[:, 1], label=r'$q_2$')
            ax_diff_multi_col.plot(t_vec, diff_sa_mc[:, 2], label=r'$q_3$')
            ax_diff_multi_col.plot(t_vec, diff_sa_mc[:, 3], label=r'$q_4$')
            ax_diff_multi_col.plot(t_vec, diff_sa_mc[:, 4], label=r'$q_5$')
            ax_diff_multi_col.plot(t_vec, diff_sa_mc[:, 5], label=r'$q_6$')
            ax_diff_multi_col.plot(t_vec, diff_sa_mc[:, 6], label=r'$q_7$')

            ax_diff_multi_col.set_xlabel('Time [s]')
            ax_diff_multi_col.set_ylabel('Position [rad]')
            ax_diff_multi_col.set_title('Difference with q_sa and q_ls')
            ax_diff_multi_col.legend(loc='ceq_inputnter right', bbox_to_anchor=(1.2, 0.5))
            fig_diff_multi_col.tight_layout()
            base_file_diff3 = "Diff_curve_q_mc_q_sa"
            suffix2 = '.jpg'
            fig_diff3 = os.path.join(rootDic, base_file_diff3 + str(__iteration_IK__) + '_' + __which__robot__ + suffix2)
            fig_diff_col.savefig(fig_diff3)

    base_file_name1 = 'Figure_time_IK_col'
    suffix = '.jpg'
    figureHandle_time_col = os.path.join(rootDic, base_file_name1 + '_' + __which__robot__ + suffix)

    fig_time1, ax_time1 = plt.subplots()
    ax_time1.plot(x_time_data, time_data_col, 'brown', lw=2)
    ax_time1.set_xlabel('Iteration Number')
    ax_time1.set_ylabel('Time [s]')
    ax_time1.set_title('Time taken to solve IK (collision avoidance)')
    fig_time1.tight_layout()
    fig_time1.savefig(figureHandle_time_col)

    base_file_name2 = 'Figure_time_IK_multi'
    figureHandle_time_multi = os.path.join(rootDic, base_file_name2 + '_' + __which__robot__ + suffix)
    fig_time2, ax_time2 = plt.subplots()
    ax_time2.plot(x_time_data, time_data_multi, 'brown', lw=2)
    ax_time2.set_xlabel('Iteration Number')
    ax_time2.set_ylabel('Time [s]')
    ax_time2.set_title('Time taken to solve IK (collision avoidance)')
    fig_time2.tight_layout()
    fig_time2.savefig(figureHandle_time_multi)


















