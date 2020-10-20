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
from input_jt_pos import find_pos_collision
import os


# In[2]:


sns.set(style="whitegrid")
sns.set_palette('tab10')


# In[3]:


#get_ipython().run_line_magic('matplotlib', 'inline')


# ## Simulation Parameters

# In[4]:


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

if __name__ == "__main__":
    q_ls = np.zeros((len(t_vec), 7))
    q_ls[0, :] = q_initial
    no_pcd_files = 0

    centroid_files = glob.glob('/home/pragna/segment_pcd/recorded_pcd/*.dat')
    for files in centroid_files:
        for i, t in enumerate(t_vec[1:]):
            J = jacobian_spatial(q_ls[i, :])
            find_pos_collision(q_input=q_ls[i, :], centroid_file=files, i=i)
            qp_ls = la.pinv(J) @ desired_trajectory_vel[i, :]
            q_ls[i+1, :] = q_ls[i, :] + qp_ls * h










