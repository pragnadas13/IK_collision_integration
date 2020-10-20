import math as m
import numpy as np
import numpy.linalg as la
from scipy.spatial.transform import Rotation


def forward_kinematics(q):
    # Link lengths
    # TODO find a better way to change link lengths
    l1, l2, l3 = 0.25, 0.2, 0.1
    
    # End-effector angle
    th = q[0] + q[1] + q[2]
    R = Rotation.from_rotvec(th * np.array([0, 0, 1]))

    # End-effector position
    p = np.array([l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1]) \
                  + l3 * np.cos(q[0] + q[1] + q[2]),
                  l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1]) \
                  + l3 * np.sin(q[0] + q[1] + q[2])])

    # In homogeneous coordinates
    # T = np.zeros((4, 4))
    # T[0:3, 0:3] = R.as_matrix()
    # T[0:2, 3] = p
    
    return p, R


def inverse_kinematics(x):
    # Link lengths
    # TODO find a better way to change link lengths
    l1, l2, l3 = 0.25, 0.2, 0.1
    
    x_bar = x[0] - l3 * m.cos(x[2])
    y_bar = x[1] - l3 * m.sin(x[2])
    
    c2 = (x_bar * x_bar + y_bar * y_bar - l1 * l1 - l2 * l2) / 2 / l1 / l2
    c2 = max(min(c2, 1), -1)
    q2 = m.atan2(m.sqrt(1 - c2 * c2), c2)
    
    q1 = m.atan2((l1 + l2 * m.cos(q2)) * y_bar - l2 * m.sin(q2) * x_bar,
                 (l1 + l2 * m.cos(q2)) * x_bar + l2 * m.sin(q2) * y_bar)

    q3 = x[2] - q1 - q2
    
    return np.array([q1, q2, q3])


def jacobian(q):
    # Link lengths
    # TODO find a better way to change link lengths
    l1, l2, l3 = 0.25, 0.2, 0.1
    q1, q2, q3 = q[0], q[1], q[2]
    
    J = np.array([[-l1*m.sin(q1) - l2*m.sin(q1 + q2) - l3*m.sin(q1 + q2 + q3),
                   -l2 * m.sin(q1 + q2) - l3 * m.sin(q1 + q2 + q3),
                   -l3 * m.sin(q1 + q2 + q3)],
                  [l1 * m.cos(q1) + l2 * m.cos(q1 + q2) + l3 * m.cos(q1 + q2 + q3),
                   l2 * m.cos(q1 + q2) + l3 * m.cos(q1 + q2 + q3),
                   l3 * m.cos(q1 + q2 + q3)],
                  [1, 1, 1]])

    return J

if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

    # Test the kinematics functions
    q0 = np.zeros(3)
    x0, R0 = forward_kinematics(q0)
    q0_i = inverse_kinematics(np.array([x0[0], x0[1], la.norm(R0.as_rotvec())]))
    
    q1 = np.array([np.pi / 2, 0, 0])
    x1, R1 = forward_kinematics(q1)
    q1_i = inverse_kinematics(np.array([x1[0], x1[1], la.norm(R1.as_rotvec())]))
    
    q2 = np.array([0, 0, m.pi / 2])
    x2, R2 = forward_kinematics(q2)
    q2_i = inverse_kinematics(np.array([x2[0], x2[1], la.norm(R2.as_rotvec())]))
    
    q3 = np.array([0, m.pi / 2, 0])
    x3, R3 = forward_kinematics(q3)
    q3_i = inverse_kinematics(np.array([x3[0], x3[1], la.norm(R3.as_rotvec())]))
    
    q4 = np.array([m.pi / 6, m.pi / 2, m.pi/3])
    x4, R4 = forward_kinematics(q4)
    q4_i = inverse_kinematics(np.array([x4[0], x4[1], la.norm(R4.as_rotvec())]))

    print('---')
    print('Test point:', q0 * 180 / m.pi)
    print('Forward kinematics solution:', x0, la.norm(R0.as_rotvec()) * 180 / m.pi)
    print('Inverse kinematics solution:', q0_i * 180 / m.pi)
    print('---')
    print('Test point:', q1 * 180 / m.pi)
    print('Forward kinematics solution:', x1, la.norm(R1.as_rotvec()) * 180 / m.pi)
    print('Inverse kinematics solution:', q1_i * 180 / m.pi)
    print('---')
    print('Test point:', q2 * 180 / m.pi)
    print('Forward kinematics solution:', x2, la.norm(R2.as_rotvec()) * 180 / m.pi)
    print('Inverse kinematics solution:', q2_i * 180 / m.pi)
    print('---')
    print('Test point:', q3 * 180 / m.pi)
    print('Forward kinematics solution:', x3, la.norm(R3.as_rotvec()) * 180 / m.pi)
    print('Inverse kinematics solution:', q3_i * 180 / m.pi)
    print('---')
    print('Test point:', q4 * 180 / m.pi)
    print('Forward kinematics solution:', x4, la.norm(R4.as_rotvec()) * 180 / m.pi)
    print('Inverse kinematics solution:', q4_i * 180 / m.pi)
