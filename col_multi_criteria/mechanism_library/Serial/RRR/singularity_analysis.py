from kinematics import forward_kinematics, jacobian
import numpy as np
import numpy.linalg as la
import math as m
import matplotlib.pyplot as plt


if __name__ == '__main__':
    q1_vec = np.linspace(0, m.pi / 2, 201)
    q2_vec = np.linspace(-m.pi, m.pi, 201)
    q3_vec = np.linspace(-m.pi, m.pi, 201)

    singular_configurations = [(q1, q2, q3)  for q1 in q1_vec for q2 in q2_vec
                              for q3 in q3_vec
                              if any(la.svd(jacobian((q1, q2, q3)))[1] < 1e-6)]

    np.savetxt('singular_configurations.csv',
               singular_configurations, delimiter=',')

    singular_points = np.array([forward_kinematics(q)[0]
                                for q in singular_configurations])
    
    fig, ax = plt.subplots()
    ax.scatter(singular_points[:, 0], singular_points[:, 1], s=1)
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.grid()
    fig.savefig('singularities.png')
