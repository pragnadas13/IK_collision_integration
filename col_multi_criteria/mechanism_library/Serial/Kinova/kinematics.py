import math as m
import numpy as np
import numpy.linalg as la
from scipy.spatial.transform import Rotation


def forward_kinematics(joint_ang):
    # joint angles of the robot
    q1, q2, q3, q4, q5, q6, q7 = joint_ang

    x0 = np.sin(q1)
    x1 = np.cos(q3)
    x2 = x0 * x1
    x3 = np.cos(q1)
    x4 = np.sin(q2)
    x5 = x3 * x4
    x6 = np.cos(q4)
    x7 = x5 * x6
    x8 = np.cos(q2)
    x9 = np.sin(q3)
    x10 = x3 * x9
    x11 = x10 * x8
    x12 = np.sin(q4)
    x13 = x0 * x9
    x14 = x1 * x3
    x15 = -x13 + x14 * x8
    x16 = x12 * x15
    x17 = np.cos(q6)
    x18 = -x16 - x7
    x19 = x17 * x18
    x20 = np.sin(q6)
    x21 = np.sin(q5)
    x22 = x11 + x2
    x23 = np.cos(q5)
    x24 = -x12 * x5 + x15 * x6
    x25 = -x21 * x22 + x23 * x24
    x26 = x20 * x25
    x27 = x0 * x4
    x28 = x27 * x6
    x29 = x13 * x8
    x30 = -x10 - x2 * x8
    x31 = x12 * x30
    x32 = x28 - x31
    x33 = x17 * x32
    x34 = x14 - x29
    x35 = x12 * x27 + x30 * x6
    x36 = -x21 * x34 + x23 * x35
    x37 = x20 * x36
    x38 = x4 * x9
    x39 = x6 * x8
    x40 = x1 * x4
    x41 = x12 * x40
    x42 = -x39 + x41
    x43 = x17 * x42
    x44 = -x12 * x8 - x40 * x6
    x45 = x21 * x38 + x23 * x44
    x46 = x20 * x45
    x47 = np.sin(q7)
    x48 = x21 * x24 + x22 * x23
    x49 = np.cos(q7)
    x50 = x17 * x25 + x18 * x20
    x51 = x21 * x35 + x23 * x34
    x52 = x17 * x36 + x20 * x32
    x53 = x21 * x44 - x23 * x38
    x54 = x17 * x45 + x20 * x42

    # end-effector position
    end_eff_pos = np.array(
        [
            -0.0118 * x0
            - 0.0128 * x11
            + 0.3143 * x16
            - 0.2874 * x19
            - 0.0128 * x2
            + 0.2874 * x26
            + 0.4208 * x5
            + 0.3143 * x7,
            -0.0128 * x14
            - 0.4208 * x27
            - 0.3143 * x28
            + 0.0128 * x29
            - 0.0118 * x3
            + 0.3143 * x31
            - 0.2874 * x33
            + 0.2874 * x37,
            0.0128 * x38
            + 0.3143 * x39
            - 0.3143 * x41
            - 0.2874 * x43
            + 0.2874 * x46
            + 0.4208 * x8
            + 0.2848,
        ]
    )

    # end-effector orientation
    end_eff_rot = Rotation.from_matrix(
        np.array(
            [
                [-x47 * x48 + x49 * x50, x47 * x50 + x48 * x49, -x19 + x26],
                [-x47 * x51 + x49 * x52, x47 * x52 + x49 * x51, -x33 + x37],
                [-x47 * x53 + x49 * x54, x47 * x54 + x49 * x53, -x43 + x46],
            ]
        )
    )

    return end_eff_pos, end_eff_rot


def inverse_kinematics_velocity(joint_ang, end_eff_vel, z=np.zeros(7), k=0):
    J = jacobian(joint_ang)
    J_inv_damped = transpose(J) @ inv(J @ transpose(J) + k ** 2 * eye(6))

    return J_inv_damped @ end_eff_vel + (eye(7) - J_inv_damped @ J) @ z


def jacobian(joint_ang):
    # joint angles of the robot
    q1, q2, q3, q4, q5, q6, q7 = joint_ang

    x0 = np.cos(q1)
    x1 = np.sin(q1)
    x2 = np.sin(q2)
    x3 = 0.4208 * x2
    x4 = np.cos(q3)
    x5 = x0 * x4
    x6 = 0.0128 * x5
    x7 = np.cos(q4)
    x8 = x1 * x2
    x9 = x7 * x8
    x10 = np.cos(q2)
    x11 = np.sin(q3)
    x12 = x1 * x11
    x13 = x10 * x12
    x14 = np.sin(q4)
    x15 = x0 * x11
    x16 = 0.3143 * x15
    x17 = x1 * x4
    x18 = x10 * x17
    x19 = -x16 - 0.3143 * x18
    x20 = np.cos(q6)
    x21 = -0.2874 * x9
    x22 = x15 + x18
    x23 = x14 * x22
    x24 = np.sin(q6)
    x25 = x13 - x5
    x26 = np.sin(q5)
    x27 = 0.2874 * x26
    x28 = x14 * x2
    x29 = x1 * x28
    x30 = -x15 - x18
    x31 = x29 + x30 * x7
    x32 = np.cos(q5)
    x33 = 0.2874 * x32
    x34 = x31 * x33
    x35 = 0.4208 * x10
    x36 = 0.0128 * x15
    x37 = x10 * x7
    x38 = 0.3143 * x37
    x39 = 0.3143 * x5
    x40 = 0.2874 * x37
    x41 = 0.2874 * x28
    x42 = x2 * x27
    x43 = x10 * x14
    x44 = x2 * x7
    x45 = 0.0128 * x12
    x46 = 0.3143 * x17
    x47 = x10 * x15
    x48 = x17 + x47
    x49 = 0.2874 * x20
    x50 = x14 * x49
    x51 = x10 * x5
    x52 = x12 - x51
    x53 = x33 * (-x17 - x47)
    x54 = x0 * x28
    x55 = 0.3143 * x12
    x56 = 0.3143 * x51
    x57 = x52 * x7
    x58 = x0 * x2
    x59 = x58 * x7
    x60 = -x12 + x51
    x61 = x14 * x60
    x62 = -x59 - x61
    x63 = x24 * x33
    x64 = -x54 + x60 * x7
    x65 = x26 * x64
    x66 = 0.2874 * x59
    x67 = 0.2874 * x61
    x68 = x26 * x48
    x69 = 0.2874 * x68
    x70 = x32 * x64
    x71 = -x13 + x5
    x72 = x25 * x33
    x73 = x22 * x7
    x74 = x14 * x30
    x75 = x26 * x71
    x76 = 0.3143 * x44
    x77 = x10 * x11
    x78 = 0.3143 * x43
    x79 = 0.2874 * x44
    x80 = 0.2874 * x43
    x81 = x2 * x4
    x82 = x11 * x28
    x83 = x11 * x2
    x84 = x33 * x83
    x85 = x28 * x4
    x86 = -x37 + x85
    x87 = x4 * x44
    x88 = -x43 - x87
    x89 = x23 + x9
    x90 = x29 - x73
    x91 = x43 + x87

    return np.array(
        [
            [
                -0.0118 * x0
                - x1 * x3
                + 0.0128 * x13
                + x14 * x19
                + x20 * (x21 - 0.2874 * x23)
                + x24 * (x25 * x27 + x34)
                - x6
                - 0.3143 * x9,
                x0 * x35
                + x0 * x38
                + x2 * x36
                + x20 * (x0 * x40 - x41 * x5)
                + x24 * (x15 * x42 + x33 * (-x0 * x43 - x44 * x5))
                - x28 * x39,
                -x10 * x6
                + x14 * (-x10 * x16 - x46)
                + x24 * (x27 * x52 + x53 * x7)
                + x45
                - x48 * x50,
                x20 * (-0.2874 * x54 - 0.2874 * x57)
                - 0.3143 * x54
                + x62 * x63
                + x7 * (-x55 + x56),
                x24 * (x53 - 0.2874 * x65),
                x20 * (-x69 + 0.2874 * x70) - x24 * (x66 + x67),
                0,
            ],
            [
                -x0 * x3
                + 0.0118 * x1
                + x14 * (x55 - x56)
                + 0.0128 * x17
                + x20 * (-x66 - x67)
                + x24 * (x33 * (x54 + x57) + x69)
                + 0.0128 * x47
                - 0.3143 * x59,
                -x1 * x35
                - x1 * x38
                - x2 * x45
                + x20 * (-x1 * x40 + x17 * x41)
                + x24 * (-x12 * x42 + x33 * (x1 * x43 + x17 * x44))
                + x28 * x46,
                x14 * (0.3143 * x13 - x39)
                + 0.0128 * x18
                + x24 * (x22 * x27 + x7 * x72)
                + x36
                - x50 * x71,
                x19 * x7
                + x20 * (0.2874 * x29 - 0.2874 * x73)
                + 0.3143 * x29
                + x63 * (-x74 + x9),
                x24 * (-x27 * x31 + x72),
                x20 * (x34 - 0.2874 * x75) - x24 * (x21 + 0.2874 * x74),
                0,
            ],
            [
                0,
                x20 * (-x4 * x80 - x79)
                + x24 * (x27 * x77 + x33 * (x28 - x37 * x4))
                - x3
                - x4 * x78
                - x76
                + 0.0128 * x77,
                x24 * (x27 * x81 + x7 * x84) + x49 * x82 + 0.0128 * x81 + 0.3143 * x82,
                x20 * (-x4 * x79 - x80) - x4 * x76 + x63 * x86 - x78,
                x24 * (-x27 * x88 + x84),
                x20 * (x27 * x83 + x33 * x88) - x24 * (x40 - 0.2874 * x85),
                0,
            ],
            [0, x1, -x58, x48, x62, x32 * x48 + x65, x20 * x62 - x24 * (-x68 + x70)],
            [
                0,
                x0,
                x8,
                x71,
                x89,
                x26 * x90 + x32 * x71,
                x20 * x89 - x24 * (x32 * x90 - x75),
            ],
            [
                -1,
                0,
                -x10,
                -x83,
                x86,
                -x26 * x91 - x32 * x83,
                x20 * x86 - x24 * (x26 * x83 - x32 * x91),
            ],
        ]
    )


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": lambda x: "{0:0.4f}".format(x)})

    # Test the kinematics functions
    q0 = np.zeros(7)
    x0, R0 = forward_kinematics(q0)
    #    q0_i = inverse_kinematics(np.array([x0[0], x0[1], la.norm(R0.as_rotvec())]))

    q1 = np.array([np.pi / 2, 0, 0, 0, 0, 0, 0])
    x1, R1 = forward_kinematics(q1)
    #    q1_i = inverse_kinematics(np.array([x1[0], x1[1], la.norm(R1.as_rotvec())]))

    q2 = np.array([0, m.pi / 2, 0, 0, 0, 0, 0])
    x2, R2 = forward_kinematics(q2)
    #    q2_i = inverse_kinematics(np.array([x2[0], x2[1], la.norm(R2.as_rotvec())]))

    q3 = np.array([0, m.pi / 2, 0, m.pi / 2, 0, m.pi / 2, 0])
    x3, R3 = forward_kinematics(q3)
    #    q3_i = inverse_kinematics(np.array([x3[0], x3[1], la.norm(R3.as_rotvec())]))

    q4 = np.array([m.pi / 6, m.pi / 2, m.pi / 3, 0, 0, 0, 0])
    x4, R4 = forward_kinematics(q4)
    #    q4_i = inverse_kinematics(np.array([x4[0], x4[1], la.norm(R4.as_rotvec())]))

    print("---")
    print("Test point 0:", q0 * 180 / m.pi)
    print("Forward kinematics solution:")
    print("End-effector position", x0)
    print("End-effector orientation")
    print(
        "\t Axis",
        R0.as_rotvec() / la.norm(R0.as_rotvec())
        if la.norm(R0.as_rotvec())
        else R0.as_rotvec(),
    )
    print("\t Angle", la.norm(R0.as_rotvec()) * 180 / m.pi)
    #    print('Inverse kinematics solution:', q0_i * 180 / m.pi)
    print("---")
    print("Test point 1:", q1 * 180 / m.pi)
    print("Forward kinematics solution:")
    print("End-effector position", x1)
    print("End-effector orientation")
    print(
        "\t Axis",
        R1.as_rotvec() / la.norm(R1.as_rotvec())
        if la.norm(R1.as_rotvec())
        else R1.as_rotvec(),
    )
    print("\t Angle", la.norm(R1.as_rotvec()) * 180 / m.pi)
    #    print('Inverse kinematics solution:', q1_i * 180 / m.pi)
    print("---")
    print("Test point 2:", q2 * 180 / m.pi)
    print("Forward kinematics solution:")
    print("End-effector position", x2)
    print("End-effector orientation")
    print(
        "\t Axis",
        R2.as_rotvec() / la.norm(R2.as_rotvec())
        if la.norm(R2.as_rotvec())
        else R2.as_rotvec(),
    )
    print("\t Angle", la.norm(R2.as_rotvec()) * 180 / m.pi)
    #    print('Inverse kinematics solution:', q2_i * 180 / m.pi)
    print("---")
    print("Test point 3:", q3 * 180 / m.pi)
    print("Forward kinematics solution:")
    print("End-effector position", x3)
    print("End-effector orientation")
    print(
        "\t Axis",
        R3.as_rotvec() / la.norm(R3.as_rotvec())
        if la.norm(R3.as_rotvec())
        else R3.as_rotvec(),
    )
    print("\t Angle", la.norm(R3.as_rotvec()) * 180 / m.pi)
    #    print('Inverse kinematics solution:', q3_i * 180 / m.pi)
    print("---")
    print("Test point 4:", q4 * 180 / m.pi)
    print("Forward kinematics solution:")
    print("End-effector position", x4)
    print("End-effector orientation")
    print(
        "\t Axis",
        R4.as_rotvec() / la.norm(R4.as_rotvec())
        if la.norm(R4.as_rotvec())
        else R4.as_rotvec(),
    )
    print("\t Angle", la.norm(R4.as_rotvec()) * 180 / m.pi)
#    print('Inverse kinematics solution:', q4_i * 180 / m.pi)
