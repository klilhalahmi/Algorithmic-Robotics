import sympy as sp  # Symbolic computation in Python
import numpy as np  # Numerical computation in Python
from math import pi, sqrt, atan # Math notations


class viper300:
    """ Describe the Viper300 6DOF robotic arm by Trossen Robotics
    The class provides the properties, transformation matrices and jacobian of the ViperX 300.    
    """

    def __init__(self):
        # Robots' joints
        self.q0 = sp.Symbol('q0')
        self.q1 = sp.Symbol('q1')
        self.q2 = sp.Symbol('q2')
        self.q3 = sp.Symbol('q3')
        self.q4 = sp.Symbol('q4')
        self.q5 = sp.Symbol('q5')
        self.q6 = sp.Symbol('q6')

        # length of the robots' links (approximated)
        self.l1 = 120 * 1e-3
        self.l2 = 300 * 1e-3
        self.l3 = 60 * 1e-3
        self.l4 = 200 * 1e-3
        self.l5 = 100 * 1e-3
        self.l6 = 200 * 1e-3

        # Calculate the transformation matrix for base to EE in operational space
        self.T = self.calculate_Tx().subs([('l1', self.l1),
                                           ('l2', self.l2),
                                           ('l3', self.l3),
                                           ('l4', self.l4),
                                           ('l5', self.l5),
                                           ('l6', self.l6)])

        self.A = self.DH_calculate().subs([('l1', self.l1),
                                           ('l2', self.l2),
                                           ('l3', self.l3),
                                           ('l4', self.l4),
                                           ('l5', self.l5),
                                           ('l6', self.l6)])

    def calculate_Tx(self):
        """ Calculate the transformation matrix for base to EE in operational space """

        # Rotation angles
        q0 = self.q0
        q1 = self.q1
        q2 = self.q2
        q3 = self.q3
        q4 = self.q4

        # Link lengths
        l1 = sp.Symbol('l1')
        l2 = sp.Symbol('l2')
        l3 = sp.Symbol('l3')
        l4 = sp.Symbol('l4')
        l5 = sp.Symbol('l5')
        l6 = sp.Symbol('l6')

        # In our physical model, the arm is pointing towards the y-axis, 
        # z is upwards, and x is pointing at the table's side

        # Rotate around the Z-axis
        T01 = sp.Matrix([[sp.cos(q0), -sp.sin(q0), 0, 0],
                         [sp.sin(q0), sp.cos(q0), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

        # Rotate around the X-axis, translate over Z
        T12 = sp.Matrix([[1, 0, 0, 0],
                         [0, sp.cos(q1), -sp.sin(q1), 0],
                         [0, sp.sin(q1), sp.cos(q1), l1],
                         [0, 0, 0, 1]])

        # Rotate around the X-axis, translate over Z and Y
        T23 = sp.Matrix([[1, 0, 0, 0],
                         [0, sp.cos(q2), -sp.sin(q2), l3],
                         [0, sp.sin(q2), sp.cos(q2), l2],
                         [0, 0, 0, 1]])

        # Rotate around the Y-axis, translate over Z and Y
        T34 = sp.Matrix([[sp.cos(q3), 0, sp.sin(q3), 0],
                         [0, 1, 0, l4],
                         [-sp.sin(q3), 0, sp.cos(q3), 0],
                         [0, 0, 0, 1]])

        # Rotate around the X-axis, translate over Y
        T45 = sp.Matrix([[1, 0, 0, 0],
                         [0, sp.cos(q4), -sp.sin(q4), l5],
                         [0, sp.sin(q4), sp.cos(q4), 0],
                         [0, 0, 0, 1]])

        # translate over Y
        T56 = sp.Matrix([[1, 0, 0, 0],
                         [0, 1, 0, l6],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

        # Building the transformation matrix
        T = T01 * T12 * T23 * T34 * T45 * T56

        # Calculating forward kinematics
        x = sp.Matrix([0, 0, 0, 1])
        Tx = T * x

        return Tx

    def get_xyz(self, q):
        """ Calculate EE location in operational space by solving the for Tx symbolically """
        return np.array(self.T.subs([('q0', q[0]),
                                     ('q1', q[1]),
                                     ('q2', q[2]),
                                     ('q3', q[3]),
                                     ('q4', q[4])]), dtype='float')

    def get_dh_xyz(self, q):
        return np.array(self.A.subs([('q0', q[0]),
                                     ('q1', q[1]),
                                     ('q2', q[2]),
                                     ('q3', q[3]),
                                     ('q4', q[4])]), dtype='float')

    def dh_matrix(self, a, phi, d, alpha):
        return sp.Matrix([[sp.cos(phi), -sp.sin(phi), 0, a],
                          [sp.sin(phi) * sp.cos(alpha), sp.cos(phi) * sp.cos(alpha), -sp.sin(alpha),
                           -d * sp.sin(alpha)],
                          [sp.sin(phi) * sp.sin(alpha), sp.cos(phi) * sp.sin(alpha), sp.cos(alpha), d * sp.cos(alpha)],
                          [0, 0, 0, 1]])

    def DH_calculate(self):
        """ Calculate the transformation matrix for base to EE in operational space """

        # Rotation angles
        q1 = self.q0
        q2 = self.q1
        q3 = self.q2
        q4 = self.q3
        q5 = self.q4
        q6 = 0

        # define the D-H parameters
        alpha_0 = 0
        alpha_1 = 0.5 * np.pi
        alpha_2 = 0
        alpha_3 = + 0.5 * np.pi
        alpha_4 = - 0.5 * np.pi
        alpha_5 = + 0.5 * np.pi

        phi_1 = q1 + 0.5 * np.pi
        phi_2 = q2 + atan(self.l2 / self.l3)
        phi_3 = q3 + atan(self.l3 / self.l2)
        phi_4 = q4
        phi_5 = q5
        phi_6 = q6

        a_0 = 0
        a_1 = 0
        a_2 = sqrt(pow(self.l2, 2) + pow(self.l3, 2))
        a_3 = 0
        a_4 = 0
        a_5 = 0

        d_1 = self.l1
        d_2 = 0
        d_3 = 0
        d_4 = self.l5 + self.l4
        d_5 = 0
        d_6 = self.l6


        ##a, alpha, d, phi
        A01 = self.dh_matrix(a_0, phi_1, d_1, alpha_0)
        A12 = self.dh_matrix(a_1, phi_2, d_2, alpha_1)
        A23 = self.dh_matrix(a_2, phi_3, d_3, alpha_2)
        A34 = self.dh_matrix(a_3, phi_4, d_4, alpha_3)
        A45 = self.dh_matrix(a_4, phi_5, d_5, alpha_4)
        A56 = self.dh_matrix(a_5, phi_6, d_6, alpha_5)

        # Building the transformation matrix
        A = A01 * A12 * A23 * A34 * A45 * A56

        # Calculating forward kinematics
        x = sp.Matrix([0, 0, 0, 1])
        Ax = A * x

        return Ax


arm = viper300()
print(arm.T)

robot_configuration = [3*pi/4, pi/3, 0, 0, pi/2]

print("Fk Location")
print(arm.get_xyz(robot_configuration))

print("DH location")
print(arm.get_dh_xyz(robot_configuration))
