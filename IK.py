from enum import Enum
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

class Optimizer(Enum):
    """ Designation of an optimization method for inverse kinematic
    
    We support two optimization methods for inverse kinematic:
    1. Standard resolved motion (STD): Based on Pseudo-inversed jacobian
    2. Dampened least squares method (DLS) or the Levenberg–Marquardt algorithm: 
        see https://en.wikipedia.org/wiki/Levenberg–Marquardt_algorithm for a detailed description
    """

    STD = 1 
    DLS = 2

class Method(Enum):
    ELISHAY_METHOD = 1
    NEWTHON_RAPHSON = 2
    CIRCLE = 3

class viper300:
    """ Describe the Viperx200 6DOF robotic arm by Trossen Robotic
    
    The class provides the properties, transformation matrices and jacobian of the ViperX 300.
    The arm is described in: https://www.trossenrobotics.com/viperx-300-robot-arm-6dof.aspx
    """

    def __init__ (self):

        # Robots' joints
        self.n_joints = 5
        self.q0 = sp.Symbol('q0')
        self.q1 = sp.Symbol('q1')
        self.q2 = sp.Symbol('q2')
        self.q3 = sp.Symbol('q3')
        self.q4 = sp.Symbol('q4')

        # length of the robots' links
        self.l1 = 126.75* 1e-3
        self.l2 = 300* 1e-3
        self.l3 = 60* 1e-3
        self.l4 = 196.38* 1e-3
        self.l5 = 103.62* 1e-3
        self.l6 = 206.58* 1e-3

        # Calculate the transformation matrix for base to EE in operational space
        self.T = self.calculate_Tx().subs([('l1', self.l1),
                                           ('l2', self.l2),
                                           ('l3', self.l3),
                                           ('l4', self.l4),
                                           ('l5', self.l5),
                                           ('l6', self.l6)])

        # Calculate the Jacobian matrix for the EE
        self.J = self.calculate_J().subs([('l1', self.l1),
                                          ('l2', self.l2),
                                          ('l3', self.l3),
                                          ('l4', self.l4),
                                          ('l5', self.l5),
                                          ('l6', self.l6)])

    def calculate_Tx(self):
        """ Calculate the transformation matrix for base to EE in operational space """

        q0 = self.q0
        q1 = self.q1
        q2 = self.q2
        q3 = self.q3
        q4 = self.q4

        l1 = sp.Symbol('l1')
        l2 = sp.Symbol('l2')
        l3 = sp.Symbol('l3')
        l4 = sp.Symbol('l4')
        l5 = sp.Symbol('l5')
        l6 = sp.Symbol('l6')

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

        T = T01 * T12 * T23 * T34 * T45 * T56
        x = sp.Matrix([0, 0, 0, 1])
        Tx = T * x

        return Tx

    def calculate_J(self):
        """ Calculate the Jacobian matrix for the EE """

        q = [self.q0, self.q1, self.q2, self.q3, self.q4]
        J = sp.Matrix.ones(3, 5)
        for i in range(3):     # x, y, z
            for j in range(5): # Five joints
                # Differentiate and simplify
                J[i, j] = sp.simplify(self.T[i].diff(q[j]))

        return J

    def get_xyz_symbolic(self, q):
        """ Calculate EE location in operational space by solving the for Tx symbolically """

        return np.array(self.T.subs([('q0', q[0]),
                                     ('q1', q[1]),
                                     ('q2', q[2]),
                                     ('q3', q[3]),
                                     ('q4', q[4])]), dtype='float')

    def calc_J_symbolic(self, q):
        """ Calculate the jacobian symbolically """
        return np.array(self.J.subs([('q0', q[0]),
                                     ('q1', q[1]),
                                     ('q2', q[2]),
                                     ('q3', q[3]),
                                     ('q4', q[4])]), dtype='float')

    def get_xyz_numeric(self, q):
        """ Calculate EE location in operational space by solving the for Tx numerically
         
        Equation was derived symbolically and was then written here manually.
        Nuerical evaluation works faster then symbolically. 
        """

        c0 = np.cos(q[0])
        c1 = np.cos(q[1])
        c2 = np.cos(q[2])
        c3 = np.cos(q[3])
        c4 = np.cos(q[4])

        s0 = np.sin(q[0])
        s1 = np.sin(q[1])
        s2 = np.sin(q[2])
        s3 = np.sin(q[3])
        s4 = np.sin(q[4])

        return np.array([[0.20658*((s0*s1*c2 + s0*s2*c1)*c3 + s3*c0)*s4 + 0.20658*(s0*s1*s2 - s0*c1*c2)*c4 + 0.3*s0*s1*s2 + 0.3*s0*s1 - 0.3*s0*c1*c2 - 0.06*s0*c1], [0.20658*((-s1*c0*c2 - s2*c0*c1)*c3 + s0*s3)*s4 + 0.20658*(-s1*s2*c0 + c0*c1*c2)*c4 - 0.3*s1*s2*c0 - 0.3*s1*c0 + 0.3*c0*c1*c2 + 0.06*c0*c1], [0.20658*(-s1*s2 + c1*c2)*s4*c3 + 0.20658*(s1*c2 + s2*c1)*c4 + 0.3*s1*c2 + 0.06*s1 + 0.3*s2*c1 + 0.3*c1 + 0.12675], [1]], dtype='float')

    def calc_J_numeric(self, q):
        """ Calculate the Jacobian for q symbolically
         
         Equation was derived symbolically and was then written here manually.
         Nuerical evaluation works faster then symbolically. 
         """

        c0 = np.cos(q[0])
        c1 = np.cos(q[1])
        c2 = np.cos(q[2])
        c3 = np.cos(q[3])
        c4 = np.cos(q[4])

        s0 = np.sin(q[0])
        s1 = np.sin(q[1])
        s2 = np.sin(q[2])
        s3 = np.sin(q[3])
        s4 = np.sin(q[4])

        s12 = np.sin(q[1] + q[2])
        c12 = np.cos(q[1] + q[2])

        return np.array([[-0.20658*s0*s3*s4 + 0.3*s1*c0 + 0.20658*s4*s12*c0*c3 - 0.06*c0*c1 - 0.20658*c0*c4*c12 - 0.3*c0*c12, (0.06*s1 + 0.20658*s4*c3*c12 + 0.20658*s12*c4 + 0.3*s12 + 0.3*c1)*s0, (0.20658*s4*c3*c12 + 0.20658*s12*c4 + 0.3*s12)*s0, -0.20658*(s0*s3*s12 - c0*c3)*s4, 0.20658*(s0*s12*c3 + s3*c0)*c4 + 0.20658*s0*s4*c12], [0.3*s0*s1 + 0.20658*s0*s4*s12*c3 - 0.06*s0*c1 - 0.20658*s0*c4*c12 - 0.3*s0*c12 + 0.20658*s3*s4*c0, -(0.06*s1 + 0.20658*s4*c3*c12 + 0.20658*s12*c4 + 0.3*s12 + 0.3*c1)*c0, -(0.20658*s4*c3*c12 + 0.20658*s12*c4 + 0.3*s12)*c0, 0.20658*(s0*c3 + s3*s12*c0)*s4, 0.20658*(s0*s3 - s12*c0*c3)*c4 - 0.20658*s4*c0*c12], [0, -0.3*s1 - 0.20658*s4*s12*c3 + 0.06*c1 + 0.20658*c4*c12 + 0.3*c12, -0.20658*s4*s12*c3 + 0.20658*c4*c12 + 0.3*c12, -0.20658*s3*s4*c12, -0.20658*s4*s12 + 0.20658*c3*c4*c12]], dtype='float')

def goto_target(arm, target, N, optimizer = Optimizer.DLS):
    """ Giving arm object, a target and optimizer, provides the required set of control signals 
    
    Returns the optimizing trajectory, error trace and arm configurazion to achieve the target.
    Target is defined in relative to the EE null position
    """

    q = np.array([[0]*arm.n_joints], dtype='float').T # Zero confoguration of the arm
    xyz_0 = (arm.get_xyz_numeric(q))[:-1]             # Zero position of the arm
    trajectory = []
    # error_tract = []
    count = 0

    for i in range(N):
        target_i = xyz_0 + (i+1)*(target - xyz_0)/N     #the target for this interval
        while count < 200*N:
            xyz_c = (arm.get_xyz_numeric(q))[:-1]      # Get current EE position
            trajectory.append(xyz_c)                   # Store to track trajectory
            xyz_d = target_i - xyz_c                  # Get vector to target
            error = np.sqrt(np.sum(xyz_d**2))          # Distance to target
            # error_tract.append(error)                  # Store distance to track error

            kp = 0.1                                   # Proportional gain term
            ux = xyz_d * kp                            # direction of movement
            J_x = arm.calc_J_numeric(q)                # Calculate the jacobian

            # Solve inverse kinematics accorting to the designated optimizaer
            if optimizer is Optimizer.STD: # Standard resolved motion
                u = np.dot(np.linalg.pinv(J_x), ux)

            elif optimizer is Optimizer.DLS: # Dampened least squares method
                u = np.dot(J_x.T, np.linalg.solve(np.dot(J_x, J_x.T) + np.eye(3) * 0.001, ux))

            q += u
            count += 1

            # Stop when within 1mm accurancy (arm mechanical accurancy limit)
            if error < .001:
                break

    print('Arm config: {}, with error: {}, achieved @ step: {}'.format(
        np.rad2deg(q.T).astype(float), error, count))
    return q, trajectory, (arm.get_xyz_numeric(q))[:-1], count

def goto_target_nr(arm, target, N):
    q = np.array([[0] * arm.n_joints], dtype='float').T  # Zero confoguration of the arm
    xyz_0 = (arm.get_xyz_numeric(q))[:-1]  # Current operational position of the arm

    count = 0
    trajectory = []
    # error_tract = []

    for i in range(N):
        target_i = xyz_0 + (i+1)*(target - xyz_0)/N
        while count < 100*N:
            xyz_current = (arm.get_xyz_numeric(q))[:-1]  # Get current EE position
            trajectory.append(xyz_current)                # Store to track trajectory
            xyz_d = target_i - xyz_current                     # Get vector to target
            error = np.sqrt(np.sum(xyz_d ** 2))      # Distance to target
            # error_tract.append(error)                # Store distance to track error

            J_theta_0 = arm.calc_J_numeric(q)  # Calculate the jacobian in theta 0
            q = q + np.dot(J_theta_0.T, xyz_d)
            count += 1

            # Stop when within 1mm accurancy (arm mechanical accurancy limit)
            if error < .001:
                break

    print('Arm config: {}, with error: {}, achieved @ step: {}'.format(
        np.rad2deg(q.T).astype(float), error, count))

    return q, trajectory, (arm.get_xyz_numeric(q))[:-1], count

def goto_circle(arm, N):
    q = np.array([[0]*arm.n_joints], dtype='float').T # Zero confoguration of the arm
    xyz_0 = (arm.get_xyz_numeric(q))[:-1]             # Zero position of the arm
    x_0 = (xyz_0)[0]
    y_0 = (xyz_0)[1]
    z_0 = (xyz_0)[2]
    trajectory = []
    # error_tract = []
    count = 0

    for i in range(N):
        alpha = 2*np.pi*(i+1)/N
        target_i = [x_0, y_0 - 0.1*np.sin(alpha), z_0 - 0.1*(1-np.cos(alpha))]     #the target for this interval
        while count < 100*N:
            xyz_c = (arm.get_xyz_numeric(q))[:-1]      # Get current EE position
            trajectory.append(xyz_c)                   # Store to track trajectory
            xyz_d = target_i - xyz_c                  # Get vector to target
            error = np.sqrt(np.sum(xyz_d**2))          # Distance to target
            # error_tract.append(error)                  # Store distance to track error

            kp = 0.1                                   # Proportional gain term
            ux = xyz_d * kp                            # direction of movement
            J_x = arm.calc_J_numeric(q)                # Calculate the jacobian

            u = np.dot(J_x.T, np.linalg.solve(np.dot(J_x, J_x.T) + np.eye(3) * 0.001, ux))

            q += u
            count += 1

            # Stop when within 1mm accurancy (arm mechanical accurancy limit)
            if error < .001:
                break

    return q, trajectory, (arm.get_xyz_numeric(q))[:-1], count


def show_move(arm, target, N, method = Method.ELISHAY_METHOD ):
    if method is Method.ELISHAY_METHOD:  # using the original method
        q, trajectory, xyz_c, count = goto_target(arm, target, N, Optimizer.DLS)

    elif method is Method.NEWTHON_RAPHSON:  # using newton raphson method
        q, trajectory, xyz_c, count = goto_target_nr(arm, target, N)

    elif method is Method.CIRCLE:  # using the original method and move in a circle
        q, trajectory, xyz_c, count = goto_circle(arm, N)

    xline = []
    yline = []
    zline = []
    steps = []

    for i in range(count):
        xline.append(trajectory[i][0])
        yline.append(trajectory[i][1])
        zline.append(trajectory[i][2])
        steps.append(i+1)

    # path of the EE in 3d graph
    xline = np.concatenate(xline)
    yline = np.concatenate(yline)
    zline = np.concatenate(zline)

    ax = plt.axes(projection='3d')
    ax.plot3D(xline, yline, zline)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.set_xlim([min(xline), max(xline)])
    ax.set_ylim([min(yline), max(yline)])
    ax.set_zlim([min(zline), max(zline)])

    plt.show()

    # error graph
    # plt.plot(steps, error_tract)
    # plt.xlabel('Iterations')
    # plt.ylabel('Error')
    # plt.show()


if __name__ == "__main__":
    arm = viper300()
    # print(arm.J)
    # print(arm.T)
    target = [[0.1], [0.367], [0.718]]
    method = Method.NEWTHON_RAPHSON
    N = 10 #number of intervals
    show_move(arm, target, N, method)


