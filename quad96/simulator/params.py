import numpy as np

from .utils import *

__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"

# this file contains all the drone i.e. crazyflie parameters and the PID values

class drone_params:
    def __init__(self):
        self.mass       = 0.027   # kg
        self.arm_length = 0.045  # meter

        self.I = np.array([[16.571710, 0.830806, 0.718277],
                           [0.830806, 16.655602, 1.800197],
                           [0.718277,  1.800197, 29.261652]])
        self.I *= 1e-6

        self.Kd = np.array([[-9.1785, 0., 0.],
                            [0., -9.1785, 0.],
                            [0., 0., -10.311]])
        self.Kd *= 1e-7
        
        # initial state of the drone [displacement, velocity, angular displacement, angular velocity]
        # [x, y, z, dx, dy, dz, phi, theta, psi, p (dphi), q (dtheta), r (dpsi)]
        self.init_state = np.array([0., 0., -0.01, 0., 0., 0., 0., 0., deg2rad(0), 0., 0., 0.]).T
        # initial drone inputs Motor voltage on scale 0-65535
        self.init_inputs = np.array([0., 0., 0., 0.]).T
        # body of drone wrt body frame (additional 1s for matrix matrix mul) and additional payload e.g. battery
        self.drone_body = np.array([[self.arm_length, 0.,  0.,   1.],
                                    [0., -self.arm_length, 0.,   1.],
                                    [-self.arm_length, 0., 0.,   1.],
                                    [0., self.arm_length,  0.,   1.],
                                    [0.,              0.,  0.,   1.],
                                    [0.,              0., -0.05, 1.]]).T


class pid_params:
    def __init__(self,  P, I, D, I_max):
        self.kP = P
        self.kI = I
        self.kD = D
        self.I_max = I_max
    
    def __neg__(self):
        return self(-self.kP, -self.kI, -self.kD)

class controller_params():
    def __init__(self):
        # PID gains for OLC and ILC
        # ---------------------------------------------- XYZ ------------------------------------------------
        # x
        self.pid_x       = pid_params(0.5, 0, 0.35, None)

        # y
        self.pid_y       = self.pid_x

        # z
        self.pid_z       = pid_params(25, 0., 15., None)
        
        # ------------------------------------------- dX dY dZ ----------------------------------------------
        # vx
        self.pid_vx       = pid_params(1, 0, 0, None)

        # vy
        self.pid_vy       = self.pid_vx

        # vz
        self.pid_vz       = pid_params(10, 0, 0., None)


        # ------------------------------------------------ RPY ----------------------------------------------
        # roll
        self.pid_roll     = pid_params(2e-4, 0, 5e-4, None)

        # pitch
        self.pid_pitch    = self.pid_roll

        # yaw
        self.pid_yaw      = pid_params(5e-4, 0, 0, None)