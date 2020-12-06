import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D

from .plot import pid_plotter
from .utils import rad2deg, C, S, deg2rad


__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"

# max tilt angle of the crazyflie
max_ang = 20


# class defination for a PID controller
class PIDcontroller:
    
    def __init__(self, controller_params, name=None, dt=0.01):

        self.dt = dt
        self.name = name

        self.err  = 0.  
        self.err_prev = 0. 
        self.err_sum  = 0.

        # gains
        self.kP, self.kI, self.kD = controller_params.kP, controller_params.kI, controller_params.kD
        self.I_max = controller_params.I_max

    def __call__(self, desired_state, current_state):
        self.err = desired_state - current_state  
        self.err_sum = self.saturate(self.err_sum + self.err)
        ret = (self.kP * self.err) + (self.kI * self.err_sum * self.dt) + (self.kD * (self.err - self.err_prev)/self.dt)
        self.err_prev = self.err
        return ret

    def saturate(self, err):
        if self.I_max is not None:
            err = np.clip(err, -self.I_max, self.I_max)
        return err


    def reset(self):
        self.err = 0.
        self.err_sum = 0.
        self.err_prev = 0.


# attitude controller of crazyflie. It just has 3 PID's for roll pitch and yawrate
class AttitudeController:
    
    def __init__(self, controller_params, plot_controller=False, dt=0.01):
        
        self.dt = dt
        self.controller_params = controller_params

        # phi
        self.pid_roll     = PIDcontroller(controller_params.pid_roll, name='roll', dt=self.dt)  

        # theta
        self.pid_pitch    = PIDcontroller(controller_params.pid_pitch, name='pitch', dt=self.dt) 

        # psi
        self.pid_yaw      = PIDcontroller(controller_params.pid_yaw, name='yaw', dt=self.dt)


        self.plot_controller = plot_controller
        

        if self.plot_controller:
            self.plot = pid_plotter()


    def __call__(self, refSig, state, time):

        # roll
        torque_x = self.pid_roll(refSig[0], state[6])
        # pitch
        torque_y = self.pid_pitch(refSig[1], state[7])
        # yaw rate
        torque_z = self.pid_yaw(refSig[2], state[11])

        u = np.array([refSig[3], torque_x, torque_y, torque_z])

        if self.plot_controller:
            self.plot.update_plot(time, state)
        return u

    def reset(self):
        self.pid_roll.reset()
        self.pid_pitch.reset()
        self.pid_yaw.reset()


# position controller working in position and velocity mode, use 6 PID's i.e. 3 for each xyz
class PositionController:

    def __init__(self, controller_params, plot_controller=False, dt=0.01, mode="position"):
        self.dt = dt
        self.mode = mode
        self.controller_params = controller_params

        # x
        self.pid_x = PIDcontroller(controller_params.pid_x, name='x', dt=self.dt)
        # y
        self.pid_y = PIDcontroller(controller_params.pid_y, name='y', dt=self.dt) 
        # z
        self.pid_z = PIDcontroller(controller_params.pid_z, name='z', dt=self.dt) 

        # velocity x
        self.pid_vx = PIDcontroller(controller_params.pid_vx, name='vx', dt=self.dt)
        # velocity y
        self.pid_vy = PIDcontroller(controller_params.pid_vy, name='vy', dt=self.dt) 
        # velocity z
        self.pid_vz = PIDcontroller(controller_params.pid_vz, name='vz', dt=self.dt) 
        
        self.plot_controller = plot_controller
        
        if self.plot_controller:
            self.plot = pid_plotter()

    def __call__(self, refSig, state, time):

        if self.mode == "position":
            roll_raw   = self.pid_x(refSig[0], state[0])
            pitch_raw  = self.pid_y(refSig[1], state[1])
            thrust_raw = self.pid_z(refSig[2], state[2])
        elif self.mode == "velocity":
            refSig[0:3] = np.clip(refSig[0:3], -1.1, 1.1)
            roll_raw   = self.pid_vx(refSig[0], state[3])
            pitch_raw  = self.pid_vy(refSig[1], state[4])
            thrust_raw = self.pid_vz(refSig[2], state[5])

        psi = state[8]      
        roll  = (pitch_raw * C(psi)) - (roll_raw  * S(psi))
        pitch = -(roll_raw  * C(psi)) - (pitch_raw * S(psi))

        # clip the angles and thrust
        roll = np.clip(roll, -deg2rad(max_ang), deg2rad(max_ang))
        pitch = np.clip(pitch, -deg2rad(max_ang), deg2rad(max_ang))
        thrust = thrust_raw*(-1000.) + 36000.
        thrust = int(np.clip(thrust, 20000, 65535))

        if self.plot_controller:
            self.plot.update_plot(time, state)

        return roll, pitch, thrust

    def reset(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()

        self.pid_vx.reset()
        self.pid_vy.reset()
        self.pid_vz.reset()

