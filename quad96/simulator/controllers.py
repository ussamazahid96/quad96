import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D

from .plot import pid_plotter, pid_pend_plotter
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


# controller responsible for balancing the inverted pendulum by adjusting the orientation of the quad
class PendulumController:

    def __init__(self, controller_params, plot_controller=False, dt=0.01):
        self.dt = dt
        self.controller_params = controller_params

        # x
        self.pid_pend_theta = PIDcontroller(controller_params.pid_pend_theta, name='theta', dt=self.dt)
        # y
        self.pid_pend_phi = PIDcontroller(controller_params.pid_pend_phi, name='phi', dt=self.dt) 
        # z
        self.pid_z = PIDcontroller(controller_params.pid_z, name='z', dt=self.dt) 
        
        self.plot_controller = plot_controller
        self.counter = 0
        if self.plot_controller:
            self.plot = pid_pend_plotter()

    def __call__(self, refSig, quad_state, pend, time):
        # form paper [1]
        br = -3/4 * (1 - pend.X[0]**2/(pend.length**2 - pend.X[1]**2)) * pend.g
        bs = -3/(4*np.cos(quad_state[6])) * (1 - pend.X[1]**2/(pend.length**2 - pend.X[0]**2)) * pend.g

        ur = self.pid_pend_theta(refSig[0], pend.X[0])/br
        up = self.pid_pend_phi(refSig[1], pend.X[1])/bs

        roll_raw   = np.arctan(ur) 
        pitch_raw  = np.arctan(up) 
        thrust_raw = self.pid_z(refSig[2], quad_state[2])


        psi = deg2rad(0)     
        roll  = (pitch_raw * C(psi)) - (roll_raw  * S(psi))
        pitch = -(roll_raw  * C(psi)) - (pitch_raw * S(psi))
        # clipping angles and thrust
        roll = np.clip(roll, -deg2rad(max_ang), deg2rad(max_ang))
        pitch = np.clip(pitch, -deg2rad(max_ang), deg2rad(max_ang))
        thrust = thrust_raw*(-1000.) + 36000.
        thrust = int(np.clip(thrust, 20000, 65535))
        
        # ploting the pend coordinates every 10 iteration if tunning the PIDs
        self.counter += 1
        if self.plot_controller and self.counter == 10:
            self.plot.update_plot(time, pend.X, quad_state)
            self.counter = 0

        return roll, pitch, thrust

    def reset(self):
        self.counter = 0
        self.pid_pend_theta.reset()
        self.pid_pend_phi.reset()
        self.pid_z.reset()


# position controller working for r and s (body frame) coordinates of pendulum
class PositionController:

    def __init__(self, controller_params, plot_controller=False, dt=0.01):
        self.dt = dt
        self.controller_params = controller_params

        # r (x body frame)
        self.pid_pend_r = PIDcontroller(controller_params.pid_pend_r, name='pend_r', dt=self.dt)
        # s (y body frame)
        self.pid_pend_s = PIDcontroller(controller_params.pid_pend_s, name='pend_s', dt=self.dt) 
        
        self.plot_controller = plot_controller
        
        if self.plot_controller:
            self.plot = pid_plotter()

    def __call__(self, refSig, quad_state, pend, time):

        bx = -pend.g/pend.X[2]
        by =  bx

        rd  = self.pid_pend_r(refSig[0], quad_state[0])/bx
        sd  = self.pid_pend_s(refSig[1], quad_state[1])/by

        if self.plot_controller:
            self.plot.update_plot(time, state)

        return rd, sd

    def reset(self):
        self.pid_pend_r.reset()
        self.pid_pend_s.reset()

