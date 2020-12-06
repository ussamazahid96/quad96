import os
import sys
import signal
import threading
import numpy as np
from time import sleep, time

from simulator.utils import *
from simulator.plot import quad_plotter
from simulator.controllers import AttitudeController, PositionController

__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"

# the main drone class combining controllers, plotters and providing the actual simulation of the quad 
# i.e. state update, setting parameters, functions to execute position and attitude commands
class Drone:
    def __init__(self, drone_params, controller_params, args=None):

        self.drone_params = drone_params
        self.args = args
        self.running = True
        # assuming that the positive z is downward
        self.g      = np.array([0., 0., 9.80665])
        self.weight_vector = drone_params.mass*self.g
        self.I_inv = np.linalg.inv(drone_params.I) 

        self.t      = 0.       
        self.att_dt = 1./500     
        self.att_steps  = int(1./self.att_dt) 
        self.pos_dt = 1./100
        self.pos_steps = int(1./self.pos_dt)
        self.st     = self.args.sim_time

        # state of the drone
        self.X      = np.copy(drone_params.init_state) 
        self.r      = self.X[0:3] 
        self.dr     = self.X[3:6]
        self.euler  = self.X[6:9] 
        self.w      = self.X[9:12]

        self.dX     = np.zeros((12,)) 

        # control input 
        self.u       = np.copy(drone_params.init_inputs)

        # crazyflie parameters
        self.c_t, self.b_t, self.a_t =  2.130295e-11, 1.032633e-6, 5.484560e-4
        self.b_omega, self.a_omega   =  0.04076521, 380.8359
        self.b_yaw, self.a_yaw       =  5.964552e-3, 1.563383e-5

        # position and attitude hold commands to be used by the functions in thread
        self.ph_command = np.array([0., 0., -3, deg2rad(0)])
        self.att_command = np.array([deg2rad(0), deg2rad(0), deg2rad(0), 0.])
        
        # controllers
        self.attitude_controller = AttitudeController(controller_params, False, dt=self.att_dt)
        self.position_controller = PositionController(controller_params, self.args.render_pid, dt=self.pos_dt, \
                                                      mode="position")      
        # threads
        self.ph_thread = None
        self.fp_thread = None
        self.att_thread = None

        # attach to the crazyflie via crazyradio
        if self.args.crazyflie:
            from crazyflie.crazyflie_class import CrazyFlie
            self.cf_handle = CrazyFlie(sim_vel=self.X[3:6])

        # render in 3d plot
        if self.args.render_quad:
            self.R_linear, _ = body2earth(self.euler)
            self.plot = quad_plotter(self.drone_params.drone_body, self.R_linear, self.r.reshape(-1, 1))
        
        # render in unity simulator
        if self.args.render_window:
            from simulator.network import ZMQPlug
            self.network_queue = None
            self.network_plug = ZMQPlug(self.X, self.network_queue, port=self.args.port)
        
        # for adding disturbances via keyboard (it will just change the state of the quad)
        if self.args.keyboard_dis and not self.args.crazyflie:
            from simulator.keyboard import KeyboardController
            self.keyboard_handle = KeyboardController(self.args, self.X)
        
        # for controlling the position of quad via keyboard (it will change the position hold command)
        # elif self.args.keyboard_pos:
        #     print("[Drone] Attaching Keyboard Handle for Position Control")
        #     self.keyboard_handle = KeyboardController(self.args, self.ph_command, scale=0.5)

        # for adding disturbances via camera (it will just change the state of the quad)
        if self.args.camera_dis:
            from camera.camera import SimpleCameraController
            self.camera_handle = SimpleCameraController(self.X, scale=0.15)

        # adding interrupt handler
        signal.signal(signal.SIGINT, self.interrupt_handle)


    def interrupt_handle(self, signal, frame):
        print('[Drone] Stopping')
        self.turnoff()

    @property
    def get_state(self):
        return self.X

    # turn off the drone and close everything with is attached
    def turnoff(self):
        if self.running:
            if self.args.crazyflie:
                self.cf_handle.close()
            if self.args.keyboard_dis:# or self.args.keyboard_pos:
                self.keyboard_handle.close()
            if self.args.render_window:
                self.network_plug.close()
            if self.args.camera_dis:
                self.camera_handle.close()
            
            self.running = False
            
            if self.ph_thread is not None:
                self.ph_thread.join()
            if self.fp_thread is not None:
                self.fp_thread.join()
            if self.att_thread is not None:
                self.att_thread.join()                
            print("[Drone] Turning Drone off...")

    # reset the drone state and controllers
    def reset(self):
        self.t       = 0

        self.X       = np.copy(self.drone_params.init_state)
        self.r       = self.X[0:3] 
        self.dr      = self.X[3:6] 
        self.euler   = self.X[6:9] 
        self.w       = self.X[9:12]
        
        self.dX      = np.zeros((12,))
        
        self.u       = np.copy(self.drone_params.init_inputs)

        self.attitude_controller.reset()
        self.position_controller.reset()

    # evaluate equations of motion for state update
    def step(self, u):
        thrust_motors = self.c_t*u[0]**2 + self.b_t*u[0] + self.a_t
        omega_motors  = self.b_omega*u[0] + self.a_omega

        self.R_linear, self.R_angular = body2earth(self.euler)
        f_thrust = 4*thrust_motors*np.array([0.,0.,-1.])
        v_body = np.dot(self.R_linear.T, self.X[3:6])
        f_drag = np.dot(self.drone_params.Kd, np.abs(omega_motors) * 4 * v_body)
        torque = u[1:]

        # linear velocity
        self.dX[0:3] = self.X[3:6] 
        # linear acceleration (from 2nd law sigma F = m d/dt (v))
        self.dX[3:6] = (1/self.drone_params.mass)*(self.weight_vector + np.dot(self.R_linear, f_thrust + f_drag))        
        # angular velocity Body frame
        self.dX[6:9] = np.dot(self.R_angular, self.w)
        # angular acceleration
        self.dX[9:12] = np.dot(self.I_inv, torque - np.cross(self.w, np.dot(self.drone_params.I, self.w)))

    # udpate the drone state
    def update_state(self, u):
        self.step(u)
        self.X     += self.dX*self.att_dt
        self.r      = self.X[0:3]
        self.dr     = self.X[3:6]
        self.euler  = self.X[6:9]
        self.w      = self.X[9:12]
        self.t     += self.att_dt
        return self.X

    # functions to set the individual values
    def set_position(self, position):
        self.X[0:3] = position
    def set_linear_velocity(self, velocity):
        self.X[3:6] = velocity
    def set_orientation(self, orientation):
        self.X[6:9] = orientation
    def set_angular_velocity(self, velocity):
        self.X[9:12] = velocity

    # function to execute the given attitude command e.g. (roll, pitch, yawrate, thrust)
    def attitude_control(self, refSig):
        j = 0
        while(j < self.st*self.att_steps and self.running):
            self.u = self.attitude_controller(refSig, self.X, self.t)
            self.X = self.update_state(self.u)
            j+=1
            if self.args.render_quad:
                self.plot.update_plot(self.R_linear, self.r.reshape(-1,1))

    # function to execute the given positon command e.g. (x, y, z, yawrate)
    def position_control(self, refSig):
        i = 0
        interval = 0
        while(i < self.st*self.pos_steps and self.running):
        # while(self.running):
            roll, pitch, thrust = self.position_controller(refSig[0:3], self.X, self.t)
            self.att_command = np.array([roll, pitch, refSig[3], thrust])
            j = 0
            while(j < self.att_steps//self.pos_steps and self.running):
                self.u = self.attitude_controller(self.att_command, self.X, self.t)
                self.X = self.update_state(self.u)
                j+=1            
            i += 1
            # interval += 1

            # if crazyflie is attached send the command to it as well
            if self.args.crazyflie:# and interval==2:
                # self.cf_handle._cf.commander.send_setpoint(rad2deg(roll), rad2deg(pitch), \
                #                                        rad2deg(refSig[3]), thrust)
                self.cf_handle._cf.commander.send_velocity_world_setpoint(self.X[3], \
                                                            self.X[4], -self.X[5], self.cf_handle._cf.yaw_rate)
            # if plotting the quad is required
            if self.args.render_quad:
                self.plot.update_plot(self.R_linear, self.r.reshape(-1,1))

    # function thread to hold a position given in position hold command
    def position_hold(self):
        while self.running:
            self.position_control(self.ph_command)
    def position_hold_threaded(self):
        self.ph_thread = threading.Thread(target=self.position_hold)
        self.ph_thread.start()

    # function to hold an attitude given in the  attitude hold command
    def att_hold(self):
        while self.running:
            self.attitude_control(self.att_command)