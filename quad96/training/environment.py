import sys
import cv2
import tty
import termios
import threading
import numpy as np
from time import sleep

from simulator.utils import *
from simulator.drone_class import Drone
from simulator.controllers import max_ang

__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"

# this file provides the drone environment for training the RL agent using the drone simulator and adding 
# target position, bounds, reward, terminate, step, reset functions along with the assistant support for providing
# feedback from position controller

class drone_env:
    def __init__(self, drone_params, controller_params, args=None, target_position=np.array([0,0,-1.5], dtype=np.float32)):
        self.args = args
        self.drone = Drone(drone_params, controller_params, args=self.args)
        self.state_size = len(self.drone.get_state)+3+6
        self.action_size = 3
        self.env_bounds = 10
        self.lower_bounds = np.array([-self.env_bounds / 2, -self.env_bounds / 2, -self.env_bounds/2])
        self.upper_bounds = np.array([self.env_bounds / 2, self.env_bounds / 2, 0])
        self.angle_bound_init = deg2rad(max_ang)
        self.angle_bound_terminal = deg2rad(45)
        self.velocity_terminal = np.abs(10)
        self.thrust_scale = 29.535
        self.scale = np.array([self.angle_bound_init, self.angle_bound_init, self.thrust_scale])
        self.action_lower_limit = np.array([-self.angle_bound_init, -self.angle_bound_init, 20000.0])
        self.action_upper_limit = np.array([self.angle_bound_init, self.angle_bound_init, 65535.0])
        self.yaw = deg2rad(0)
        self.yaw_rate = np.zeros(1)
        self.target_position = np.hstack([target_position, self.yaw_rate])
        self.time_limit = self.args.sim_time
        # for calculating the reward
        self.w_p, self.w_v, self.w_e = 0.6, 0.2, 0.2
        # attitude steps taken for 1 position step
        self.steps = 1 if self.args.train else 5
        self._error_distance = 0
        self._velocity_magnitude = 0
        self._max_tilt = 0
        self.running = self.drone.running
        self.reward = 0
        self.camera_handle = None

        # if camera based position controller is required add `c` key controller and camera controller
        if self.args.camera_pos:
            from camera.camera import HandGestureCameraController
            self.camera_handle = HandGestureCameraController()
            from simulator.keyboard import KeyboardController_CD
            self.keyboard_handle = KeyboardController_CD(self.args, self.target_position, camera=self.camera_handle)
       
        # attach the keyboard for position control if crazyflie's keyboard is null
        if self.args.keyboard_pos:
            from simulator.keyboard import KeyboardController
            self.keyboard_handle = KeyboardController(self.args, self.target_position, scale=0.5)

    @property
    def error_distance(self):
        self._error_distance = np.linalg.norm(self.target_position[:3] - self.drone.r)
        return self._error_distance
    
    @property
    def velocity_magnitude(self):
        self._velocity_magnitude = np.linalg.norm(self.drone.dr)
        return self._velocity_magnitude
    
    @property
    def max_tilt(self):
        self._max_tilt = max(np.abs(self.drone.euler[0]), np.abs(self.drone.euler[1]))
        return self._max_tilt
    
    # reward function
    def get_reward(self):        
        self.reward = self.w_p * np.exp(-self.error_distance) + \
                      self.w_v * np.exp(-self.velocity_magnitude) + \
                      self.w_e * np.exp(-self.max_tilt)
        return self.reward


    def reset(self):
        self.drone.attitude_controller.reset()
        self.drone.position_controller.reset()
        self.drone.pend_controller.reset()
        self.drone.pend.reset()#pose=np.random.uniform([-0.1, -0.1], [0.1, 0.1]))
        x     =  np.random.uniform(-1, 1)
        y     =  np.random.uniform(-1, 1)
        z     =  np.random.uniform(-0.5, -2.5) 
        roll  =  0#np.random.uniform(-1, 1)*self.angle_bound_init
        pitch =  0#np.random.uniform(-1, 1)*self.angle_bound_init
        yaw   =  self.yaw#np.random.uniform(-1,1)*np.pi
        self.drone.set_position(np.array([x, y, z]))
        self.drone.set_linear_velocity(np.array([0, 0, 0]))
        self.drone.set_orientation(np.array([roll, pitch, yaw]))
        self.drone.set_angular_velocity(np.array([0, 0, 0]))
        self.drone.t = 0
        delta = self.target_position[:3] - self.drone.get_state[:3]
        state = np.concatenate((delta, self.drone.get_state[3:], [self.error_distance], \
                               [self.velocity_magnitude], [self.max_tilt], self.drone.pend.X), axis=0) 
        return state

    # env step function 
    def step(self, action):
        j = 0
        # mapping [-29.5,29.5] thrust to [0-65535]
        action[2] = action[2]*(-1000.) + 36000.
        action = np.clip(action, self.action_lower_limit, self.action_upper_limit)
        att_command = np.array([action[0], action[1], self.target_position[-1], action[2]])
        while(j < self.steps and self.drone.running):
            self.drone.u = self.drone.attitude_controller(att_command, self.drone.X, self.drone.t)
            self.drone.X = self.drone.update_state(self.drone.u)
            j += 1
        if self.args.camera_pos:
            cv2.imshow("Video Feed", self.camera_handle.frame)
            waitkey = cv2.waitKey(1) & 0xFF
        self.get_reward()
        done = self.terminate(self.drone.get_state)
        delta = self.target_position[:3] - self.drone.get_state[:3]
        next_state = np.concatenate((delta, self.drone.get_state[3:], [self.error_distance], \
                                    [self.velocity_magnitude], [self.max_tilt], self.drone.pend.X), axis=0)
        return next_state, self.reward, done, None
    
    # determine when to terminate the episode
    def terminate(self, state):
        done = False
        if np.any(self.drone.get_state[0:3] < self.lower_bounds) or np.any(self.drone.get_state[0:3] > self.upper_bounds):
            self.reward = -1
            done = True
        if np.linalg.norm(state[3:6]) > self.velocity_terminal:
            done = True
        if np.any(np.abs(state[6:8]) > self.angle_bound_terminal):
            done = True
        if self.drone.t > self.time_limit:
            done = True
        if not self.drone.pend.running:
            done = True
        return done
    
    def close(self):
        self.drone.turnoff()
        if self.args.camera_pos:
            self.camera_handle.close()
            self.keyboard_handle.close()
        if self.args.keyboard_pos:
            self.keyboard_handle.close()
        print("[Env] Closing env...")
        
    # for providing feedback from the position controller in a given state
    def assistant(self):
        rd, sd = self.drone.position_controller(self.target_position[:3], self.drone.X, self.drone.pend, self.drone.t)
        pend_position = np.array([rd, sd, self.target_position[2]])
        roll, pitch, thrust = self.drone.pend_controller(pend_position, self.drone.X, self.drone.pend, self.drone.t)
        thrust = (thrust - 36000.)/(-1000.)
        actions = np.array([roll, pitch, thrust])
        return actions, None
