import sys
import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D

from .utils import rad2deg, C, S, deg2rad


__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"


# class definations for plotting the pend in 3d matplotlib plot
class inv_pend_plotter:
    def __init__(self, pend_state, offset, max_tilt):
        self.xdata = []
        self.ydata = []
        self.zdata = []
        self.offset = offset
        self.max_tilt = max_tilt
        self.pend_state = pend_state
        self.fig = plt.figure(frameon=False)
        self.ax = Axes3D.Axes3D(self.fig)#, azim=-80, elev=15)
        
        self.xlim, self.ylim, self.zlim = .5, .5, 1.
        self.ax.set_xlabel('X')
        self.ax.set_xlim3d([-self.xlim, self.xlim])
        self.ax.set_ylabel('Y')
        self.ax.set_ylim3d([-self.ylim, self.ylim])
        self.ax.set_zlabel('Z')
        self.ax.set_zlim3d([0, -self.zlim])
        
        self.pendulum, = self.ax.plot3D([],[],[], zdir='z', color='blue', linewidth=1, antialiased=False)
        self.mass, = self.ax.plot3D([],[],[], zdir='z', color='red', marker='o', antialiased=False)
        
        self.fig.canvas.mpl_connect('key_press_event', self.keypress_routine)

        self.pendulum.set_data(np.array([self.pend_state[0]+self.offset[0], self.offset[0]]),
                               np.array([self.pend_state[1]+self.offset[1], self.offset[1]]))
        self.pendulum.set_3d_properties(np.array([self.pend_state[2]+self.offset[2], self.offset[2]]))

        self.mass.set_data(self.pend_state[0]+self.offset[0], 
                           self.pend_state[1]+self.offset[1])
        self.mass.set_3d_properties(self.pend_state[2]+self.offset[2])

    def update_plot(self, pend_state, offset):
        self.offset = offset
        self.pend_state = pend_state 

        self.pendulum.set_data(np.array([self.pend_state[0]+self.offset[0], self.offset[0]]),
                               np.array([self.pend_state[1]+self.offset[1], self.offset[1]]))
        self.pendulum.set_3d_properties(np.array([self.pend_state[2]+self.offset[2], self.offset[2]]))

        self.mass.set_data(self.pend_state[0]+self.offset[0], 
                           self.pend_state[1]+self.offset[1])
        self.mass.set_3d_properties(self.pend_state[2]+self.offset[2])

        plt.pause(0.00000001)
        
        if self.pend_state[0]**2 + self.pend_state[1]**2 >= self.max_tilt**2:
            plt.show(block=True)
   
    def keypress_routine(self, event):
        sys.stdout.flush()
        if event.key == 'a':
            self.pend_state[0] -= 0.1
        elif event.key == 'd':
            self.pend_state[0] += 0.1
        elif event.key == 'w':
            self.pend_state[1] += 0.1
        elif event.key == 'x':
            self.pend_state[1] -= 0.1
        elif event.key == 'q':
            exit(0)

# class definations for plotting the quad with pend in 3d matplotlib plot
class quad_plotter:
    def __init__(self, drone_body, R_linear, position, pend):
        self.xdata = []
        self.ydata = []
        self.zdata = []
        self.pend = pend
        self.drone_body = drone_body
        self.position = position
        self.fig = plt.figure(frameon=False)
        self.ax = Axes3D.Axes3D(self.fig)#, azim=-90, elev=90)
        
        self.xlim, self.ylim, self.zlim = .3, .3, .7
        self.ax.set_xlabel('X')
        self.ax.set_xlim3d([-self.xlim, self.xlim])
        self.ax.set_ylabel('Y')
        self.ax.set_ylim3d([-self.ylim, self.ylim])
        self.ax.set_zlabel('Z')
        self.ax.set_zlim3d([0, -self.zlim])
        
        self.arm13, = self.ax.plot3D([],[],[], zdir='z', color='green', linewidth=2, antialiased=False)
        self.arm24, = self.ax.plot3D([],[],[], zdir='z', color='green', linewidth=2, antialiased=False)
        self.shadow, = self.ax.plot3D([],[],[], zdir='z', color='red', marker='.', antialiased=False)

        self.pendulum, = self.ax.plot3D([],[],[], zdir='z', color='blue', linewidth=1, antialiased=False)
        self.mass, = self.ax.plot3D([],[],[], zdir='z', color='red', marker='o', antialiased=False)
        
        self.fig.canvas.mpl_connect('key_press_event', self.keypress_routine)       
        
        wHb = np.concatenate((R_linear, position), axis=1)
        
        drone_fig = np.matmul(wHb, self.drone_body)
        self.arm13.set_data(drone_fig[0,[0, 2]], drone_fig[1,[0, 2]])
        self.arm13.set_3d_properties(drone_fig[2, [0,2]])
        self.arm24.set_data(drone_fig[0,[1,3]], drone_fig[1,[1,3]])
        self.arm24.set_3d_properties(drone_fig[2,[1,3]])

        self.shadow.set_data(drone_fig[0,5], drone_fig[1,5])
        self.shadow.set_3d_properties(0)

        self.pendulum.set_data(np.array([self.pend.X[0]+position[0,0], position[0,0]]),
                               np.array([self.pend.X[1]+position[1,0], position[1,0]]))
        self.pendulum.set_3d_properties(np.array([self.pend.X[2]+position[2,0], position[2,0]]))

        self.mass.set_data(self.pend.X[0]+position[0,0], 
                           self.pend.X[1]+position[1,0])
        self.mass.set_3d_properties(self.pend.X[2]+position[2,0]) 


    def update_plot(self, R_linear, position): 
        self.position = position       
        wHb = np.concatenate((R_linear, position), axis=1)
        drone_fig = np.matmul(wHb, self.drone_body)
        self.arm13.set_data(drone_fig[0,[0, 2]], drone_fig[1,[0, 2]])
        self.arm13.set_3d_properties(drone_fig[2, [0,2]])
        self.arm24.set_data(drone_fig[0,[1,3]], drone_fig[1,[1,3]])
        self.arm24.set_3d_properties(drone_fig[2,[1,3]])

        self.shadow.set_data(drone_fig[0,5], drone_fig[1,5])
        self.shadow.set_3d_properties(0)

        self.pendulum.set_data(np.array([self.pend.X[0]+position[0,0], position[0,0]]),
                               np.array([self.pend.X[1]+position[1,0], position[1,0]]))
        self.pendulum.set_3d_properties(np.array([self.pend.X[2]+position[2,0], position[2,0]]))

        self.mass.set_data(self.pend.X[0]+position[0,0], 
                           self.pend.X[1]+position[1,0])
        self.mass.set_3d_properties(self.pend.X[2]+position[2,0]) 

        plt.pause(0.00000001)
        if(position[-1] > 0):
            plt.show(block=True)
   
    def keypress_routine(self, event):
        sys.stdout.flush()
        if event.key == 'a':
            self.position[0] -= 0.05
        elif event.key == 'd':
            self.position[0] += 0.05
        elif event.key == 'w':
            self.position[1] += 0.05
        elif event.key == 'x':
            self.position[1] -= 0.05
        elif event.key == 't':
            self.position[2] -= 0.05
        elif event.key == 'g':
            self.position[2] += 0.05
        elif event.key == 'q':
            exit(0)


# class definaiton to plot the state against time. This is used for manual PID tuning
class pid_pend_plotter:

    def __init__(self):

        self.quad_x = plt.subplot(2,3,1)
        self.quad_x.set_title("quad_x")
        plt.grid()
        self.quad_y = plt.subplot(2,3,2)
        self.quad_y.set_title("quad_y")
        plt.grid()

        self.quad_z = plt.subplot(2,3,3)
        self.quad_z.set_title("quad_z")
        plt.grid()

        self.pend_x = plt.subplot(2,3,4)
        self.pend_x.set_title("pend_x")
        plt.grid()

        self.pend_y = plt.subplot(2,3,5)
        self.pend_y.set_title("pend_y")
        plt.grid()

        self.pend_z = plt.subplot(2,3,6)
        self.pend_z.set_title("pend_z")
        plt.grid()

    def update_plot(self, time, pend_state, quad_state):        
        self.pend_x.plot(time, pend_state[0], 'b.')
        self.pend_y.plot(time, pend_state[1], 'b.')
        self.pend_z.plot(time, -pend_state[2], 'b.')

        self.quad_x.plot(time, quad_state[0], 'r.')
        self.quad_y.plot(time, quad_state[1], 'r.')
        self.quad_z.plot(time, -quad_state[2], 'r.')

# class definaiton to plot the state against time. This is used for manual PID tuning
class pid_plotter:

    def __init__(self):

        self.phi_plt = plt.subplot(2,3,1)
        self.phi_plt.set_title("phi[deg]")
        plt.grid()
        self.theta_plt = plt.subplot(2,3,2)
        self.theta_plt.set_title("theta[deg]")
        plt.grid()

        self.psi_plt = plt.subplot(2,3,3)
        self.psi_plt.set_title("psi[deg]")
        plt.grid()

        self.x_plt = plt.subplot(2,3,4)
        self.x_plt.set_title("x[m]")
        plt.grid()

        self.y_plt = plt.subplot(2,3,5)
        self.y_plt.set_title("y[m]")
        plt.grid()

        self.z_plt = plt.subplot(2,3,6)
        self.z_plt.set_title("z_dot [m]")
        plt.grid()

    def update_plot(self, time, state):        
        self.x_plt.plot(time, state[0], 'b.')
        self.y_plt.plot(time, state[1], 'b.')
        # here -ve z axis in converted to +ve
        self.z_plt.plot(time, -state[2], 'b.')

        self.phi_plt.plot(time, rad2deg(state[6]), 'b.')
        self.theta_plt.plot(time, rad2deg(state[7]), 'b.')
        self.psi_plt.plot(time, rad2deg(state[8]), 'b.')