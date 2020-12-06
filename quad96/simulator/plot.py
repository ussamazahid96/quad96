import sys
import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D

from .utils import rad2deg, C, S, deg2rad

__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"

# class definations for plotting the quad in 3d matplotlib plot
class quad_plotter:
    def __init__(self, drone_body, R_linear, position):
        self.xdata = []
        self.ydata = []
        self.zdata = []
        self.drone_body = drone_body
        self.position = position
        self.fig = plt.figure(frameon=False)
        self.ax = Axes3D.Axes3D(self.fig)#, azim=-2, elev=9)
        
        self.xlim, self.ylim, self.zlim = .25, .25, 1.5
        self.ax.set_xlabel('X')
        self.ax.set_xlim3d([-self.xlim, self.xlim])
        self.ax.set_ylabel('Y')
        self.ax.set_ylim3d([-self.ylim, self.ylim])
        self.ax.set_zlabel('Z')
        self.ax.set_zlim3d([0, -self.zlim])
        
        # body of quad
        self.arm13, = self.ax.plot3D([],[],[], zdir='z', color='green', linewidth=2, antialiased=False)
        self.arm24, = self.ax.plot3D([],[],[], zdir='z', color='green', linewidth=2, antialiased=False)
        self.shadow, = self.ax.plot3D([],[],[], zdir='z', color='red', marker='.', antialiased=False)
        self.hub, = self.ax.plot3D([],[],[], zdir='z', color='blue', marker='o', markersize=6, antialiased=False)
        
        self.fig.canvas.mpl_connect('key_press_event', self.keypress_routine)
        
        # map to world frame and plot
        wHb = np.concatenate((R_linear, position), axis=1)
        drone_fig = np.matmul(wHb, self.drone_body)
        self.arm13.set_data(drone_fig[0,[0, 2]], drone_fig[1,[0, 2]])
        self.arm13.set_3d_properties(drone_fig[2, [0,2]])
        self.arm24.set_data(drone_fig[0,[1,3]], drone_fig[1,[1,3]])
        self.arm24.set_3d_properties(drone_fig[2,[1,3]])
        self.hub.set_data(drone_fig[0,5], drone_fig[1,5])
        self.hub.set_3d_properties(drone_fig[2,5])
        self.shadow.set_data(drone_fig[0,5], drone_fig[1,5])
        self.shadow.set_3d_properties(0)

    # map the currend position to the world frame and plot
    def update_plot(self, R_linear, position): 
        self.position = position       
        wHb = np.concatenate((R_linear, position), axis=1)
        drone_fig = np.matmul(wHb, self.drone_body)
        self.arm13.set_data(drone_fig[0,[0, 2]], drone_fig[1,[0, 2]])
        self.arm13.set_3d_properties(drone_fig[2, [0,2]])
        self.arm24.set_data(drone_fig[0,[1,3]], drone_fig[1,[1,3]])
        self.arm24.set_3d_properties(drone_fig[2,[1,3]])
        self.hub.set_data(drone_fig[0,5], drone_fig[1,5])
        self.hub.set_3d_properties(drone_fig[2,5])
        self.shadow.set_data(drone_fig[0,5], drone_fig[1,5])
        self.shadow.set_3d_properties(0)
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
class pid_plotter:

    def __init__(self):
        # self.fig = plt.figure(figsize=(10, 6))

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
        self.z_plt.set_title("z[m]")
        plt.grid()

    def update_plot(self, time, state):        
        self.x_plt.plot(time, state[0], 'b.')
        self.y_plt.plot(time, state[1], 'b.')
        # here -ve z axis in converted to +ve
        self.z_plt.plot(time, -state[2], 'b.')

        self.phi_plt.plot(time, rad2deg(state[6]), 'b.')
        self.theta_plt.plot(time, rad2deg(state[7]), 'b.')
        self.psi_plt.plot(time, rad2deg(state[8]), 'b.')
