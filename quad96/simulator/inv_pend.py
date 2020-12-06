import numpy as np
from simulator.utils import deg2rad
from simulator.plot import inv_pend_plotter


__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"

# class defination for the inverted pendulum and the state update function

class InvPend:

    def __init__(self):

        self.length = 0.5
        self.g = 9.81
        self.running = True
        self.max_angle = deg2rad(45)
        self.max_tilt = np.abs(self.length * np.sin(self.max_angle))

        self.offset = np.array([0., 0., 0.])
        self.doffset = np.zeros(3,)
        self.quad_acc_x, self.quad_acc_y = 0., 0.

        self.t = 0
        self.dt = 1./500
        self.X = np.zeros(6,)
        self.dX = np.zeros(6,)

        init_xy = np.array([min(0., self.length), min(0., self.length)])
        self.set_pose(init_xy)

        # self.plotter = inv_pend_plotter(self.X, self.offset, self.max_tilt)


    def set_pose(self, new_xy):
        self.X[0:2] = np.clip(new_xy, -self.length, self.length)
        self.X[2]   = -np.sqrt(self.length**2 - np.clip(new_xy[0]**2 + new_xy[1]**2, 0, self.length**2))


    def step(self):
        self.dX[:3] = self.X[3:]
        self.dX[3:] = self.solve_lagrangian()

    # from paper [2]
    def solve_lagrangian(self):
        d2v = np.zeros(3,)
        a, b, c = self.X[0], self.X[1], self.X[2]
        da, db  = self.X[3], self.X[4]

        if np.abs(c) < 1e-1:
            return d2v

        etta_zeta = np.dot( np.array([ [b, c**2*b] , [a, c**2*a] ]),  np.array([ (a*da + b*db)**2 , (da**2 + db**2) ]) )

        etta_zeta = etta_zeta + self.g * c**3 * np.array( [b+c*self.quad_acc_y, a+c*self.quad_acc_x] )

        etta_zeta_2d = np.array( [ [etta_zeta[0], etta_zeta[1]], [etta_zeta[1], etta_zeta[0]] ] ) 

        d2v[0:2] = 1./(self.length**2 * c**4) * np.dot(np.array([a*b, (a**2-self.length**2)]), etta_zeta_2d)

        return d2v

    # reset the pend state
    def reset(self, pose=np.zeros(2,)):
        self.t = 0
        self.X[:] = np.zeros(6,)
        self.dX[:] = np.zeros(6,)
        self.set_pose(pose)
        self.running = True

    # update the pend state
    def update_state(self, acc_x=None, acc_y=None):
        if self.running:
            if acc_x is not None:
                self.quad_acc_x = acc_x
            if acc_y is not None:
                self.quad_acc_y = acc_y

            self.step()
            self.X += self.dX*self.dt
            self.X[2] = self.update_z()
            self.t += self.dt

            # self.doffset[0] += self.quad_acc_x*self.dt
            # self.doffset[1] += self.quad_acc_y*self.dt
            # self.offset += self.doffset*self.dt
            # self.plotter.update_plot(self.X, self.offset)
            
            if self.X[0]**2 + self.X[1]**2 >= self.max_tilt**2:
                self.running = False        
        
        return self.X

    # set z from given xy and L
    def update_z(self):
        new_z = -np.sqrt(self.length**2 - np.clip(self.X[0]**2 + self.X[1]**2, 0, self.length**2))
        return new_z


if __name__=='__main__':
    pend = InvPend()
    while pend.running:
        pend.update_state()