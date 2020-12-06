import sys
import tty
import time
import signal
import struct
import termios
import argparse
import threading
import bluepy as b
import numpy as np
try:
    import os
    if os.environ['BOARD'] == 'Ultra96':
        os.environ["DISPLAY"] = ":0"
except KeyError:
    pass
from pynput.keyboard import Key, Listener, Controller

try:
    from .bledrv import BLEDriver
except:
    from bledrv import BLEDriver

import cflib.crtp as crtp
from cflib.crazyflie import Crazyflie
from cflib.crtp.crtpstack import CRTPPort
from cflib.crtp.crtpstack import CRTPPacket

__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"

orig_settings = termios.tcgetattr(sys.stdin)


HOVER = 0
ALT_HOLD = 19
START_PROP_TEST = 56

# the main Crazyflie class which combines the crazyflie with the given driver e.g. bluetooth/Radio 
# with keyboard controller and provides functions to send commands e.g. take off land set a parameter etc

class CrazyFlie():
    def __init__(self, ble=False, sim_vel=np.array([0.,0.,0.])):
        print("[CrazyFlie] Attaching CrazyFlie Plug with Keyboard handle")

        if ble:        
            link = BLEDriver()
            inter = link.scan_interface()[0][0]
            # inter = "e0:c3:b3:86:a6:13"
            link.connect(inter)
            self._cf = Crazyflie(link=link, rw_cache="./")
            self.rate = 0.05
            print("[CrazyFlie] Connected to Crazyflie on bluetooth {}.".format(inter))
        else:
            crtp.init_drivers()
            self.inter = crtp.scan_interfaces()
            self._cf = Crazyflie(rw_cache="./")
            self._cf.open_link(self.inter[0][0])
            time.sleep(1) # wait for a while to let crazyflie fetch the TOC
            self.rate = 0.01
            print("[CrazyFlie] Connected to Crazyflie on radio {}.".format(self.inter))
        
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.hover_thrust = 36330
        self.tilt_angle = 10

        self.x_dot = 0.0
        self.y_dot = 0.0
        self.z_dot = 0.0
        self.yaw_rate = 0.0
        self.step = 0.1
        self.max_xyz_speed = 0.4
        self.max_yaw_rate = 90.

        self.x, self.y, self.z = 0,0,0.5
        
        self.running = True

        self.roll_calib_offset = 0
        self.pitch_calib_offset = 0

        signal.signal(signal.SIGINT, self.interrupt_handle)
        # tty.setcbreak(sys.stdin)
        self.keyboard_handle = Listener(on_press=self.on_press, on_release=self.on_release)
        self.keyboard_handle.start()

        # unlocking thrust protection
        self._cf.commander.send_setpoint(0, 0, 0, 0)
        self.sim_vel = sim_vel

        # if self.sim_vel is not None:
        self.hover_thread_flag = True
        self.hover_thread = threading.Thread(target=self.hover_threaded)
        # self.set_param(START_PROP_TEST)

    # what to do on a key press
    def on_press(self, key):
        try: 
            if key.char == 'w':
                self.z_dot = self.max_xyz_speed
                self.hover_thrust += 500
                self.z += 0.5
            elif key.char == 's':
                self.z_dot = -self.max_xyz_speed
                self.hover_thrust -= 500
                self.z -= 0.5

            if key.char == 'a':
                self.yaw -= 10
                self.yaw_rate = -self.max_yaw_rate
            elif key.char == 'd':
                self.yaw += 10
                self.yaw_rate = self.max_yaw_rate

            if key.char == 'i':
                self.pitch = self.tilt_angle
                self.x_dot = self.max_xyz_speed
                self.x += 0.1
            elif key.char == 'k':
                self.pitch = -self.tilt_angle
                self.x_dot = -self.max_xyz_speed
                self.x -= 0.1

            if key.char == 'j':
                self.roll = -self.tilt_angle
                self.y_dot = self.max_xyz_speed
                self.y += 0.5
            elif key.char == 'l':
                self.roll = self.tilt_angle
                self.y_dot = -self.max_xyz_speed
                self.y -= 0.5

            if key.char == 'c':
                self.x_dot = self.sim_vel[0]
                self.y_dot = self.sim_vel[1]
                self.z_dot = -self.sim_vel[2]

            if key.char == 'r':
                self.running = True

        except AttributeError:
            if key == Key.up:
                self.pitch_calib_offset = min(self.pitch_calib_offset+1, 5)
            elif key == Key.down:
                self.pitch_calib_offset = max(self.pitch_calib_offset-1, -5)

            if key == Key.left:
                self.roll_calib_offset = min(self.roll_calib_offset+1, 5)
            elif key == Key.right:
                self.roll_calib_offset = max(self.roll_calib_offset-1, -5)

            if key == Key.shift:
                self.tilt_angle = min(self.tilt_angle + 5, 45)
                self.max_xyz_speed = min(self.max_xyz_speed + self.step, 1)
                self.max_yaw_rate = min(self.max_yaw_rate + 5, 90)
            elif key == Key.ctrl:
                self.tilt_angle = max(self.tilt_angle - 5, 5)
                self.max_xyz_speed = max(self.max_xyz_speed - self.step, -1)
                self.max_yaw_rate = max(self.max_yaw_rate - 5, -90)

            # stop the crazyflie (to be used in case if Crazyflie is getting out of control)
            if key == Key.space:
                self.running = False
                self._cf.commander.send_stop_setpoint()

    # when a key is released
    def on_release(self, key):
        try:
            if key.char == 'w' or key.char == 's':
                self.z_dot = 0.
            elif key.char == 'a' or key.char == 'd':
                self.yaw_rate = 0.
            elif key.char == 'i' or key.char == 'k':
                self.pitch = 0.
                self.x_dot = 0.
            elif key.char == 'j' or key.char == 'l':
                self.roll = 0.
                self.y_dot = 0.
            if key.char == 'c':
                self.x_dot = 0.
                self.y_dot = 0.
                self.z_dot = 0.

        except AttributeError:
            if key == Key.esc:
                return False

    def interrupt_handle(self, signal, frame):
        print('\n[CrazyFlie] Stopping')
        self.close()
        exit(0)

    def close(self):
        self.running = False
        self._cf.commander.send_stop_setpoint()        
        self._cf.close_link()        
        Controller().press(Key.esc)
        Controller().release(Key.esc)
        self.keyboard_handle.join()
        self.hover_thread_flag = False
        self.hover_thread.join()
        # termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)
        print("[CrazyFlie] Closed...")

    def print_rpyt(self):
        print("roll = {:.2f}, pitch = {:.2f}, yaw = {:.2f},   thrust = {:.2f}, SA = {:.2f}\r".format(\
                   self.roll,     self.pitch,     self.yaw, self.hover_thrust, self.tilt_angle), end="")

    def print_vx_vy_vz_yr(self):
        print("x_dot = {:.2f}, y_dot = {:.2f}, z_dot = {:.2f}, yaw_rate = {:.2f},  Max Speed = {:.2f}\r".format(\
                   self.x_dot,     self.y_dot,     self.z_dot,     self.yaw_rate, self.max_xyz_speed), end="")

    def takeoff(self):
        for i in range(int(1*20)):
            # self.print_vx_vy_vz_yr()
            self._cf.commander.send_velocity_world_setpoint(0, 0, 0.1, 0)
            # self._cf.commander.send_setpoint(self.roll + self.roll_calib_offset, \
            #                              self.pitch + self.pitch_calib_offset, \
            #                              self.yaw, 10000)
            time.sleep(0.01)        
        # self._cf.commander.send_setpoint(0, 0, 0, self.hover_thrust)

    def hover(self):
        self.set_param(ALT_HOLD)
        while self.running:
            self.print_vx_vy_vz_yr()    
            # self.print_rpyt()
            self._cf.commander.send_velocity_world_setpoint(self.x_dot, self.y_dot, self.z_dot, self.yaw_rate)
            # self._cf.commander.send_setpoint(self.roll + self.roll_calib_offset, \
            #                              self.pitch + self.pitch_calib_offset, \
            #                              self.yaw, self.hover_thrust)
            time.sleep(self.rate)


    def hover_threaded(self):
        self.set_param(ALT_HOLD)
        while self.hover_thread_flag:
            while self.running:
                self.print_vx_vy_vz_yr()    
                # self.print_rpyt()
                self._cf.commander.send_velocity_world_setpoint(self.x_dot, self.y_dot, self.z_dot, self.yaw_rate)
                # self._cf.commander.send_setpoint(self.roll + self.roll_calib_offset, \
                #                              self.pitch + self.pitch_calib_offset, \
                #                              self.yaw, self.hover_thrust)
                time.sleep(self.rate)
            time.sleep(self.rate)

    def land(self):
        while self.hover_thrust != 0:
            self.hover_thrust -= 2000
            if self.hover_thrust < 0:
                self.hover_thrust = 0
            self.print_rpyt()
            self._cf.commander.send_setpoint(0, 0, 0, self.hover_thrust)
            time.sleep(0.4)

    def set_param(self, value):
        WRITE_CHANNEL = 2
        pk = CRTPPacket()
        pk.set_header(CRTPPort.PARAM, WRITE_CHANNEL)
        pk.data = struct.pack('<H', value)
        self._cf.send_packet(pk)



def get_args():
    parser = argparse.ArgumentParser(description="Crazyflie bluetooth/radio test.")
    parser.add_argument('--ble', action='store_true', help="Use bluetooth")
    return parser.parse_args()


def main():
    args = get_args()
    try:
        cf = CrazyFlie(ble=args.ble)
        # cf.takeoff()
        # cf.hover()
        # cf.close()
        cf.hover_thread.start()
        while cf.hover_thread_flag:
            pass
    except Exception as e:
        print("\n", e)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)



if __name__ == "__main__":
    main()
