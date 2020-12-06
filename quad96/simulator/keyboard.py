try:
    import os
    if os.environ['BOARD'] == 'Ultra96':
        os.environ["DISPLAY"] = ":0"
except KeyError:
    pass
import sys
import tty
import termios
import numpy as np
from simulator.utils import deg2rad
from pynput.keyboard import Key, Listener, Controller


__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"

# this file contains the definations for the keyboard controller used position control 

class KeyboardController:
    def __init__(self, args, input_ptr=None, scale=0.2):
        print("[Keyboard] Attaching Keyboard Handle")
        self.args = args
        self.factor = 2
        self.scale = scale
        self.input_ptr = input_ptr
        self.orig_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin)
        self.keyboard_handle = Listener(on_press=self.on_press, on_release=self.on_release)
        self.keyboard_handle.start()

    def on_press(self, key):
        try:
            if key.char == 'w':
                self.input_ptr[2] -= self.scale
            elif key.char == 's':
                self.input_ptr[2] += self.scale
            
            if key.char == 'a':
                self.input_ptr[self.yaw_idx]  -= deg2rad(90)
            elif key.char == 'd':
                self.input_ptr[self.yaw_idx]  += deg2rad(90)
            
            if key.char == 'i':
                self.input_ptr[0] += self.scale
            elif key.char == 'k':
                self.input_ptr[0] -= self.scale
            
            if key.char == 'j':
                self.input_ptr[1] += self.scale
            elif key.char == 'l':
                self.input_ptr[1] -= self.scale
        
        except AttributeError:
            if key == Key.esc:
                return False

    def on_release(self, key):
        try:
            if key.char == 'a' or key.char == 'd':
                self.input_ptr[self.yaw_idx] = 0
        except AttributeError:
            pass

    def close(self):
        Controller().press(Key.esc)
        Controller().release(Key.esc)
        self.keyboard_handle.join()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.orig_settings)
        print("[Keyboard] Keyboard handle closed...") 

# the function which directs the camera output to the drone when holding the `c` key

class KeyboardController_CD:
    def __init__(self, args, input_ptr=None, scale=0.2, camera=None):
        print("[Keyboard_CD] Attaching Keyboard Handle for Camera Direct")

        self.args = args
        self.factor = 2
        self.scale = scale
        self.camera = camera
        self.input_ptr = input_ptr
        self.input_ptr_copy = np.copy(self.input_ptr)
        self.yaw_idx = 11 if len(self.input_ptr) == 12 else 3
        self.orig_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin)
        self.keyboard_handle = Listener(on_press=self.on_press, on_release=self.on_release)
        self.keyboard_handle.start()

    def on_press(self, key):
        try:

            if key.char == 'c':
                self.input_ptr[0] = self.factor*self.camera.x_dot
                self.input_ptr[1] = self.factor*self.camera.y_dot
                self.input_ptr[2] = self.factor*self.camera.z_dot + self.input_ptr_copy[2]
        
        except AttributeError:
            if key == Key.esc:
                return False

    def on_release(self, key):
        try:
            if key.char == 'a' or key.char == 'd':
                self.input_ptr[self.yaw_idx] = 0
        except AttributeError:
            pass

    def close(self):
        Controller().press(Key.esc)
        Controller().release(Key.esc)
        self.keyboard_handle.join()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.orig_settings)
        print("[Keyboard_CD] Keyboard handle closed...") 
