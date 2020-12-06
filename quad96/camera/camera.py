try:
    import os
    if os.environ['BOARD'] == 'Ultra96':
        os.environ["DISPLAY"] = ":1"
except KeyError:
    pass
import cv2
import time
import signal
import threading
import numpy as np
import multiprocessing

__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"

# this file contains definations for camera controllers

font = cv2.FONT_HERSHEY_SIMPLEX

class SimpleCameraController:
    def __init__(self, state_ptr=None, scale=1):
        print("[SimpleCameraController] Attaching Camera Handle for adding Disturbance")        
        self.running = True
        self.scale = scale
        self.state_ptr = state_ptr
        self.camera = cv2.VideoCapture(0)
        
        (_, self.frame) = self.camera.read()
        h, w = self.frame.shape[:2]
        self.frame = cv2.resize(self.frame, (w//2, h//2))
        # w = 320, h = 240
        self.ih, self.iw = self.frame.shape[:2]

        self.dead_radius = 60//2
        self.kernelopen  = np.ones((5,5))
        self.kernelclose = np.ones((10,10)) 

        # hsv value for the object of interest    
        self.low  = np.array([100, 130,   0])
        self.high = np.array([120, 255, 255])

        self.camera_handle = threading.Thread(target=self.run)
        self.camera_handle.start()

    # convert xy to polar coordinates
    def xy2pol(self, x,y):
        dis = np.sqrt(x**2 + y**2)
        angle = np.arctan2(y, x)*180/np.pi
        return dis, angle

    # get frame and apply a morphological operation
    def get_morph_frame(self):
        (_, frame) = self.camera.read()
        frame = cv2.resize(frame, (self.iw, self.ih))        
        self.frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.low, self.high)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernelopen)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernelclose)
        return mask

    # get polar coordinates for a contour having the max area
    def get_max_cont_polar(self, conts):
        c = max(conts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        cx, cy = x+w//2, y+h//2
        cv2.rectangle(self.frame,(x,y),(x+w,y+h), (0,255,0), 2)
        cv2.circle(self.frame, (cx, cy), 5, (0, 0, 255), 2)
        cx = cx - self.iw//2
        cy = self.ih//2 - cy
        dis, angle = self.xy2pol(cx, cy)
        return dis, angle

    # main thread function
    def run(self):
        while self.running:
            temp_frame = self.get_morph_frame()            
            conts = cv2.findContours(temp_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            conts = conts[0] if len(conts) == 2 else conts[1]
            if len(conts) != 0:
                dis, angle = self.get_max_cont_polar(conts)
                if (dis > self.dead_radius) and (-45 <= angle < 45):
                    # print("Right")
                    self.state_ptr[1] -= self.scale
                elif (dis > self.dead_radius) and (45 <= angle < 90+45):
                    # print("Up")
                    self.state_ptr[2] -= self.scale
                elif (dis > self.dead_radius) and (angle >= 90+45 or angle < -180+45):
                    # print("Left")
                    self.state_ptr[1] += self.scale
                elif (dis > self.dead_radius) and (-180+45 <= angle < -45):
                    # print("down")
                    self.state_ptr[2] += self.scale
            cv2.circle(self.frame, (self.iw//2, self.ih//2), self.dead_radius, (255, 0, 0), 2)

    def close(self):
        if self.running:
            self.running = False
            self.camera_handle.join()
            self.camera.release()
            cv2.destroyAllWindows()
            print("[SimpleCameraController] Camera Handle Closed...")


class HandGestureCameraController:
    def __init__(self, scale=1):
        print("[HandGestureCameraController] Attaching Camera Handle for position control")        
        self.running = True
        self.scale = scale
        self.x_dot = 0.
        self.y_dot = 0.
        self.z_dot = 0.
        self.yaw_rate = 0.
        self.camera = cv2.VideoCapture(0)
        
        (_, self.frame) = self.camera.read()
        h, w = self.frame.shape[:2]
        self.frame = cv2.resize(self.frame, (w//2, h//2))
        # w = 320, h = 240
        self.ih, self.iw = self.frame.shape[:2]

        filt_size = 5
        self.kernelopen  = np.ones((filt_size,filt_size))
        self.kernelclose = np.ones((filt_size,filt_size))  
        # if hand/object of interest is inside this radius, do nothing
        self.dead_radius = 60//2

        # hsv value for the object of interest    
        self.low  = np.array([103, 174,  85])
        self.high = np.array([117, 255, 255])

        # calibrating background for background subtraction
        self.calib()

        # Main loop
        self.camera_handle = threading.Thread(target=self.run)
        self.camera_handle.start()

    # calibrate background for the first 60 frames
    def calib(self):
        i = 0
        self.back = None
        while i < 60:    
            frame = self.get_morph_frame()
            if self.back is None:
                self.back = frame.copy()
                self.back = np.float32(self.back)
            else:                
                cv2.accumulateWeighted(frame.copy(), self.back, 0.1)
            i+=1

            back=cv2.convertScaleAbs(self.back)
            img=cv2.absdiff(back,frame)
            
            cv2.imshow('Calibrating background...',img)
            waitkey = cv2.waitKey(1) & 0xFF 
            if waitkey == ord("q"):
                self.close()
                exit(0)

        self.back = cv2.convertScaleAbs(self.back)
        cv2.destroyAllWindows()        

    # convert xy to polar coordinates
    def xy2pol(self, x,y):
        dis = np.sqrt(x**2 + y**2)
        angle = np.arctan2(y, x)*180/np.pi
        return dis, angle

    # get frame and apply a morphological operation
    def get_morph_frame(self):
        (_, frame) = self.camera.read()
        frame = cv2.resize(frame, (self.iw, self.ih))        
        self.frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.low, self.high)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernelopen)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernelclose)
        return mask


    # function to count the no. of fingers
    def defects(self, c):
        cnt = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)
        hull = cv2.convexHull(cnt, returnPoints=False)
        fcnt = 0 
        try:        
            defects = cv2.convexityDefects(cnt,hull)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])

                    a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                    if angle <= np.pi / 3: 
                        fcnt += 1
                        cv2.circle(self.frame, far, 4, [0, 0, 255], -1)
                if fcnt > 0:
                    fcnt += 1
            cv2.putText(self.frame, str(fcnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        except Exception as e:
            pass
        return fcnt

    # get polar coordinates for a contour having the max area
    def get_max_cont_polar(self, conts):
        c = max(conts, key=lambda x: cv2.contourArea(x))
        # cv2.drawContours(self.frame, [c], -1, (255,255,0), 2)
        hull = cv2.convexHull(c)
        cv2.drawContours(self.frame, [hull], -1, (0, 255,255), 2)
        
        fcnt = self.defects(c)
        
        x,y,w,h = cv2.boundingRect(c)
        # cv2.rectangle(self.frame,(x,y),(x+w,y+h), (0,255,0), 2)
        cx, cy = x+w//2, y+h//2
        cv2.circle(self.frame, (cx, cy), 5, (0, 255, 0), 2)
        cx = cx - self.iw//2
        cy = self.ih//2 - cy
        dis, angle = self.xy2pol(cx, cy)
        return cx, cy, fcnt, dis, angle

    def run(self):
        while self.running:
            temp_frame = self.get_morph_frame()
            temp_frame = cv2.absdiff(self.back, temp_frame)            
            conts = cv2.findContours(temp_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            conts = conts[0] if len(conts) == 2 else conts[1]
            cv2.circle(self.frame, (self.iw//2, self.ih//2), self.dead_radius, (255, 0, 0), 2)
            if len(conts) != 0:
                cx, cy, fcnt, dis, angle = self.get_max_cont_polar(conts)

                # if fingers detected
                if fcnt > 0:

                    if (dis > self.dead_radius) and (-45 <= angle < 45):
                        # print("Right")
                        self.x_dot = 0.
                        self.y_dot = -(cx/(self.iw//2)) #-self.scale
                        self.z_dot = 0.
                        self.yaw_rate = 0.   
                    elif (dis > self.dead_radius) and (45 <= angle < 90+45):
                        # print("Up")
                        self.x_dot = 0.
                        self.y_dot = 0.
                        self.z_dot = -(cy/(self.ih//2)) #self.scale
                        self.yaw_rate = 0.   
                    elif (dis > self.dead_radius) and (angle >= 90+45 or angle < -180+45):
                        # print("Left")
                        self.x_dot = 0.
                        self.y_dot = -(cx/(self.iw//2)) #self.scale
                        self.z_dot = 0.
                        self.yaw_rate = 0.   
                    elif (dis > self.dead_radius) and (-180+45 <= angle < -45):
                        # print("Down")
                        self.x_dot = 0.
                        self.y_dot = 0.
                        self.z_dot = -(cy/(self.ih//2)) #-self.scale
                        self.yaw_rate = 0.   
                    else:
                        self.x_dot = 0.
                        self.y_dot = 0.
                        self.z_dot = 0.
                        self.yaw_rate = 0.   
                
                # if no fingers detected e.g. fist
                elif fcnt == 0:

                    if (dis > self.dead_radius) and (-45 <= angle < 45):
                        # print("Right")
                        self.x_dot = 0.
                        self.z_dot = 0.
                        self.y_dot = 0.
                        # self.yaw_rate = 50
                    elif (dis > self.dead_radius) and (45 <= angle < 90+45):
                        # print("Up")
                        self.x_dot = cy/(self.ih//2) #self.scale
                        self.y_dot = 0.
                        self.z_dot = 0.
                        self.yaw_rate = 0.   
                    elif (dis > self.dead_radius) and (angle >= 90+45 or angle < -180+45):
                        # print("Left")
                        self.x_dot = 0.
                        self.z_dot = 0.
                        self.y_dot = 0.
                        # self.yaw_rate = -50
                    elif (dis > self.dead_radius) and (-180+45 <= angle < -45):
                        # print("Down")
                        self.x_dot = cy/(self.ih//2)#-self.scale
                        self.y_dot = 0.
                        self.z_dot = 0.
                        self.yaw_rate = 0.   
                    else:
                        self.x_dot = 0.
                        self.y_dot = 0.
                        self.z_dot = 0.
                        self.yaw_rate = 0.   
  
            else:
                self.x_dot = 0.
                self.y_dot = 0.
                self.z_dot = 0.
                self.yaw_rate = 0.  
            

    def close(self):
        if self.running:
            self.running = False
            self.camera_handle.join()
            self.camera.release()
            cv2.destroyAllWindows()
            print("[HandGestureCameraController] Camera Handle Closed...")

if __name__=='__main__':
    cc = HandGestureCameraController()
    while True:
        print("x_dot = {:.2f}, y_dot = {:.2f}, z_dot = {:.2f}, yaw_rate = {:.2f}\r".format(\
                   cc.x_dot,     cc.y_dot,     cc.z_dot,     cc.yaw_rate), end="")        
        cv2.imshow("Video Feed", cc.frame)
        waitkey = cv2.waitKey(1) & 0xFF 
        if waitkey == ord("q"):
            cc.close()
            exit(0)        


