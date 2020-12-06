try:
    import os
    if os.environ['BOARD'] == 'Ultra96':
        os.environ["DISPLAY"] = ":1"
except KeyError:
    pass
import cv2
import numpy as np


__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"

# class defination to find the hsv values of object of interest

class hsvfinder:
    def __init__(self):

        cv2.namedWindow('Image')
        self.camera = cv2.VideoCapture(0)
        
        (_, self.frame) = self.camera.read()
        h, w = self.frame.shape[:2]
        self.frame = cv2.resize(self.frame, (w//2, h//2))
        # w = 320, h = 240
        self.ih, self.iw = self.frame.shape[:2]


        cv2.createTrackbar('Hmin', 'Image', 0, 179, self.nothing)
        cv2.createTrackbar('Smin', 'Image', 0, 255, self.nothing)
        cv2.createTrackbar('Vmin', 'Image', 0, 255, self.nothing)
        cv2.createTrackbar('Hmax', 'Image', 0, 179, self.nothing)
        cv2.createTrackbar('Smax', 'Image', 0, 255, self.nothing)
        cv2.createTrackbar('Vmax', 'Image', 0, 255, self.nothing)

        cv2.setTrackbarPos('Hmax', 'Image', 179)
        cv2.setTrackbarPos('Smax', 'Image', 255)
        cv2.setTrackbarPos('Vmax', 'Image', 255)

        self.Hmin  = self.Smin  = self.Vmin  = 0, 0, 0
        self.Hmax  = self.Smax  = self.Vmax  = 0, 0, 0
        self.pHmin = self.pSmin = self.pVmin = 0, 0, 0
        self.pHmax = self.pSmax = self.pVmax = 0, 0, 0



    def nothing(self, x):
        pass


    def run(self):

        while True:
            (_, frame) = self.camera.read()
            frame = cv2.resize(frame, (self.iw, self.ih))                
            frame = cv2.flip(frame, 1)

            self.Hmin = cv2.getTrackbarPos('Hmin', 'Image')
            self.Smin = cv2.getTrackbarPos('Smin', 'Image')
            self.Vmin = cv2.getTrackbarPos('Vmin', 'Image')

            self.Hmax = cv2.getTrackbarPos('Hmax', 'Image')
            self.Smax = cv2.getTrackbarPos('Smax', 'Image')
            self.Vmax = cv2.getTrackbarPos('Vmax', 'Image')

            lower = np.array([self.Hmin, self.Smin, self.Vmin])
            upper = np.array([self.Hmax, self.Smax, self.Vmax])

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            output = cv2.bitwise_and(frame, frame, mask=mask)

            if ( (self.pHmin != self.Hmin) | (self.pSmin != self.Smin) | (self.pVmin != self.Vmin) | \
                 (self.pHmax != self.Hmax) | (self.pSmax != self.Smax) | (self.pVmax != self.Vmax) ):
                print("self.low = np.array([%d, %d, %d])" % (self.Hmin, self.Smin, self.Vmin))
                print("self.high = np.array([%d, %d, %d])" % (self.Hmax, self.Smax, self.Vmax))

                self.pHmin = self.Hmin
                self.pSmin = self.Smin
                self.pVmin = self.Vmin
                self.pHmax = self.Hmax
                self.pSmax = self.Smax
                self.pVmax = self.Vmax

            cv2.imshow('Image', output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.camera.release()
                cv2.destroyAllWindows()
                break

if __name__=='__main__':
    hsv = hsvfinder()
    hsv.run()


