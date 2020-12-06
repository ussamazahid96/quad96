import numpy as np


__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"

# utility function for converting degrees to rad, body frame to earth (world) frame, sin, cos, tan

def rad2deg(rad):
    ret = rad*(180/np.pi)
    return ret

def deg2rad(deg):
    return deg*(np.pi/180)

def S(x):
    return np.sin(x)

def C(x):
    return np.cos(x)

def T(x):
    return np.tan(x)

def body2earth(angles):

    sin_phi   = S(angles[0])
    sin_theta = S(angles[1])
    sin_psi   = S(angles[2])

    cos_phi   = C(angles[0])
    cos_theta = C(angles[1])
    cos_psi   = C(angles[2]) 

    tan_theta = T(angles[1])

    R_linear =[[ (cos_psi * cos_theta) , (cos_psi * sin_theta * sin_phi - sin_psi * cos_phi) , (cos_psi * sin_theta * cos_phi + sin_psi * sin_phi) ],
               [ (sin_psi * cos_theta) , (sin_psi * sin_theta * sin_phi + cos_psi * cos_phi) , (sin_psi * sin_theta * cos_phi - cos_psi * sin_phi) ],
               [          (-sin_theta) ,                               (cos_theta * sin_phi) ,                               (cos_theta * cos_phi) ]]
    
    R_angular = [[1., sin_phi*tan_theta, cos_phi*tan_theta],
                 [0.,          cos_phi,         -sin_phi],
                 [0., sin_phi/cos_theta, cos_phi/cos_theta]]

    return np.array(R_linear), np.array(R_angular)