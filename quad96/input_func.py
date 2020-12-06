# this file is used by vitis ai for the calibration

import numpy as np

data = np.load('./deploy/calib_data.npz')['data']

batch_size=1

def calib_input(iter):

    calib_data = data[iter*batch_size:(iter+1)*batch_size]

    return {'input_data_actor_local': calib_data.reshape(1,1,1,-1)}

if __name__ == '__main__':

    arr = calib_input(1)
    arr = arr["input_data"]
    print(arr.shape)
