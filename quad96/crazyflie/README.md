# Simple bluetooth driver for crazyflie to work with keyboard

Use `w, s` for up/down, `a,d` for yawrate, `i, k` for forward/backward, `j, l` for left/right, `space, r` for pause/resume, `left shift, left ctrl` for increasing/decreasing the speed. When turning on the crazyflie, make sure that the forward direction is pointing away from you (i.e. the back of crazyflie should be facing you, so pressing `w` will make it move away in the forward direction).

Install the crazyflie [libraries](https://github.com/bitcraze/crazyflie-lib-python) and [client](https://github.com/bitcraze/crazyflie-clients-python)
```
pip3 install bluepy pynput

# for bluetooth
sudo python3 crazyflie_class.py --ble

# for crazyradio
sudo python3 crazyflie_class.py
```