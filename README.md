# quad96

### For Inverted Pendulum on Quad switch to `flying_pend` branch

Quadcopter control using Deep Reinforcement Learning and Hand Gestures on ultra96 using DPU

Trained using twin delayed deep deterministic policy gradient (TD3), rank based prioritized experience replay and feedback

## Video Demo

### Simulation Demo

[![IMAGE ALT TEXT](https://yt-embed.herokuapp.com/embed?v=cKm3FKpj5zY)](https://youtu.be/cKm3FKpj5zY?t=0s "Simulation Demo")


### Crazyflie Demo

[![IMAGE ALT TEXT](https://yt-embed.herokuapp.com/embed?v=KBOMOkM78mY)](https://youtu.be/KBOMOkM78mY?t=0s "Crazyflie Demo")


## Hardware Requirements

-> [Ultra96-V1/V2](https://www.96boards.org/product/ultra96/)

-> USB Keyboard

-> USB Hub (Optional)

-> USB Camera (Optional)

-> [Crazyflie](https://www.bitcraze.io/products/crazyflie-2-1/) (Optional)

-> [CrazyRadio PA](https://www.bitcraze.io/products/crazyradio-pa/) (Optional)


## Setting up PC having Ubuntu 20.04

Clone the [Vitis-Ai](https://github.com/Xilinx/Vitis-AI) repo.
```
git clone --recurse-submodules https://github.com/Xilinx/Vitis-AI 
cd vitis-ai/
git checkout a5a4a48839ce5dee9542bdbcdda07d0ea98d88d1
```

Make sure to have docker installed. Pull the vitis-ai docker image and run it.
```
docker pull xilinx/vitis-ai:1.2.82
./docker_run.sh xilinx/vitis-ai:1.2.82

# then in docker install the dependencies
conda activate vitis-ai-tensorflow
pip install keras==2.2.5 pycairo gym
git clone https://github.com/ussamazahid96/quad96.git

# run compile for ultra96 DPU
cd quad96/quad96/deploy/host
./compile

# start training
cd ../../
./quad_run.sh
```

Take a look inside `quad_run.sh`. You can also pass the argument `--render_window` to training and eval commands and start the unity simulator with ip in `quad96/env_window/env_ip.txt` with your selected port (otherwise default port is `4567`) `127.0.0.1:<your port>` to see what the quad is doing while training/eval.

## Setting up Ultra96 V1/V2 having pynq v2.5 

Connect to wifi (take a look at `connect_wifi.py` on how to do it), clone 1uad96 repo, and install the dependencies

```
sudo apt update && sudo apt upgrade -y && sudo apt install -y vnc4server python3-tk
sudo pip3 install bluepy pynput
git clone https://github.com/ussamazahid96/quad96.git --depth 1
cd quad96
./vnc.sh # (you can edit this file to set your  desired resolution and set password for the first time configuration)
cd ../
```

Install [DPU-PYNQ](https://github.com/Xilinx/DPU-PYNQ)

```
git clone https://github.com/Xilinx/DPU-PYNQ.git
cd DPU-PYNQ
git checkout 5af296d67b1a90ce09447d7466d669c0b0e41f2f
cd upgrade
su # switch to superuser
make
pip3 install pynq-dpu
cd $PYNQ_JUPYTER_NOTEBOOKS
pynq get-notebooks pynq-dpu -p .
``` 

Install Crazyflie libraries (required for using crazyflie) and client (optional)

```
git clone https://github.com/bitcraze/crazyflie-lib-python.git
cd crazyflie-lib-python/
git checkout 12f2a49427e6342459065410d0fef154abe04a37
./setup_linux.sh

# client is optional
sudo apt install python3-pyqt5 libzmq3-dev
sudo pip3 install cython==0.29.21
git clone https://github.com/bitcraze/crazyflie-clients-python.git
cd crazyflie-clients-python/
git checkout a0d617aeb20183741f10994b626cc7adef22219e
python3 setup.py develop
```

### Test the crazyflie with Crazyradio or Bluetooth (built-in bluetooth only working for ultra96-V2)

Attach the keyboard to ultra96 and use the keys `w, s` for z-axis, `i, k` for x-axis, `j, l` for y-axis and `a, d` for yaw rate motion control. Moreover you and increase/decrease the speed using `left shift/left control` and pressing `space` will cause the crazyflie to fall off. To resume the flight after fall of press `r`. When turning on the crazyflie, make sure that the forward direction is pointing away from you (i.e. the back of crazyflie should be facing you, so pressing `w` will make it move away in the forward direction).

```
# for crazyradio
cd quad96/quad96/crazyflie
sudo python3 crazyflie_class.py

# for bluetooth
./blue.sh # (run this only once if you are using the ultra96-v2's own bluetooth and not the external one)
sudo python3 crazyflie_class.py --ble
```

### Calibrate the camera to detect the hand/object of interest

Attach the camera to ultra96 (if you are going to use) and calibrate for HSV values of your desired object of interest. Press `q` to quit calibration.

```
cd quad96/quad96/
sudo python3 -m camera.thresholder
```

Now paste the obtained `self.low` and `self.high` values in `camera.py` line  `130-131`. Next run the camera controller to make sure the hand gestures are being recognized correctly. Press `q` to quit.
```
sudo python3 -m camera.camera
```
The camera will calibrate the background for the first 60 frames, make sure while calibrating background, to not show it the object of interest you are going to detect.

### Run the TD3PG agent on DPU and view in the simulator

Note the ip address of the ultra96. Unity simulator will connect to this ip address. Launch the unity environment window simulator on the pc.

```
# on PC
cd quad96/env_window/
vi env_ip.txt # and replace the ip address only. Keep the port same unless you are manually changing the port on ultra96 using --port <port number>

# launch the environment window
./build/unity_sim.x86_64

```

Now on ultra96 launch the agent on dpu. If you want to use camera input to the agent pass `--camera_pos` and hold `c` on the attached keyboard whenever you are ready and the camera detection is stable. Now move the hand (while holding `c`) in any direction and the quad in the unity simulator will follow. If you want to use keyboard only for the movement control, pass `--keyboard_pos` and the same keys will be used for movement control as for crazyflie i.e. `w a s d i j k l`.

```
sudo python3 -m ultra96.test_quad --episodes 10 --sim_time 5 --render_window <your argument i.e. --camera_pos or --keyboard_pos>
```

Note: Unity simulator need to be reset (by pressing `r` on PC) for every new connection (this is the requirement by zmq). So whenever you run any command which connects to the simulator, make sure that the simulator is in reset state, otherwise the connection will not be established. Also, by default for ultra96, the scripts are designed to work with the vnc display for camera and attached keyboard (not the laptop keyboard via vnc). So if you are not using the vnc server make sure to have a look and set the `DISPLAY` variable correctly according to your settings in `camera.py` and `keyboard.py`

###  Run the TD3PG agent on DPU and connect with Crazyflie

Crazyflie can also be connected to the environment simulation and the output of the TD3PG running on DPU can be directed to the crazyflie (via crazyradio only, not with bluetooth). But here it is open loop control with crazyflie i.e. no position feedback, just the estimate from the simulation. In this open loop settings crazyflie will very loosly follow the simulation, but it is noticable. The close loop control requires [loco positioning system](https://www.bitcraze.io/products/loco-positioning-system/) to give the position feedback, which can significatly improve the performance of the implemented TD3PG. Run the follow command to attach the crazyflie. Again, when turning on the crazyflie, make sure that the forward direction is pointing away from you (i.e. the back of crazyflie should be facing you, so pressing `w` will make it move away in the forward direction).

```
sudo python3 -m ultra96.test_quad --episodes 10 --sim_time 5 --render_window --camera_pos --crazyflie
```

Once the simulation has been started and camera is detecting the object, first use the keyboard to move the crazyflie to higher level and desired positon in the space. Now holding `c` will:

1) Direct the camera input to the DPU.

2) Direct the output from the simulator to crazyflie.     

Slowly move the hand/object of interest (while holding `c`) and out of the blue circle (dead area) and both the simulation and the crazyflie will follow. Again, if at any point crazyflie is getting out of control, press `space` to cause the crazyflie to fall off. To resume again, press `r`, and again take the crazyflie to the desired point in space and hold `c` while moving hand/object of interest. 

Note: Do not use `--keyboard_pos` argument when using `--crazyflie` because now crazyflie's keyboard controller will be active.

# References

[1] [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf)

[2] [System Identification of the Crazyflie 2.0 NanoQuadrocopter](http://mikehamer.info/assets/papers/Crazyflie%20Modelling.pdf)


# Acknowledgements

Rank Based Prioritized Experience Replay taken from [here](https://github.com/Damcy/prioritized-experience-replay)

