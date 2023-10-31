# quad96

Quadcopter pole balancing using Deep Reinforcement Learning on Ultra96 with DPU

Trained using twin delayed deep deterministic policy gradient (TD3), experience replay and feedback

## Video Demo

[![IMAGE ALT TEXT](https://yt-embed.herokuapp.com/embed?v=b2clTUmQJPg)](https://youtu.be/b2clTUmQJPg "Simulation Demo")


## Hardware Requirements

-> [Ultra96-V1/V2](https://www.96boards.org/product/ultra96/)

-> USB Keyboard

-> USB Hub (Optional)

-> USB Camera (Optional)

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
### Calibrate the camera to detect the hand/object of interest

Attach the camera to ultra96 (if you are going to use) and calibrate for HSV values of your desired object of interest. Press `q` to quit calibration.

```
cd quad96/quad96/
sudo python3 -m camera.thresholder
```

Now paste the obtained `self.low` and `self.high` values in `camera.py` line  `119-120`. Next run the camera controller to make sure the hand gestures are being recognized correctly. Press `q` to quit.
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

Now on ultra96 launch the agent on dpu. If you want to use camera input to the agent pass `--camera_pos` and hold `c` on the attached keyboard whenever you are ready and the camera detection is stable. Now move the hand (while holding `c`) in any direction and the quad in the unity simulator will follow. If you want to use keyboard only for the movement control, pass `--keyboard_pos` and use: `w a s d i j k l`.

```
sudo python3 -m ultra96.test_quad --episodes 10 --sim_time 10 --render_window <your argument i.e. --camera_pos or --keyboard_pos>
```

Note: Unity simulator need to be reset (by pressing `r` on PC) for every new connection (this is the requirement by zmq). So whenever you run any command which connects to the simulator, make sure that the simulator is in reset state, otherwise the connection will not be established. Also, by default for ultra96, the scripts are designed to work with the vnc display for camera and attached keyboard (not the laptop keyboard via vnc). So if you are not using the vnc server make sure to have a look and set the `DISPLAY` variable correctly according to your settings in `camera.py` and `keyboard.py`

## Pendulum-v0 as bonus :)

In `trainer.py` set `output = 0.` and comment out the assistant

In `actor_critic.py` set `neurons = 512`, actor and critic lr = 0.0002, change loss to `gymloss` and last layer init as
```
last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
net = layers.Dense(self.action_size, kernel_initializer=last_init)(net)
```

In `agent.py` set buffer size to 50000, `tau = 0.01` and use the following noise process
```
self.exploration_mu = 0 
self.exploration_sigma =  0.1
self.target_smoothing_mu = 0
self.target_smoothing_sigma = 0.2
self.target_smoothing_clip = 0.5
```
Pass the argument `--env Pendulum-v0` to the training command.

# References

[1] Zhang, C., Hu, H., Gu, D., & Wang, J. (2017). Cascaded control for balancing an inverted pendulum on a flying quadrotor. <i>Robotica,</i> <i>35</i>(6), 1263-1279. doi:10.1017/S0263574716000035

[2] R. Figueroa, A. Faust, P. Cruz, L. Tapia and R. Fierro, "Reinforcement learning for balancing a flying inverted pendulum," Proceeding of the 11th World Congress on Intelligent Control and Automation, Shenyang, 2014, pp. 1787-1793, doi: 10.1109/WCICA.2014.7052991.

[3] [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf)

[4] [System Identification of the Crazyflie 2.0 NanoQuadrocopter](http://mikehamer.info/assets/papers/Crazyflie%20Modelling.pdf)
