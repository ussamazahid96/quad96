import sys
import cv2
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K

from simulator.utils import deg2rad
from simulator.drone_class import Drone
from training.environment import drone_env
from training.trainer import RLtrainer
from simulator.params import drone_params, controller_params

__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"


# the main entry point which can run the trainer for training or run a simple simulation for testing

def get_args():
    parser = argparse.ArgumentParser(description="Quadcopter Sim.")
    parser.add_argument('--run_sim', action='store_true', help="Run Simulation.")
    parser.add_argument('--sim_time', type=float, default=1, help="Simulation time")

    parser.add_argument('--camera_dis', action='store_true', help="Camera Disturbance.")
    parser.add_argument('--camera_pos', action='store_true', help="Camera Position Control.")
    parser.add_argument('--keyboard_dis', action='store_true', help="Keyboard Disturbance.")
    parser.add_argument('--keyboard_pos', action='store_true', help="Keyboard Position Control.")
    
    parser.add_argument('--render_pid', action='store_true', help="show the pid plots.")
    parser.add_argument('--render_quad', action='store_true', help="Render the Quadcopter Simulation in Python Plot.")
    parser.add_argument('--port', type=int, default=4567, help="Port to use for Unity Simulator Connection.")
    parser.add_argument('--render_window', action='store_true', help="Render the Quadcopter Simulation in Unity.")

    parser.add_argument('--train', action='store_true', help="train the RL agent.")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint.")
    parser.add_argument('--eval', action='store_true', help="Evaluate the saved model.")
    parser.add_argument('--test_quantized', action='store_true', help="Test the Quantized Graph.")
    parser.add_argument('--episodes', type=int, default=100, help="Number of episodes to train.")
    parser.add_argument('--points', type=int, default=100, help="Number of data points for calibration.")
    parser.add_argument('--export', action='store_true', help="Export Graph for VAI Quantization and Compilation.")
    parser.add_argument('--env', type=str, default="quad", help="Environment to train on [quad, Pendulum-v0, ...].")
    parser.add_argument('--gen_calib_data', action='store_true', help="Generate calibration data for VAI calibration.")


    return parser.parse_args()

def run_sim(args):
    try:
        print("[Main] Running Quadcopter Simulation")
        drone1_params = drone_params()
        controller1_params = controller_params()
        drone1 = Drone(drone1_params, controller1_params, args=args)
        
        # command = "circle"
        # drone1.follow_path(command)

        # command = np.array([deg2rad(0), deg2rad(-20), deg2rad(0), 36330.])
        # drone1.attitude_control(command)
        while drone1.running:
            drone1.position_control(drone1.ph_command)
                
        # drone1.position_hold_threaded()
        # print("[Main] Waiting in loop")
        # if args.camera_pos or args.camera_dis:
        #     while drone1.running:
        #         cv2.imshow("Video Feed", drone1.camera_handle.frame)
        #         waitkey = cv2.waitKey(1) & 0xFF 
        #         if waitkey == ord("q"):
        #             drone1.turnoff()
        # else:
        #     while drone1.running:
        #         pass
        
        if args.render_pid:
            plt.show(block=True)



        drone1.turnoff()

    except Exception as e:
        drone1.turnoff()
        print("ERROR: {}".format(e))


def trainRLagent(args):
    # try:

    if args.export:
        K.set_learning_phase(0)
    
    trainer = RLtrainer(args)
    
    if args.eval:
        trainer.eval()
    elif args.train:
        trainer.train()

    if args.export:
        trainer.export()
    if args.gen_calib_data:
        trainer.gen_calib_data()
    if args.test_quantized:
        trainer.test_quantized()
    
    # except Exception as e:
    #     trainer.env.close()
    #     print("ERROR: {}".format(e))

def main():
    args = get_args()
    if args.run_sim:
        run_sim(args)
    else:
        trainRLagent(args)

if __name__=='__main__':
    main()