import signal
import argparse
import numpy as np
from pynq_dpu import DpuOverlay

from training.environment import drone_env
from simulator.params import drone_params, controller_params

__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"

# the main RL agent using the DPU-PYNQ for execution

class RL_agent:
    def __init__(self, elf_file, env):
        self.overlay = DpuOverlay("dpu.bit")
        self.overlay.set_runtime("vart")
        self.overlay.load_model(elf_file)
        self.dpu = self.overlay.runner
        self.env = env
        self.scale = self.env.scale

        self.inputTensors = self.dpu.get_input_tensors()
        outputTensors = self.dpu.get_output_tensors()
        tensorformat = self.dpu.get_tensor_format()
        if tensorformat == self.dpu.TensorFormat.NCHW:
            outputHeight = outputTensors[0].dims[2]
            outputWidth = outputTensors[0].dims[3]
            outputChannel = outputTensors[0].dims[1]
        elif tensorformat == self.dpu.TensorFormat.NHWC:
            outputHeight = outputTensors[0].dims[1]
            outputWidth = outputTensors[0].dims[2]
            outputChannel = outputTensors[0].dims[3]
        else:
            raise ValueError("Input format error.")

        self.outputSize = outputHeight*outputWidth*outputChannel
        self.tanh = np.empty(self.outputSize)
        
        shape_in = (1,) + tuple([self.inputTensors[0].dims[i] for i in range(self.inputTensors[0].ndims)][1:])
        shape_out = (1, outputHeight, outputWidth, outputChannel)
        self.input_data = []
        self.output_data = []
        self.input_data.append(np.empty((shape_in), dtype = np.float32, order = 'C'))
        self.output_data.append(np.empty((shape_out), dtype = np.float32, order = 'C'))
        self.input = self.input_data[0]
        signal.signal(signal.SIGINT, self.interrupt_handle)

    def interrupt_handle(self, signal, frame):
        print('[Ultra96] Stopping')
        self.env.close()
        exit(0)

    def act(self, state):
        self.input[0,...] = state.reshape(
            self.inputTensors[0].dims[1],
            self.inputTensors[0].dims[2],
            self.inputTensors[0].dims[3])
        job_id = self.dpu.execute_async(self.input_data, self.output_data)
        self.dpu.wait(job_id)
        temp = [j.reshape(1, self.outputSize) for j in self.output_data]
        self.tanh = self.calculate_tanh(temp[0][0])
        action = self.tanh*self.scale
        return action

    def post_process(self, outputs):
        throttle = np.random.normal(outputs[0], np.square(outputs[3]))
        roll = np.random.normal(outputs[1], np.square(outputs[4]))
        pitch = np.random.normal(outputs[2], np.square(outputs[5]))
        return np.clip(np.array([throttle, roll, pitch]), -1, 1)


    def calculate_tanh(self, data):
        result = np.tanh(data)
        return result


def get_args():
    parser = argparse.ArgumentParser(description="Quadcopter Sim.")
    parser.add_argument('--run_sim', action='store_true', help="Run Simulation.")
    parser.add_argument('--sim_time', type=float, default=1, help="Simulation time")

    parser.add_argument('--camera_dis', action='store_true', help="Camera Disturbance.")
    parser.add_argument('--camera_pos', action='store_true', help="Camera Position Control.")
    parser.add_argument('--keyboard_dis', action='store_true', help="Keyboard Disturbance.")
    parser.add_argument('--keyboard_pos', action='store_true', help="Keyboard Position Control.")
    parser.add_argument('--crazyflie', action='store_true', help="Connect to CrazyFlie.")
    
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


# the main function running the dpu and interfacing it with the drone env for testing
def main():
    args = get_args()
    env = drone_env(drone_params(), controller_params(), args=args)
    agent = RL_agent('dpu_TD3PGagent_quad.elf', env)
    average = []
    for i in range(args.episodes):
        state = env.reset()
        state = np.reshape(state, [1,1,1, env.state_size])
        done = False
        score = 0
        while True:#not done:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            state = np.reshape(state, [1,1,1, env.state_size])
            score += reward
        average.append(score)
        avg = np.mean(average[-40:])
        print("Episode: {}/{}, Score: {:.3f}, Average: {:.3f}".format(i+1, args.episodes, score, avg))
    env.close()


if __name__=='__main__':
    main()






