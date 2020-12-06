import os
import time
import pylab
import math
import signal
import pickle
import warnings
import numpy as np
import random as rnd

import gym

from training.agent import TD3PG
from training.environment import drone_env
from simulator.params import drone_params, controller_params
from simulator.utils import deg2rad

import tensorflow as tf
import keras.backend as K
from keras.backend.tensorflow_backend import set_session

__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"

# the main trainer which takes an environment and trains an agent on it

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

seed_value=0
os.environ['PYTHONHASHSEED']=str(seed_value)
rnd.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

class RLtrainer:
    
    def __init__(self, args):
        self.args = args
        self.running = True
        self.start_episode = 1
        self.score_history = []
        self.max_average = -10000
        self.scores, self.episodes, self.average = [], [], []

        if not os.path.exists('./training/Models'):
            os.makedirs('./training/Models')

        if self.args.env == "quad":
            self.env = drone_env(drone_params(), controller_params(), args=args)
            self.state_size = self.env.state_size
            self.action_size = self.env.action_size
            self.scale = self.env.scale
        else:
            self.env = gym.make(self.args.env)
            self.state_size = self.env.observation_space.shape[0]
            self.action_size = self.env.action_space.shape[0]
            self.scale = self.env.action_space.high
            self.env.seed(seed_value)
        print("State Size = {}".format(self.state_size))
        print("Action Size = {}".format(self.action_size))
        print("Scale = {}".format(self.scale))
        self.agent = TD3PG(self.args, self.state_size, self.action_size, self.scale)
        if self.args.resume is not None:
            self.load(self.args.resume) 
        
        signal.signal(signal.SIGINT, self.interrupt_handle)

    pylab.figure(figsize=(10, 5))
    def PlotModel(self, score, episode, postfix):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(np.mean(self.scores[-40:]))
        if str(episode)[-1:] == "0":
            pylab.plot(self.episodes, self.scores, 'b')
            pylab.plot(self.episodes, self.average, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
        try:
            pylab.savefig("./training/Models/reward_plot_{}.png".format(postfix))
        except OSError:
            pass
        return self.average[-1]

    
    def interrupt_handle(self, signal, frame):
        print('[Trainer] Stopping')
        self.running = False


    def train(self):
        for e in range(self.start_episode, self.args.episodes+1):
            state = self.env.reset()
            self.agent.reset()
            done, score, SAVING, steps = False, 0, '', 0
            while not done:
                # self.env.render()
                # output = 0. # for gym
                output, _ = self.env.assistant() # for quad env only
                action, _ = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.remember(state, action, output, reward, next_state, int(done))
                state = next_state
                score += reward
                steps += 1            
                self.agent.replay()    
                if not self.running:
                    self.env.close()
                    exit(0)
            average = self.PlotModel(score, e, "train")
            if average >= self.max_average:
                self.max_average = average
                path = "./training/Models/best.tar"
                self.save(e, path)
            elif (e) % 10 == 0:
                path = "./training/Models/checkpoint.tar"
                self.save(e, path)
            else:
                path = "None"
            print("Episode: {}/{}, Steps: {}, Score: {:.3f}, Average: {:.2f}, Saving at: {}".format(e, \
                        self.args.episodes, steps, score, average, path))
        self.env.close()

    def eval(self):
        self.scores, self.episodes, self.average = [], [], []
        for e in range(self.start_episode, self.args.episodes+1):
            state = self.env.reset()
            state = np.reshape(state, [1,1,1, self.state_size])
            done = False
            score = 0
            while not done:
                # self.env.render()
                action, _ = self.agent.act(state)
                state, reward, done, _ = self.env.step(action)
                state = np.reshape(state, [1,1,1, self.state_size])
                score += reward
                if not self.running:
                    self.env.close()
                    exit(0)
            average = self.PlotModel(score, e, "eval")
            print("Episode: {}/{}, Score: {:.3f}, Average: {:.3f}".format(e, self.args.episodes, score, average))
        self.env.close()


    def save(self, e, postfix):
        actor_local = np.array(self.agent.actor_local.model.get_weights())
        actor_target = np.array(self.agent.actor_target.model.get_weights())
        critic_local = np.array(self.agent.critic_local.model.get_weights())
        critic_target = np.array(self.agent.critic_target.model.get_weights())
        critic2_local = np.array(self.agent.critic2_local.model.get_weights())
        critic2_target = np.array(self.agent.critic2_target.model.get_weights())
        savedict = {"actor_local": actor_local, "actor_target": actor_target, "critic_local": critic_local, \
                    "critic_target": critic_target, "critic2_local": critic2_local, "critic2_target": critic2_target,\
                    "start_episode": e+1, "max_average": self.max_average, "scores": self.scores, \
                    "episodes": self.episodes, "average": self.average}
        with open(postfix, 'wb') as f:
            pickle.dump(savedict, f)

    def load(self, path):
        print("Loading checkpoint {}".format(path))
        with open(path, 'rb') as f:
            loaddict = pickle.load(f)
        self.agent.actor_local.model.set_weights(loaddict["actor_local"])
        if not self.args.export and not self.args.eval:
            self.agent.actor_target.model.set_weights(loaddict["actor_target"])
            self.agent.critic_local.model.set_weights(loaddict["critic_local"])
            self.agent.critic_target.model.set_weights(loaddict["critic_target"])
            self.agent.critic2_local.model.set_weights(loaddict["critic2_local"])
            self.agent.critic2_target.model.set_weights(loaddict["critic2_target"])
            self.start_episode = loaddict["start_episode"]
        self.scores = loaddict["scores"]
        self.average = loaddict["average"]
        self.episodes = loaddict["episodes"]
        self.max_average = loaddict["max_average"]

    # export the actor graph
    def export(self):  
        saver = tf.train.Saver()
        tf_session = K.get_session()
        input_graph_def = tf_session.graph.as_graph_def()
        save_path = saver.save(tf_session, './deploy/checkpoint.ckpt')
        tf.train.write_graph(input_graph_def, './deploy/', 'actor.pb', as_text=False)
        print("[Trianer] Actor graph exported under ./deploy/")
        self.env.close()

    # gen calibration data for vitis ai
    def gen_calib_data(self):
        state_array = []
        state = self.env.reset()
        state_array.append(state)
        for i in range(self.args.points):
            action, _ = self.agent.act(state)
            next_state, reward, done, _ = self.env.step(action)
            state_array.append(next_state)
            if done:
                state = self.env.reset()
                state_array.append(state)
        state_array = np.array(state_array)
        np.savez('./deploy/calib_data.npz', data = state_array[:self.args.points,:])
        print("[Trainer] Calibration data exported under ./deploy/ ...")
        self.env.close()


    # to test the quantized graph obtained from vai
    def test_quantized(self):
        tf.keras.backend.clear_session()
        import tensorflow.contrib.decent_q
        with tf.gfile.GFile('./deploy/quantized/quantize_eval_model.pb', "rb") as f:
            graph = tf.GraphDef()
            graph.ParseFromString(f.read())

        tf.import_graph_def(graph,name = '')
        input_data = tf.get_default_graph().get_tensor_by_name('input_data_actor_local'+':0')
        logits = tf.get_default_graph().get_tensor_by_name('output_logits_actor_local/Identity'+':0')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.initializers.local_variables())
            for e in range(self.start_episode, self.args.episodes+1):
                state = self.env.reset()
                done = False
                score = 0
                while not done:
                    state = np.reshape(state, [1,1,1, self.state_size])
                    prediction = logits.eval(feed_dict={input_data: state})
                    prediction = np.tanh(prediction)[0]
                    action = prediction*self.scale 
                    state, reward, done, _ = self.env.step(action)
                    score += reward
                    if not self.running:
                        self.env.close()
                        exit(0)
                average = self.PlotModel(score, e, "eval_q")
                print("Episode: {}/{}, Score: {:.3f}, Average: {:.3f}".format(e, self.args.episodes, score, average))
        self.env.close()
