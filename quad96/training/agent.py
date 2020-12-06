import os
import random
import numpy as np
from collections import deque, namedtuple

from simulator.utils import deg2rad
from training.actor_critic import Actor, Critic
from training.replay_buffer import PERBuffer

__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"


# defination for the actal TD3 agent having actor, critic, experience replay 
# and providing act, remember, train functions

class TD3PG():
    """Reinforcement Learning agent that learns using TD3PG."""
    def __init__(self, args, state_size, action_size, scale=1.):
        
        self.args = args
        self.state_size = state_size
        self.action_size = action_size
        self.scale = scale
        
        # Actor (Policy) Model
        self.actor_local = Actor(self.args, self.state_size, self.action_size, 'local')
        self.actor_target = Actor(self.args, self.state_size, self.action_size, 'target')
        
        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        
        # Critic2 (Value) Model
        self.critic2_local = Critic(self.state_size, self.action_size)
        self.critic2_target = Critic(self.state_size, self.action_size)
        
        # Initialize target model parameters with local model parameters
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.critic2_target.model.set_weights(self.critic2_local.model.get_weights())
        
        # Noise process
        self.exploration_mu = np.zeros(3) 
        self.exploration_sigma =  np.array([deg2rad(0), deg2rad(0), 0])
        self.target_smoothing_mu = np.zeros(3)
        self.target_smoothing_sigma = np.array([deg2rad(2), deg2rad(2), 0.2])
        self.target_smoothing_clip = np.array([deg2rad(5), deg2rad(5), 0.5])
        
        # Replay memory
        self.buffer_size = int(1e4)
        self.batch_size = 64
        
        # Rank based Prioritized Experience Replay (PER)
        conf = {'size': self.buffer_size,
                'learn_start': self.batch_size,
                'partition_num': self.batch_size,
                'total_step': self.args.episodes*200,
                'batch_size': self.batch_size}
        self.memory = PERBuffer(conf, self.buffer_size)
                
        # Algorithm parameters
        self.gamma = 0.99
        self.tau = 0.002
        self.itr = 0
        self.reb_itr = 0
        self.update_interval = 2

    def reset(self):
        pass

    # rememeber a tuple for training
    def remember(self, state, action, output, reward, next_state, done):
        state = np.reshape(state, [1,1, self.state_size])
        next_state = np.reshape(next_state, [1,1, self.state_size])
        self.memory.add(state, action, output, reward, next_state, done)

    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        states = np.reshape(states, [1,1,1, self.state_size])
        action = self.actor_local.model.predict(states)[0]
        action *= self.scale
        if not self.args.eval:
            action += np.random.normal(self.exploration_mu, self.exploration_sigma, self.action_size)
        action = np.clip(action, -self.scale, self.scale)
        return list(action), None  # add some noise for exploration
        # return action

    def replay(self):
        """Update policy and value parameters using given batch of experience tuples."""
        # Learn, if enough samples are available in memory
        
        if len(self.memory) < self.batch_size*40:
            return
        
        batch, weights, batch_idxes = self.memory.sample(self.batch_size)
        states          = np.zeros((self.batch_size, 1,1, self.state_size))
        actions         = np.zeros((self.batch_size, self.action_size))
        outputs         = np.zeros((self.batch_size, self.action_size))
        rewards         = np.zeros((self.batch_size,))
        next_states     = np.zeros((self.batch_size, 1,1,self.state_size))
        dones           = np.zeros((self.batch_size,))
        for k, (s0, a, o, r, s1, done) in enumerate(batch):
            states[k]  = s0
            actions[k] = a
            outputs[k] = o
            rewards[k] = r
            if not done:
                next_states[k] = s1
                dones[k] = 1

        
        # Get predicted next-state actions and Q values from target models
        actions_next  = self.actor_target.model.predict_on_batch(next_states)
        actions_next *= self.scale
        actions_next += np.clip(np.random.normal(self.target_smoothing_mu, self.target_smoothing_sigma, \
                                                 self.action_size), \
                                -self.target_smoothing_clip, self.target_smoothing_clip)
        actions_next = np.clip(actions_next, -self.scale, self.scale)

        # Compute Q targets for current states 
        Q_targets_next1 = self.critic_target.model.predict_on_batch([next_states, actions_next])
        Q_targets_next2 = self.critic2_target.model.predict_on_batch([next_states, actions_next])
        Q_targets_next  = np.minimum(Q_targets_next1, Q_targets_next2)
        Q_targets = rewards + self.gamma * Q_targets_next.reshape(-1) * dones

        # train critic model (local)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets, sample_weight=weights)
        self.critic2_local.model.train_on_batch(x=[states, actions], y=Q_targets, sample_weight=weights)


        # update priories in PER buffer
        Q_estimate = self.critic_local.model.predict_on_batch([states, actions])
        # Q_estimate2 = self.critic2_local.model.predict_on_batch([states, actions])
        # Q_estimate = np.maximum(Q_estimate1, Q_estimate2)
        TD_error = np.abs(Q_targets - Q_estimate.reshape(-1)) + 1e-6  
        
        # Rank Based
        self.memory.update_priority(batch_idxes, TD_error)
        self.reb_itr += 1
        if self.reb_itr == 100:
            self.memory.rebalance()
            self.reb_itr = 0

        self.itr += 1


        if self.itr==self.update_interval:
            # Train actor model (local)
            action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
            action_gradients *= self.scale
            outputs /= self.scale
            action_gradients = np.hstack([action_gradients, outputs])
            self.actor_local.model.train_on_batch(x=[states], y=[action_gradients])

            # Soft-update target models
            self.soft_update(self.actor_local.model, self.actor_target.model)
            self.soft_update(self.critic_local.model, self.critic_target.model)
            self.soft_update(self.critic2_local.model, self.critic2_target.model)
            
            self.itr = 0

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"
        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
