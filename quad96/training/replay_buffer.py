import numpy as np

# simple replay buffer

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.output = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def add(self, state, action, output, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.output[index] = output
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1. - done
        self.mem_cntr += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states   = self.state_memory[batch]
        actions  = self.action_memory[batch]
        output   = self.output[batch]
        rewards  = self.reward_memory[batch]
        states_  = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, output, rewards, states_, terminal, None

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)