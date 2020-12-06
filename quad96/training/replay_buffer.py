import random
import numpy as np
from collections import deque

from training import rank_based


class PERBuffer(object):

  def __init__(self, conf, buffer_size):

    self.buffer_size = buffer_size
    self.mem_cntr = 0
    self.replay_memory = rank_based.Experience(conf)

  def sample(self, batch_size):

    batch, w, e_id = self.replay_memory.sample(self.mem_cntr)
    self.e_id=e_id
    self.w_id=w
    return batch, self.w_id, self.e_id


  def size(self):
    return self.buffer_size

  def add(self, state, action, output, reward, next_state, done):
    self.replay_memory.store((state, action, output, reward, next_state, done))
    self.mem_cntr += 1

  def __len__(self):
    return min(self.mem_cntr, self.buffer_size)

  def rebalance(self):
    self.replay_memory.rebalance()

  def update_priority(self, indices, delta):
    self.replay_memory.update_priority(indices, delta)
