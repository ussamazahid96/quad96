import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import layers, models, optimizers, regularizers

__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"

# this file contains the model definations i.e. Actor and Critic and the special actor loss function 
# which use the feedback from the position controller

neurons = 128


class Actor:
    """Actor (Policy) Model."""
    def __init__(self, args, state_size, action_size, name):
        """Initialize parameters and build model."""
        self.args = args
        self.state_size = state_size
        self.action_size = action_size
        self.name = name
        self.lr_actor = 1e-3
        self.loss_mse_scale = 0.5
        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states to actions."""

        # special loss for quad
        def quadloss(action_grad, actions):
            action_gradients, outputs = action_grad[:,:self.action_size], action_grad[:,self.action_size:]
            loss1 =  K.mean(-action_gradients * actions)
            loss2 =  K.mean(K.square(outputs - actions))
            loss = (1-self.loss_mse_scale)*loss1 + (self.loss_mse_scale)*loss2 
            return loss

        # for gym environments
        def gymloss(action_grad, actions):
            action_gradients, outputs = action_grad[:,:self.action_size], action_grad[:,self.action_size:]
            loss =  K.mean(-action_gradients * actions)
            return loss

        last_activation = None if self.args.export else 'tanh'

        states = layers.Input(shape=(1,1, self.state_size,), name='input_data_actor_{}'.format(self.name))
        
        net = layers.Conv2D(neurons, 1)(states)
        net = layers.Activation(tf.nn.relu6)(net)

        net = layers.Flatten()(net)

        net = layers.Dense(neurons)(net)
        net = layers.Activation(tf.nn.relu6)(net)

        net = layers.Dense(self.action_size)(net)
        outputs = layers.Activation(last_activation, name='output_logits_actor_{}'.format(self.name))(net)

        self.model = models.Model(inputs=[states], outputs=outputs)
        self.model.compile(optimizer=optimizers.Adam(lr=self.lr_actor), loss=quadloss)


class Critic:
    """Critic (Value) Model."""
    def __init__(self, state_size, action_size):
        """Initialize parameters and build model."""
        self.lr_critic = 0.001
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    def build_model(self):

        # Define input layers
        states = layers.Input(shape=(1,1, self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')        
        # states
        net_states = layers.Flatten()(states)

        net = layers.Concatenate()([net_states, actions])
        net = layers.Dense(neurons)(net)
        net = layers.Activation('relu')(net)

        net = layers.Dense(neurons)(net)
        net = layers.Activation('relu')(net)

        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)       
        
        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)
        
        # self.optimizer = optimizers.Adam(self.lr_critic)
        
        self.model.compile(optimizer=optimizers.Adam(self.lr_critic), loss='mse')
        action_gradients = K.gradients(Q_values, actions)
        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
