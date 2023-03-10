
import tensorflow as tf
import numpy as np
from noise import OUNoise
from critic_ddpg import Critic
from actor_ddpg import Actor
from memory import ReplayBuffer

# Hyper Parameters:

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99


class DDPG:

    def __init__(self, env):
        self.name = 'DDPG' 
        self.environment = env

        self.state_dim = env.observation_space.shape[0]
        print(env.action_space)
        self.action_dim = env.action_space.shape[0]

        self.sess = tf.InteractiveSession()

        self.actor_network = Actor(self.sess,self.state_dim,self.action_dim)
        self.critic_network = Critic(self.sess,self.state_dim,self.action_dim)
        
        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)

    def train(self):

        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch,[BATCH_SIZE,self.action_dim])

        # Calculate y_batch
        
        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch,next_action_batch)
        y_batch = []  
        for i in range(len(minibatch)): 
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch,[BATCH_SIZE,1])
        # Update critic by minimizing the loss L
        self.critic_network.train(y_batch,state_batch,action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch,action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch,state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    def noise_action(self,state):
        # Select action 
        action = self.actor_network.action(state)
        return action+self.exploration_noise.noise()

    def action(self,state):
        action = self.actor_network.action(state)
        return action

    def perceive(self,state,action,reward,next_state,done):
        # Store in replay buffer
        self.replay_buffer.add(state,action,reward,next_state,done)

        if self.replay_buffer.count() >  REPLAY_START_SIZE:
            self.train()

        if done:
            self.exploration_noise.reset()