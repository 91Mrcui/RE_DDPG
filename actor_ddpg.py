import tensorflow as tf
import numpy as np
import math

# parameters from paper
LAYER1_SIZE=400
LAYER2_SIZE=300
LR=0.0001
TAU=0.001
BATCH=64

class Actor:
    def __init__(self,sess,state_dim,action_dim):
        self.sess=sess #session
        self.state_dim=state_dim
        self.action_dim=action_dim
        # create network
        self.input_state,self.output_action,self.net=self.create(state_dim,action_dim)
        # create target network
        self.target_input_state,self.target_output_action,self.target_update,self.target_net = self.create_target(state_dim,action_dim,self.net)
        # train methods
        self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
        # gradients
        self.parameters_gradients = tf.gradients(self.output_action,self.net,-self.q_gradient_input)
        # use adamoptimizer as the paper said
        self.optimizer = tf.train.AdamOptimizer(LR).apply_gradients(zip(self.parameters_gradients,self.net))

        self.sess.run(tf.initialize_all_variables())
        self.sess.run(self.target_update)


    # create actor network
    def create(self,state_dim,action_dim):
        state_input = tf.placeholder("float",[None,state_dim])
        # create variables
        # as the paper set
        w1=self.variable([state_dim,LAYER1_SIZE],state_dim)
        b1=self.variable([LAYER1_SIZE],state_dim)
        w2=self.variable([LAYER1_SIZE,LAYER2_SIZE],LAYER1_SIZE)
        b2=self.variable([LAYER2_SIZE],LAYER1_SIZE)
        w3 = tf.Variable(tf.random_uniform([LAYER2_SIZE,action_dim],-3e-3,3e-3))
        b3 = tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3))
        
        layer1 = tf.nn.relu(tf.matmul(state_input,w1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1,w2) + b2)
        #Compress the value between - 1 and 1
        output = tf.tanh(tf.matmul(layer2,w3) + b3)
        
        return state_input,output,[w1,b1,w2,b2,w3,b3]
    
    def create_target(self,state_dim,action_dim,net):
        state_input = tf.placeholder("float",[None,state_dim])
        ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer1 = tf.nn.relu(tf.matmul(state_input,target_net[0]) + target_net[1])
        layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + target_net[3])
        output = tf.tanh(tf.matmul(layer2,target_net[4]) + target_net[5])

        return state_input,output,target_update,target_net



    def variable(self,shape,f):
        # random min=-1/math.sqrt(f),max=1/math.sqrt(f)
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self,q_gradient_batch,state_batch):
        self.sess.run(self.optimizer,feed_dict={
			self.q_gradient_input:q_gradient_batch,
			self.state_input:state_batch
			})

    def actions(self,state_batch):
        return self.sess.run(self.action_output,feed_dict={
			self.state_input:state_batch
			})

    def action(self,state):
        return self.sess.run(self.action_output,feed_dict={
			self.state_input:[state]
			})[0]


    def target_actions(self,state_batch):
        return self.sess.run(self.target_action_output,feed_dict={
			self.target_state_input:state_batch
			})






