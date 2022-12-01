import numpy as np
import gym
import random
import copy
import pickle
from collections import namedtuple, deque
# %tensorflow_version 1.x
from keras.models import Sequential,Model
from keras.layers import Dense, Concatenate, Input, concatenate, Flatten, BatchNormalization, Add
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import tensorflow.compat.v1 as tf1

ENV_NAME = "BipedalWalker-v3"

record_filename = './record_ddpg_keras.dat'

TAU = 0.001        
GAMMA = 0.99   

LR_ACTOR = 0.0001
LR_CRITIC = 0.001

BUFFER_SIZE = 10**6
BATCH_SIZE = 64  

MAX_EPISODES = 10001
MAX_STEPS = 1602

class OUNoise:
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = seed
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class Actor:
    def __init__(self, state_size, action_size, seed, lrn_rate, fc_units=600, fc1_units=300):

        self.state_size = state_size
        self.action_size = action_size
        self.lr = lrn_rate
        self.seed = seed

        state = Input((state_size,))
        l1 = Dense(fc_units,activation='relu')(state)
        bn1 = BatchNormalization(axis=1, epsilon=1e-5)(l1)
        l2 = Dense(fc1_units,activation='relu')(bn1)
        bn2 = BatchNormalization(axis=1, epsilon=1e-5) (l2)
        out = Dense(action_size,activation='tanh')(bn2)

        self.model = Model(inputs = [state],outputs = [out])
        self.optimize = self.optimizer()

    def optimizer(self):
        action_gdts = K.placeholder(shape=(None, self.action_size))
        params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.model.trainable_weights)
        
        return K.function(inputs=[self.model.input, action_gdts], outputs=[K.constant(1)], updates=[tf1.train.AdamOptimizer(self.lr).apply_gradients(grads)])

class Critic:
    def __init__(self, state_size, action_size, seed, lrn_rate, fcs1_units=600, fcs2_units=300, fca1_units=300):

        state = Input((state_size,))
        action = Input((action_size,))
        fcs1 = Dense(fcs1_units, activation='relu')(state)
        bn1 = BatchNormalization(axis=1, epsilon=1e-5)(fcs1)
        fcs2 = Dense(fcs2_units)(bn1)
        fca1 = Dense(fca1_units)(action)
        added = Add()([fca1,fcs2])
        out = Dense(1,activation='relu') (added)

        self.model = Model(inputs=[state, action],outputs=out)
        self.model.compile(loss="mse", optimizer=Adam(lr=lrn_rate))
        self.model.summary()

        self.action_grads = K.function([self.model.input[0], self.model.input[1]], K.gradients(self.model.output, [self.model.input[1]]))

    def get_gradients(self, states, actions):
        return self.action_grads([states, actions])

class DDPGSolverBPW():
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random_seed

        self.actor_local = Actor(state_size, action_size, random_seed, LR_ACTOR)
        self.actor_target = Actor(state_size, action_size, random_seed, LR_ACTOR)
        
        self.critic_local = Critic(state_size, action_size, random_seed, LR_CRITIC)
        self.critic_target = Critic(state_size, action_size, random_seed, LR_CRITIC)
        
        self.soft_update(self.critic_local.model, self.critic_target.model, 1)
        self.soft_update(self.actor_local.model, self.actor_target.model, 1)
     
        self.noise = OUNoise(action_size, random_seed)
        self.memory = deque(maxlen=BUFFER_SIZE)
    
    def remember(self, state, action, reward, state_next, done):
        self.memory.append((state, action, reward, state_next, done))
    

    def get_action(self, state, add_noise=True):
        action = self.actor_local.model.predict(state)[0]
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states = np.float32([arr[0] for arr in batch])
        actions = np.int32([arr[1] for arr in batch])
        rewards = np.float32([arr[2] for arr in batch])
        state_nexts = np.float32([arr[3] for arr in batch])
        terminals = np.float32([arr[4] for arr in batch])

        actions_next = self.actor_target.model.predict(state_nexts)
        Q_targets_next = self.critic_target.model.predict(x=[state_nexts, actions_next])
        Q_targets = Q_targets_next
        for i in range(BATCH_SIZE):
            Q_targets[i] = rewards[i] + (GAMMA * Q_targets_next[i] * (1 - terminals[i]))

        self.critic_local.model.fit(x=[states, actions],y=Q_targets,batch_size=BATCH_SIZE,verbose=0)
        
        actions = self.actor_local.model.predict(states)
        grads = self.critic_target.get_gradients(states, actions)

        self.actor_local.optimize([states, np.array(grads).reshape((-1, self.action_size))])

        self.soft_update(self.critic_local.model, self.critic_target.model, TAU)
        self.soft_update(self.actor_local.model, self.actor_target.model, TAU)                     


    def soft_update(self, local_model, target_model, tau):
        target_wts = np.array(target_model.get_weights())
        q_wts = np.array(local_model.get_weights())
        target_model.set_weights((target_wts * (1-tau)) + (q_wts * tau))

def bipedalwalkerDDPG():
    env = gym.make(ENV_NAME)
    env.seed(10)
    observation_space = env.observation_space.shape[0]
    ddpgsolver = DDPGSolverBPW(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=10)

    scores_deque = deque(maxlen=100)
    for ep in range(1, MAX_EPISODES+1):
        state = env.reset()
        ddpgsolver.reset()
        total_reward = 0
        for step in range(MAX_STEPS):
            env.render()
            state = np.reshape(state, [1, observation_space])
            action = ddpgsolver.get_action(state)
            state_next, reward, terminal, info = env.step(action)

            state = np.reshape(state, [observation_space])
            ddpgsolver.remember(state, action, reward, state_next, terminal)
            state = state_next
            total_reward += reward
            if terminal:
                scores_deque.append(total_reward)
                print("Run: " + str(ep) + ", Average Score: " + str(np.mean(scores_deque)) + ", score: " + str(total_reward))
                data = [ep,total_reward,np.mean(scores_deque)]
                with open(record_filename, "ab") as f:
                        pickle.dump(data, f)  
                break 
            ddpgsolver.experience_replay()
        
        if ep % 100 == 0:
            ddpgsolver.actor_local.model.save_weights('./checkpoints/checkpoint_actor'+str(ep))
            ddpgsolver.critic_local.model.save_weights('./checkpoints/checkpoint_critic'+str(ep))
            print('Model Saved for episode ',ep)

if __name__ == '__main__':
    bipedalwalkerDDPG()