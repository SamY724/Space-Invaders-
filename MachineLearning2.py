

#Importing required modules

import gym

from ale_py import ALEInterface

from ale_py.roms import SpaceInvaders

import numpy as np

from keras.models import Sequential

from keras.layers import Dense, Flatten, Convolution2D

import keras.optimizer_v2.adam

from rl.agents import DQNAgent

from rl.policy import EpsGreedyQPolicy,GreedyQPolicy

from rl.memory import SequentialMemory







ale = ALEInterface()

ale.loadROM(SpaceInvaders)



env = gym.make('SpaceInvaders', render_mode = 'human')





# Observing counts of states and actions given the environment

states = env.observation_space.shape[0]

actions = env.action_space.n

height,width,channels = env.observation_space.shape







# Creating a function to initialise the model architype we have selected

def createModel(height,width,channels,actions):

    model = Sequential()

    model.add(Convolution2D(32,(8,8),strides=(4,4), activation='relu', input_shape=(3,height,width,channels)))

    model.add(Convolution2D(64,(4,4),strides=(2,2),activation='relu'))

    model.add(Convolution2D(64,(3,3),activation='relu'))

    model.add(Flatten())

    model.add(Dense(512,activation='relu'))

    model.add(Dense(256,activation='relu'))

    model.add(Dense(actions,activation='linear'))

    return model



# Assigning model to a variable for easy calling/plugging into other functions

model = createModel(height,width,channels,actions)



 



# Creating a function to initialise the agent architype we have selected

def createAgent(model,actions):

    policy = EpsGreedyQPolicy(eps=0.1)

    memory = SequentialMemory(limit=1000, window_length = 3)

    dqn = DQNAgent(model=model, memory=memory, policy=policy, enable_dueling_network=True, dueling_type='avg', nb_actions = actions,nb_steps_warmup = 5000)

    return dqn





# Assigning the agent to a variable for ease of storage

# Compiling the agent variable in question

# Finally the agent is fitted with the specified envrionment, with a verbose of 2 for some information to be relayed back to the terminal

dqn = createAgent(model,actions)

dqn.compile(keras.optimizer_v2.adam.Adam(lr=1e-4))

dqn.fit(env,nb_steps = 1000000, visualize=False, verbose=2)





# Creating a variable to keep track of the scores of our agent when being tested against the environment with n-episodes

scores = dqn.test(env, nb_episodes=15, visualize=False)





# Finally, we print these scores out to the terminal for observation

print(f' The mean episode score' , np.mean(scores.history['episode_reward']))



 













