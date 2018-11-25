import gym
env=gym.make("Taxi-v2")
env.reset()
env.observation_space.n
env.render()

env.action_space.n

env.env.s = 114
env.render()

env.step(1)
env.render()


(14, -1, False, {'prob': 1.0})

state, reward, done, info = env.step(1)

state = env.reset()
counter = 0
reward = None
while reward != 20:
    state, reward, done, info = env.step(env.action_space.sample())
    counter += 1

print(counter)

import numpy as np
Q = np.zeros([env.observation_space.n, env.action_space.n])

G = 0

alpha = 0.618

for episode in range(1,1001):
    done = False
    G, reward = 0,0
    state = env.reset()
    while done != True:
            action = np.argmax(Q[state]) #1
            state2, reward, done, info = env.step(action) #2
            Q[state,action] += alpha * (reward + np.max(Q[state2]) - Q[state,action]) #3
            G += reward
            state = state2   
    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode,G))
