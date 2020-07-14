import gym
import numpy as np
import time, pickle, os

env = gym.make('FrozenLake-v0')

epsilon = 0.9
total_episodes = 5000
max_steps = 100

lr_rate = 0.81
gamma = 0.95

Q = np.zeros((env.observation_space.n, env.action_space.n))
    
def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

def learn(state, state2, reward, action):
    predict = Q[state, action]
    target = reward + gamma * np.max(Q[state2, :])
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)

# Start
for episode in range(total_episodes):
    state = env.reset()
    t = 0
    
    while t < max_steps:
        env.render()

        action = choose_action(state)  

        state2, reward, done, info = env.step(action)  

        learn(state, state2, reward, action)

        state = state2

        t += 1
       
        if done:
            break

        time.sleep(0.1)

print(Q)

with open("frozenLake_qTable.pkl", 'wb') as f:
    pickle.dump(Q, f)
