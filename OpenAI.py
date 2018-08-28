import gym
env = gym.make(“Taxi-v2”)

import gym
env = gym.make(‘CartPole-v0’)
obs = env.reset()
for _ in range(1000):
env.render()
env.step(env.action_space.sample())
