from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time
from matplotlib import pyplot as plt
from stable_baselines3 import PPO

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

model = PPO.load('mario_model')

obs =env.reset()
obs = obs.copy()
done = True
while True:
    if done:
        state = env.reset()
    action, _state = model.predict(obs)
    obs, reward, done, info = env.step(action)
    obs = obs.copy()
    env.render()
