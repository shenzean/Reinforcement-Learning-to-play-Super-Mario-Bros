from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time
from matplotlib import pyplot as plt
from stable_baselines3 import PPO

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

tensorboard_log = r'./tensorboard_log/'

model = PPO("CnnPolicy",env,verbose=1,
            tensorboard_log = tensorboard_log)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()



