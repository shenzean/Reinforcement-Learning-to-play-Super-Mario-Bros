from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.monitor import Monitor
import os 

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
monitor_dir = r'./monitor_log/'
os.makedirs(monitor_dir,exist_ok=True)
env = Monitor(env,monitor_dir)
env = GrayScaleObservation(env,keep_dim=True)
env = DummyVecEnv([lambda:env])
env = VecFrameStack(env,4,channels_order='last')

state = env.reset()

def show_img_4(state):
    plt.figure(figsize=(20,16))
    for idx in range (state.shape[3]):
        plt.subplot(1,4,idx+1)
        plt.imshow(state[0][:,:,idx])
    plt.show()

state, reward, done, info = env.step([env.action_space.sample()])
show_img_4(state)



