import os
import gym
from gym import spaces

import sys

import pytorch_rl.teacher as teacher
import gym_minigrid
import gym

def make_env(env_id, seed, rank,useTeacher):
    def _thunk():
        
        env=gym.make(env_id)
        if useTeacher:            
            print('adding Teacher...')
            env=teacher.Teacher(env)
            print('done!')
        else:
            print('no Teacher')

        env.seed(seed + rank)
        #if log_dir is not None:
        #    env = bench.Monitor(env, os.path.join(log_dir, str(rank)))

        # Maxime: until RL code supports dict observations, squash observations into a flat vector
        #if isinstance(env.observation_space, spaces.Dict):
        #    print('dic state not supported. we use a Flat wrapper')
        #    env = FlatObsWrapper(env)
        env=WrapPyTorch(env)
        return env

    return _thunk

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.spaces['image'].shape
        self.observation_space = spaces.Box(
            self.observation_space.spaces['image'].low[0,0,0],
            self.observation_space.spaces['image'].high[0,0,0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],dtype=float
        )

    def observation(self, observation):
        #print('observation', observation)
        #print(observation)
        observation['image']=observation['image'].transpose(2, 0, 1)
        return (observation)
    
    #def _observation(self, observation):
        #return observation['image'].transpose(2, 0, 1)
    
    

