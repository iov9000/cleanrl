import sys
import os

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import gym

gym.envs.register(
     id='MoveP-v0',
     entry_point='envs.envs2d:MoveP',
     max_episode_steps=1000)

gym.envs.register(
     id='CMoveP-v0',
     entry_point='envs.envs2d:CMoveP',
     max_episode_steps=1000)

gym.envs.register(
     id='MovePoint-v0',
     entry_point='envs.envs2d:MovePoint',
     max_episode_steps=1000)

import numpy as np
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from itertools import count

import pickle

# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

# from irl.utils import AntCFWrapper
def train_expert(env, n_timesteps, fname):

    ppo_kwargs = {
        'learning_rate':0.0003, 
        'n_steps': 2048, 
        'batch_size': 64, 
        'n_epochs': 3, 
        'gamma': 0.99, 
        'gae_lambda': 0.95, 
        'clip_range': 0.2, 
        'clip_range_vf':None, 
        'ent_coef': 0.0, 
        'vf_coef': 0.5, 
        #'max_grad_norm':0.5
    }


    model = PPO('MlpPolicy', env, verbose=1, **ppo_kwargs)
    model.learn(total_timesteps=n_timesteps)

    model.save('model_' + fname + str(n_timesteps))
    save_expert_traj_dict('CMoveP-v0', env, model, nr_trajectories=10, min_total_rew=-50000)
    
    return model

def save_expert_traj_dict(env_name, env, model, nr_trajectories=10, min_total_rew=100):
    num_steps = 0
    expert_obs = []
    expert_acs = []
    expert_rew = []
    expert_done = []

    if isinstance(nr_trajectories, list):
        nr_trajectories = nr_trajectories[0]

    cnt = 0
    while cnt < nr_trajectories:
        ob = env.reset()
        done = False
        total_reward = 0
        episode_obs = []
        episode_acs = []
        episode_rew = []
        episode_done = []

        while not done:
            ac, _states = model.predict(ob)
            next_ob, reward, done, _ = env.step(ac)
            ob = next_ob
            total_reward += reward
            episode_obs.append(ob)
            episode_acs.append(ac)
            episode_rew.append(reward)
            episode_done.append(done)

            num_steps += 1

        print("episode:", cnt, "reward:", total_reward)
        if total_reward > min_total_rew:
            cnt += 1
            expert_obs.extend(episode_obs)
            expert_acs.extend(episode_acs)
            expert_rew.extend(episode_rew)
            expert_done.extend(episode_done)

    filename = f"demo_hf_{env_name}_{nr_trajectories}.pkl" 
    expert_obs = np.stack(expert_obs)
    expert_acs = np.stack(expert_acs)
    expert_rew = np.stack(expert_rew)
    expert_done = np.stack(expert_done)

    d = {'obs': expert_obs, 'acs': expert_acs, 'rew': expert_rew, 'done': expert_done}
    pickle.dump(d, open(filename, 'wb'))

if __name__ == '__main__':
    env_name = sys.argv[1]
    algo = sys.argv[2]
    nr_traj = int(sys.argv[3])
    min_total_rew = int(sys.argv[4])

    env = gym.make(env_name, waypoints=[(175, 175)])
    train_expert(env, n_timesteps=1000000, fname='mp')


    

    

