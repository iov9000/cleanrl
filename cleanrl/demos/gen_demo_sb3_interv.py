import sys
import os
import argparse

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import gymnasium as gym

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
def train_expert(args, env, algo, n_timesteps, fname, n_traj, model=None, min_total_rew=-50000, label=None):

    if algo == 'ppo':
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

        if model is None:
            model = PPO('MlpPolicy', env, verbose=1, **ppo_kwargs)
            model.learn(total_timesteps=n_timesteps)
        else:
            model = PPO.load(model)
    elif algo == 'sac':
        if model is None:
            model = SAC('MlpPolicy', env, verbose=1)
            model.learn(total_timesteps=n_timesteps)
        else:
            model = SAC.load(model)

    model.save('model_' + fname + str(n_timesteps))
    save_expert_traj_dict(env.unwrapped.spec.id, env, model, nr_trajectories=n_traj, min_total_rew=min_total_rew, label=label)
    
    return model

def save_expert_traj_dict(env_name, env, model, nr_trajectories=10, min_total_rew=100, label=''):
    num_steps = 0
    expert_obs = []
    expert_acs = []
    expert_rew = []
    expert_done = []

    if isinstance(nr_trajectories, list):
        nr_trajectories = nr_trajectories[0]

    cnt = 0
    while cnt < nr_trajectories:
        ob, _ = env.reset()
        done = False
        total_reward = 0
        episode_obs = []
        episode_acs = []
        episode_rew = []
        episode_done = []

        while not done:
            ac, _states = model.predict(ob)
            next_ob, reward, term, trunc, _ = env.step(ac)
            done = term or trunc
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

    filename = f"demo_hf_{env_name}_{nr_trajectories}_{label}.pkl" 
    expert_obs = np.stack(expert_obs)
    expert_acs = np.stack(expert_acs)
    expert_rew = np.stack(expert_rew)
    expert_done = np.stack(expert_done)

    d = {'obs': expert_obs, 'acs': expert_acs, 'rew': expert_rew, 'done': expert_done}
    pickle.dump(d, open(filename, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Ant-v3",
                        help="the name of the environment")
    parser.add_argument("--algo", type=str, default="sac",
                        help="algorithm")
    parser.add_argument("--nr_traj", default=10, type=int,
                        required=False, help="nr of trajectories to save")
    parser.add_argument("--min_rew", default=0, type=float,
                        required=False, help="minimum cum. rew. to save traj.")
    parser.add_argument("--eps0", default=0.0, type=float,
                        required=False, help="initial state perturbation")
    parser.add_argument("--epsdyn", default=0.0, type=float,
                        required=False, help="dynamics perturbation")
    parser.add_argument("--perturb_parameter", default='body_mass', type=str, nargs='+',
                        required=False, help="specify which sim parameter to perturb")

    parser.add_argument("--reward_components", type=str, default="", nargs='+',  help="reward_components to change")
    parser.add_argument("--reward_components_values", type=float, nargs='+', help="reward_components values")

    parser.add_argument('--record_video', default=False, required=False,
                        action='store_true', help='record video')
    parser.add_argument('--normalize_obs', default=False,
                        action='store_true', required=False, help='normalize obs')
    parser.add_argument("--n_timesteps", default=1000000, type=int,
                        required=False, help="nr of timesteps to train")
    parser.add_argument("--model", default=None, type=str,
                        required=False, help="(optional) use pretrained model")
    parser.add_argument("--label", default='', type=str,
                        required=False, help="additional label for demo file")
    args = parser.parse_args()

    kwargs = {}
    for i, comp in enumerate(args.reward_components):
        kwargs[comp] = args.reward_components_values[i]

    env = gym.make(args.env_name, **kwargs)
    train_expert(args, env, args.algo, n_timesteps=args.n_timesteps, 
                                fname=args.env_name, 
                                n_traj=args.nr_traj, 
                                model=args.model,
                                min_total_rew=args.min_rew,
                                label = args.label + str(kwargs))


    

    

