import sys
import os
import gymnasium as gym
# import pybullet
#import gym

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO, SAC, TD3, DQN
from sb3_contrib import TQC
from stable_baselines3.common.evaluation import evaluate_policy
from itertools import count
import torch

import argparse
#import safety_gymnasium

import pickle

gym.envs.register(
     id='CMovePND-v0',
     entry_point='envs.envs2d:CMovePND',
     max_episode_steps=1000)

def save_expert_traj_dict(opt, env, model):
    num_steps = 0
    expert_obs = []
    expert_acs = []
    expert_rew = []
    expert_done = []

    exp_support_obs = []
    exp_support_acs = []
    exp_support_rew = []
    exp_support_done = []

    env_name = opt.env_name
    nr_trajectories = args.nr_traj
    min_total_rew = args.min_rew
    extraflag = args.extraflag

    if isinstance(nr_trajectories, list):
        nr_trajectories = nr_trajectories[0]

    cnt = 0
    while cnt < nr_trajectories:
        ob = env.reset()[0]
        done = False
        total_reward = 0
        episode_obs = []
        episode_acs = []
        episode_rew = []
        episode_done = []
        
        support_obs = []
        support_acs = []
        support_rew = []
        support_done = []

        coinflip = np.random.rand() > 0.5
        if opt.action_noise > 0 and coinflip:
            noise = opt.action_noise
        else:
            noise = 0.0

        while not done:
            if args.custom_model is None:
                ac, _states = model.predict(ob)
            else:
                ob_ = torch.FloatTensor(ob).unsqueeze(0)
                if args.algo == 'ppo':
                    ac, _, _, _ = model.get_action_and_value(ob_)
                else: # SAC
                    ac, _, _ = model.get_action(ob_)

            ac = ac + np.random.randn(*ac.shape)*noise

            # save "backup" state
            if hasattr(env, 'model'):
                Qpos = env.sim.get_state().qpos
                Qvel = env.sim.get_state().qvel
            else:
                saved_state, agent, tgt = env.get_state()

            for i in range(opt.n_eps_walk):
                # reset to saved state on beginning of random walk
                if hasattr(env, 'model'):
                    env.set_state(Qpos, Qvel)
                else:
                    env.set_state(agent, tgt)

                for j in range(opt.len_eps_walk):

                    random_ac = env.action_space.sample()
                    next_ob_s, reward_s, term_s, trunc_s, _ = env.step(random_ac)
                    # TODO: figure out what to do with terminations..
                    done_s = term_s or trunc_s
                    support_obs.append(next_ob_s)
                    support_acs.append(random_ac)
                    support_rew.append(reward_s)
                    support_done.append(done_s)

            # carry on with previously saved state
            if hasattr(env, 'model'):
                env.set_state(Qpos, Qvel)
            else:
                env.set_state(agent, tgt)

            ac = ac.squeeze().detach().numpy()
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
            exp_support_obs.extend(support_obs)
            exp_support_acs.extend(support_acs)
            exp_support_rew.extend(support_rew)
            exp_support_done.extend(support_done)

    filename = f"demo_hf_{env_name}_{nr_trajectories}{extraflag}.pkl" 
    expert_obs = np.stack(expert_obs)
    expert_acs = np.stack(expert_acs)
    expert_rew = np.stack(expert_rew)
    expert_done = np.stack(expert_done)
    
    exp_support_obs = np.stack(exp_support_obs)
    exp_support_acs = np.stack(exp_support_acs)
    exp_support_rew = np.stack(exp_support_rew)
    exp_support_done = np.stack(exp_support_done)

    d = {'obs': expert_obs, 'acs': expert_acs, 'rew': expert_rew, 'done': expert_done,
         'support_obs': exp_support_obs, 'support_acs': exp_support_acs, 
         'support_rew': exp_support_rew, 'support_done': exp_support_done}

    pickle.dump(d, open(filename, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="HalfCheetah-v3",
                        help="the name of the environment")
    parser.add_argument("--env_kwargs", type=str, default="", nargs='+',  help="env kwargs")
    parser.add_argument("--env_kwargs_values", type=float, nargs='+', help="env kwargs values")

    parser.add_argument("--algo", type=str, default="sac",
                        help="algorithm")
    parser.add_argument("--nr_traj", default=0, type=int,
                        required=False, help="nr of trajectories to save")
    parser.add_argument("--min_rew", default=0, type=float,
                        required=False, help="minimum cum. rew. to save traj.")
    parser.add_argument("--eps0", default=0.0, type=float,
                        required=False, help="initial state perturbation")
    parser.add_argument("--epsdyn", default=0.0, type=float,
                        required=False, help="dynamics perturbation")
    parser.add_argument("--action_noise", default=0.0, type=float,
                        required=False, help="action perturbation")
    parser.add_argument("--n_eps_walk", default=5, type=int,
                        required=False, help="random walk number")
    parser.add_argument("--len_eps_walk", default=3, type=int,
                        required=False, help="random walk length")
    parser.add_argument("--perturb_parameter", default='body_mass', type=str, nargs='+',
                        required=False, help="specify which sim parameter to perturb")
    parser.add_argument('--record_video', default=False, required=False,
                        action='store_true', help='record video')
    parser.add_argument('--normalize_obs', default=False,
                        action='store_true', required=False, help='normalize obs')
    parser.add_argument("--custom_model", default=None, type=str,
                        required=False, help="use custom model to generate demos")
    parser.add_argument('--use_hparams_kwargs', default=False, required=False,
                        action='store_true', help='record video')
    parser.add_argument("--extraflag", type=str, default="_support", help="extraflag for name")

    args = parser.parse_args()

    env_kwargs = {}
    for i, comp in enumerate(args.env_kwargs):
        env_kwargs[comp] = args.env_kwargs_values[i]

    if args.custom_model is None:
        env = gym.make(args.env_name, **env_kwargs)

        checkpoint = load_from_hub(
            repo_id=f"sb3/{args.algo}-{args.env_name}",
            filename=f"{args.algo}-{args.env_name}.zip",
        )
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
        if args.algo == 'sac':
            model = SAC.load(checkpoint, custom_objects=custom_objects)
        elif args.algo == 'ppo':
            model = PPO.load(checkpoint, custom_objects=custom_objects)
        elif args.algo == 'dqn':
            model = DQN.load(checkpoint, custom_objects=custom_objects)
        elif args.algo == 'td3':
            model = TD3.load(checkpoint, custom_objects=custom_objects)
        elif args.algo == 'tqc':
            model = TQC.load(checkpoint, custom_objects=custom_objects)
    else:
        # TODO: train and load safety models!
        hparams = pickle.load(open(os.path.join(args.custom_model, "hparams.pkl"), 'rb'))

        model_env_kwargs = {}
        for i, comp in enumerate(hparams['env_kwargs']):
            model_env_kwargs[comp] = hparams['env_kwargs_values'][i]

        if args.use_hparams_kwargs:        
            env = gym.make(args.env_name, **model_env_kwargs)
        else:
            env = gym.make(args.env_name, **env_kwargs)

        if args.algo == 'sac':
            from sac_continuous_action import Actor, SoftQNetwork
            model = Actor(env)
            model.load_state_dict(torch.load(os.path.join(args.custom_model, "best_actor.pt")))
        elif args.algo == 'ppo':
            from ppo_univ import Agent
            model = Agent(env)
            model.load_state_dict(torch.load(os.path.join(args.custom_model, "best_actor.pt")))

    if args.normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)

    print(hparams)

    if hparams is not None:
        if hparams['clip_action'] and isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env) 
        if hparams['normalize_obs'] and isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env, lambda obs: np.clip(obs, -10, 10))

    save_expert_traj_dict(args, env, model)


    

    

