import sys
import os
# import gymnasium as gym
import gym
import numpy as np
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO, SAC, DQN, A2C, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from itertools import count

import pickle

# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

# from irl.utils import AntCFWrapper

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
        # ob, _ = env.reset(seed=1)
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
    expert_obs = np.stack(np.squeeze(expert_obs))
    expert_obs = np.reshape(expert_obs, (expert_obs.shape[0], 4,84,84))
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

    checkpoint = load_from_hub(
        repo_id=f"sb3/{algo}-{env_name}",
        filename=f"{algo}-{env_name}.zip",
    )
    custom_objects = {
      "learning_rate": 0.0,
      "lr_schedule": lambda _: 0.0,
      "clip_range": lambda _: 0.0,
    }
    if algo == 'sac':
        model = SAC.load(checkpoint, custom_objects=custom_objects)
    elif algo == 'ppo':
        model = PPO.load(checkpoint, custom_objects=custom_objects)
    elif algo == 'dqn':
        model = DQN.load(checkpoint, custom_objects=custom_objects)
    else:        
        model = A2C.load(checkpoint, custom_objects=custom_objects)



    env = gym.make(env_name)
    env = make_atari_env(env_name, n_envs=1, seed=0)
    # Stack 4 frames
    env = VecFrameStack(env, n_stack=4)
    #env = gym.wrappers.ResizeObservation(env, (84, 84))
    #env = gym.wrappers.GrayScaleObservation(env)
    #env = gym.wrappers.FrameStack(env, 4)

    save_expert_traj_dict(env_name, env, model, nr_traj, min_total_rew)


    

    

