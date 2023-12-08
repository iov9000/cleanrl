import sys
import os
import gymnasium as gym
#import gym
import numpy as np

import argparse

import pickle 

# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

# from irl.utils import AntCFWrapper

def save_expert_traj(env_name, env, model, nr_trajectories=10, min_total_rew=100):
    num_steps = 0
    expert_traj = []

    if isinstance(nr_trajectories, list):
        nr_trajectories = nr_trajectories[0]

    cnt = 0
    while cnt < nr_trajectories:
        ob = env.reset()[0]
        done = False
        total_reward = 0
        episode_traj = []

        while not done:
            ac, _states = model.predict(ob)
            next_ob, reward, term, trunc, _ = env.step(ac)
            done = term or trunc
            ob = next_ob
            total_reward += reward
            if env_name == 'Ant-v3':
                stacked_vec = np.hstack([np.squeeze(ob[:27]), np.squeeze(ac), reward, done])
            else:
                stacked_vec = np.hstack([np.squeeze(ob), np.squeeze(ac), reward, done])
            episode_traj.append(stacked_vec)
            num_steps += 1

        print("episode:", cnt, "reward:", total_reward)
        if total_reward > min_total_rew:
            cnt += 1
            expert_traj.extend(episode_traj)

    filename = f"demo_hf_{env_name}_{nr_trajectories}.npy" 
    expert_traj = np.stack(expert_traj)
    np.save(filename, expert_traj)

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
        ob = env.reset()[0]
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

    filename = f"demo_hf_{env_name}_{nr_trajectories}.pkl" 
    expert_obs = np.stack(expert_obs)
    expert_acs = np.stack(expert_acs)
    expert_rew = np.stack(expert_rew)
    expert_done = np.stack(expert_done)

    d = {'obs': expert_obs, 'acs': expert_acs, 'rew': expert_rew, 'done': expert_done}
    pickle.dump(d, open(filename, 'wb'))

def concat_key_in_demo_list(d_list, key):
    l = []
    for d in d_list:
        l.append(d[key])

    return np.concatenate(l, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="AdroitHandDoor-v1",
                        help="the name of the environment")
    parser.add_argument("--demofile", type=str, default="",
                        help="algorithm")

    args = parser.parse_args()

    demo = pickle.load(open(args.demofile,'rb'))
    env_name = args.env_name

    filename1 = f"demo_hf_{env_name}_1.pkl" 
    filename4 = f"demo_hf_{env_name}_4.pkl" 
    filename10 = f"demo_hf_{env_name}_10.pkl" 
    filename25 = f"demo_hf_{env_name}_25.pkl" 
    rews = []
    for dem in demo:
        dem['done'] = np.zeros(len(dem['observations']))
        dem['done'][-1] = 1
        rews.append(np.sum(dem['rewards']))

    r_srt = np.sort(rews)[::-1]
    r_idx = np.argsort(rews)[::-1]

    demo = [demo[i] for i in r_idx]

    for n in [1,4,10,25]:
        d = {'obs': concat_key_in_demo_list(demo[:n], 'observations'), 
             'acs': concat_key_in_demo_list(demo[:n], 'actions'), 
             'rew': concat_key_in_demo_list(demo[:n], 'rewards'), 
             'done': concat_key_in_demo_list(demo[:n], 'done'),
             'init_state_dict': [d['init_state_dict'] for d in demo]}
        filename = f"demo_hf_{env_name}_{n}.pkl" 
        pickle.dump(d, open(filename, 'wb'))




    

    

