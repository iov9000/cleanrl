import sys
import os
import gymnasium as gym
#import gym
import numpy as np

import argparse
import h5py
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
    print(len(d_list))
    for d in d_list:
        l.append(d[key])

    return np.concatenate(l, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="AdroidHandDoor-v1",
                        help="the name of the environment")
    parser.add_argument("--demofile", type=str, default="",
                        help="algorithm")
    parser.add_argument("--n_demos", type=int, default=1,
                        help="extract n demos from file")    
    parser.add_argument('--sorted', default=False, required=False, 
        action='store_true', help='pre-sort demos')

    args = parser.parse_args()

    #demo = pickle.load(open(args.demofile,'rb'))
    demo_file = h5py.File(args.demofile, 'r+')

    env_name = args.env_name

    obs = list(demo_file['observations'])
    acs = list(demo_file['actions'])
    rew = list(demo_file['rewards'])
    terms = list(demo_file['terminals'])
    truncs = list(demo_file['timeouts'])
    infos = list(demo_file['infos'])

    ends = np.where(terms)
    trajs = []
    bgn_idx = 0

    for e in ends[0]:
        traj = {}
        # cut last dim (TODO deal with it differently)
        traj['observations'] = demo_file['observations'][bgn_idx:e+1,:-1]
        traj['actions'] = demo_file['actions'][bgn_idx:e+1]
        traj['rewards'] = demo_file['rewards'][bgn_idx:e+1]
        traj['done'] = demo_file['terminals'][bgn_idx:e+1]
        traj['infos'] = demo_file['infos'][bgn_idx:e+1]
        bgn_idx = e+1

        trajs.append(traj)
    

    filename = f"demo_hf_{env_name}_{args.n_demos}.pkl" 

    if sorted:
        rews = []
        for dem in trajs:
            rews.append(np.sum(dem['rewards']))

        r_srt = np.sort(rews)[::-1]
        r_idx = np.argsort(rews)[::-1]

        print(r_srt)
        trajs_sorted = [trajs[i] for i in r_idx]
    else:
        trajs_sorted = trajs

    n = args.n_demos
    d = {'obs': concat_key_in_demo_list(trajs_sorted[:n], 'observations'), 
            'acs': concat_key_in_demo_list(trajs_sorted[:n], 'actions'), 
            'rew': concat_key_in_demo_list(trajs_sorted[:n], 'rewards'), 
            'done': concat_key_in_demo_list(trajs_sorted[:n], 'done'),
            'infos': [traj['infos'] for traj in trajs_sorted]}

    print(d['obs'].shape)
    print(d['acs'].shape)
    print(d['rew'].shape)
    print(d['done'].shape)

    filename = f"demo_hf_{env_name}_{n}.pkl" 
    pickle.dump(d, open(filename, 'wb'))




    

    

