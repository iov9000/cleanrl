import sys
import numpy as np

if __name__ == '__main__':
    demo_name = sys.argv[1]
    demo = np.load(demo_name)
    
    num_episodes = np.sum(demo[:,-1])
    rew_avg = np.mean(demo[:,-2])
    rew_std = np.std(demo[:,-2])
    rew_min = np.min(demo[:,-2])
    rew_max = np.max(demo[:,-2])
    
    ep_rew_list = []
    ep_rew = 0
    for sard in demo:
        ep_rew += sard[-2]
        if sard[-1] == 1:
            ep_rew_list.append(ep_rew)
            print("episode_reward", ep_rew)
            ep_rew = 0
        
    ep_rew_avg = np.mean(ep_rew_list)
    ep_rew_std = np.std(ep_rew_list)
    ep_rew_min = np.min(ep_rew_list)
    ep_rew_max = np.max(ep_rew_list)


    print("Demo file stats")
    print(demo_name)
    print("-------------")
    print("Number of episodes: ", num_episodes)
    print("Reward stats: ", rew_avg, " +- ", rew_std) 
    print("Reward min / max", rew_min, " / ", rew_max) 
    print("Episode reward stats: ", ep_rew_avg, " +- ", ep_rew_std) 
    print("Episode reward min / max", ep_rew_min, " / ", ep_rew_max) 


