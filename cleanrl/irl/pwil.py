import gym
import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import Adam
from torch.nn.utils import spectral_norm
from irl.utils import ResNetAIRLDisc, MiniGridCNN
import copy
from sklearn import preprocessing
import ot

def get_trajectory_list(demos):
    done_cnt = 0
    trajs = []
    ep = []
    for i, t in enumerate(demos):
        ep.append(t)
        if t[-1] == 1:
            done_cnt += 1
            trajs.append(np.array(ep))
            ep = []
        
    return trajs

class PWILRewarder(object):
    """Rewarder class to compute PWIL rewards."""

    def __init__(self,
                 demonstrations,
                 subsampling,
                 env,
                 num_demonstrations=1,
                 time_horizon=1000.,
                 alpha=5.,
                 beta=5.,
                 observation_only=False):

        if num_demonstrations == -1:
            self.num_demonstrations = len(demonstrations)
        else:
            self.num_demonstrations = num_demonstrations 
        self.time_horizon = time_horizon
        self.subsampling = subsampling

        # Observations and actions are flat.
        ob_shapes = list(env.observation_space.shape)
        ac_shapes = list(env.action_space.shape)
        if len(ac_shapes) == 0:
            dim_act = 1
        else:
            dim_act = ac_shapes[-1]
        dim_obs = ob_shapes[-1]

        self.reward_sigma = beta * time_horizon / np.sqrt(dim_act + dim_obs)
        self.reward_scale = alpha

        self.observation_only = observation_only
        self.demonstrations = self.filter_demonstrations(get_trajectory_list(demonstrations))
        #self.vectorized_demonstrations = self.vectorize(self.demonstrations)
        if self.observation_only:            
            demos = np.concatenate(self.demonstrations)
            self.vectorized_demonstrations = demos[:,:dim_obs]
        else:
            demos = np.concatenate(self.demonstrations)
            self.vectorized_demonstrations = demos[:,:dim_obs+dim_act]
        
        self.scaler = self.get_scaler()
        self.reset()

    def filter_demonstrations(self, demonstrations):
        filtered_demonstrations = []
        print("Number of demonstrations: ", len(demonstrations))
        #np.random.shuffle(demonstrations)
        for episode in demonstrations[:self.num_demonstrations]:
            # Random episode start.
            #random_offset = np.random.randint(0, self.subsampling)
            random_offset = 1
            print("Random offset: ", random_offset)
            # Subsampling.
            print("Full episode length", len(episode))
            subsampled_episode = episode[random_offset::self.subsampling]
            print("Subsampled episode length", len(subsampled_episode))
            # Specify step types of demonstrations.
            # for transition in subsampled_episode:
            #     transition['step_type'] = dm_env.StepType.MID
            # subsampled_episode[0]['step_type'] = dm_env.StepType.FIRST
            # subsampled_episode[-1]['step_type'] = dm_env.StepType.LAST
            filtered_demonstrations.append(subsampled_episode)
        return filtered_demonstrations

    def get_scaler(self):
        """Defines a scaler to derive the standardized Euclidean distance."""
        scaler = preprocessing.StandardScaler()
        scaler.fit(self.vectorized_demonstrations)
        return scaler

    def reset(self):
        """Makes all expert transitions available and initialize weights."""
        self.expert_atoms = copy.deepcopy(
            self.scaler.transform(self.vectorized_demonstrations)
        )
        num_expert_atoms = len(self.expert_atoms)
        self.expert_weights = np.ones(num_expert_atoms) / (num_expert_atoms)

    def compute_reward(self, obs, action=None):
        """Computes reward as presented in Algorithm 1."""
        # Scale observation and action.
        if action is None:
            agent_atom = obs
        else:
            if not isinstance(action, int):
                action = np.array([action])
            if len(obs.shape) != len(action.shape):
                obs = np.expand_dims(obs, axis=0)
            agent_atom = np.squeeze(np.concatenate([obs, action], -1))

        agent_atom = np.expand_dims(agent_atom, axis=0)  # add dim for scaler
        agent_atom = self.scaler.transform(agent_atom)[0]

        # reset if list exhausted
        if (len(self.expert_atoms) == 1):
            self.reset()
        
        cost = 0.
        # As we match the expert's weights with the agent's weights, we might
        # raise an error due to float precision, we substract a small epsilon from
        # the agent's weights to prevent that.
        weight = 1. / self.time_horizon - 1e-6
        norms = np.linalg.norm(self.expert_atoms - agent_atom, axis=1)

        while weight > 0:
            # Get closest expert state action to agent's state action.
            argmin = norms.argmin()
            expert_weight = self.expert_weights[argmin]

            # Update cost and weights.
            if weight >= expert_weight:
                weight -= expert_weight
                cost += expert_weight * norms[argmin]
                self.expert_weights = np.delete(self.expert_weights, argmin, 0)
                self.expert_atoms = np.delete(self.expert_atoms, argmin, 0)
                norms = np.delete(norms, argmin, 0)
            else:
                cost += weight * norms[argmin]
                self.expert_weights[argmin] -= weight
                weight = 0

        reward = self.reward_scale * np.exp(-self.reward_sigma * cost)
        return reward.astype('float32')

    def compute_w2_dist_to_expert(self, trajectory):
        """Computes Wasserstein 2 distance to expert demonstrations."""
        self.reset()
        if self.observation_only:
            trajectory = [t['observation'] for t in trajectory]
        else:
            trajectory = [np.concatenate([t['observation'], t['action']])
                          for t in trajectory]

        trajectory = self.scaler.transform(trajectory)
        trajectory_weights = 1./len(trajectory) * np.ones(len(trajectory))
        cost_matrix = ot.dist(
            trajectory, self.expert_atoms, metric='euclidean')
        w2_dist = ot.emd2(trajectory_weights, self.expert_weights, cost_matrix)
        return w2_dist


class PWILReward(gym.Wrapper):
    def __init__(self, env, opt, demos):
        super().__init__(env=env)
        # self.reward_fn = make_network(**reward_fn_spec)
        self.use_actions = opt.use_actions
        self.n_demos = opt.n_demos
        self.subsampling=opt.subsampling
        self.use_actions = opt.use_actions
        self.demos = demos
        self.obs = None
        self.pwil = PWILRewarder(demos, subsampling=self.subsampling, env=env, 
                                    num_demonstrations=self.n_demos, 
                                    observation_only=(not opt.use_actions))

    def step(self, action):
        # also, here, WANT state before applying action
        next_obs, gt_reward, done, info = self.env.step(action)
        if self.obs is None:
            obs = next_obs
        else:
            obs = self.obs
        info['gt_reward'] = gt_reward
        # reset the pwil because the atoms are exhausted
        self.pwil.reset()
        if self.use_actions:
            reward = self.pwil.compute_reward(obs, action)
        else:
            reward = self.pwil.compute_reward(obs, action=None)

        self.obs = next_obs

        return next_obs, reward, done, info

    