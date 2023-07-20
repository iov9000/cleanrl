import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import PIL.Image as Image
import gym
import random
import torch 
import time
import imageio
import datetime
from argparse import Namespace
import argparse 

import ot

from itertools import count
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from datetime import datetime
from tqdm.notebook import tqdm
#font = cv2.FONT_HERSHEY_COMPLEX_SMALL 

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# stable baselines imports
import stable_baselines3 as sb
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, EvalCallback
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO as PPOSB
from stable_baselines3 import TD3 as TD3SB
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import SAC as SACSB
from stable_baselines3 import DQN as DQNSB

from irl.swil import SwilIRDiffReward
from irl.utils import load_hf_demos, prepare_batch_update_irl, enhance_demos_with_barycenters, demos_gen_dict, MujocoTestWrapper, save_traj_dict

gym.envs.register(
     id='MoveP-v0',
     entry_point='envs.envs2d:MoveP',
     max_episode_steps=1000,
)

gym.envs.register(
     id='Lap-v0',
     entry_point='envs.env_lap:Laparos',
     max_episode_steps=10000,
)

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")

    parser.add_argument('--normalize_obs', default=True, 
        action='store_true', required=False, help='normalize observations')  
    parser.add_argument('--normalize_rewards', default=True, 
        action='store_true', required=False, help='normalize irl rewards')  
    parser.add_argument('--clip_action', default=True, 
        action='store_true', required=False, help='clip actions')

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="MoveP-v0",
        help="the id of the environment")
    parser.add_argument("--n_timesteps", type=int, default=10000000, #1e6
        help="total timesteps of the experiments")
    parser.add_argument("--lr", type=float, default=3e-4, # 3e-4
        help="the learning rate of the optimizer")
    parser.add_argument("--num_envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num_steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--agent_size", type=int, default=64, help='actor-critic network size')
    parser.add_argument("--gamma", type=float, default=0.95, # 0.99
        help="the discount factor gamma")
    parser.add_argument("--gae_lambda", type=float, default=0.9, #0.95
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num_minibatches", type=int, default=32, # 32
        help="the number of mini-batches")
    parser.add_argument("--update_epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--clip_coef", type=float, default=0.3,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent_coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None,
        help="the target KL divergence threshold")

    # environment specific arguments (Lap-v0)
    parser.add_argument('--goal_conditioning', default=True, 
        action='store_false', required=False, help='normalize irl rewards')  
    parser.add_argument('--done_on_collision', default=True, 
        action='store_false', required=False, help='normalize irl rewards') 
    parser.add_argument('--collision_loss', default=False, 
        action='store_true', required=False, help='end episode on collision')  
    parser.add_argument('--continuous_actions', default=False, 
        action='store_true', required=False, help='normalize irl rewards')  
    parser.add_argument('--render_meshes', default=True, 
        action='store_true', required=False, help='normalize irl rewards')  
    parser.add_argument("--step_size", type=float, default=0.3,
        help="step magnitude")
    parser.add_argument("--state_spec", type=str, default="r3",
        help="state spec (r3/r3so3)")

    # SWIL specific 
    parser.add_argument("--n_proj", type=int, default=100, help="number of projections for SWIL")
    parser.add_argument("--poly_degree", type=int, default=4, help="degree of polynomial for SWIL")
    parser.add_argument("--swil_reward_type", type=str, default='linear', help="SWIL reward type (linear / attn")
    parser.add_argument("--max_gsw", default=False, required=False, action='store_true', help="use max GSW loss")
    parser.add_argument("--swil_loss", default='gsw', type=str, required=False, help="loss_type:")
    parser.add_argument("--gsw_df", default='poly', type=str, required=False, help="Radon transform defining function: nn/poly/linear/circular")

    parser.add_argument("--proj_layer", default=False, required=False, action='store_true', 
                                        help="use nn instead of mean of projections")
    parser.add_argument("--linear_proj", default=False, required=False, action='store_true', 
                                        help="linear projection only")
    parser.add_argument("--swil_ae", default=False, required=False, action='store_true', help="use autoencoder")

    # Discriminator specific arguments
    parser.add_argument("--demo_dir", type=str, default="./demos",
        help="expert demo folder")
    parser.add_argument("--n_demos", type=int, default=10,
        help="number of demos to use")
    parser.add_argument('--enhance_demos', default=False, required=False, 
        action='store_true', help='enhance demos using W2 barycenters')
    parser.add_argument('--use_actions', default=True, required=False, 
        action='store_false', help='use actions in discriminator')
    parser.add_argument('--use_dones', default=False, required=False, 
        action='store_true', help='use actions in discriminator')
    parser.add_argument('--use_next_obs', default=False, required=False, 
        action='store_true', help='use actions in discriminator')
    parser.add_argument('--d_layer_dims', default=[128, 128], nargs='+', required=False,
                   help='discriminator MLP layer dimensions')
    parser.add_argument('--irm_coeff', default=0, type=float, 
        required=False, help='use IRM penalty on discriminator')
    parser.add_argument('--irl_batch_size', default=64, type=int, required=False, help='batch size for discriminator updates')
    parser.add_argument('--irl_init_epochs', default=0, type=int, required=False, help='number of discriminator updates')
    parser.add_argument('--irl_epochs', default=1, type=int, required=False, help='number of discriminator updates')
    parser.add_argument('--lip_coeff', default=0, type=float, required=False, help='use gradient (spectral/mixup) penalty on discriminator')
    parser.add_argument('--gsw_coeff', default=1.0, type=float, required=False, help='ae loss coeff')
    parser.add_argument('--recon_coeff', default=0.0, type=float, required=False, help='ae loss coeff')
    parser.add_argument('--l2_coeff', default=0.0, type=float, required=False, help='use l2 penalty on discriminator')
    parser.add_argument('--disc_lr', default=3e-4, type=float, required=False, help='discriminator learning rate')
    parser.add_argument('--use_cnn_base', default=False, required=False, 
        action='store_true', help='use cnn base for discriminator (only for visual obs)')
    parser.add_argument('--use_disc_bias', default=True, required=False, 
        action='store_true', help='use bias in disc arch')
    parser.add_argument("--disc_nonlin", type=str, default="tanh",
        help="divergence type")
    parser.add_argument('--normalize_irl_rewards', default=True, 
        action='store_false', required=False, help='normalize irl rewards')  

    # SWIL specific arguments
    parser.add_argument('--new_theta', default=False, 
        action='store_true', required=False, help='sample new direction of keep one during training')  
    parser.add_argument('--sw_poly_deg', default=2, type=int, 
        required=False, help='GSW polynomial degree')
    parser.add_argument('--sw_eval_freq', default=1, type=int, 
        required=False, help='GSW evaluation frequency')
    parser.add_argument('--sw_sample_size', default=1000, type=int, 
        required=False, help='GSW evaluation frequency')
    parser.add_argument('--swil_rew', default='replacement_loss', type=str, 
        required=False, help='SWIL loss type (atom insertion vs. atom replacement)')
    parser.add_argument('--radon_df_type', default='linear', type=str, 
        required=False, help='GRT defining function')
    parser.add_argument('--measurable_space', default='s', type=str, 
        required=False, help='GRT defining function')

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    non_default = {
        opt.dest: getattr(args, opt.dest)
        for opt in parser._option_string_actions.values()
        if hasattr(args, opt.dest) and opt.default != getattr(args, opt.dest)
    }

    # fmt: on
    return args, non_default

def plot_densities(policy_atoms, expert_atoms, theta, step, log_path):
    f = plt.gcf()
    ax = plt.gca()
    plt.scatter(10, 10, s=200, c='b', marker='o', alpha=1.0)
    plt.scatter(190, 190, s=200, c='r',
                marker='*', alpha=1.0)
    plt.scatter(expert_atoms[:, 0], expert_atoms[:, 1],
                s=20, c='g', alpha=0.3, label='expert')
    plt.scatter(policy_atoms[:, 0], policy_atoms[:, 1], s=20,
                c='m', alpha=0.3, label='agent')
    
    plt.axis('equal')
    plt.tight_layout()
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    #plt.legend()
    plt.savefig(os.path.join(log_path, f"swil_scatter_{step}.png"))
    plt.clf()

def plot_scatter_cdf(policy_atoms, expert_atoms, 
                    pa_subset, ea_subset, 
                    data1_sorted, data2_sorted, theta, 
                    sw, step, log_path):
    # Calculate the ECDFs
    x1 = torch.squeeze(data1_sorted).detach().numpy()
    # x1[5]+=0.1
    y1 = np.arange(1, len(x1) + 1) / len(x1)
    x2 = torch.squeeze(data2_sorted).detach().numpy()
    y2 = np.arange(1, len(x2) + 1) / len(x2)

    # Calculate the rectangle coordinates
    left = np.minimum(x1[:-1], x2[:-1])
    right = np.maximum(x1[:-1], x2[:-1])
    bottom = np.minimum(y1[:-1], y2[:-1])
    height = np.abs(y1[1:] - y2[:-1])

    # Create the figure and axes
    fig, ax = plt.subplots(1,3, figsize=(15, 5))

    # plot scatter
    #ax[0].scatter(10, 10, s=200, c='b', marker='o', alpha=1.0)
    #ax[0].scatter(190, 190, s=200, c='r',
    #            marker='*', alpha=1.0)
    ax[0].scatter(expert_atoms[:, 0], expert_atoms[:, 1],
                s=20, c='g', alpha=0.3, label='expert')
    ax[0].scatter(policy_atoms[:, 0], policy_atoms[:, 1], s=20,
                c='m', alpha=0.3, label='agent')
    ax[0].scatter(ea_subset[:, 0], ea_subset[:, 1],
                s=20, c='b', alpha=0.4, label='expert_ss')
    ax[0].scatter(pa_subset[:, 0], pa_subset[:, 1], s=20,
                c='r', alpha=0.4, label='agent_ss')

    # Plot the ECDFsg
    ax[1].scatter(x1, np.zeros_like(x1))
    ax[1].scatter(x2, np.zeros_like(x2))
    ax[1].step(x1, y1, label='Policy CDF')
    ax[1].step(x2, y2, label='Expert CDF')

    # Draw the rectangles
    for l, r, b, h in zip(left, right, bottom, height):
        rect = plt.Rectangle((l, b), r - l, h, alpha=0.3, facecolor='green')
        ax[1].add_patch(rect)

    # Add labels and legend
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[0].set_title('State occupancy for slice XY')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Cumulative Probability')
    ax[1].set_title('Distance between ECDFs')
    ax[1].legend()

    ax[2].plot(sw)
    ax[2].set_xlabel('steps')
    ax[2].set_ylabel('SW distance')
    ax[2].set_title('Distance between ECDFs')
    plt.savefig(os.path.join(log_path, f"swil_scatter_cdf_{step}.png"))
    plt.clf()

class SWILUpdateCallback(BaseCallback):
    def __init__(self, opt, demos, verbose=0, id="", log_path=None):
        super(SWILUpdateCallback, self).__init__(verbose)
        self.cnt = 0
        self.opt = opt
        self.eval_freq = opt.sw_eval_freq
        self.sw_dists = []
        self.demos = demos
        self.id = id
        self.log_path = log_path
        #self.aux_writer = SummaryWriter(log_path)

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True

    def on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # update the sorted slices and compute 
        if self.cnt%self.eval_freq == 0:
            # eval sw distance
            rb = self.locals['rollout_buffer']
            env = self.locals['env']
            n_demo_samples = min(self.opt.sw_sample_size, self.opt.num_steps)
            idx = np.random.randint(0, len(self.demos['obs'])-1, n_demo_samples)
            idx_rb = np.random.randint(0, len(rb.observations)-1, n_demo_samples)

            # M(SxS)?
            if self.opt.measurable_space == 'sxs':
                rb_sample = np.squeeze(np.concatenate(
                    [rb.observations[idx_rb], rb.observations[idx_rb+1]], -1))
                demo_sample = np.concatenate(
                    [self.demos['obs'][idx], self.demos['obs'][idx+1]], -1)
            elif self.opt.measurable_space == 'sxa':
                rb_sample = np.squeeze(np.concatenate(
                    [rb.observations[idx_rb], rb.actions[idx_rb]], -1))
                demo_sample = np.concatenate(
                    [self.demos['obs'][idx], self.demos['acs'][idx]], -1)
            else:
                rb_sample = np.squeeze(rb.sample(n_demo_samples).observations)
                demo_sample = np.concatenate([self.demos['obs'][idx]])

            # shuffling?
            policy_atoms = torch.Tensor(rb_sample)
            #policy_atoms = torch.Tensor(rb.observations)
            expert_atoms = torch.Tensor(demo_sample)

            if self.opt.max_gsw:
                sw_dist = env.envs[0].gsw.max_gsw(policy_atoms, expert_atoms)
            else:
                sw_dist = env.envs[0].gsw.gsw(policy_atoms, expert_atoms, new_theta=self.opt.new_theta)
            
            self.sw_dists.append(sw_dist.detach().numpy())
            theta = env.envs[0].gsw.theta
            self.logger.record("irl/sw_dist", sw_dist)
            p_sl = env.envs[0].gsw.Xslices_sorted[...,0]
            exp_sl = env.envs[0].gsw.Yslices_sorted[..., 0]
            # plot a projection, cdfs of projection and distances
            plot_scatter_cdf(rb.observations.squeeze(), self.demos['obs'], 
                            rb_sample, demo_sample, p_sl, exp_sl, 
                            theta, self.sw_dists, self.cnt, self.log_path)

        self.cnt +=1

    def _on_training_end(self) -> None:
        pass
    

if __name__ == "__main__":
    args, non_default = parse_args()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    # create output folder with non default arguments in the name
    id_string = str(non_default).replace(' ','').replace(':','').replace('Namespace','').replace('{','').replace('}','').replace("'","_").replace(",","_")
    run_name = f"{ts}_{args.env_id}__{args.exp_name}_seed_{args.seed}_np{args.n_proj}_{id_string}"

    # save model
    output_dir = f"runs/{run_name}_sb3"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    test_env = gym.make(args.env_id)
    seed = args.seed
    env = gym.make(args.env_id)
    env = SwilIRDiffReward(env, args)

    # seed environment
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    kwargs = {
        'learning_rate': args.lr,
        'n_steps': args.num_steps,
        'n_epochs': args.update_epochs
        #'buffer_size': 1000000,
    }
    demos = load_hf_demos(args.env_id, args.demo_dir, n_demos=args.n_demos)['all']
    #fig = plt.figure(figsize=(10,10))

    model = PPOSB('MlpPolicy', env, verbose=0, **kwargs, tensorboard_log=output_dir)
    #model = DQNSB('MlpPolicy', env, verbose=1, **kwargs, tensorboard_log=output_dir)
    # define callbacks
    swil_cb = SWILUpdateCallback(args, demos, log_path=output_dir)
    eval_cb = EvalCallback(
        test_env, best_model_save_path=output_dir, log_path=output_dir, 
        eval_freq=10000, deterministic=False)
    cb_list = CallbackList([eval_cb, swil_cb])
    model.learn(total_timesteps=args.n_timesteps, callback=cb_list)

    model.save(os.path.join(output_dir, 'model.sb3'))
    #save_expert_traj(env, model, nr_trajectories=10, render=False, 
    #                 filename=fname+str(n_timesteps), deterministic=save_deterministic)

