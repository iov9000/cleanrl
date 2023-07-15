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
    parser.add_argument("--num_steps", type=int, default=1024, # 2048
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--agent_size", type=int, default=64, help='actor-critic network size')
    parser.add_argument("--gamma", type=float, default=0.95, # 0.99
        help="the discount factor gamma")
    parser.add_argument("--gae_lambda", type=float, default=0.9, #0.95
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num_minibatches", type=int, default=32, # 32
        help="the number of mini-batches")
    parser.add_argument("--update_epochs", type=int, default=5, #10
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
    parser.add_argument('--new_theta', default=False, 
        action='store_true', required=False, help='sample new direction of keep one during training')  
    parser.add_argument('--sw_poly_deg', default=2, type=int, 
        required=False, help='GSW polynomial degree')
    parser.add_argument('--sw_eval_freq', default=10, type=int, 
        required=False, help='GSW evaluation frequency')
    parser.add_argument('--sw_sample_size', default=1000, type=int, 
        required=False, help='GSW evaluation frequency')
    parser.add_argument('--swil_rew', default='insertion_loss', type=str, 
        required=False, help='SWIL loss type (atom insertion vs. atom replacement)')
    parser.add_argument('--radon_df_type', default='linear', type=str, 
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


class SWILUpdateCallback(BaseCallback):
    def __init__(self, opt, demos, verbose=0, id="", log_path=None):
        super(SWILUpdateCallback, self).__init__(verbose)
        self.cnt = 0
        self.opt = opt
        self.eval_freq = opt.sw_eval_freq
        self.demos = demos
        self.id = id
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
        if self.cnt%self.eval_freq == 0:
            # eval sw distance
            rb = self.locals['replay_buffer']
            env = self.locals['env']
            idx = np.random.randint(0, len(self.demos['obs']), self.opt.sw_sample_size)

            policy_atoms = torch.Tensor(rb.sample(self.opt.sw_sample_size).observations)
            expert_atoms = torch.Tensor(self.demos['obs'][idx])

            if self.opt.max_gsw:
                sw_dist = env.envs[0].gsw.max_gsw(policy_atoms, expert_atoms)
            else:
                sw_dist = env.envs[0].gsw.gsw(policy_atoms, expert_atoms, new_theta=self.opt.new_theta)
            self.logger.record("irl/sw_dist", sw_dist)

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
        
    test_env = gym.make('MoveP-v0')
    seed = args.seed
    env = gym.make('MoveP-v0')
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
        'buffer_size': 10000,
    }
    demos = load_hf_demos(args.env_id, args.demo_dir, n_demos=args.n_demos)['all']

    #model = PPOSB('MlpPolicy', env, verbose=1, **kwargs, tensorboard_log=f"runs/{id_string}_sb3")
    model = DQNSB('MlpPolicy', env, verbose=1, **kwargs, tensorboard_log=output_dir)
    # define callbacks
    swil_cb = SWILUpdateCallback(args, demos)
    eval_cb = EvalCallback(
        test_env, best_model_save_path=output_dir, log_path=output_dir, 
        eval_freq=10000, deterministic=False)
    cb_list = CallbackList([eval_cb, swil_cb])
    model.learn(total_timesteps=args.n_timesteps, callback=cb_list)

    model.save(os.path.join(output_dir, 'model.sb3'))
    #save_expert_traj(env, model, nr_trajectories=10, render=False, 
    #                 filename=fname+str(n_timesteps), deterministic=save_deterministic)

