# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from datetime import datetime
from distutils.util import strtobool

import gymnasium as gym

gym.envs.register(
     id='MoveP-v0',
     entry_point='envs.envs2d:MoveP',
     max_episode_steps=1000,
)

gym.envs.register(
     id='CMoveP-v0',
     entry_point='envs.envs2d:CMoveP',
     max_episode_steps=1000,
)

import numpy as np
#import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical

import ot 

from irl.swil import SwilReward, SWILDiscriminator
from irl.utils import load_hf_demos, evaluate_model, prepare_batch_update_irl,\
     enhance_demos_with_barycenters, demos_gen_dict, save_traj_dict

from parse_tb import read_tb_file, create_df_from_ea

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--exp-id", type=str, default="",
        help="additional identifier of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--logging_interval", type=int, default=500,
        help="logging interval")
    parser.add_argument("--eval_interval", type=int, default=10000,
        help="evaluation interval")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # environment specific arguments
    parser.add_argument('--normalize_obs', default=False, 
        action='store_true', required=False, help='normalize irl rewards')  
    parser.add_argument('--clip_action', default=False, 
        action='store_true', required=False, help='normalize irl rewards')  

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v3",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")

    # SWIL specific 
    parser.add_argument("--n_proj", type=int, default=1, help="number of projections for SWIL")
    parser.add_argument("--poly_degree", type=int, default=4, help="degree of polynomial for SWIL")
    parser.add_argument("--swil_reward_type", type=str, default='linear', help="SWIL reward type (linear / attn")
    parser.add_argument("--swil_rew", type=str, default='repl_loss', help="mean or replacement")
    parser.add_argument("--repl_loss_type", type=str, default='diff', help="replacement type: diff or a")
    parser.add_argument("--max_gsw", default=False, required=False, action='store_true', help="use max GSW loss")
    parser.add_argument("--swil_loss", default='surr_loss', type=str, required=False, help="loss_type:")
    parser.add_argument("--gsw_df", default='nn', type=str, required=False, help="Radon transform defining function: nn/poly/linear/circular")
    parser.add_argument('--measurable_space', default='sxa', type=str, 
        required=False, help='measurable space (s, sxs, sxa)')

    parser.add_argument("--proj_layer", default=False, required=False, action='store_true', 
                                        help="use nn instead of mean of projections")
    parser.add_argument("--linear_proj", default=False, required=False, action='store_true', 
                                        help="linear projection only")
    parser.add_argument("--swil_ae", default=False, required=False, action='store_true', help="use autoencoder")
    parser.add_argument("--i_c", type=float, default=0.1, help="degree of polynomial for SWIL")
    parser.add_argument("--min_beta", type=int, default=1.0, help="degree of polynomial for SWIL")
    parser.add_argument("--max_q_len", type=int, default=1, help="max number of projected atom sets in queue")
    parser.add_argument("--max_swil_step", type=int, default=100000000, help="only update projections until some timestep")
    parser.add_argument("--replace_atoms", default=False, required=False, action='store_true', help="actually replace atoms with policy samples")

    parser.add_argument("--use_sb_ppo", default=False, required=False, action='store_true', help="use autoencoder")

    # Discriminator specific arguments
    parser.add_argument("--demo_dir", type=str, default="./demos",
        help="expert demo folder")
    parser.add_argument("--n_demos", type=int, default=10,
        help="number of demos to use")
    parser.add_argument("--subsample", type=int, default=1,
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
    parser.add_argument('--irl_init_epochs', default=1  , type=int, required=False, help='number of discriminator updates')
    parser.add_argument('--irl_epochs', default=1, type=float, required=False, help='number of discriminator updates')
    parser.add_argument('--swil_period', default=1, type=float, required=False, help='number of discriminator updates')

    parser.add_argument('--lip_coeff', default=0, type=float, required=False, help='use gradient (spectral/mixup) penalty on discriminator')
    parser.add_argument('--gsw_coeff', default=1.0, type=float, required=False, help='ae loss coeff')
    parser.add_argument('--vb_coeff', default=1e-4, type=float, required=False, help='ae loss coeff')
    parser.add_argument('--recon_coeff', default=0.0, type=float, required=False, help='ae loss coeff')
    parser.add_argument('--l2_coeff', default=0.0, type=float, required=False, help='use l2 penalty on discriminator')
    parser.add_argument('--disc_lr', default=3e-4, type=float, required=False, help='discriminator learning rate')
    parser.add_argument("--scheduler_gamma", type=float, default=1.0,
        help="exponential decay rate of the learning rate scheduler")
    parser.add_argument("--swil_vb", type=int, default=0,
        help="variational bottleneck dimension")
    parser.add_argument('--use_cnn_base', default=False, required=False, 
        action='store_true', help='use cnn base for discriminator (only for visual obs)')
    parser.add_argument('--use_disc_bias', default=True, required=False, 
        action='store_false', help='use bias in disc arch')
    parser.add_argument('--use_weight_norm', default=False, required=False, 
        action='store_true', help='use weight norm')
    parser.add_argument('--use_ll_weight_norm', default=False, required=False, 
        action='store_true', help='use weight norm only in last layer')
    parser.add_argument('--use_spectral_norm', default=False, required=False, 
        action='store_true', help='use spectral_norm')
    parser.add_argument("--disc_nonlin", type=str, default="tanh",
        help="divergence type")
    parser.add_argument('--normalize_irl_rewards', default=True, 
        action='store_true', required=False, help='normalize irl rewards')  

    parser.add_argument('--on_policy', default=False, required=False, 
        action='store_true', help='use bias in disc arch')
    parser.add_argument("--resume", type=str, default=None, help="resume training from checkpoint")
    parser.add_argument("--load_disc", type=str, default=None, help="resume training from checkpoint")
    parser.add_argument("--eval_test_eps", type=float, default=0.0, help="test perturbation eps")
    parser.add_argument('--verbose', default=False, required=False, 
        action='store_true', help='verbose output')
    parser.add_argument('--drop_batch', default=False, required=False, 
        action='store_true', help='drop batch in irl epochs')   

    args = parser.parse_args()
    # fmt: on
    non_default = {
        opt.dest: getattr(args, opt.dest)
        for opt in parser._option_string_actions.values()
        if hasattr(args, opt.dest) and opt.default != getattr(args, opt.dest)
    }

    return args, non_default

def make_env(opt, env_id, seed, idx, capture_video, run_name, disc):
    def thunk():
        env = gym.make(env_id, render_mode='rgb_array')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        if opt.clip_action and isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)
        if opt.normalize_obs and isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env, lambda obs: np.clip(obs, -10, 10))
        env = SwilReward(env, opt, disc)        
        #env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class GumbelSoftmax(torch.distributions.RelaxedOneHotCategorical):
    '''
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob 
    is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    '''

    def sample(self, sample_shape=torch.Size()):
        '''Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical'''
        u = torch.empty(self.logits.size(), device=self.logits.device,
                        dtype=self.logits.dtype).uniform_(0, 1)
        noisy_logits = self.logits - torch.log(-torch.log(u))
        return torch.argmax(noisy_logits, dim=-1)

    def log_prob(self, value):
        '''value is one-hot or relaxed'''
        if value.shape != self.logits.shape:
            value = F.one_hot(value.long(), self.logits.shape[-1]).float()
            assert value.shape == self.logits.shape
        return - torch.sum(- value * F.log_softmax(self.logits, -1), -1)


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, env.single_action_space.n)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def choose_action(self, state, alpha, sample=False):
        with torch.no_grad():
            q = self.forward(state)
            logits = F.softmax(q/alpha, dim=1)
            # if sample:
            dist = Categorical(logits)
            action = dist.sample()  # if sample else dist.mean
            # else:
            #     action = torch.argmax(dist, dim=1)

        return action, logits

if __name__ == "__main__":
    args, non_default = parse_args()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    ua = 'ua' if args.use_actions else ''
    # create output folder with non default arguments in the name
    id_string = str(non_default).replace(' ','').replace(':','').replace('Namespace','').replace('{','').replace('}','').replace("'","_").replace(",","_")
    run_name = f"{ts}_{args.env_id}__{args.exp_name}_seed_{args.seed}_irm{args.irm_coeff}_lip{args.lip_coeff}_{ua}_{id_string}"
    if len(run_name) > 220:
        run_name = run_name[:220]
    results_dict = {'eval_mu': [], 'eval_std': []}

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)

    demos = load_hf_demos(args, n_demos=args.n_demos)
    if args.enhance_demos:
        print("Enhancing demos by computing Wasserstein barycenters")
        demos = enhance_demos_with_barycenters(demos)
    # get env specs
    test_env = gym.make(args.env_id)
    disc = SWILDiscriminator(test_env, args)

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args, args.env_id, args.seed, 0, args.capture_video, run_name, disc)])

    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)

    if args.resume is not None:
        qf1.load_state_dict(torch.load(os.path.join(args.resume, "best_qf1.pt") , map_location=device))
        qf2.load_state_dict(torch.load(os.path.join(args.resume, "best_qf2.pt") , map_location=device))
    
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    init = True # flag for IRL discriminator pretraining

    ######## gather a batch to define a SWIL distance so we don't lose initial samples
    obs = envs.reset(seed=args.seed)[0]
    b_obs, b_acs, b_dones = [], [], [] 
    #b_obs.append(torch.from_numpy(obs).type(torch.get_default_dtype()))
    for i in range(args.batch_size):
        actions = np.array([envs.single_action_space.sample()
                            for _ in range(envs.num_envs)])
        next_obs, rewards, term, trunc, infos = envs.step(actions)
        dones = term or trunc
        b_obs.append(torch.from_numpy(next_obs).type(torch.get_default_dtype()))
        b_acs.append(torch.from_numpy(actions).type(torch.get_default_dtype()))
        b_dones.append(torch.from_numpy(dones).type(torch.get_default_dtype()))

    b_obs = torch.stack(b_obs)
    b_acs = torch.stack(b_acs)
    b_dones = torch.stack(b_dones)

    gen_demos = demos_gen_dict(demos['all'], args.batch_size)

    for d in gen_demos:
        if len(d['obs']) < args.batch_size:
            obs_ = b_obs[:len(d['obs'])]
            actions_ = b_acs[:len(d['obs'])]
            dones_ = b_dones[:len(d['obs'])]
        else:
            obs_ = b_obs
            actions_ = b_acs
            dones_ = b_dones
        # GAIL Discriminator Update
        update_dict = prepare_batch_update_irl(envs, args, d, obs_, actions_, dones_, qf1)
        loss_dict = disc.compute_loss(update_dict)

        loss = loss_dict['d_loss'] + args.irm_coeff * loss_dict['grad_penalty'] + disc.beta * loss_dict['ib_loss'] 
        if args.irm_coeff > 1.0:
            loss /= args.irm_coeff

        disc.update(loss)

    #####
    obs = envs.reset(seed=args.seed)[0]
    best_rew = -100000
    for global_step in range(args.total_timesteps):
        irl_trigger = False
        try:
            # ALGO LOGIC: put action logic here
            if global_step < args.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                actions, _ = qf1.choose_action(torch.Tensor(obs).to(device), alpha)  
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, term, trunc, infos = envs.step(actions)
            dones = term or trunc

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if not isinstance(infos, list):
                infos = [infos]

            for item in infos:
                # TODO: fix this for new Gymnasium
                if "final_info" in item.keys():
                    irl_trigger = True
                    if item["final_info"][0]['episode']["l"] > best_rew:
                        best_rew = item["final_info"][0]['episode']["r"]
                        torch.save(qf1.state_dict(), f"runs/{run_name}/best_qf1.pt")
                        torch.save(qf2.state_dict(), f"runs/{run_name}/best_qf2.pt")
                        torch.save(disc.state_dict(), f"runs/{run_name}/best_disc.pt")

                    if args.verbose:
                        print(
                            f"global_step={global_step}, episodic_return={item['final_info'][0]['episode']['r']}")
                    if global_step % 1 == 0:
                        writer.add_scalar(
                            "charts/episodic_return", item["final_info"][0]['episode']["r"], global_step)
                        writer.add_scalar(
                            "charts/episodic_return_irl", item["final_info"][0]['episode']["ep_rew_irl"], global_step)
                        writer.add_scalar(
                            "charts/episodic_length", item["final_info"][0]['episode']["l"], global_step)
                    break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d:
                    real_next_obs[idx] = infos[idx]["final_observation"][0]
                    # TODO: check if 0 index is always correct
            rb.add(obs, real_next_obs, actions, rewards, dones, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    q1_tgt = qf1_target(data.next_observations)
                    q2_tgt = qf2_target(data.next_observations)
                    target_v_1 = alpha * torch.logsumexp(q1_tgt/alpha, dim=1, keepdim=True)
                    target_v_2 = alpha * torch.logsumexp(q2_tgt/alpha, dim=1, keepdim=True)
                    # maybe use minimum here!
                    next_q_value_1 = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (target_v_1).view(-1)
                    next_q_value_2 = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (target_v_2).view(-1)

                qf1_a_values = qf1(data.observations)
                qf1_a_values = qf1_a_values.gather(1, data.actions.long()).view(-1)
                qf2_a_values = qf2(data.observations)
                qf2_a_values = qf2_a_values.gather(1, data.actions.long()).view(-1)

                qf1_loss = F.mse_loss(qf1_a_values, next_q_value_1)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value_2)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi = qf1.choose_action(data.observations, alpha)
                    alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

                # update the target networks
                if global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                if global_step % args.logging_interval == 0:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    if args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                
                # Optimize the discriminator
                if init:
                    n_irl_epochs = args.irl_init_epochs
                else:
                    n_irl_epochs = args.irl_epochs
                init = False

                irl_condition = global_step % args.swil_period == 0
                if args.swil_period == -1:
                    irl_condition = irl_trigger
                    
                if irl_condition and global_step < args.max_swil_step:
                    for epoch in range(n_irl_epochs):            
                        # GAIL Discriminator Update 
                        gen_demos = demos_gen_dict(demos['all'], args.batch_size)
                        for d in gen_demos:
                            if len(d['obs']) < args.batch_size:
                                obs_ = data.observations[:len(d['obs'])]
                                actions_ = data.actions[:len(d['obs'])]
                                dones_ = data.dones[:len(d['obs'])]
                            else:
                                obs_ = data.observations
                                actions_ = data.actions
                                dones_ = data.dones
                            # GAIL Discriminator Update
                            update_dict = prepare_batch_update_irl(envs, args, d, obs_, actions_, dones_, qf1)
                            loss_dict = disc.compute_loss(update_dict)

                            loss = loss_dict['d_loss'] + args.irm_coeff * loss_dict['grad_penalty'] + disc.beta * loss_dict['ib_loss'] 
                            if args.irm_coeff > 1.0:
                                loss /= args.irm_coeff

                            disc.update(loss)
                    
                            if global_step % args.logging_interval == 0:
                                writer.add_scalar('irl/swil_gsw_dist', loss_dict['d_loss'], global_step)

                                if args.irm_coeff > 0:
                                    writer.add_scalar('irl/swil_irm_loss', loss_dict['grad_penalty'], global_step)
                                if args.swil_vb > 0:
                                    writer.add_scalar('irl/swil_vib_loss', loss_dict['ib_loss'], global_step)

                # evaluate at regular intervals and save 
                if global_step % args.eval_interval == 0:
                    mu, std, d = evaluate_model(args, (qf1,alpha), device, test_env, wrap_mujoco=args.eval_test_eps>0)
                    print(f"Evaluation at {global_step}: {mu}, {std}")
                    writer.add_scalar('eval/mu', mu, global_step)
                    writer.add_scalar('eval/std', std, global_step)
                    ref_dist = ot.sliced_wasserstein_distance(
                        d['obs'], demos['all']['obs'][:len(d['obs'])], n_projections=100, seed=args.seed)
                    writer.add_scalar('irl/ref_sw_dist', ref_dist, global_step)

                    results_dict['eval_mu'].append(mu)
                    results_dict['eval_std'].append(std)

        except KeyboardInterrupt:
            break

    envs.close()

    # final eval
    torch.save(qf1.state_dict(), f"runs/{run_name}/qf1.pt")
    torch.save(qf2.state_dict(), f"runs/{run_name}/qf2.pt")
    torch.save(disc.state_dict(), f"runs/{run_name}/disc.pt")

    ## test model and save trajectories
    #test_env = MujocoTestWrapper(gym.make(args.env_id), eps=args.test_perturbation_eps)

    #  XXX f"runs/{args.exp_id}_dict.pkl" append to results dict
    mu, std, d = evaluate_model(args, (qf1,alpha), device, test_env)
    results_dict['final_mu'] = mu
    results_dict['final_std'] = std
    writer.add_scalar('eval/mu', mu, args.total_timesteps)
    writer.add_scalar('eval/std', std, args.total_timesteps)

    writer.close()

    # save df for easier parsing
    # ea = read_tb_file(f"runs/{run_name}")
    # df = create_df_from_ea(ea, "charts/episodic_return")
    # df.to_pickle(f"results_pkl/results_{run_name}_df.pkl")


