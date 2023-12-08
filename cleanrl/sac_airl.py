# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from datetime import datetime
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
#import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from irl.airl import AirlReward, AIRLDiscriminator
from irl.utils import load_hf_demos, evaluate_model, prepare_batch_update_irl,\
     enhance_demos_with_barycenters, demos_gen_dict, save_traj_dict, AbsorbAfterDoneWrapper, FrankaObsWrapper

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
    parser.add_argument("--capture_video_interval", type=int, default=200,
        help="capture video interval")
   
    # environment specific arguments
    parser.add_argument('--normalize_obs', default=False, 
        action='store_true', required=False, help='normalize irl rewards')  
    parser.add_argument('--clip_action', default=False, 
        action='store_true', required=False, help='normalize irl rewards')  
   
    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v3",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=3000000,
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

    # compat
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
        
    # Discriminator specific arguments
    parser.add_argument("--demo_dir", type=str, default="./demos",
        help="expert demo folder")
    parser.add_argument("--n_demos", type=int, default=10,
        help="number of demos to use")
    parser.add_argument("--subsample", type=int, default=1,
        help="number of demos to use")
    parser.add_argument('--enhance_demos', default=False, required=False, 
        action='store_true', help='enhance demos using W2 barycenters')
    parser.add_argument('--use_disc_bias', default=True, required=False, 
        action='store_false', help='use bias in disc arch')
    parser.add_argument('--use_actions', default=True, required=False, 
        action='store_false', help='use actions in discriminator')
    parser.add_argument('--d_layer_dims', default=[128, 128], nargs='+', required=False,
                   help='discriminator MLP layer dimensions')
    parser.add_argument('--airl_gamma', default=0.99, type=float, 
        required=False, help='gamma for AIRL advantage')
    parser.add_argument('--irm_coeff', default=0, type=float, 
        required=False, help='use IRM penalty on discriminator')
    parser.add_argument("--irm_loss_type", type=str, default="sym", help="e1,e0,p0,p1,sym")

    parser.add_argument('--airl_resnet', default=False, action='store_true', help='use resnet for discriminator')
    parser.add_argument('--d_out_reward', default=False, action='store_true', help='use d_out_reward')    
    parser.add_argument('--irl_epochs', default=1, type=float, required=False, help='number of discriminator updates')
    parser.add_argument('--lip_coeff', default=0, type=float, required=False, help='use gradient (spectral/mixup) penalty on discriminator')
    parser.add_argument('--l2_coeff', default=0, type=float, required=False, help='use l2 penalty on discriminator')
    parser.add_argument('--disc_lr', default=3e-4, type=float, required=False, help='discriminator learning rate')
    parser.add_argument('--disc_update_period', default=-1, type=int, required=False, help='discriminator update period')
    parser.add_argument("--disc_nonlin", type=str, default="tanh", help="nonlinearity for discriminator")
    parser.add_argument('--use_cnn_base', default=False, required=False, 
        action='store_true', help='use cnn base for discriminator (only for visual obs)')

    parser.add_argument('--use_weight_norm', default=False, required=False, 
        action='store_true', help='use weight norm')
    parser.add_argument('--use_spectral_norm', default=False, required=False, 
        action='store_true', help='use spectral_norm')
    parser.add_argument('--use_absorb_wrapper', default=False, required=False, 
        action='store_true', help='use_absorb_wrapper')

    parser.add_argument("--use_sb_ppo", default=False, required=False, action='store_true', help="use autoencoder")
    parser.add_argument("--div", type=str, default="rkl", help="divergence type")

    parser.add_argument("--resume", type=str, default=None, help="resume training from checkpoint")
    parser.add_argument("--load_disc", type=str, default=None, help="load discriminator from checkpoint")

    parser.add_argument('--drop_batch', default=False, required=False, 
        action='store_true', help='drop batch in irl epochs')  
    parser.add_argument('--on_policy', default=False, required=False, 
        action='store_true', help='use bias in disc arch')
    parser.add_argument('--verbose', default=False, required=False, 
        action='store_true', help='verbose output')

    args = parser.parse_args()
    # fmt: on
    non_default = {
        opt.dest: getattr(args, opt.dest)
        for opt in parser._option_string_actions.values()
        if hasattr(args, opt.dest) and opt.default != getattr(args, opt.dest)
    }

    return args, non_default


def make_env(opt, env_id, seed, idx, capture_video, run_name, agent, disc, demos):
    def thunk():
        if 'Adroit' in env_id:
            env = gym.make(env_id, max_episode_steps=500, render_mode='rgb_array')
            env.set_env_state(demos['all']['init_state_dict'][0])
        elif 'Franka' in env_id:
            env = gym.make(env_id, tasks_to_complete=['microwave', 'kettle', 'light switch'], render_mode='rgb_array')
            env = FrankaObsWrapper(env)
        else:
            env = gym.make(env_id, render_mode='rgb_array')        

        if opt.use_absorb_wrapper:  # DAC
            env = AbsorbAfterDoneWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger = lambda x: x % opt.capture_video_interval == 0)
        if opt.clip_action and isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)
        if opt.normalize_obs and isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env, lambda obs: np.clip(obs, -10, 10))
        env = AirlReward(env, opt, agent, disc)        
        #env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk



# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, a=None):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        if a is None:
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            y_t = torch.tanh(a)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(a)
            # Enforcing Action Bound
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    args, non_default = parse_args()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    ua = 'ua' if args.use_actions else ''
    # create output folder with non default arguments in the name
    id_string = str(non_default).replace(' ','').replace(':','').replace('Namespace','').replace('{','').replace('}','').replace("'","_").replace(",","_")
    run_name = f"{ts}_{args.env_id}__{args.exp_name}_seed_{args.seed}_irm{args.irm_coeff}_lip{args.lip_coeff}_{ua}_{id_string}"
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
    actor = Actor(test_env).to(device)


    disc = AIRLDiscriminator(test_env, args)

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args, args.env_id, args.seed, 0, args.capture_video, run_name, actor, disc, demos)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)

    if args.resume:
        actor.load_state_dict(torch.load(os.path.join(args.resume, "actor.pt") , map_location=device))
        qf1.load_state_dict(torch.load(os.path.join(args.resume, "qf1.pt") , map_location=device))
        qf2.load_state_dict(torch.load(os.path.join(args.resume, "qf2.pt") , map_location=device))
        disc.load_state_dict(torch.load(os.path.join(args.resume, "disc.pt") , map_location=device))

    if args.load_disc:
        disc.load_state_dict(torch.load(os.path.join(args.load_disc, "disc.pt") , map_location=device))

    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

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
    obs = envs.reset(seed=args.seed)[0]
    for global_step in range(args.total_timesteps):
        irl_trigger = False
        try:
            # ALGO LOGIC: put action logic here
            if global_step < args.learning_starts:
                actions = np.array([envs.single_action_space.sample()
                                   for _ in range(envs.num_envs)])
            else:
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
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
                    next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = qf1(data.observations, data.actions).view(-1)
                qf2_a_values = qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(
                        args.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = actor.get_action(data.observations)
                        qf1_pi = qf1(data.observations, pi)
                        qf2_pi = qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        if args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(data.observations)
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
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    if args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

                # Optimize the discriminator
                irl_condition = global_step % args.disc_update_period == 0
                if args.disc_update_period == -1:
                    irl_condition = irl_trigger
                    
                if irl_condition:                    
                    for epoch in range(args.irl_epochs):            
                        # GAIL Discriminator Update
                                    
                        gen_demos = demos_gen_dict(demos['all'], args.batch_size)
                        for d in gen_demos:
                            if len(d['obs']) < args.batch_size:
                                if args.drop_batch:
                                    break
                                else:
                                    obs_ = data.observations[:len(d['obs'])]
                                    actions_ = data.actions[:len(d['obs'])]
                                    dones_ = data.dones[:len(d['obs'])]
                            else:
                                obs_ = data.observations
                                actions_ = data.actions
                                dones_ = data.dones

                            # GAIL Discriminator Update
                            update_dict = prepare_batch_update_irl(envs, args, d,
                                            obs_, actions_, dones_, actor, compute_lprobs=True)

                            loss_dict = disc.compute_loss(update_dict)

                            loss = loss_dict['d_loss'] + args.irm_coeff * loss_dict['grad_penalty'] + args.lip_coeff * loss_dict['lip_penalty']
                            if args.irm_coeff > 1.0:
                                loss /= args.irm_coeff

                            disc.update(loss)

                            if global_step % args.logging_interval == 0:
                                writer.add_scalar('irl/airl_bce_loss', loss_dict['d_loss'], global_step)
                                if args.irm_coeff > 0:
                                    writer.add_scalar('irl/airl_irm_loss', loss_dict['grad_penalty'], global_step)
                                writer.add_scalar('irl/airl_r_p', loss_dict['policy_reward'], global_step)
                                writer.add_scalar('irl/airl_v_p', loss_dict['policy_value'], global_step)
                                writer.add_scalar('irl/airl_r_e', loss_dict['expert_reward'], global_step)
                                writer.add_scalar('irl/airl_v_e', loss_dict['expert_value'], global_step)
                                writer.add_scalar('irl/airl_bce_loss', loss_dict['d_loss'], global_step)

                #  evaluate at regular intervals and save
                if global_step % args.eval_interval == 0:
                    mu, std, d = evaluate_model(args, actor, device)
                    print(f"Evaluation at {global_step}: {mu}, {std}")
                    writer.add_scalar('eval/mu', mu, global_step)
                    writer.add_scalar('eval/std', std, global_step)
                    results_dict['eval_mu'].append(mu)
                    results_dict['eval_std'].append(std)

        except KeyboardInterrupt:
            break


    envs.close()

    #  eval
    torch.save(actor.state_dict(), f"runs/{run_name}/actor.pt")
    torch.save(qf1.state_dict(), f"runs/{run_name}/qf1.pt")
    torch.save(qf2.state_dict(), f"runs/{run_name}/qf2.pt")
    torch.save(disc.state_dict(), f"runs/{run_name}/disc.pt")

    # test model and save trajectories
    # test_env = MujocoTestWrapper(gym.make(args.env_id), eps=args.test_perturbation_eps)
    mu, std, d = evaluate_model(args, actor, device)
    results_dict['final_mu'] = mu
    results_dict['final_std'] = std
    writer.add_scalar('eval/mu', mu, args.total_timesteps)
    writer.add_scalar('eval/std', std, args.total_timesteps)

    writer.close()

    # save df for easier parsing
    # ea = read_tb_file(f"runs/{run_name}")
    # df = create_df_from_ea(ea, "charts/episodic_return")
    # df.to_pickle(f"results_{run_name}_df.pkl")
