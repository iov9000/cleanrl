# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

from datetime import datetime

import gymnasium as gym

import numpy as np
#import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from mg.buffer import MultiStepReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple, deque

BufferData = namedtuple("BufferData",
                        ["observations", "actions", "rewards",
                         "next_observations", "next_actions", "next_rewards",
                         "dones", "indices", "ki", "probs"])

from irl.utils import FrankaObsWrapper, evaluate_model

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
    parser.add_argument('--save_eval', default=False, required=False, 
        action='store_true', help='verbose output')
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

    # environment specific arguments (wrappers)
    parser.add_argument('--normalize_obs', default=False, 
        action='store_true', required=False, help='normalize obs')  
    parser.add_argument('--clip_action', default=False, 
        action='store_true', required=False, help='clip action')  

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v3",
        help="the id of the environment")
    parser.add_argument("--reward_components", type=str, default="", nargs='+',  help="reward_components to change")
    parser.add_argument("--reward_components_values", type=float, nargs='+', help="reward_components values")

    # multigrid specific arguments
    parser.add_argument("--nstep_distribution", type=str, default='ucb_gradient_bandit', help="distribution over scales")
    # parser.add_argument("--ks", type=int, nargs='+', default=[1,10], help="multigrid scales")
    parser.add_argument("--ks", type=int, default=10, help="multigrid scales")
    parser.add_argument("--mode", type=str, default='', required=False, help='mg or last')  
    parser.add_argument("--n_ensemble", type=int, default=10, help="ensemble size")
    parser.add_argument("--ucb_reset_interval", type=int, default=50000, help="when to reset / update UCB arms")
    parser.add_argument("--ucb_update_rule", type=str, default='minvar', required=False, help='minvar/maxvar/else')  
    parser.add_argument("--ucb_polyak_avg", type=float, default=1.0, required=False, 
                        help='Polyak averaging update for n-step arms')
    parser.add_argument("--prioritized_replay", default=False, 
        action='store_true', required=False, help='prioritize buffer items based on td-error')

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
    parser.add_argument("--policy-frequency", type=int, default=1,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument('--verbose', default=False, required=False, 
        action='store_true', help='verbose output')

    args = parser.parse_args()

    non_default = {
        opt.dest: getattr(args, opt.dest)
        for opt in parser._option_string_actions.values()
        if hasattr(args, opt.dest) and opt.default != getattr(args, opt.dest)
    }

    return args, non_default


def make_env(opt, env_id, seed, idx, capture_video, run_name, kwargs):
    def thunk():
        if 'Franka' in env_id:
            env = gym.make(env_id, tasks_to_complete=['microwave'])
            env = FrankaObsWrapper(env)
        else:
            env = gym.make(env_id, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        #env.seed(seed)

        if opt.clip_action and isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)
        if opt.normalize_obs and isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env, lambda obs: np.clip(obs, -10, 10))

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

class HeadNet(nn.Module):
    def __init__(self, reshape_size, n_actions=4):
        super(HeadNet, self).__init__()
        self.fc1 = nn.Linear(reshape_size, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CoreNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(
            env.single_observation_space.shape).prod() + np.prod(
            env.single_action_space.shape), 256)
        self.reshape_size = 256

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x


class EnsembleNet(nn.Module):
    def __init__(self, env, n_actions=1, n_ensemble=1):
        super(EnsembleNet, self).__init__()
        self.core_net = CoreNet(env)
        reshape_size = self.core_net.reshape_size
        self.net_list = nn.ModuleList(
            [HeadNet(reshape_size=reshape_size, n_actions=n_actions)
             for _ in range(n_ensemble)])

    def _core(self, x, a=None):
        if a is not None:
            x = torch.cat([x, a], 1)
        return self.core_net(x)

    def _heads(self, x):
        return [net(x) for net in self.net_list]

    def forward(self, x, a, k=None):
        x = torch.cat([x, a], 1)
        if k is not None:
            return self.net_list[k](self.core_net(x))
        else:
            core_cache = self._core(x)
            net_heads = self._heads(core_cache)
            all_heads = torch.stack(net_heads, dim=0).squeeze()
            return all_heads


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        if isinstance(env.observation_space, gym.spaces.Dict):
            ob_shape = env.observation_space['observation'].shape
        else:
            ob_shape = env.single_observation_space.shape

        self.fc1 = nn.Linear(np.array(ob_shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
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

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    args, non_default = parse_args()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    # create output folder with non default arguments in the name
    id_string = str(non_default).replace(' ','').replace(':','').replace('Namespace','').replace('{','').replace('}','').replace("'","_").replace(",","_")
    run_name = f"{ts}_{args.env_id}__{args.exp_name}_seed_{args.seed}_{id_string}"
    if len(run_name) > 220:
        run_name = run_name[:220]
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

    env_kwargs = {}
    for i, comp in enumerate(args.reward_components):
        env_kwargs[comp] = args.reward_components_values[i]

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args, args.env_id, args.seed, 0, args.capture_video, run_name, env_kwargs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    if 'Adroit' in args.env_id:
        test_env = gym.make(args.env_id, max_episode_steps=500)
    elif 'Franka' in args.env_id:
        test_env = gym.make(args.env_id, tasks_to_complete=['microwave', 'kettle', 'light switch'])
        test_env = FrankaObsWrapper(test_env)
    else:
        test_env = gym.make(args.env_id, **env_kwargs) 
    
    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = EnsembleNet(
        envs, n_actions=1, n_ensemble=args.n_ensemble).to(device)
    qf1_target = EnsembleNet(
        envs, n_actions=1, n_ensemble=args.n_ensemble).to(device) 
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.q_lr)
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
    if isinstance(envs.observation_space, gym.spaces.Dict):
        ob_space = envs.observation_space['observation']
    else:
        ob_space = envs.single_observation_space

    # rb = ReplayBuffer(
    #     args.buffer_size,
    #     ob_space,
    #     envs.single_action_space,
    #     device,
    #     handle_timeout_termination=True,
    # )
    ks = [args.ks]
    ks = np.sort(ks)[::-1]
    assert len(ks) == 1, "only one ki is supported"

    rb = MultiStepReplayBuffer(args,
            observation_shape=envs.single_observation_space.shape,
            action_shape=envs.single_action_space.shape,
            stack_size=1,
            batch_size=args.batch_size,
            replay_capacity=args.buffer_size,
            update_horizon=ks[0],
            nstep_distribution=args.nstep_distribution,
            mode=args.mode,
            gamma=args.gamma,
            observation_dtype=envs.single_observation_space.dtype,
            action_dtype=envs.single_action_space.dtype)

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset(seed=args.seed)[0]
    best_rew = -100000
    average10 = deque(maxlen=10)
    average100 = deque(maxlen=100)
    kis = []

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, term, trunc, infos = envs.step(actions)
        dones = term or trunc

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if not isinstance(infos, list):
            infos = [infos]

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for item in infos:
            if "final_info" in item.keys():
                irl_trigger = True
                if item["final_info"][0]['episode']["r"] > best_rew:
                    best_rew = item["final_info"][0]['episode']["r"]
                    torch.save(actor.state_dict(), f"runs/{run_name}/best_actor.pt")
                    torch.save(qf1.state_dict(), f"runs/{run_name}/best_qf1.pt")

                if args.verbose:
                    print(
                        f"global_step={global_step}, episodic_return={item['final_info'][0]['episode']['r']}")
                if global_step % 1 == 0:
                    writer.add_scalar(
                        "charts/episodic_return", item["final_info"][0]['episode']["r"], global_step)
                    writer.add_scalar(
                        "charts/episodic_length", item["final_info"][0]['episode']["l"], global_step)
                break


        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["final_observation"][0]
        # rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        if args.prioritized_replay:
            priority = rb.sum_tree.max_recorded_priority
        else:
            priority = 1.
        
        # TODO: why only [0]?
        rb.add(obs, actions, rewards[0], term[0])
        #rb.add(obs, actions, rewards, dones, priority=priority)
        
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            batch = rb.sample_transition_batch(args.batch_size)
            # see definition of namedtuple above
            data = BufferData(
                torch.tensor(batch[0]).squeeze().to(device),
                torch.tensor(batch[1]).to(device),
                torch.tensor(batch[2]).to(device),
                torch.tensor(batch[3]).squeeze().to(device),
                torch.tensor(batch[4]).to(device),
                torch.tensor(batch[5]).to(device),
                torch.tensor(batch[6]).to(device),
                batch[7],
                batch[8],
                None)

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data.next_observations)
                qf1_next_targets = qf1_target(data.next_observations,
                                              next_state_actions)
                qf1_next_targets = qf1_next_targets - alpha * next_state_log_pi.squeeze()
                discount = torch.tensor(
                    args.gamma ** data.ki, dtype=torch.float32).to(device)
                next_q_values = data.rewards.flatten() + (
                        1 - data.dones.flatten()) * discount * qf1_next_targets
            qf1_a_values = qf1(data.observations, data.actions)

            total_td_error = torch.abs(
                qf1_a_values - next_q_values).sum(dim=0).detach()
            weights = torch.ones_like(next_q_values[0]).to(device)
            if args.prioritized_replay:
                weights = data.weights
            # mask = torch.tensor(data.ki == (head_i + 1))
            mask = torch.ones_like(weights).to(device)

            total_q_loss = torch.mean(
                (qf1_a_values - next_q_values) ** 2 * weights * mask,
                dim=1).sum(dim=0)

            # optimize the model
            q_optimizer.zero_grad()
            total_q_loss.backward()
            q_optimizer.step()

            if args.prioritized_replay:
                rb.set_priority(data.indices, total_td_error)

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pis = qf1(data.observations, pi)

                    min_qf_pi = qf1_pis.min(dim=0)[0]
                    max_qf_pi = qf1_pis.max(dim=0)[0]
                    mean_qf_pi = qf1_pis.mean(dim=0)
                    std_qf_pi = qf1_pis.std(dim=0)

                    rb.update_mean_std(
                        mean_qf_pi.mean().item(),
                        std_qf_pi.mean().item(),
                        min_qf_pi.mean().item(),
                        max_qf_pi.mean().item(),
                        nstep=data.ki[0])
                    kis.append(data.ki[0])
                    assert data.ki.max() == data.ki.min(), "ki should be the same"

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

            if global_step % 100 == 0:
                writer.add_scalar("Train/Total TD-Error",
                                    total_td_error.mean().item(),
                                    global_step)
                for head_i in range(args.n_ensemble):
                    writer.add_scalar(f"Train/qf1_values_{head_i}",
                                        qf1_a_values[head_i].mean().item(),
                                        global_step)
                writer.add_scalar("Train/total_q_loss", total_q_loss.item(),
                                    global_step)
                writer.add_scalar("Train/actor_loss", actor_loss.item(),
                                    global_step)
                writer.add_histogram("Train/kis", data.ki, global_step)
                writer.add_histogram("Train/kis", np.array(kis),
                                        global_step, bins="auto")
                kis = []
                for head_i in range(ks[0]):
                    writer.add_scalar(f"UCB/mean_value_{head_i}",
                                        rb.ucb_mean[head_i],
                                        global_step)
                    writer.add_scalar(f"UCB/std_value_{head_i}",
                                        rb.ucb_std[head_i],
                                        global_step)
                    writer.add_scalar(f"UCB/nstep_count_{head_i}",
                                        rb.ucb_nstep_count[head_i],
                                        global_step)
                    writer.add_scalar(f"UCB/ucb_{head_i}",
                                        rb.ucb[head_i],
                                        global_step)
                writer.add_scalar("Train/alpha", alpha, global_step)
                if args.autotune:
                    writer.add_scalar("Train/alpha_loss", alpha_loss.item(),
                                        global_step)
                
                #run.log({"actor_loss": actor_loss.item(),
                #            "alpha": alpha,
                #            "alpha_loss": alpha_loss.item()
                #            if args.autotune else None})
                #for head_i in range(args.n_ensemble):
                #    run.log({f"qf1_values_{head_i}": qf1_a_values[head_i].mean().item()})

            #  evaluate at regular intervals and save
            if global_step % args.eval_interval == 0:
                if args.save_eval:
                    mu, std, d = evaluate_model(args, actor, device, test_env, save_eval=f"runs/{run_name}/traj_{global_step}.pkl")
                else:
                    mu, std, d = evaluate_model(args, actor, device, test_env)
                print(f"Evaluation at {global_step}: {mu}, {std}")
                writer.add_scalar('eval/mu', mu, global_step)
                writer.add_scalar('eval/std', std, global_step)

    envs.close()
    writer.close()
