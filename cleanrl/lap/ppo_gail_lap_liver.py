# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import sys
import argparse
import os

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
sys.path.append(os.getcwd())

import random
import time
from datetime import datetime
from distutils.util import strtobool

import faulthandler; faulthandler.enable()

import matplotlib.pyplot as plt
import numpy as np
import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from torch.utils.tensorboard import SummaryWriter
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

import gymnasium as gym

gym.envs.register(
     id='Lap-v0',
     entry_point='envs.env_lap:Laparos',
     max_episode_steps=1000,
)

gym.envs.register(
     id='LapLiver-v0',
     entry_point='envs.env_lap_liver:LapLiver',
     max_episode_steps=500,
)


from irl.gail import GailReward, GAILDiscriminator
from irl.utils import load_hf_demos, load_hf_demos_name, load_lap_demos, prepare_batch_update_irl,\
                    enhance_demos_with_barycenters, demos_gen_dict, MujocoTestWrapper, save_traj_dict, evaluate_model

from parse_tb import read_tb_file, create_df_from_ea

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
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
    parser.add_argument('--goal_conditioning', default=False, 
        action='store_true', required=False, help='concatenate target to state')  
    parser.add_argument('--condition_on_n_tasks_completed', default=False, 
        action='store_true', required=False, help='add num tasks completed as conditioner')  
    parser.add_argument('--done_on_collision', default=True, 
        action='store_false', required=False, help='end episode on collision') 
    parser.add_argument('--collision_loss', default=False, 
        action='store_true', required=False, help='end episode on collision')  
    parser.add_argument('--hard_collisions', default=True, 
        action='store_false', required=False, help='dont progress on collision')  
    parser.add_argument('--continuous_actions', default=True, 
        action='store_false', required=False, help='use continuous actions')  
    parser.add_argument('--render_meshes', default=True, 
        action='store_true', required=False, help='render meshes when using env.render')  
    parser.add_argument("--step_size", type=float, default=3.0,
        help="step magnitude")
    parser.add_argument("--vis_threshold_factor", type=float, default=100.0,
        help="step magnitude")
    parser.add_argument("--state_spec", type=str, default="r3",
        help="state spec (r3/r3so3)")

    # environment specific arguments
    parser.add_argument('--normalize_obs', default=False, 
        action='store_true', required=False, help='normalize observations')  
    parser.add_argument('--clip_action', default=True, 
        action='store_true', required=False, help='clip action')  
    parser.add_argument('--time_aware', default=False, 
        action='store_true', required=False, help='use time aware wrapper and concatenate time to state')  

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="LapLiver-v0",
        help="the id of the environment")
    parser.add_argument("--env_kwargs", type=str, default="", nargs='+',  help="env kwargs")
    parser.add_argument("--env_kwargs_values", type=float, nargs='+', help="env kwargs values")

    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--agent_size", type=int, default=64, help='actor-critic network size')
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2000,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument('--anneal_lr', default=True, required=False, 
        action='store_false', help='anneal policy lr')
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--agent_nonlin", type=str, default="tanh",
        help="nonlinearity type")

    # Discriminator specific arguments
    parser.add_argument("--demo_dir", type=str, default="./demos",
        help="expert demo folder")
    parser.add_argument("--n_demos", type=int, default=10,
        help="number of demos to use")
    parser.add_argument('--enhance_barys', default=0, required=False, 
        type=int, help='enhance demos using W2 barycenters')
    parser.add_argument("--diverse_demos", type=str, default=None,
        help="diverse demos for generalization experiments")

    parser.add_argument('--use_actions', default=False, required=False, 
        action='store_true', help='use actions in discriminator')
    parser.add_argument('--use_dones', default=False, required=False, 
        action='store_true', help='use actions in discriminator')
    parser.add_argument('--use_next_obs', default=False, required=False, 
        action='store_true', help='use actions in discriminator')
    parser.add_argument('--d_layer_dims', default=[128, 128], nargs='+', required=False,
                   help='discriminator MLP layer dimensions')
    parser.add_argument('--irm_coeff', default=0, type=float, 
        required=False, help='use IRM penalty on discriminator')
    parser.add_argument("--irm_loss_type", type=str, default="sym", help="e1,e0,p0,p1,sym")
    parser.add_argument('--anneal_irm', default=False, required=False, 
        action='store_true', help='use bias in disc arch')
    parser.add_argument('--drop_batch', default=False, required=False, 
        action='store_true', help='drop batch in irl epochs')   
    
    parser.add_argument('--irl_epochs', default=1, type=int, required=False, help='number of discriminator updates')
    parser.add_argument('--irl_init_epochs', default=1, type=int, required=False, help='number of discriminator updates')
    parser.add_argument('--lip_coeff', default=0, type=float, required=False, help='use gradient (spectral/mixup) penalty on discriminator')
    parser.add_argument('--lip_p', default=1, type=float, required=False, help='use gradient (spectral/mixup) penalty on discriminator')

    parser.add_argument('--l2_coeff', default=0, type=float, required=False, help='use l2 penalty on discriminator')
    parser.add_argument('--disc_lr', default=3e-4, type=float, required=False, help='discriminator learning rate')
    parser.add_argument('--use_cnn_base', default=False, required=False, 
        action='store_true', help='use cnn base for discriminator (only for visual obs)')

    parser.add_argument('--use_weight_norm', default=False, required=False, 
        action='store_true', help='use weight norm')
    parser.add_argument('--use_spectral_norm', default=False, required=False, 
        action='store_true', help='use spectral_norm')
    parser.add_argument('--use_absorb_wrapper', default=False, required=False, 
        action='store_true', help='use_absorb_wrapper')
    parser.add_argument('--use_curr_traj_only', default=False, required=False, 
        action='store_true', help='only used last sampled trajectory to do matching')

    parser.add_argument('--use_disc_bias', default=True, required=False, 
        action='store_false', help='use bias in disc arch')
    parser.add_argument("--div", type=str, default="rkl",
        help="divergence type")
    parser.add_argument("--disc_nonlin", type=str, default="prelu",
        help="divergence type")
    parser.add_argument('--normalize_irl_rewards', default=True, 
        action='store_true', required=False, help='normalize irl rewards') 

    parser.add_argument("--use_sb_ppo", default=False, required=False, action='store_true', help="use autoencoder")

    parser.add_argument('--on_policy', default=True, required=False, 
        action='store_false', help='use bias in disc arch')
    parser.add_argument("--resume", type=str, default=None, help="resume training from checkpoint")
    parser.add_argument("--load_disc", type=str, default=None, help="resume training from checkpoint")
    parser.add_argument('--eval_on_different_seed', default=False, required=False, 
        action='store_true', help='evaluate on random seed')
    parser.add_argument("--test_perturbation_eps", type=float, default=0.0, help="test perturbation eps")
    parser.add_argument('--verbose', default=False, required=False, 
        action='store_true', help='verbose output')

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    # fmt: on
    non_default = {
        opt.dest: getattr(args, opt.dest)
        for opt in parser._option_string_actions.values()
        if hasattr(args, opt.dest) and opt.default != getattr(args, opt.dest)
    }
    cli = str(sys.argv).replace('--','').replace('verbose','').replace("'","")
    cli = cli.replace(" ","_").replace(",","").replace(".py","").replace('[','').replace(']','')

    return args, non_default, cli


def make_env(opt, env_id, seed, idx, capture_video, run_name, gamma, disc, demos):
    def thunk():
        env = gym.make(args.env_id, gc=opt.goal_conditioning, 
                             state_spec=opt.state_spec, 
                             step_size=opt.step_size,
                             done_on_collision=opt.done_on_collision,
                             hard_collisions=opt.hard_collisions,
                             plot_meshes=opt.render_meshes,
                             continuous_actions=opt.continuous_actions,
                             condition_on_n_tasks_completed=opt.condition_on_n_tasks_completed,
                             collision_loss=opt.collision_loss,
                             ref_demo = demos,
                             vis_threshold_factor = opt.vis_threshold_factor,
                             render_mode='rgb_array')        

        env = gym.wrappers.RecordEpisodeStatistics(env)
        if opt.time_aware:
            env = gym.wrappers.TimeAwareObservation(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger = lambda x: x % 500 == 0)
        if opt.clip_action and isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)
        if opt.normalize_obs and isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = GailReward(env, opt, disc)
        if opt.normalize_irl_rewards and isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        #env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Agent(nn.Module):
    def __init__(self, envs, hidden_size=64, nonlin='tanh', discrete=False):        
        super().__init__()

        if nonlin == 'relu':
            nonlin = nn.ReLU()
        elif nonlin == 'prelu':
            nonlin = nn.PReLU()
        else:
            nonlin = nn.Tanh()

        self.discrete = discrete
        if self.discrete:
            output_dim = int(envs.single_action_space.n)
        else:
            output_dim = np.prod(envs.single_action_space.shape)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_size)),
            nonlin,
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nonlin,
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_size)),
            nonlin,
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nonlin,
            layer_init(nn.Linear(hidden_size, output_dim), std=0.01),
        )
        if not self.discrete:
            self.actor_logstd = nn.Parameter(torch.zeros(1,output_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value_c(self, x, action=None, eval=False):
        action_mean = self.actor_mean(x)
        action_logstd = torch.tanh(self.actor_logstd.expand_as(action_mean))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (action_logstd + 1)
        action_std = torch.exp(log_std)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        if eval:
            return action_mean, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
        else:
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_action_and_value_d(self, x, action=None):
        logits = self.actor_mean(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_action_and_value(self, x, action=None, eval=False):
        if self.discrete:
            return self.get_action_and_value_d(x, action)
        else:
            return self.get_action_and_value_c(x, action)


if __name__ == "__main__":
    args, non_default, cli = parse_args()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    ua = 'ua' if args.use_actions else ''
    # create output folder with non default arguments in the name
    id_string = str(non_default).replace(' ','').replace(':','').replace('Namespace','').replace('{','').replace('}','').replace("'","_").replace(",","_")
    #run_name = f"{ts}_{args.env_id}__{args.exp_name}_seed_{args.seed}_irm{args.irm_coeff}_lip{args.lip_coeff}_{ua}_{id_string}"
    run_name = f"{ts}_{cli}".replace('/','_')
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
    #writer.add_hparams(vars(args))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    import pickle 
    pickle.dump(vars(args),open(f"runs/{run_name}/hparams.pkl",'wb'))

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
    for i, comp in enumerate(args.env_kwargs):
        env_kwargs[comp] = args.env_kwargs_values[i]

    if args.diverse_demos is not None:
        demo_list = [f for f in os.listdir(args.demo_dir) if args.diverse_demos in f and args.env_id in f]
        print("-> Training on union of demos:", demo_list)
        demos = load_hf_demos_name(args, demo_list)
    else:
        if 'Lap' in args.env_id:
            demos = load_lap_demos('demos/davos_data', args.n_demos, obs_shape=3, offset=100)
        else:
            demos = load_hf_demos(args.env_id, args.demo_dir, n_demos=args.n_demos)


    test_env = gym.make(args.env_id, gc=args.goal_conditioning, 
                             state_spec=args.state_spec, 
                             step_size=args.step_size,
                             done_on_collision=args.done_on_collision,
                             hard_collisions=args.hard_collisions,
                             plot_meshes=args.render_meshes,
                             continuous_actions=args.continuous_actions,
                             condition_on_n_tasks_completed=args.condition_on_n_tasks_completed,
                             collision_loss=args.collision_loss,
                             vis_threshold_factor = args.vis_threshold_factor,
)        
    if args.normalize_obs and isinstance(test_env.action_space, gym.spaces.Box):
            env = gym.wrappers.NormalizeObservation(test_env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    if args.clip_action and isinstance(test_env.action_space, gym.spaces.Box):
        test_env = gym.wrappers.ClipAction(test_env)

    if args.enhance_barys > 0:
        print("Enhancing demos by computing Wasserstein barycenters")
        demos = enhance_demos_with_barycenters(args, demos, test_env)

    # get env specs
    discrete = not isinstance(test_env.action_space, gym.spaces.Box)
    disc = GAILDiscriminator(test_env, args)
    if args.load_disc:
        disc.load_state_dict(torch.load(os.path.join(args.load_disc, "disc.pt") , map_location=device))

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args, args.env_id, args.seed + i, i, args.capture_video, run_name, args.gamma, disc, demos) for i in range(args.num_envs)]
    )
    #assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args.agent_size, discrete=discrete).to(device)
    if args.resume:
        agent.load_state_dict(torch.load(os.path.join(args.resume, "best_actor.pt") , map_location=device))

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset(seed=args.seed)[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    init = True

    # logging variables
    irl_count = 0
    best_rew = -10000

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, term, trunc, info = envs.step(action.cpu().numpy())
            done = term or trunc            
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            if not isinstance(info, list):
                info = [info]

            for item in info:
                if "final_info" in item.keys():
                    if item["final_info"][0]['episode']["r"] > best_rew:
                        best_rew = item["final_info"][0]['episode']["r"]
                        torch.save(agent.state_dict(), f"runs/{run_name}/best_actor.pt")
                        torch.save(disc.state_dict(), f"runs/{run_name}/best_disc.pt")

                    if args.verbose:
                        print(
                            f"global_step={global_step}, episodic_return={item['final_info'][0]['episode']['r']}")
                    if global_step % 1 == 0:
                        writer.add_scalar(
                            "charts/episodic_return", item["final_info"][0]['episode']["r"], global_step)
                        writer.add_scalar(
                            "charts/episodic_length", item["final_info"][0]['episode']["l"], global_step)
                    break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Optimize the discriminator
        if init:
            n_irl_epochs = args.irl_init_epochs
        else:
            n_irl_epochs = args.irl_epochs
        init = False

        irl_count += 1
        for epoch in range(n_irl_epochs):            
            gen_demos = demos_gen_dict(demos['all'], args.batch_size)
            for d in gen_demos:
                if len(d['obs']) < args.batch_size:
                    if args.drop_batch:
                        break
                    else:
                        obs_ = obs[:len(d['obs'])]
                        actions_ = actions[:len(d['obs'])]
                        dones_ = dones[:len(d['obs'])]
                else:
                    obs_ = obs
                    actions_ = actions
                    dones_ = dones

                update_dict = prepare_batch_update_irl(envs, args, d, obs_, actions_, dones_, agent)

                loss_dict = disc.compute_loss(update_dict)

                loss = loss_dict['d_loss'] + args.irm_coeff * loss_dict['grad_penalty'] + args.lip_coeff * loss_dict['lip_penalty']
                if args.irm_coeff > 1.0:
                    loss /= args.irm_coeff

                disc.update(loss)
            
                if global_step % args.logging_interval  == 0:
                    writer.add_scalar('irl/gail_bce_loss', loss_dict['d_loss'], global_step)
                    if args.irm_coeff > 0:
                        writer.add_scalar('irl/gail_irm_loss', loss_dict['grad_penalty'], global_step)


        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if global_step % args.logging_interval  == 0:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            if args.verbose:
                print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        #  evaluate at regular intervals and save
        if global_step % args.eval_interval == 0:
            mu, std, d = evaluate_model(args, agent, device, test_env)

            print(f"Evaluation at {global_step}: {mu}, {std}")
            writer.add_scalar('eval/mu', mu, global_step)
            writer.add_scalar('eval/std', std, global_step)

    envs.close()
    writer.close()

    print("Total IRL steps:", irl_count)

    ## save model
    torch.save(agent.state_dict(), f"runs/{run_name}/agent.pt")
    torch.save(disc.state_dict(), f"runs/{run_name}/disc.pt")

