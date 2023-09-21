# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import sys
import random
import time
from datetime import datetime
from distutils.util import strtobool
from tqdm import tqdm

module_path = os.path.abspath(os.path.join('../ccl'))
if module_path not in sys.path:
    sys.path.append(module_path)

import gymnasium as gym

import numpy as np
#import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from torch.utils.tensorboard import SummaryWriter

from irl.utils import load_hf_demos, prepare_batch_update_irl, enhance_demos_with_barycenters,\
                    demos_gen_dict, AntCFWrapper, MujocoTestWrapper, save_traj_dict, evaluate_model

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

    parser.add_argument('--normalize_obs', default=True, 
        action='store_true', required=False, help='normalize irl rewards')  
    parser.add_argument('--clip_action', default=True, 
        action='store_true', required=False, help='normalize irl rewards')  

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v3",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--agent_size", type=int, default=64, help='actor-critic network size')
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
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
    parser.add_argument('--d_layer_dims', default=[128, 128], nargs='+', required=False,
                   help='discriminator MLP layer dimensions')
    parser.add_argument('--airl_gamma', default=0.99, type=float, 
        required=False, help='gamma for AIRL advantage')
    parser.add_argument('--airl_resnet', default=False, action='store_true', help='use resnet for discriminator')
    parser.add_argument('--irm_coeff', default=0, type=float, 
        required=False, help='use IRM penalty on discriminator')
    parser.add_argument('--irl_epochs', default=1, type=float, required=False, help='number of discriminator updates')
    parser.add_argument('--lip_coeff', default=0, type=float, required=False, help='use gradient (spectral/mixup) penalty on discriminator')
    parser.add_argument('--l2_coeff', default=0, type=float, required=False, help='use l2 penalty on discriminator')
    parser.add_argument('--disc_lr', default=3e-4, type=float, required=False, help='discriminator learning rate')
    parser.add_argument('--use_cnn_base', default=False, required=False, 
        action='store_true', help='use cnn base for discriminator (only for visual obs)')
    parser.add_argument('--use_disc_bias', default=True, required=False, 
        action='store_false', help='use bias in disc arch')
    parser.add_argument('--on_policy', default=True, required=False, 
        action='store_false', help='use bias in disc arch')
    parser.add_argument("--resume", type=str, default=None, help="resume training from checkpoint")
    parser.add_argument("--test_perturbation_eps", type=float, default=0.0, help="test perturbation eps")
    parser.add_argument('--verbose', default=False, required=False, 
        action='store_true', help='verbose output')
    parser.add_argument("--disc_nonlin", type=str, default="prelu",
        help="divergence type")
    parser.add_argument('--normalize_irl_rewards', default=True, 
        action='store_true', required=False, help='normalize irl rewards')  

    parser.add_argument("--test_eps", type=float, default=0.0, help="test perturbation eps")
    parser.add_argument("--perturbation_parameters", type=str, nargs='+', default='', help="which env parameter to perturb")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    non_default = {
        opt.dest: getattr(args, opt.dest)
        for opt in parser._option_string_actions.values()
        if hasattr(args, opt.dest) and opt.default != getattr(args, opt.dest)
    }

    return args, non_default

def make_env(opt, env_id, seed, idx, capture_video, run_name, gamma):
    def thunk():
        env = gym.make(env_id)
        #if opt.env_id == 'Ant-v3':
        #    print("Wrapping Ant-v3 env with AntCFWrapper")
        #    env = AntCFWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        if opt.clip_action:
            env = gym.wrappers.ClipAction(env)
        if opt.normalize_obs:
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        if opt.normalize_irl_rewards:
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
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_size)),
            nonlin,
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nonlin,
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_size)),
            nonlin,
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nonlin,
            layer_init(nn.Linear(hidden_size, output_dim), std=0.01),
        )
        if not self.discrete:
            self.actor_logstd = nn.Parameter(torch.zeros(1, output_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value_c(self, x, action=None, eval=False):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        if eval:
            return action_mean, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
        else:
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_action_and_value_d(self, x, action=None, eval=False):
        logits = self.actor_mean(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_action_and_value(self, x, action=None, eval=False):
        if self.discrete:
            return self.get_action_and_value_d(x, action, eval)
        else:
            return self.get_action_and_value_c(x, action, eval)


if __name__ == "__main__":
    args, non_default = parse_args()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    ua = 'ua' if args.use_actions else ''
    # create output folder with non default arguments in the name
    id_string = str(non_default).replace(' ','').replace(':','').replace('Namespace','').replace('{','').replace('}','').replace("'","_").replace(",","_")
    run_name = f"{ts}_{args.env_id}__{args.exp_name}_seed_{args.seed}_irm{args.irm_coeff}_lip{args.lip_coeff}_{ua}_{id_string}"
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

    demos = load_hf_demos(args, args.n_demos)
    if args.enhance_demos:
        print("Enhancing demos by computing Wasserstein barycenters")
        demos = enhance_demos_with_barycenters(demos)
    # get env specs
    test_env = gym.make(args.env_id)
    discrete = not isinstance(test_env.action_space, gym.spaces.Box)

    #if args.env_id == 'Ant-v3':
    #    print("Wrapping Ant-v3 test env with AntCFWrapper")
    #    test_env = AntCFWrapper(test_env)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args, args.env_id, args.seed + i, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args.agent_size, discrete).to(device)
    if args.resume:
        agent.load_state_dict(torch.load(os.path.join(args.resume, "agent.pt") , map_location=device))

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if discrete:
        loss = nn.CrossEntropyLoss()
    else:
        loss = nn.MSELoss()

    #actions = torch.zeros((args.batch_size, args.num_envs) + envs.single_action_space.shape).to(device)

    # Behavioural cloning loop:
    global_step = 0
    #args.batch_size = 128

    for epoch in tqdm(range(args.update_epochs)):            
        # GAIL Discriminator Update (exhaust demos)
        gen_demos = demos_gen_dict(demos['all'], args.batch_size, shuffle=True)
        for d in gen_demos:
            if len(d['obs']) < args.batch_size:
                break
            global_step += 1 * args.num_envs

            obs = torch.tensor(d['obs'], dtype=torch.float32).to(device)
            exp_ac = torch.tensor(d['acs'], dtype=torch.float32).to(device)
        
            #actions, logprob, _, value = agent.get_action_and_value(obs)
            
            _, log_prob, entropy, _ = agent.get_action_and_value(obs, exp_ac)
            log_prob = log_prob.mean()
            entropy = entropy.mean() if entropy is not None else None

            ent_loss = -args.ent_coef * (entropy if entropy is not None else torch.zeros(1))
            neglogp = -log_prob
            loss = neglogp + ent_loss

            #l = loss(actions, exp_ac)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % args.logging_interval == 0:
                writer.add_scalar("BC loss", loss, global_step)
        
                #  evaluate at regular intervals and save
            if global_step % args.eval_interval == 0:
                if isinstance(test_env.action_space, gym.spaces.Box):
                    det = True
                else:
                    det = False
                mu, std, d = evaluate_model(args, agent, device, deterministic = det, wrap_mujoco=args.test_eps > 0)
                print(f"Evaluation at {global_step}: {mu}, {std}")
                writer.add_scalar('eval/mu', mu, global_step)
                writer.add_scalar('eval/std', std, global_step)

    envs.close()

    ## save model
    torch.save(agent.state_dict(), f"runs/{run_name}/agent.pt")

    ## test model and save trajectories
    test_env = MujocoTestWrapper(gym.make(args.env_id), eps=args.test_perturbation_eps)
    if args.clip_action:
        test_env = gym.wrappers.ClipAction(test_env)
    if args.normalize_obs:
        test_env = gym.wrappers.NormalizeObservation(test_env)
        test_env = gym.wrappers.TransformObservation(test_env, lambda obs: np.clip(obs, -10, 10))
    save_traj_dict(args.env_id, test_env, agent, device)

    writer.close()

    # save df for easier parsing
    # if args.total_timesteps > 0:    
    #     ea = read_tb_file(f"runs/{run_name}")
    #     df = create_df_from_ea(ea, "charts/episodic_return")
    #     df.to_pickle(f"results_{run_name}_df.pkl")
