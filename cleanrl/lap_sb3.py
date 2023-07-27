from pathlib import Path
import pickle
import pprint
import numpy as np
import gymnasium as gym
import random
import torch 
import torch.functional as F
import torch.nn as nn
import argparse
import argunparse
import subprocess
from datetime import datetime

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN as DQNSB

gym.envs.register(
     id='Lap-v0',
     entry_point='envs.env_lap:Laparos',
     max_episode_steps=10000,
)


class GAILDiscriminator(nn.Module):
    def __init__(self, env, opt):
        super(GAILDiscriminator, self).__init__()

        # parse options
        self.env = env
        self.opt = opt
        self.layer_dims = opt.d_layer_dims
        self.lr = opt.disc_lr
        self.use_actions = opt.use_actions
        self.irm_coeff = opt.irm_coeff
        self.l2_coeff = opt.l2_coeff
        self.lip_coeff = opt.lip_coeff
        self.use_cnn_base = opt.use_cnn_base
        self.bias = opt.use_disc_bias
        self.is_atari = 'atari' in self.opt.exp_name

        if opt.disc_nonlin == 'relu':
            nonlin = nn.ReLU()
        elif opt.disc_nonlin == 'tanh':
            nonlin = nn.Tanh()
        else:
            nonlin = nn.PReLU()

        ob_shapes = list(env.observation_space.shape)
        ac_shapes = list(env.action_space.shape)

        if not ac_shapes:
            ac_shapes = [1]
        if self.use_actions:
            self.layer_dims = [ob_shapes[-1] + ac_shapes[-1]] + self.layer_dims
        else:
            self.layer_dims = [ob_shapes[-1]] + self.layer_dims
        ac_len = ac_shapes[0]

        layer_dims = self.layer_dims

        if self.is_atari:
            self.base = AtariCNNBase(opt,env, self.use_actions)
            self.discriminator = nn.Linear(512, 1, bias=self.bias)
        else:
            if self.use_cnn_base:
                self.base = MiniGridCNN(layer_dims, self.use_actions)
            else:
                self.base = nn.Sequential(torch.nn.Linear(
                    self.layer_dims[0], self.layer_dims[1], self.bias), nonlin)

            self.discriminator_layers = []
            for i in range(2, len(layer_dims)):
                self.discriminator_layers += [torch.nn.Linear(
                    in_features=layer_dims[i - 1], out_features=layer_dims[i],
                    bias=self.bias), nonlin]

            self.discriminator_layers += [torch.nn.Linear(
                in_features=layer_dims[-1], out_features=1, bias=self.bias)]

            self.discriminator = nn.Sequential(*self.discriminator_layers)

        if torch.cuda.is_available():
            self.base.cuda()
            self.discriminator.cuda()

        base_params = list(self.base.parameters())
        d_params = list(self.discriminator.parameters())
        base_params.extend(d_params)
        self.d_optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                                            weight_decay=self.l2_coeff)

    def base_forward(self,ob, ac):
        if self.use_actions and (self.use_cnn_base or self.is_atari):
            base_out = self.base(ob, ac)
        elif self.use_actions and not (self.use_cnn_base or self.is_atari):
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)
        
        return base_out

    def forward(self, ob, ac):
        base_out = self.base_forward(ob, ac)
        d_out = self.discriminator(base_out)

        return d_out

    def get_reward(self, ob, ac):
        base_out = self.base_forward(ob, ac)

        d_out = self.discriminator(base_out)
        self.reward = - torch.squeeze(torch.log(torch.sigmoid(d_out) + 1e-8))
        return self.reward

    def compute_loss(self, update_dict):
        d_out = self.forward(update_dict['all_obs'], update_dict['all_acs'])
        self.bce_loss = F.binary_cross_entropy_with_logits(
            d_out, update_dict['collision'])
        self.loss = self.bce_loss

        return self.bce_loss
        
    def update(self, loss):
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()


# collision classifier: learn to predict paths which lead to failures (buffer pruning)
class CCUpdateCallback(BaseCallback):
    def __init__(self, opt, disc, demos, verbose=0, id="", log_path=None):
        super(CCUpdateCallback, self).__init__(verbose)
        self.cnt = 0
        self.opt = opt
        self.demos = demos
        self.disc = disc
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
        rb = self.locals['replay_buffer'] 
        env = self.locals['env']
        # loop over buffer 
        for d in rb:
            len_obs = args.batch_size
            if len(d['obs']) < args.batch_size:
                if args.drop_batch:
                    break
                else:
                    len_obs = len(d['obs'])

            obs = d.observations
            actions = torch.squeeze(d.actions)
            dones = d.dones

            expert_obs = d['obs']
            expert_acs = d['acs']

            expert_obs_t = torch.from_numpy(
                expert_obs).type(torch.get_default_dtype())
            expert_acs_t = torch.from_numpy(
                expert_acs).type(torch.get_default_dtype())

            all_obs = torch.cat([expert_obs_t, obs], 0)
            all_acs = torch.cat([expert_acs_t, actions], 0)

            loss_dict = self.disc.compute_loss(
                {'expert_obs': expert_obs_t, 'expert_acs': expert_acs_t,
                 'policy_obs': obs, 'policy_acs': actions,
                 'all_obs': all_obs, 'all_acs': all_acs})

            loss = loss_dict['d_loss'] + args.irm_coeff * \
                loss_dict['grad_penalty'] + \
                args.lip_coeff * loss_dict['lip_penalty']
            if args.irm_coeff > 1.0:
                loss /= args.irm_coeff

            self.disc.update(loss)

            self.logger.record('irl/gail_bce_loss', loss_dict['d_loss'])
            if self.opt.irm_coeff > 0:
                self.logger.record('irl/gail_irm_loss',
                                   loss_dict['grad_penalty'])

        self.cnt += 1

    def _on_training_end(self) -> None:
        pass


def save_expert_traj(env, model, nr_trajectories=10, file_path=None,
                     deterministic=True):
    num_steps = 0
    expert_obs = []
    expert_acs = []
    expert_rew = []
    expert_done = []

    cnt = 0
    while cnt < nr_trajectories:
        ob = env.reset()
        done = False
        total_reward = 0
        episode_obs = []
        episode_acs = []
        episode_rew = []
        episode_done = []

        while not done:
            ac, _states = model.predict(ob)
            next_ob, reward, term, trunc, _ = env.step(ac)
            ob = next_ob
            total_reward += reward
            done = term or trunc
            episode_obs.append(ob)
            episode_acs.append(ac)
            episode_rew.append(reward)
            episode_done.append(done)
            num_steps += 1

        cnt += 1
        expert_obs.extend(episode_obs)
        expert_acs.extend(episode_acs)
        expert_rew.extend(episode_rew)
        expert_done.extend(episode_done)

    expert_obs = np.stack(expert_obs)
    expert_acs = np.stack(expert_acs)
    expert_rew = np.stack(expert_rew)
    expert_done = np.stack(expert_done)
    data = {'obs': expert_obs, 'acs': expert_acs, 'rew': expert_rew,
            'done': expert_done}
    if file_path is not None:
        pickle.dump(data, open(str(file_path), 'wb'))
    return data


def submit_jobs(args):

    # If the same seed should be used for all experiments
    random.seed(43)
    if isinstance(args.seed, list) and len(args.seed) == 1 \
            and int(args.seed[0]) == -1:
        seeds = random.sample(range(1, 1000), args.repetitions)
    elif isinstance(args.seed, list) and len(args.seed) == 1:
        seeds = np.repeat(args.seed[0], args.repetitions)
    elif isinstance(args.seed, list) and len(args.seed) > 1:
        assert len(args.seed) == args.repetitions
        seeds = args.seed
    elif args.seed is None:
        seeds = random.sample(range(1, 1000), args.repetitions)
    else:
        raise NotImplemented

    lsf_out_dir = args.results_dir / 'outs'
    lsf_out_dir.mkdir(parents=True, exist_ok=True, mode=0o755)

    if args.no_gpu:
        args.any_gpu = True

    new_args_dict = vars(args).copy()
    for i in range(args.repetitions):
        seed = seeds[i]

        time = datetime.now().strftime('%H%M%S%f')
        run_id = f"lap_{args.run_id}_{i}_seed_{seed}_{time}"
        print(f"Run {run_id} with seed: {seed}")
        euler_slurm = (f"sbatch --mem-per-cpu={args.memory} "
                       f"{'--gpus=1 ' if not args.no_gpu else ''}"
                       f"{'--gpus=gtx_1080:1 ' if not args.any_gpu else ''}"
                       f"-J {run_id} "
                       f"-o {lsf_out_dir / run_id} "
                       f"--time={args.time}:00:00 --wrap ")
        euler_lsf = (f"bsub -R rusage[mem={args.memory}"
                     f"{',ngpus_excl_p=1' if not args.any_gpu else ''}] "
                     f"{'-R select[gpu_model0==NVIDIAGeForceGTX1080Ti] ' if not args.any_gpu else ''}"
                     f"-o {lsf_out_dir / run_id} "
                     f"-W {args.time}:00 ")
        new_args_dict['seed'] = seed
        new_args_dict['run_id'] = run_id
        new_args_dict['repetitions'] = 1
        unparser = argunparse.ArgumentUnparser()
        newargs = unparser.unparse(**new_args_dict)
        newargs = newargs + f" --no_submit"

        if args.batch_system == 'slurm':
            command = f"python euler_train.py {newargs}"
            subprocess.check_output(args=euler_slurm.split() + [command])
        elif args.batch_system == 'lsf':
            command = euler_lsf + f"python euler_train.py {newargs}"
            subprocess.check_output(args=command.split())
        else:
            NotImplemented(f"batch system {args.batch_system} is not "
                           f"supported on the cluster.")


def run_training(args):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    # create output folder with non default arguments in the name
    id_string = str(args.non_default).replace(' ', '').replace(
        ':', '').replace('Namespace', '').replace('{', '').replace(
        '}', '').replace("'", "_").replace(",", "_")

    seed = args.seed
    env = gym.make('Lap-v0',
                   gc=args.goal_conditioning,
                   state_spec=args.state_spec,
                   step_size=args.step_size,
                   done_on_collision=args.done_on_collision,
                   hard_collisions=args.hard_collisions,
                   plot_meshes=args.render_meshes,
                   collision_loss=args.collision_loss)

    # collision classification loss wrapper

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    kwargs = {
        'learning_rate': args.lr,
        #'buffer_size': 1000000,
        #'normalize': "{'norm_obs': True, 'norm_reward': False}",
        'n_steps': args.num_steps,
        'batch_size': args.minibatch_size,
        'n_epochs': args.update_epochs,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'clip_range': args.clip_coef,
        'clip_range_vf': None,
        #'ent_coef': 'auto',
        #'vf_coef':0.5,
        #'gradient_steps': 20,
        #'tau': 0.005,
        #'target_update_interval': 1
        'max_grad_norm': args.max_grad_norm
    }

    id_string = f"{id_string}_{ts}"
    exp_dir = args.results_dir / id_string
    exp_dir.mkdir(parents=True, exist_ok=True)
    model = DQNSB('MlpPolicy', env, verbose=1,
                  tensorboard_log=exp_dir / "sb3")
    model.learn(total_timesteps=args.n_timesteps)

    # save model
    model_dir = exp_dir / 'sb3_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir))
    save_expert_traj(env, model, nr_trajectories=10,
                     file_path=exp_dir / 'expert_traj.pkl',
                     deterministic=True)


def main():
    euler_res = "/cluster/project/jbuhmann/workspace/lap/results/"
    parser = argparse.ArgumentParser(description="Laparos experiments")
    parser.add_argument("--results_dir", type=Path, default=Path(euler_res),
                        help="Path in which results of training are/will be "
                             "located")
    parser.add_argument("--exp-name", type=str, default="laparos",
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument('--normalize_obs', default=True,
                        action='store_true', required=False,
                        help='normalize observations')
    parser.add_argument('--normalize_rewards', default=True,
                        action='store_true', required=False,
                        help='normalize irl rewards')
    parser.add_argument('--clip_action', default=True, action='store_true',
                        required=False, help='clip actions')

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Lap-v0",
                        help="the id of the environment")
    parser.add_argument("--n_timesteps", type=int, default=10000000,  # 1e6
                        help="total timesteps of the experiments")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument("--num_steps", type=int, default=1024,  # 2048
                        help="the number of steps to run in each environment "
                             "per policy rollout")
    parser.add_argument("--agent_size", type=int, default=64,
                        help='actor-critic network size')
    parser.add_argument("--anneal_lr", type=lambda x: bool(strtobool(x)),
                        default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and "
                             "value networks")
    parser.add_argument("--gamma", type=float, default=0.95,  # 0.99
                        help="the discount factor gamma")
    parser.add_argument("--gae_lambda", type=float, default=0.9,  # 0.95
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num_minibatches", type=int, default=32,  #  32
                        help="the number of mini-batches")
    parser.add_argument("--update_epochs", type=int, default=5,  # 10
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

    #  environment specific arguments
    parser.add_argument('--goal_conditioning', default=True,
                        action='store_false', required=False,
                        help='normalize irl rewards')
    parser.add_argument('--condition_on_n_tasks_completed', default=True,
                        action='store_false', required=False,
                        help='normalize irl rewards')
    parser.add_argument('--done_on_collision', default=True,
                        action='store_false', required=False,
                        help='normalize irl rewards')
    parser.add_argument('--collision_loss', default=False,
                        action='store_true', required=False,
                        help='end episode on collision')
    parser.add_argument('--hard_collisions', default=True,
                        action='store_false', required=False,
                        help='dont progress on collision')
    parser.add_argument('--continuous_actions', default=False,
                        action='store_true', required=False,
                        help='normalize irl rewards')
    parser.add_argument('--render_meshes', default=True,
                        action='store_true', required=False,
                        help='normalize irl rewards')
    parser.add_argument("--step_size", type=float, default=0.3,
                        help="step magnitude")
    parser.add_argument("--state_spec", type=str, default="r3",
                        help="state spec (r3/r3so3)")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    non_default = {
        opt.dest: getattr(args, opt.dest)
        for opt in parser._option_string_actions.values()
        if hasattr(args, opt.dest) and opt.default != getattr(args, opt.dest)
    }
    args.non_default = non_default

    print("   Experiment parameters: ")
    print("-" * 100)
    pprint.pprint(vars(args), indent=5)
    print("-" * 100)

    if args.no_submit:
        run_training(args)
    else:
        submit_jobs(args)


if __name__ == '__main__':
    main()
