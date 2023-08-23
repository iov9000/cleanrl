from pathlib import Path
import pickle
import pprint
import numpy as np
import gymnasium as gym
import random
import torch
import argparse
import argunparse
import subprocess
from datetime import datetime
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import DQN as DQNSB

from irl.gail import GailReward, GAILDiscriminator
from irl.utils import load_hf_demos, load_lap_demos, demos_gen_dict

# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)


gym.envs.register(
     id='Lap-v0',
     entry_point='envs.env_lap:Laparos',
     max_episode_steps=10000,
)


class GAILUpdateCallback(BaseCallback):
    def __init__(self, opt, disc, demos, verbose=0, id="", log_path=None):
        super(GAILUpdateCallback, self).__init__(verbose)
        self.cnt = 0
        self.opt = opt
        self.demos = demos
        self.disc = disc
        self.id = id
        # self.aux_writer = SummaryWriter(log_path)

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
        #  account for training starts! TODO: fix this.. since it's off-policy, need to figure out timing
        if self.cnt % 1000 == 0:
            gen_demos = demos_gen_dict(self.demos, self.opt.batch_size)
            for d in gen_demos:
                len_obs = self.opt.batch_size
                if len(d['obs']) < self.opt.batch_size:
                    if self.opt.drop_batch:
                        break
                    else:
                        len_obs = len(d['obs'])

                obs = rb.sample(len_obs).observations
                actions = torch.squeeze(rb.sample(len_obs).actions)
                dones = rb.sample(len_obs).dones

                expert_obs = d['obs']
                expert_acs = d['acs']

                expert_obs_t = torch.from_numpy(expert_obs).type(
                    torch.get_default_dtype())
                expert_acs_t = torch.from_numpy(expert_acs).type(
                    torch.get_default_dtype())

                all_obs = torch.cat([expert_obs_t, obs], 0)
                all_acs = torch.cat([expert_acs_t, actions], 0)

                loss_dict = self.disc.compute_loss(
                    {'expert_obs': expert_obs_t, 'expert_acs': expert_acs_t,
                     'policy_obs': obs, 'policy_acs': actions,
                     'all_obs': all_obs, 'all_acs': all_acs})

                loss = loss_dict['d_loss'] + self.opt.irm_coeff * \
                       loss_dict['grad_penalty'] + self.opt.lip_coeff * loss_dict[
                           'lip_penalty']
                if self.opt.irm_coeff > 1.0:
                    loss /= self.opt.irm_coeff

                self.disc.update(loss)

                self.logger.record('irl/gail_bce_loss', loss_dict['d_loss'])
                if self.opt.irm_coeff > 0:
                    self.logger.record('irl/gail_irm_loss',
                                       loss_dict['grad_penalty'])

        self.cnt += 1

    def _on_training_end(self) -> None:
        pass


class NumTasksDoneCallBack(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, env, _log_freq=2000):
        self.env = env
        self._log_freq = _log_freq
        super(NumTasksDoneCallBack, self).__init__()

    def _on_training_start(self):
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here,
        # should be done with try/except.
        self.tb_formatter = next(
            formatter for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.tb_formatter.writer.add_scalar(
            "env/num_tasks", self.env.curr_tgt, self.num_timesteps)
        self.tb_formatter.writer.flush()

        if len(self.env.num_tasks_completed[-20:]) > 0:
            tasks_hist = np.array(self.env.num_tasks_completed[-20:])
            self.tb_formatter.writer.add_histogram(
                "env/num_tasks_hist", tasks_hist, self.num_timesteps)


def load_lap_forward_demos(file_path, n_demos):
    demos = pickle.load(open(file_path, 'rb'))
    for k, v in demos.items():
        demos[k] = v[:n_demos]
    data = {k: [] for k in demos.keys()}
    for k, v in demos.items():
        for i in range(len(v)):
            data[k].extend(v[i])
    for k, v in data.items():
        data[k] = np.array(v)
    return data


def save_expert_traj(env, model, nr_trajectories=10, file_path=None,
                     deterministic=True):
    num_steps = 0
    expert_obs = []
    expert_acs = []
    expert_rew = []
    expert_done = []

    cnt = 0
    while cnt < nr_trajectories:
        ob, _ = env.reset()
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


def run_training(args, parser):
    if isinstance(args.seed, list):
        args.seed = int(args.seed[0])
    elif args.seed is None:
        args.seed = random.sample(range(1, 1000), args.repetitions)[0]

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    # create output folder with non default arguments in the name
    non_default = {
        opt.dest: str(getattr(args, opt.dest))
        for opt in parser._option_string_actions.values()
        if hasattr(args, opt.dest) and opt.default != getattr(args, opt.dest)
    }
    id_string = "_".join([f"{key}_{val}" for key, val in non_default.items()])
    id_string = "default" if id_string == '' else id_string

    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    seed = args.seed
    env = gym.make(
        'Lap-v0',
        gc=args.goal_conditioning,
        condition_on_n_tasks_completed=args.condition_on_n_tasks_completed,
        state_spec=args.state_spec,
        step_size=args.step_size,
        done_on_collision=args.done_on_collision,
        hard_collisions=args.hard_collisions,
        plot_meshes=args.render_meshes,
        collision_loss=args.collision_loss)
    test_env = gym.make(
        'Lap-v0',
        gc=args.goal_conditioning,
        condition_on_n_tasks_completed=args.condition_on_n_tasks_completed,
        state_spec=args.state_spec,
        step_size=args.step_size,
        done_on_collision=args.done_on_collision,
        hard_collisions=args.hard_collisions,
        plot_meshes=args.render_meshes,
        collision_loss=args.collision_loss)
    disc = GAILDiscriminator(test_env, args)
    env = GailReward(env, args, disc)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    kwargs = {
        'learning_rate': args.lr,
        'buffer_size': 1000000,
    }

    if args.demos == 'davos':
        demos = load_lap_demos(
            'demos/davos_data', args.n_demos, obs_shape=3)['all']
    elif args.demos == 'fw':
        demos = load_lap_forward_demos(args.demo_path, args.n_demos)
    else:
        NotImplemented("Only 'davos' and 'fw' demos are supported.")
        return

    id_string = f"{id_string}_{ts}"
    exp_dir = args.results_dir / id_string
    exp_dir.mkdir(parents=True, exist_ok=True)

    model = DQNSB('MlpPolicy', env, **kwargs, verbose=1,
                  tensorboard_log=exp_dir / "sb3")

    nt_logs = NumTasksDoneCallBack(env, _log_freq=args.eval_freq // 2)
    gail_cb = GAILUpdateCallback(args, disc, demos)
    eval_cb = EvalCallback(
        test_env, best_model_save_path=exp_dir, log_path=exp_dir / "test",
        eval_freq=args.eval_freq, deterministic=False)

    cb_list = [gail_cb, eval_cb, nt_logs]
    callback_list = CallbackList(cb_list)
    model.learn(total_timesteps=args.n_timesteps, callback=callback_list)

    # save model
    model_dir = exp_dir / 'sb3_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir))
    save_expert_traj(env, model, nr_trajectories=10,
                     file_path=exp_dir / 'expert_traj.pkl',
                     deterministic=True)


def submit_jobs(args, parser=None):

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
    if parser is not None:
        for key in new_args_dict.keys():
            if isinstance(parser.get_default(key), bool):
                store_false = parser.get_default(key)
                if store_false:
                    new_args_dict[key] = not new_args_dict[key]
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
            command = f"python lap_gail_dqn.py {newargs}"
            subprocess.check_output(args=euler_slurm.split() + [command])
        elif args.batch_system == 'lsf':
            command = euler_lsf + f"python lap_sb3.py {newargs}"
            subprocess.check_output(args=command.split())
        else:
            NotImplemented(f"batch system {args.batch_system} is not "
                           f"supported on the cluster.")


def main():
    euler_res = "/cluster/project/jbuhmann/workspace/lap/results/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path(euler_res),
                        help="Path in which results of training are/will be "
                             "located")
    parser.add_argument("--run_id", type=str, default="laparos",
                        help="the name of this experiment")
    parser.add_argument("--seed",
                        nargs="+",
                        default=None,
                        help="Random seed to use. If None, randomly generated "
                             "seeds for each experiment will be used. If a "
                             "single value, the same seed will be used for "
                             "all the experiments. If the a list of seeds, "
                             "each seed will be used for each of the "
                             "experiments. In this case, the length of seed "
                             "must be the same as number of repetitions (-r)."
                             "if it's set to -1 the same set of  seeds will "
                             "be used for the sake of  experiments' "
                             "consistency.")

    parser.add_argument('--normalize_obs', default=True,
                        action='store_true', required=False,
                        help='normalize observations')
    parser.add_argument('--normalize_rewards', default=True,
                        action='store_true', required=False,
                        help='normalize irl rewards')
    parser.add_argument('--clip_action', default=True,
                        action='store_true', required=False,
                        help='clip actions')
    parser.add_argument("--eval_freq", type=int, default=2000,
                        help="the frequency of evaluation")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Lap-v0",
                        help="the id of the environment")
    parser.add_argument("--n_timesteps", type=int, default=10000000,  # 1e6
                        help="total timesteps of the experiments")
    parser.add_argument("--lr", type=float, default=3e-4,  # 3e-4
                        help="the learning rate of the optimizer")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument("--num_steps", type=int, default=1024,  # 2048
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--agent_size", type=int, default=64,
                        help='actor-critic network size')
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

    #  environment specific arguments (Lap-v0)
    parser.add_argument('--goal_conditioning', default=True,
                        action='store_false', required=False,
                        help='Condition on goal')
    parser.add_argument('--condition_on_n_tasks_completed', default=True,
                        action='store_false', required=False,
                        help='Condition on number of tasks completed')
    parser.add_argument('--done_on_collision', default=False,
                        action='store_false', required=False,
                        help='Terminate episode on collision')
    parser.add_argument('--collision_loss', default=False,
                        action='store_true', required=False,
                        help='Penalize collisions')
    parser.add_argument('--hard_collisions', default=False,
                        action='store_false', required=False,
                        help='dont progress on collision')
    parser.add_argument('--continuous_actions', default=False,
                        action='store_true', required=False,
                        help='Continuous actions')
    parser.add_argument('--render_meshes', default=True,
                        action='store_true', required=False,
                        help='Render meshes')
    parser.add_argument("--step_size", type=float, default=0.3,
                        help="step magnitude")
    parser.add_argument("--state_spec", type=str, default="r3",
                        help="state spec (r3/r3so3)")

    # SWIL specific
    parser.add_argument("--n_proj", type=int, default=100,
                        help="number of projections for SWIL")
    parser.add_argument("--poly_degree", type=int, default=4,
                        help="degree of polynomial for SWIL")
    parser.add_argument("--swil_reward_type", type=str, default='linear',
                        help="SWIL reward type (linear / attn")
    parser.add_argument("--max_gsw", default=False, required=False,
                        action='store_true', help="use max GSW loss")
    parser.add_argument("--swil_loss", default='gsw', type=str, required=False,
                        help="loss_type:")
    parser.add_argument("--gsw_df", default='poly', type=str, required=False,
                        help="Radon transform defining function: nn/poly/linear/circular")

    parser.add_argument("--proj_layer", default=False, required=False,
                        action='store_true',
                        help="use nn instead of mean of projections")
    parser.add_argument("--linear_proj", default=False, required=False,
                        action='store_true',
                        help="linear projection only")
    parser.add_argument("--swil_ae", default=False, required=False,
                        action='store_true', help="use autoencoder")

    # Discriminator specific arguments
    parser.add_argument("--demos", type=str, default="fw",
                        help="type of demos. `fw` or `davos`.")
    parser.add_argument("--demo_path", type=str,
                        default=Path("./demos"), help="expert demo path")
    parser.add_argument("--n_demos", type=int, default=None,
                        help="number of demos to use. If None, use all demos")
    parser.add_argument('--enhance_barys', default=0, required=False,
                        type=int, help='enhance demos using W2 barycenters')
    parser.add_argument('--use_actions', default=True, required=False,
                        action='store_false',
                        help='use actions in discriminator')
    parser.add_argument('--d_layer_dims', default=[128, 128], nargs='+',
                        required=False,
                        help='discriminator MLP layer dimensions')
    parser.add_argument('--irm_coeff', default=0, type=float,
                        required=False,
                        help='use IRM penalty on discriminator')
    parser.add_argument("--irm_loss_type", type=str, default="sym",
                        help="e1,e0,p0,p1,sym")

    parser.add_argument('--drop_batch', default=False, required=False,
                        action='store_true', help='drop batch in irl epochs')

    parser.add_argument('--irl_epochs', default=1, type=int, required=False,
                        help='number of discriminator updates')
    parser.add_argument('--irl_init_epochs', default=1, type=int,
                        required=False, help='number of discriminator updates')
    parser.add_argument('--lip_coeff', default=0, type=float, required=False,
                        help='use gradient (spectral/mixup) penalty on discriminator')
    parser.add_argument('--l2_coeff', default=0, type=float, required=False,
                        help='use l2 penalty on discriminator')
    parser.add_argument('--disc_lr', default=3e-4, type=float, required=False,
                        help='discriminator learning rate')
    parser.add_argument('--use_cnn_base', default=False, required=False,
                        action='store_true',
                        help='use cnn base for discriminator (only for visual obs)')
    parser.add_argument('--use_disc_bias', default=True, required=False,
                        action='store_false', help='use bias in disc arch')
    parser.add_argument("--div", type=str, default="rkl",
                        help="divergence type")
    parser.add_argument("--disc_nonlin", type=str, default="prelu",
                        help="divergence type")
    parser.add_argument('--normalize_irl_rewards', default=True,
                        action='store_true', required=False,
                        help='normalize irl rewards')

    parser.add_argument('--on_policy', default=True, required=False,
                        action='store_false', help='use bias in disc arch')
    parser.add_argument("--resume", type=str, default=None,
                        help="resume training from checkpoint")
    parser.add_argument("--load_disc", type=str, default=None,
                        help="resume training from checkpoint")

    parser.add_argument("--test_perturbation_eps", type=float, default=0.0,
                        help="test perturbation eps")
    parser.add_argument('--verbose', default=False, required=False,
                        action='store_true', help='verbose output')

    # Cluster arguments
    parser.add_argument('--no_submit', default=False, required=False,
                        action='store_true',
                        help='Do not submit to Euler. Default will submit.')
    parser.add_argument('--any_gpu', default=False, required=False,
                        action='store_true',
                        help='Do not select a specific GPU in oder to wait '
                             'less in the queue. Selecting a specific GPU is '
                             'important for reproducibility in order to run '
                             'deterministic experiments, but this could be '
                             'ignored sometimes.')
    parser.add_argument('--no_gpu', default=False, required=False,
                        action='store_true',
                        help='Do not request any gpus.')
    parser.add_argument("--batch_system",
                        type=str,
                        default='slurm',
                        choices=["slurm", "lsf"],
                        help="Which batch system to use for submission. "
                             "lsf is becoming  obsolete on Euler very soon.")
    parser.add_argument("-s", "--scratch", action='store_true',
                        help="Whether to first copy the dataset to the "
                             "scratch storage of Leonhard. Do not use on "
                             "other systems than Leonhard.")
    parser.add_argument("-m", "--memory",
                        type=int,
                        default=8000,
                        help="Memory allocated for each leonhard job. This "
                             "will be ignored of Leonhard is not selected.")
    parser.add_argument("-t", "--time",
                        type=int,
                        default=4,
                        help="Number of hours requested for the job on "
                             "Leonhard. For virtual models usually it "
                             "requires more time than this default value.")
    parser.add_argument("-r", "--repetitions",
                        type=int,
                        default=1,
                        help="Number of repetitions to run_mlagents the same "
                             "experiment")

    args = parser.parse_args()
    print("   Experiment parameters: ")
    print("-" * 100)
    pprint.pprint(vars(args), indent=5)
    print("-" * 100)

    if args.no_submit:
        run_training(args, parser)
    else:
        submit_jobs(args, parser)


if __name__ == '__main__':
    main()


