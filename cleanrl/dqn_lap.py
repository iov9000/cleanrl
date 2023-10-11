# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
import random
import time
from distutils.util import strtobool

from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

gym.envs.register(
    id='Lap-v0',
    entry_point='envs.env_lap_new:Laparos',
    max_episode_steps=10000,
)

from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
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

    # Algorithm specific arguments
    parser.add_argument("--env_id", type=str, default="Lap-v0",
        help="the id of the environment")
    parser.add_argument("--num_envs", type=int, default=1,
        help="number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=1000000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=10000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.1,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=50000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")

    # environment specific arguments
    parser.add_argument('--goal_conditioning', default=True, 
        action='store_false', required=False, help='normalize irl rewards')  
    parser.add_argument('--condition_on_n_tasks_completed', default=True, 
        action='store_false', required=False, help='normalize irl rewards')  
    parser.add_argument('--done_on_collision', default=True, 
        action='store_false', required=False, help='normalize irl rewards') 
    parser.add_argument('--collision_loss', default=False, 
        action='store_true', required=False, help='end episode on collision')  
    parser.add_argument('--hard_collisions', default=True, 
        action='store_false', required=False, help='dont progress on collision')  
    parser.add_argument('--continuous_actions', default=False, 
        action='store_true', required=False, help='normalize irl rewards')  
    parser.add_argument('--render_meshes', default=True, 
        action='store_true', required=False, help='normalize irl rewards')  
    parser.add_argument("--step_size", type=float, default=0.3,
        help="step magnitude")
    parser.add_argument("--state_spec", type=str, default="r3",
        help="state spec (r3/r3so3)")

    # environment specific arguments
    parser.add_argument('--normalize_obs', default=True, 
        action='store_true', required=False, help='normalize observations')  
    parser.add_argument('--normalize_rewards', default=True, 
        action='store_true', required=False, help='normalize irl rewards')  
    parser.add_argument('--clip_action', default=True, 
        action='store_true', required=False, help='clip actions')

    parser.add_argument('--verbose', default=False, required=False, 
        action='store_true', help='verbose output')
    args = parser.parse_args()

    non_default = {
        opt.dest: getattr(args, opt.dest)
        for opt in parser._option_string_actions.values()
        if hasattr(args, opt.dest) and opt.default != getattr(args, opt.dest)
    }

    # fmt: on
    return args, non_default


def make_env(args, env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, gc=args.goal_conditioning, 
                             state_spec=args.state_spec, 
                             step_size=args.step_size,
                             done_on_collision=args.done_on_collision,
                             hard_collisions=args.hard_collisions,
                             plot_meshes=args.render_meshes,
                             collision_loss=args.collision_loss)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        #env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


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

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args, args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

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
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

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
                    print(f"global_step={global_step}, episodic_return={item['final_info'][0]['episode']['r']}")
                if global_step % 1 == 0:
                    writer.add_scalar("charts/episodic_return", item["final_info"][0]['episode']["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["final_info"][0]['episode']["l"], global_step)
                    writer.add_scalar("charts/epsilon", epsilon, global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["final_observation"][0]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                #print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    envs.close()
    writer.close()
