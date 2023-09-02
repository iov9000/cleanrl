import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import ot
import gym
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

def load_hf_demos(env_name, folder=None, n_demos=10, bary_enhance=False):
    if folder is None:
        folder = "demos"
    expert_demos = {}
    try:
        #expert_demos['all'] = np.load(os.path.join(folder, f"demo_hf_{env_name}_{n_demos}.npy"))
        expert_demos['all'] = pickle.load(open(os.path.join(folder, f"demo_hf_{env_name}_{n_demos}.pkl"), 'rb'))
    except:
        print("Generate demos from HuggingFace Hub first")
        assert False

    return expert_demos


def load_lap_demos(folder='demos/davos_data', n_demos=1, bary_enhance=False, offset=100, obs_shape=3):
    expert_demos = {}
    obs = []
    acs = []
    rew = []
    dones = []
    cnt = 0

    for f in os.listdir(folder):
        cnt += 1
        if 'npy' in f:
            demo = {}
            o = np.load(os.path.join(folder, f))[offset:, :obs_shape]
            a = np.zeros([len(o)])
            r = np.zeros([len(o)])
            d = np.zeros([len(o)]) 
            d[-1] = 1

            obs.append(o)
            acs.append(a)
            rew.append(r)
            dones.append(d)
        if cnt > n_demos:
            break

    expert_demos['obs'] = np.concatenate(obs,0)
    expert_demos['acs'] = np.concatenate(acs,0)
    expert_demos['rew'] = np.concatenate(rew,0)
    expert_demos['done'] = np.concatenate(dones,0)

    return {'all': expert_demos}


def get_trajectory_list(demos, oa_cat=False):
    done_cnt = 0
    trajs = []
    ep = []
    obs = demos['obs']
    acs = demos['acs']
    dones = demos['done']
    if len(acs.shape) == 1:
        acs = np.expand_dims(acs,-1)

    for o,a,d in zip(obs,acs,dones):
        if oa_cat:
            oa = np.concatenate([o,a])
        else:
            oa = o
        ep.append(oa.astype(np.float32))
        if int(d) == 1:
            done_cnt += 1
            trajs.append(np.array(ep))
            ep = []
        
    return trajs

def compute_sw_barycenter(a,b,lr=1e3,nb_iter_max=1000,bary_size=1000,n_proj=100):
    x1_torch = torch.from_numpy(a)
    x2_torch = torch.from_numpy(b)
    xbinit = np.random.randn(bary_size, a.shape[-1]).astype(np.float32)
    xbary_torch = torch.tensor(xbinit).requires_grad_(True)

    #x_all = np.zeros((nb_iter_max, xbary_torch.shape[0], 2))

    loss_iter = []

    # generator for random permutations
    gen = torch.Generator()
    gen.manual_seed(42)

    alpha = 0.5 # adaptive alpha schedule?

    for i in tqdm(range(nb_iter_max)):

        loss = alpha * ot.sliced_wasserstein_distance(xbary_torch, x2_torch, n_projections=n_proj, seed=gen) \
            + (1 - alpha) * ot.sliced_wasserstein_distance(xbary_torch, x1_torch, n_projections=n_proj, seed=gen)

        loss_iter.append(loss.clone().detach().cpu().numpy())
        loss.backward()

        # performs a step of projected gradient descent
        with torch.no_grad():
            grad = xbary_torch.grad
            xbary_torch -= grad * lr  # / (1 + i / 5e1)  # step
            xbary_torch.grad.zero_()
            #x_all[i, :, :] = xbary_torch.clone().detach().cpu().numpy()

    xb = xbary_torch.clone().detach().cpu().numpy()
    
    return xb

def enhance_demos_with_barycenters(args, demos, test_env, lr=1e3,nb_iter_max=1000,bary_size=1000,n_proj=100):
    demo_list = get_trajectory_list(demos['all'], args.use_actions)
    pairs = [(a, b) for idx, a in enumerate(demo_list) for b in demo_list[idx + 1:]]
    barys = []
    for i,(a,b) in enumerate(pairs):
        if i == args.enhance_barys:
            break
        else:
            barys.append(compute_sw_barycenter(a,b,lr,nb_iter_max,bary_size,n_proj))

    f = plt.figure(figsize=(10,10))
    for b in barys:
        ob_shape = test_env.observation_space.shape[-1]
        if len(test_env.action_space.shape) == 0:
            ac_shape = 1
        else:
            ac_shape = test_env.action_space.shape[-1]
        o = b[:,:ob_shape]
        a = b[:,ob_shape:ob_shape+ac_shape]
        d = b[:,-1]
        plt.scatter(o[:,0],o[:,1])
        if hasattr(test_env, 'get_reward'):
            r = test_env.get_reward(o) 
        else:
            r = np.zeros(d.shape)

        # truncate actions 
        a_ = np.squeeze(a).astype(np.int32)
        a_ = np.minimum(np.maximum(0,a_),4,a_)

        demos['all']['obs'] = np.concatenate([demos['all']['obs'], o],0)
        demos['all']['acs'] = np.concatenate([demos['all']['acs'], a_],0)
        demos['all']['done'] = np.concatenate([demos['all']['done'], d],0)
        demos['all']['rew'] = np.concatenate([demos['all']['rew'], r],0)

    plt.show()

    #demo_list.extend(barys)
    
    return demos
    
def demos_gen(data, batch_size):
    """  Yields batch of specified size """
    if batch_size<=0:
       return
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]

def demos_gen_dict(data, batch_size, shuffle=False):
    """  Yields batch of specified size """
    if batch_size <= 0:
       return
    
    b_inds = np.arange(len(data['obs']))
    if shuffle:
        np.random.shuffle(b_inds)

    for i in range(0, len(data['obs']), batch_size):
        end = i + batch_size
        mb_inds = b_inds[i:end]
        yield {'obs': data['obs'][mb_inds], 
               'acs': data['acs'][mb_inds],
               'rew': data['rew'][mb_inds],
               'done': data['done'][mb_inds]}

def prepare_batch_update_irl(env, opt, expert_demos, obs, acs, dones, policy):
    if opt.use_sb_ppo:
        ac_sample = env.action_space.sample()
    else:
        ac_sample = env.single_action_space.sample()

    #print(obs.shape, acs.shape, dones.shape, expert_demos['obs'].shape, expert_demos['acs'].shape, expert_demos['done'].shape)

    if isinstance(ac_sample, int) or isinstance(ac_sample, np.int64):  
        ac_shape = 1
    else:
        ac_shape = ac_sample.shape[-1]

    if isinstance(obs, torch.Tensor):
        obs = obs.cpu().numpy()
    if isinstance(acs, torch.Tensor):
        acs = acs.cpu().numpy()
    if isinstance(dones, torch.Tensor):
        dones = np.squeeze(dones.cpu().numpy())

    # flatten first dimension to use samples from all env
    if 'atari' in opt.exp_name:
        #acs = np.reshape(acs, (-1,1))
        obs = np.reshape(obs, [-1, *obs.shape[2:]])
        acs = np.reshape(acs, [-1, ac_shape]) 

        dones = np.reshape(dones, (-1))
    else:
        obs = np.reshape(obs, [-1, obs.shape[-1]])
        acs = np.reshape(acs, [-1, ac_shape])
        if ac_shape == 1:
            acs = np.squeeze(acs,-1)
        
        # DONES make a big difference?
        #print(dones.shape)
        dones = np.reshape(dones, (-1))

    obs_next = np.concatenate([obs[1:], np.expand_dims(obs[-1], 0)], axis=0)

    # sample expert_demos
    if isinstance(expert_demos, dict):
        rewards = expert_demos['rew']
        expert_obs = expert_demos['obs']
        expert_acs = expert_demos['acs']
        expert_dones = np.squeeze(expert_demos['done'])
        #expert_dones = expert_demos['done']
    else:
        expert_ob_ac_done_reward = expert_demos#[np.random.randint(0, expert_demos.shape[0], opt.batch_size), :]
        expert_dones = expert_ob_ac_done_reward[:, -1]
        rewards = expert_ob_ac_done_reward[:, -2]
        expert_ob_ac = expert_ob_ac_done_reward[:, :-2]
        expert_obs = expert_ob_ac[:, :-ac_shape]
        expert_acs = expert_ob_ac[:, -ac_shape:]

    expert_obs_next = np.concatenate([expert_obs[1:], np.expand_dims(expert_obs[-1], 0)],
                                     axis=0)  # repeat last observation

    N = expert_obs.shape[0]
    T = 1000

    policy_traj = np.empty((N, T, obs.shape[-1]), dtype=np.float32)
    a_buffer = np.empty((N, T-1, ac_shape), dtype=np.float32)
    expert_traj = np.empty((N, T, obs.shape[-1]), dtype=np.float32)

    # convert to torch tensors
    obs_t = torch.from_numpy(obs).type(torch.get_default_dtype())
    acs_t = torch.from_numpy(acs).type(torch.get_default_dtype())   
    expert_obs_t = torch.from_numpy(expert_obs).type(torch.get_default_dtype())
    expert_acs_t = torch.from_numpy(expert_acs).type(torch.get_default_dtype())


    # policy_ob_ac = np.concatenate([obs, acs], 1)
    # eval lprobs conditioned on obs, acs

    #print(expert_obs_t.shape, expert_acs_t.shape, obs_t.shape, acs_t.shape)

    with torch.no_grad():

        if torch.cuda.is_available():
            if opt.on_policy:
                if opt.use_sb_ppo:
                    _, _, expert_lprobs_t = policy.policy.forward(expert_obs_t.cuda())
                    _, _, policy_lprobs_t = policy.policy.forward(obs_t.cuda())
                else:
                    _, expert_lprobs_t, _, _ = policy.get_action_and_value(expert_obs_t.cuda(), expert_acs_t.cuda())
                    _, policy_lprobs_t, _, _ = policy.get_action_and_value(obs_t.cuda(), acs_t.cuda())
            else:
                _, expert_lprobs_t, _ = policy.get_action(expert_obs_t.cuda())
                _, policy_lprobs_t, _ = policy.get_action(obs_t.cuda())
        else:
            if opt.on_policy:
                if opt.use_sb_ppo:
                    _, _, expert_lprobs_t = policy.policy.forward(expert_obs_t)
                    _, _, policy_lprobs_t = policy.policy.forward(obs_t)
                else:
                    _, expert_lprobs_t, _, _ = policy.get_action_and_value(expert_obs_t, expert_acs_t)
                    _, policy_lprobs_t, _, _ = policy.get_action_and_value(obs_t, acs_t)
            else:
                _, expert_lprobs_t, _ = policy.get_action(expert_obs_t)
                _, policy_lprobs_t, _ = policy.get_action(obs_t)

    all_obs = np.concatenate([expert_obs, obs], axis=0)
    all_next_obs = np.concatenate([expert_obs_next, obs_next], axis=0)
    all_acs = np.concatenate([expert_acs, acs], axis=0)
    lprobs = np.concatenate([expert_lprobs_t.cpu().numpy(), policy_lprobs_t.cpu().numpy()], axis=0)
    all_dones = np.concatenate([expert_dones, dones], axis=0)

    expert_obs_t = torch.from_numpy(expert_obs).type(torch.get_default_dtype())
    expert_acs_t = torch.from_numpy(expert_acs).type(torch.get_default_dtype())
    expert_obs_next_t = torch.from_numpy(expert_obs_next).type(torch.get_default_dtype())
    expert_dones_t = torch.from_numpy(expert_dones).type(torch.get_default_dtype())
    policy_obs_t = torch.from_numpy(obs).type(torch.get_default_dtype())
    policy_acs_t = torch.from_numpy(acs).type(torch.get_default_dtype())
    policy_obs_next_t = torch.from_numpy(obs_next).type(torch.get_default_dtype())
    policy_dones_t = torch.from_numpy(dones).type(torch.get_default_dtype())
    all_obs_t = torch.from_numpy(all_obs).type(torch.get_default_dtype())
    all_obs_next_t = torch.from_numpy(all_next_obs).type(torch.get_default_dtype())
    all_acs_t = torch.from_numpy(all_acs).type(torch.get_default_dtype())
    all_lprobs_t = torch.from_numpy(lprobs).type(torch.get_default_dtype())
    all_dones_t = torch.from_numpy(all_dones).type(torch.get_default_dtype())

    if torch.cuda.is_available():
        expert_obs_t = expert_obs_t.cuda()
        expert_obs_next_t = expert_obs_next_t.cuda()
        expert_acs_t = expert_acs_t.cuda()
        expert_lprobs_t = expert_lprobs_t.cuda()
        expert_dones_t = expert_dones_t.cuda()
        policy_obs_next_t = policy_obs_next_t.cuda()
        policy_obs_t = policy_obs_t.cuda()
        policy_acs_t = policy_acs_t.cuda()
        policy_lprobs_t = policy_lprobs_t.cuda()
        policy_dones_t = policy_dones_t.cuda()

        all_obs_next_t = all_obs_next_t.cuda()
        all_obs_t = all_obs_t.cuda()
        all_acs_t = all_acs_t.cuda()
        all_lprobs_t = all_lprobs_t.cuda()
        all_dones_t = all_dones_t.cuda()

    update_dict = {}
    update_dict['expert_obs'] = expert_obs_t
    update_dict['expert_obs_next'] = expert_obs_next_t
    update_dict['expert_acs'] = expert_acs_t
    update_dict['expert_lprobs'] = expert_lprobs_t
    update_dict['expert_dones'] = expert_dones_t

    update_dict['policy_obs'] = policy_obs_t
    update_dict['policy_obs_next'] = policy_obs_next_t
    update_dict['policy_acs'] = policy_acs_t
    update_dict['policy_lprobs'] = policy_lprobs_t
    update_dict['policy_dones'] = policy_dones_t

    update_dict['all_obs'] = all_obs_t
    update_dict['all_obs_next'] = all_obs_next_t
    update_dict['all_acs'] = all_acs_t
    update_dict['all_lprobs'] = all_lprobs_t
    update_dict['all_dones'] = all_dones_t

    return update_dict


class MiniGridCNN(nn.Module):
    def __init__(self, layer_dims, use_actions=False):
        super(MiniGridCNN, self).__init__()
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
        )
        if use_actions:
            self.lin = nn.Linear(65, layer_dims[1])
        else:
            self.lin = nn.Linear(64, layer_dims[1])

        self.use_actions = use_actions

    def forward(self, x, a=None):
        x = torch.transpose(x, 1, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        if a is not None and self.use_actions:
            if len(x.shape) != len(a.shape):
                ac = torch.unsqueeze(a, -1)
            x = self.lin(torch.cat([x, torch.unsqueeze(a, -1)], 1))
        else:
            x = self.lin(x)
        return x

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AtariCNNBase(nn.Module):
    def __init__(self, opt, env, use_actions=False):
        super().__init__()
        ob_shapes = list(env.observation_space.shape)
        ac_shapes = list(env.action_space.shape)
        if not ac_shapes:
            ac_shapes = [1]

        if use_actions:
            ac_dim = ac_shapes[-1]
        else:
            ac_dim = 0

        self.obs_network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU())
            #nn.Flatten())
            
        self.obs_acs_network = nn.Sequential(
            layer_init(nn.Linear(64 * 7 * 7 + ac_dim, 512)),
            nn.ReLU(),
        )
        #self.critic = layer_init(nn.Linear(512, 1), std=1)

    def forward(self, ob, ac=None):
        if len(ob.shape) == 3:
            ob = ob.unsqueeze(0)

        ob_enc = self.obs_network(ob/255)
        ob_enc = ob_enc.reshape(ob_enc.shape[0], -1)
        if ac is not None:
            if len(ac.shape) == 0:
                ac = ac.unsqueeze(0).unsqueeze(0)
            elif len(ac.shape) == 1:
                ac = ac.unsqueeze(0)

            ob_enc = torch.cat([ob_enc, ac], dim=-1)
            x = self.obs_acs_network(ob_enc)
        else:
            x = self.obs_acs_network(ob_enc)
        
        return x



class ResNetAIRLDisc(nn.Module):
    def __init__(
        self,
        input_dim,
        num_layer_blocks=2,
        hid_dim=100,
        hid_act='relu',
        use_bn=False,
        clamp_magnitude=10.0
    ):
        super().__init__()

        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()

        self.clamp_magnitude = clamp_magnitude
        self.input_dim = input_dim

        self.first_fc = nn.Linear(input_dim, hid_dim)

        self.blocks_list = nn.ModuleList()
        for i in range(num_layer_blocks - 1):
            block = nn.ModuleList()
            block.append(nn.Linear(hid_dim, hid_dim))
            if use_bn: block.append(nn.BatchNorm1d(hid_dim))
            block.append(hid_act_class())
            self.blocks_list.append(nn.Sequential(*block))
        self.blocks_list = self.blocks_list

        self.last_fc = nn.Linear(hid_dim, 1)

    def forward(self, batch):
        x = self.first_fc(batch)
        for block in self.blocks_list:
            x = x + block(x)
        output = self.last_fc(x)
        if self.clamp_magnitude is not None:
            output = torch.clamp(output, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        return output  

class AntCFWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.env.observation_space = gym.spaces.Box(low = -np.inf*np.ones((27,)), 
                                            high = np.inf*np.ones((27,)),
                                            dtype = np.float32)
    
        super().__init__(env=self.env)

    def reset(self):
        o = self.env.reset()
        return o[:27]

    def step(self, action):
        o, r, d, i = self.env.step(action)
        return o[:27], r, d, i

class MujocoTestWrapper(gym.Wrapper):
    def __init__(self, env, eps=0.1):
        self.env = env
        self.eps = eps

        self.perturb(eps)     
        super().__init__(env=self.env)

    def perturb(self, eps):
        self.init_qpos = np.array(self.init_qpos)
        self.init_qvel = np.array(self.init_qvel)
        self.init_qpos = self.init_qpos * eps*np.random.randn(*self.init_qpos.shape)
        self.init_qvel = self.init_qvel * eps*np.random.randn(*self.init_qvel.shape)
        self.init_qpos = tuple(self.init_qpos)
        self.init_qvel = tuple(self.init_qvel)     

    def reset(self):
        o = self.env.reset()
        self.perturb(self.eps)     

        return o

    def step(self, action):
        o, r, d, i = self.env.step(action)
        o = o + self.eps*np.random.randn(*o.shape)
        return o, r, d, i


def evaluate_model(args, actor, device):
    test_env = gym.make(args.env_id)
    if args.clip_action:
        test_env = gym.wrappers.ClipAction(test_env)
    if args.normalize_obs:
        test_env = gym.wrappers.NormalizeObservation(test_env)
        test_env = gym.wrappers.TransformObservation(
            test_env, lambda obs: np.clip(obs, -10, 10))
    mu, std, d = save_traj_dict(args, args.env_id, test_env, actor, device)

    test_env.close()

    return mu, std, d

def save_traj_dict(opt, env_name, env, model, device, nr_trajectories=10, min_total_rew=-1e6):
    num_steps = 0
    expert_obs = []
    expert_acs = []
    expert_rew = []
    expert_done = []

    if isinstance(nr_trajectories, list):
        nr_trajectories = nr_trajectories[0]

    cnt = 0
    ep_rew = []
    while cnt < nr_trajectories:
        ob, _ = env.reset(seed=opt.seed)
        done = False
        total_reward = 0
        episode_obs = []
        episode_acs = []
        episode_rew = []
        episode_done = []

        while not done:
            with torch.no_grad():
                if 'sac' in opt.exp_name:
                    action, _, _ = model.get_action(torch.Tensor(ob).to(device).unsqueeze(0))
                else:
                    action, logprob, _, value = model.get_action_and_value(torch.Tensor(ob).to(device).unsqueeze(0))
            next_ob, reward, term, trunc, _ = env.step(action.squeeze().detach().cpu().numpy())
            done = term or trunc
            ob = next_ob
            total_reward += reward
            episode_obs.append(ob)
            episode_acs.append(action)
            episode_rew.append(reward)
            episode_done.append(done)

            num_steps += 1

        ep_rew.append(total_reward)
        print("episode:", cnt, "reward:", total_reward)
        if total_reward > min_total_rew:
            cnt += 1
            expert_obs.extend(episode_obs)
            expert_acs.extend(episode_acs)
            expert_rew.extend(episode_rew)
            expert_done.extend(episode_done)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{ts}_model_{env_name}_{nr_trajectories}.pkl" 
    expert_obs = np.stack(expert_obs)
    expert_acs = np.stack(expert_acs)
    expert_rew = np.stack(expert_rew)
    expert_done = np.stack(expert_done)

    ep_rew = np.array(ep_rew)

    print("Mean / std over {nr_trajectories} episodes: ", np.mean(ep_rew), np.std(ep_rew))

    d = {'obs': expert_obs, 'acs': expert_acs, 'rew': expert_rew, 'done': expert_done}
    #pickle.dump(d, open(filename, 'wb'))

    return np.mean(ep_rew), np.std(ep_rew), d
