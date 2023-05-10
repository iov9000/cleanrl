import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import ot

def load_hf_demos(env_name, folder=None, n_demos=10, bary_enhance=False):
    if folder is None:
        folder = "demos"
    expert_demos = {}
    try:
        expert_demos['all'] = np.load(os.path.join(folder, f"demo_hf_{env_name}_{n_demos}.npy"))
    except:
        print("Generate demos from HuggingFace Hub first")
        assert False

    return expert_demos

def get_trajectory_list(demos):
    done_cnt = 0
    trajs = []
    ep = []
    for i, t in enumerate(demos):
        ep.append(t)
        if t[-1] == 1:
            done_cnt += 1
            trajs.append(np.array(ep))
            ep = []
        
    return trajs

def compute_sw_barycenter(a,b,lr=1e3,nb_iter_max=1000,bary_size=1000,n_proj=100):
    x1_torch = torch.tensor(a)
    x2_torch = torch.tensor(b)
    xbinit = np.random.randn(bary_size, a.shape[-1])
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

def enhance_demos_with_barycenters(demos,lr=1e3,nb_iter_max=1000,bary_size=1000,n_proj=100,n_barys=20):
    demo_list = get_trajectory_list(demos['all'])
    pairs = [(a, b) for idx, a in enumerate(demo_list) for b in demo_list[idx + 1:]]
    barys = []
    for i,(a,b) in enumerate(pairs):
        barys.append(compute_sw_barycenter(a,b,lr,nb_iter_max,bary_size,n_proj))
        if i > n_barys:
            break
    
    demo_list.extend(barys)
    
    return {'all': np.concatenate(demo_list,0)}
    
def demos_gen(data, batch_size):
    """  Yields bacth of specified size """
    if batch_size<=0:
       return
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]

def prepare_batch_update_irl(env, opt, expert_demos, obs, acs, dones, policy):
    ac_sample = env.single_action_space.sample()
    if isinstance(ac_sample, int):
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
    obs = np.reshape(obs, [-1, obs.shape[-1]])
    acs = np.reshape(acs, [-1, ac_shape])
    obs_next = np.concatenate([obs[1:], np.expand_dims(obs[-1], 0)], axis=0)

    # sample expert_demos
    expert_ob_ac_done_reward = expert_demos#[np.random.randint(0, expert_demos.shape[0], opt.batch_size), :]
    expert_dones = expert_ob_ac_done_reward[:, -1]
    rewards = expert_ob_ac_done_reward[:, -2]
    expert_ob_ac = expert_ob_ac_done_reward[:, :-2]
    expert_obs = expert_ob_ac[:, :-ac_shape]
    expert_acs = expert_ob_ac[:, -ac_shape:]
    expert_obs_next = np.concatenate([expert_obs[1:], np.expand_dims(expert_obs[-1], 0)],
                                     axis=0)  # repeat last observation

    N = expert_demos.shape[0]
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
    with torch.no_grad():
        if torch.cuda.is_available():
            if opt.on_policy:
                _, expert_lprobs_t, _, _ = policy.get_action_and_value(expert_obs_t.cuda(), expert_acs_t.cuda())
                _, policy_lprobs_t, _, _ = policy.get_action_and_value(obs_t.cuda(), acs_t.cuda())
            else:
                _, expert_lprobs_t, _ = policy.get_action(expert_obs_t.cuda())
                _, policy_lprobs_t, _ = policy.get_action(obs_t.cuda())
        else:
            if opt.on_policy:
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

