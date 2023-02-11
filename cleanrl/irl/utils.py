import os
import numpy as np
import torch

def load_hf_demos(env_name, folder=None):
    if folder is None:
        folder = "demos"
    expert_demos = {}
    try:
        expert_demos['all'] = np.load(os.path.join(folder, f"demo_hf_{env_name}.npy"))
    except:
        print("Generate demos from HuggingFace Hub first")
        assert False

    return expert_demos

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
    expert_ob_ac_done_reward = expert_demos[np.random.randint(0, expert_demos.shape[0], opt.batch_size), :]
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
