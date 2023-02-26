import gym

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import Adam
from torch.nn.utils import spectral_norm
from irl.utils import ResNetAIRLDisc, MiniGridCNN

class fIRLDiscriminator(nn.Module):
    def __init__(self, env, opt):
        super(fIRLDiscriminator, self).__init__()

        # parse options
        self.env = env
        self.layer_dims = opt.d_layer_dims
        self.lr = opt.disc_lr
        self.use_actions = opt.use_actions
        self.irm_coeff = opt.irm_coeff
        self.l2_coeff = opt.l2_coeff
        self.lip_coeff = opt.lip_coeff
        self.use_cnn_base = opt.use_cnn_base
        self.bias = opt.use_disc_bias
        self.use_resnet = opt.firl_use_disc_resnet

        self.clamp_magnitude = 10

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

        if self.lip_coeff > 0:
            f = lambda x: spectral_norm(x)
        else:
            f = lambda x: x

        # reward function g_\psi
        if self.use_cnn_base:
            self.base = MiniGridCNN(layer_dims, self.use_actions)
        else:
            self.base = nn.Sequential(f(torch.nn.Linear(self.layer_dims[0],
                                                      self.layer_dims[1], self.bias)),
                                      torch.nn.ReLU())

        self.reward_layers = []
        for i in range(2, len(self.layer_dims)):
            self.reward_layers += [f(torch.nn.Linear(in_features=self.layer_dims[i - 1],
                                                   out_features=self.layer_dims[i],
                                                   bias=self.bias)),
                                                   torch.nn.ReLU()]

        self.reward_layers += [f(torch.nn.Linear(in_features=self.layer_dims[-1],
                                               out_features=1,
                                               bias=self.bias))]
        self.reward = nn.Sequential(*self.reward_layers)

        # shaping function h_\phi
        if self.use_cnn_base:
            self.base_v = MiniGridCNN(layer_dims, use_actions=False)
        else:
            self.base_v = nn.Sequential(f(torch.nn.Linear(self.layer_dims[0],
                                                        self.layer_dims[1], self.bias)),
                                        torch.nn.Tanh())

        if self.use_resnet:
            self.disc=ResNetAIRLDisc(input_dim=self.layer_dims[1])
        else:          
            self.disc_layers = []
            for i in range(2, len(self.layer_dims)):
                self.disc_layers += [f(torch.nn.Linear(in_features=self.layer_dims[i - 1],
                                                    out_features=self.layer_dims[i],
                                                    bias=self.bias)),
                                    torch.nn.Tanh()]

            self.disc_layers += [f(torch.nn.Linear(in_features=self.layer_dims[-1],
                                                out_features=1,
                                                bias=self.bias))]
            self.disc = nn.Sequential(*self.disc_layers)

        if torch.cuda.is_available():
            self.base.cuda()
            self.base_v.cuda()
            self.reward.cuda()
            self.disc.cuda()

        self.reward_optimizer = Adam(list(self.base.parameters()) + list(self.reward.parameters()), lr=self.lr, 
                                weight_decay=self.l2_coeff)
        self.d_optimizer = Adam(list(self.base_v.parameters()) +  list(self.disc.parameters()), lr=self.lr, 
                                weight_decay=self.l2_coeff)

    def forward(self, ob, next_ob, ac, lprobs):
        # forward the nn models
        reward = self.get_reward(ob, ac)
        reward = torch.clamp(reward, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)

        return reward

    def disc_forward(self, ob, ac):
        if self.use_actions and self.use_cnn_base:
            base_out = self.base_v(ob, ac)
        elif self.use_actions and not self.use_cnn_base:
            base_out = self.base_v(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base_v(ob)

        d_out = self.disc(base_out)
        return d_out

    def get_reward(self, ob, ac):
        if self.use_actions and self.use_cnn_base:
            base_out = self.base(ob, ac)
        elif self.use_actions and not self.use_cnn_base:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)

        # rew, v, v_n, d_out = self.forward(ob, next_ob, ac, lprobs) TODO??
        # return torch.sigmoid

        reward = self.reward(base_out)
        reward = torch.clamp(reward, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)

        return reward

    def compute_penalty(self, logits, y):
        scale = torch.tensor(1.).requires_grad_()
        loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        g1 = autograd.grad(loss[0::2].mean(), [scale], create_graph=True)[0]
        g2 = autograd.grad(loss[1::2].mean(), [scale], create_graph=True)[0]
        return (g1 * g2).sum()

    def get_value(self, ob, ac):
        if self.use_actions and self.use_cnn_base:
            base_out = self.base(ob, ac)
        elif self.use_actions and not self.use_cnn_base:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)

        # rew, v, v_n, d_out = self.forward(ob, next_ob, ac, lprobs) TODO??
        return self.value(base_out)

    def irm_penalty(self, logits, y):
        scale = torch.tensor(1.).requires_grad_()
        loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)

    def lip_penalty(self, update_dict):
        obs_epsilon = torch.rand(update_dict['policy_obs'].shape)
        interp_obs = obs_epsilon * update_dict['policy_obs'] + (1 - obs_epsilon) * update_dict['expert_obs']
        interp_obs.requires_grad = True  # For gradient calculation

        obs_epsilon = torch.rand(update_dict['policy_obs_next'].shape)
        interp_obs_next = obs_epsilon * update_dict['policy_obs_next'] + (1 - obs_epsilon) * update_dict[
            'expert_obs_next']
        interp_obs_next.requires_grad = True  # For gradient calculation

        action_epsilon = torch.rand(update_dict['policy_acs'].shape)
        interp_acs = action_epsilon * update_dict['policy_acs'] + (1 - action_epsilon) * update_dict['expert_acs']
        interp_acs.requires_grad = True
        encoder_input = [interp_obs, interp_acs, interp_obs_next]
        _, _, _, estimate = self.forward(interp_obs,
                                         interp_obs_next, interp_acs, update_dict['policy_lprobs'])

        gradient = torch.autograd.grad(estimate.sum(), encoder_input, create_graph=True)[0]
        # Norm's gradient could be NaN at 0. Use our own safe_norm
        safe_norm = (torch.sum(gradient ** 2, dim=1) + 1e-8).sqrt()
        gradient_mag = torch.mean((safe_norm - 1) ** 2)

        return gradient_mag

    def compute_loss(self, update_dict):
        reward_loss, norm_logits = self.f_div_disc_loss('rkl', False, update_dict)

        # -> Jensen Shannon? Mixture of experts?
        #grad_penalty = self.irm_penalty(d_out, labels)
        # grad_penalty = self.irm_penalty(expert_estimate, torch.zeros(expert_estimate.size()))
        grad_penalty = 0

        # lip_penalty = self.lip_penalty(update_dict)
        loss = reward_loss + self.irm_coeff * grad_penalty
        if self.irm_coeff > 1.0:
            loss /= self.irm_coeff

        output_dict = {}
        output_dict['total_loss'] = loss
        output_dict['d_loss'] = reward_loss
        output_dict['grad_penalty'] = grad_penalty
        # output_dict['lip_penalty'] = lip_penalty

        # return self.loss, discriminator_loss, policy_estimate.mean(), expert_estimate.mean(), self.grad_penalty
        return output_dict

    def update_disc(self, update_dict):
        d_out = self.disc_forward(update_dict['all_obs'], update_dict['all_acs'])
        expert_out, policy_out = torch.chunk(d_out, chunks=2, dim=0)

        ## XXX: switched order: 1 for expert, 0 for policy..
        labels = torch.cat([torch.ones(expert_out.size()),
                            torch.zeros(policy_out.size())])

        bce_loss = F.binary_cross_entropy_with_logits(d_out, labels)
        grad_penalty = self.irm_penalty(d_out, labels)
        # grad_penalty = self.irm_penalty(expert_estimate, torch.zeros(expert_estimate.size()))

        # lip_penalty = self.lip_penalty(update_dict)
        loss = bce_loss + self.irm_coeff * grad_penalty
        if self.irm_coeff > 1.0:
            loss /= self.irm_coeff

        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step() 

        return loss

    def sum_over_dones(self, t, dones):
        j = 0
        sums = []
        for i in torch.where(dones > 0):
            print(i,j)
            sums.append(torch.sum(t[j:i],0))
            j=i
        
        return torch.stack(sums).type(torch.get_default_dtype())

    def f_div_disc_loss(self, div: str, IS: bool, update_dict, expert_trajs=None):
        # please add eps to expert density, not here
        assert div in ['fkl', 'rkl', 'js']
        #s, _, log_a = samples
        #if expert_trajs is not None:
        #    assert expert_trajs.ndim == 3 and expert_trajs.shape[1:] == s.shape[1:]
        #    s = np.concatenate((s, expert_trajs), axis=0) # NOTE: this won't change samples variable
        #N, T, d = s.shape

        T = len(update_dict['all_obs'])
        #s_vec = s.reshape(-1, d)
        logits = self.disc_forward(update_dict['all_obs'], update_dict['all_acs']) # torch vector

        if div == 'fkl':
            t1 = torch.exp(logits) # (N*T,) p/q TODO: clip
        elif div == 'rkl':
            t1 = logits # (N*T,) log (p/q)
        elif div == 'js':  # https://pytorch.org/docs/master/generated/torch.nn.Softplus.html
            t1 = F.softplus(logits) # (N*T,) log (1 + p/q)

        #t1 = (-t1).view(N, T).sum(1) # NOTE: sign (N,)
        #t2 = self.reward(torch.FloatTensor(s_vec)).view(N, T).sum(1) # (N,)    
        # sum over trajectory dimension.. -> def sum_over_traj (until done)
        rew = self.get_reward(update_dict['all_obs'], update_dict['all_acs'])

        #t2 = self.sum_over_dones(rew, update_dict['all_dones'])
        t2 = rew

        if IS:
            with torch.no_grad():
                traj_reward = self.get_reward(update_dict['all_obs'], update_dict['all_acs']) # (N,)
                traj_reward = self.sum_over_dones(traj_reward, update_dict['all_dones'])
            traj_log_prob = self.sum_over_dones(update_dict['all_lprobs'], update_dict['all_dones']) # (N,)
            t2 = self.sum_over_dones(update_dict['all_lprobs'], update_dict['all_dones'])
            IS_ratio = F.softmax(torch.FloatTensor(traj_reward - traj_log_prob), dim=0) # normalized weight
            surrogate_objective = (IS_ratio * t1 * t2).sum() - (IS_ratio * t1).sum() * (IS_ratio * t2).sum()
        else:
            surrogate_objective = (t1 * t2).mean() - t1.mean() * t2.mean() # sample covariance
        
        # TODO: covariance regularization based on logits / reward output?

        surrogate_objective /= T

        return surrogate_objective, t1 / T # log of geometric mean w.r.t. traj (0 is the borderline)

    def update(self, loss):
        self.reward_optimizer.zero_grad()
        loss.backward()
        self.reward_optimizer.step()

    def save(self, fname):
        torch.save(self.state_dict(), fname)


class firlReward(gym.Wrapper):
    def __init__(self, env, opt, disc):
        super().__init__(env=env)
        # self.reward_fn = make_network(**reward_fn_spec)
        self.use_actions = opt.use_actions
        self.discriminator = disc
        self.obs = None

    def step(self, action):
        next_obs, gt_reward, done, info = self.env.step(action)
        info['gt_reward'] = gt_reward
        if self.obs is not None:
            obs_t = torch.tensor(self.obs, dtype=torch.get_default_dtype())
        else:
            obs_t = torch.tensor(next_obs, dtype=torch.get_default_dtype())

        acs_t = torch.tensor(action, dtype=torch.get_default_dtype())
        next_obs_t = torch.tensor(next_obs, dtype=torch.get_default_dtype())

        with torch.no_grad():
            # get discriminator reward and train on that
            #print([p.norm(2) for p in self.discriminator.parameters()])
            irl_reward = self.discriminator.get_reward(
                obs_t, acs_t).cpu().numpy()
            #irl_reward = self.discriminator.get_reward(obs_t, acs_t, next_obs_t).cpu().numpy()
        self.obs = next_obs

        return next_obs, irl_reward, done, info