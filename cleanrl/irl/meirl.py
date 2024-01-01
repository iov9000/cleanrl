import gymnasium as gym

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import Adam
from torch.nn.utils import spectral_norm, weight_norm
from irl.utils import ResNetAIRLDisc, MiniGridCNN, AtariCNNBase


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    # torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def linlayer(in_dim, out_dim, bias=True, wnorm=False, snorm=False):
    # return layer_init(nn.Linear(in_dim, out_dim, bias=bias))
    if wnorm:
        return weight_norm(nn.Linear(in_dim, out_dim, bias=bias), 'weight')
    elif snorm:
        return spectral_norm(nn.Linear(in_dim, out_dim, bias=bias), 'weight')
    else:
        return nn.Linear(in_dim, out_dim, bias=bias)


class MEIRLDiscriminator(nn.Module):
    def __init__(self, env, opt, clamp_magnitude = 10.0):
        super(MEIRLDiscriminator, self).__init__()

        # parse options
        self.env = env
        self.opt = opt
        self.layer_dims = opt.d_layer_dims
        self.lr = opt.disc_lr
        self.use_actions = opt.use_actions
        self.use_dones = opt.use_dones
        self.use_next_obs = opt.use_next_obs        
        self.irm_coeff = opt.irm_coeff
        self.l2_coeff = opt.l2_coeff
        self.lip_coeff = opt.lip_coeff
        self.use_cnn_base = opt.use_cnn_base
        self.bias = opt.use_disc_bias
        self.is_atari = 'atari' in self.opt.exp_name
        self.snorm = opt.use_spectral_norm
        self.wnorm = opt.use_weight_norm
        self.identity_features = opt.identity_features

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

        dim0 = ob_shapes[-1]
        if self.use_actions:
            dim0 = dim0 + ac_shapes[-1]
        if opt.use_dones:
            dim0 = dim0 + 1
        if opt.use_next_obs:
            dim0 = dim0 + ob_shapes[-1]

        self.layer_dims = [dim0] + self.layer_dims
        ac_len = ac_shapes[0]

        layer_dims = self.layer_dims

        self.clamp_magnitude = clamp_magnitude
        layer_dims = self.layer_dims

        if self.identity_features:
            self.base = nn.Identity()
            self.phi = nn.Identity()
            self.discriminator = linlayer(in_dim=dim0,
                                                   out_dim=1,
                                                    bias=self.bias,
                                                    wnorm=self.wnorm,
                                                    snorm=self.snorm)
        else:
            if self.is_atari:
                self.base = AtariCNNBase(opt,env, self.use_actions)
                self.discriminator = nn.Linear(512, 1, bias=self.bias)
            else:
                if self.use_cnn_base:
                    self.base = MiniGridCNN(layer_dims, self.use_actions)
                else:
                    self.base = nn.Sequential(linlayer(self.layer_dims[0], self.layer_dims[1], 
                                                        self.bias, self.wnorm, self.snorm),
                                            nonlin)

                self.discriminator_layers = []
                for i in range(2, len(layer_dims)):
                    self.discriminator_layers += [linlayer(in_dim=layer_dims[i - 1],
                                                                out_dim=layer_dims[i],
                                                                bias=self.bias,
                                                                wnorm=self.wnorm,
                                                                snorm=self.snorm),
                                                nonlin]
            self.phi = nn.Sequential(*self.discriminator_layers)

            self.discriminator_layers += [linlayer(in_dim=layer_dims[-1],
                                                    out_dim=1,
                                                        bias=self.bias,
                                                        wnorm=self.wnorm,
                                                        snorm=self.snorm)]

            self.discriminator = nn.Sequential(*self.discriminator_layers)

        if torch.cuda.is_available():
            self.base.cuda()
            self.phi.cuda()
            self.discriminator.cuda()

        # self.module_list = nn.ModuleList([self.base, self.discriminator])
        self.d_optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coeff)

    def base_fwd(self, ob, ac, nob=None, d=None):
        #  match tensor sizes
        if ac is not None:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
        if d is not None and len(ob.shape) != len(d.shape):
            d = torch.unsqueeze(d, -1)

        input_ = [ob]
        if self.use_actions:
            input_.append(ac)
        if self.use_next_obs:
            input_.append(nob)
        if self.use_dones:
            input_.append(d)

        if (self.use_cnn_base or self.is_atari):
            base_out = self.base(*input_)

        else:
            base_out = self.base(torch.cat(input_, axis=-1))

        return base_out

    def forward(self, ob, ac=None, nob=None, d=None):
        base_out = self.base_fwd(ob, ac, nob, d)

        phi = self.phi(base_out)
        d_out = self.discriminator(phi)
        output = torch.clamp(d_out, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        return d_out, phi

    def get_reward(self, ob, ac):
        if self.use_actions and (self.use_cnn_base or self.is_atari):
            base_out = self.base(ob, ac)
        elif self.use_actions and not (self.use_cnn_base or self.is_atari):
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)

        phi = self.phi(base_out)
        d_out = self.discriminator(phi)
        self.reward = torch.clamp(d_out, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        return self.reward

    def irm_penalty(self, logits, y):
        scale = torch.tensor(1.).requires_grad_()
        #loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        loss = F.mse_loss(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.discriminator(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()

        return grad_pen

    def compute_loss(self, update_dict):    
        r_policy, self.phi_policy = self.forward(update_dict['policy_obs'], update_dict['policy_acs'])
        r_expert, self.phi_expert = self.forward(update_dict['expert_obs'], update_dict['expert_acs'])

        #self.diff_loss = r_policy.mean() - torch.exp(r_expert).mean()
        self.diff_loss = r_policy.mean() - r_expert.mean()
        r_out = torch.cat([r_expert, r_policy], dim=0)

        #self.grad_penalty = torch.norm(self.phi_expert-self.phi_policy)
        self.grad_penalty = self.irm_penalty(r_policy, r_expert)

        if self.opt.lip_coeff > 0:
            lip_penalty, grad_mix = self.lip_penalty(update_dict, self.opt.lip_p)
            grad_mix_norm = torch.norm(grad_mix)
        else:
            lip_penalty, grad_mix_norm = 0,0

        self.loss = self.diff_loss + self.irm_coeff * self.grad_penalty + self.lip_coeff*lip_penalty

        output_dict = {}
        output_dict['total_loss'] = self.loss
        output_dict['d_loss'] = self.diff_loss
        output_dict['grad_penalty'] = self.grad_penalty
        output_dict['lip_penalty'] = lip_penalty

        return output_dict

    def lip_penalty(self, update_dict, p=1):
        policy_obs = update_dict['policy_obs']
        policy_acs = update_dict['policy_acs']
        policy_obs_next = update_dict['policy_obs_next']
        policy_dones = update_dict['policy_dones']
        exp_obs = update_dict['expert_obs']
        exp_acs = update_dict['expert_acs']
        exp_obs_next = update_dict['expert_obs_next']
        exp_dones = update_dict['expert_dones']

        if len(exp_obs.shape) != len(exp_dones.shape):
            exp_dones = torch.unsqueeze(exp_dones, -1)

        if len(policy_obs.shape) != len(policy_dones.shape):
            policy_dones = torch.unsqueeze(policy_dones, -1)

        obs_epsilon = torch.rand(policy_obs.shape)
        interp_obs = obs_epsilon * policy_obs + (1 - obs_epsilon) * exp_obs
        interp_obs.requires_grad = True  # For gradient calculation
        
        input_ = [interp_obs]
        if self.use_actions:
            acs_epsilon = torch.rand(policy_acs.shape)
            interp_acs = acs_epsilon * policy_acs + (1 - acs_epsilon) * exp_acs
            interp_acs.requires_grad = True  # For gradient calculation
            input_.append(interp_acs)

        if self.use_next_obs:
            nobs_epsilon = torch.rand(policy_obs_next.shape)
            interp_next_obs = nobs_epsilon * policy_obs_next + (1 - nobs_epsilon) * exp_obs_next
            interp_next_obs.requires_grad = True  # For gradient calculation
            input_.append(interp_next_obs)
        if self.use_dones:
            d_epsilon = torch.rand(policy_dones.shape)
            interp_d = d_epsilon * policy_dones + (1 - d_epsilon) * exp_dones
            interp_d.requires_grad = True  # For gradient calculation
            input_.append(interp_d)

        estimate, phi = self.forward(*input_)

        # policy_obs = update_dict['policy_obs']
        # expert_obs = update_dict['expert_obs']
        # policy_obs.requires_grad = True
        # expert_obs.requires_grad = True

        gradient_mix = torch.autograd.grad(estimate.sum(), input_, create_graph=True)[0]
        # gradient_p = torch.autograd.grad(
        #     estimate.sum(), policy_obs, create_graph=True)[0]
        # gradient_e = torch.autograd.grad(
        #     estimate.sum(), expert_obs, create_graph=True)[0]

        # Norm's gradient could be NaN at 0. Use our own safe_norm
        safe_norm = (torch.sum(gradient_mix ** 2, dim=1) + 1e-8).sqrt()
        # L1
        gradient_mag = torch.mean((safe_norm - p) ** 2)

        return gradient_mag, gradient_mix #, gradient_e, gradient_p


    def update(self, loss):
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()

class MeirlReward(gym.Wrapper):
    def __init__(self, env, opt, disc):
        super().__init__(env=env)
        # self.reward_fn = make_network(**reward_fn_spec)
        self.use_actions = opt.use_actions
        self.discriminator = disc
        self.obs = None
        self.episode_return = 0

    def step(self, action):
        next_obs, gt_reward, term, trunc, info = self.env.step(action)
        done = term or trunc
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
        self.episode_return += irl_reward
        if done:
            info['episode']["ep_rew_irl"] = self.episode_return
            self.episode_return = 0

        return next_obs, irl_reward, term, trunc, info