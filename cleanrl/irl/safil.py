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

import copy

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


class SafILDiscriminator(nn.Module):
    def __init__(self, env, opt):
        super(SafILDiscriminator, self).__init__()

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

        if opt.disc_nonlin == 'relu':
            nonlin = nn.ReLU()
        elif opt.disc_nonlin == 'tanh':
            nonlin = nn.Tanh()
        else:
            nonlin = nn.PReLU()

        if isinstance(env.observation_space, gym.spaces.Dict):
            ob_shapes = list(env.observation_space['observation'].shape)
        else:
            ob_shapes = list(env.observation_space.shape)
        #ob_shapes = list(env.observation_space.shape)
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

            self.discriminator_layers += [linlayer(in_dim=layer_dims[-1],
                                                   out_dim=1,
                                                        bias=self.bias,
                                                        wnorm=self.wnorm,
                                                        snorm=self.snorm)]

            self.discriminator = nn.Sequential(*self.discriminator_layers)


        self.safe_base = copy.deepcopy(self.base)
        self.safe_discriminator = copy.deepcopy(self.discriminator)

        if torch.cuda.is_available():
            self.base.cuda()
            self.discriminator.cuda()
            self.safe_base.cuda()
            self.safe_discriminator.cuda()

        self.d_optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coeff)

    def base_forward(self, base, ob, ac):
        if self.use_actions and (self.use_cnn_base or self.is_atari):
            base_out = base(ob, ac)
        elif self.use_actions and not (self.use_cnn_base or self.is_atari):
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = base(ob)
        
        return base_out

    def base_fwd(self, base, ob, ac, nob=None, d=None):
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
            base_out = base(*input_)

        else:
            base_out = base(torch.cat(input_, axis=-1))

        return base_out

    def forward(self, ob, ac=None, nob=None, d=None):
        base_out = self.base_fwd(self.base, ob, ac, nob, d)
        sbase_out = self.base_fwd(self.safe_base, ob, ac, nob, d)

        d_out = self.discriminator(base_out)
        sd_out = self.safe_discriminator(sbase_out)

        if self.opt.div == 'fkl':
            d_out_ = torch.exp(d_out) # (N*T,) p/q TODO: clip
        elif self.opt.div == 'rkl':
            d_out_ = d_out # (N*T,) log (p/q)
        elif self.opt.div == 'js':  # https://pytorch.org/docs/master/generated/torch.nn.Softplus.html
            d_out_ = F.softplus(d_out) # (N*T,) log (1 + p/q)

        return d_out_, F.softplus(sd_out)

    def get_reward(self, ob, ac, nob=None, d=None):
        base_out = self.base_fwd(self.base, ob, ac, nob, d)
        d_out = self.discriminator(base_out)


        if self.opt.div == 'fkl':
            d_out_ = torch.exp(d_out)  # (N*T,) p/q TODO: clip
        elif self.opt.div == 'rkl':
            d_out_ = d_out  # (N*T,) log (p/q)
        elif self.opt.div == 'js':  # https://pytorch.org/docs/master/generated/torch.nn.Softplus.html
            d_out_ = F.softplus(d_out)  # (N*T,) log (1 + p/q)

        # XXX: log D vs log(1-D)!!!!
        self.reward = - torch.squeeze(torch.log(torch.sigmoid(d_out) + 1e-8))

        if self.opt.use_safe_disc_reward:
            sbase_out = self.base_fwd(self.safe_base, ob, ac, nob, d)
            sd_out = self.safe_discriminator(sbase_out)
            safe_reward = - torch.squeeze(torch.log(torch.sigmoid(sd_out) + 1e-8))

            return self.reward + safe_reward
        else:
            return self.reward

    def irm_penalty(self, logits, y):
        scale = torch.tensor(1.).requires_grad_()
        loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        #safe_norm = (torch.sum(grad ** 2, dim=-1) + 1e-8).sqrt()
        # L1
        #gradient_mag = torch.mean((safe_norm - 1) ** 2)
        #return gradient_mag, grad
        return torch.sum(grad ** 2), grad

    def compute_penalty(self, logits, y):
        scale = torch.tensor(1.).requires_grad_()
        loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        g1 = autograd.grad(loss[0::2].mean(), [scale], create_graph=True)[0]
        g2 = autograd.grad(loss[1::2].mean(), [scale], create_graph=True)[0]
        return (g1 * g2).sum()

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

        estimate, _ = self.forward(*input_)

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

    # def compute_lip_pen(self, update_dict):
    #     alpha = torch.rand(update_dict['expert_obs'].size(0), 1)
    #     expert_data = torch.cat([update_dict['expert_obs'], update_dict['expert_acs']], dim=1)
    #     policy_data = torch.cat([update_dict['policy_obs'], update_dict['policy_acs']], dim=1)

    #     alpha = alpha.expand_as(expert_data).to(expert_data.device)

    #     mixup_data = alpha * expert_data + (1 - alpha) * policy_data
    #     mixup_data.requires_grad = True

    #     disc = self.discriminator(mixup_data)
    #     ones = torch.ones(disc.size()).to(disc.device)
    #     grad = autograd.grad(
    #         outputs=disc,
    #         inputs=mixup_data,
    #         grad_outputs=ones,
    #         create_graph=True,
    #         retain_graph=True,
    #         only_inputs=True)[0]

    #     grad_pen = (grad.norm(2, dim=1) - 1).pow(2).mean()

    #     return grad_pen

    def compute_loss(self, update_dict):
        self.policy_obs = update_dict['policy_obs']
        self.policy_acs = update_dict['policy_acs']
        policy_obs_next = update_dict['policy_obs_next']
        policy_dones = update_dict['policy_dones']
        exp_obs = update_dict['expert_obs']
        exp_acs = update_dict['expert_acs']
        exp_obs_next = update_dict['expert_obs_next']
        exp_dones = update_dict['expert_dones']
        supp_obs = update_dict['support_obs']
        supp_acs = update_dict['support_acs']
        supp_obs_next = update_dict['support_obs_next']
        supp_dones = update_dict['support_dones']

        policy_out, policy_sout = self.forward(self.policy_obs, self.policy_acs, policy_obs_next, policy_dones)
        expert_out, _ = self.forward(exp_obs, exp_acs, exp_obs_next, exp_dones)
        _, support_sout = self.forward(supp_obs, supp_acs, supp_obs_next, supp_dones)

        d_out = torch.cat([expert_out, policy_out])
        sd_out = torch.cat([support_sout, policy_out])

        expert_loss = F.binary_cross_entropy_with_logits(
            expert_out,
            torch.ones(expert_out.size()))
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_out,
            torch.zeros(policy_out.size()))

        labels = torch.cat([torch.zeros(expert_out.size()),
                            torch.ones(policy_out.size())])
        labels_ = torch.cat([torch.zeros(support_sout.size()),
                            torch.ones(policy_out.size())])

        labels_flipped = torch.cat([torch.ones(expert_out.size()),
                            torch.zeros(policy_out.size())])

        self.bce_loss = F.binary_cross_entropy_with_logits(d_out, labels)
        self.bce_loss_s = F.binary_cross_entropy_with_logits(sd_out, labels_)

        # self.grad_penalty = self.irm_penalty(d_out, labels)
        if self.opt.lip_coeff > 0:
            lip_penalty, grad_mix = self.lip_penalty(update_dict, self.opt.lip_p)
            grad_mix_norm = torch.norm(grad_mix)
        else:
            lip_penalty, grad_mix_norm = 0,0
        ## TODO: check IRM penalty form! -> ones not zeros!
        if self.opt.irm_loss_type == 'e1':
            self.grad_penalty, grad_irm = self.irm_penalty(
                expert_out, torch.ones(expert_out.size()))
        elif self.opt.irm_loss_type == 'e0':
            self.grad_penalty, grad_irm = self.irm_penalty(
                expert_out, torch.zeros(expert_out.size()))
        if self.opt.irm_loss_type == 'p1':
            self.grad_penalty, grad_irm = self.irm_penalty(
                policy_out, torch.zeros(policy_out.size()))
        elif self.opt.irm_loss_type == 'p0':
            self.grad_penalty, grad_irm = self.irm_penalty(
                policy_out, torch.zeros(policy_out.size()))
        elif self.opt.irm_loss_type == 'e1p0_sep':
            self.grad_penalty, grad_irm = self.irm_penalty(policy_out, torch.zeros(policy_out.size())) +\
                self.irm_penalty(expert_out, torch.ones(expert_out.size()))    
        elif self.opt.irm_loss_type == 'e1p0':
            self.grad_penalty, grad_irm = self.irm_penalty(d_out, labels_flipped)
        elif self.opt.irm_loss_type == 'e1p0sym':
            self.grad_penalty, grad_irm = self.irm_penalty(d_out, labels_flipped) + self.irm_penalty(d_out, labels)
        else:
            self.grad_penalty, grad_irm = self.irm_penalty(d_out, labels)


        #self.loss = self.bce_loss + self.irm_coeff * self.grad_penalty + self.lip_coeff * lip_penalty

        output_dict = {}
        output_dict['d_loss'] = self.bce_loss
        output_dict['sd_loss'] = self.bce_loss_s

        output_dict['lip_penalty'] = lip_penalty
        output_dict['lip_grad_mix'] = grad_mix_norm
        output_dict['expert_bce_loss'] = expert_loss
        output_dict['policy_bce_loss'] = policy_loss
        #output_dict['lip_grad_e'] = grad_e
        #output_dict['lip_grad_p'] = grad_p
        output_dict['grad_penalty'] = self.grad_penalty
        output_dict['irm_grad'] = grad_irm

        return output_dict
        
    def update(self, loss):
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()

class SafilReward(gym.Wrapper):
    def __init__(self, env, opt, disc):
        super().__init__(env=env)
        # self.reward_fn = make_network(**reward_fn_spec)
        self.use_actions = opt.use_actions
        self.discriminator = disc
        self.episode_return = 0
        self.obs = None

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
        done_t = torch.tensor(done, dtype=torch.get_default_dtype())

        with torch.no_grad():
            # get discriminator reward and train on that
            # TODO: use second discriminator as reward?
            irl_reward = self.discriminator.get_reward(
                obs_t, acs_t, next_obs_t, done_t).cpu().numpy()

            #irl_reward = self.discriminator.get_reward(obs_t, acs_t, next_obs_t).cpu().numpy()

        self.episode_return += irl_reward
        if done:
            info['episode']["ep_rew_irl"] = self.episode_return
            self.episode_return = 0

        self.obs = next_obs

        return next_obs, irl_reward, term, trunc, info