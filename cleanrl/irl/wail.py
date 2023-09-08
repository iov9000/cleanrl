import gym

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import Adam
from torch.nn.utils import spectral_norm, weight_norm
from irl.utils import ResNetAIRLDisc, MiniGridCNN

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

class WAILDiscriminator(nn.Module):
    def __init__(self, opt, env, bias=False):
        super(WAILDiscriminator, self).__init__()
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
        self.use_actions = opt.use_actions
        self.epsilon = opt.epsilon
        self.is_atari = 'atari' in self.opt.exp_name

        ob_shapes = list(env.observation_space.shape)
        ac_shapes = list(env.action_space.shape)
        if not ac_shapes:
            ac_shapes = [1]
        if self.use_actions:
            self.layer_dims = [ob_shapes[-1] + ac_shapes[-1]] +  self.layer_dims
        else:
             self.layer_dims = [ob_shapes[-1]] +  self.layer_dims
        ac_len = ac_shapes[0]

        if opt.disc_nonlin == 'relu':
            nonlin = nn.ReLU()
        elif opt.disc_nonlin == 'tanh':
            nonlin = nn.Tanh()
        else:
            nonlin = nn.PReLU()

        if self.use_cnn_base:
            self.base = MiniGridCNN(self.layer_dims, self.use_actions)
        else:
            self.base = nn.Sequential(linlayer(self.layer_dims[0],
                                                self.layer_dims[1], 
                                                bias, 
                                                self.opt.use_weight_norm, 
                                                self.opt.use_spectral_norm),
                                      nonlin)

        self.discriminator_layers = []
        for i in range(2, len(self.layer_dims)):
            self.discriminator_layers += [linlayer(self.layer_dims[i-1],
                                                self.layer_dims[i], 
                                                bias, 
                                                self.opt.use_weight_norm, 
                                                self.opt.use_spectral_norm),
                                          nonlin]

        self.discriminator_layers += [linlayer(self.layer_dims[-1],
                                               1,
                                               bias,
                                               self.opt.use_weight_norm,
                                               self.opt.use_spectral_norm),
                                        nonlin]

        print(self.discriminator_layers)

        self.discriminator = nn.Sequential(*self.discriminator_layers)

        if torch.cuda.is_available():
            self.base.cuda()
            self.discriminator.cuda()

        # self.module_list = nn.ModuleList([self.base, self.discriminator])

        base_params = list(self.base.parameters())
        d_params = list(self.discriminator.parameters())
        base_params.extend(d_params)
        self.d_optimizer = Adam(base_params, lr=self.lr, weight_decay=self.l2_coeff)

    def base_forward(self, ob, ac):
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
        self.reward = self.forward(ob, ac)
        return self.reward

    def irm_penalty(self, logits, y):
        scale = torch.tensor(1.).requires_grad_()
        loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)

    def compute_penalty(self, logits, y):
        scale = torch.tensor(1.).requires_grad_()
        loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        g1 = autograd.grad(loss[0::2].mean(), [scale], create_graph=True)[0]
        g2 = autograd.grad(loss[1::2].mean(), [scale], create_graph=True)[0]
        return (g1 * g2).sum()

    # lipschitz penalty
    def lip_penalty(self, update_dict):
        interp_inputs = []
        for policy_input, expert_input in zip(update_dict['policy_obs'], update_dict['expert_obs']):
            obs_epsilon = torch.rand(policy_input.shape)
            interp_input = obs_epsilon * policy_input + \
                (1 - obs_epsilon) * expert_input
            interp_input.requires_grad = True  # For gradient calculation
            interp_inputs.append(interp_input)
        if self.use_actions:
            action_epsilon = torch.rand(update_dict['policy_acs'].shape)

            dones_epsilon = torch.rand(update_dict['policy_dones'].shape)
            action_inputs = torch.cat(
                [
                    action_epsilon * update_dict['policy_acs']
                    + (1 - action_epsilon) * update_dict['expert_acs'],
                    dones_epsilon * update_dict['policy_dones'] +
                    (1 - dones_epsilon) * update_dict['expert_dones'],
                ],
                dim=1,
            )
            action_inputs.requires_grad = True
            hidden, _ = self.encoder(interp_inputs, action_inputs)
            encoder_input = tuple(interp_inputs + [action_inputs])
        else:
            hidden, _ = self.encoder(interp_inputs)
            encoder_input = tuple(interp_inputs)

        estimate = self.forward(hidden).squeeze(1).sum()
        gradient = torch.autograd.grad(
            estimate, encoder_input, create_graph=True)[0]
        # Norm's gradient could be NaN at 0. Use our own safe_norm
        safe_norm = (torch.sum(gradient ** 2, dim=1) + self.EPSILON).sqrt()
        gradient_mag = torch.mean((safe_norm - 1) ** 2)
        return gradient_mag

    def compute_grad_pen(self, update_dict, lambda_=10):
        expert_state = update_dict['expert_obs']
        expert_action = update_dict['expert_acs']
        policy_state = update_dict['policy_obs']
        policy_action = update_dict['policy_acs']

        alpha = torch.rand(expert_state.size(0), 1)
        if self.use_actions:
            expert_data = torch.cat([expert_state, expert_action], dim=1)
            policy_data = torch.cat([policy_state, policy_action], dim=1)
        else:
            expert_data = expert_state
            policy_data = policy_state

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.discriminator(self.base(mixup_data))
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
        expert_out = self.forward(update_dict['expert_obs'], update_dict['expert_acs'])
        policy_out = self.forward(update_dict['policy_obs'], update_dict['policy_acs'])

        lip_penalty = self.compute_grad_pen(update_dict)
        #self.ot_loss = torch.sum(expert_out - policy_out -\
        #    (1/(4*self.epsilon)*(expert_out - policy_out - torch.norm(expert_out - policy_out))))
        self.ot_loss = torch.mean(expert_out) - torch.mean(torch.exp(policy_out)) + 1 
        #self.ot_loss = torch.mean(expert_out) - torch.mean(policy_out) + 1 

        # self.grad_penalty = self.irm_penalty(d_out, labels)
        self.grad_penalty = self.irm_penalty(
            expert_out, torch.zeros(expert_out.size()))
        self.loss = self.ot_loss + self.irm_coeff * self.grad_penalty + self.lip_coeff*lip_penalty

        output_dict = {}
        output_dict['d_loss'] = self.ot_loss
        output_dict['grad_penalty'] = self.grad_penalty
        output_dict['lip_penalty'] = lip_penalty

        return output_dict
        
    def update(self, loss):
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()


class WailReward(gym.Wrapper):
    def __init__(self, env, opt, disc):
        super().__init__(env=env)
        # self.reward_fn = make_network(**reward_fn_spec)
        self.use_actions = opt.use_actions
        self.discriminator = disc
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

        with torch.no_grad():
            # get discriminator reward and train on that
            #print([p.norm(2) for p in self.discriminator.parameters()])
            irl_reward = self.discriminator.get_reward(
                obs_t, acs_t).cpu().numpy()
            #irl_reward = self.discriminator.get_reward(obs_t, acs_t, next_obs_t).cpu().numpy()
        self.obs = next_obs

        return next_obs, irl_reward[0], term, trunc, info