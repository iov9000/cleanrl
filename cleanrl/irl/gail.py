import gym

import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import Adam
from torch.nn.utils import spectral_norm
from irl.utils import ResNetAIRLDisc, MiniGridCNN, AtariCNNBase

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
                self.base = nn.Sequential(torch.nn.Linear(self.layer_dims[0],
                                                        self.layer_dims[1], self.bias),
                                        nonlin)

            self.discriminator_layers = []
            for i in range(2, len(layer_dims)):
                self.discriminator_layers += [torch.nn.Linear(in_features=layer_dims[i - 1],
                                                            out_features=layer_dims[i],
                                                            bias=self.bias),
                                            nonlin]

            self.discriminator_layers += [torch.nn.Linear(in_features=layer_dims[-1],
                                                        out_features=1,
                                                        bias=self.bias)]

            self.discriminator = nn.Sequential(*self.discriminator_layers)

        if torch.cuda.is_available():
            self.base.cuda()
            self.discriminator.cuda()

        # self.module_list = nn.ModuleList([self.base, self.discriminator])
        base_params = list(self.base.parameters())
        d_params = list(self.discriminator.parameters())
        base_params.extend(d_params)
        #self.d_optimizer = Adam(base_params, lr=self.lr, weight_decay=self.l2_coeff)
        self.d_optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coeff)

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

        if self.opt.div == 'fkl':
            d_out_ = torch.exp(d_out) # (N*T,) p/q TODO: clip
        elif self.opt.div == 'rkl':
            d_out_ = d_out # (N*T,) log (p/q)
        elif self.opt.div == 'js':  # https://pytorch.org/docs/master/generated/torch.nn.Softplus.html
            d_out_ = F.softplus(d_out) # (N*T,) log (1 + p/q)

        return d_out_

    def get_reward(self, ob, ac):
        base_out = self.base_forward(ob, ac)

        d_out = self.discriminator(base_out)
        self.reward = - torch.squeeze(torch.log(torch.sigmoid(d_out) + 1e-8))
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

    def lip_penalty(self, update_dict):
        obs_epsilon = torch.rand(update_dict['policy_obs'].shape)
        interp_obs = obs_epsilon * update_dict['policy_obs'] + (1 - obs_epsilon) * update_dict['expert_obs']
        interp_obs.requires_grad = True  # For gradient calculation

        if self.use_actions:
            action_epsilon = torch.rand(update_dict['policy_acs'].shape)
            interp_acs = action_epsilon * update_dict['policy_acs'] + (1 - action_epsilon) * update_dict['expert_acs']
            interp_acs.requires_grad = True
            encoder_input = [interp_obs, interp_acs]
            estimate = self.forward(interp_obs, interp_acs)
        else:            
            encoder_input = [interp_obs]
            estimate = self.forward(interp_obs, ac=None)

        gradient = torch.autograd.grad(estimate.sum(), encoder_input, create_graph=True)[0]
        # Norm's gradient could be NaN at 0. Use our own safe_norm
        safe_norm = (torch.sum(gradient ** 2, dim=1) + 1e-8).sqrt()
        gradient_mag = torch.mean((safe_norm - 1) ** 2)

        return gradient_mag

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
        d_out = self.forward(update_dict['all_obs'], update_dict['all_acs'])
        expert_out, policy_out = torch.chunk(d_out, chunks=2, dim=0)

        expert_loss = F.binary_cross_entropy_with_logits(
            expert_out,
            torch.ones(expert_out.size()))
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_out,
            torch.zeros(policy_out.size()))

        labels = torch.cat([torch.zeros(expert_out.size()),
                            torch.ones(policy_out.size())])

        labels_flipped = torch.cat([torch.ones(expert_out.size()),
                            torch.zeros(policy_out.size())])

        self.bce_loss = F.binary_cross_entropy_with_logits(d_out, labels)
        # self.grad_penalty = self.irm_penalty(d_out, labels)
        lip_penalty = self.lip_penalty(update_dict)
        ## TODO: check IRM penalty form! -> ones not zeros!
        if self.opt.irm_loss_type == 'e1':
            self.grad_penalty = self.irm_penalty(
                expert_out, torch.ones(expert_out.size()))
        elif self.opt.irm_loss_type == 'e0':
            self.grad_penalty = self.irm_penalty(
                expert_out, torch.zeros(expert_out.size()))
        if self.opt.irm_loss_type == 'p1':
            self.grad_penalty = self.irm_penalty(
                policy_out, torch.zeros(policy_out.size()))
        elif self.opt.irm_loss_type == 'p0':
            self.grad_penalty = self.irm_penalty(
                policy_out, torch.zeros(policy_out.size()))
        elif self.opt.irm_loss_type == 'e1p0_sep':
            self.grad_penalty = self.irm_penalty(policy_out, torch.zeros(policy_out.size())) +\
                self.irm_penalty(expert_out, torch.ones(expert_out.size()))    
        elif self.opt.irm_loss_type == 'e1p0':
            self.grad_penalty = self.irm_penalty(d_out, labels_flipped)
        elif self.opt.irm_loss_type == 'e1p0sym':
            self.grad_penalty = self.irm_penalty(d_out, labels_flipped) + self.irm_penalty(d_out, labels)
        else:
            self.grad_penalty = self.irm_penalty(d_out, labels)


        self.loss = self.bce_loss + self.irm_coeff * self.grad_penalty + self.lip_coeff * lip_penalty

        output_dict = {}
        output_dict['d_loss'] = self.bce_loss
        output_dict['lip_penalty'] = lip_penalty
        output_dict['grad_penalty'] = self.grad_penalty

        return output_dict
        
    def update(self, loss):
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()

class GailReward(gym.Wrapper):
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