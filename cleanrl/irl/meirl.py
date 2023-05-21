import gym
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import Adam
from torch.nn.utils import spectral_norm
from irl.utils import ResNetAIRLDisc, MiniGridCNN, AtariCNNBase

class MEIRLDiscriminator(nn.Module):
    def __init__(self, env, opt, clamp_magnitude = 10.0):
        super(MEIRLDiscriminator, self).__init__()

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

        ob_shapes = list(env.observation_space.shape)
        ac_shapes = list(env.action_space.shape)
        if not ac_shapes:
            ac_shapes = [1]
        if self.use_actions:
            self.layer_dims = [ob_shapes[-1] + ac_shapes[-1]] + self.layer_dims
        else:
            self.layer_dims = [ob_shapes[-1]] + self.layer_dims
        ac_len = ac_shapes[0]

        self.clamp_magnitude = clamp_magnitude
        layer_dims = self.layer_dims

        if self.use_cnn_base:
            self.base = MiniGridCNN(layer_dims, self.use_actions)
        else:
            self.base = nn.Sequential(torch.nn.Linear(self.layer_dims[0],
                                                      self.layer_dims[1], self.bias),
                                      torch.nn.PReLU())

        self.discriminator_layers = []
        for i in range(2, len(layer_dims)):
            self.discriminator_layers += [torch.nn.Linear(in_features=layer_dims[i - 1],
                                                          out_features=layer_dims[i],
                                                          bias=self.bias),
                                          torch.nn.PReLU()]

        self.phi = nn.Sequential(*self.discriminator_layers)

        self.discriminator_layers += [torch.nn.Linear(in_features=layer_dims[-1],
                                                      out_features=1,
                                                      bias=self.bias)]
        self.discriminator = nn.Sequential(*self.discriminator_layers)

        if torch.cuda.is_available():
            self.base.cuda()
            self.phi.cuda()
            self.discriminator.cuda()

        # self.module_list = nn.ModuleList([self.base, self.discriminator])
        self.d_optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coeff)

    def forward(self, ob, ac):
        if self.use_actions and (self.use_cnn_base or self.is_atari):
            base_out = self.base(ob, ac)
        elif self.use_actions and not (self.use_cnn_base or self.is_atari):
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)

        phi = self.phi(base_out)
        d_out = self.discriminator(phi)
        #output = torch.clamp(d_out, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
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
        self.reward = d_out #torch.clamp(d_out, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        return self.reward

    # lipschitz penalty
    def lip_penalty(self, update_dict):
        interp_inputs = []
        for policy_input, expert_input in zip(update_dict['policy_obs'], update_dict['expert_obs']):
            obs_epsilon = torch.rand(policy_input.shape)
            interp_input = obs_epsilon * policy_input + (1 - obs_epsilon) * expert_input
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
        gradient = torch.autograd.grad(estimate, encoder_input, create_graph=True)[0]
        # Norm's gradient could be NaN at 0. Use our own safe_norm
        safe_norm = (torch.sum(gradient ** 2, dim=1) + self.EPSILON).sqrt()
        gradient_mag = torch.mean((safe_norm - 1) ** 2)
        return gradient_mag

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

        self.diff_loss = r_expert.mean() - r_policy.mean()
        self.grad_penalty = torch.norm(self.phi_expert-self.phi_policy)
        self.loss = self.diff_loss + self.irm_coeff * self.grad_penalty

        output_dict = {}
        output_dict['total_loss'] = self.loss
        output_dict['d_loss'] = self.diff_loss
        output_dict['grad_penalty'] = self.grad_penalty
        output_dict['lip_penalty'] = 0

        return output_dict

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