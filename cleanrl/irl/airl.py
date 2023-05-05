import gym

import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import Adam
from torch.nn.utils import spectral_norm
from irl.utils import ResNetAIRLDisc, MiniGridCNN

class AIRLDiscriminator(nn.Module):
    def __init__(self, env, opt):
        super(AIRLDiscriminator, self).__init__()

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
        self.gamma = opt.airl_gamma

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
                                      torch.nn.PReLU())

        if opt.airl_resnet:
            self.reward = ResNetAIRLDisc(layer_dims[1])
        else:
            self.reward_layers = []
            for i in range(2, len(self.layer_dims)):
                self.reward_layers += [f(torch.nn.Linear(in_features=self.layer_dims[i - 1],
                                                    out_features=self.layer_dims[i],
                                                    bias=self.bias)),
                                                    torch.nn.PReLU()]

            self.reward_layers += [f(torch.nn.Linear(in_features=self.layer_dims[-1],
                                                out_features=1,
                                                bias=self.bias))]
            self.reward = nn.Sequential(*self.reward_layers)

        # shaping function h_\phi
        if self.use_cnn_base:
            self.base_v = MiniGridCNN(layer_dims, use_actions=False)
        else:
            self.base_v = nn.Sequential(f(torch.nn.Linear(ob_shapes[-1],
                                                        self.layer_dims[1], self.bias)),
                                        torch.nn.PReLU())

        if opt.airl_resnet:
            self.value = ResNetAIRLDisc(layer_dims[1])
        else:
            self.value_layers = []
            for i in range(2, len(self.layer_dims)):
                self.value_layers += [f(torch.nn.Linear(in_features=self.layer_dims[i - 1],
                                                    out_features=self.layer_dims[i],
                                                    bias=self.bias)),
                                    torch.nn.PReLU()]

            self.value_layers += [f(torch.nn.Linear(in_features=self.layer_dims[-1],
                                                out_features=1,
                                                bias=self.bias))]
            self.value = nn.Sequential(*self.value_layers)

            if torch.cuda.is_available():
                self.base.cuda()
                self.base_v.cuda()
                self.reward.cuda()
                self.value.cuda()

        # self.module_list = nn.ModuleList([self.base, self.base_v, self.reward, self.value])

        self.d_optimizer = Adam(list(self.base.parameters()) + list(self.base_v.parameters()) +
                                list(self.reward.parameters()) + list(self.value.parameters()), lr=self.lr, 
                                weight_decay=self.l2_coeff)

    def forward(self, ob, next_ob, ac, lprobs):
        # forward the nn models
        fitted_value_n = self.value(self.base_v(next_ob))
        fitted_value = self.value(self.base_v(ob))
        reward = self.get_reward(ob, ac)

        # calculate discriminator probability according to AIRL structure
        qfn = reward + self.gamma * fitted_value_n
        log_p_tau = torch.squeeze(qfn - fitted_value)
        # log probabilities of expert actions under policy
        log_q_tau = lprobs

        # log_pq = torch.log(torch.sum(torch.exp(torch.cat([log_p_tau, log_q_tau], dim=0))))
        # d_out = torch.exp(log_p_tau - log_pq)
        # TEST:
        d_out = torch.sigmoid(log_p_tau - log_q_tau)
        # d_out = torch.sigmoid(log_q_tau - log_p_tau)

        return reward, fitted_value, fitted_value_n, d_out

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
        return self.reward(base_out)

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

        if self.use_actions:
            action_epsilon = torch.rand(update_dict['policy_acs'].shape)
            interp_acs = action_epsilon * update_dict['policy_acs'] + (1 - action_epsilon) * update_dict['expert_acs']
            interp_acs.requires_grad = True
            encoder_input = [interp_obs, interp_acs, interp_obs_next]
            _, _, _, estimate = self.forward(interp_obs,
                                         interp_obs_next, interp_acs, update_dict['policy_lprobs'])
        else:            
            encoder_input = [interp_obs, interp_obs_next]
            _, _, _, estimate = self.forward(interp_obs,
                                         interp_obs_next, ac=None, lprobs=update_dict['policy_lprobs'])

        gradient = torch.autograd.grad(estimate.sum(), encoder_input, create_graph=True)[0]
        # Norm's gradient could be NaN at 0. Use our own safe_norm
        safe_norm = (torch.sum(gradient ** 2, dim=1) + 1e-8).sqrt()
        gradient_mag = torch.mean((safe_norm - 1) ** 2)

        return gradient_mag

    def compute_loss(self, update_dict):
        # Define log p(tau) = r(s,a) + gamma * V(s') - V(s)
        r_p, v_p, _, policy_estimate = self.forward(update_dict['policy_obs'],
                                                update_dict['policy_obs_next'], update_dict['policy_acs'],
                                                update_dict['policy_lprobs'])
        r_e, v_e, _, expert_estimate = self.forward(update_dict['expert_obs'],
                                                update_dict['expert_obs_next'], update_dict['expert_acs'],
                                                update_dict['expert_lprobs'])

        # label convention: experts 0, policy 1
        labels = torch.cat([torch.zeros(expert_estimate.size()),
                            torch.ones(policy_estimate.size())])
        d_out = torch.cat([expert_estimate, policy_estimate], dim=0)
        if torch.cuda.is_available():
            labels = labels.cuda()

        discriminator_loss = -(
                torch.log(expert_estimate + 1e-6)
                + torch.log(1.0 - policy_estimate + 1e-6)
        ).mean()

        # loss_pi = -F.logsigmoid(-policy_estimate).mean()
        # loss_exp = -F.logsigmoid(expert_estimate).mean()
        # loss_disc = loss_pi + loss_exp

        # TODO: only compute w.r.t. expert_estimate!
        # what are labels -> need to be expert ids!!
        # -> Jensen Shannon? Mixture of experts?
        # grad_penalty = self.irm_penalty(d_out, labels)
        lip_penalty = self.lip_penalty(update_dict)
        grad_penalty_ = self.irm_penalty(policy_estimate, torch.zeros(expert_estimate.size()))
        grad_penalty__ = self.irm_penalty(expert_estimate, torch.ones(expert_estimate.size()))
        grad_penalty = self.irm_penalty(d_out, labels)

        # lip_penalty = self.lip_penalty(update_dict)
        loss = discriminator_loss + self.irm_coeff * grad_penalty + self.lip_coeff * lip_penalty
        if self.irm_coeff > 1.0:
            loss /= self.irm_coeff

        output_dict = {}
        output_dict['total_loss'] = loss
        output_dict['d_loss'] = discriminator_loss
        output_dict['policy_estimate'] = policy_estimate
        output_dict['expert_estimate'] = expert_estimate
        output_dict['policy_reward'] = torch.mean(r_p)
        output_dict['expert_reward'] = torch.mean(r_e)
        output_dict['policy_value'] = torch.mean(v_p)
        output_dict['expert_value'] = torch.mean(v_e)
        output_dict['grad_penalty'] = grad_penalty
        output_dict['lip_penalty'] = lip_penalty

        # return self.loss, discriminator_loss, policy_estimate.mean(), expert_estimate.mean(), self.grad_penalty
        return output_dict

    def update(self, loss):
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()

    def save(self, fname):
        torch.save(self.state_dict(), fname)

class AirlReward(gym.Wrapper):
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
