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

def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) / torch.exp( log_var ) )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )

def log_Normal_standard(x, average=False, dim=None):
    log_normal = -0.5 * torch.pow( x , 2 )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )

def vampprior_kld_vae(sample_dict, n_pseudo_inputs):
    mu = sample_dict['z1_mu'] 
    logvar = sample_dict['z1_logvar']
    z = sample_dict['z1']
    z_p_mu = sample_dict['z1_p_mu']
    z_p_logvar = sample_dict['z1_p_logvar']

    z_expanded = z.unsqueeze(1)                                
    means = z_p_mu.unsqueeze(0)
    logvars = z_p_logvar.unsqueeze(0)
    log_normal = log_Normal_diag(z_expanded, means, logvars, dim=2) - np.log(n_pseudo_inputs)
    #if args.attention and args.attention_type == 'nn':
    #    log_normal = vdb.attention_weights * log_normal
    log_normal_max, _ = torch.max(log_normal, 1)
    log_p_z = log_normal_max + torch.log(torch.sum(torch.exp(log_normal - log_normal_max.unsqueeze(1)), 1))
    log_q_z = log_Normal_diag(z, mu, logvar, dim=1)
    kld = -(log_p_z - log_q_z)
    kld = kld.mean()
    return kld

def vampprior_kld_twolayervae(sample_dict, n_pseudo_inputs, use_vampprior=True):
    log_p_z1 = log_Normal_diag(sample_dict['z1'], sample_dict['z1_p_mu'], sample_dict['z1_p_logvar'], dim=1)
    log_q_z1 = log_Normal_diag(sample_dict['z1'], sample_dict['z1_mu'], sample_dict['z1_logvar'], dim=1)
    log_q_z2 = log_Normal_diag(sample_dict['z2'], sample_dict['z2_mu'], sample_dict['z2_logvar'], dim=1)
    if use_vampprior:
        z_expanded = sample_dict['z2'].unsqueeze(1)
        means = sample_dict['z2_p_mu'].unsqueeze(0)
        logvars = sample_dict['z2_p_logvar'].unsqueeze(0)
        log_normal = log_Normal_diag(z_expanded, means, logvars, dim=2) - np.log(n_pseudo_inputs)
        log_normal_max, _ = torch.max(log_normal, 1)
        log_p_z2 = log_normal_max + torch.log(torch.sum(torch.exp(log_normal - log_normal_max.unsqueeze(1)), 1))
    else:
        log_p_z2 = log_Normal_standard(sample_dict['z2'], dim=1)
    kld = -(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)
    kld = kld.mean()
    return kld

def gaussian_kld(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1)


class VAILDiscriminator(nn.Module):
    def __init__(self, env, opt,  use_cnn_base=False, vae_type='TwoLayer', 
                demos=None, alpha_beta=1e-4, bias=False):
        super(VAILDiscriminator, self).__init__()
        self.layer_dims = opt.d_layer_dims
        self.use_actions = opt.use_actions
        self.lr = opt.disc_lr
        self.latent_dim = opt.vail_latent_dim
        self.vae_type = vae_type
        self.n_pseudo_inputs = opt.n_pseudo_inputs
        self.pseudo_inputs_grad = opt.pseudo_inputs_grad
        self.i_c = opt.i_c
        self.alpha_beta = alpha_beta
        self.use_vampprior = opt.use_vampprior
        self.use_actions = opt.use_actions
        self.irm_coeff = opt.irm_coeff
        self.lip_coeff = opt.lip_coeff
        self.l2_coeff = opt.l2_coeff

        ob_shapes = list(env.observation_space.shape)
        ac_shapes = list(env.action_space.shape)
        if not ac_shapes:
            ac_shapes = [1]
        if self.use_actions:
            self.layer_dims = [ob_shapes[-1] + ac_shapes[-1]] + self.layer_dims
        else:
            self.layer_dims = [ob_shapes[-1]] + self.layer_dims
        ac_len = ac_shapes[0]

        self.use_cnn_base = use_cnn_base

        if use_cnn_base:
            self.base = MiniGridCNN(self.layer_dims, self.use_actions)
        else:
            self.base = nn.Sequential(linlayer(self.layer_dims[0],
                                                      self.layer_dims[1], bias, opt.use_weight_norm, opt.use_spectral_norm),
                                      torch.nn.PReLU())

        self.num_inputs = self.layer_dims[1]
        self.nonlinear = torch.nn.PReLU()

        self.vp_dict = {}

        # encoder z_2: q(z1|x,z2)
        self.vp_dict['z_2'] = nn.Linear(self.num_inputs, self.latent_dim)
        self.vp_dict['z_2_mu'] = nn.Linear(self.latent_dim, self.latent_dim)
        self.vp_dict['z_2_logvar'] = nn.Linear(self.latent_dim, self.latent_dim)
        # encoder z_1: q(z1|x,z2)
        self.vp_dict['z_1_x']= nn.Linear(self.num_inputs, self.latent_dim)
        self.vp_dict['z_1_z2'] = nn.Linear(self.latent_dim, self.latent_dim)
        self.vp_dict['z_1_joint'] = nn.Linear(2 * self.latent_dim, self.latent_dim)
        self.vp_dict['z_1_mu']= nn.Linear(self.latent_dim, self.latent_dim)
        self.vp_dict['z_1_logvar']= nn.Linear(self.latent_dim, self.latent_dim)
        # decoder z1: p(z1|z2)
        self.vp_dict['p_z1'] = nn.Linear(self.latent_dim, self.latent_dim)
        self.vp_dict['p_z1_mu'] = nn.Linear(self.latent_dim, self.latent_dim)
        self.vp_dict['p_z1_logvar'] = nn.Linear(self.latent_dim, self.latent_dim)
        # decoder x: p(x|z1,z2)
        self.vp_dict['p_x_z1']= nn.Linear(self.latent_dim, self.latent_dim)
        self.vp_dict['p_x_z2'] = nn.Linear(self.latent_dim, self.latent_dim)
        self.vp_dict['p_x_joint'] = nn.Linear(2 * self.latent_dim, self.latent_dim)
        self.vp_dict['final'] = nn.Linear(self.latent_dim, 1)

        if self.use_vampprior:
            self.pseudo_2 = nn.Hardtanh(min_val=0.0, max_val=1.0)
            if demos is not None:
                # make sure that random numbers are distinct and do not repeat themselves
                indices = np.random.choice(demos.shape[0], size=self.n_pseudo_inputs, replace=False)
                # indices = np.random.randint(0, self.demonstrations.shape[0], size=self.args.number_pseudo_inputs)
                self.pseudo_inputs = torch.nn.Parameter(demos[indices], requires_grad=False)
                self.vp_dict['pseudo_1'] = nn.Linear(self.num_inputs, self.num_inputs)
            else:
                self.pseudo_inputs = torch.nn.Parameter(torch.eye(self.n_pseudo_inputs),
                                                    requires_grad=self.pseudo_inputs_grad)
                self.vp_dict['pseudo_1'] = nn.Linear(self.n_pseudo_inputs, self.num_inputs)

        self._beta = torch.tensor(1.0, dtype=torch.float)

        if torch.cuda.is_available():
            self.base.cuda()
            #self.discriminator.cuda()
            for k,v in self.vp_dict.items():
                v.cuda()

        # self.module_list = nn.ModuleList([self.base, self.discriminator])
        base_params = list(self.base.parameters())
        for k,v in self.vp_dict.items():
            base_params.extend(list(v.parameters()))
        self.d_optimizer = Adam(base_params, lr=self.lr, weight_decay=self.l2_coeff)

        self.vp_dict['final'].weight.data.mul_(1.0)
        self.vp_dict['final'].bias.data.mul_(0.0)

    def get_reward(self, ob, ac):
        d_out, _ = self.forward(ob,ac)
        self.reward = - torch.squeeze(torch.log(d_out + 1e-8))
        return self.reward

    def normalize(self, x):
        mean = torch.mean(x)
        std = torch.std(x)
        x_out = (x - mean) / std
        return x_out

    def encoder_z2(self, x):
        h = self.nonlinear(self.vp_dict['z_2'](x))
        return self.vp_dict['z_2_mu'](h), self.vp_dict['z_2_logvar'](h)

    def encoder_z1(self, x, z2):
        h1 = self.nonlinear(self.vp_dict['z_1_x'](x))
        h2 = self.nonlinear(self.vp_dict['z_1_z2'](z2))
        x_z2 = torch.cat((h1, h2), -1)
        h = self.nonlinear(self.vp_dict['z_1_joint'](x_z2))
        return self.vp_dict['z_1_mu'](h), self.vp_dict['z_1_logvar'](h)

    def irm_penalty(self, logits, y):
        scale = torch.tensor(1.).requires_grad_()
        loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def vail_discriminator(self, z1):
        h1 = self.nonlinear(self.vp_dict['p_z1'](z1))
        z1_p_mu = self.vp_dict['p_z1_mu'](h1)
        z1_p_logvar = self.vp_dict['p_z1_logvar'](h1)

        return torch.sigmoid(self.vp_dict['final'](h1)), z1_p_mu, z1_p_logvar

    def vp_discriminator(self, z1, z2):
        h1 = self.nonlinear(self.vp_dict['p_z1'](z1)) # TODO: z1 or z2?
        z1_p_mu = self.vp_dict['p_z1_mu'](h1)
        z1_p_logvar = self.vp_dict['p_z1_logvar'](h1)
        h2 = self.nonlinear(self.vp_dict['p_x_z1'](z1))
        h3 = self.nonlinear(self.vp_dict['p_x_z2'](z2))
        z1_z2 = torch.cat((h2, h3), -1)
        h = self.nonlinear(self.vp_dict['p_x_joint'](z1_z2))
        return torch.sigmoid(self.vp_dict['final'](h)), z1_p_mu, z1_p_logvar

    def forward(self, ob, ac):
        if self.use_actions and self.use_cnn_base:
            base_out = self.base(ob, ac)
        elif self.use_actions and not self.use_cnn_base:
            base_out = self.base(self.normalize(torch.cat([ob, ac], axis=-1)))
        else:
            base_out = self.base(self.normalize(ob))

        #if self.normalize_input:
        #    x = self.normalize(base_out)
        ## XXX: be careful with exponents in reparametrization -> normalize 

        z2_mu, z2_logvar = self.encoder_z2(base_out)
        z2 = self.reparameterize(z2_mu, z2_logvar)
        z1_mu, z1_logvar = self.encoder_z1(base_out, z2)
        z1 = self.reparameterize(z1_mu, z1_logvar)
        if self.use_vampprior:
            z2_mu, z2_logvar = self.encoder_z2(base_out)
            z2 = self.reparameterize(z2_mu, z2_logvar)
            z1_mu, z1_logvar = self.encoder_z1(base_out, z2)
            z1 = self.reparameterize(z1_mu, z1_logvar)
            d_out, z1_p_mu, z1_p_logvar = self.vp_discriminator(z1, z2)
            pseudos_1 = self.vp_dict['pseudo_1'](self.pseudo_inputs)
            pseudos_2 = self.pseudo_2(pseudos_1)
            z2_p_mu, z2_p_logvar = self.encoder_z2(pseudos_2)
        else:
            z2_mu, z2_logvar = self.encoder_z2(base_out)
            z2 = self.reparameterize(z2_mu, z2_logvar)
            d_out, z1_p_mu, z1_p_logvar = self.vail_discriminator(z2)
            z2_p_mu, z2_p_logvar = None, None

        sample_dict = {}
        sample_dict['z1'] = z1
        sample_dict['z1_mu'] = z1_mu
        sample_dict['z1_logvar'] = z1_logvar
        sample_dict['z2'] = z2
        sample_dict['z2_mu'] = z2_mu
        sample_dict['z2_logvar'] = z2_logvar
        sample_dict['z1_p_mu'] = z1_p_mu
        sample_dict['z1_p_logvar'] = z1_p_logvar
        sample_dict['z2_p_mu'] = z2_p_mu
        sample_dict['z2_p_logvar'] = z2_p_logvar

        return d_out, sample_dict

    def redraw_pseudo_inputs_demonstrations(self, demos):
        # redrawing new samples from demonstrations
        indices = np.random.choice(demos.shape[0], size=self.opt.number_pseudo_inputs, replace=False)
        # indices = np.random.randint(0, self.demonstrations.shape[0], size=self.args.number_pseudo_inputs)
        pseudo_inputs = torch.nn.Parameter(demos[indices], requires_grad=False)
        return pseudo_inputs

    def compute_loss(self, update_dict):
        if self.vae_type == 'TwoLayer':
            d_out_policy, policy_sample_dict= self.forward(update_dict['policy_obs'], update_dict['policy_acs'])
            d_out_expert, expert_sample_dict = self.forward(update_dict['expert_obs'], update_dict['expert_acs'])
            l_kld = vampprior_kld_twolayervae(policy_sample_dict, self.n_pseudo_inputs, self.use_vampprior)
            e_kld = vampprior_kld_twolayervae(expert_sample_dict, self.n_pseudo_inputs, self.use_vampprior)
        else:
            d_out_policy, policy_sample_dict= self.forward(update_dict['policy_obs'], update_dict['policy_acs'])
            d_out_expert, expert_sample_dict = self.forward(update_dict['expert_obs'], update_dict['expert_acs'])

            if self.use_vampprior:
                l_kld = vampprior_kld_vae(policy_sample_dict)
                e_kld = vampprior_kld_vae(expert_sample_dict['z1_mu'], expert_sample_dict['z1_logvar'], 
                                                expert_sample_dict['z1'], 
                                                expert_sample_dict['z1_p_mu'], 
                                                expert_sample_dict['z1_p_logvar'])            
            else:
                l_kld = gaussian_kld(policy_sample_dict['z1_mu'], policy_sample_dict['z1_logvar'])
                l_kld = l_kld.mean()
                e_kld = gaussian_kld(expert_sample_dict['z1_mu'], expert_sample_dict['z1_logvar'])
                e_kld = e_kld.mean()
        
        kld = 0.5 * (l_kld + e_kld)
        bottleneck_loss = kld - self.i_c

        labels = torch.cat([torch.zeros(d_out_expert.size()),
                            torch.ones(d_out_policy.size())])
        d_out = torch.cat([d_out_expert, d_out_policy], dim=0)
        if torch.cuda.is_available():
            labels = labels.cuda()

        discriminator_loss = -(
                torch.log(d_out_expert + 1e-6)
                + torch.log(1.0 - d_out_policy + 1e-6)
        ).mean()
        
        with torch.no_grad():
            self._beta = torch.max(torch.tensor(0.0), self._beta + self.alpha_beta * bottleneck_loss)
        vdb_loss = discriminator_loss + self._beta * bottleneck_loss
        
        grad_penalty = self.irm_penalty(d_out, labels)

        output_dict = {}
        output_dict['d_loss'] = vdb_loss
        output_dict['grad_penalty'] = grad_penalty
        output_dict['vib_loss'] = bottleneck_loss.item()
        output_dict['vib_beta'] = self._beta.item()

        # log gradients
        # if self.vae_type == 'VAE':
        #     stats_dict['Policy/Norm_grad_last_layer'] = torch.norm(vdb.fc5.weight.grad).item()
        # else:
        #     stats_dict['Policy/Norm_grad_last_layer'] = torch.norm(vdb.final.weight.grad).item()

        # if self.use_vampprior:
        #     if args.pseudo_inputs_grad:
        #         stats_dict['Policy/Norm_grad_pseudoinputs'] = torch.norm(vdb.pseudo_input.grad).item()
        #     if args.attention and args.attention_type == 'nn':
        #         stats_dict['Policy/Norm_grad_attention_nn_parameter'] = torch.norm(vdb.attention_weights.grad).item()

        #if args.change_pseudo_inputs_dem:
        #    vdb.redraw_pseudo_inputs_demonstrations()

        return output_dict

    def update(self, loss):
        self.d_optimizer.zero_grad()
        #loss.backward(retain_graph=True)
        loss.backward()
        self.d_optimizer.step()

    def save(self, fname):
        torch.save(self.state_dict(), fname)


class VailReward(gym.Wrapper):
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

        return next_obs, irl_reward, term, trunc, info