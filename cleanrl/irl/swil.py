import gym
import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from torch.nn.utils import spectral_norm
from irl.utils import ResNetAIRLDisc, MiniGridCNN, AtariCNNBase
from irl.gsw_utils import gsw, get_slice
from tqdm import tqdm

def get_random_projections(latent_dim: int, num_samples: int, proj_dist='normal') -> torch.Tensor:
    """
    Returns random samples from latent distribution's (Gaussian)
    unit sphere for projecting the encoded samples and the
    distribution samples.

    :param latent_dim: (Int) Dimensionality of the latent space (D)
    :param num_samples: (Int) Number of samples required (S)
    :return: Random projections from the latent unit sphere
    """
    if proj_dist == 'normal':
        rand_samples = torch.randn(num_samples, latent_dim)
    elif proj_dist == 'cauchy':
        rand_samples = torch.distributions.Cauchy(torch.tensor([0.0]),
                                   torch.tensor([1.0])).sample((num_samples, latent_dim)).squeeze()
    else:
        raise ValueError('Unknown projection distribution.')

    rand_proj = rand_samples / rand_samples.norm(dim=1).view(-1,1)
    return rand_proj # [S x D]

def get_generalized_random_projections(projection_model, latent_dim: int, num_samples: int, proj_dist: str) -> torch.Tensor:
    """
    Returns random samples from latent distribution's (Gaussian)
    unit sphere for projecting the encoded samples and the
    distribution samples.

    :param latent_dim: (Int) Dimensionality of the latent space (D)
    :param num_samples: (Int) Number of samples required (S)
    :return: Random projections from the latent unit sphere
    """

    # flows?
    rand_samples = projection_model(torch.randn(num_samples, latent_dim))
        
    rand_proj = rand_samples / rand_samples.norm(dim=1).view(-1,1)
    return rand_proj # [S x D]

def approximate_sw(X, Y, centering=True):
    """
    Approximates SW with Wasserstein distance between Gaussian approximations
    """
    if len(X.shape) == 1:
        X = X.unsqueeze(0)
    d = X.shape[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Convert numpy arrays to torch tensors
    X = torch.tensor(X, dtype=torch.float, device=device)
    Y = torch.tensor(Y, dtype=torch.float, device=device)
    if centering:
        # Center the data
        mean_X = torch.mean(X, dim=0)
        mean_Y = torch.mean(Y, dim=0)
        X = X - mean_X
        Y = Y - mean_Y
    # Approximate SW
    m2_Xc = torch.mean(torch.linalg.norm(X, dim=1) ** 2) / d
    m2_Yc = torch.mean(torch.linalg.norm(Y, dim=1) ** 2) / d
    sw = torch.abs(m2_Xc ** (1 / 2) - m2_Yc ** (1 / 2))
    return sw

def compute_swd(z_expert: torch.Tensor,
                z_policy: torch.Tensor,
                p: float,
                reg_weight: float,
                n_proj: int,
                latent_dim: int) -> torch.Tensor:
    """
    Computes the Sliced Wasserstein Distance (SWD) - which consists of
    randomly projecting the encoded and prior vectors and computing
    their Wasserstein distance along those projections.

    :param z: Latent samples # [N  x D]
    :param p: Value for the p^th Wasserstein distance
    :param reg_weight:
    :return:
    """
    prior_z = z_policy
    device = z_expert.device

    proj_matrix = get_random_projections(latent_dim, n_proj).transpose(0,1).to(device)
    #gen_proj_matrix = get_generalized_random_projections(latent_dim, ns_proj).transpose(0,1).to(device)

    latent_projections = z_expert.matmul(proj_matrix) # [N x S]
    prior_projections = prior_z.matmul(proj_matrix) # [N x S]

    # The Wasserstein distance is computed by sorting the two projections
    # across the batches and computing their element-wise l2 distance
    w_dist = torch.sort(latent_projections.t(), dim=1)[0] - \
             torch.sort(prior_projections.t(), dim=1)[0]
    w_dist = w_dist.pow(p)
    return reg_weight * w_dist.mean()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    #torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def linlayer(in_dim, out_dim, bias=True):
    #return layer_init(nn.Linear(in_dim, out_dim, bias=bias))
    return nn.Linear(in_dim, out_dim, bias=bias)

class SWILDiscriminator(nn.Module):
    def __init__(self, env, opt):
        super(SWILDiscriminator, self).__init__()

        self.env = env
        self.opt = opt
        self.gsw_df = opt.gsw_df
        self.layer_dims = opt.d_layer_dims
        self.lr = opt.disc_lr
        self.use_actions = opt.use_actions
        self.use_dones = opt.use_dones
        self.use_next_obs = opt.use_next_obs
        self.irm_coeff = opt.irm_coeff
        self.l2_coeff = opt.l2_coeff
        self.lip_coeff = opt.lip_coeff
        self.use_cnn_base = opt.use_cnn_base
        use_cnn_base = self.use_cnn_base
        self.bias = opt.use_disc_bias
        bias = self.bias
        self.n_proj = opt.n_proj
        self.l2_coeff = opt.l2_coeff
        self.reward_type = opt.swil_reward_type
        self.is_atari = 'atari' in self.opt.exp_name

        if opt.disc_nonlin == 'relu':
            nonlin = nn.ReLU()
        elif opt.disc_nonlin == 'leakyrelu':
            nonlin = nn.LeakyReLU()
        elif opt.disc_nonlin == 'tanh':
            nonlin = nn.Tanh()
        else:
            nonlin = nn.PReLU()

        #if opt.swil_ae:
            # constrain the dimension if using an autoencoder
        #    self.layer_dims[1] = 10

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

        if opt.gsw_df == 'nn':

            if use_cnn_base:
                self.base = MiniGridCNN(self.layer_dims, self.use_actions)
            elif self.is_atari:
                self.base = AtariCNNBase(opt, env, self.use_actions)
            elif opt.linear_proj:
                self.base = linlayer(self.layer_dims[0], self.n_proj, bias)
            else:
                #self.base = nn.Sequential(torch.nn.Linear(self.layer_dims[0],
                #                                          self.layer_dims[1], bias), nonlin)
                self.base = nn.Sequential(linlayer(self.layer_dims[0], self.layer_dims[1], bias), nonlin)

            if self.is_atari:
                self.reward = nn.Linear(512,self.n_proj)
            elif opt.linear_proj:
                self.reward = nn.Identity()
            else:
                if self.reward_type == 'linear':
                    self.reward_layers = []
                    for i in range(2, len(self.layer_dims)):
                        # self.reward_layers += [torch.nn.Linear(in_features=self.layer_dims[i - 1],
                        #                                     out_features=self.layer_dims[i],
                        #                                     bias=bias), nonlin]
                        self.reward_layers += [linlayer(self.layer_dims[i - 1], self.layer_dims[i], bias), nonlin]

                    # self.reward_layers += [torch.nn.Linear(in_features=self.layer_dims[-1],
                    #                                     out_features=self.n_proj,
                    #                                     bias=bias)]
                    self.reward_layers += [linlayer(self.layer_dims[-1], self.n_proj, bias)]
                    self.reward = nn.Sequential(*self.reward_layers)
                else:
                    self.reward = nn.MultiheadAttention(embed_dim=self.layer_dims[1], num_heads = 1)

            if self.opt.proj_layer:
                # self.proj_layer = nn.Sequential(nn.Linear(in_features=self.n_proj,
                #                                         out_features=1,
                #                                         bias=bias), torch.nn.Tanh())
                self.proj_layer = nn.Sequential(linlayer(self.n_proj, 1, bias), torch.nn.Tanh())

            if self.opt.swil_ae:
                self.ae_layers = []
                for i in reversed(range(2, len(self.layer_dims))):
                    self.ae_layers += [torch.nn.Linear(in_features=self.n_proj,
                                                            out_features=self.layer_dims[i],
                                                            bias=bias), nonlin]
                self.ae_layers += [torch.nn.Linear(in_features=self.layer_dims[-1],
                                                        out_features=self.layer_dims[0],
                                                        bias=bias)]
                self.recon_net = nn.Sequential(*self.ae_layers)

            if torch.cuda.is_available():
                self.base.cuda()
                self.reward.cuda()
                if self.opt.proj_layer:
                    self.proj_layer.cuda()
            
            self.d_optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coeff)#, eps=1e-5)
            self.d_scheduler = ExponentialLR(self.d_optimizer, gamma=opt.scheduler_gamma)
        else:
            self.gsw_module = GSW('poly', nofprojections=opt.n_proj, degree=opt.poly_degree)


        self.policy_obs = torch.randn([opt.batch_size, *ob_shapes])#, requires_grad=True)
        self.policy_acs = torch.randn([opt.batch_size, *ac_shapes])#, requires_grad=True)
        self.expert_obs = torch.randn([opt.batch_size, *ob_shapes])
        self.expert_nobs = torch.randn([opt.batch_size, *ob_shapes])
        self.expert_acs = torch.randn([opt.batch_size, *ac_shapes])
        self.expert_nobs = torch.randn([opt.batch_size, *ob_shapes])

        # self.module_list = nn.ModuleList([self.base, self.base_v, self.reward, self.value])
        #self.d_optimizer = Adam(list(self.base.parameters()) + list(self.reward.parameters()), 
        #                lr=self.lr, weight_decay=self.l2_coeff)

        #self.d_optimizer = Adam([self.policy_obs, self.policy_acs], lr=self.lr)

    def forward(self, ob, ac, nob=None, d=None, lprobs=None):
        # forward the nn models
        gsw_dist = self.gsw_dist(ob, ac, nob, d, self.expert_obs, self.expert_acs, random=False)

        # get random projections based on reward
        return gsw_dist

    def proj(self, ob, ac, nob=None, d=None):
        base_out = self.base_fwd(ob,ac)

        # rew, v, v_n, d_out = self.forward(ob, next_ob, ac, lprobs) TODO??
        # potentially use more sophisticated mechanism
        if self.opt.proj_layer:
            return self.proj_layer(self.reward(base_out))
        else:
            # XXX: additional noise?
            return self.reward(base_out)# + torch.randn_like(self.reward(base_out))

    def base_fwd(self, ob, ac, nob=None, d=None):
        if self.use_actions and (self.use_cnn_base or self.is_atari):
            base_out = self.base(ob, ac)
        elif self.use_actions and not (self.use_cnn_base or self.is_atari):
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)

        return base_out

    def weight_reset(self):
        reset_parameters = getattr(self.reward, "reset_parameters", None)
        if callable(reset_parameters):
            self.reward.reset_parameters()

    def get_reward(self, ob, ac, nob=None,d=None):
        if self.opt.gsw_df == 'nn':
            base_out = self.base_fwd(ob,ac,nob,d)
            if self.reward_type == 'linear':
                if self.opt.proj_layer:
                    return self.proj_layer(self.reward(base_out))
                else:
                    return torch.mean(self.reward(base_out),-1)
                    #return torch.min(self.reward(base_out),-1)[0]
            else:   
                return torch.mean(self.reward(base_out,base_out,base_out)[0],-1)
        elif self.opt.swil_loss == 'approx_sw': 
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            return approximate_sw(torch.cat([ob,ac],-1),torch.cat([self.expert_obs, self.expert_acs],-1))
        elif self.opt.swil_loss == 'atom_gsw': 
            return self.atom_gsw(ob, ac, self.expert_obs, self.expert_acs, random=False)
        else:
            # compute distance for single ob,ac and batch of experts via projections
            if self.gsw_df == 'nn':
                gsw_dist = self.gsw_dist_nn(ob, ac, self.expert_obs, self.expert_acs, random=False)
            else:
                gsw_dist = self.gsw_dist(ob, ac, self.expert_obs, self.expert_acs)

            # rew, v, v_n, d_out = self.forward(ob, next_ob, ac, lprobs) TODO??
            # potentially use more sophisticated mechanism for averaging out projections
            # Self-Attention sliced wasserstein distances is a weighted sum? 
            return gsw_dist

    def get_slice(self, ob, ac, theta):
        ''' Slices samples from distribution X~P_X
            Inputs:
                X:  Nxd matrix of N data samples
                theta: parameters of g (e.g., a d vector in the linear case)
        '''

        if self.use_actions:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            X = torch.cat([ob, ac], axis=-1)
        else:
            X = ob

        if self.ftype == 'linear':
            return self.gsw_module.linear(X, theta)
        elif self.ftype == 'poly':
            return self.gsw_module.poly(X, theta)
        elif self.ftype == 'circular':
            return self.gsw_module.circular(X, theta)
        else:
            raise Exception('Defining function not implemented')

    def prepare_x(self, ob, ac):
        if self.use_actions:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            X = torch.cat([ob, ac], axis=-1)
        else:
            X = ob

        return X

    def gsw_dist(self, obs_pi, acs_pi, obs_exp, acs_exp, nobs_pi=None, d_pi=None, nobs_exp=None, d_exp=None):
        X_exp = self.prepare_x(obs_exp, acs_exp)
        X_pi = self.prepare_x(obs_pi, acs_pi)

        return self.gsw_module.gsw(X_exp, X_pi)

    def max_gsw_dist(self, obs_pi, acs_pi, nobs_pi, d_pi, obs_exp, acs_exp, nobs_exp, d_exp):
        X_exp = self.prepare_x(obs_exp, acs_exp)
        X_pi = self.prepare_x(obs_pi, acs_pi)

        return self.gsw_module.max_gsw(X_exp, X_pi)

    def gsw_dist_nn(self, obs_pi, acs_pi, nobs_pi, d_pi, obs_exp, acs_exp, nobs_exp, d_exp, random=False):
        '''
        Calculates GSW between two empirical state-action distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        '''
        # N,dn = X.shape
        # M,dm = Y.shape
        # assert dn==dm and M==N
        
        if random:
            self.reward.reset()
        
        # project slices
        pi_slices = self.proj(obs_pi, acs_pi)
        exp_slices = self.proj(obs_exp, acs_exp)

        # sort slices
        pi_slices_sorted = torch.sort(pi_slices, dim=0)[0]
        exp_slices_sorted = torch.sort(exp_slices, dim=0)[0]
        
        return torch.sqrt(torch.sum((pi_slices_sorted-exp_slices_sorted)**2))

    def atom_gsw(self, obs_pi, acs_pi, obs_exp, acs_exp, random=False):
        '''
        Calculates distance for single atom in 1d by identifying expert neighbours
        '''
        # N,dn = X.shape
        # M,dm = Y.shape
        # assert dn==dm and M==N
        
        if random:
            self.reward.reset()
        
        # project slices 
        pi_slices = self.proj(obs_pi, acs_pi)
        exp_slices = self.proj(obs_exp, acs_exp)
        all_slices = torch.cat([pi_slices.unsqueeze(0),exp_slices],0)

        # sort slices
        slices_sorted_idx = torch.argsort(all_slices, dim=0)
        idx0 = torch.where(slices_sorted_idx==0)[0]
        idx_exp0 = idx0-1
        d0 = torch.sqrt(torch.sum(pi_slices - exp_slices[idx_exp0])**2)

        if torch.all(idx0 > 0):
            idx_exp0 = idx0-1
            d0 = torch.sqrt(torch.sum(pi_slices - exp_slices[idx_exp0])**2)
        else:
            d0 = torch.tensor(0.)
        if torch.all(idx0 < len(exp_slices)-1):
            idx_exp1 = idx0+1
            d1 = torch.sqrt(torch.sum(pi_slices - exp_slices[idx_exp1])**2)
        else:
            d1 = torch.tensor(0.)
        
        return d0 + d1

    # Knothe-Rosenblatt transport -> need a way to design partitioning?
    def kr(self, obs_pi, acs_pi, obs_exp, acs_exp, random=False):
        return 0    

    def max_gsw(self, obs_pi, acs_pi, obs_exp, acs_exp, iterations=10,lr=1e-4):
        # N,dn = X.shape
        # M,dm = Y.shape
        # assert dn==dm and M==N

        #self.weight_reset()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        total_loss = np.zeros((iterations,))
        for i in tqdm(range(iterations)):
            optimizer.zero_grad()
            loss = -self.gsw(obs_pi, acs_pi, obs_exp, acs_exp,random=False)
            total_loss[i] = loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
        
        return self.gsw(obs_pi, acs_pi, obs_exp, acs_exp,random=False)

    def compute_loss(self, update_dict):
        # compute sliced Wasserstein distance here 
        self.policy_obs = update_dict['policy_obs']
        self.policy_acs = update_dict['policy_acs']
        policy_obs_next = update_dict['policy_obs_next']
        policy_dones = update_dict['policy_dones']

        exp_obs = update_dict['expert_obs']
        exp_acs = update_dict['expert_acs']
        exp_obs_next = update_dict['expert_obs_next']
        exp_dones = update_dict['expert_dones']

        if self.opt.swil_loss == 'surr_loss':
            # surrogate loss from maxSWGAN paper
            surr_loss = torch.sum(self.proj(self.policy_obs, self.policy_acs)) - torch.sum(self.proj(exp_obs,exp_acs)) 
            d_loss = surr_loss
        elif self.opt.swil_loss == 'approx_sw':
            obs = self.policy_obs
            acs = self.policy_acs
            if len(obs.shape) != len(acs.shape):
                acs = torch.unsqueeze(acs, -1)
            if len(exp_obs.shape) != len(exp_acs.shape):
                exp_acs = torch.unsqueeze(exp_acs, -1)
            d_loss = approximate_sw(torch.cat([obs, acs],-1), 
                                    torch.cat([exp_obs, exp_acs],-1))
        else:
            if self.opt.max_gsw:
                gsw_dist = self.max_gsw(self.policy_obs, self.policy_acs,
                            exp_obs, exp_acs)
            else:
                gsw_dist = self.gsw(self.policy_obs, self.policy_acs, policy_obs_next, policy_dones,
                            exp_obs, exp_acs, exp_obs_next, exp_dones, random=False)
            d_loss = gsw_dist

        output_dict = {}
        if self.opt.swil_ae:
            # meaningful reconstruction: action prediction?
            base_out = self.base_fwd(exp_obs,exp_acs)
            recon = self.recon_net(self.reward(base_out))
            if self.use_actions:
                if len(exp_acs.shape) == 1:
                    exp_acs = exp_acs.unsqueeze(-1)
                output_dict['recon_loss'] = F.mse_loss(recon, torch.cat([exp_obs,exp_acs],-1))
            else:
                output_dict['recon_loss'] = F.mse_loss(recon, exp_obs)
        else:
            output_dict['recon_loss'] = 0

        output_dict['d_loss'] = d_loss
        output_dict['grad_penalty'] = 0
        output_dict['lip_penalty'] = 0

        # TODO: classification or reconstruction loss for 

        return output_dict

    def update(self, loss):
        self.d_optimizer.zero_grad()
        if self.opt.swil_loss != 'approx_sw':        
            loss.backward()
        self.d_optimizer.step()
        self.d_scheduler.step()


class SwilReward(gym.Wrapper):
    def __init__(self, env, opt, disc):
        super().__init__(env=env)
        # self.reward_fn = make_network(**reward_fn_spec)
        self.use_actions = opt.use_actions
        self.discriminator = disc
        self.obs = None
        # calculate reward at the end of trajecotiry
        self.traj = [] 

    def step(self, action):
        next_obs, gt_reward, done, info = self.env.step(action)
        info['gt_reward'] = gt_reward
        if self.obs is not None:
            obs_t = torch.tensor(self.obs, dtype=torch.get_default_dtype())
        else:
            obs_t = torch.tensor(next_obs, dtype=torch.get_default_dtype())

        acs_t = torch.tensor(action, dtype=torch.get_default_dtype())
        next_obs_t = torch.tensor(next_obs, dtype=torch.get_default_dtype())
        done_t = torch.tensor(done, dtype=torch.get_default_dtype())

        if done:
            self.traj=[]
            
        with torch.no_grad():
            # get discriminator reward and train on that
            #print([p.norm(2) for p in self.discriminator.parameters()])
            irl_reward = self.discriminator.get_reward(
                obs_t, acs_t, next_obs_t, done_t).cpu().numpy()
            #irl_reward = self.discriminator.get_reward(obs_t, acs_t, next_obs_t).cpu().numpy()
        self.obs = next_obs

        return next_obs, irl_reward, done, info


class SwilFlowReward(gym.Wrapper):
    def __init__(self, env, opt, disc):
        super().__init__(env=env)
        # self.reward_fn = make_network(**reward_fn_spec)
        self.use_actions = opt.use_actions
        self.discriminator = disc
        self.obs = None
        self.n_proj = opt.n_proj
        self.sw_poly_deg = opt.sw_poly_deg
        self.radon_df_type = opt.radon_df_type
        # calculate reward at the end of trajecotiry
        self.traj = []

    def compute_sw_grad(self, rho_pi, rho_e):
        #loss = ot.sliced_wasserstein_distance(x1_torch, x2_torch, n_projections=100, seed=gen)
        rho_pi.requires_grad_(True)
        loss = gsw(rho_pi, rho_e, ftype=self.radon_df_type,
                   n_proj=self.n_proj, degree=self.sw_poly_deg)
        loss.backward()

        return rho_pi.grad

    def step(self, action):
        next_obs, gt_reward, done, info = self.env.step(action)
        info['gt_reward'] = gt_reward
        if self.obs is not None:
            obs_t = torch.tensor(self.obs, dtype=torch.get_default_dtype())
        else:
            obs_t = torch.tensor(next_obs, dtype=torch.get_default_dtype())

        acs_t = torch.tensor(action, dtype=torch.get_default_dtype())
        next_obs_t = torch.tensor(next_obs, dtype=torch.get_default_dtype())
        done_t = torch.tensor(done, dtype=torch.get_default_dtype())

        if done:
            self.traj = []

        with torch.no_grad():
            # get discriminator reward and train on that
            #print([p.norm(2) for p in self.discriminator.parameters()])
            d_obs = next_obs_t-obs_t
            #sw_grad_obs = 
            irl_reward = self.discriminator.get_reward(
                obs_t, acs_t, next_obs_t, done_t).cpu().numpy()
            #irl_reward = self.discriminator.get_reward(obs_t, acs_t, next_obs_t).cpu().numpy()
        self.obs = next_obs

        return next_obs, irl_reward, done, info

"""
SWIL reward class based on difference of increasing arrangement maps
"""
class SwilIRDiffReward(gym.Wrapper):
    def __init__(self, env, opt):
        super().__init__(env=env)
        # self.reward_fn = make_network(**reward_fn_spec)
        self.use_actions = opt.use_actions
        self.obs = None
        self.opt = opt
        self.n_proj = opt.n_proj
        self.sw_poly_deg = opt.sw_poly_deg
        self.radon_df_type = opt.radon_df_type
        # TODO: makes GSW module shared across vec envs?
        self.gsw = GSW(ftype=self.radon_df_type, nofprojections=self.n_proj, degree=self.sw_poly_deg)
        # calculate reward at the end of trajecotiry
        self.traj = []

    def compute_gsw(self, policy_atoms, expert_atoms):
        dist, theta, sorted_proj_pi, sorted_proj_exp = self.gsw.gsw(policy_atoms, expert_atoms, ftype=self.radon_df_type,
                                          n_proj=self.n_proj, degree=self.sw_poly_deg)
        self.sorted_proj_pi = sorted_proj_pi
        self.sorted_proj_exp = sorted_proj_exp
        self.theta = theta

        return dist

    def step(self, action):
        next_obs, gt_reward, done, info = self.env.step(action)
        info['gt_reward'] = gt_reward
        if self.obs is not None:
            obs_t = torch.tensor(self.obs, dtype=torch.get_default_dtype())
        else:
            obs_t = torch.tensor(next_obs, dtype=torch.get_default_dtype())

        acs_t = torch.tensor(action, dtype=torch.get_default_dtype())
        next_obs_t = torch.tensor(next_obs, dtype=torch.get_default_dtype())
        done_t = torch.tensor(done, dtype=torch.get_default_dtype())

        if done:
            self.traj = []

        rew = torch.tensor([0.])
        if self.gsw.theta is not None:
            with torch.no_grad():
                # project next state and rank it as part of previous evaluation
                if self.opt.measurable_space == 'sxs':
                    obs_nextobs = torch.cat([obs_t, next_obs_t],-1)
                    obs_t_slice = self.gsw.get_slice(
                        torch.unsqueeze(obs_nextobs,0), self.gsw.theta)
                else:
                    obs_t_slice = self.gsw.get_slice(
                        torch.unsqueeze(next_obs_t,0), self.gsw.theta)

                sorted_proj = self.gsw.Xslices_sorted
                sorted_proj_tgt = self.gsw.Yslices_sorted
                n = len(sorted_proj)

                idx = torch.searchsorted(sorted_proj.T, obs_t_slice.T)
                # shift extreme indices
                idx[idx==0] +=1
                idx[idx==n] -=1
                w = 1/(n*(n+1))
                diff = sorted_proj[1:]-sorted_proj[:-1]

                # TODO: what if target CDF is left or mixed?
                for j,i in enumerate(idx):
                    if self.opt.swil_rew == 'insertion_loss':
                        #w = 1
                        # new atom contribution
                        rew += w*torch.sum(sorted_proj[i,j]-obs_t_slice[0,j] +
                                        sorted_proj[i-1,j]-obs_t_slice[0,j])
                        # reweighting of integrals
                        rew -= torch.sum(w*diff[:i-1,j])
                        rew += torch.sum(w*diff[i:,j])
                    else:
                        # calculate diff when replacing atom
                        a_prev = (sorted_proj_tgt[i,j] - sorted_proj[i,j])**2
                        a_new = (sorted_proj_tgt[i,j] - obs_t_slice[0, j])**2

                        rew += w*(a_new - a_prev)

                        # replace atom in original array with new slice
                        #if (a_new - a_prev) < 0:
                        #    self.gsw.Xslices_sorted[i, j] = obs_t_slice[0, j]
                    #else:
                        # replace atom with obs_t_slice and calculate integral difference
                        
                
                    #rew += torch.sum((sorted_proj[idx]-sorted_proj_tgt[idx])**2)

                rew = -rew
            #irl_reward = self.discriminator.get_reward(obs_t, acs_t, next_obs_t).cpu().numpy()
        self.obs = next_obs

        return next_obs, float(rew.detach().numpy()), done, info

class GSW():
    def __init__(self, ftype='poly', nofprojections=10, 
                    degree=2, radius=2., use_cuda=False):

        self.ftype=ftype
        self.nofprojections=nofprojections
        self.degree=degree
        self.radius=radius
        if torch.cuda.is_available() and use_cuda:
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')
        self.theta = None # Thi is for max-GSW
        self.Y = None

    def update_theta(self):            
        self.theta = self.random_slice(self.theta_shape)

    def gsw(self, X, Y, new_theta=False):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        '''
        #print(X.shape)
        #N,dn = X.shape
        #M,dm = Y.shape
        #assert dn==dm and M==N
        self.Y = Y
        self.theta_shape = X.shape[-1]
        if self.ftype=='index':
            self.theta = torch.randint(0, X.shape[-1], (self.nofprojections,))
        else:
            if self.theta is None or new_theta:
                self.theta = self.random_slice(X.shape[-1])

        Xslices=self.get_slice(X,self.theta)
        Yslices=self.get_slice(Y,self.theta)

        self.Xslices_sorted=torch.sort(Xslices,dim=0)[0]
        self.Yslices_sorted=torch.sort(Yslices,dim=0)[0]

        return torch.sqrt(torch.sum((self.Xslices_sorted-self.Yslices_sorted)**2))

    def max_gsw(self,X,Y,iterations=50,lr=1e-4):
        #N, dn = X.shape
        #M, dm = Y.shape
        dn = X.shape[-1]
        device = self.device
        #assert dn==dm and M==N
        if self.ftype == 'linear':
            theta=torch.randn((1,dn),device=device,requires_grad=True)
            theta.data/=torch.sqrt(torch.sum((theta.data)**2))
        elif self.ftype == 'poly':
            dpoly=self.homopoly(dn,self.degree)
            theta=torch.randn((1,dpoly),device=device,requires_grad=True)
            theta.data/=torch.sqrt(torch.sum((theta.data)**2))
        elif self.ftype == 'circular':
            theta=torch.randn((1,dn),device=device,requires_grad=True)
            theta.data/=torch.sqrt(torch.sum((theta.data)**2))
            theta.data*=self.radius
        self.theta=theta

        optimizer=torch.optim.Adam([self.theta],lr=lr)
        total_loss=np.zeros((iterations,))
        for i in range(iterations):
            optimizer.zero_grad()
            loss=-self.gsw(X.to(self.device),Y.to(self.device))
            total_loss[i]=loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            self.theta.data/=torch.sqrt(torch.sum(self.theta.data**2))

        return self.gsw(X.to(self.device),Y.to(self.device))

    def gsl2(self,X,Y,theta=None):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        '''
        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N
        if theta is None:
            theta=self.random_slice(dn)

        Xslices=self.get_slice(X,theta)
        Yslices=self.get_slice(Y,theta)

        Yslices_sorted=torch.sort(Yslices,dim=0)

        return torch.sqrt(torch.sum((Xslices-Yslices)**2))

    def get_slice(self,X,theta):
        ''' Slices samples from distribution X~P_X
            Inputs:
                X:  Nxd matrix of N data samples
                theta: parameters of g (e.g., a d vector in the linear case)
        '''
        if theta is None:
            theta=self.random_slice(X.shape[-1])
            self.theta = theta

        if self.ftype=='linear':
            return self.linear(X,theta)
        elif self.ftype=='poly':
            return self.poly(X,theta)
        elif self.ftype=='circular':
            return self.circular(X,theta)
        elif self.ftype=='index':
            return self.random_data_index(X,theta)
        else:
            raise Exception('Defining function not implemented')

    def random_slice(self,dim):
        if self.ftype=='linear':
            theta=torch.randn((self.nofprojections,dim))
            theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
        elif self.ftype=='poly':
            dpoly=self.homopoly(dim,self.degree)
            theta=torch.randn((self.nofprojections,dpoly))
            theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
        elif self.ftype=='circular':
            theta=torch.randn((self.nofprojections,dim))
            theta=torch.stack([self.radius*th/torch.sqrt((th**2).sum()) for th in theta])
        return theta.to(self.device)

    def random_data_index(self,X,theta):
        return X[...,theta]

    def linear(self,X,theta):
        if len(theta.shape)==1:
            return torch.matmul(X,theta)
        else:
            return torch.matmul(X,theta.t())

    def poly(self,X,theta):
        ''' The polynomial defining function for generalized Radon transform
            Inputs
            X:  Nxd matrix of N data samples
            theta: Lxd vector that parameterizes for L projections
            degree: degree of the polynomial
        '''
        if len(X.shape)>1:
            N = X.shape[0]
            d = X.shape[1]
        else:
            N = 1
            d = X.shape[0]
            X = X.unsqueeze(0)
        assert theta.shape[1]==self.homopoly(d,self.degree)
        powers=list(self.get_powers(d,self.degree))
        HX=torch.ones((N,len(powers))).to(self.device)
        for k,power in enumerate(powers):
            for i,p in enumerate(power):
                HX[:,k]*=X[:,i]**p
        if len(theta.shape)==1:
            return torch.matmul(HX,theta)
        else:
            return torch.matmul(HX,theta.t())

    def circular(self,X,theta):
        ''' The circular defining function for generalized Radon transform
            Inputs
            X:  Nxd matrix of N data samples
            theta: Lxd vector that parameterizes for L projections
        '''
        N,d=X.shape
        if len(theta.shape)==1:
            return torch.sqrt(torch.sum((X-theta)**2,dim=1))
        else:
            return torch.stack([torch.sqrt(torch.sum((X-th)**2,dim=1)) for th in theta],1)

    def get_powers(self,dim,degree):
        '''
        This function calculates the powers of a homogeneous polynomial
        e.g.

        list(get_powers(dim=2,degree=3))
        [(0, 3), (1, 2), (2, 1), (3, 0)]

        list(get_powers(dim=3,degree=2))
        [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
        '''
        if dim == 1:
            yield (degree,)
        else:
            for value in range(degree + 1):
                for permutation in self.get_powers(dim - 1,degree - value):
                    yield (value,) + permutation

    def homopoly(self,dim,degree):
        '''
        calculates the number of elements in a homogeneous polynomial
        '''
        return len(list(self.get_powers(dim,degree)))