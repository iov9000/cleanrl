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


class SWILDiscriminator(nn.Module):
    def __init__(self, env, opt, use_cnn_base=False, bias=False):
        super(SWILDiscriminator, self).__init__()

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
        self.n_proj = opt.n_proj
        self.l2_coeff = opt.l2_coeff
        self.reward_type = opt.swil_reward_type
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

        self.use_cnn_base = use_cnn_base

        if use_cnn_base:
            self.base = MiniGridCNN(self.layer_dims, self.use_actions)
        elif self.is_atari:
            self.base = AtariCNNBase(opt, env, self.use_actions)
        else:
            self.base = nn.Sequential(torch.nn.Linear(self.layer_dims[0],
                                                      self.layer_dims[1], bias),
                                      nonlin)

        if self.is_atari:
            self.reward = nn.Linear(512,self.n_proj)
        else:
            if self.reward_type == 'linear':
                self.reward_layers = []
                for i in range(2, len(self.layer_dims)):
                    self.reward_layers += [torch.nn.Linear(in_features=self.layer_dims[i - 1],
                                                        out_features=self.layer_dims[i],
                                                        bias=bias), nonlin]

                self.reward_layers += [torch.nn.Linear(in_features=self.layer_dims[-1],
                                                    out_features=self.n_proj,
                                                    bias=bias)]
                self.reward = nn.Sequential(*self.reward_layers)
            else:
                self.reward = nn.MultiheadAttention(embed_dim=self.layer_dims[1], num_heads = 1)

        if self.opt.proj_layer:
            self.proj_layer = nn.Sequential(nn.Linear(in_features=self.n_proj,
                                                    out_features=1,
                                                    bias=bias), torch.nn.Tanh())

        if self.opt.swil_ae:
            self.ae_layers = []
            for i in reversed(range(2, len(self.layer_dims))):
                self.ae_layers += [torch.nn.Linear(in_features=self.layer_dims[i - 1],
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

        self.policy_obs = torch.randn([opt.batch_size, *ob_shapes], requires_grad=True)
        self.policy_acs = torch.randn([opt.batch_size, *ac_shapes], requires_grad=True)
        self.expert_obs = torch.randn([opt.batch_size, *ob_shapes])
        self.expert_acs = torch.randn([opt.batch_size, *ac_shapes])
        # self.module_list = nn.ModuleList([self.base, self.base_v, self.reward, self.value])
        #self.d_optimizer = Adam(list(self.base.parameters()) + list(self.reward.parameters()), 
        #                lr=self.lr, weight_decay=self.l2_coeff)
        self.d_optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coeff)
        self.d_scheduler = ExponentialLR(self.d_optimizer, gamma=opt.scheduler_gamma)
        #self.d_optimizer = Adam([self.policy_obs, self.policy_acs], lr=self.lr)

    def forward(self, ob, ac, lprobs):
        # forward the nn models
        # reward = self.get_reward(ob, ac)
        gsw_dist = self.gsw(ob, ac, self.expert_obs, self.expert_acs, random=False)

        # get random projections based on reward
        return gsw_dist

    def proj(self, ob, ac):
        base_out = self.base_fwd(ob,ac)

        # rew, v, v_n, d_out = self.forward(ob, next_ob, ac, lprobs) TODO??
        # potentially use more sophisticated mechanism
        return self.reward(base_out)

    def base_fwd(self,ob,ac):
        if self.use_actions and (self.use_cnn_base or self.is_atari):
            base_out = self.base(ob, ac)
        elif self.use_actions and not (self.use_cnn_base or self.is_atari):
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)

        return base_out

    def get_reward(self, ob, ac):
        base_out = self.base_fwd(ob,ac)

        if self.opt.swil_loss == 'gsw':
            if self.reward_type == 'linear':
                if self.opt.proj_layer:
                    return self.proj_layer(self.reward(base_out))
                else:
                    return torch.mean(self.reward(base_out),-1)
            else:   
                return torch.mean(self.reward(base_out,base_out,base_out)[0],-1)
        elif self.opt.swil_loss == 'approx_sw': 
            return approximate_sw(torch.cat([ob,ac],-1),torch.cat([self.expert_obs, self.expert_acs],-1))
        else:
            # compute distance for single ob,ac and batch of experts via projections
            gsw_dist = self.gsw(ob, ac, self.expert_obs, self.expert_acs, random=False)
            
            # rew, v, v_n, d_out = self.forward(ob, next_ob, ac, lprobs) TODO??
            # potentially use more sophisticated mechanism for averaging out projections
            # Self-Attention sliced wasserstein distances is a weighted sum? 
            return gsw_dist

    def gsw(self, obs_pi, acs_pi, obs_exp, acs_exp, random=False):
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

    def max_gsw(self, obs_pi, acs_pi, obs_exp, acs_exp, iterations=50,lr=1e-4):
        # N,dn = X.shape
        # M,dm = Y.shape
        # assert dn==dm and M==N

        # self.reward.reset()
        
        optimizer = torch.optim.Adam(list(self.base.parameters()) + list(self.reward.parameters()),lr=lr)
        total_loss=np.zeros((iterations,))
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
        exp_obs = update_dict['expert_obs']
        exp_acs = update_dict['expert_acs']

        if self.opt.swil_loss == 'surr_loss':
            # surrogate loss from maxSWGAN paper
            surr_loss = torch.sum(self.proj(self.policy_obs,self.policy_acs)) - torch.sum(self.proj(exp_obs,exp_acs)) 
            d_loss = surr_loss
        elif self.opt.swil_loss == 'approx_sw':
            d_loss = approximate_sw(torch.cat([self.policy_obs, self.policy_acs],-1), 
                                    torch.cat([exp_obs, exp_acs],-1))
        else:
            if self.opt.max_gsw:
                gsw_dist = self.max_gsw(self.policy_obs, self.policy_acs,
                            exp_obs, exp_acs)
            else:
                gsw_dist = self.gsw(self.policy_obs, self.policy_acs,
                            exp_obs, exp_acs, random=False)
            d_loss = gsw_dist


        output_dict = {}
        if self.opt.swil_ae:
            # meaningful reconstruction: action prediction?
            base_out = self.base_fwd(exp_obs,exp_acs)
            recon = self.recon_net(base_out)
            output_dict['recon_loss'] = F.mse_loss(recon, torch.cat([exp_obs,exp_acs],-1))
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