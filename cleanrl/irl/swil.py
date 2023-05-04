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
            self.base = nn.Sequential(torch.nn.Linear(self.layer_dims[0],
                                                      self.layer_dims[1], bias),
                                      torch.nn.PReLU())

        self.reward_layers = []
        for i in range(2, len(self.layer_dims)):
            self.reward_layers += [torch.nn.Linear(in_features=self.layer_dims[i - 1],
                                                   out_features=self.layer_dims[i],
                                                   bias=bias),
                                   torch.nn.PReLU()]

        self.reward_layers += [torch.nn.Linear(in_features=self.layer_dims[-1],
                                               out_features=self.n_proj,
                                               bias=bias)]
        self.reward = nn.Sequential(*self.reward_layers)

        if torch.cuda.is_available():
            self.base.cuda()
            self.reward.cuda()

        self.policy_obs = torch.randn([opt.batch_size, *ob_shapes], requires_grad=True)
        self.policy_acs = torch.randn([opt.batch_size, *ac_shapes], requires_grad=True)
        self.expert_obs = torch.randn([opt.batch_size, *ob_shapes])
        self.expert_acs = torch.randn([opt.batch_size, *ac_shapes])
        # self.module_list = nn.ModuleList([self.base, self.base_v, self.reward, self.value])
        self.d_optimizer = Adam(list(self.base.parameters()) + list(self.reward.parameters()), 
                        lr=self.lr, weight_decay=self.l2_coeff)
        #self.d_optimizer = Adam([self.policy_obs, self.policy_acs], lr=self.lr)

    def forward(self, ob, ac, lprobs):
        # forward the nn models
        # reward = self.get_reward(ob, ac)
        gsw_dist = self.gsw(ob, ac, self.expert_obs, self.expert_acs, random=False)

        # get random projections based on reward
        return gsw_dist

    def proj(self, ob, ac):
        if self.use_actions and self.use_cnn_base:
            base_out = self.base(ob, ac)
        elif self.use_actions and not self.use_cnn_base:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)

        # rew, v, v_n, d_out = self.forward(ob, next_ob, ac, lprobs) TODO??
        # potentially use more sophisticated mechanism
        return self.reward(base_out)

    def get_reward(self, ob, ac):
        if self.use_actions and self.use_cnn_base:
            base_out = self.base(ob, ac)
        elif self.use_actions and not self.use_cnn_base:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)

        # compute distance for single ob,ac and batch of experts via projections
        #gsw_dist = self.gsw(ob, ac, self.expert_obs, self.expert_acs, random=False)
        
        # rew, v, v_n, d_out = self.forward(ob, next_ob, ac, lprobs) TODO??
        # potentially use more sophisticated mechanism for averaging out projections
        # Self-Attention sliced wasserstein distances is a weighted sum? 
        return torch.mean(self.reward(base_out),-1)
        #return gsw_dist

    def gsw(self, obs_pi, acs_pi, obs_exp, acs_exp, random=True):
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
        pi_slices=self.proj(obs_pi, acs_pi)
        exp_slices=self.proj(obs_exp, acs_exp)

        # sort slices
        pi_slices_sorted = torch.sort(pi_slices, dim=0)[0]
        exp_slices_sorted = torch.sort(exp_slices, dim=0)[0]
        
        return torch.sqrt(torch.sum((pi_slices_sorted-exp_slices_sorted)**2))

    def max_gsw(self,X,Y,iterations=50,lr=1e-4):
        # N,dn = X.shape
        # M,dm = Y.shape
        # assert dn==dm and M==N

        # self.reward.reset()
        
        optimizer = torch.optim.Adam(self.reward.parameters(),lr=lr)
        total_loss=np.zeros((iterations,))
        for i in range(iterations):
            optimizer.zero_grad()
            loss = -self.gsw(X.to(self.device),Y.to(self.device),random=False)
            total_loss[i] = loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
        
        return self.gsw(X.to(self.device),Y.to(self.device),random=False)

    def compute_loss(self, update_dict):
        # compute sliced Wasserstein distance here 
        self.policy_obs = update_dict['policy_obs']
        self.policy_acs = update_dict['policy_acs']
        gsw_dist = self.gsw(self.policy_obs, self.policy_acs,
                            update_dict['expert_obs'], update_dict['expert_acs'], random=False)

        output_dict = {}
        output_dict['d_loss'] = gsw_dist
        output_dict['grad_penalty'] = 0
        output_dict['lip_penalty'] = 0

        return output_dict

    def update(self, loss):
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()


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