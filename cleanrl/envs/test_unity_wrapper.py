import numpy as np
import random
from mlagents_envs.environment import UnityEnvironment
# from gym_unity.envs import UnityToGymWrapper
from env_fast import UnityToGymWrapper

SEED = 123

random.seed(SEED)
np.random.seed(SEED)

exec_path = "fast/fast_ic_ca.app"
unity_env = UnityEnvironment(file_name=exec_path, no_graphics=True, seed=SEED)
env = UnityToGymWrapper(unity_env, uint8_visual=False, flatten_branched=False, allow_multiple_obs=False, action_space_seed=SEED)

for i in range(1000):
    action = env.action_space.sample()
    o,r,t,tr,i = env.step(action)
    print(np.sum(o))
