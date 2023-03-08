import os
import re
import sys
import pickle as pkl

import pandas as pd
import numpy as np


if __name__ == "__main__":
  # data = pkl.load(open(sys.argv[1],'rb'))
  # print(data)
  
  def parse_fname(fname):    
    intgr = re.findall(r'\d+', fname)
    floats = re.findall(r"\d+\.\d+", fname)
    name = re.findall(r"[a-zA-Z0-9]*-[-a-zA-Z0-9]", fname)
    seed = intgr[-3]
    n_epc = intgr[-1]
    n_epi = intgr[-2]

    if 'airl' in fname:
      irl_algo = 'airl'
    if 'gail' in fname:
      irl_algo = 'gail'
    if 'firl' in fname:
      irl_algo = 'firl'
    
    if len(floats) > 0:
      irm_coeff = floats[0]
    else:
      irm_coeff = 0
      
    return int(seed), float(irm_coeff), name[0]+'3', irl_algo

  sac_folder = os.path.join(sys.argv[1], 'sac')
  ppo_folder = os.path.join(sys.argv[1], 'ppo')

  df_dict_ppo = {}
  df_dict_ppo['airl'] = {'Ant-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}},
                    'HalfCheetah-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}},
                    'Hopper-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}},
                    'Walker2d-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}}}

  df_dict_ppo['gail'] = {'Ant-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}},
                    'HalfCheetah-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}},
                    'Hopper-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}},
                    'Walker2d-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}}}

  df_dict_ppo['firl'] = {'Ant-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}},
                    'HalfCheetah-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}},
                    'Hopper-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}},
                    'Walker2d-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}}}

  df_dict_sac = {}
  df_dict_sac['airl'] = {'Ant-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}},
                    'HalfCheetah-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}},
                    'Hopper-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}},
                    'Walker2d-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}}}

  df_dict_sac['gail'] = {'Ant-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}},
                    'HalfCheetah-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}},
                    'Hopper-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}},
                    'Walker2d-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}}}

  df_dict_sac['firl'] = {'Ant-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}},
                    'HalfCheetah-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}},
                    'Hopper-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}},
                    'Walker2d-v3': {0.0:{},0.1:{},1.0:{},10.0:{},100.0:{}}}

  for d in os.listdir(sac_folder):
    seed, irm, name, irl_algo = parse_fname(d) 
    df = pd.read_pickle(os.path.join(sac_folder, d))
    df_dict_sac[irl_algo][name][irm][seed] = max(df['Value'])

  for d in os.listdir(ppo_folder):
    seed, irm, name, irl_algo = parse_fname(d) 
    df = pd.read_pickle(os.path.join(ppo_folder, d))
    df_dict_ppo[irl_algo][name][irm][seed] = max(df['Value'])

  for d in os.listdir(sac_folder):
    seed, irm, name, irl_algo = parse_fname(d) 
    df_dict_sac[irl_algo][name][irm]['mean'] = np.mean(list(df_dict_sac[irl_algo][name][irm].values()))
    df_dict_sac[irl_algo][name][irm]['std'] = np.std(list(df_dict_sac[irl_algo][name][irm].values()))

  for d in os.listdir(ppo_folder):
    seed, irm, name, irl_algo = parse_fname(d) 
    df_dict_ppo[irl_algo][name][irm]['mean'] = np.mean(list(df_dict_ppo[irl_algo][name][irm].values()))
    df_dict_ppo[irl_algo][name][irm]['std'] = np.std(list(df_dict_ppo[irl_algo][name][irm].values()))


  def result_dict(d_all):
    df = pd.DataFrame.from_dict({(i,j, k): d_all[i][j][k] for i in d_all.keys() 
                                                 for j in d_all[i].keys()
                                                 for k in d_all[i][j].keys()},
                                                 orient='index')

    return df

  df_sac = result_dict(df_dict_sac)
  df_ppo = result_dict(df_dict_ppo)

  print(df_sac)
  print(df_ppo)





