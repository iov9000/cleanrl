from pathlib import Path
from argparse import Namespace, ArgumentParser
import sys
import re
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib notebook
import numpy as np
import pandas as pd
# import seaborn as sns  
import pickle as pkl
from tqdm import tqdm
from tensorboard.backend.event_processing import event_accumulator

def read_tb_file(tb_file):
    ea = event_accumulator.EventAccumulator(str(tb_file),
                                            size_guidance={ 
                                                event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                                                event_accumulator.IMAGES: 0,
                                                event_accumulator.AUDIO: 0,
                                                event_accumulator.SCALARS: 0,
                                                event_accumulator.HISTOGRAMS: 0,})
    ea.Reload()
    return ea

def create_df_from_ea(ea, tag):
    steps = [x.step for x in ea.Scalars(tag)]
    values = [x.value for x in ea.Scalars(tag)]
    t0 = ea.Scalars(tag)[0].wall_time
    time = [x.wall_time-t0 for x in ea.Scalars(tag)]

    df = pd.DataFrame({"Time": time, "Step": steps, "Value": values})
    return df

def create_mu_std_df_from_ea(ea_dict, tag, filter_best=False, n_best=5):
    min_len = 1e6
    
    df_list = []
    for name, ea in ea_dict.items():
        df = create_df_from_ea(ea, tag)
        if len(df) < min_len:
            min_len = len(df)
        df_list.append(df)

    return create_mu_std_df(df_list, filter_best, n_best)

def create_mu_std_df(df_list, filter_best=False, n_best=5):
    print(df_list)
    val_list = []
    for df in df_list:
        val_list.append(df["Value"])
    val_list = np.array(val_list)
    
    means = np.mean(val_list, axis=-1)
    idx = np.argsort(-means)
    if filter_best:
        vals = val_list[idx[:n_best]]
    else:
        vals = val_list
    x_axis = df_list[0]["Step"][:min_len]
    
    mu = np.mean(vals, axis=0)
    median = np.median(vals, axis=0)
    std = np.std(vals, axis=0)
    q1 = np.quantile(vals, 0.25, axis=0)
    q3 = np.quantile(vals, 0.75, axis=0)
    best = val_list[idx[0]]
    return pd.DataFrame({"Step": x_axis, "mu": mu, "std": std, "best": best, "median": median, "q1": q1, "q3": q3})

def load_dirs(tasks_dir, tags, save_path=None):
    # init dicts
    df_dict = {}
    for tag in tags:
        df_dict[tag] = {}
        
    for task_dir in tqdm(tasks_dir.iterdir()):        
        ea_dict = {}

        if len(list((task_dir).iterdir())) > 0:
            try:
                ea_dict[task_dir] = read_tb_file(task_dir)
            except:
                print(f"WARNING: No data in {run_dir}. Skipped.")
                continue

        if ea_dict:
            for tag in tags:
                df_dict[tag][task_dir.name] = create_mu_std_df(
                    ea_dict, tag, filter_best=True, n_best=5).copy()
    
    if save_path:
        with open(save_path, 'wb') as handle:
            pkl.dump(df_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
    return df_dict

def plot_mu_std(x_axis, mu, q1=None, q3=None, std=None, fig=None, title=None, label='', yuplim=None, 
                ydownlim=None, interpolate=False, max_num_tasks=None):
    if fig is None:
        fig = plt.figure(figsize=[12,10])
        
    if interpolate:
        from scipy.interpolate import make_interp_spline, BSpline
        std = std[::2]
        new_size = std.shape[0]
        
        xnew = np.linspace(x_axis.min(), x_axis.max(), new_size)
        spl = make_interp_spline(x_axis, mu, k=3)  # type: BSpline
        power_smooth = spl(xnew)
        
        x_axis = xnew
        power_smooth = np.clip(power_smooth, np.min(mu), np.max(mu))
        mu = power_smooth

    if std is not None:
        uperror = mu+std
        downerror = mu-std
        
        uperror = np.clip(uperror, np.min(mu), np.max(mu))
        downerror = np.clip(downerror, np.min(mu), np.max(mu))
    else:
        uperror = q3
        downerror = q1    
    
    if yuplim is not None:
        uperror = np.where(uperror > yuplim, yuplim, uperror)
    if ydownlim is not None:
        downerror = np.where(downerror < ydownlim, ydownlim, downerror)
    plt.plot(x_axis, mu, label=label)
    plt.fill_between(x_axis, downerror, uperror, alpha=.1)
    plt.xlabel("Timesteps", fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    
    if title is not None:
        plt.title(title)

#     plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    if max_num_tasks:
        plt.yticks(np.arange(max_num_tasks+1), 
                   ["{0:1.1f}\%".format(100.0 * i/max_num_tasks) for i in np.arange(max_num_tasks+1)])
#     plt.legend(loc="lower right")
#     plt.legend(loc="upper left")
    
    return fig

def create_mean_std_plot(ea_dict, tag,  title, smoothing=3.0, savefig_fmt=None, filter_best=False):
    df = create_mu_std_df(ea_dict, tag)
    fig = plot_mu_std(title, 'forward',df["steps"], df["mu"], df["std"])
    if savefig_fmt is not None:
        plt.savefig(title + '.' + savefig_fmt, format=savefig_fmt, dpi=300)
        
    return fig

def parse_fname(fname):    
    intgr = re.findall(r'\d+', fname)
    floats = re.findall(r"\d+\.\d+", fname)
    name = re.findall(r"[a-zA-Z0-9]*-[-a-zA-Z0-9]", fname)
    seed = intgr[-3]
    n_epc = intgr[-1]
    n_epi = intgr[-2]
    if len(floats) > 0:
        irm_coeff = floats[0]
    else:
        irm_coeff = 0
        
    return int(seed), float(irm_coeff), name

def load_to_dict(p):
    print(p)
    ea = read_tb_file(p)
    ea_dict_airl[d] = ea
    return ea 

if __name__ == '__main__':
    tags = ["charts/episodic_return"]
    run_dir = Path(sys.argv[1])
    
    ea_dict = {}
    for d in run_dir.iterdir():
        print(d)
        ea_dict[d] = read_tb_file(d)

    df_dict = {}
    df_dict['sac'] = {'Ant-v3': {0.0:{},0.01:{},0.1:{},1.0:{},10.0:{},50.0:{},100.0:{}},
                      'HalfCheetah-v3': {0.0:{},0.01:{},0.1:{},1.0:{},10.0:{},50.0:{},100.0:{}},
                      'Hopper-v3': {0.0:{},0.01:{},0.1:{},1.0:{},10.0:{},50.0:{},100.0:{}},
                      'Walker2d-v3': {0.0:{},0.01:{},0.1:{},1.0:{},10.0:{},50.0:{},100.0:{}}}
    df_dict['ppo'] = {'Ant-v3': {0.0:{},0.01:{},0.1:{},1.0:{},10.0:{},50.0:{},100.0:{}},
                      'HalfCheetah-v3': {0.0:{},0.01:{},0.1:{},1.0:{},10.0:{},50.0:{},100.0:{}},
                      'Hopper-v3': {0.0:{},0.01:{},0.1:{},1.0:{},10.0:{},50.0:{},100.0:{}},
                      'Walker2d-v3': {0.0:{},0.01:{},0.1:{},1.0:{},10.0:{},50.0:{},100.0:{}}}

    df_dict_max = {}
    df_dict_flat = {}
    irm_coeffs = [0.0, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]

    df_dict_max['sac'] = {'Ant-v3': {0.0:{},0.01:{},0.1:{},1.0:{},10.0:{},50.0:{},100.0:{}},
                      'HalfCheetah-v3': {0.0:{},0.01:{},0.1:{},1.0:{},10.0:{},50.0:{},100.0:{}},
                      'Hopper-v3': {0.0:{},0.01:{},0.1:{},1.0:{},10.0:{},50.0:{},100.0:{}},
                      'Walker2d-v3': {0.0:{},0.01:{},0.1:{},1.0:{},10.0:{},50.0:{},100.0:{}}}
    df_dict_max['ppo'] = {'Ant-v3': {0.0:{},0.01:{},0.1:{},1.0:{},10.0:{},50.0:{},100.0:{}},
                      'HalfCheetah-v3': {0.0:{},0.01:{},0.1:{},1.0:{},10.0:{},50.0:{},100.0:{}},
                      'Hopper-v3': {0.0:{},0.01:{},0.1:{},1.0:{},10.0:{},50.0:{},100.0:{}},
                      'Walker2d-v3': {0.0:{},0.01:{},0.1:{},1.0:{},10.0:{},50.0:{},100.0:{}}}

    for k,v in ea_dict.items():
        seed, irm, name = parse_fname(str(k))
        name = name[0]+'3'
        df = create_df_from_ea(v, "charts/episodic_return")

        if 'sac' in str(k):
            df_dict['sac'][name][irm][seed] = df 
            df_dict_max['sac'][name][irm][seed] = max(df["Value"])

        if 'ppo' in str(k):
            df_dict['ppo'][name][irm][seed] = df 
            #print(k, max(df["Value"]))
            df_dict_max['ppo'][name][irm][seed] = max(df["Value"])

        if 'sac' in str(k):
            df_dict_flat[f"sac_{name}_{irm}_seed{seed}"] = df 
        if 'ppo' in str(k):
            df_dict_flat[f"ppo_{name}_{irm}_seed{seed}"] = df

    pkl.dump(df_dict, open(f"results_{run_dir}_df.pkl", "wb"))
    pkl.dump(df_dict_max, open(f"results_{run_dir}_df_max.pkl", "wb"))
    #df_dict = load_dirs(run_dir, tags)
