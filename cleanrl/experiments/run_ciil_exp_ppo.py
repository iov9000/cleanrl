import subprocess
import argparse
from datetime import datetime
from pathlib import Path

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', type=Path, default=Path('/cluster/project/jbuhmann/oivan/cleanrl/cleanrl/slurm_out'), required=False, help='output dir path')
    p.add_argument('--time', default=24, required=False, help='which algorithms to use')
    p.add_argument('--exp_id', default='', required=False, help='which algorithms to use')
    p.add_argument('--gpu', default=False, required=False, action='store_true', help='use subproc for parallel envs')
    p.add_argument('--test_run', default=False, required=False, action='store_true', help='use subproc for parallel envs')

    args = p.parse_args()

    
    seeds = [515, 234, 894, 322, 72]
    # seeds = [234]
    # algos = ['ppo', 'sac']
    # algos = ['ppo', 'sac']
    algos = ['ppo']
    disc_types = ['gail', 'airl', 'wail', 'swil']
    # irm_coeffs = [0.0, 0.1, 1.0, 10.0, 100.0]
    # n_projs = [1,5,20,50]
    # n_projs = [1,20,100]
    # n_projs = [1]
    # irm_coeffs = [0.1, 1.0, 10.0, 50.0, 100.0]
    # irm_coeffs = [0.0, 1.0, 100.0]
    # irm_coeffs = [50.0, 100.0]
    envs = ['Ant-v3', 'Walker2d-v3', 'Hopper-v3', 'HalfCheetah-v3']
    # envs = ['Humanoid-v3']
    # lins = ['--use_spectral_norm', '--use_weight_norm', '']
    # lins = ['--use_spectral_norm', '']
    # lins = ['--use_spectral_norm', '']
    lins = ['', '--use_spectral_norm', '--lip_coeff=1.0', '--lip_coeff=10.0', '--irm_coeff=50.0', '--irm_coeff=100.0', '--irm_coeff=1000.0']
    # lins = ['']
    # lins = ['--use_weight_norm --linear_proj']
    # usenobs = ['--use_next_obs', '', '--use_next_obs --use_dones']
    # usenobs = ['--use_next_obs', '', '--use_next_obs --use_dones']
    usenobs = ['']
    # envs = ['Walker2d-v3', 'Hopper-v3', 'HalfCheetah-v3']
    # envs = ['HalfCheetah-v3']
    # use_acs = ['', '--use_actions']
    # use_acs = ['--use_actions', '']
    # use_acs = ['--use_actions']
    use_acs = ['']
    # n_demos = [1,4,10]
    # n_demos = [1,4,10,25]
    n_demos = [10]
    # n_envs = [1,8,16]
    ent_coeffs = [0.0]
    # ent_coeffs = [0.0, 0.01]
    # disc_nonlins = ['tanh', 'prelu', 'relu']
    # disc_nonlins = ['tanh', 'leakyrelu']
    disc_nonlins = ['tanh']
    # lip_coeffs = [1.0]
    # lip_coeffs = [0.0, 1.0]
    # lip_coeff = ['--use_actions']
    MEM_PER_CPU = 16000
    GPU_PER_CPU = 1
    # gpu_string = f"--gpus={GPU_PER_CPU} " if args.gpu else ""
    gpu_string = f"--gpus={GPU_PER_CPU} " if args.gpu else ""
    # f"--gpus=gtx_1080:1 "
    
    for seed in seeds:
        for lin in lins:
            for disc in disc_types:
                for ua in usenobs:
                    for n_demo in n_demos:
                        for d_nl in disc_nonlins:
                            for ec in ent_coeffs:
                                for algo in algos:
                                    for env_name in envs:
                                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                                        ua_rbl = ua.replace(" ", "")
                                        lin_rbl = lin.replace(" ", "")
                                        exp_id = (f"{ts}_{algo}_{disc}_{env_name}_seed_{seed}_{args.exp_id}__{ua_rbl}_{lin_rbl}")
                                        euler_slurm = (f"sbatch --mem-per-cpu={MEM_PER_CPU} {gpu_string} " 
                                                    f"-o {args.output_dir / exp_id} "
                                                    f"--time={args.time}:00:00 --wrap ")
                                        euler_lsf = 'bsub -W 24:00 -R "rusage[scratch=5000,mem=8000,ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceGTX1080Ti]" '
                                        # exp_cmd = (f"python {algo}_{disc_type}.py  --seed={seed} --irm_coeff={irm_coeff} --env_id {env_name}")
                                        exp_cmd = (f"python {algo}_{disc}.py --exp_id={args.exp_id} --total-timesteps 3000000 --n_demos={n_demo} --ent-coef={ec} {lin} --disc_nonlin={d_nl} --verbose --seed={seed} {ua} --env-id {env_name}")
                                        cluster_cmd = euler_slurm + exp_cmd 
                                        print("Running command: ", cluster_cmd)
                                        # subprocess.run(cluster_cmd, shell=True)
                                        subprocess.check_output(args=euler_slurm.split() + [exp_cmd])

