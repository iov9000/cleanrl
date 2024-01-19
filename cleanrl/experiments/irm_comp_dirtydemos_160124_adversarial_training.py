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

    
    seeds = [515, 234, 894, 72, 322]
    algos = ['sac']
    disc_types = ['airl','gail','meirl']
    envs = ['Ant-v3', 'HalfCheetah-v3', 'Humanoid-v3', 'Hopper-v3', 'Walker2d-v3', 'BipedalWalker-v3', 'LunarLanderContinuous-v2']
    # lins = ['--use_spectral_norm', '--use_weight_norm']
    lins = ['']
    # lins = ['--use_weight_norm --linear_proj']
    # usenobs = ['--use_next_obs', '', '--use_next_obs --use_dones']
    # usenobs = ['--use_next_obs', '', '--use_next_obs --use_dones']
    # n_demos = [1,4,10]
    irm_coeffs = [0.0, 0.1, 1.0, 10.0, 100.0]
    lip_coeffs = [0.0, 1.0, 10.0]
    disc_nonlins = ['tanh']
    usenobs = ['']
    MEM_PER_CPU = 4000
    GPU_PER_CPU = 1
    gpu_string = f"--gpus={GPU_PER_CPU} " if args.gpu else ""
    # f"--gpus=gtx_1080:1 "
    
    for seed in seeds:
        for lin in lins:
            for disc in disc_types:
                for irm_c in irm_coeffs:
                    for lip_c in lip_coeffs:
                        for algo in algos:
                            for ua in usenobs:
                                for env_name in envs:
                                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    # ua_rbl = ua.replace(" ", "")
                                    # lin_rbl = lin.replace(" ", "")
                                    exp_id = (f"{ts}_{algo}_{disc}_{env_name}_seed_{seed}_{args.exp_id}_irm{irm_c}_lip{lip_c}")
                                    euler_slurm = (f"sbatch --mem-per-cpu={MEM_PER_CPU} {gpu_string} " 
                                                f"-o {args.output_dir / exp_id} "
                                                f"--time={args.time}:00:00 --wrap ")
                                    euler_lsf = 'bsub -W 24:00 -R "rusage[scratch=5000,mem=8000,ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceGTX1080Ti]" '
                                    # exp_cmd = (f"python {algo}_{disc_type}.py  --seed={seed} --irm_coeff={irm_coeff} --env_id {env_name}")
                                    exp_cmd = (f"python {algo}_{disc}.py --total-timesteps 1000000 "
                                              f"{lin} --verbose --seed={seed} --env-id {env_name} {ua} --diverse_demos dirty "
                                              f"--irm_coeff {irm_c} --lip_coeff {lip_c}")
                                    cluster_cmd = euler_slurm + exp_cmd 
                                    print("Running command: ", cluster_cmd)
                                    # subprocess.run(cluster_cmd, shell=True)
                                    subprocess.check_output(args=euler_slurm.split() + [exp_cmd])

