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
    algos = ['sac']
    envs = ['AdroitHandDoor-v1', 'AdroitHandPen-v1', 'FrankaKitchen-v1']
    lins = ['', '--replace_atoms']
    usenobs = ['', '--use_next_obs']
    n_demos = [1, 10]
    qlens = [1,10,100]
    # subsamplings = [1,20,100]
    subsamplings = [1]
    periods = [1, 100]
    ent_coeffs = [0.0]
    disc_nonlins = ['tanh']
    MEM_PER_CPU = 8000
    GPU_PER_CPU = 1
    # gpu_string = f"--gpus={GPU_PER_CPU} " if args.gpu else ""
    gpu_string = f"--gpus={GPU_PER_CPU} " if args.gpu else ""
    # f"--gpus=gtx_1080:1 "
    
    for seed in seeds:
        for lin in lins:
            for ua in usenobs:
                for n_demo in n_demos:
                    for d_nl in disc_nonlins:
                        for per in periods:
                            for algo in algos:
                                for ql in qlens:
                                    for ss in subsamplings:
                                        for env_name in envs:
                                            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                                            ua_rbl = ua.replace(" ", "")
                                            lin_rbl = lin.replace(" ", "")
                                            exp_id = (f"{ts}_{algo}_swil_{env_name}_seed_{seed}_{args.exp_id}__{ua_rbl}_{lin_rbl}")
                                            euler_slurm = (f"sbatch --mem-per-cpu={MEM_PER_CPU} {gpu_string} " 
                                                        f"-o {args.output_dir / exp_id} "
                                                        f"--time={args.time}:00:00 --wrap ")
                                            euler_lsf = 'bsub -W 24:00 -R "rusage[scratch=5000,mem=8000,ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceGTX1080Ti]" '
                                            # exp_cmd = (f"python {algo}_{disc_type}.py  --seed={seed} --irm_coeff={irm_coeff} --env_id {env_name}")
                                            exp_cmd = (f"python {algo}_swil.py --exp-id={args.exp_id} --n_demos={n_demo} "
                                                    f"--swil_period={per} {lin} --subsample={ss} --max_q_len={ql} --disc_nonlin=tanh --verbose "
                                                    f"--seed={seed} {ua} --env-id {env_name}")
                                            cluster_cmd = euler_slurm + exp_cmd 
                                            print("Running command: ", cluster_cmd)
                                            # subprocess.run(cluster_cmd, shell=True)
                                            subprocess.check_output(args=euler_slurm.split() + [exp_cmd])
    
    lins = ['--use_spectral_norm']
    usenobs = ['']
    for seed in seeds:
        for lin in lins:
            for ua in usenobs:
                for n_demo in n_demos:
                    for algo in algos:
                        for ss in subsamplings:
                            for env_name in envs:
                                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                                ua_rbl = ua.replace(" ", "")
                                lin_rbl = lin.replace(" ", "")
                                exp_id = (f"{ts}_{algo}_gail_{env_name}_seed_{seed}_{args.exp_id}__{ua_rbl}_{lin_rbl}")
                                euler_slurm = (f"sbatch --mem-per-cpu={MEM_PER_CPU} {gpu_string} " 
                                            f"-o {args.output_dir / exp_id} "
                                            f"--time={args.time}:00:00 --wrap ")
                                euler_lsf = 'bsub -W 24:00 -R "rusage[scratch=5000,mem=8000,ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceGTX1080Ti]" '
                                # exp_cmd = (f"python {algo}_{disc_type}.py  --seed={seed} --irm_coeff={irm_coeff} --env_id {env_name}")
                                exp_cmd = (f"python {algo}_gail.py --exp-id={args.exp_id} --n_demos={n_demo} "
                                        f"--swil_period={per} {lin} --subsample={ss} --max_q_len={ql} --disc_nonlin=tanh --verbose "
                                        f"--seed={seed} {ua} --env-id {env_name}")
                                cluster_cmd = euler_slurm + exp_cmd 
                                print("Running command: ", cluster_cmd)
                                # subprocess.run(cluster_cmd, shell=True)
                                subprocess.check_output(args=euler_slurm.split() + [exp_cmd])

