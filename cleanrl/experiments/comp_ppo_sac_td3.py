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
    algos = ['ppo', 'sac', 'td3']
    n_projs = [1]
    envs = ['Ant-v3', 'Walker2d-v3', 'Hopper-v3', 'HalfCheetah-v3', 'Humanoid-v3']
    lins = ['']
    usenobs = ['--use_next_obs', '', '--use_next_obs --use_dones']
    use_acs = ['']
    n_demos = [1,10]
    use_acs = ['']
    qlen = [1,5,10]
    ent_coeffs = [0.0]
    disc_nonlins = ['tanh']
    MEM_PER_CPU = 8000
    GPU_PER_CPU = 1
    gpu_string = f"--gpus={GPU_PER_CPU} " if args.gpu else ""
    # f"--gpus=gtx_1080:1 "
    
    for seed in seeds:
        for lin in lins:
            for ua in usenobs:
                for n_demo in n_demos:
                    for ql in qlen:
                        for algo in algos:
                            for env_name in envs:
                                if algo == 'ppo':
                                    periods = [1,10,100]
                                else:
                                    periods = [100]
                                for per in periods:
                                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    ua_rbl = ua.replace(" ", "")
                                    lin_rbl = lin.replace(" ", "")
                                    exp_id = (f"{ts}_{algo}_swil_{env_name}_seed_{seed}_{args.exp_id}_ql{ql}_per{per}_{ua_rbl}_{lin_rbl}")
                                    euler_slurm = (f"sbatch --mem-per-cpu={MEM_PER_CPU} {gpu_string} " 
                                                f"-o {args.output_dir / exp_id} "
                                                f"--time={args.time}:00:00 --wrap ")
                                    euler_lsf = 'bsub -W 24:00 -R "rusage[scratch=5000,mem=8000,ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceGTX1080Ti]" '
                                    # exp_cmd = (f"python {algo}_{disc_type}.py  --seed={seed} --irm_coeff={irm_coeff} --env_id {env_name}")
                                    exp_cmd = (f"python {algo}_swil.py --n_demos={n_demo} "
                                                f"--swil_period={per} {lin} --disc_nonlin=tanh --max_q_len {ql} "
                                                f"--verbose --seed={seed} --swil_loss surr_loss {ua} --env-id {env_name}")
                                    cluster_cmd = euler_slurm + exp_cmd 
                                    print("Running command: ", cluster_cmd)
                                    # subprocess.run(cluster_cmd, shell=True)
                                    subprocess.check_output(args=euler_slurm.split() + [exp_cmd])

    n_projs = [1]
    lins = ['', '--use_spectral_norm']
    usenobs = ['']
    use_acs = ['']
    use_acs = ['']
    ent_coeffs = [0.0]
    
    for seed in seeds:
        for lin in lins:
            for ua in usenobs:
                for n_demo in n_demos:
                    for algo in algos:
                        for env_name in envs:
                            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                            ua_rbl = ua.replace(" ", "")
                            lin_rbl = lin.replace(" ", "")
                            exp_id = (f"{ts}_{algo}_gail_{env_name}_seed_{seed}_{args.exp_id}_{ua_rbl}_{lin_rbl}")
                            euler_slurm = (f"sbatch --mem-per-cpu={MEM_PER_CPU} {gpu_string} " 
                                        f"-o {args.output_dir / exp_id} "
                                        f"--time={args.time}:00:00 --wrap ")
                            euler_lsf = 'bsub -W 24:00 -R "rusage[scratch=5000,mem=8000,ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceGTX1080Ti]" '
                            # exp_cmd = (f"python {algo}_{disc_type}.py  --seed={seed} --irm_coeff={irm_coeff} --env_id {env_name}")
                            exp_cmd = (f"python {algo}_gail.py --n_demos={n_demo} "
                                    f"{lin} --disc_nonlin=tanh "
                                    f"--verbose --seed={seed} {ua} --env-id {env_name}")
                            cluster_cmd = euler_slurm + exp_cmd 
                            print("Running command: ", cluster_cmd)
                            # subprocess.run(cluster_cmd, shell=True)
                            subprocess.check_output(args=euler_slurm.split() + [exp_cmd])

