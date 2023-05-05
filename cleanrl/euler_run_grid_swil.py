import subprocess
import argparse
from datetime import datetime
from pathlib import Path

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', type=Path, default=Path('/cluster/project/jbuhmann/oivan/cleanrl/cleanrl/slurm_out'), required=False, help='output dir path')
    p.add_argument('--time', default=72, required=False, help='which algorithms to use')
    p.add_argument('--exp_id', default='', required=False, help='which algorithms to use')
    p.add_argument('--test_run', default=False, required=False, action='store_true', help='use subproc for parallel envs')

    args = p.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    seeds = [515, 234, 894, 322, 72]
    # seeds = [515]
    # algos = ['ppo', 'sac']
    algos = ['ppo', 'sac']
    disc_types = ['swil']
    # irm_coeffs = [0.0, 0.1, 1.0, 10.0, 100.0]
    irm_coeffs = [0.0]
    n_projs = [1,5,10,100]
    # irm_coeffs = [0.1, 1.0, 10.0, 50.0, 100.0]
    # irm_coeffs = [0.0, 1.0, 100.0]
    # irm_coeffs = [50.0, 100.0]
    envs = ['Ant-v3', 'Walker2d-v3', 'Hopper-v3', 'HalfCheetah-v3']
    # envs = ['HalfCheetah-v3']
    use_acs = ['', '--use_actions']
    # use_acs = ['']
    lip_coeffs = [1.0]
    # lip_coeffs = [0.0, 1.0]
    # lip_coeff = ['--use_actions']
    MEM_PER_CPU = 8000
    GPU_PER_CPU = 1
    
    for seed in seeds:
        for disc_type in disc_types:
            for irm_coeff in irm_coeffs:
                for n_proj in n_projs:
                    for ua in use_acs:
                        for algo in algos:
                            for env_name in envs:
                                exp_id = (f"{ts}_{algo}_{disc_type}_{env_name}_seed_{seed}_{args.exp_id}_{irm_coeff}_lip{lip_coeff}_{ua}")
                                euler_slurm = (f"sbatch --mem-per-cpu={MEM_PER_CPU} --gpus={GPU_PER_CPU} "
                                            f"--gpus=gtx_1080:1 "
                                            f"-o {args.output_dir / exp_id} "
                                            f"--time={args.time}:00:00 --wrap ")
                                euler_lsf = 'bsub -W 24:00 -R "rusage[scratch=5000,mem=8000,ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceGTX1080Ti]" '
                                # exp_cmd = (f"python {algo}_{disc_type}.py  --seed={seed} --irm_coeff={irm_coeff} --env_id {env_name}")
                                exp_cmd = (f"python {algo}_{disc_type}.py  --verbose --seed={seed} --n_proj={n_proj} --irm_coeff={irm_coeff} {ua} --lip_coeff={lip_coeff} --env-id {env_name}")
                                cluster_cmd = euler_slurm + exp_cmd 
                                print("Running command: ", cluster_cmd)
                                # subprocess.run(cluster_cmd, shell=True)
                                subprocess.check_output(args=euler_slurm.split() + [exp_cmd])

