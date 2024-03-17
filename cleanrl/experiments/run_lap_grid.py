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

    
    seeds = [515, 234, 894, 322, 72, 913, 76, 147, 410, 728]
    algos = ['sac']
    disc_types = ['gail', 'airl']
    # irm_coeffs = [0.0, 0.1, 1.0, 10.0, 100.0]
    irm_coeffs = [0.0, 1.0, 10.0]
    lip_coeffs = [0.0, 1.0, 10.0]
    lins = ['',  '--use_spectral_norm']
    for irm_c in irm_coeffs:
        for lip_c in lip_coeffs:
            lins.append(f"--irm_coeff={irm_c} --lip_coeff={lip_c}")

    usenobs = ['']
    use_acs = ['']
    n_demos = [10]
    ent_coeffs = [0.0]
    disc_nonlins = ['tanh']
    MEM_PER_CPU = 3000
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
                            for algo in algos:
                                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                                lin_rbl = lin.replace(" ", "")
                                exp_id = (f"{ts}_{algo}_{disc}_{env_name}_seed_{seed}_{args.exp_id}_{lin_rbl}")
                                euler_slurm = (f"sbatch --mem-per-cpu={MEM_PER_CPU} {gpu_string} " 
                                            f"-o {args.output_dir / exp_id} "
                                            f"--time={args.time}:00:00 --wrap ")
                                euler_lsf = 'bsub -W 24:00 -R "rusage[scratch=5000,mem=8000,ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceGTX1080Ti]" '
                                # exp_cmd = (f"python {algo}_{disc_type}.py  --seed={seed} --irm_coeff={irm_coeff} --env_id {env_name}")
                                exp_cmd = (f"python lap/ppo_{disc}_lap_liver.py --exp-id={args.exp_id} --n_demos={n_demo} "
                                        f" {lin} --disc_nonlin={d_nl} --verbose "
                                        f"--seed={seed} --env-id {env_name} "
                                        f"--collision_loss --step_size 1.0 "
                                        f"--vis_threshold_factor 50 --total-timesteps 1000000")
                                cluster_cmd = euler_slurm + exp_cmd 
                                print("Running command: ", cluster_cmd)
                                # subprocess.run(cluster_cmd, shell=True)
                                subprocess.check_output(args=euler_slurm.split() + [exp_cmd])

