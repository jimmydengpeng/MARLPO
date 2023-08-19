""" This a formal evaluation script for all algos.
Usage:
Just specify the abs. or rel. path to an 'algo' or 'experiment folder' 
folder, e.g. ippo.

Note:
1. The 'algo' or 'experiment folder' contains one or more 'trial folders',
    which is usually of different seeds;
2. Each 'trial folder' contains many 'checkpoint folders',
    in which the 'params.json' and 'progress.csv' exists.

Example:
evaluation
├── ippo/
│   ├─ trial_0_seed=5000/
│   │    ├─ checkpoint_100/
│   │    ├─ params.json
│   │    └─ progress.csv
│   └─ trial_1_seed=6000
└── copo/
    ├─ trial_0_seed=5000/
    └─ trial_1_seed=6000/
"""

import os
import os.path as osp
import copy
import time
import argparse
import json
import numpy as np
import pandas as pd

from rich.panel import Panel
from rich.progress import Progress
import logging

from ray.rllib.utils.deprecation import logger
logger.setLevel(logging.ERROR)

from ray.rllib.algorithms import Algorithm

from env.env_wrappers import get_rllib_compatible_env, get_neighbour_env
from env.env_utils import get_metadrive_ma_env_cls
from evaluation.recoder import RecorderEnv
from utils.debug import printPanel, print, dict_to_panel


def get_eval_env(env_name: str, env_config={}):
    '''Args:
    env_name: may be 'MultiAgentIntersectionEnv' 
        or abbreviation, e.g., 'Intersection' 
    '''
    env_cls, abbr_name = get_metadrive_ma_env_cls(env_name, return_abbr=True) 
    env_name, env_cls = get_rllib_compatible_env(
                            get_neighbour_env(env_cls),
                        return_class=True)
    env = env_cls(env_config)
    return RecorderEnv(env), abbr_name


def get_env_and_start_seed(trial_path):
    param_path = os.path.join(trial_path, "params.json")
    assert os.path.exists(param_path)
    with open(param_path, "r") as f:
        param = json.load(f)

    if "env_config" not in param:
        print(param)
        raise ValueError()

    start_seed = param["env_config"]["start_seed"]
    # may be 'MultiAgentIntersectionEnv' or abbreviation, e.g., 'Intersection' 
    env_name = param["env"]
    return env_name, start_seed, param


def get_algo_from_checkpoint(ckpt_pth):
    algo = Algorithm.from_checkpoint(ckpt_pth)
    return algo


def compute_actions(algo: Algorithm, obs, extra_actions=False):
    if extra_actions:
        actions, states, infos = algo.compute_actions(observations=obs, full_fetch=True)
        return actions, infos
    else:
        actions = algo.compute_actions(observations=obs)
        return actions

def get_brief_epi_res(res, print_out=False):
    res_to_print = {}
    keys_rate = ['success_rate', 'crash_rate', 'out_rate']
    keys_other = ['episode_reward_mean']
    rate = 0
    for k in res:
        if k in keys_rate:
            _r = round(float(res[k])*100, 2)
            res_to_print[k] = str(_r) + ' %'
            rate += res[k]
            if k == 'success_rate':
                res_to_print[k] = '[cyan]' + res_to_print[k]+ '[/]'

    res_to_print['maxstep_rate'] = str(round((1 - rate), 2)) + ' %'
     
    for k in res:
        if k in keys_other:
            res_to_print[k] = round(res[k], 2)

    if print_out:
        printPanel(res_to_print, title=f"Episode {ep_count}/{num_episodes} • Algo {res['algo'].upper()}")
    return res_to_print

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default="marlpo/evaluation/demo_raw_checkpoints/ippo", 
        help="The path ending up with your exp_name."
    )
    parser.add_argument(
        "--num_episodes", type=int, default=2, 
        help="How many episodes you want to run for a single checkpoint."
    )
    parser.add_argument(
        "--render", action='store_true'
    )
    return parser


def get_checkpoint_infos(root):
    checkpoint_infos = []
    trial_paths = [(osp.join(root, p), p) for p in os.listdir(root) if osp.isdir(osp.join(root, p))]
    task_trial = progress.add_task("[cyan]Processing Trials", total=len(trial_paths))
    for p_idx, (trial_path, trial_name) in enumerate(trial_paths):
        progress.console.log(f"Loading {p_idx+1}/{len(trial_paths)} trials.")

        raw_env_name, start_seed, ckpt_params = get_env_and_start_seed(trial_path)

        should_wrap_cc_env = "CCPPO" in trial_name
        should_wrap_copo_env = "CoPO" in trial_name

        # Get checkpoint path
        ckpt_paths = [] # [(ckpt_path, ckpt_idx), ...]
        for ckpt_path in os.listdir(trial_path):
            if "checkpoint" in ckpt_path:
                ckpt_paths.append((ckpt_path, int(ckpt_path.split("_")[1])))

        # All checkpoints will be evaluated
        ckpt_paths = sorted(ckpt_paths, key=lambda p: p[1]) # [('checkpoint_490', 490), ...]

        for ckpt_path, ckpt_idx in ckpt_paths:
            # ckpt_file_path = osp.join(root, trial_path, ckpt_path, ckpt_path.replace("_", "-"))
            ckpt_file_path = osp.join(root, trial_path, ckpt_path)

            progress.console.log(
                f"We will evaluate checkpoint: Algo-[red]{algo_name}[/], "
                f"Env-[green]{raw_env_name}[/], Seed-{start_seed}, "
                f"Ckpt-{ckpt_idx}"
            )

            ckpt_info = {
                "path": ckpt_file_path,
                "ckpt_idx": ckpt_idx,
                "algo": algo_name,
                "env": raw_env_name,
                "seed": start_seed,
                "trial": trial_name,
                "trial_path": trial_path,
                "should_wrap_copo_env": should_wrap_copo_env,
                "should_wrap_cc_env": should_wrap_cc_env
            }

            # ckpt_info["config"] = ckpt_config

            checkpoint_infos.append(ckpt_info)
        progress.advance(task_trial)

    progress.remove_task(task_trial)
    return checkpoint_infos


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    with Progress(transient=False) as progress:
        progress.console.log("Evaluation begins. The results will be saved at: ", "./evaluate_results/\n")

        root = args.root
        num_episodes = args.num_episodes
        should_render = args.render

        root = os.path.abspath(root)
        algo_name = root.split('/')[-1]

        # == Get Checkpoints ==
        checkpoint_infos = get_checkpoint_infos(root)
                
        # for info in checkpoint_infos:
        #     printPanel(info)

        # == Evaluating ==
        os.makedirs("evaluate_results", exist_ok=True)

        saved_results = []

        result_name = f"{algo_name}_evaluate_results"
        result_save_path = f"evaluate_results/{result_name}.csv"
        num_ckpts = len(checkpoint_infos)
   
        num_step_estimate = None
        step_esti_coeff = 0.5

        task_ckpt = progress.add_task("[cyan]Checkpoint", total=num_ckpts)
        task_epi = progress.add_task("[orange1]Episode", total=num_episodes)
        task_step = progress.add_task("[red]Step", total=num_step_estimate)

        progress.console.log(f'\nEvaluating total {num_ckpts} checkpoints...')
        for ckpt_idx, ckpt_info in enumerate(checkpoint_infos):
            assert os.path.isdir(ckpt_info["path"]), ckpt_info
            # task_step = progress.add_task("[red]Step", total=num_step_estimate)
            progress.reset(task_epi)
            # ckpt_info = copy.deepcopy(ckpt_info)

            # Setup environment
            env, abbr_env_name = get_eval_env(env_name=ckpt_info["env"], env_config={'horizon': 100})

            algo = get_algo_from_checkpoint(ckpt_info["path"])

            # if ckpt_info["should_wrap_copo_env"]:
            #     lcf_mean, lcf_std = get_lcf_from_checkpoint(ckpt_info["trial_path"])
            # else:
            #     lcf_mean = lcf_std = 0.0

            progress.console.log(
                f"Evaluating Algo-[red]{ckpt_info['algo']}[/], "
                f"Env-[green]{abbr_env_name}[/], Seed-{ckpt_info['seed']}, "
                f"Ckpt-{ckpt_info['ckpt_idx']}:"
            )
            # if ckpt_info["should_wrap_copo_env"]:
            #     print("We are using CoPO environment! The LCF is set to Mean {}, STD {}".format(lcf_mean, lcf_std))

            # Evaluate this checkpoint for sufficient episodes.
            try:
                o, info = env.reset()
                tm = {"__all__": False}
                start = time.time()
                last_epi_start_time = time.time()
                ep_count = 0
                step_count = 0
                ep_times = []
                while True:
                    # Step the environment
                    actions = compute_actions(algo, o, extra_actions=False)
                    o, r, tm, tc, info = env.step(actions)
                    step_count += 1
                    progress.advance(task_step)

                    if should_render:
                        env.render(mode="topdown")

                    # if step_count % 100 == 0:
                    #     progress.console.log(
                    #         f"Evaluating {ckpt_info['algo']} {abbr_env_name} {ckpt_info['seed']}, "
                    #         f"Num episodes: {ep_count+1}/{num_episodes}, "
                    #         f"Episode steps: {step_count}, "
                    #         f"(Epsode time {time.time()-last_epi_start_time:.2f}, "
                    #         f"Total time {time.time() - start:.2f})"
                    #     )

                    # Reset the environment
                    if tm["__all__"]:
                        # policy_function.reset()
                        ep_count += 1
                        progress.advance(task_epi)

                        ep_times.append(time.time() - last_epi_start_time)
                        last_epi_start_time = time.time()



                        res = env.get_episode_result()
                        res.update(ckpt_info)
                        res["episode"] = ep_count
                        res["env"] = abbr_env_name
                        saved_results.append(res)
                        df = pd.DataFrame(saved_results)

                        progress.console.log(
                            f"Finish {ep_count}/{num_episodes} episodes with {time.time() - start:.3f} s! "
                            f"Average {np.mean(ep_times):.3f} s per episode!\n"
                        )

                        # progress.console.log('\n')

                        # progress.console.log(f"Backup data is saved at: {path}")
                        progress.console.log(
                            dict_to_panel(
                                get_brief_epi_res(res), 
                                title=f"Episode {ep_count}/{num_episodes} • "
                                      f"Algo {res['algo'].upper()} • "
                                      f"Ckpt {ckpt_idx+1}/{num_ckpts}"
                            )
                        )
                        df.to_csv(result_save_path)

                        progress.stop_task(task_step)
                        task_step


                        tm = {"__all__": False}
                        if ep_count >= num_episodes:
                            env.close()
                            progress.update(task_step, completed=num_step_estimate)
                            # task_step = progress.add_task("[red]Step", total=num_step_estimate)
                            break

                        o, info = env.reset()
                        if num_step_estimate is None:
                            num_step_estimate = step_count
                        else:
                            num_step_estimate = (1-step_esti_coeff) * num_step_estimate + step_esti_coeff * step_count
                        step_count = 0
                        print('num_step_estimate', num_step_estimate)
                        progress.reset(task_step, completed=0, total=num_step_estimate)
                        # progress.remove_task(task_step)
                        # progress.reset(task_step)
                        # task_step = progress.add_task("[red]Step", total=num_step_estimate)

            except Exception as e:
                print("Error encountered: ", e)

            progress.advance(task_ckpt)


        df = pd.DataFrame(saved_results)
        path = f"evaluate_results/{result_name}.csv"
        print("Final data is saved at: ", path)
        df.to_csv(path)
