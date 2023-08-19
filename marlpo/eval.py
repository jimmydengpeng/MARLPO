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
├── ippo
│   ├─ trial_0_seed=5000
│   └─ trial_1_seed=6000
└── copo
    ├─ trial_0_seed=5000
    └─ trial_1_seed=6000
"""

import os
import os.path as osp
import re
import copy
import time
import argparse
import json
import numpy as np
import pandas as pd
from rich.progress import Progress
# from copo.algo_ccppo.ccppo import get_ccppo_env
# from copo.algo_svo.svo_env import get_svo_env
# from copo.ccenv import get_ccenv
# from copo.eval.get_policy_function_from_checkpoint import get_policy_function_from_checkpoint, get_lcf_from_checkpoint
from ray.rllib.algorithms import Algorithm

from evaluation.recoder import RecorderEnv
# from metadrive.envs.marl_envs import MultiAgentIntersectionEnv, MultiAgentRoundaboutEnv, MultiAgentTollgateEnv, \
    # MultiAgentBottleneckEnv, MultiAgentParkingLotEnv, MultiAgentMetaDrive

from env.env_wrappers import get_rllib_compatible_env, get_neighbour_env
from env.env_utils import get_metadrive_ma_env_cls
from utils.debug import printPanel


def get_eval_env(env_name: str):
    '''Args:
    env_name: may be 'MultiAgentIntersectionEnv' 
        or abbreviation, e.g., 'Intersection' 
    '''
    env_cls, abbr_name = get_metadrive_ma_env_cls(env_name, return_abbr=True) 
    env_name, env_cls = get_rllib_compatible_env(
                            get_neighbour_env(env_cls),
                        return_class=True)
    env = env_cls({})
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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default="marlpo/evaluation/demo_raw_checkpoints/ippo", help="The path ending up with your exp_name."
    )
    parser.add_argument(
        "--num_episodes", type=int, default=2, help="How many episodes you want to run for a single checkpoint."
    )
    parser.add_argument(
        "--render", action='store_true')
    return parser

def get_checkpoint_infos(root):
    checkpoint_infos = []
    trial_paths = [(osp.join(root, p), p) for p in os.listdir(root) if osp.isdir(osp.join(root, p))]
    with Progress(transient=False) as progress:
        task_trial = progress.add_task("[cyan]Processing Trials", total=len(trial_paths))
        for p_idx, (trial_path, trial_name) in enumerate(trial_paths):
            progress.console.log(f"Finish {p_idx+1}/{len(trial_paths)} trials.")

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

    return checkpoint_infos


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    

    print("Evaluation begins. The results will be saved at: ", "./evaluate_results/")

    # == Get Checkpoints ==
    root = args.root
    num_episodes = args.num_episodes
    should_render = args.render

    root = os.path.abspath(root)
    algo_name = root.split('/')[-1]

    checkpoint_infos = get_checkpoint_infos(root)
            
    # for info in checkpoint_infos:
    #     printPanel(info)

    # == Evaluating ==
    os.makedirs("evaluate_results", exist_ok=True)

    saved_results = []

    result_name = f"{algo_name}_evaluate_results"

    with Progress() as progress:
        task_ckpt = progress.add_task("[orange]Evaluating Checkpoint", total=len(checkpoint_infos))
        task_epi = progress.add_task("[cyan]Episode", total=num_episodes)
        task_step = progress.add_task("[bright_black]Env step", total=None)

        for ckpt_idx, ckpt_info in enumerate(checkpoint_infos):
            assert os.path.isdir(ckpt_info["path"]), ckpt_info
            # ckpt_info = copy.deepcopy(ckpt_info)

            # Setup environment
            env, formal_env_name = get_eval_env(env_name=ckpt_info["env"])

            algo = get_algo_from_checkpoint(ckpt_info["path"])
            print(algo)

            # exit()
            # if ckpt_info["should_wrap_copo_env"]:
            #     lcf_mean, lcf_std = get_lcf_from_checkpoint(ckpt_info["trial_path"])
            # else:
            #     lcf_mean = lcf_std = 0.0


            print(
                f"\n === Evaluating Algo-{ckpt_info['algo']}_Env-{formal_env_name}_Seed-{ckpt_info['seed']}_Ckpt-{ckpt_info['ckpt_idx']} ==="
            )
            # if ckpt_info["should_wrap_copo_env"]:
            #     print("We are using CoPO environment! The LCF is set to Mean {}, STD {}".format(lcf_mean, lcf_std))

            # Evaluate this checkpoint for sufficient episodes.
            try:
                o, info = env.reset()
                tm = {"__all__": False}
                start = time.time()
                last_time = time.time()
                ep_count = 0
                step_count = 0
                ep_times = [0]
                while True:

                    # Step the environment
                    actions = compute_actions(algo, o, extra_actions=False)
                    o, r, tm, tc, info = env.step(actions)
                    step_count += 1

                    if should_render:
                        env.render(mode="topdown")

                    if step_count % 100 == 0:
                        print(
                            f"Evaluating {ckpt_info['algo']} {formal_env_name} {ckpt_info['seed']}, Num episodes: {ep_count}, "
                            f"Num steps in this episode: {step_count} (Ep time {np.mean(ep_times):.2f}, "
                            f"Total time {time.time() - start:.2f})"
                        )

                    # Reset the environment
                    if tm["__all__"]:
                        # policy_function.reset()

                        step_count = 0
                        ep_count += 1
                        progress.advance(task_epi)

                        ep_times.append(time.time() - last_time)
                        last_time = time.time()

                        print("Finish {} episodes with {:.3f} s!\n".format(ep_count, time.time() - start))
                        res = env.get_episode_result()
                        res.update(ckpt_info)
                        res["episode"] = ep_count
                        res["env"] = formal_env_name
                        saved_results.append(res)
                        df = pd.DataFrame(saved_results)
                        res_to_print = res.copy()
                        for k in ['path', 'trial', 'trial_path']:
                            res_to_print.pop(k)
                        print(
                            printPanel(res_to_print, title=f'Eval Result for Episode {ep_count}/{num_episodes} {algo_name}')
                        )

                        path = f"evaluate_results/{result_name}.csv"
                        print("Backup data is saved at: ", path)
                        df.to_csv(path)

                        tm = {"__all__": False}
                        if ep_count >= num_episodes:
                            break

                        o, info = env.reset()

            except Exception as e:
                print("Error encountered: ", e)

            progress.advance(task_ckpt)


            df = pd.DataFrame(saved_results)
            path = f"evaluate_results/{result_name}.csv"
            print("Final data is saved at: ", path)
            df.to_csv(path)
