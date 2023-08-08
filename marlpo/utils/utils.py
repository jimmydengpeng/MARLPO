import argparse, sys
from typing import List, Union


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--num-agents", type=int)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--num-workers", type=int,)
    parser.add_argument("--num-cpus-per-worker", type=float, default=1.0)
    parser.add_argument("--num-gpus-per-trial", type=float, default=0.25)
    parser.add_argument("--test", action="store_true")
    return parser



def get_other_training_configs(
    args, 
    algo_name, 
    exp_des, 
    scene, 
    num_agents: Union[int, List[int]], 
    seeds: list,
    test: bool,
):
    '''Args:

        Returns:
            num_agents: list
    '''
    test = args.test if args.test else test
    if test:
        exp_name = "TEST"
        num_rollout_workers = 0
        seeds = [5000]
        num_agents = [2]
    else:
        exp_name = get_exp_name(algo_name, exp_des, scene, num_agents)
        if not isinstance(num_agents, list) and isinstance(num_agents, int):
            num_agents = [num_agents]
        num_rollout_workers = get_num_workers()

    if args.num_workers:
        num_rollout_workers = args.num_workers
    if args.num_agents is not None:
        num_agents = [args.num_agents] 

    return num_agents, exp_name, num_rollout_workers, seeds, test


def get_abbr_scene(name: str):
    scenes = {
        "roundabout":   "Round",
        "intersection": "Inter",
        "tollgate":     "Tollg",
        "bottleneck":   "Bottn",
        "parkinglot":   "Parkl"
    }
    assert name in scenes
    return scenes[name]


def get_num_agents_str(num_agents):
    if isinstance(num_agents, int):
        return str(num_agents)

    if isinstance(num_agents, list):
        l = []
        for n in num_agents:
            l.append(str(n))
        return '-'.join(l)


def get_exp_name(algo_name, exp_des, scene, num_agents):
    EXP_SUFFIX = ('_' if exp_des else '') + exp_des
    return algo_name + f"_{get_abbr_scene(scene)}_{get_num_agents_str(num_agents)}agents" + EXP_SUFFIX


def get_num_workers():
    if sys.platform.startswith('darwin'):
        return 4
    elif sys.platform.startswith('linux'):
        return 4
    else:
        return 0


def get_training_resources():
    if sys.platform.startswith('darwin'):
        # No GPUs
        return dict(
            num_gpus=0,
            # num_cpus_per_trainer_worker
            # num_cpus_per_worker=0.5,
        )
    elif sys.platform.startswith('linux'):
        # 1 GPU
        return dict(
            num_gpus=0,
            # num_cpus_per_worker=0.5,
            # num_gpus_per_learner_worker=,
        )
    else:
        return {}