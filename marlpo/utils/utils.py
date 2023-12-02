import argparse, sys
from typing import List, Union
from env.env_utils import get_abbr_scene, get_env_default_num_agents

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
    num_agents: Union[int, List[int]] = None, 
    seeds: list = [5000],
    test: bool = False,
):
    '''
    Args:
        num_agents: 用于生成实验名, 返回一个list
        seeds: 可能会变

    Returns:
        num_agents: list
    '''
    test = (args.test or test or False) # True if any or False if all False
    if test:
        exp_name = "TEST"
        num_rollout_workers = 0
        seeds = [5000]
        num_agents = [2]
    else:
        if num_agents is None:
            num_agents = get_env_default_num_agents(scene)
        num_rollout_workers, num_cpus_per_worker = get_training_resources(args)
        exp_name = get_exp_name(algo_name, exp_des, scene, num_agents, num_rollout_workers)
        if not isinstance(num_agents, list) and isinstance(num_agents, int):
            num_agents = [num_agents]

    if args.num_agents is not None:
        num_agents = [args.num_agents] 

    return num_agents, exp_name, num_rollout_workers, num_cpus_per_worker, seeds, test


def get_num_agents_str(num_agents):
    if isinstance(num_agents, int):
        return str(num_agents)

    if isinstance(num_agents, list):
        l = []
        for n in num_agents:
            l.append(str(n))
        return '-'.join(l)


def get_exp_name(algo_name, exp_des, scene, num_agents, num_workers):
    EXP_SUFFIX = ('_' if exp_des else '') + exp_des
    return algo_name + f"_{get_abbr_scene(scene)}_{get_num_agents_str(num_agents)}agents_{num_workers}workers" + EXP_SUFFIX


def get_num_workers(args):
    if args.num_workers:
        return args.num_workers
    if sys.platform.startswith('darwin'):
        return 8
    elif sys.platform.startswith('linux'):
        return 8
    else:
        return 0


def get_training_resources(args):
    num_w = get_num_workers(args)
    if num_w == 8:
        num_cpus_w = 0.125
    elif num_w == 4:
        num_cpus_w = 0.25
    else:
        raise NotImplementedError
    return num_w, num_cpus_w