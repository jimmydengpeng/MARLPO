import argparse
import sys


def get_other_training_resources():
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

def get_num_workers():
    if sys.platform.startswith('darwin'):
        return 4
    elif sys.platform.startswith('linux'):
        return 4
    else:
        return 0



def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--num-cpus-per-worker", type=float, default=1.0)
    parser.add_argument("--num-gpus-per-trial", type=float, default=0.25)
    parser.add_argument("--test", action="store_true")
    return parser


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
    
def get_agent_str(num_agents):
    if isinstance(num_agents, int):
        return str(num_agents)

    if isinstance(num_agents, list):
        l = []
        for n in num_agents:
            l.append(str(n))
        return '-'.join(l)
