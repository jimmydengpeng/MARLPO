from typing import List, Tuple
import logging
import math
import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch
from utils.debug import get_logger

# logger = logging.getLogger('ray.rllib')
logger = get_logger()

ORIGINAL_REWARDS = "original_rewards"
NEI_REWARDS = "nei_rewards"
SVO = 'svo'
NEXT_VF_PREDS = 'next_vf_preds'
HAS_NEIGHBOURS = 'has_neighbours'
NUM_NEIGHBOURS = 'num_neighbours'
ATTENTION_MAXTRIX = 'attention_maxtrix'

NEI_REWARDS_MODE = 'nei_rewards_mode'

MEAN_NEI_REWARDS = 'mean'                 # ─╮ 
MAX_NEI_REWARDS = 'max_nei_rewards'                   #  │
NEAREST_NEI_REWARDS = 'nearest_nei_reward'            #  │──> Choose 1 alternatively
SUM_NEI_REWARDS = 'sum'
ATTENTIVE_ONE_NEI_REWARD = 'attentive_one_nei_reward' #  │
ATTENTIVE_ALL_NEI_REWARD = 'attentive_all_nei_reward' # ─╯



def add_neighbour_rewards(
    config: dict,
    sample_batch: SampleBatch,
) -> Tuple[List, List]:
    '''Args:
        config: a Policy Config.
        sample_batch: sample_batch to be added to.
    '''
    infos = sample_batch[SampleBatch.INFOS]
    nei_rewards = []
    num_neighbours = []
    has_neighbours = []

    assert isinstance(infos, np.ndarray)

    for i, info in enumerate(infos):
        assert isinstance(info, dict)
        # Abormal, when env reset, the 1st info will be {0: {'agent0': ..., 'agent1': ...}}
        if NEI_REWARDS not in info:
            # logger.warning(
            #     "No 'nei_rewards' key in info for agent{} at t_{}!".format(
            #     sample_batch['agent_index'][i], 
            #     sample_batch[SampleBatch.T][i])
            # )
            assert set(info.keys()) == set([0])
            assert sample_batch['t'][0] == 0
            keys = [info.keys() for info in infos]
            assert [0] == [i for i, k in enumerate(keys) if k == set([0])]
                
            agent_id = f"agent{sample_batch['agent_index'][0]}"

            true_info = info[0][agent_id]
            info = true_info
        if i == 0:
            continue    
        if i > 0 and i < len(infos) - 1:
            assert np.isclose(sample_batch[SampleBatch.REWARDS][i], infos[i+1]['step_reward']), (sample_batch[SampleBatch.REWARDS][i], infos[i+1]['step_reward'])

        # Normal
        # == NEI_REWARDS 列表不为空, 即有邻居 ==
        assert NEI_REWARDS in info
        # only use a max num of neighbours
        nei_rewards_t = info[NEI_REWARDS][: config['num_neighbours']]
        assert isinstance(nei_rewards_t, list)
        num_neighbours.append(len(nei_rewards_t))
        has_neighbours.append(len(nei_rewards_t) > 0)

        if nei_rewards_t: # === nei_rewards_t 列表不为空 ===
            assert len(info[NEI_REWARDS]) > 0
            nei_r = 0
            # 1. == 使用基于规则的邻居奖励 ==
            if config[NEI_REWARDS_MODE] == MEAN_NEI_REWARDS:
                nei_r = np.mean(nei_rewards_t)
            elif config[NEI_REWARDS_MODE] == MAX_NEI_REWARDS:
                nei_r = np.max(nei_rewards_t)
            elif config[NEI_REWARDS_MODE] == NEAREST_NEI_REWARDS:
                nei_r = nei_rewards_t[0]
            elif config[NEI_REWARDS_MODE] == SUM_NEI_REWARDS:
                nei_r = np.sum(nei_rewards_t)

            # 2. or == 使用注意力选择一辆车或多辆车 ==
            else:
                raise NotImplementedError

            nei_rewards.append(nei_r)
        
        else: # == NEI_REWARDS 列表为空, 即没有邻居 ==
            assert len(info[NEI_REWARDS]) == 0
            ego_r = sample_batch[SampleBatch.REWARDS][i-1]
            # 1. == 使用自己的奖励当做邻居奖励 ==
            if config.get('nei_reward_if_no_nei', None) == 'self':
                nei_rewards.append(ego_r)
            # 2. == 此时邻居奖励为0 ==
            elif config.get('nei_reward_if_no_nei', None) == '0':
                nei_rewards.append(0.)
            # 3. == 默认为0 ==
            else:
                nei_rewards.append(0.)

    assert len(nei_rewards) + 1 == len(infos)
    if nei_rewards:
        nei_rewards.append(nei_rewards[-1])
        has_neighbours.append(has_neighbours[-1])
        num_neighbours.append(num_neighbours[-1])
    else:
        nei_rewards.append(0)
        has_neighbours.append(False)
        num_neighbours.append(0)
    nei_rewards = np.array(nei_rewards).astype(np.float32)
    has_neighbours = np.array(has_neighbours)
    num_neighbours = np.array(num_neighbours)

    sample_batch[NEI_REWARDS] = nei_rewards
    sample_batch[HAS_NEIGHBOURS] = has_neighbours
    sample_batch[NUM_NEIGHBOURS] = num_neighbours

    return sample_batch
