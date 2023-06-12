from typing import Any, Tuple
import numpy as np
from ray.rllib.connectors.connector import AgentConnector, ConnectorContext
from ray.rllib.connectors.registry import get_connector, register_connector
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentConnectorDataType

from marlpo.utils.debug import print, inspect, printPanel, pretty_print, dict_to_str

NEIGHBOUR_INFOS = "neighbour_infos"


"""
class MyAgentConnector(AgentConnector):
    def __init__(self, ctx: ConnectorContext):
        super().__init__(ctx)

    def transform(self, ac_data: AgentConnectorDataType) -> AgentConnectorDataType:
        '''Args:
            Types:
                ac_data.data.sample_batch['obs']: nd.array(1, 91)
                ac_data.data.sample_batch['infos']: nd.array(1, ) of dict
                raw_dict['agent_index']: int
                ac_data.agent_id: str
        '''

        msg = {}
        msg['ac_data.data'] = ac_data.data
        printPanel(msg, 'MyAgentConnector')
        exit()
        # raw_dict = ac_data.data.raw_dict
        # if 'agent_index' not in raw_dict:
        #     msg = {}
        #     msg['[Warning]'] = "'agent_index' not in raw_dict"
        #     msg['raw_dict'] = raw_dict
        #     msg['ac_data.agent_id'] = ac_data.agent_id
        #     printPanel(msg, title=f'{self.__class__.__name__}')
        # printPanel(raw_dict)
        # assert raw_dict['t'] != -1 and str(raw_dict['agent_index']) in ac_data.agent_id, (str(raw_dict['agent_index']), ac_data.agent_id)

        neighbour_info = {}
        # neighbour_info['agent_index'] = raw_dict['agent_index'] # int

        # add agent_id into info
        neighbour_info['agent_id'] = ac_data.agent_id # str

        infos = ac_data.data.sample_batch[SampleBatch.INFOS][0]
        # raw_dict['t'] == -1 时 没有infos 为{}
        neighbour_info['neighbours'] = infos.get('neighbours', [])
        neighbour_info['neighbours_distance'] = infos.get('neighbours_distance', [])

        ac_data.data.sample_batch[NEIGHBOUR_INFOS] = np.array([neighbour_info])

        # assert raw_dict['agent_index'] == int(ac_data.agent_id), (type(raw_dict['agent_index']), type(ac_data.agent_id))


        # msg = {}

        # msg['ac_data.env_id'] = ac_data.env_id
        # msg['ac_data.agent_id'] = ac_data.agent_id
        # # msg['ac_data.data.raw_dict'] = ac_data.data.raw_dict
        # msg['ac_data.data.sample_batch.obs'] = ac_data.data.sample_batch['obs'].shape
        # msg['ac_data.data.sample_batch.infos[0].shape'] = ac_data.data.sample_batch['infos'].shape
        # # msg['ac_data.data.sample_batch.infos[0]'] = ac_data.data.sample_batch['infos'][0]
        # msg['ac_data.data.sample_batch.NEIGHBOUR_INFOS'] = ac_data.data.sample_batch[NEIGHBOUR_INFOS]
        # # printPanel(dict_to_str(msg), title=f"from {self.__class__.__name__}")

        del ac_data.data.sample_batch[SampleBatch.INFOS]
        return ac_data

    def to_state(self) -> Tuple[str, Any]:
        return MyAgentConnector.__name__, None
    
    @staticmethod
    def from_state(ctx: ConnectorContext, params: Any):
        return MyAgentConnector(ctx)

"""


class ModEnvInfoAgentConnector(AgentConnector):
    def __init__(self, ctx: ConnectorContext):
        super().__init__(ctx)

    def transform(self, ac_data: AgentConnectorDataType) -> AgentConnectorDataType:
        '''Args:
            Types:
                ac_data.data.sample_batch['obs']: nd.array(1, 91)
                ac_data.data.sample_batch['infos']: nd.array(1, ) of dict
                raw_dict['agent_index']: int
                ac_data.agent_id: str
        '''

        msg = {}

        assert isinstance(ac_data.data, dict)
        infos = ac_data.data.get(SampleBatch.INFOS, {})

        desired_keys = [
            'velocity',
            'steering',
            'step_reward',
            'acceleration',
            'cost',
            'episode_length',
            'episode_reward',

            'arrive_dest', 
            'crash', 
            'crash_vehicle', 
            'crash_object', 
            'out_of_road', 
            'max_step', 

            'neighbours', 
            'neighbours_distance'
        ]
        
            # 删除不需要的键
        keys_to_del = infos.keys() - set(desired_keys)
        for k in keys_to_del:
            del infos[k]

        # 加入每个info所属的agent_id
        infos['agent_id'] = ac_data.agent_id

        # TODO: 在env.reset()中获取
        for k in desired_keys:
            if k not in infos:
                infos[k] = []

        # neighbour_info = {}
        # neighbour_info['agent_index'] = raw_dict['agent_index'] # int

        # add agent_id into info
        # neighbour_info['agent_id'] = ac_data.agent_id # str
        # ac_data.data[NEIGHBOUR_INFOS] = neighbour_info

        msg['ac_data.data'] = ac_data.data
        # printPanel(msg, f'{self.__class__.__name__}')

        return ac_data

    def to_state(self) -> Tuple[str, Any]:
        return ModEnvInfoAgentConnector.__name__, None
    
    @staticmethod
    def from_state(ctx: ConnectorContext, params: Any):
        return ModEnvInfoAgentConnector(ctx)


register_connector(ModEnvInfoAgentConnector.__name__, ModEnvInfoAgentConnector)