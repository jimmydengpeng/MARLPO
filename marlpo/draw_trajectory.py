import sys
from typing import Optional, Tuple, Union, Iterable

import numpy as np
import json
import pickle
import pygame

from ray.rllib.algorithms import Algorithm

from metadrive.envs.marl_envs import MultiAgentIntersectionEnv
from metadrive.policy.idm_policy import ManualControllableIDMPolicy
from metadrive.constants import Decoration, TARGET_VEHICLES
from metadrive.obs.top_down_obs_impl import WorldSurface, VehicleGraphics, LaneGraphics
from metadrive.constants import PGLineType, PGLineColor
from metadrive.type import MetaDriveType
from metadrive.obs.top_down_obs_impl import LaneGraphics
from metadrive.component.map.nuplan_map import NuPlanMap
from metadrive.component.map.scenario_map import ScenarioMap
from metadrive.constants import Decoration, TARGET_VEHICLES
from metadrive.obs.top_down_obs_impl import WorldSurface, VehicleGraphics, LaneGraphics
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.utils.interpolating_line import InterpolatingLine


from metadrive.obs.top_down_obs_impl import WorldSurface, PositionType

from env.env_copo import get_lcf_env, get_lcf_from_checkpoint
from env.env_wrappers import get_rllib_compatible_env, get_neighbour_env
from env.env_utils import get_metadrive_ma_env_cls

from utils.debug import pretty_print, printPanel, get_logger
logger = get_logger()

pygame.init()

color_white = (255, 255, 255)


COLOR_BLIND = [ (2, 158, 115), (204, 120, 188), (1, 115, 178), (202, 145, 97)]
COLOR_DEFAULT = [(85, 168, 226), (255, 138, 35), (103, 213, 103), (220, 62, 63)]
COLOR_DEFAULT = [(249,65,68), (248,150,30),(144,190,109),(39,125,161)]	

COLOR_BUSINESS = [(200,45,49), (0,127,84),(25,79,151),(189,107,8)]	
COLOR_RETRO = [(251,80,80), (2, 158, 115),(7,128,207),(243,112,33)]	
COLOR_MY = [(9, 163, 135), (213, 64, 0), (7, 128, 207), (222, 173, 5)]	
COLOR_GRUVBOX = [(9, 163, 135), (213, 64, 0), (7, 128, 207), (222, 173, 5)]	

# Green, Red, Blue, Brown

def draw_top_down_trajectory(
    surface: WorldSurface, episode_data: dict, entry_differ_color=False, exit_differ_color=False, color_list=None
):
    if entry_differ_color or exit_differ_color:
        assert color_list is not None
    color_map = {}
    if not exit_differ_color and not entry_differ_color:
        color_type = 0
    elif exit_differ_color ^ entry_differ_color:
        color_type = 1
    else:
        color_type = 2

    if entry_differ_color:
        # init only once
        if "spawn_roads" in episode_data:
            spawn_roads = episode_data["spawn_roads"]
        else:
            spawn_roads = set()
            first_frame = episode_data["frame"][0]
            for state in first_frame[TARGET_VEHICLES].values():
                spawn_roads.add((state["spawn_road"][0], state["spawn_road"][1]))
            logger.debug(f'2-------- {spawn_roads}')
        keys = [road[0] for road in list(spawn_roads)]
        keys.sort()
        color_map = {key: color for key, color in zip(keys, color_list)}

    for frame in episode_data["frame"]:
        last = None
        for k, state, in frame[TARGET_VEHICLES].items():
            if color_type == 0:
                color = color_white
            elif color_type == 1:
                if exit_differ_color:
                    key = state["destination"][1]
                    if key not in color_map:
                        color_map[key] = color_list.pop()
                    color = color_map[key]
                else:
                    color = color_map[state["spawn_road"][0]]
            else:
                key_1 = state["spawn_road"][0]
                key_2 = state["destination"][1]
                if key_1 not in color_map:
                    color_map[key_1] = dict()
                if key_2 not in color_map[key_1]:
                    color_map[key_1][key_2] = color_list.pop()
                color = color_map[key_1][key_2]
            start = state["position"]
            pygame.draw.circle(surface, color, surface.pos2pix(start[0], start[1]), 2)
            # center = surface.pos2pix(start[0], start[1])
            # pygame.draw.rect(surface, color, pygame.Rect(center[0]-1, center[1]-1, 2, 2))
    for step, frame in enumerate(episode_data["frame"]):
        for k, state in frame[TARGET_VEHICLES].items():
            if "done" not in state: #TODO
                break

            if not state["done"]:
                continue
            start = state["position"]
            if state["done"]:
                pygame.draw.circle(surface, (0, 0, 0), surface.pos2pix(start[0], start[1]), 7)
    return surface



def show_map_and_traj(file_path, algo_name, env_name):
    import matplotlib.pyplot as plt
    # from metadrive.obs.top_down_renderer import draw_top_down_map
    import cv2

    # BEST:
    # screen_size = (1000, 1000)
    # film_size = (1500, 1500)

    # BEST:
    # screen_size = (800, 800)
    # film_size = (1200, 1200)

    screen_size = (800, 800)
    film_size = (1400, 1400)

    screen_size = (1000, 1000)
    film_size = (1950, 1950)

    screen = pygame.display.set_mode(screen_size)
    # screen.fill((255,255,255))

    # 'marlpo/test_traj.pkl', 'rb'
    with open(file_path, 'rb') as file:
        traj = pickle.load(file)
    logger.info(f"frame length: {len(traj['frame'])}")

    env = MultiAgentIntersectionEnv()
    env.reset()

    # road_color = (35,35,35) # (80, 80, 80)
    road_color = (220,220,220)

    # get map
    m = draw_top_down_map(
        env.current_map, 
        draw_drivable_area=False, 
        return_surface=True, 
        reverse_color=False,
        film_size=film_size,
        road_color=road_color,
    )
    # assert isinstance(m, WorldSurface)
    logger.info(f'map.origin: {m.origin}')

    # 像素翻转
    # pixels = pygame.surfarray.pixels2d(m)
    # pixels ^= 2**32 - 1
    # del pixels

    # 设置轨迹的颜色主题
    colors = COLOR_BLIND
    # colors = COLOR_DEFAULT
    # colors = COLOR_RETRO
    # colors = COLOR_MY

    m = draw_top_down_trajectory(
        m, traj, entry_differ_color=True, color_list=colors
    )

    m = pygame.transform.flip(m, flip_x=True, flip_y=False)

    screen.blit(m, (-(film_size[0]-screen_size[0])/2, -(film_size[1]-screen_size[1])/2))



    pygame.display.update()

    pygame.image.save(screen, f"trajectories/{env_name}_{algo_name}.png")

    while True:
        handle_pygame_event()


def handle_pygame_event():
    for event in pygame.event.get():
        if event.type == pygame.QUIT or \
            (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            pygame.quit()
            sys.exit()


def get_single_frame(env, dones, infos) -> dict:
    frame = env.engine.traffic_manager.get_global_states()
    # `frame` is a dict, its keys & values: 
    #  1. 'target_vehicles': (dict) vehicle.id -> vehicle_state(dict) 
    #      .. vehicle_state(dict): state_name -> state_value
    #  2. 'object_to_agent': (dict) vehicle.id -> agent_id
    #  3. 'agent_to_object': (dict) agent_id -> vehicle.id

    vid_2_aid = frame['object_to_agent']
    for vid, state in frame[TARGET_VEHICLES].items():
        agent_id = vid_2_aid[vid]
        if agent_id in dones:
            info = infos[agent_id]
            #FIXME: more accurate
            if 'crash' in info:
                state["done"] = info['crash']
            else:
                state["done"] = False
        else:
            state["done"] = False
            # printPanel(frame['agent_to_object'])
            # printPanel(dones)
            # logger.warning('no agent in done dict: {}'.format(agent_id))
            
    return frame



def record_traj():
    import matplotlib.pyplot as plt
    from metadrive.obs.top_down_renderer import draw_top_down_map, draw_top_down_trajectory
    import json
    import cv2
    import pygame

    pygame.init()

    SCENE = 'intersection'
    env_cls, abbr_name = get_metadrive_ma_env_cls(SCENE, return_abbr=True) 

    TEST_CoPO = True # <~~
    if TEST_CoPO:
        env_cls = get_lcf_env(env_cls)
    else:
        env_cls = get_neighbour_env(env_cls)
    env_name, env_cls = get_rllib_compatible_env(env_cls, return_class=True)




    env_cls = MultiAgentIntersectionEnv
    env = env_cls(dict(
            use_render=False,
            num_agents=30,
            return_single_space=True,
    ))

    env.reset()
    map = env.current_map
    print(map, type(map))


    # with open("metasvodist_inter_best.json", "r") as f:
    #     traj = json.load(f)

    b_box = map.road_network.get_bounding_box() # (x0, x1, y0, y1) , (0.1, 150.4, -69.9, 80.4) 
    x_len = b_box[1] - b_box[0]
    y_len = b_box[3] - b_box[2]
    max_len = max(x_len, y_len)

    print('bounding_box:', b_box, x_len, y_len, max_len)

    should_stop = False
    frame = []
    while not should_stop:

        o, r, tm, tc, info = env.step({agent_id: [0, 0] for agent_id in env.vehicles.keys()})

        env.render(
            mode="top_down",
            film_size=(800, 800), 
            screen_size=(800, 800),
        )

        frame.append(get_single_frame(env, tm))

        if tm["__all__"]:
            should_stop = True
            # env.reset()
    
    # state = env.vehicles['agent0'].get_state()
    # printPanel(state)

    env.close()

    traj = {'frame': frame}

    with open('marlpo/test_traj.pkl', 'wb') as file:
        pickle.dump(traj, file)


# mod from metadrive-0.3.0.1
def draw_top_down_map(
    map,
    resolution: Iterable = (512, 512),
    draw_drivable_area=True,
    return_surface=False,
    film_size=None,
    reverse_color=False,
    road_color=color_white,
) -> Optional[Union[np.ndarray, pygame.Surface]]:
    import cv2
    film_size = film_size or map.film_size # map.film_size: (1024, 1024) (对于多智体环境默认为)
    surface = MyWorldSurface(film_size, 0, pygame.Surface(film_size))
    if reverse_color:
        surface.WHITE, surface.BLACK = surface.BLACK, surface.WHITE
        surface.__init__(film_size, 0, pygame.Surface(film_size))
    b_box = map.road_network.get_bounding_box()
    x_len = b_box[1] - b_box[0]
    y_len = b_box[3] - b_box[2]
    max_len = max(x_len, y_len)
    # scaling and center can be easily found by bounding box
    scaling = film_size[1] / max_len - 0.1
    surface.scaling = scaling
    centering_pos = ((b_box[0] + b_box[1]) / 2, (b_box[2] + b_box[3]) / 2)
    surface.move_display_window_to(centering_pos)

    if isinstance(map, ScenarioMap):
        if draw_drivable_area:
            for lane_info in map.road_network.graph.values():
                LaneGraphics.draw_drivable_area(lane_info.lane, surface, color=road_color)
        else:
            for id, data in map.blocks[-1].map_data.items():
                if ScenarioDescription.POLYLINE not in data:
                    continue
                type = data.get("type", None)
                if "boundary" in id:
                    num_seg = int(len(data[ScenarioDescription.POLYLINE]) / 10)
                    for i in range(num_seg):
                        # 10 points
                        end = min((i + 1) * 10, len(data[ScenarioDescription.POLYLINE]))
                        waymo_line = InterpolatingLine(np.asarray(data[ScenarioDescription.POLYLINE][i * 10:end]))
                        LaneGraphics.display_scenario(waymo_line, type, surface)

                    if (i + 1) * 10 < len(data[ScenarioDescription.POLYLINE]):
                        end = len(data[ScenarioDescription.POLYLINE])
                        waymo_line = InterpolatingLine(np.asarray(data[ScenarioDescription.POLYLINE][(i + 1) * 10:end]))
                        LaneGraphics.display_scenario(waymo_line, type, surface)
                else:
                    waymo_line = InterpolatingLine(np.asarray(data[ScenarioDescription.POLYLINE]))
                    LaneGraphics.display_scenario(waymo_line, type, surface)

    elif isinstance(map, NuPlanMap):
        if draw_drivable_area:
            for lane_info in map.road_network.graph.values():
                LaneGraphics.draw_drivable_area(lane_info.lane, surface, color=road_color)
        else:
            for block in map.attached_blocks + [map.boundary_block]:
                for boundary in block.lines.values():
                    line = InterpolatingLine(boundary.points)
                    LaneGraphics.display_nuplan(line, boundary.type, boundary.color, surface)

    else:
        for _from in map.road_network.graph.keys():
            decoration = True if _from == Decoration.start else False
            for _to in map.road_network.graph[_from].keys():
                for l in map.road_network.graph[_from][_to]:
                    if draw_drivable_area:
                        LaneGraphics.draw_drivable_area(l, surface, color=road_color)
                    else:
                        two_side = True if l is map.road_network.graph[_from][_to][-1] or decoration else False
                        # LaneGraphics.display(l, surface, two_side, use_line_color=True)
                        LaneGraphics.display(l, surface, two_side, use_line_color=False, color=road_color)
    if return_surface:
        return surface
    ret = cv2.resize(pygame.surfarray.pixels_red(surface), resolution, interpolation=cv2.INTER_LINEAR)
    return ret


class MyWorldSurface(WorldSurface):
    def __init__(self, size: Tuple[int, int], flags: object, surf: pygame.SurfaceType) -> None:
        super().__init__(size, flags, surf)
        self.fill(self.WHITE)


class WorldSurface(pygame.Surface):
    """
    A pygame Surface implementing a local coordinate system so that we can move and zoom in the displayed area.
    From highway-env, See more information on its Github page: https://github.com/eleurent/highway-env.
    """

    BLACK = (0, 0, 0)
    GREY = (100, 100, 100)
    GREEN = (50, 200, 0)
    YELLOW = (200, 200, 0)
    WHITE = (255, 255, 255)
    INITIAL_SCALING = 5.5
    INITIAL_CENTERING = [0.5, 0.5]
    SCALING_FACTOR = 1.3
    MOVING_FACTOR = 0.1
    LANE_LINE_COLOR = (35, 35, 35)

    def __init__(self, size: Tuple[int, int], flags: object, surf: pygame.SurfaceType) -> None:
        surf.fill(pygame.Color("Black"))
        super().__init__(size, flags, surf)
        self.raw_size = size
        self.raw_flags = flags
        self.raw_surface = surf
        self.origin = np.array([0, 0])
        self.scaling = self.INITIAL_SCALING
        self.centering_position = self.INITIAL_CENTERING
        # self.fill(self.BLACK) # before!
        self.fill(self.WHITE)

    def pix(self, length: float) -> int:
        """
        Convert a distance [m] to pixels [px].

        :param length: the input distance [m]
        :return: the corresponding size [px]
        """
        return int(length * self.scaling)

    def pos2pix(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert two world coordinates [m] into a position in the surface [px]

        :param x: x world coordinate [m]
        :param y: y world coordinate [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pix(x - self.origin[0]), self.pix(y - self.origin[1])

    def vec2pix(self, vec: PositionType) -> Tuple[int, int]:
        """
        Convert a world position [m] into a position in the surface [px].

        :param vec: a world position [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pos2pix(vec[0], vec[1])

    def is_visible(self, vec: PositionType, margin: int = 50) -> bool:
        """
        Is a position visible in the surface?
        :param vec: a position
        :param margin: margins around the frame to test for visibility
        :return: whether the position is visible
        """
        x, y = self.vec2pix(vec)
        return -margin < x < self.get_width() + margin and -margin < y < self.get_height() + margin

    def move_display_window_to(self, position: PositionType) -> None:
        """
        Set the origin of the displayed area to center on a given world position.

        :param position: a world position [m]
        """
        self.origin = position - np.array(
            [
                self.centering_position[0] * self.get_width() / self.scaling,
                self.centering_position[1] * self.get_height() / self.scaling
            ]
        )

    def copy(self):
        ret = WorldSurface(size=self.raw_size, flags=self.raw_flags, surf=self.raw_surface)
        ret.origin = self.origin
        ret.scaling = self.scaling
        ret.centering_position = self.centering_position
        ret.blit(self, (0, 0))
        return ret



if __name__ == '__main__':
    # record_traj()
    # traj_path = 'marlpo/traj_copo_inter.pkl'
    # traj_path = 'marlpo/traj_scpo_inter.pkl'

    # traj_path = 'trajectories/traj/traj_ippo_inter.pkl'
    # traj_path = 'trajectories/traj/traj_scpo_inter.pkl'
    traj_path = 'trajectories/traj/traj_ccppo_inter.pkl'

    _tjpth = traj_path.split('/')[-1].split('.')[0].split('_')
    algo_name = _tjpth[1]
    env_name = _tjpth[2]

    logger.debug(f'drawing traj for algo: {algo_name}, env: {env_name}')

    show_map_and_traj(traj_path, algo_name, env_name)
