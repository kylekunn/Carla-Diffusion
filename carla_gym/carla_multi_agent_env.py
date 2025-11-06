import random
import socket
import time

import carla
import gymnasium as gym
import numpy as np
from stable_baselines3.common.utils import set_random_seed

from .core.obs_manager.obs_manager_handler import ObsManagerHandler
from .core.task_actor.ego_vehicle.ego_vehicle_handler import EgoVehicleHandler
from .core.task_actor.scenario_actor.scenario_actor_handler import ScenarioActorHandler
from .core.zombie_vehicle.zombie_vehicle_handler import ZombieVehicleHandler
from .core.zombie_walker.zombie_walker_handler import ZombieWalkerHandler
from .utils.dynamic_weather import WeatherHandler
from .utils.traffic_light import TrafficLightHandler


def get_random_seed():
    t = int(time.time() * 1000.0)
    t = (
        ((t & 0xFF000000) >> 24)
        + ((t & 0x00FF0000) >> 8)
        + ((t & 0x0000FF00) << 8)
        + ((t & 0x000000FF) << 24)
    )
    return t


def wait_for_port(host, port, timeout=30):
    """Wait for a port to be ready before connecting."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection((host, port), 1):
                return
        except:
            time.sleep(0.2)
    raise RuntimeError(f"CARLA RPC port {host}:{port} not ready after {timeout}s")


class CarlaMultiAgentEnv(gym.Env):
    def __init__(
        self,
        carla_map,
        host,
        port,
        seed,
        no_rendering,
        obs_configs,
        reward_configs,
        terminal_configs,
        all_tasks,
    ):
        self.all_tasks = all_tasks
        self.obs_configs = obs_configs
        self.carla_map = carla_map
        self.seed = seed

        self.name = self.__class__.__name__

        self.om_handler = None
        self.ev_handler = None
        self.zw_handler = None
        self.zv_handler = None
        self.sa_handler = None
        self.wt_handler = None

        self.port = port
        self.no_rendering = no_rendering
        self.obs_configs = obs_configs
        self.reward_configs = reward_configs
        self.terminal_configs = terminal_configs

        self._init_client(host, port)

        # observation spaces
        self.observation_space = self.om_handler.observation_space
        for env_id in self.observation_space:
            self.observation_space[env_id] = gym.spaces.Dict(
                {
                    **self.observation_space[env_id],
                    "cur_waypoint": gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                    ),
                    "target_waypoint": gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                    ),
                    "next_waypoint": gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                    ),
                    "next_command": gym.spaces.Discrete(7),
                }
            )

        # define action spaces exposed to agent
        # throttle, steer, brake
        self.action_space = gym.spaces.Dict(
            {
                ego_vehicle_id: gym.spaces.Box(
                    low=np.array([0.0, -1.0, 0.0]),
                    high=np.array([1.0, 1.0, 1.0]),
                    dtype=np.float32,
                )
                for ego_vehicle_id in obs_configs.keys()
            }
        )

        self.task_idx = 0
        self.shuffle_task = True
        self._task = self.all_tasks[self.task_idx].copy()

    def set_task_idx(self, task_idx):
        self.task_idx = task_idx
        self.shuffle_task = False
        self._task = self.all_tasks[self.task_idx].copy()

    @property
    def num_tasks(self):
        return len(self.all_tasks)

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, task):
        self._task = task

    def reset(self, seed=None, **options):
        if self.shuffle_task:
            self.task_idx = np.random.choice(self.num_tasks)
            self.task = self.all_tasks[self.task_idx].copy()
        self.random_load_map()
        TrafficLightHandler.reset(self.world)

        self.wt_handler.reset(self.task["weather"])

        ev_spawn_locations = self.ev_handler.reset(self.task["ego_vehicles"])

        self.sa_handler.reset(self.task["scenario_actors"], self.ev_handler.ego_vehicles)

        self.zw_handler.reset(self.task["num_zombie_walkers"], ev_spawn_locations)

        self.zv_handler.reset(self.task["num_zombie_vehicles"], ev_spawn_locations)

        self.om_handler.reset(self.ev_handler.ego_vehicles)

        self.world.tick()

        snap_shot = self.world.get_snapshot()
        self._timestamp = {
            "step": 0,
            "frame": snap_shot.timestamp.frame,
            "relative_wall_time": 0.0,
            "wall_time": snap_shot.timestamp.platform_timestamp,
            "relative_simulation_time": 0.0,
            "simulation_time": snap_shot.timestamp.elapsed_seconds,
            "start_frame": snap_shot.timestamp.frame,
            "start_wall_time": snap_shot.timestamp.platform_timestamp,
            "start_simulation_time": snap_shot.timestamp.elapsed_seconds,
        }

        _, _, info_dict = self.ev_handler.tick(self.timestamp)
        # get obeservations
        obs_dict = self.om_handler.get_observation(self.timestamp)
        cur_waypoints = self.ev_handler.get_current_pos()
        target_waypoints = self.ev_handler.get_target_pos()
        next_waypoints = self.ev_handler.get_next_pos()
        for ev_id in self.om_handler._obs_managers.keys():
            next_waypoint, next_command = next_waypoints[ev_id]
            obs_dict[ev_id]["cur_waypoint"] = np.array(
                [cur_waypoints[ev_id].x, cur_waypoints[ev_id].y]
            )
            obs_dict[ev_id]["target_waypoint"] = np.array(
                [target_waypoints[ev_id].x, target_waypoints[ev_id].y]
            )
            obs_dict[ev_id]["next_waypoint"] = np.array(
                [
                    next_waypoint.transform.location.x,
                    next_waypoint.transform.location.y,
                ]
            )
            obs_dict[ev_id]["next_command"] = next_command.value

        return obs_dict, info_dict

    def step(self, control_dict=None):
        self.ev_handler.apply_control(control_dict)
        self.sa_handler.tick()
        # tick world
        self.world.tick()

        # update timestamp
        snap_shot = self.world.get_snapshot()
        self._timestamp["step"] = snap_shot.timestamp.frame - self._timestamp["start_frame"]
        self._timestamp["frame"] = snap_shot.timestamp.frame
        self._timestamp["wall_time"] = snap_shot.timestamp.platform_timestamp
        self._timestamp["relative_wall_time"] = (
            self._timestamp["wall_time"] - self._timestamp["start_wall_time"]
        )
        self._timestamp["simulation_time"] = snap_shot.timestamp.elapsed_seconds
        self._timestamp["relative_simulation_time"] = (
            self._timestamp["simulation_time"] - self._timestamp["start_simulation_time"]
        )

        reward_dict, done_dict, info_dict = self.ev_handler.tick(self.timestamp)

        # Get Observations
        obs_dict = self.om_handler.get_observation(self.timestamp)
        cur_waypoints = self.ev_handler.get_current_pos()
        target_waypoints = self.ev_handler.get_target_pos()
        next_waypoints = self.ev_handler.get_next_pos()
        for ev_id in self.om_handler._obs_managers.keys():
            next_waypoint, next_command = next_waypoints[ev_id]
            obs_dict[ev_id]["cur_waypoint"] = np.array(
                [cur_waypoints[ev_id].x, cur_waypoints[ev_id].y]
            )
            obs_dict[ev_id]["target_waypoint"] = np.array(
                [target_waypoints[ev_id].x, target_waypoints[ev_id].y]
            )
            obs_dict[ev_id]["next_waypoint"] = np.array(
                [
                    next_waypoint.transform.location.x,
                    next_waypoint.transform.location.y,
                ]
            )
            obs_dict[ev_id]["next_command"] = next_command.value

        self.wt_handler.tick(snap_shot.timestamp.delta_seconds)
        return obs_dict, reward_dict, done_dict, {}, info_dict

    def remake_task(self):
        self.om_handler = ObsManagerHandler(self.obs_configs)
        self.ev_handler = EgoVehicleHandler(
            self.client, self.reward_configs, self.terminal_configs, self.tm
        )
        self.zw_handler = ZombieWalkerHandler(self.client)
        self.zv_handler = ZombieVehicleHandler(self.client, tm_port=self.tm.get_port())
        self.sa_handler = ScenarioActorHandler(self.client)

    def random_load_map(self):
        if self.wt_handler is not None:
            self.clean()
            self.tm.set_synchronous_mode(False)
            self.set_sync_mode(False, set_tm=False)
        
        current_map = random.choice(self.carla_map)
        
        # Safe load_world: wait for RPC ready, do empty handshake, wait for GL context
        # Get current world for empty handshake
        world = self.client.get_world()
        
        # Wait 2 seconds for UE4 to build GL context
        time.sleep(2)
        
        # Check if we need to load a different map
        # Extract base map name for comparison (handle different formats)
        current_map_name = world.get_map().name
        
        # Normalize map names for comparison
        # CARLA map names can be in formats like:
        # - "Town04"
        # - "Carla/Maps/Town04"
        # - "/Game/Carla/Maps/Town04"
        def normalize_map_name(map_name):
            """Extract base map name from various formats."""
            if not map_name:
                return ""
            # Remove path prefixes
            if "/" in map_name:
                map_name = map_name.split("/")[-1]
            # Remove common prefixes
            if map_name.startswith("Town"):
                return map_name
            return map_name
        
        current_map_normalized = normalize_map_name(current_map)
        current_map_name_normalized = normalize_map_name(current_map_name)
        
        if current_map_name_normalized != current_map_normalized:
            try:
                time.sleep(1)
                # Use original map name for load_world (CARLA handles format conversion)
                self.world = self.client.load_world(current_map)
                # Wait additional time after loading
                time.sleep(1)
            except Exception as e:
                raise
        else:
            self.world = world
        
        self.tm.set_random_device_seed(get_random_seed())
        self.remake_task()
        self.wt_handler = WeatherHandler(self.world)
        self.set_sync_mode(True)
        self.set_no_rendering_mode(self.world, self.no_rendering)

    def _init_client(self, host, port):
        # Wait for RPC port to be ready before connecting
        wait_for_port(host, port, timeout=30)
        
        client = None
        while client is None:
            try:
                # Use worker_threads=0 to avoid multi-threading race conditions
                client = carla.Client(host, port, worker_threads=0)
                client.set_timeout(30.0)
            except RuntimeError as re:
                if "timeout" not in str(re) and "time-out" not in str(re):
                    pass
                client = None

        self.client = client
        self.tm = self.client.get_trafficmanager(self.port + 6000)
        self.random_load_map()
        set_random_seed(self.seed, using_cuda=True)
        self.world.tick()
        TrafficLightHandler.reset(self.world)

    def set_sync_mode(self, sync, set_tm=True):
        settings = self.world.get_settings()
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = 0.1
        settings.deterministic_ragdolls = True
        self.world.apply_settings(settings)
        if set_tm:
            self.tm.set_synchronous_mode(sync)

    @staticmethod
    def set_no_rendering_mode(world, no_rendering):
        settings = world.get_settings()
        settings.no_rendering_mode = no_rendering
        world.apply_settings(settings)

    @property
    def timestamp(self):
        return self._timestamp.copy()

    def set_timestamp(self, timestamp):
        self._timestamp = timestamp

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def close(self):
        self.clean()
        self.set_sync_mode(False)
        self.client = None
        self.world = None
        self.tm = None

    def clean(self):
        self.sa_handler.clean()
        self.zw_handler.clean()
        self.zv_handler.clean()
        self.om_handler.clean()
        self.ev_handler.clean()
        self.wt_handler.clean()

        self.sa_handler = None
        self.zw_handler = None
        self.zv_handler = None
        self.om_handler = None
        self.ev_handler = None
        self.wt_handler = None

        self.world.tick()
