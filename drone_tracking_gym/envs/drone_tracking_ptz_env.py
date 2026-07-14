# drone_tracking_gym/envs/drone_tracking_ptz_env.py
from pathlib import Path
import numpy as np
from gym import spaces
from blendtorch import btt

CONTINUOUS = True
OUTPUT_MASK = False
#"basic_tracking"
#"dynamic_tracking"
#"obstacle_tracking"
# DRONE_SCENARIO = "obstacle_tracking"
from .Scenario import DRONE_SCENARIO

class DroneTrackingEnv_PTZ(btt.env.OpenAIRemoteEnv):

    def __init__(self, render_every=1, real_time=True, seed=None, rank=0):
        super().__init__(version="0.0.1")
        start_port = 11000 + rank * 10
        if DRONE_SCENARIO == "obstacle_tracking":
            scene_name = "drone_tracking_obstacles.blend"
        else:
            scene_name = "drone_tracking.blend"
        self.launch(
            scene=Path(__file__).parent / scene_name,
            script=Path(__file__).parent / "drone_tracking_ptz.blend.py",   # 新脚本
            real_time=real_time,
            render_every=1
        )
        if CONTINUOUS:
            self.action_space = spaces.Box(
                np.array([-1, -1, -1]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )
        else:
            self.action_space = spaces.Discrete(7)
        self.obs = np.zeros((3, 160, 160))
        if OUTPUT_MASK:
            self.observation_space = spaces.Box(low=0, high=1, shape=(4, 160, 160), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=1, shape=(3, 160, 160), dtype=np.float32)

    def step(self, action):
        assert self._env, "Environment not running."
        obs, reward, done, info = self._env.step(action)
        self.obs = obs
        return obs, reward, done, info

    def render(self, mode="human"):
        frame = np.transpose(self.obs, axes=[1,2,0]) * 255
        frame = frame.astype(np.uint8)
        frame = frame[:,:,::-1]
        return frame