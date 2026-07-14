

import os
import gym
import torch
import torch.nn as nn
import numpy as np
import drone_tracking_gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.ppo_recurrent import RecurrentPPO

import matplotlib.pyplot as plt
import drone_tracking_gym.register_extended_envs
from gym.wrappers import RecordVideo

# ========== 1.  CustomCNN ==========
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    
        
    def normalize(self, img):
        return 1 -(img-np.min(img))/(np.max(img)-np.min(img))

    def viz_feats(self, observations):
        a = nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0)
        b = nn.ReLU()
        a = a.to("cuda")
        b = b.to("cuda")
        c = a(observations)
        feats_detached = c.detach().cpu().numpy()
        feats_avgd = np.sum(feats_detached[0], axis=0)

        feats_avgd = self.normalize(feats_avgd)
        plt.imshow(feats_avgd)
        plt.savefig('features_lay_1.png')
        plt.colorbar()
        plt.show()

# ========== 2.  ==========
# MODEL_PATH = r"E:\lcc\project_space\UAV-tracking-ftz\models\Tracking_Obstacle_lr_7.5e-05_loaded_model_best_2"
# MODEL_PATH = r"E:\lcc\project_space\UAV-tracking-ftz\models\Tracking_Dynamic_lr_0.0001_best"
# MODEL_PATH = r"E:\lcc\project_space\UAV-tracking-ftz\models\Tracking_Basic_lr_0.0001"

from drone_tracking_gym.envs.Scenario import DRONE_SCENARIO
#"basic_tracking"
#"dynamic_tracking"
#"obstacle_tracking"

from env_id import ENV_NAME
# "hue60"
# "hue60m"
# "dark15"
# "dark05"

if DRONE_SCENARIO=="basic_tracking":
    MODEL_PATH = r"E:\lcc\project_space\UAV-tracking-ftz\models\Tracking_Basic_lr_0.0001"
elif DRONE_SCENARIO=="dynamic_tracking":
    MODEL_PATH = r"E:\lcc\project_space\UAV-tracking-ftz\models\Tracking_Dynamic_lr_0.0001_best"
else:
    MODEL_PATH = r"E:\lcc\project_space\UAV-tracking-ftz\models\Tracking_Obstacle_lr_7.5e-05_loaded_model_best_2"

if ENV_NAME=="hue60":
    ENV_ID = "blendtorch-drone_tracking_ptz_hue60-v0"
elif ENV_NAME=="hue60m":
    ENV_ID = "blendtorch-drone_tracking_ptz_hue60m-v0"
elif ENV_NAME=="dark15":
    ENV_ID = "blendtorch-drone_tracking_ptz_dark15-v0"
elif ENV_NAME=="dark05":
    ENV_ID = "blendtorch-drone_tracking_ptz_dark05-v0"
elif ENV_NAME=="ori":
    ENV_ID = "blendtorch-drone_tracking_ptz-v0"

TEST_RUN_NAME = DRONE_SCENARIO+"_"+ENV_NAME+"_wo"

NUM_EPISODES = 10               

SEED = 42

import random
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


VIDEO_DIR = f"./videos/{TEST_RUN_NAME}"
LOG_DIR = f"./log_dirs/{TEST_RUN_NAME}"
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==========  RecordVideo + Monitor ==========
def make_env_with_video():

    env = gym.make(ENV_ID, real_time=False)
    # RecordVideo
    env = RecordVideo(env, VIDEO_DIR, episode_trigger=lambda ep: True, name_prefix=f"episode", video_length=0)
    env = Monitor(env, LOG_DIR)   # 用于记录奖励
    return env


vec_env = DummyVecEnv([make_env_with_video])

# ==========  ==========
model = RecurrentPPO.load(MODEL_PATH, env=vec_env, learning_rate=0,  verbose=1, device="cuda")  # 或 "cpu"

# ==========  ==========
obs = vec_env.reset()
obs, reward, done, info = vec_env.step([[0,0,0]])
print(obs)
print(info)
# assert 1==2
obs = vec_env.reset()
lstm_states = None
episode_starts = np.ones((1,), dtype=bool)

episode_returns = []
episode_lengths = []
current_return = 0.0
current_length = 0
episode_count = 0

while episode_count < NUM_EPISODES+1:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, infos = vec_env.step(action)
    current_return += rewards[0]
    current_length += 1
    episode_starts = dones

    if dones[0]:
        episode_count += 1
        if episode_count==1:
            pass
        else:
            episode_returns.append(current_return)
            episode_lengths.append(current_length)
            print(f"Episode {episode_count}: return={current_return:.2f}, length={current_length}")
        current_return = 0.0
        current_length = 0



stats_file = os.path.join(LOG_DIR, "test_stats.csv")
with open(stats_file, 'w') as f:
    f.write("episode,return,length\n")
    for ep, ret, leng in zip(range(1, episode_count+1), episode_returns, episode_lengths):
        f.write(f"{ep},{ret},{leng}\n")

print("\n===== Test Results =====")
print(f"Mean return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
print(f"Mean length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
