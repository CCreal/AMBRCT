# generate_reference_images.py
import os
import numpy as np
import gym
import cv2
import json
import drone_tracking_gym
import drone_tracking_gym.register_extended_envs
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder



from drone_tracking_gym.envs.Scenario import DRONE_SCENARIO
#"basic_tracking"
#"dynamic_tracking"
#"obstacle_tracking"


if DRONE_SCENARIO=="basic_tracking":
    MODEL_PATH = r"E:\lcc\project_space\UAV-tracking-ftz\models\Tracking_Basic_lr_0.0001"
elif DRONE_SCENARIO=="dynamic_tracking":
    MODEL_PATH = r"E:\lcc\project_space\UAV-tracking-ftz\models\Tracking_Dynamic_lr_0.0001_best"
else:
    MODEL_PATH = r"E:\lcc\project_space\UAV-tracking-ftz\models\Tracking_Obstacle_lr_7.5e-05_loaded_model_best_2"

NUM_EPISODES = 30


ENV_ID = "blendtorch-drone_tracking_ptz-v0"   
OUTPUT_DIR = "./reference_data/"+DRONE_SCENARIO+"_"+"all"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")


SEED = 1234

import random
np.random.seed(SEED)
import torch
torch.manual_seed(SEED)
random.seed(SEED)


def make_env():
    env = gym.make(ENV_ID, real_time=False)
    return env


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    env = DummyVecEnv([make_env])
    model = RecurrentPPO.load(MODEL_PATH, env=env, verbose=1, device="cuda")

    metadata = []   

    obs = env.reset()
        #for continuous 
    obs, reward, done, info = env.step([[0,0,0]])

    for ep in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        step_cnt = 0
        while not done:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, reward, done, info = env.step(action)



            img = obs[0]          # shape: (C, H, W) 或 (H, W, C)

            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))

            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)


            img_file = f"ep{ep}_step{step_cnt}.png"
            cv2.imwrite(os.path.join(IMAGES_DIR, img_file), img)

            info = info[0]
            pan, tilt, zoom = info['ptz'] if 'ptz' in info else (None, None, None)
            metadata.append({
                "episode": ep,
                "step": step_cnt,
                "pan": float(pan) if pan is not None else None,
                "tilt": float(tilt) if tilt is not None else None,
                "zoom": float(zoom) if zoom is not None else None,
                "image": img_file
            })
            step_cnt += 1


    with open(os.path.join(OUTPUT_DIR, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    env.close()
    print(f"Saved {len(metadata)} frames to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()