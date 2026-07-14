# test_color_robustness_mkl.py


import os
import gym
import torch
import numpy as np
import cv2
import json
from scipy.spatial import cKDTree
import drone_tracking_gym
import drone_tracking_gym.register_extended_envs
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import RecordVideo
from sb3_contrib.ppo_recurrent import RecurrentPPO

EPS = 1e-8

# ================== MKL  ==================
def mkl(A, B):
    """MKL """
    Da2, Ua = np.linalg.eig(A)
    Da2 = np.diag(Da2)
    Da2[Da2 < 0] = 0
    Da = np.sqrt(Da2 + EPS)
    C = Da @ np.transpose(Ua) @ B @ Ua @ Da
    Dc2, Uc = np.linalg.eig(C)
    Dc2 = np.diag(Dc2)
    Dc2[Dc2 < 0] = 0
    Dc = np.sqrt(Dc2 + EPS)
    Da_inv = np.diag(1.0 / (np.diag(Da) + EPS))
    T = Ua @ Da_inv @ Uc @ Dc @ np.transpose(Uc) @ Da_inv @ np.transpose(Ua)
    return T


# ==================  ==================
def compute_mean_cov(img):

    pixels = img.reshape(-1, img.shape[-1])  # (N,3)
    mean = np.mean(pixels, axis=0)
    cov = np.cov(pixels, rowvar=False)
    return mean, cov


def color_transfer_mkl_with_smoothing(target_img, ref_mean, ref_cov,
                                      smooth_mean=None, smooth_cov=None,
                                      alpha=0.7):

    cur_mean, cur_cov = compute_mean_cov(target_img)


    if smooth_mean is None:
        smooth_mean = cur_mean.copy()
        smooth_cov = cur_cov.copy()
    else:
        smooth_mean = alpha * smooth_mean + (1 - alpha) * cur_mean
        smooth_cov = alpha * smooth_cov + (1 - alpha) * cur_cov


    T = mkl(smooth_cov, ref_cov)


    H, W, C = target_img.shape
    pixels = target_img.reshape(-1, C)          # (N,3)
    pixels_trans = (pixels - smooth_mean) @ T + ref_mean
    pixels_trans = np.clip(pixels_trans, 0.0, 1.0)
    transferred_img = pixels_trans.reshape(H, W, C)

    return transferred_img, smooth_mean, smooth_cov


# ==================  ==================
class ReferenceStatsRetriever:
    def __init__(self, metadata_dir, images_dir):

        self.images_dir = images_dir
        with open(os.path.join(metadata_dir, "metadata.json"), 'r') as f:
            self.metadata = json.load(f)

        self.ptz_list = []      
        self.ref_stats = []     
        self.image_paths = []    

        for item in self.metadata:
            pan = item['pan']
            tilt = item['tilt']
            zoom = item['zoom']
            if pan is None or tilt is None or zoom is None:
                continue
            img_path = os.path.join(images_dir, item['image'])

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"Warning: cannot read {img_path}, skip.")
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = img_rgb.astype(np.float32) / 255.0
            mean, cov = compute_mean_cov(img)
            self.ptz_list.append([pan, tilt, zoom])
            self.ref_stats.append((mean, cov))
            self.image_paths.append(img_path)

        self.ptz_array = np.array(self.ptz_list)
        self.tree = cKDTree(self.ptz_array)

    def get_closest_stats(self, pan, tilt, zoom):

        query = np.array([pan, tilt, zoom])
        dist, idx = self.tree.query(query)
        return self.ref_stats[idx]


# ================== Gym Wrapper ==================
class ColorRobustnessMKLWrapper(gym.Wrapper):
    def __init__(self, env, retriever, alpha=0.7,
                 debug_render=False,          
                 debug_save_dir=None,         
                 debug_save_interval=10):     
        super().__init__(env)
        self.retriever = retriever
        self.alpha = alpha
        self.smooth_mean = None
        self.smooth_cov = None
        self.debug_render = debug_render
        self.debug_save_dir = debug_save_dir
        self.debug_save_interval = debug_save_interval
        self.step_counter = 0

        if self.debug_save_dir:
            os.makedirs(self.debug_save_dir, exist_ok=True)


        if self.debug_render:
            cv2.namedWindow('Transferred Image', cv2.WINDOW_NORMAL)

    def reset(self, **kwargs):
        self.smooth_mean = None
        self.smooth_cov = None
        self.step_counter = 0
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        #  PTZ
        if isinstance(info, dict):
            ptz = info.get('ptz', None)
        else:
            ptz = info[0].get('ptz', None) if isinstance(info, (list, tuple)) and len(info) > 0 else None

        if ptz is not None and len(ptz) == 3:
            pan, tilt, zoom = ptz
            ref_mean, ref_cov = self.retriever.get_closest_stats(pan, tilt, zoom)
            # print(f"[Debug] ref_mean = {ref_mean}")

            #  obs ：(C,H,W) -> (H,W,C)  [0,1]
            if obs.ndim == 3 and obs.shape[0] == 3:
                obs = np.transpose(obs, (1, 2, 0))
            if obs.max() > 1.0:
                obs = obs.astype(np.float32) / 255.0
            else:
                obs = obs.astype(np.float32)


            obs_rgb = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)


            transferred_rgb, self.smooth_mean, self.smooth_cov = color_transfer_mkl_with_smoothing(
                obs_rgb, ref_mean, ref_cov, self.smooth_mean, self.smooth_cov, self.alpha
            )
            transferred_rgb = transferred_rgb.astype(np.float32)

            transferred_bgr = cv2.cvtColor(transferred_rgb, cv2.COLOR_RGB2BGR)


            if self.debug_render or self.debug_save_dir:
                disp_img = (transferred_bgr * 255).astype(np.uint8)
                if self.debug_render:
                    cv2.imshow('Transferred Image', disp_img)
                    cv2.waitKey(1)
                if self.debug_save_dir and (self.step_counter % self.debug_save_interval == 0):
                    save_path = os.path.join(self.debug_save_dir, f'step_{self.step_counter:06d}.png')
                    cv2.imwrite(save_path, disp_img)
                    print(f"[Debug] Saved transferred image to {save_path}")

   
            obs = np.transpose(transferred_bgr, (2, 0, 1))

        self.step_counter += 1
        return obs, reward, done, info


# ===================================
def main():

    from drone_tracking_gym.envs.Scenario import DRONE_SCENARIO
    from env_id import ENV_NAME

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

    from env_id import NUM_REF, ALPHA
    # NUM_REF = 5000
    # ALPHA = 0.7

    TEST_RUN_NAME = DRONE_SCENARIO+"_"+ENV_NAME+"_w"+"_A"+str(ALPHA)+"_N"+str(NUM_REF)
    NUM_EPISODES = 10
    
    VIDEO_DIR = f"./videos/{TEST_RUN_NAME}"
    LOG_DIR = f"./log_dirs/{TEST_RUN_NAME}"

    REFERENCE_METADATA_DIR = "./reference_data/"+DRONE_SCENARIO+"_"+str(NUM_REF)
    REFERENCE_IMAGES_DIR = os.path.join(REFERENCE_METADATA_DIR, "images")
    
    # ALPHA = 0

    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    SEED = 42
    import random
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)


    retriever = ReferenceStatsRetriever(REFERENCE_METADATA_DIR, REFERENCE_IMAGES_DIR)

    def make_env():
        env = gym.make(ENV_ID, real_time=False)
        env = ColorRobustnessMKLWrapper(
            env, retriever, alpha=ALPHA,
            debug_render=False,                    
            debug_save_dir=None,       
            debug_save_interval=10             
        )
        env = RecordVideo(env, VIDEO_DIR, episode_trigger=lambda ep: True, name_prefix="episode", video_length=0)
        env = Monitor(env, LOG_DIR)
        return env

    vec_env = DummyVecEnv([make_env])
    model = RecurrentPPO.load(MODEL_PATH, env=vec_env, learning_rate=0, verbose=1, device="cuda")

    obs = vec_env.reset()
    obs, reward, done, info = vec_env.step([[0,0,0]])
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
    vec_env.close()


if __name__ == "__main__":
    main()