import gym
import numpy as np
import cv2

class DarkEnv(gym.Wrapper):

    def __init__(self, env, factor=0.4):
        super().__init__(env)
        self.factor = factor
        self._input_format = None
        self._is_normalized = None
        self._last_obs = None          

    def _detect_format(self, obs):
        if obs.ndim == 3:
            c_first = obs.shape[0] in (1,3,4)
            c_last = obs.shape[-1] in (1,3,4)
            if c_first and not c_last:
                self._input_format = 'channels_first'
            elif c_last and not c_first:
                self._input_format = 'channels_last'
            else:
                self._input_format = 'channels_last'
        else:
            raise ValueError(f"Unsupported observation dimension: {obs.ndim}")
        
        if obs.dtype == np.uint8 or obs.max() > 1.0:
            self._is_normalized = False
        else:
            self._is_normalized = True
    
    def _adjust(self, obs):
        if self._input_format is None:
            self._detect_format(obs)
        if not self._is_normalized:
            obs = obs / 255.0
        obs = obs * self.factor
        obs = np.clip(obs, 0, 1)
        if not self._is_normalized:
            obs = (obs * 255).astype(np.uint8)
        else:
            obs = obs.astype(np.float32)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._adjust(obs)
        self._last_obs = obs          
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs) 
        return obs

    def render(self, mode='rgb_array', **kwargs):
        if mode == 'rgb_array':
            if self._last_obs is not None:

                if self._input_format == 'channels_first':
                    frame = np.transpose(self._last_obs, (1, 2, 0))
                else:
                    frame = self._last_obs
                if self._is_normalized:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)

                frame = frame[:, :, ::-1]
                return frame
            else:
                return self.env.render(mode=mode, **kwargs)
        else:
            return self.env.render(mode=mode, **kwargs)


class HueShiftEnv(gym.Wrapper):
    """
    hue_shift ：
        hue_shift > 0  -> walm 0~80
        hue_shift = 0  -> ori
        hue_shift < 0  -> cool -80~0
    """
    def __init__(self, env, hue_shift=20):
        super().__init__(env)
        self.hue_shift = hue_shift
        self._input_format = None
        self._is_normalized = None
        self._last_obs = None

    def _detect_format(self, obs):
        if obs.ndim == 3:
            c_first = obs.shape[0] in (1,3,4)
            c_last = obs.shape[-1] in (1,3,4)
            if c_first and not c_last:
                self._input_format = 'channels_first'
            elif c_last and not c_first:
                self._input_format = 'channels_last'
            else:
                self._input_format = 'channels_last'
        else:
            raise ValueError(f"Unsupported observation dimension: {obs.ndim}")
        if obs.dtype == np.uint8 or obs.max() > 1.0:
            self._is_normalized = False
        else:
            self._is_normalized = True

    def _adjust(self, obs):
        if self._input_format is None:
            self._detect_format(obs)

        if self._input_format == 'channels_first':
            img = np.transpose(obs, (1, 2, 0))
        else:
            img = obs
        if not self._is_normalized:
            img = img.astype(np.uint8)
        else:
            img = (img * 255).astype(np.uint8)


        b, g, r = cv2.split(img)


        k = np.clip(self.hue_shift / 100.0, -0.6, 0.6)

        if k >= 0:   
            r = np.clip(r * (1 + k * 1.2), 0, 255)   
            g = np.clip(g * (1 + k * 0.8), 0, 255)   
            b = np.clip(b * (1 - k * 1.5), 0, 255)   
        else:        
            k_abs = -k
            b = np.clip(b * (1 + k_abs * 1.2), 0, 255)
            g = np.clip(g * (1 - k_abs * 0.5), 0, 255)
            r = np.clip(r * (1 - k_abs * 0.8), 0, 255)

        img_new = cv2.merge([b, g, r]).astype(np.uint8)


        if self._is_normalized:
            img_new = img_new.astype(np.float32) / 255.0
        if self._input_format == 'channels_first':
            obs_new = np.transpose(img_new, (2, 0, 1))
        else:
            obs_new = img_new
        return obs_new

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._adjust(obs)
        self._last_obs = obs
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        return obs

    def render(self, mode='rgb_array', **kwargs):
        if mode == 'rgb_array':
            if self._last_obs is not None:
                if self._input_format == 'channels_first':
                    frame = np.transpose(self._last_obs, (1, 2, 0))
                else:
                    frame = self._last_obs
                if self._is_normalized:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)

                frame = frame[:, :, ::-1]
                return frame
            else:
                return self.env.render(mode=mode, **kwargs)
        else:
            return self.env.render(mode=mode, **kwargs)
