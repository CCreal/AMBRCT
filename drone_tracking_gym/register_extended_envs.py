# drone_tracking_gym/register_extended_envs.py
from gym.envs.registration import register
from drone_tracking_gym.envs.drone_tracking_ptz_env import DroneTrackingEnv_PTZ
from drone_tracking_gym.envs.lighting_variants import  HueShiftEnv

# ----------------------  PTZ ----------------------
register(
    id="blendtorch-drone_tracking_ptz-v0",
    entry_point="drone_tracking_gym.envs.drone_tracking_ptz_env:DroneTrackingEnv_PTZ",
)



def _make_ptz_hue60(real_time=False):
    return HueShiftEnv(DroneTrackingEnv_PTZ(real_time), hue_shift=60)


register(
    id="blendtorch-drone_tracking_ptz_hue60-v0",
    entry_point="drone_tracking_gym.register_extended_envs:_make_ptz_hue60",
)

def _make_ptz_hue60m(real_time=False):
    return HueShiftEnv(DroneTrackingEnv_PTZ(real_time), hue_shift=-60)

register(
    id="blendtorch-drone_tracking_ptz_hue60m-v0",
    entry_point="drone_tracking_gym.register_extended_envs:_make_ptz_hue60m",
)
