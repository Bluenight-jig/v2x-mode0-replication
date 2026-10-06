"""
v2x_env_arms.py -- capability ablations for R2-5 / R3-e (aa-achievement.md §7c). A NEW file.

Two environments that override ONE component of each vehicle's action and then call the
unchanged parent step. Simulator, bridge, rewards and observations are identical to the
environment every other run uses.

  Arm S (V2XEnvArmS): channel choice as learned; power FIXED at 23 dBm (the top level).
  Arm P (V2XEnvArmP): power as learned; channel drawn uniformly at random every interval.

Mode 0c learns both components. Mode 0c against Arm S isolates what learned power adds;
Mode 0c against Arm P isolates what learned channel choice adds.

The actor still outputs M x 5 actions. The overridden component has no effect on the outcome,
so its gradient is pure noise and its entropy stays high -- the logged entropy includes it.
It is therefore not a learned behaviour in either arm, and the paper should say so.
"""
import numpy as np

from envs.v2x_env import V2XEnv

FIXED_POWER_DBM = 23.0


class V2XEnvArmS(V2XEnv):
    """Learned channel, fixed 23 dBm power."""

    def __init__(self, cfg):
        super().__init__(cfg)
        idx = np.flatnonzero(np.isclose(self.POWER_DBM, FIXED_POWER_DBM))
        if len(idx) != 1:
            raise ValueError(f"{FIXED_POWER_DBM} dBm is not exactly one of the power levels {self.POWER_DBM}")
        self._fixed_pwr = int(idx[0])

    def step(self, actions):
        a = np.asarray(actions).astype(np.int64)
        return super().step((a // self.n_pwr) * self.n_pwr + self._fixed_pwr)


class V2XEnvArmP(V2XEnv):
    """Learned power, uniformly random channel every interval."""

    def step(self, actions):
        a = np.asarray(actions).astype(np.int64)
        subch = np.random.randint(0, self.M, size=a.shape)
        return super().step(subch * self.n_pwr + (a % self.n_pwr))
