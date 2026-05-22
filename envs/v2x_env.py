import os, sys, io
import numpy as np
import traci
from gymnasium import Env, spaces
from envs.ns3_bridge import NS3Bridge


class V2XEnv(Env):

    # Discrete Tx power levels in dBm — NR-V2X PC5 power class 3.
    # 5 levels: [-10, 0, 10, 16, 23] ≈ [0.1, 1, 10, 40, 200] mW
    POWER_DBM = np.array([-10.0, 0.0, 10.0, 16.0, 23.0], dtype=np.float32)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg=cfg; self.N=cfg["n_vehicles"]; self.M=cfg["n_subchannels"]
        self.n_pwr=len(self.POWER_DBM); self.sumo_cfg=cfg["sumo_cfg"]
        self.max_steps=cfg.get("max_steps",200)

        self.n_actions=self.M*self.n_pwr
        self.action_space=spaces.MultiDiscrete([self.n_actions]*self.N)

        # obs per vehicle (obs_dim = 3+M+2):
        #   [0]     x/1000             normalised position [km]
        #   [1]     y/1000
        #   [2]     speed/33.3         normalised (1.0 = 120 km/h)
        #   [3..M+2] chan_quality[k]   per-vehicle EMA PDR on channel k [0,1]
        #             0=always failed, 1=always delivered on this channel.
        #             PRIVATE per vehicle — avoids herding instability.
        #   [M+3]   last_pdr           own last-step PDR [0,1]
        #   [M+4]   vehicle_class      0.0=M0 (safety), 1.0=M1 (non-safety)
        #             Allows actor to apply different strategies per class.
        self.obs_dim = 3 + self.M + 3   # +3: last_pdr + vehicle_class + vehicle_id_norm (Path A.5)   # +2: last_pdr AND vehicle_class
        self.observation_space=spaces.Box(
            low=0.,high=np.inf,shape=(self.N,self.obs_dim),dtype=np.float32)
        self.global_state_dim=self.N*self.obs_dim+self.M

        self.bridge=NS3Bridge(
            n_vehicles=self.N, port=cfg.get("ns3_port",5556),
            binary=os.path.expanduser(
                cfg.get("bridge_binary","~/v2x_thesis/ns3_bridge/v2x_bridge")),
            fc_ghz=cfg.get("fc_ghz",5.9))

        self._bridge_started=False; self.vehicle_ids=[]; self._t=0

        # Per-vehicle channel quality EMA (N × M), neutral prior 0.5
        self._chan_quality=np.full((self.N,self.M),0.5,dtype=np.float32)
        self._last_subch  =np.zeros(self.N,dtype=np.int32)
        self._last_pdr    =np.full(self.N,0.5,dtype=np.float32)
        self._ema_alpha   =cfg.get("ema_alpha",0.3)
        # ema_alpha: EMA smoothing for channel quality update
        # 0.1=slow/smooth, 0.3=balanced (default), 0.5=fast/reactive

        # ── PSM-001: M0/M1 traffic class assignment ───────────────────
        self.m0_ratio = cfg.get("m0_ratio", 0.50)
        self.m0_count = int(self.N * self.m0_ratio)
        self.m1_count = self.N - self.m0_count
        # vehicle_classes[i] = 0 → M0 (safety-critical)
        # vehicle_classes[i] = 1 → M1 (non-safety / infotainment)
        # First m0_count vehicles (by spawn order) are M0; remainder M1.
        self.vehicle_classes = np.array(
            [0]*self.m0_count + [1]*self.m1_count, dtype=np.int32)

        # ── Resource pool partitioning (Config C only) ────────────────
        self.m0_subchannel_count = cfg.get("m0_subchannel_count", 2)
        # m0_subchannels: indices [0, m0_subchannel_count) reserved for M0
        # m1_subchannels: indices [m0_subchannel_count, M) for M1
        self.m0_subchannels = list(range(self.m0_subchannel_count))
        self.m1_subchannels = list(range(self.m0_subchannel_count, self.M))
        self.demand_separation = cfg.get("demand_separation", False)
        self.M_m0 = cfg.get("m0_subchannel_count", 2)
        self.M_m1 = self.M - self.M_m0
        # demand_separation=False → Config A/B (shared pool, no hard split)
        # demand_separation=True  → Config C (hard M0/M1 pool separation)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self._bridge_started:
            self.bridge.start(); self._bridge_started=True
        if traci.isLoaded(): traci.close()

        # Suppress TraCI "Retrying in 1 seconds" polling noise.
        # sys.stdout/stderr redirect works for ALL TraCI versions —
        # no stdout/stderr keyword arg support needed in traci.start().
        _oo,_oe=sys.stdout,sys.stderr
        sys.stdout=sys.stderr=io.StringIO()
        try:
            traci.start(["sumo","-c",self.sumo_cfg,
                         "--no-warnings","--no-step-log",
                         "--log","/dev/null","--error-log","/dev/null"])
        finally:
            sys.stdout,sys.stderr=_oo,_oe

        for _ in range(60): traci.simulationStep()
        # 60 steps = 6 s at 0.1s step-length — lets vehicles spawn.
        # Increase to 100 if vehicle list is empty after reset.

        self.vehicle_ids=list(traci.vehicle.getIDList())[:self.N]
        self._t=0
        self._chan_quality=np.full((self.N,self.M),0.5,dtype=np.float32)
        self._last_subch  =np.zeros(self.N,dtype=np.int32)
        self._last_pdr    =np.full(self.N,0.5,dtype=np.float32)

        obs=self._observe()
        return obs, {"global_state":self._global_state()}

    # ------------------------------------------------------------------
    def step(self, actions:np.ndarray):
        subch,pwr_idx=actions//self.n_pwr, actions%self.n_pwr

        # Config C action masking — inactive when demand_separation=False.
        # When active, M0 vehicles are restricted to M0 subchannel pool
        # and M1 vehicles to M1 pool, enforcing hard resource separation.
        if self.demand_separation:
            for i in range(self.N):
                if self.vehicle_classes[i] == 0:
                    subch[i] = subch[i] % self.M_m0
                else:
                    subch[i] = self.M_m0 + (subch[i] % self.M_m1)
        powers_dBm=self.POWER_DBM[pwr_idx]
        traci.simulationStep(); self._t+=1
        positions=self._positions()
        phy=self.bridge.step(positions,subch,powers_dBm)

        # Update per-vehicle channel quality EMA
        for i in range(self.N):
            k=int(subch[i])
            self._chan_quality[i,k]=(
                (1.-self._ema_alpha)*self._chan_quality[i,k]
                +self._ema_alpha*float(phy["pdr"][i]))
        self._last_subch=subch.copy(); self._last_pdr=phy["pdr"].copy()

        obs=self._observe(); reward=self._reward(phy)
        # ── M0 collision diagnostic (Path A.5) ────────────────────────
        # Fraction of M0 vehicles that share a subchannel with at least one
        # other vehicle (M0 or M1) this TTI.
        m0_mask_arr = (self.vehicle_classes == 0)
        if m0_mask_arr.any():
            m0_subch = subch[m0_mask_arr]
            collisions = 0
            for i_idx, ch in enumerate(m0_subch):
                others_on_same = ((subch == ch).sum() - 1)
                if others_on_same > 0:
                    collisions += 1
            m0_collision_rate = collisions / len(m0_subch)
        else:
            m0_collision_rate = 0.0

        # Within-pool (intra-class only) collision rates — Phase C metric.
        # Under demand_separation=False, m0_collision_rate_within_pool
        # is a STRICT SUBSET of m0_collision_rate (excludes M0-M1 cross-
        # class collisions). Under demand_separation=True, the two are
        # identical by construction — useful as a sanity check.
        _m0_idx = np.where(self.vehicle_classes == 0)[0]
        _m1_idx = np.where(self.vehicle_classes == 1)[0]
        _m0_intra = sum(1 for i in _m0_idx
                        if any(j != i and subch[j] == subch[i] for j in _m0_idx))
        _m1_intra = sum(1 for i in _m1_idx
                        if any(j != i and subch[j] == subch[i] for j in _m1_idx))
        m0_collision_within_pool = _m0_intra / max(len(_m0_idx), 1)
        m1_collision_within_pool = _m1_intra / max(len(_m1_idx), 1)

        info={
            "global_state":    self._global_state(),
            "sinr_dB":         phy["sinr_dB"],
            "pdr":             phy["pdr"],
            "subchannels":     subch,
            "vehicle_classes": self.vehicle_classes,
            "m0_collision_rate": float(m0_collision_rate),
            "m0_collision_rate_within_pool": float(m0_collision_within_pool),
            "m1_collision_rate_within_pool": float(m1_collision_within_pool),
            "pdr_m0":          phy["pdr"][self.vehicle_classes==0],
            "pdr_m1":          phy["pdr"][self.vehicle_classes==1],
            "sinr_dB_m0":      phy["sinr_dB"][self.vehicle_classes==0],
            "sinr_dB_m1":      phy["sinr_dB"][self.vehicle_classes==1],
        }
        terminated=self._t>=self.max_steps
        truncated =len(traci.vehicle.getIDList())<2
        return obs,reward,terminated,truncated,info

    # ------------------------------------------------------------------
    def _positions(self):
        pos=np.zeros((self.N,2),dtype=np.float32)
        live=set(traci.vehicle.getIDList())
        for i,vid in enumerate(self.vehicle_ids):
            if vid in live:
                x,y=traci.vehicle.getPosition(vid); pos[i]=[x,y]
        return pos

    def _observe(self):
        obs=np.zeros((self.N,self.obs_dim),dtype=np.float32)
        live=set(traci.vehicle.getIDList())
        for i,vid in enumerate(self.vehicle_ids):
            if vid not in live: continue
            x,y=traci.vehicle.getPosition(vid); spd=traci.vehicle.getSpeed(vid)
            obs[i]=np.concatenate([
                [x/1000.,y/1000.,spd/33.3],
                self._chan_quality[i],        # per-vehicle quality map
                [self._last_pdr[i]],          # own last PDR
                [float(self.vehicle_classes[i]), i / max(self.N - 1, 1)]  # 0.0=M0, 1.0=M1
            ])
        return obs

    def _global_state(self):
        load=np.zeros(self.M,dtype=np.float32)
        for k in range(self.M):
            load[k]=float(np.sum(self._last_subch==k))
        # load[k]: number of vehicles on channel k (0=empty, N=all on same)
        return np.concatenate([self._observe().flatten(),load])

    def _reward(self, phy: dict) -> np.ndarray:
        """
        Class-differentiated reward — Fix A3 applied.

        M0 (safety-critical, unchanged):
            r_i = alpha * PDR_i
                  + beta  * clip(SINR_dB_i / 20, 0, 1)
                  + gamma_team * mean(PDR_M0)
            alpha=1.0  beta=0.3  gamma_team=0.5

        M1 (non-safety, FIX A3):
            r_i = delta * clip(SINR_dB_i / 20, 0, 1)         # ceiling 40 -> 20 dB
                  - eta * (1 - mean(PDR_M0))                  # cooperative penalty
            delta=0.3  eta=0.3

        Why the change: original M1 reward saturated SINR at 40 dB. Under
        shared-pool scheduling this incentivised M1 to grab the cleanest
        subchannels (its reward kept growing past where M0's saturated),
        pushing M0 onto worse channels. Lowering the ceiling to 20 dB
        removes the misaligned incentive; the eta penalty makes M1 share
        responsibility for M0 reliability under the shared-actor policy.
        """
        alpha      = self.cfg.get("alpha",      1.0)
        beta       = self.cfg.get("beta",       0.3)
        gamma_team = self.cfg.get("gamma_team", 0.5)
        delta      = self.cfg.get("delta",      0.3)
        eta        = self.cfg.get("eta",        0.3)   # NEW (Fix A3)

        pdr = phy["pdr"]
        sdB = np.clip(phy["sinr_dB"], -20., 40.)
        rewards = np.zeros(self.N, dtype=np.float32)
        m0_mask = (self.vehicle_classes == 0)
        m1_mask = (self.vehicle_classes == 1)

        m0_pdr_mean = float(pdr[m0_mask].mean()) if m0_mask.any() else 0.0

        if m0_mask.any():
            rewards[m0_mask] = (
                alpha * pdr[m0_mask]
                + beta * np.clip(sdB[m0_mask] / 20., 0., 1.)
                + gamma_team * m0_pdr_mean
            )

        if m1_mask.any():
            rewards[m1_mask] = (
                delta * np.clip(sdB[m1_mask] / 20., 0., 1.)
                - eta * (1.0 - m0_pdr_mean)
            )

        return rewards


    def close(self):
        if traci.isLoaded(): traci.close()
        self.bridge.close()
