"""
ns3_bridge.py — Python ZMQ REQ client for the v2x_bridge subprocess.
"""
import os, time, signal, json, subprocess
from typing import Optional
import zmq, numpy as np


class NS3Bridge:
    def __init__(self,
                 n_vehicles: int   = 10,
                 port:       int   = 5556,
                 binary:     str   = os.path.expanduser(
                                 "~/v2x_thesis/ns3_bridge/v2x_bridge"),
                 fc_ghz:     float = 5.9,
                 timeout_ms: int   = 8000):
        # port      : change if 5556 is in use
        # timeout_ms: 8 s — increase on slow machines
        self.n_vehicles=n_vehicles; self.port=port
        self.binary=binary; self.fc_ghz=fc_ghz; self.timeout_ms=timeout_ms
        self._proc=None; self._ctx=None; self._socket=None

    def start(self):
        cmd=[self.binary,f"--port={self.port}",f"--fcGHz={self.fc_ghz}"]
        self._proc=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        time.sleep(1.5)  # wait for bridge to bind socket
        self._ctx=zmq.Context()
        self._socket=self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO,self.timeout_ms)
        self._socket.connect(f"tcp://localhost:{self.port}")
        print(f"[NS3Bridge] Connected tcp://localhost:{self.port}")

    def step(self, positions, subchannels, powers_dBm):
        payload={"vehicles":positions.tolist(),
                 "subchannels":subchannels.tolist(),
                 "powers_dBm":powers_dBm.tolist()}
        self._socket.send_string(json.dumps(payload))
        try:
            reply=json.loads(self._socket.recv_string())
        except zmq.Again:
            print("[NS3Bridge] WARNING: timeout — using fallback values")
            reply={"sinr_lin":[1.0]*self.n_vehicles,
                   "sinr_dB": [0.0]*self.n_vehicles,
                   "pdr":     [0.5]*self.n_vehicles}
        return {k:np.array(v,dtype=np.float32) for k,v in reply.items()}

    def close(self):
        if self._socket: self._socket.close()
        if self._ctx:    self._ctx.term()
        if self._proc and self._proc.poll() is None:
            self._proc.send_signal(signal.SIGTERM); self._proc.wait(timeout=5)
        print("[NS3Bridge] Shutdown complete.")
