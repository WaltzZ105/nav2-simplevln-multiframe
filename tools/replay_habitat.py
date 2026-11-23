"""Headless renderer using the simulator configuration from SimpleVLN."""

from pathlib import Path

import habitat
import numpy as np
from habitat.sims import make_sim
from PIL import Image


class HabitatRenderer:
    def __init__(self, upstream, scenes_dir, gpu):
        self.config = habitat.get_config(
            config_path=str(upstream / "DataProcess/config/vln_r2r.yaml")
        ).habitat.simulator
        self.scenes_dir = scenes_dir
        self.gpu = gpu
        self.scene = None
        self.sim = None
        self.black_frames = 0

    def start(self, root):
        self.black_frames = 0
        scene_id = Path(root["scene"])
        scene_name = scene_id.stem
        scene = self.scenes_dir / "mp3d" / scene_name / f"{scene_name}.glb"
        if not scene.is_file() or not scene.with_suffix(".navmesh").is_file():
            raise FileNotFoundError(f"MP3D scene or navmesh missing: {scene}")
        if scene != self.scene:
            self.close()
            with habitat.config.read_write(self.config):
                self.config.scene = str(scene)
                self.config.habitat_sim_v0.gpu_device_id = self.gpu
            self.sim = make_sim(id_sim=self.config.type, config=self.config)
            self.scene = scene
            if not self.sim.pathfinder.is_loaded:
                raise ValueError(f"Habitat did not load a navmesh for {scene}")
        self.restore(root["position"], root["rotation_xyzw"])

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None
            self.scene = None

    def pose(self):
        state = self.sim.get_agent_state()
        return state.position.copy(), [state.rotation.x, state.rotation.y, state.rotation.z, state.rotation.w]

    def restore(self, position, rotation):
        if not self.sim.set_agent_state(np.asarray(position, dtype=np.float32), list(rotation)):
            raise ValueError("Habitat rejected the recorded agent pose")

    def current(self):
        return self.sim.get_observations_at()["rgb"]

    def step(self, action):
        return self.sim.step(action)["rgb"]

    def save(self, frame, path):
        if frame.shape[:2] != (480, 640) or frame.dtype != np.uint8:
            raise ValueError(f"Unexpected Habitat RGB for {path}")
        self.black_frames += not bool(frame.any())
        Image.fromarray(frame).convert("RGB").save(path)
