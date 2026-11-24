from robot_properties_nyu_finger.config import NYUFingerConfig, NYUFingerDoubleConfig0, NYUFingerDoubleConfig1
from cto.envs.multi_articulators import mj_spec_scale_body, Articulator, MultiArticulatorEnv

import numpy as np


class Finger(Articulator):
    def __init__(self, env, config: NYUFingerConfig, prefix: str):
        super().__init__(env, config, prefix)


class FingerDoubleAndBox(MultiArticulatorEnv):

    def __init__(self, params, box_pose, ee_init=None, render=False, dt=0.001):
        super().__init__(articulator_class=Finger, params=params,
                         configs=[NYUFingerDoubleConfig0(idx=0),
                                  NYUFingerDoubleConfig1(idx=1)],
                         box_pose=box_pose,
                         ee_init=ee_init, render=render, dt=dt)

        # Collision geoms of the moving links (upper / middle / lower)
        for finger, prefix, i in zip(self.articulators, self.prefixes, (0, 1)):
            for link in ("upper", "middle", "lower"):
                finger_link = self.scene_spec.body(f"{prefix}finger{i}_{link}_link")
                finger.geom_ids.extend([geom.id for geom in finger_link.geoms])
