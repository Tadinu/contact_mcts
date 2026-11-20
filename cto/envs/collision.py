import numpy as np
import mujoco as mj


class CollisionDetector:
    """in_collision(q, margin): any robot link/tool closer than `margin` to the box, or q outside the joint limits."""

    def __init__(self, env, robot):
        self.env = env
        self.robot = robot

    def compute_distances(self, q, max_distance=1.0):
        self.robot.set_configuration(q)
        fromto = np.zeros(6)
        return np.array([mj.mj_geomDistance(self.env.model, self.env.data, g, self.env.box_geom_id,
                                            max_distance, fromto) for g in self.robot.geom_ids])

    def in_collision(self, q, margin=0):
        if self.robot.violates_limits(q):
            return True
        ds = self.compute_distances(q, max_distance=max(margin, 0.0))
        return bool((ds < margin).any())
