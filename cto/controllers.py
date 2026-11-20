import pinocchio as pin
import numpy as np


class ImpedanceController():
    def __init__(self, P, D, pin_robot, frame_id):
        # setup control parameters
        self.P = P
        self.D = D
        self.pin_robot = pin_robot
        self.frame_id = frame_id

    def compute_torque(self, q, dq, x_ref, dx_ref, f_ref):
        # 1. Compute Jacobians (also updating kinematics) and update frame placements for the current q
        pin.computeJointJacobians(self.pin_robot.model, self.pin_robot.data, q)
        pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)

        # 2. Directly get the world-aligned Jacobian (replaces your adjoint math)
        J = pin.getFrameJacobian(self.pin_robot.model, self.pin_robot.data, self.frame_id,
                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        Jov = J[:3, :]  # Linear velocity part

        # 3. Compute Cartesian error and desired joint torques
        x = self.pin_robot.data.oMf[self.frame_id].translation
        dx = Jov @ dq
        force_des = self.P @ (x_ref - x) + self.D @ (dx_ref - dx) + f_ref

        return Jov.T @ force_des
