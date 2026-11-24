import pinocchio as pin
import numpy as np

from cto.kinematics import pin_jac


class ImpedanceController:
    def __init__(self, P, D, pin_robot, frame_id):
        # setup control parameters
        self.P = P
        self.D = D
        self.pin_robot = pin_robot
        self.frame_id = frame_id

    def compute_torque(self, q, dq, x_ref, dx_ref, f_ref):
        J = pin_jac(self.pin_robot, self.frame_id, q)
        Jov = J[:3]
        x = self.pin_robot.data.oMf[self.frame_id].translation
        dx = Jov @ dq
        force_des = self.P @ (x_ref - x) + self.D @ (dx_ref - dx) + f_ref
        return Jov.T @ force_des


class NullspaceImpedanceController(ImpedanceController):
    """Cartesian impedance + feed-forward force at the tool tip, plus a null-space posture task.

    Same call signature as cto.controllers.ImpedanceController, so it is a drop-in replacement:
        tau = J^T [ P (x_ref - x) + D (dx_ref - dx) + f_ref ]  +  N^T [ Kn (q_null - q) - Dn dq ]
    with N^T = I - J^T (J^+)^T projecting the posture torque into the null space of the 3D tip task.
    Gravity compensation is done by the environment (qfrc_bias), not here.
    """

    def __init__(self, P, D, pin_robot, frame_id, q_null, Kn=5.0, Dn=1.0):
        super().__init__(P, D, pin_robot, frame_id)
        self.q_null = np.asarray(q_null, dtype=float)
        self.Kn, self.Dn = Kn, Dn

    def compute_torque(self, q, dq, x_ref, dx_ref, f_ref):
        # Jov: 3 x nv, world-aligned
        J = pin_jac(self.pin_robot, self.frame_id, q)
        J = J[:3]

        x = self.pin_robot.data.oMf[self.frame_id].translation
        dx = J @ dq
        force_des = self.P @ (x_ref - x) + self.D @ (dx_ref - dx) + f_ref
        tau = J.T @ force_des

        N_T = np.eye(len(q)) - J.T @ np.linalg.pinv(J.T)  # null-space projector (transposed)
        tau += N_T @ (self.Kn * (self.q_null - q) - self.Dn * dq)
        return tau
