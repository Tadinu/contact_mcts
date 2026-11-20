from __future__ import annotations

import numpy as np
import pinocchio as pin


def pin_forward_kinematics(pin_robot: pin.RobotWrapper, q: np.ndarray, frame_id: int,
                           compute_jac: bool = False) -> pin.SE3:
    """Computes forward kinematics and returns a copy of the target frame's SE3 placement."""
    q = np.asarray(q, dtype=float).ravel()
    if compute_jac:
        pin.computeJointJacobians(pin_robot.model, pin_robot.data, q)
    else:
        pin.forwardKinematics(pin_robot.model, pin_robot.data, q)
    return pin.updateFramePlacement(pin_robot.model, pin_robot.data, frame_id).copy()


def pin_oriented_jacobian(pin_robot: pin.RobotWrapper, q: np.ndarray, frame_id: int) -> np.ndarray:
    """Computes the 6xN frame Jacobian located at the frame, aligned with the world frame."""
    q = np.asarray(q, dtype=float).ravel()
    return pin.computeFrameJacobian(
        pin_robot.model,
        pin_robot.data,
        q,
        frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )


def damped_pinv(J: np.ndarray, damping: float = 1e-3) -> np.ndarray:
    """Computes the damped least-squares pseudoinverse without explicit matrix inversion."""
    JJT = J @ J.T
    np.fill_diagonal(JJT, JJT.diagonal() + damping)
    return np.linalg.solve(JJT, J).T


def pin_ik(pin_robot: pin.RobotWrapper,
           p_des: np.ndarray,
           frame_id: int,
           q_init: np.ndarray | None = None,
           max_it: int = 100,
           alpha: float = 0.1,
           threshold: float = 0.02,
           damping: float = 1e-4,
           q_null: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    """Solves 3D position inverse kinematics using damped least squares with optional null-space control."""
    model, data = pin_robot.model, pin_robot.data
    p_des = np.asarray(p_des, dtype=float).ravel()
    q = pin.neutral(model) if q_init is None else np.array(q_init, dtype=float, copy=True).ravel()
    if q_null is not None:
        q_null = np.asarray(q_null, dtype=float).ravel()

    for _ in range(max_it):
        pin_forward_kinematics(pin_robot, q, frame_id, compute_jac=True)

        err_vec = p_des - data.oMf[frame_id].translation
        if np.linalg.norm(err_vec) <= threshold:
            return q, True

        Jlin = pin.getFrameJacobian(model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
        Jlin_inv = damped_pinv(Jlin, damping=damping)
        dq = Jlin_inv @ err_vec

        if q_null is not None:
            dq_null = pin.difference(model, q, q_null)
            dq += dq_null - Jlin_inv @ (Jlin @ dq_null)

        q = pin.integrate(model, q, alpha * dq)

    pin_forward_kinematics(pin_robot, q, frame_id)
    if np.linalg.norm(p_des - data.oMf[frame_id].translation) <= threshold:
        return q, True

    return q, False
