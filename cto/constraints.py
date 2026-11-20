import casadi as ca
import numpy as np


# --------------------------------------------------------------------------- #
# Constraint helpers (replace cto.constraints for CasADi)
# These mirror cto/constraints.py (verified against the original source).
# --------------------------------------------------------------------------- #
def friction_cone(opti, f_local, mu):
    """Friction pyramid, identical to cto.constraints.friction_cone:
    f_z >= 0, |f_x| <= mu f_z, |f_y| <= mu f_z   (all linear => the problems are QPs)."""
    fx, fy, fz = f_local[0], f_local[1], f_local[2]
    opti.subject_to(fz >= 0)
    opti.subject_to(fx - mu * fz <= 0)
    opti.subject_to(-fx - mu * fz <= 0)
    opti.subject_to(fy - mu * fz <= 0)
    opti.subject_to(-fy - mu * fz <= 0)


def sliding_friction(opti, f_local, sliding_direction, mu):
    """f_t = -mu * f_n * d,  f_n >= 0  (Coulomb friction opposing the sliding direction)."""
    d = np.asarray(sliding_direction, dtype=float).ravel()[:2]
    # trajectory.py passes the contact-point velocity (not unit length); the original normalises it too.
    norm = np.linalg.norm(d)
    d = d / norm if norm > 0 else d
    opti.subject_to(f_local[2] >= 0)
    opti.subject_to(f_local[:2] == -mu * f_local[2] * d)


def contact_surface(opti, location, weights, simplices):
    """location is a convex combination of the surface vertices."""
    opti.subject_to(weights >= 0)
    opti.subject_to(ca.sum1(weights) == 1)
    opti.subject_to(location == simplices.T @ weights)
