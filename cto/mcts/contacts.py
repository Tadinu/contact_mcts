"""
CasADi port of cto/mcts/contacts.py.

Contacts are built on a casadi.Opti instance. Anything that changes between
solves (location, force, surface, orientation, simplices) is an Opti parameter,
so the problem is built once and only `opti.set_value` is called on update.
"""
from types import SimpleNamespace

import casadi as ca
import numpy as np

from cto.constraints import friction_cone, sliding_friction, contact_surface

_OFF = 1.0  # penalty on the (unused) local force of switched-off contacts
_REG = 1e-7  # keeps auxiliary variables well posed (negligible next to the 1e4..1e6 weights)


def _fz0(params):
    """Initial normal force guess: strictly inside the cone, order of the force scale."""
    return 0.1 * float(np.max(np.abs(np.asarray(params.max_force, dtype=float))))


# --------------------------------------------------------------------------- #
# Contacts
# --------------------------------------------------------------------------- #
class EnvironmentContact(object):
    """Fixed body-frame location, variable force, optional sliding friction."""

    def __init__(self, location, orientation, params, type="sticking", sliding_direction=None):
        self.type = "environment_" + type
        self.location = np.asarray(location, dtype=float)
        self.orientation = np.asarray(orientation, dtype=float)
        self.params = params
        self.kind = type
        self.sliding_direction = sliding_direction

    def build(self, opti):
        """Add this contact to `opti`; returns an object with .force .torque .cost"""
        # local-frame force is the variable; body force = R f_local  (R orthogonal => f_local = R^T force)
        f_local = opti.variable(3)
        opti.set_initial(f_local, [0.0, 0.0, _fz0(self.params)])
        force = self.orientation @ f_local
        if self.kind == "sticking":
            friction_cone(opti, f_local, self.params.environment_friction)
        elif self.kind == "sliding":
            sliding_friction(opti, f_local, self.sliding_direction, self.params.environment_friction)

        torque = ca.cross(self.location, force)
        cost = 1e-1 * ca.sumsqr(force)
        return SimpleNamespace(type=self.type, force=force, torque=torque, cost=cost,
                               location=self.location, f_local=f_local)


class FixedForceContact(object):
    """Force is a parameter; location is a decision variable on the contact surface.

    Sticking is built into the parametrisation (no equality rows that are 0 == 0 when
    not sticking, which made the problem over-determined for IPOPT):

        w = s * w_prev + (1 - s) * u,      u on the simplex

    s = 1 (sticking): w copies the previous weights;  s = 0: w = u is free.
    A convex combination of simplex points stays on the simplex, so only u needs
    u >= 0, sum(u) == 1.
    """

    def __init__(self, opti, params, prev=None, sticking=None):
        n_v = len(params.simplices[0])
        self.force_param = opti.parameter(3)
        self.simplices = opti.parameter(n_v, 3)  # zeros when surface == 0 -> location = 0
        self.in_contact = opti.parameter()  # kept for API compatibility
        self.force = np.zeros(3)
        self.surface = 0

        self.aux_weights = opti.variable(n_v)  # u
        opti.set_initial(self.aux_weights, np.ones(n_v) / n_v)  # centre of the simplex
        opti.subject_to(self.aux_weights >= 0)
        opti.subject_to(ca.sum1(self.aux_weights) == 1)

        if prev is None or sticking is None:
            self.location_weights = self.aux_weights
        else:
            self.location_weights = sticking * prev.location_weights + (1 - sticking) * self.aux_weights

        self.location = self.simplices.T @ self.location_weights
        self.torque = ca.cross(self.location, self.force_param)  # = -skew(f) @ r
        self.cost = ca.sumsqr(self.location) + _REG * ca.sumsqr(self.aux_weights)

    def set_values(self, opti, force, surface, params):
        self.force = np.asarray(force, dtype=float)
        self.surface = surface
        opti.set_value(self.force_param, self.force)
        n_v = len(params.simplices[0])
        if surface == 0:
            opti.set_value(self.in_contact, 0.0)
            opti.set_value(self.simplices, np.zeros((n_v, 3)))
        else:
            opti.set_value(self.in_contact, 1.0)
            opti.set_value(self.simplices, params.simplices[surface - 1])


class FixedLocationContact(object):
    """Location is a parameter; force is a decision variable inside the friction cone."""

    def __init__(self, opti, params):
        self.location = np.zeros(3)
        self.surface = 0
        self.r = opti.parameter(3)
        self.rot = opti.parameter(3, 3)  # contact->body rotation; zeros when surface == 0
        self.in_contact = opti.parameter()

        # f_local is the force in the contact frame; the original uses f_local = rot^T force,
        # i.e. force = rot f_local. With rot = 0 the force is exactly 0 (replaces force == in_contact*aux).
        self.f_local = opti.variable(3)
        self.force = self.rot @ self.f_local
        self.torque = ca.cross(self.r, self.force)  # = skew(r) @ force

        # bounded force (body frame, elementwise)
        opti.subject_to(opti.bounded(-params.max_force, self.force, params.max_force))
        # fc must be bounded for ALL contacts: when a contact is off (rot = 0) the box above
        # does not touch fc at all, leaving a flat, unbounded direction inside the cone.
        # |force_i| <= M for a rotated force implies |fc_i| <= sqrt(3) M, so this is no restriction.
        fc_max = np.sqrt(3.0) * np.asarray(params.max_force, dtype=float)
        opti.subject_to(opti.bounded(-fc_max, self.f_local, fc_max))
        # friction cone on the local force
        # Optional planning margin: params.friction_coeff_plan < params.friction_coeff plans with a
        # smaller friction coefficient than the simulator/robot has (default: no margin).
        mu_plan = params.get("friction_coeff_plan", params.friction_coeff)
        friction_cone(opti, self.f_local, mu_plan)

        # off-contact: strongly convex penalty pins fc to 0 (no effect when in contact)
        self.cost = (ca.sumsqr(self.force) + _REG * ca.sumsqr(self.f_local)
                     + (1 - self.in_contact) * _OFF * ca.sumsqr(self.f_local))
        # start strictly inside the cone, not at its apex
        opti.set_initial(self.f_local, [0.0, 0.0, _fz0(params)])

    def set_values(self, opti, location, surface, params):
        self.location = np.asarray(location, dtype=float)
        self.surface = surface
        opti.set_value(self.r, self.location)
        if surface == 0:
            opti.set_value(self.in_contact, 0.0)
            opti.set_value(self.rot, np.zeros((3, 3)))
        else:
            opti.set_value(self.in_contact, 1.0)
            opti.set_value(self.rot, params.contact_frame_orientation[surface - 1])
