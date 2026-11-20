"""
CasADi port of the biconvex force/location problems.

Differences from the cvxpy version:
  * Each sub-problem is built ONCE as a CasADi Opti problem; `update` only
    calls opti.set_value on parameters (no re-canonicalisation).
  * Exact derivatives via algorithmic differentiation, solved with IPOPT
    (interior point, handles the friction cone natively, no SOC reformulation).
  * Warm starting between calls (primal values of the previous solve).
  * Contact classes are defined here, since cvxpy-based ones can't be reused.
    ASSUMPTIONS are marked with "ASSUMPTION".
"""
import casadi as ca
from dotmap import DotMap
import numpy as np
import pinocchio as pin

from cto.mcts.contacts import FixedForceContact, FixedLocationContact
from cto.trajectory import integrate_trajectory

# --------------------------------------------------------------------------- #
# Solver helper
# --------------------------------------------------------------------------- #
# All sub-problems are convex QPs (friction pyramid, linear constraints).
#   "ipopt" (default): reliable (30/30 random plans), same optimum as the cvxpy version
#   "highs"          : faster, but failed on 2/30 plans in testing -> experimental
# (daqp / qpoases / proxqp / osqp did not converge reliably through CasADi's SQP wrapper.)
# QP solvers are run through CasADi's sqpmethod. Select with params.qp_solver = "..."
# or by changing DEFAULT_SOLVER.
DEFAULT_SOLVER = "ipopt"


def _configure_solver(opti, verbose, solver=DEFAULT_SOLVER):
    if solver == "ipopt":
        opti.solver("ipopt", {"print_time": False,
                              "expand": True,  # MX -> SX: much faster function evaluation
                              "ipopt.hessian_constant": "yes",  # QP: Hessian/Jacobians do not depend on x
                              "ipopt.jac_c_constant": "yes",
                              "ipopt.jac_d_constant": "yes",
                              "ipopt.print_level": 5 if verbose else 0,
                              "ipopt.sb": "yes",
                              "ipopt.tol": 1e-8,
                              "ipopt.max_iter": 500,
                              "ipopt.acceptable_tol": 1e-5,
                              "ipopt.acceptable_iter": 10})
        # for linear solver speed, install HSL and add: "ipopt.linear_solver": "ma57"
    else:
        qp_opts = {"osqp": {"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iter": 100000, "verbose": False}}
        opti.solver("sqpmethod", {"qpsol": solver,
                                  "qpsol_options": qp_opts.get(solver, {}),
                                  "max_iter": 5,
                                  "convexify_strategy": "none",
                                  "print_time": False,
                                  "print_iteration": bool(verbose),
                                  "print_header": bool(verbose),
                                  "print_status": bool(verbose)})


class _Warm(object):
    """Keeps the last primal solution to warm-start the next solve, and remembers
    how the solver was configured (opti.solver(...) must NOT be re-called every
    solve: it throws away CasADi's cached derivative functions)."""

    def __init__(self):
        self.x = None
        self.cfg = None

    def apply(self, opti):
        if self.x is not None:
            opti.set_initial(opti.x, self.x)

    def store(self, opti, sol):
        self.x = sol.value(opti.x)


# For a QP, sqpmethod stops with "Search_Direction_Becomes_Too_Small" once the (zero) step
# shows it has converged, so that status counts as success there.
_OK_STATUS = {"Solve_Succeeded", "Solved_To_Acceptable_Level",
              "Search_Direction_Becomes_Too_Small", "Success"}


class _DebugSol(object):
    """Mimics OptiSol.value() using opti.debug (last iterate) after an accepted status."""

    def __init__(self, opti):
        self.opti = opti

    def value(self, expr):
        return self.opti.debug.value(expr)


def _solve(opti, warm, verbose, name, solver=DEFAULT_SOLVER):
    """Solve and raise a descriptive error (sub-problem name + solver status)."""
    if warm.cfg != (verbose, solver):
        _configure_solver(opti, verbose, solver)
        warm.cfg = (verbose, solver)
    warm.apply(opti)
    try:
        sol = opti.solve()
    except RuntimeError:
        status = opti.stats().get("return_status", "unknown")
        if status not in _OK_STATUS:
            raise RuntimeError("%s failed: solver status '%s'" % (name, status))
        sol = _DebugSol(opti)  # converged, but CasADi reports it as an exception
    warm.store(opti, sol)
    return sol


# --------------------------------------------------------------------------- #
# Sub-problems
# --------------------------------------------------------------------------- #
class ForceProblem(object):
    def __init__(self, locations, contact_plan, params):
        self.horizon = len(contact_plan)
        self.contact_plan = contact_plan
        self.params = params
        self.locations = locations
        self.warm = _Warm()

    def setup(self):
        self.opti = ca.Opti()
        self.contacts = []  # numeric-force contacts
        self.env = []
        costs = []

        for i in range(self.horizon):
            curr_mode, n = self.contact_plan[i]
            curr_contacts, wrenches = [], []

            for _ in curr_mode:
                flc = FixedLocationContact(self.opti, self.params)
                curr_contacts.append(flc)
                costs.append(flc.cost)
                wrenches.append((flc.force, flc.torque))

            env = []
            for ec in self.params.environment_contacts[n]:
                b = ec.build(self.opti)
                env.append(b)
                costs.append(b.cost)
                wrenches.append((b.force, b.torque))
            self.env.append(env)
            self.contacts.append(curr_contacts)

            total_force = sum(w[0] for w in wrenches)
            total_torque = sum(w[1] for w in wrenches)
            costs.append(1e4 * ca.sumsqr(total_force - self.params.traj_desired.total_force[n]))
            costs.append(1e6 * ca.sumsqr(total_torque - self.params.traj_desired.total_torque[n]))

        self.opti.minimize(sum(costs))
        self.update(self.contact_plan, self.locations)

    def update(self, contact_plan, locations):
        self.contact_plan = contact_plan
        self.locations = locations
        opti = self.opti

        for i in range(self.horizon):
            curr_mode, _ = self.contact_plan[i]
            for c, s in enumerate(curr_mode):
                self.contacts[i][c].set_values(opti, self.locations[i][c], s, self.params)

    def solve(self, verbose=False):
        sol = _solve(self.opti, self.warm, verbose, "ForceProblem", self.params.get("qp_solver", DEFAULT_SOLVER))
        self._sol = sol
        return float(sol.value(self.opti.f)), self.opti

    def get_forces(self):
        forces = []
        for i in range(self.horizon):
            force = [np.asarray(self._sol.value(c.force)).reshape(3) for c in self.contacts[i]]
            force += [np.asarray(self._sol.value(b.force)).reshape(3) for b in self.env[i]]
            forces.append(force)
        return forces


class LocationProblem(object):
    def __init__(self, forces, contact_plan, params):
        self.horizon = len(contact_plan)
        self.contact_plan = contact_plan
        self.params = params
        self.forces = forces
        self.sticking_value = np.zeros((self.horizon, params.n_contacts))
        self.warm = _Warm()

    def set_sticking_contact_flags(self):
        self.sticking_value[:] = 0.0
        for i in range(1, self.horizon):
            curr_mode, _ = self.contact_plan[i]
            prev_mode, _ = self.contact_plan[i - 1]
            for c, s in enumerate(curr_mode):
                self.sticking_value[i, c] = 1.0 if (s == prev_mode[c] and s != 0) else 0.0

    def setup(self):
        self.opti = ca.Opti()
        self.sticking = self.opti.parameter(self.horizon, self.params.n_contacts)
        self.environment_torque = self.opti.parameter(self.horizon, 3)
        self.opti.set_value(self.environment_torque, np.zeros((self.horizon, 3)))
        self.contacts = []
        costs = []

        for i in range(self.horizon):
            curr_mode, n = self.contact_plan[i]
            curr_contacts = []

            for c, _ in enumerate(curr_mode):
                prev = self.contacts[i - 1][c] if i >= 1 else None
                ffc = FixedForceContact(self.opti, self.params, prev,
                                        self.sticking[i, c] if i >= 1 else None)
                curr_contacts.append(ffc)
                costs.append(ffc.cost)
            self.contacts.append(curr_contacts)

            total_torque = sum(f.torque for f in curr_contacts) + self.environment_torque[i, :].T
            costs.append(1e6 * ca.sumsqr(total_torque - self.params.traj_desired.total_torque[n]))

        self.opti.minimize(sum(costs))
        self.update(self.contact_plan, self.forces)

    def update(self, contact_plan, forces):
        self.contact_plan = contact_plan
        self.forces = forces
        self.set_sticking_contact_flags()
        self.opti.set_value(self.sticking, self.sticking_value)

        for i in range(self.horizon):
            curr_mode, _ = self.contact_plan[i]
            for c, s in enumerate(curr_mode):
                self.contacts[i][c].set_values(self.opti, self.forces[i][c], s, self.params)

    def solve(self, verbose=False):
        sol = _solve(self.opti, self.warm, verbose, "LocationProblem", self.params.get("qp_solver", DEFAULT_SOLVER))
        self._sol = sol
        return float(sol.value(self.opti.f)), self.opti

    def get_locations(self):
        locations = []
        for i in range(self.horizon):
            location = [np.asarray(self._sol.value(c.location)).reshape(3) for c in self.contacts[i]]
            # NOTE: kept from original -- indexes environment_contacts by i, not by n
            location += [ec.location for ec in self.params.environment_contacts[i]]
            locations.append(location)
        return locations


# --------------------------------------------------------------------------- #
# Alternating (biconvex) solver -- unchanged logic
# --------------------------------------------------------------------------- #
class BiconvexProblem(object):
    """Alternates force (locations fixed) and location (forces fixed) solves.

    max_it >= 2 is required for accurate wrench balance: with a single pass the forces are
    computed for face-centre locations and the location step cannot fully repair the torque
    (~1e-3 N m residual, which spins a 0.5 kg / 8e-4 kg m^2 cube by radians). The second
    pass feeds the optimised locations back into the force problem (see solve()).
    """

    def __init__(self, contact_plan, params, max_it=2):
        self.contact_plan = contact_plan
        self.params = params
        self.locations = initialize_locations(self.contact_plan, self.params)
        self.forces = initialize_forces(self.contact_plan, self.params)
        self.max_it = max_it

        self.fp = ForceProblem(self.locations, self.contact_plan, self.params)
        self.lp = LocationProblem(self.forces, self.contact_plan, self.params)

    def setup(self):
        self.fp.setup()
        self.lp.setup()

    def update(self, contact_plan):
        self.contact_plan = contact_plan
        self.locations = initialize_locations(self.contact_plan, self.params)
        self.forces = initialize_forces(self.contact_plan, self.params)
        self.fp.update(contact_plan, self.locations)
        self.lp.update(contact_plan, self.forces)
        # different contact plan: last solution is not a good initial guess
        self.fp.warm.x = None
        self.lp.warm.x = None

    def solve(self, verbose=True):
        last_cost = 0
        for _ in range(self.max_it):
            fp_cost, _ = self.fp.solve(verbose=verbose)
            self.forces = self.fp.get_forces()
            self.lp.update(self.contact_plan, self.forces)
            lp_cost, _ = self.lp.solve(verbose=verbose)
            self.locations = self.lp.get_locations()
            self.fp.update(self.contact_plan, self.locations)  # feed optimised locations back into the force problem
            last_cost = fp_cost + lp_cost

        sol = DotMap()
        sol.cost = last_cost
        sol.force_residual, sol.torque_residual = self.wrench_residuals()
        sol.forces = self.forces
        sol.locations = self.locations
        sol.forces_world = transform_forces_world(self.forces, self.params)
        sol.locations_world = transform_locations_world(self.locations, self.params)
        sol.contacts = self.lp.contacts
        return sol

    def wrench_residuals(self):
        """Per-step ||sum f - f_des|| and ||sum r x f - tau_des|| for the current solution.
        Large values mean the contact plan cannot produce the desired motion."""
        rf, rt = [], []
        for i in range(len(self.contact_plan)):
            n = self.contact_plan[i][1]
            f = np.sum(self.forces[i], axis=0)
            t = np.sum(np.cross(self.locations[i], self.forces[i]), axis=0)
            rf.append(np.linalg.norm(f - self.params.traj_desired.total_force[n]))
            rt.append(np.linalg.norm(t - self.params.traj_desired.total_torque[n]))
        return np.array(rf), np.array(rt)


def initialize_locations(contact_plan, params):
    horizon = params.horizon
    locations = []
    for i in range(horizon):
        location = []
        mode = contact_plan[i][0]
        for s in mode:
            if s == 0:
                location.append(np.zeros(3))
            else:
                location.append(np.mean(params.simplices[s - 1], axis=0))
        location += [ec.location for ec in params.environment_contacts[i]]
        locations.append(location)
    return locations


def initialize_forces(contact_plan, params):
    horizon = params.horizon
    forces = []
    for i in range(horizon):
        force = [np.zeros(3) for _ in contact_plan[i][0]]
        force += [np.zeros(3) for _ in params.environment_contacts[i]]
        forces.append(force)
    return forces


def transform_locations_world(locations_body, params):
    q = params.traj_desired.q
    locations_world = []
    for n in range(params.horizon):
        curr_pose = pin.XYZQUATToSE3(q[n])
        p, R = curr_pose.translation, curr_pose.rotation
        locations_world.append([p + R @ r for r in locations_body[n]])
    return locations_world


def transform_forces_world(forces_body, params):
    q = params.traj_desired.q
    forces_world = []
    for n in range(params.horizon):
        R = pin.XYZQUATToSE3(q[n]).rotation
        forces_world.append([R @ f for f in forces_body[n]])
    return forces_world


def integrate_solution(sol, params, idx_start=None, idx_end=None):
    if idx_start is None:
        idx_start = 0
    if idx_end is None:
        idx_end = len(params.traj_desired.q)

    traj_length = idx_end - idx_start
    total_force = np.zeros((traj_length, 3))
    total_torque = np.zeros((traj_length, 3))

    pose_start = params.traj_desired.q[idx_start]
    forces = sol.forces[idx_start:idx_end]
    locations = sol.locations[idx_start:idx_end]

    for n in range(traj_length):
        total_force[n] = np.sum(forces[n], axis=0)
        total_torque[n] = np.sum(np.cross(locations[n], forces[n]), axis=0)

    return integrate_trajectory(pose_start, idx_start, idx_end,
                                total_force, total_torque, params)
