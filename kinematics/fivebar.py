"""Kinematics of a symmetric planar five-bar robot.

Motor 1 is at (-d/2, 0), motor 2 at (d/2, 0). Angles are in radians,
zero points north (+Y), and positive rotation is clockwise.
Each chain has motor-to-elbow length l1 and elbow-to-tip length l2.
Use consistent length units. Positions and angles are NumPy arrays.

FK defaults to the highest tip; IK defaults to the widest elbow spacing
on that FK branch. Equal-height tips are ordered by X (rightmost is
"upper"). Joint limits and link collisions are not modeled.
"""

import math

import numpy as np


_RTOL = 1e-12


def _validate_lengths(*lengths):
    if not all(math.isfinite(v) and v > 0 for v in lengths):
        raise ValueError("Link lengths and motor spacing must be finite and positive.")


def _validate_branch(branch):
    if branch not in ("upper", "lower"):
        raise ValueError("branch must be 'upper' or 'lower'.")


def elbow_positions(q1, q2, l1, d):
    """Return the left and right elbow coordinates as a (2, 2) array."""
    _validate_lengths(l1, d)
    if not all(math.isfinite(q) for q in (q1, q2)):
        raise ValueError("Motor angles must be finite.")
    return np.array(
        [
            [-d / 2 + l1 * math.sin(q1), l1 * math.cos(q1)],
            [d / 2 + l1 * math.sin(q2), l1 * math.cos(q2)],
        ]
    )


def _circle_intersections(c0, r0, c1, r1):
    """Return isolated circle intersections (one at tangency)."""
    delta = c1 - c0
    distance = math.hypot(*delta)
    scale = max(r0, r1, distance)
    # Normalize so the tolerance is independent of the chosen length unit.
    a, b, s = r0 / scale, r1 / scale, distance / scale
    if s <= _RTOL:
        if abs(a - b) <= _RTOL:
            raise ValueError("Coincident circles: infinitely many configurations.")
        raise ValueError("Unreachable configuration: concentric circles.")
    if s > a + b + _RTOL or s < abs(a - b) - _RTOL:
        raise ValueError("Unreachable configuration: links cannot meet.")

    along = (a * a - b * b + s * s) / (2 * s)
    height = math.sqrt(max(0.0, a * a - along * along))
    direction = delta / distance
    center = c0 + scale * along * direction
    if height <= _RTOL:
        return (center,)
    offset = scale * height * np.array([-direction[1], direction[0]])
    return (center + offset, center - offset)


def fk(q1, q2, l1, l2, d, *, branch="upper"):
    """Return [X, Y] for motor angles q1, q2.

    branch="upper" selects the highest-Y tip, "lower" the lowest.
    At equal Y, "upper" selects the largest X. Tangency has one solution.
    Raise ValueError for invalid geometry, unreachable angles, or coincident
    elbows (the tip then has infinitely many possible positions).
    """
    _validate_lengths(l1, l2, d)
    _validate_branch(branch)
    e1, e2 = elbow_positions(q1, q2, l1, d)
    tips = _circle_intersections(e1, l2, e2, l2)
    select = max if branch == "upper" else min
    return select(tips, key=lambda p: (p[1], p[0])).copy()


def ik_solutions(x, y, l1, l2, d, *, branch="upper"):
    """Return all isolated [q1, q2] solutions on the requested FK branch.

    Solutions are sorted by decreasing distance between elbows. Angles are
    wrapped to [-pi, pi). There are at most four solutions. Raise ValueError
    if the target is unreachable, no solution uses this FK branch, or an
    entire chain has infinitely many solutions (tip at its motor, l1=l2).
    Coincident-elbow configurations are excluded because FK is undefined.
    """
    _validate_lengths(l1, l2, d)
    _validate_branch(branch)
    if not all(math.isfinite(v) for v in (x, y)):
        raise ValueError("Target coordinates must be finite.")

    tip = np.array([x, y], dtype=float)
    motors = np.array([[-d / 2, 0.0], [d / 2, 0.0]])
    candidates = [_circle_intersections(motor, l1, tip, l2) for motor in motors]
    solutions = []
    tolerance = _RTOL * max(l1, l2, d)
    for e1 in candidates[0]:
        for e2 in candidates[1]:
            angles = np.array(
                [
                    math.atan2(elbow[0] - motor[0], elbow[1] - motor[1])
                    for elbow, motor in zip((e1, e2), motors)
                ]
            )
            angles = (angles + math.pi) % (2 * math.pi) - math.pi
            # Use the same assembly choice as FK, so fk(*ik(...)) agrees.
            try:
                selected_tip = fk(*angles, l1, l2, d, branch=branch)
            except ValueError:
                continue
            if math.hypot(*(selected_tip - tip)) <= 10 * tolerance:
                spacing = math.hypot(*(e2 - e1))
                solutions.append((spacing, angles))

    if not solutions:
        raise ValueError(
            f"No isolated IK solution on the {branch!r} FK branch; "
            "try the other branch."
        )
    solutions.sort(key=lambda item: item[0], reverse=True)
    return [angles for _, angles in solutions]


def ik(x, y, l1, l2, d, *, branch="upper", elbows="open"):
    """Return [q1, q2] for a target [x, y].

    Default: highest-tip FK branch, then maximum distance between elbows.
    elbows="closed" selects the minimum distance on the requested branch.
    Use ik_solutions() to inspect intermediate configurations as well.
    """
    if elbows not in ("open", "closed"):
        raise ValueError("elbows must be 'open' or 'closed'.")
    solutions = ik_solutions(x, y, l1, l2, d, branch=branch)
    return solutions[0] if elbows == "open" else solutions[-1]


def jacobian(q1, q2, l1, l2, d, *, branch="upper"):
    """Return J such that [Xdot, Ydot] = J @ [q1dot, q2dot].

    Differentiating |P-E_i|^2 = l2^2 gives A @ Pdot = B @ qdot:
    row i of A is P-E_i; B_ii = (P-E_i) dot dE_i/dq_i.
    The derivative follows the chosen assembly locally; upper/lower
    selection may switch discontinuously when the two tips have equal Y.

    Raise ValueError at a parallel singularity (collinear distal links),
    where no unique forward velocity exists. At a serial singularity, J
    is returned but can be rank deficient. Units: length per radian.
    """
    tip = fk(q1, q2, l1, l2, d, branch=branch)
    elbows = elbow_positions(q1, q2, l1, d)
    # Normalize by l2 so the singularity check is independent of units.
    a = (tip - elbows) / l2
    determinant = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    if abs(determinant) <= _RTOL:
        raise ValueError("Parallel singularity: the Jacobian is undefined.")
    elbow_derivatives = l1 * np.array(
        [
            [math.cos(q1), -math.sin(q1)],
            [math.cos(q2), -math.sin(q2)],
        ]
    )
    b = np.diag(np.sum(a * elbow_derivatives, axis=1))
    return np.linalg.solve(a, b)
