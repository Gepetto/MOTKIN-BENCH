# Five-bar kinematics

[`fivebar.py`](fivebar.py) contains only numerical geometry and kinematics.
[`plot_fivebar.py`](plot_fivebar.py) provides the plotting helper and a runnable example.

- Motor 1 is at `(-d/2, 0)`; motor 2 is at `(d/2, 0)`.
- `l1` is the motor-to-elbow length; `l2` is the elbow-to-tip length.
- Angles are radians, **clockwise-positive from north**.
- X points right and Y points up. Use consistent length units.

| Function | Returns |
| --- | --- |
| `fk(q1, q2, l1, l2, d)` | Tip position `[X, Y]` |
| `ik(x, y, l1, l2, d)` | Motor angles `[q1, q2]` |
| `ik_solutions(x, y, l1, l2, d)` | Available solutions, widest elbows first |
| `jacobian(q1, q2, l1, l2, d)` | Matrix satisfying `XYdot = J @ qdot` |
| `elbow_positions(q1, q2, l1, d)` | Left and right elbow positions |
| `plot_configuration(q1, q2, l1, l2, d)` | Matplotlib axes, from `plot_fivebar` |

From a script or Python session at the repository root:

```python
from kinematics import fk, ik, jacobian
from kinematics.plot_fivebar import plot_configuration
import matplotlib.pyplot as plt

geometry = (0.06, 0.10, 0.10)
q = ik(0.0, 0.12, *geometry)
position = fk(*q, *geometry)
velocity = jacobian(*q, *geometry) @ [0.1, -0.1]
plot_configuration(*q, *geometry)
plt.show()
```

From the repository root:

```bash
python3 -m kinematics.plot_fivebar
python3 -m unittest discover -s kinematics -v
```

The `kinematics` package leaves Python's standard-library `math` module untouched.
Run the examples as modules from the repository root.

FK defaults to the highest tip (`branch="upper"`). IK chooses the widest
elbow spacing on that branch. Use `branch="lower"` consistently across
functions for the other assembly, or `elbows="closed"` in IK for minimum
elbow spacing. Returned IK angles lie in `[-pi, pi)`; equal-height FK tips
are ordered by X, with the rightmost considered upper.

Invalid geometry, unreachable targets, and underdetermined poses raise
`ValueError`. The Jacobian also raises at parallel singularities; serial
singularities can produce a rank-deficient matrix. Its derivative follows the
chosen assembly locally. Joint limits, link collisions, and trajectory
continuity are outside the numerical solver.
