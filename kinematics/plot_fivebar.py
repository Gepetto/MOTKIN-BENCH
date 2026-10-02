"""Plot a five-bar configuration. Run this file for a simple example."""

import numpy as np

from kinematics import elbow_positions, fk, ik


def plot_configuration(q1, q2, l1, l2, d, *, branch="upper", ax=None):
    """Draw the robot and return a Matplotlib Axes; use plt.show() to display.

    Supply ax to overlay or arrange configurations. Matplotlib is only
    needed when this helper is called.
    """
    import matplotlib.pyplot as plt

    tip = fk(q1, q2, l1, l2, d, branch=branch)
    e1, e2 = elbow_positions(q1, q2, l1, d)
    m1, m2 = np.array([[-d / 2, 0.0], [d / 2, 0.0]])
    if ax is None:
        _, ax = plt.subplots()
    ax.plot([m1[0], m2[0]], [0, 0], "k--", linewidth=1, label="Fixed base")
    for motor, elbow, label in ((m1, e1, "Left arm"), (m2, e2, "Right arm")):
        points = np.array([motor, elbow, tip])
        ax.plot(points[:, 0], points[:, 1], "o-", linewidth=3, label=label)
    ax.plot([m1[0], m2[0]], [0, 0], "ks", markersize=8)
    ax.plot(*tip, "ko", markersize=7)
    ax.plot(0, 0, "k+", markersize=8)
    for label, point in (("M1", m1), ("M2", m2), ("E1", e1), ("E2", e2), ("P", tip)):
        ax.annotate(label, point, xytext=(6, 6), textcoords="offset points")
    ax.set(xlabel="X", ylabel="Y", title=f"Five-bar robot ({branch} branch)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.margins(0.15)
    ax.legend()
    return ax



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    geometry = (0.06, 0.10, 0.10)
    q = ik(0.0, 0.12, *geometry)
    plot_configuration(*q, *geometry)
    plt.show()
