"""Separate CNRS pen-drawing example using the five-bar kinematics module."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kinematics import elbow_positions, ik, ik_solutions, jacobian
from kinematics.plot_fivebar import plot_configuration


@dataclass
class Stroke:
    """A straight line (2 controls) or cubic Bezier (4 controls), in meters."""

    name: str
    controls: np.ndarray
    connector: bool = False

    def evaluate(self, u):
        u = np.asarray(u, dtype=float)[..., None]
        p = self.controls
        if len(p) == 2:
            return (1 - u) * p[0] + u * p[1]
        return (
            (1 - u) ** 3 * p[0]
            + 3 * (1 - u) ** 2 * u * p[1]
            + 3 * (1 - u) * u**2 * p[2]
            + u**3 * p[3]
        )

    def derivative(self, u):
        u = np.asarray(u, dtype=float)[..., None]
        p = self.controls
        if len(p) == 2:
            return np.broadcast_to(p[1] - p[0], u.shape[:-1] + (2,))
        return (
            3 * (1 - u) ** 2 * (p[1] - p[0])
            + 6 * (1 - u) * u * (p[2] - p[1])
            + 3 * u**2 * (p[3] - p[2])
        )


def cnrs_path(
    width=0.080, height=0.025, baseline=0.090, center_x=0.0, *, border_margin=0.005
):
    """Draw CNRS, then go up from the S and trace a closed rectangular frame.

    The pen stays down. Set border_margin=0 to draw only the letters.
    """
    if not all(
        np.isfinite(v) for v in (width, height, baseline, center_x, border_margin)
    ):
        raise ValueError("Drawing dimensions must be finite.")
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive.")
    if border_margin < 0:
        raise ValueError("Border margin must be nonnegative.")
    letter_width = width / 5
    gap = width / 15
    left = center_x - width / 2
    strokes = []

    def add(name, letter, points, connector=False):
        offset = np.array([left + letter * (letter_width + gap), baseline])
        controls = np.array(points, dtype=float) * [letter_width, height] + offset
        strokes.append(Stroke(name, controls, connector))

    # C: start at its upper-right tip, finish at its lower-right tip.
    add("C upper", 0, [(1, 1), (0.35, 1), (0, 0.92), (0, 0.5)])
    add("C lower", 0, [(0, 0.5), (0, 0.08), (0.35, 0), (1, 0)])
    add("C to N", 0, [(1, 0), (4 / 3, 0)], True)

    add("N left", 1, [(0, 0), (0, 1)])
    add("N diagonal", 1, [(0, 1), (1, 0)])
    add("N right up", 1, [(1, 0), (1, 1)])
    add("N right retrace", 1, [(1, 1), (1, 0)])
    add("N to R", 1, [(1, 0), (4 / 3, 0)], True)

    add("R stem", 2, [(0, 0), (0, 1)])
    add("R top", 2, [(0, 1), (0.45, 1)])
    # Two quarter ellipses, with horizontal tangents at the top and bottom.
    k = 0.5522847498307936
    add(
        "R bowl upper",
        2,
        [(0.45, 1), (0.45 + 0.55 * k, 1), (1, 0.75 + 0.25 * k), (1, 0.75)],
    )
    add(
        "R bowl lower",
        2,
        [(1, 0.75), (1, 0.75 - 0.25 * k), (0.45 + 0.55 * k, 0.5), (0.45, 0.5)],
    )
    add("R middle", 2, [(0.45, 0.5), (0, 0.5)])
    add("R leg", 2, [(0, 0.5), (1, 0)])
    add("R to S", 2, [(1, 0), (4 / 3, 0)], True)

    # S drawn from its lower-left tip to its upper-right tip.
    add("S bottom", 3, [(0, 0), (0.75, 0), (1, 0.10), (1, 0.25)])
    add("S lower", 3, [(1, 0.25), (1, 0.45), (0.8, 0.5), (0.5, 0.5)])
    add("S upper", 3, [(0.5, 0.5), (0.2, 0.5), (0, 0.55), (0, 0.75)])
    add("S top", 3, [(0, 0.75), (0, 0.95), (0.25, 1), (1, 1)])
    if border_margin > 0:
        right = left + width
        top = baseline + height + border_margin
        bottom = baseline - border_margin
        join = np.array([right, top])
        strokes.append(
            Stroke("S to frame", np.array([strokes[-1].controls[-1], join]), True)
        )
        corners = np.array(
            [
                join,
                [right + border_margin, top],
                [right + border_margin, bottom],
                [left - border_margin, bottom],
                [left - border_margin, top],
                join,
            ]
        )
        names = ("top right", "right", "bottom", "left", "top close")
        for name, start, end in zip(names, corners[:-1], corners[1:]):
            strokes.append(Stroke(f"Frame {name}", np.array([start, end])))
    return strokes


def preview(strokes, geometry, output):
    """Save PNG and SVG previews before generating any motor trajectory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    points = np.concatenate([s.evaluate(np.linspace(0, 1, 80)) for s in strokes])
    # Check the complete stroke, including all letter-to-letter connections.
    angles = np.array([ik(*p, *geometry) for p in points])
    fig, (detail, workspace) = plt.subplots(
        2,
        1,
        figsize=(11, 9),
        gridspec_kw={"height_ratios": [1, 1.2]},
        layout="constrained",
    )
    fig.set_facecolor("#faf9f6")
    for ax in (detail, workspace):
        ax.set_facecolor("#faf9f6")
        ax.spines[["top", "right"]].set_visible(False)

    for stroke in strokes:
        p = stroke.evaluate(np.linspace(0, 1, 100)) * 1000
        detail.plot(
            p[:, 0],
            p[:, 1],
            color="#da8130" if stroke.connector else "#183d54",
            linewidth=3.5,
            solid_capstyle="round",
        )
    first, last = points[0] * 1000, points[-1] * 1000
    for p, color, label in ((first, "#298b67", "Start"), (last, "#bd4949", "Finish")):
        detail.plot(*p, "o", color=color, markersize=8, zorder=5)
        offset = (0, -20) if label == "Start" else (0, 12)
        detail.annotate(
            label,
            p,
            xytext=offset,
            textcoords="offset points",
            ha="center",
            color=color,
            weight="bold",
        )
    detail.set_aspect("equal")
    detail.set(xlabel="X [mm]", ylabel="Y [mm]", title="CNRS · continuous pen path")
    detail.margins(x=0.06, y=0.35)
    detail.grid(alpha=0.12)
    detail.text(
        0.5,
        -0.27,
        "Orange: pen-down connections   ·   N: right stem traced twice",
        transform=detail.transAxes,
        ha="center",
        color="#52606a",
        fontsize=10,
    )

    # The kinematics helper draws in the units supplied; show this panel in mm.
    plot_configuration(*angles[0], *(np.array(geometry) * 1000), ax=workspace)
    workspace.plot(
        points[:, 0] * 1000,
        points[:, 1] * 1000,
        color="#183d54",
        linewidth=2,
        label="Pen path",
    )
    workspace.set(
        xlabel="X [mm]", ylabel="Y [mm]", title="Robot at the start of the drawing"
    )
    workspace.legend(loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("CNRS drawing preview", fontsize=20, weight="bold")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"cnrs_preview.{suffix}", dpi=170)
    plt.close(fig)
    return output / "cnrs_preview.png"


@dataclass
class Trajectory:
    time: np.ndarray
    xy: np.ndarray
    xy_velocity: np.ndarray
    q: np.ndarray
    q_velocity: np.ndarray
    stroke_index: np.ndarray
    stroke_info: list


def sample_path(strokes, speed=0.010, rate=100.0):
    """Time the path at a uniform rate, with pen speed bounded by speed.

    Each primitive uses arc-length parameterization and a quintic time law.
    Velocity and acceleration vanish at every primitive boundary, including
    corners and reversals. Smooth curve joins also pause in this first version.
    """
    if not np.isfinite(speed) or not np.isfinite(rate) or speed <= 0 or rate <= 0:
        raise ValueError("Speed and sample rate must be finite and positive.")
    if not strokes:
        raise ValueError("The path must contain at least one stroke.")
    positions, velocities, indices, info = [], [], [], []
    elapsed_ticks = 0
    previous_end = None
    for index, stroke in enumerate(strokes):
        if previous_end is not None and not np.allclose(
            previous_end, stroke.controls[0], rtol=0, atol=1e-12
        ):
            raise ValueError("Pen-down strokes must meet without a jump.")
        previous_end = stroke.controls[-1]
        parameter = np.linspace(0, 1, 2001)
        curve = stroke.evaluate(parameter)
        distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        length = distance[-1]
        if not np.isfinite(length) or length <= 0 or np.any(np.diff(distance) <= 0):
            raise ValueError(
                "Every stroke must have positive length and a regular tangent."
            )

        # max(d/du [10u^3 - 15u^4 + 6u^5]) = 1.875.
        ticks = max(20, int(np.ceil(1.875 * length / speed * rate)))
        duration = ticks / rate
        u = np.arange(ticks + 1) / ticks
        progress = u**3 * (10 - 15 * u + 6 * u * u)
        curve_u = np.interp(progress * length, distance, parameter)
        position = stroke.evaluate(curve_u)
        tangent = stroke.derivative(curve_u)
        tangent = tangent / np.linalg.norm(tangent, axis=1)[:, None]
        path_speed = (30 * u * u * (1 - u) ** 2) * length / duration
        velocity = tangent * path_speed[:, None]
        keep = slice(None) if index == 0 else slice(1, None)
        positions.append(position[keep])
        velocities.append(velocity[keep])
        indices.append(np.full(len(position[keep]), index, dtype=int))
        info.append(
            {
                "index": index,
                "name": stroke.name,
                "connector": stroke.connector,
                "start_sample": elapsed_ticks,
                "end_sample": elapsed_ticks + ticks,
                "length_m": float(length),
                "duration_s": duration,
            }
        )
        elapsed_ticks += ticks
    return (
        np.arange(elapsed_ticks + 1) / rate,
        np.concatenate(positions),
        np.concatenate(velocities),
        np.concatenate(indices),
        info,
    )


def motor_trajectory(strokes, geometry, speed=0.010, rate=100.0):
    """Follow the initial open IK mode continuously and compute motor speeds."""
    time, xy, xy_velocity, indices, info = sample_path(strokes, speed, rate)
    l1, _l2, d = geometry
    motors = np.array([[-d / 2, 0.0], [d / 2, 0.0]])

    def elbow_mode(q, point):
        arms = elbow_positions(*q, l1, d) - motors
        target = point - motors
        cross = arms[:, 0] * target[:, 1] - arms[:, 1] * target[:, 0]
        return np.sign(cross)

    q = np.empty_like(xy)
    q_velocity = np.empty_like(xy)
    mode = None
    for index, (point, velocity) in enumerate(zip(xy, xy_velocity)):
        try:
            candidates = ik_solutions(*point, *geometry)
            if index == 0:
                chosen = candidates[0]  # Widest elbows at the starting pose.
                mode = elbow_mode(chosen, point)
            else:
                candidates = [
                    candidate
                    for candidate in candidates
                    if np.array_equal(elbow_mode(candidate, point), mode)
                ]
                if not candidates:
                    raise ValueError("The path cannot stay in the starting elbow mode.")
                # Unwrap around the previous angle, then choose the nearest solution.
                candidates = [
                    q[index - 1]
                    + (candidate - q[index - 1] + np.pi) % (2 * np.pi)
                    - np.pi
                    for candidate in candidates
                ]
                chosen = min(candidates, key=lambda a: np.linalg.norm(a - q[index - 1]))
                if np.max(np.abs(chosen - q[index - 1])) > np.pi / 2:
                    raise ValueError(
                        "Motor angle jump: use a slower or more finely sampled path."
                    )
            j = jacobian(*chosen, *geometry)
            if np.linalg.svd(j, compute_uv=False)[-1] <= 1e-8 * max(geometry):
                raise ValueError("The path reaches a serial singularity.")
            q[index] = chosen
            q_velocity[index] = np.linalg.solve(j, velocity)
        except ValueError as error:
            raise ValueError(
                f"At sample {index}, t={time[index]:.3f} s, "
                f"XY={point.tolist()} m: {error}"
            ) from error
    return Trajectory(time, xy, xy_velocity, q, q_velocity, indices, info)


def save_trajectory(trajectory, geometry, output, parameters):
    """Export motor positions/speeds, Cartesian reference, metadata, and plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = np.column_stack(
        (
            trajectory.time,
            trajectory.q,
            trajectory.q_velocity,
            trajectory.xy,
            trajectory.xy_velocity,
            trajectory.stroke_index,
        )
    )
    csv_path = output / "cnrs_trajectory.csv"
    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        comments="",
        header="time_s,q1_rad,q2_rad,dq1_rad_s,dq2_rad_s,x_m,y_m,vx_m_s,vy_m_s,stroke_index",
        fmt=["%.9f"] + ["%.12f"] * 8 + ["%d"],
    )
    metadata = {
        "geometry_m": dict(zip(("l1", "l2", "d"), geometry)),
        "parameters": parameters,
        "angles": "Radians, zero north, positive clockwise; motor 1 left.",
        "assembly": "Upper tip, widest elbows initially, same elbow mode throughout.",
        "pen": "Down throughout. N right stem retraced. No approach or homing move.",
        "timing": "Quintic arc-length timing; rest at every primitive boundary.",
        "samples": len(trajectory.time),
        "duration_s": float(trajectory.time[-1]),
        "path_length_m": sum(item["length_m"] for item in trajectory.stroke_info),
        "peak_pen_speed_m_s": float(
            np.max(np.linalg.norm(trajectory.xy_velocity, axis=1))
        ),
        "peak_motor_speed_rad_s": np.max(
            np.abs(trajectory.q_velocity), axis=0
        ).tolist(),
        "start_q_rad": trajectory.q[0].tolist(),
        "finish_q_rad": trajectory.q[-1].tolist(),
        "strokes": trajectory.stroke_info,
    }
    (output / "cnrs_trajectory.json").write_text(json.dumps(metadata, indent=2) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    for motor, color in ((0, "#227798"), (1, "#da8130")):
        axes[0].plot(
            trajectory.time,
            np.degrees(trajectory.q[:, motor]),
            color=color,
            label=f"Motor {motor + 1}",
        )
        axes[1].plot(
            trajectory.time,
            np.degrees(trajectory.q_velocity[:, motor]),
            color=color,
            label=f"Motor {motor + 1}",
        )
    axes[0].set(ylabel="Angle [deg]", title="CNRS · motor position and speed")
    axes[1].set(xlabel="Time [s]", ylabel="Speed [deg/s]")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(output / "cnrs_motor_profiles.png", dpi=160)
    plt.close(fig)
    return csv_path, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--l1", type=float, default=0.06, help="Powered link length [m]"
    )
    parser.add_argument(
        "--l2", type=float, default=0.10, help="Passive link length [m]"
    )
    parser.add_argument("--d", type=float, default=0.10, help="Motor spacing [m]")
    parser.add_argument(
        "--width", type=float, default=0.080, help="Lettering width [m]"
    )
    parser.add_argument("--height", type=float, default=0.025, help="Letter height [m]")
    parser.add_argument("--baseline", type=float, default=0.090, help="Baseline Y [m]")
    parser.add_argument(
        "--center-x", type=float, default=0.0, help="Drawing center X [m]"
    )
    parser.add_argument(
        "--border-margin",
        type=float,
        default=0.005,
        help="Frame margin [m]; 0 omits it",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument(
        "--speed", type=float, default=0.010, help="Maximum pen speed [m/s]"
    )
    parser.add_argument(
        "--rate", type=float, default=100.0, help="CSV sample rate [Hz]"
    )
    parser.add_argument(
        "--preview-only", action="store_true", help="Only render the XY preview"
    )
    args = parser.parse_args()
    strokes = cnrs_path(
        args.width,
        args.height,
        args.baseline,
        args.center_x,
        border_margin=args.border_margin,
    )
    path = preview(strokes, (args.l1, args.l2, args.d), args.output)
    print(f"Preview: {path}")
    if args.preview_only:
        return
    geometry = (args.l1, args.l2, args.d)
    trajectory = motor_trajectory(strokes, geometry, args.speed, args.rate)
    parameters = {
        key: value
        for key, value in vars(args).items()
        if key not in ("output", "preview_only", "l1", "l2", "d")
    }
    csv_path, metadata = save_trajectory(trajectory, geometry, args.output, parameters)
    print(f"CSV: {csv_path}")
    print(
        f"{metadata['samples']} samples, {metadata['duration_s']:.2f} s, "
        f"{1000 * metadata['path_length_m']:.1f} mm of pen travel"
    )


if __name__ == "__main__":
    main()
