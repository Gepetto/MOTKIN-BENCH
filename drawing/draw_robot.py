#!/usr/bin/env python3
"""Move to the first CSV pose, wait for Enter, then draw using motkin_pcb."""

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
from motkin_pcb import MotorUsbController

DEFAULT_KP = 12.0
DEFAULT_KD = 0.3
DEFAULT_APPROACH_SECONDS = 2.0

DEFAULT_CSV = Path(__file__).parent / "output" / "cnrs_trajectory.csv"


def load_trajectory(path):
    """Read and validate the complete file before enabling any motor."""
    table = np.genfromtxt(path, delimiter=",", names=True, ndmin=1)
    columns = ("time_s", "q1_rad", "q2_rad", "dq1_rad_s", "dq2_rad_s")
    if table.dtype.names is None or not set(columns).issubset(table.dtype.names):
        raise ValueError(f"CSV must contain these columns: {', '.join(columns)}")
    data = np.column_stack([table[name] for name in columns])
    if len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("CSV needs at least two rows of finite values.")
    timestamps, q, velocity = data[:, 0], data[:, 1:3], data[:, 3:5]
    if abs(timestamps[0]) > 1e-9 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("CSV time must start at zero and increase strictly.")
    if not np.allclose(velocity[[0, -1]], 0, rtol=0, atol=1e-8):
        raise ValueError("The trajectory must start and finish at rest.")
    return timestamps, q, velocity


def interpolate(t, timestamps, q, velocity):
    """Cubic Hermite interpolation; position and velocity agree between rows."""
    t = np.clip(t, timestamps[0], timestamps[-1])
    i = min(np.searchsorted(timestamps, t, side="right") - 1, len(timestamps) - 2)
    dt = timestamps[i + 1] - timestamps[i]
    u = (t - timestamps[i]) / dt
    position = (
        (2 * u**3 - 3 * u * u + 1) * q[i]
        + (u**3 - 2 * u * u + u) * dt * velocity[i]
        + (-2 * u**3 + 3 * u * u) * q[i + 1]
        + (u**3 - u * u) * dt * velocity[i + 1]
    )
    speed = (
        (6 * u * u - 6 * u) / dt * q[i]
        + (3 * u * u - 4 * u + 1) * velocity[i]
        + (-6 * u * u + 6 * u) / dt * q[i + 1]
        + (3 * u * u - 2 * u) * velocity[i + 1]
    )
    return position, speed


class SettlingWindow:
    """Position envelope and time-weighted RMS speed over a trailing window.

    Integrate squared speed, so positive/negative oscillations cannot cancel.
    Samples use a zero-order hold until the following sample's timestamp.
    """

    def __init__(self, duration):
        self.duration = duration
        self.samples = deque()
        self.span = 0.0
        self.max_error = np.zeros(2)
        self.rms_speed = np.zeros(2)

    def add(self, now, position_error, velocity):
        self.samples.append((now, *np.abs(position_error), *np.abs(velocity)))
        cutoff = now - self.duration
        # Keep the sample active at the left edge of the window.
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()
        data = np.asarray(self.samples)
        self.max_error = np.max(data[:, 1:3], axis=0)
        intervals = np.diff(np.maximum(data[:, 0], cutoff))
        self.span = float(np.sum(intervals))
        if self.span > 0:
            mean_squared = (
                np.sum(data[:-1, 3:5] ** 2 * intervals[:, None], axis=0) / self.span
            )
            self.rms_speed = np.sqrt(mean_squared)
        else:
            self.rms_speed = data[-1, 3:5].copy()

    def ready(self, position_tolerance, speed_tolerance):
        return (
            self.span >= self.duration - 1e-9
            and np.all(self.max_error <= position_tolerance)
            and np.all(self.rms_speed <= speed_tolerance)
        )


def draw(
    robot,
    trajectory,
    *,
    kp=DEFAULT_KP,
    kd=DEFAULT_KD,
    approach_seconds=DEFAULT_APPROACH_SECONDS,
    control_rate=500.0,
    timeout_ms=20,
    position_tolerance=0.03,
    speed_tolerance=0.1,
    settle_timeout=5.0,
    settle_window=0.25,
    speed_multiplier=2.0,
    prompt=input,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Approach, wait for Enter, play the CSV, and hold for pen removal.

    Use inside a MotorUsbController context. CSV angles go directly to m0
    (left) and m1 (right): radians, clockwise-positive, zero at north.
    """
    numeric = (
        kp,
        kd,
        approach_seconds,
        control_rate,
        position_tolerance,
        speed_tolerance,
        settle_timeout,
        settle_window,
        speed_multiplier,
    )
    if not np.isfinite(numeric).all():
        raise ValueError("Control settings must be finite.")
    if (
        min(
            kp,
            approach_seconds,
            control_rate,
            position_tolerance,
            speed_tolerance,
            settle_timeout,
            settle_window,
            speed_multiplier,
        )
        <= 0
        or kd < 0
    ):
        raise ValueError("Control settings must be positive (kd may be zero).")
    if not 1 <= timeout_ms <= 65535 or 1000 / control_rate >= timeout_ms:
        raise ValueError("Watchdog must be 1..65535 ms and exceed the control period.")
    timestamps, q, velocity = trajectory
    playback_times = timestamps / speed_multiplier
    velocity = velocity * speed_multiplier
    period = 1 / control_rate

    def command(q, v, *, watchdog=timeout_ms, block=False):
        robot.m0.set(q=float(q[0]), v=float(v[0]), kp=kp, kd=kd, iff=0.0)
        robot.m1.set(q=float(q[1]), v=float(v[1]), kp=kp, kd=kd, iff=0.0)
        robot.update(timeout_ms=watchdog, block=block)

    def stream(duration, reference):
        """Use elapsed time; skip missed ticks instead of replaying stale rows."""
        start = clock()
        finish = start + duration
        while True:
            now = clock()
            elapsed = duration if now >= finish else now - start
            q, v = reference(elapsed)
            command(q, v)
            if elapsed >= duration:
                return
            now = clock()
            tick = int((now - start) / period) + 1
            deadline = min(start + tick * period, start + duration)
            delay = deadline - now
            if delay <= 0 and now < start + duration:
                delay = min(period, start + duration - now)
            sleep(max(0.0, delay))

    robot.initialize()  # Library initialization uses timeout_ms=0.
    measured = np.array([robot.m0.q, robot.m1.q])
    if not np.isfinite(measured).all():
        raise ValueError("Initial encoder readings must be finite.")
    target = q[0]

    # Printing and terminal input only happen while a zero-timeout hold is active.
    command(measured, [0, 0], watchdog=0, block=True)
    print(f"Moving to the first pose in {approach_seconds:g} s; keep the pen raised.")
    delta = target - measured

    def approach(t):
        u = t / approach_seconds
        progress = u**3 * (10 - 15 * u + 6 * u * u)
        progress_speed = 30 * u * u * (1 - u) ** 2 / approach_seconds
        return measured + delta * progress, delta * progress_speed

    stream(approach_seconds, approach)

    # Continue feeding the watchdog until the measured first pose has settled.
    start = clock()
    settling = SettlingWindow(settle_window)
    while True:
        command(target, [0, 0])
        now = clock()
        measured_q = np.array([robot.m0.q, robot.m1.q])
        measured_v = np.array([robot.m0.v, robot.m1.v])
        if not np.isfinite(measured_q).all() or not np.isfinite(measured_v).all():
            raise RuntimeError("Non-finite motor feedback while settling.")
        position_error = target - measured_q
        settling.add(now, position_error, measured_v)
        if settling.ready(position_tolerance, speed_tolerance):
            break
        if now - start >= settle_timeout:
            raise TimeoutError(
                f"Could not settle at the first pose within {settle_timeout:g} s."
            )
        sleep(period)

    # This sends timeout=0 to firmware AND waits for its echo before blocking.
    command(target, [0, 0], watchdog=0, block=True)
    prompt("At the first pose. Lower the pen, then press Enter to start drawing... ")
    # The first streamed packet re-enables the finite watchdog.
    stream(playback_times[-1], lambda t: interpolate(t, playback_times, q, velocity))

    # Hold the finished drawing while the pen is removed.
    command(q[-1], [0, 0], watchdog=0, block=True)
    prompt("Drawing complete. Lift the pen, then press Enter to disable the motors... ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--port", help="Serial port; otherwise use the library default")
    parser.add_argument("--kp", type=float, default=DEFAULT_KP)
    parser.add_argument("--kd", type=float, default=DEFAULT_KD)
    parser.add_argument(
        "--approach-seconds", type=float, default=DEFAULT_APPROACH_SECONDS
    )
    parser.add_argument(
        "--speed-multiplier",
        type=float,
        default=2.0,
        help="Drawing playback speed factor (default: 2; 1 = CSV timing)",
    )
    parser.add_argument(
        "--control-rate", type=float, default=500.0, help="USB command rate [Hz]"
    )
    parser.add_argument(
        "--timeout-ms", type=int, default=20, help="Watchdog during motion [ms]"
    )
    parser.add_argument(
        "--position-tolerance", type=float, default=0.03, help="Settling error [rad]"
    )
    parser.add_argument(
        "--speed-tolerance", type=float, default=0.1, help="Settling RMS speed [rad/s]"
    )
    parser.add_argument(
        "--settle-window", type=float, default=0.25, help="Settling RMS window [s]"
    )
    parser.add_argument(
        "--settle-timeout", type=float, default=5.0, help="Settling timeout [s]"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate CSV without opening USB"
    )
    args = parser.parse_args()
    if not np.isfinite(args.speed_multiplier) or args.speed_multiplier <= 0:
        parser.error("--speed-multiplier must be finite and strictly positive")
    trajectory = load_trajectory(args.csv)
    duration = trajectory[0][-1] / args.speed_multiplier
    print(f"{args.csv.name}: {args.speed_multiplier:g}x playback, {duration:.2f} s.")
    if args.dry_run:
        return

    with MotorUsbController(port=args.port, timeout_ms=args.timeout_ms) as robot:
        draw(
            robot,
            trajectory,
            kp=args.kp,
            kd=args.kd,
            approach_seconds=args.approach_seconds,
            control_rate=args.control_rate,
            timeout_ms=args.timeout_ms,
            position_tolerance=args.position_tolerance,
            speed_tolerance=args.speed_tolerance,
            settle_timeout=args.settle_timeout,
            settle_window=args.settle_window,
            speed_multiplier=args.speed_multiplier,
        )


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(str(error)) from None
