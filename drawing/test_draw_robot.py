"""Hardware-free controller tests, including firmware-watchdog semantics."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from drawing.draw_robot import (
    DEFAULT_CSV,
    SettlingWindow,
    draw,
    interpolate,
    load_trajectory,
)


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeMotor:
    def __init__(self, q):
        self.q, self.v = q, 0.0
        self.i, self.i_target = 0.0, 0.0

    def set(self, **values):
        self.target = values


class FakeRobot:
    def __init__(self, clock, follow=True):
        self.clock = clock
        self.follow = follow
        self.m0, self.m1 = FakeMotor(0.4), FakeMotor(-0.2)
        self.commands = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def initialize(self):
        self.initialized = True

    def update(self, timeout_ms=None, block=False):
        if len(self.commands) > 10000:
            raise AssertionError("Controller loop is not advancing simulated time")
        if self.commands:
            previous = self.commands[-1]
            if previous["timeout_ms"] and (
                self.clock.now - previous["time"] > previous["timeout_ms"] / 1000 + 1e-9
            ):
                raise RuntimeError("Firmware watchdog expired")
        q = np.array([self.m0.target["q"], self.m1.target["q"]])
        v = np.array([self.m0.target["v"], self.m1.target["v"]])
        self.commands.append(
            {
                "time": self.clock.now,
                "timeout_ms": timeout_ms,
                "block": block,
                "q": q,
                "v": v,
            }
        )
        if self.follow:
            self.m0.q, self.m1.q = q
            self.m0.v, self.m1.v = v


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.robot = FakeRobot(self.clock)
        self.trajectory = (
            np.array([0.0, 0.1, 0.2]),
            np.array([[0.1, -0.2], [0.12, -0.15], [0.13, -0.14]]),
            np.array([[0.0, 0.0], [0.1, 0.15], [0.0, 0.0]]),
        )

    def run_draw(self, prompt, **options):
        with self.robot:
            draw(
                self.robot,
                self.trajectory,
                approach_seconds=0.25,
                clock=self.clock.monotonic,
                sleep=self.clock.sleep,
                prompt=prompt,
                **options,
            )

    def test_absolute_targets_and_pause_watchdog(self):
        pause_indices = []

        def prompt(message):
            last = self.robot.commands[-1]
            self.assertEqual(last["timeout_ms"], 0)
            self.assertTrue(last["block"])
            assert_allclose(last["v"], 0)
            expected = self.trajectory[1][0 if not pause_indices else -1]
            assert_allclose(last["q"], expected)
            pause_indices.append(len(self.robot.commands) - 1)
            self.clock.sleep(30)  # Much longer than the 20 ms motion watchdog.

        self.run_draw(prompt)
        self.assertEqual(len(pause_indices), 2)
        first = pause_indices[0]
        self.assertEqual(self.robot.commands[first + 1]["timeout_ms"], 20)
        self.assertTrue(self.robot.closed)
        assert_allclose(self.robot.commands[0]["q"], [0.4, -0.2])
        assert_allclose(self.robot.commands[-1]["q"], [0.13, -0.14])
        deltas = np.diff([entry["q"] for entry in self.robot.commands], axis=0)
        self.assertLess(np.max(np.abs(deltas)), 0.02)

    def test_playback_multiplier_scales_duration_and_velocity(self):
        approach_and_settling = []
        for requested in (None, 0.5, 1.0, 3.0):
            with self.subTest(multiplier=requested):
                self.setUp()
                multiplier = 2.0 if requested is None else requested
                pauses = []
                original = [array.copy() for array in self.trajectory]

                def prompt(_):
                    pauses.append((self.clock.now, len(self.robot.commands)))  # noqa: B023

                options = {} if requested is None else {"speed_multiplier": requested}
                self.run_draw(prompt, **options)
                self.assertEqual(len(pauses), 2)
                start, first_command = pauses[0]
                finish, last_command = pauses[1]
                self.assertAlmostEqual(finish - start, 0.2 / multiplier, places=10)
                approach_and_settling.append(start - self.robot.commands[0]["time"])
                for command in self.robot.commands[first_command:last_command]:
                    elapsed = (command["time"] - start) * multiplier
                    q, v = interpolate(elapsed, *self.trajectory)
                    assert_allclose(command["q"], q, atol=1e-10)
                    assert_allclose(command["v"], v * multiplier, atol=1e-9)
                for actual, expected in zip(self.trajectory, original):
                    assert_allclose(actual, expected)
        assert_allclose(approach_and_settling, approach_and_settling[0], atol=1e-10)

    def test_interrupt_and_eof_during_pause_close_controller(self):
        for error in (KeyboardInterrupt, EOFError):
            with self.subTest(error=error):
                self.setUp()

                def prompt(_):
                    self.assertEqual(self.robot.commands[-1]["timeout_ms"], 0)
                    raise error()  # noqa: B023

                with self.assertRaises(error):
                    self.run_draw(prompt)
                self.assertTrue(self.robot.closed)

    def test_settling_failure_does_not_prompt(self):
        self.robot.follow = False
        with self.assertRaisesRegex(TimeoutError, "Could not settle"):
            self.run_draw(
                lambda _: self.fail("Must not start drawing"), settle_timeout=0.1
            )
        self.assertTrue(self.robot.closed)

    def test_persistent_motion_does_not_start_drawing(self):
        update = self.robot.update

        def moving_feedback(**kwargs):
            update(**kwargs)
            self.robot.m0.v = 0.2

        self.robot.update = moving_feedback
        with self.assertRaisesRegex(TimeoutError, "Could not settle"):
            self.run_draw(
                lambda _: self.fail("Must not start drawing"), settle_timeout=0.1
            )
        self.assertTrue(self.robot.closed)

    def test_brief_speed_spikes_do_not_reset_settling(self):
        update = self.robot.update

        def noisy_feedback(**kwargs):
            update(**kwargs)
            # Five Hz spikes prevent the old uninterrupted 0.25 s dwell.
            self.robot.m0.v = 0.16 if len(self.robot.commands) % 100 == 0 else 0.07

        self.robot.update = noisy_feedback
        self.run_draw(lambda _: None, settle_timeout=0.6)
        self.assertTrue(self.robot.closed)

    def test_clockwise_commands_reconstruct_csv_path(self):
        _, q, _velocity = load_trajectory(DEFAULT_CSV)
        table = np.genfromtxt(DEFAULT_CSV, delimiter=",", names=True)
        tip = np.column_stack((table["x_m"], table["y_m"]))
        # Independent physical formula: clockwise from north gives +sin(q) in X.
        elbows = np.stack((0.06 * np.sin(q), 0.06 * np.cos(q)), axis=2)
        elbows[:, 0, 0] -= 0.05
        elbows[:, 1, 0] += 0.05
        assert_allclose(
            np.linalg.norm(tip[:, None, :] - elbows, axis=2), 0.10, atol=1e-11
        )
        self.assertLess(elbows[0, 0, 0], -0.05)
        self.assertGreater(elbows[0, 1, 0], 0.05)
        assert_allclose(q[0], [-0.789646856696, 0.151569291774], atol=1e-12)

    def test_communication_failure_closes_controller(self):
        original_update = self.robot.update

        def failing_update(**kwargs):
            if len(self.robot.commands) == 10:
                raise RuntimeError("USB disconnected")
            return original_update(**kwargs)

        self.robot.update = failing_update
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            self.run_draw(lambda _: self.fail("Unexpected prompt"))
        self.assertTrue(self.robot.closed)

    def test_interpolation_matches_csv_and_derivative(self):
        timestamps, q, velocity = self.trajectory
        for t, expected_q, expected_v in zip(timestamps, q, velocity):
            position, speed = interpolate(t, *self.trajectory)
            assert_allclose(position, expected_q, atol=1e-12)
            assert_allclose(speed, expected_v, atol=1e-12)
        for t in np.linspace(0.001, 0.199, 34):
            position, speed = interpolate(t, *self.trajectory)
            step = 1e-6
            numeric = (
                interpolate(t + step, *self.trajectory)[0]
                - interpolate(t - step, *self.trajectory)[0]
            ) / (2 * step)
            assert_allclose(speed, numeric, atol=1e-8)

    def test_real_drawing_csv_loads(self):
        timestamps, q, v = load_trajectory(DEFAULT_CSV)
        metadata = json.loads(DEFAULT_CSV.with_suffix(".json").read_text())
        self.assertEqual(len(timestamps), metadata["samples"])
        self.assertAlmostEqual(timestamps[-1], metadata["duration_s"])
        assert_allclose(v[[0, -1]], 0)
        self.assertEqual(q.shape, (len(timestamps), 2))

    def test_invalid_csv_rejected(self):
        header = "time_s,q1_rad,q2_rad,dq1_rad_s,dq2_rad_s\n"
        invalid = [
            "time_s,wrong\n0,1\n1,1\n",
            header + "0,0,0,0,0\n",
            header + "0,0,0,0,0\n0,0,0,0,0\n",
            header + "1,0,0,0,0\n2,0,0,0,0\n",
            header + "0,nan,0,0,0\n1,0,0,0,0\n",
            header + "0,0,0,1,0\n1,0,0,0,0\n",
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            for text in invalid:
                path.write_text(text)
                with self.subTest(csv=text), self.assertRaises(ValueError):
                    load_trajectory(path)

    def test_invalid_control_settings_rejected_before_initialization(self):
        for options in (
            {"timeout_ms": 0},
            {"timeout_ms": 65536},
            {"control_rate": 10},
            {"kp": -1},
            {"speed_tolerance": 0},
            {"settle_window": 0},
            {"speed_multiplier": 0},
            {"speed_multiplier": -1},
            {"speed_multiplier": np.nan},
            {"speed_multiplier": np.inf},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_draw(lambda _: self.fail("Unexpected prompt"), **options)
        self.assertFalse(hasattr(self.robot, "initialized"))


class SettlingWindowTests(unittest.TestCase):
    def test_reported_feedback_is_accepted_after_full_window(self):
        window = SettlingWindow(0.25)
        error = np.array([0.016844, 0.001239])
        velocity = np.array([0.066451, -0.014702])
        for t in np.arange(125) * 0.002:
            window.add(t, error, velocity)
            self.assertFalse(window.ready(0.03, 0.1))
        window.add(0.25, error, velocity)
        self.assertTrue(window.ready(0.03, 0.1))

    def test_rms_is_weighted_by_time_not_sample_count(self):
        window = SettlingWindow(0.25)
        window.add(0.0, [0, 0], [0.2, 0])
        window.add(0.01, [0, 0], [0, 0])
        window.add(0.24, [0, 0], [0, 0])
        window.add(0.25, [0, 0], [0, 0])
        assert_allclose(window.rms_speed, [0.04, 0], atol=1e-12)
        self.assertTrue(window.ready(0.03, 0.1))

    def test_alternating_velocity_does_not_cancel(self):
        window = SettlingWindow(0.25)
        for i in range(126):
            window.add(i * 0.002, [0, 0], [0.2 * (-1) ** i, 0])
        assert_allclose(window.rms_speed, [0.2, 0], atol=1e-12)
        self.assertFalse(window.ready(0.03, 0.1))

    def test_position_excursion_must_leave_window_before_acceptance(self):
        window = SettlingWindow(0.25)
        for i in range(151):
            window.add(i * 0.002, [0.04 if i == 50 else 0.0, 0], [0, 0])
        # The current error is zero, but a recent sample was outside tolerance.
        self.assertFalse(window.ready(0.03, 0.1))
        assert_allclose(window.max_error, [0.04, 0])
        for i in range(151, 181):
            window.add(i * 0.002, [0, 0], [0, 0])
        self.assertTrue(window.ready(0.03, 0.1))


if __name__ == "__main__":
    unittest.main()
