"""Checks for pen continuity, timing, and the exported motor trajectory."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from drawing.cnrs import cnrs_path, motor_trajectory, sample_path, save_trajectory
from kinematics import fk


class CNRSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = (0.06, 0.10, 0.10)
        cls.strokes = cnrs_path()
        cls.trajectory = motor_trajectory(cls.strokes, cls.geometry)

    def test_continuous_path_and_horizontal_connections(self):
        for first, second in zip(self.strokes, self.strokes[1:]):
            assert_allclose(first.controls[-1], second.controls[0], atol=1e-15)
        connections = [
            s for s in self.strokes if s.connector and s.name != "S to frame"
        ]
        self.assertEqual(len(connections), 3)
        for stroke in connections:
            assert_allclose(stroke.controls[:, 1], 0.09)
            self.assertGreater(stroke.controls[1, 0], stroke.controls[0, 0])
        assert_allclose(self.strokes[5].controls, self.strokes[6].controls[::-1])

    def test_clockwise_start_angles(self):
        assert_allclose(
            self.trajectory.q[0], [-0.789646856696, 0.151569291774], atol=1e-12
        )

    def test_drawing_dimensions(self):
        points = np.concatenate(
            [s.evaluate(np.linspace(0, 1, 1001)) for s in self.strokes]
        )
        assert_allclose(points.min(axis=0), [-0.045, 0.085], atol=1e-12)
        assert_allclose(points.max(axis=0), [0.045, 0.120], atol=1e-12)

    def test_frame_starts_upward_and_closes(self):
        letters = cnrs_path(border_margin=0)
        connection = next(s for s in self.strokes if s.name == "S to frame")
        assert_allclose(connection.controls[0], letters[-1].controls[-1])
        assert_allclose(connection.controls[1] - connection.controls[0], [0, 0.005])
        frame = [s for s in self.strokes if s.name.startswith("Frame ")]
        assert_allclose(frame[0].controls[0], frame[-1].controls[-1])
        length = 0.0
        for stroke in frame:
            delta = stroke.controls[1] - stroke.controls[0]
            self.assertEqual(np.count_nonzero(np.abs(delta) > 1e-12), 1)
            length += np.linalg.norm(delta)
        self.assertAlmostEqual(length, 2 * (0.09 + 0.035))
        assert_allclose(self.trajectory.xy[-1], [0.04, 0.12], atol=1e-12)

    def test_border_margin_option(self):
        letters = cnrs_path(border_margin=0)
        points = np.concatenate([s.evaluate(np.linspace(0, 1, 100)) for s in letters])
        assert_allclose(points.min(axis=0), [-0.04, 0.09])
        assert_allclose(points.max(axis=0), [0.04, 0.115])
        for margin in (-0.001, np.nan):
            with self.assertRaises(ValueError):
                cnrs_path(border_margin=margin)

    def test_uniform_timing_and_speed_limit(self):
        t = self.trajectory
        assert_allclose(np.diff(t.time), 0.01, rtol=0, atol=1e-13)
        self.assertEqual(t.time[0], 0)
        self.assertLessEqual(np.linalg.norm(t.xy_velocity, axis=1).max(), 0.010 + 1e-12)
        assert_allclose(t.q_velocity[[0, -1]], 0, atol=1e-14)
        for stroke in t.stroke_info:
            for index in (stroke["start_sample"], stroke["end_sample"]):
                assert_allclose(t.xy_velocity[index], 0, atol=1e-14)
                assert_allclose(t.q_velocity[index], 0, atol=1e-14)

    def test_motor_positions_reconstruct_pen_path(self):
        t = self.trajectory
        for q, point in zip(t.q[::37], t.xy[::37]):
            assert_allclose(fk(*q, *self.geometry), point, atol=1e-11)
        self.assertLess(np.abs(np.diff(t.q, axis=0)).max(), 0.02)
        self.assertTrue(np.isfinite(t.q_velocity).all())

    def test_motor_speed_matches_position_derivative(self):
        t = self.trajectory
        numeric = np.gradient(t.q, 0.01, axis=0, edge_order=2)
        assert_allclose(t.q_velocity, numeric, rtol=0.003, atol=2e-4)

    def test_pen_speed_matches_position_derivative(self):
        t = self.trajectory
        numeric = np.gradient(t.xy, 0.01, axis=0, edge_order=2)
        assert_allclose(t.xy_velocity, numeric, rtol=0.003, atol=1e-5)

    def test_export_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path, metadata = save_trajectory(
                self.trajectory, self.geometry, directory, {"rate": 100, "speed": 0.010}
            )
            data = np.genfromtxt(csv_path, delimiter=",", names=True)
            self.assertEqual(len(data), len(self.trajectory.time))
            assert_allclose(data["q1_rad"], self.trajectory.q[:, 0], atol=1e-12)
            assert_allclose(
                data["dq2_rad_s"], self.trajectory.q_velocity[:, 1], atol=1e-12
            )
            assert_allclose(data["time_s"], self.trajectory.time, atol=1e-9)
            self.assertEqual(metadata["samples"], len(data))
            self.assertTrue((Path(directory) / "cnrs_trajectory.json").exists())

    def test_invalid_timing_and_disconnected_path(self):
        for speed, rate in ((0, 100), (0.01, 0), (np.nan, 100)):
            with self.assertRaises(ValueError):
                sample_path(self.strokes, speed, rate)
        with self.assertRaisesRegex(ValueError, "jump"):
            sample_path([self.strokes[0], self.strokes[-1]])

    def test_unreachable_drawing(self):
        with self.assertRaises(ValueError):
            motor_trajectory(cnrs_path(baseline=1), self.geometry)


if __name__ == "__main__":
    unittest.main()
