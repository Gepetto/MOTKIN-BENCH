"""Geometric, branch-selection, and differential checks for fivebar."""

import io
import math
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose

from kinematics import fk, ik, ik_solutions, jacobian
from kinematics.plot_fivebar import plot_configuration


class FiveBarTests(unittest.TestCase):
    geometry = (0.06, 0.10, 0.10)

    def elbows(self, q, l1, d):
        return np.array(
            [
                [-d / 2 + l1 * math.sin(q[0]), l1 * math.cos(q[0])],
                [d / 2 + l1 * math.sin(q[1]), l1 * math.cos(q[1])],
            ]
        )

    def test_zero_angles_and_both_fk_branches(self):
        l1, l2, d = self.geometry
        height = math.sqrt(l2**2 - (d / 2) ** 2)
        assert_allclose(fk(0, 0, *self.geometry), [0, l1 + height])
        assert_allclose(fk(0, 0, *self.geometry, branch="lower"), [0, l1 - height])

    def test_positive_quarter_turn_points_right(self):
        l1, l2, d = self.geometry
        expected = [l1, math.sqrt(l2 * l2 - d * d / 4)]
        assert_allclose(
            fk(math.pi / 2, math.pi / 2, *self.geometry), expected, atol=1e-13
        )

    def test_inverse_enumeration_and_open_closed_defaults(self):
        target = np.array([0.0, 0.12])
        solutions = ik_solutions(*target, *self.geometry)
        self.assertEqual(len(solutions), 4)
        spacings = []
        for q in solutions:
            self.assertTrue(np.all(q >= -math.pi) and np.all(q < math.pi))
            assert_allclose(fk(*q, *self.geometry), target, atol=1e-13)
            elbows = self.elbows(q, self.geometry[0], self.geometry[2])
            spacings.append(np.linalg.norm(elbows[1] - elbows[0]))
        self.assertEqual(spacings, sorted(spacings, reverse=True))
        assert_allclose(ik(*target, *self.geometry), solutions[0])
        assert_allclose(ik(*target, *self.geometry, elbows="closed"), solutions[-1])
        self.assertLess(solutions[0][0], 0)
        self.assertGreater(solutions[0][1], 0)

    def test_target_only_on_lower_branch(self):
        with self.assertRaisesRegex(ValueError, "branch"):
            ik(0, -0.12, *self.geometry)
        q = ik(0, -0.12, *self.geometry, branch="lower")
        assert_allclose(fk(*q, *self.geometry, branch="lower"), [0, -0.12], atol=1e-13)

    def test_link_lengths_and_round_trips_over_workspace(self):
        rng = np.random.default_rng(42)
        checked = 0
        for geometry in (self.geometry, (0.15, 0.10, 0.08), (1.0, 1.0, 1.0)):
            l1, l2, d = geometry
            motors = np.array([[-d / 2, 0], [d / 2, 0]])
            for q in rng.uniform(-math.pi, math.pi, size=(100, 2)):
                for branch in ("upper", "lower"):
                    try:
                        tip = fk(*q, *geometry, branch=branch)
                    except ValueError:
                        continue
                    elbows = self.elbows(q, l1, d)
                    assert_allclose(np.linalg.norm(elbows - motors, axis=1), l1)
                    assert_allclose(np.linalg.norm(tip - elbows, axis=1), l2)
                    recovered = ik(*tip, *geometry, branch=branch)
                    assert_allclose(
                        fk(*recovered, *geometry, branch=branch),
                        tip,
                        rtol=1e-10,
                        atol=1e-11,
                    )
                    checked += 1
        self.assertGreater(checked, 250)

    def test_jacobian_against_central_differences(self):
        rng = np.random.default_rng(7)
        step = 1e-6
        checked = 0
        for q in rng.uniform(-1.0, 1.0, size=(60, 2)):
            for branch in ("upper", "lower"):
                try:
                    analytic = jacobian(*q, *self.geometry, branch=branch)
                except ValueError:
                    continue
                numeric = np.column_stack(
                    [
                        (
                            fk(*(q + delta), *self.geometry, branch=branch)
                            - fk(*(q - delta), *self.geometry, branch=branch)
                        )
                        / (2 * step)
                        for delta in np.eye(2) * step
                    ]
                )
                assert_allclose(analytic, numeric, rtol=2e-6, atol=1e-9)
                checked += 1
        self.assertGreater(checked, 80)

    def test_length_unit_invariance(self):
        q = ik(0.015, 0.12, *self.geometry)
        tip = fk(*q, *self.geometry)
        j = jacobian(*q, *self.geometry)
        for scale in (1e-6, 1e3, 1e6):
            lengths = np.array(self.geometry) * scale
            assert_allclose(fk(*q, *lengths) / scale, tip, atol=1e-13)
            assert_allclose(jacobian(*q, *lengths) / scale, j, atol=1e-13)
            assert_allclose(ik(*(tip * scale), *lengths), q, atol=1e-12)

    def test_tangent_fk_and_parallel_singularity(self):
        for branch in ("upper", "lower"):
            assert_allclose(fk(0, 0, 1, 1, 2, branch=branch), [0, 1])
            with self.assertRaisesRegex(ValueError, "singularity"):
                jacobian(0, 0, 1, 1, 2, branch=branch)

    def test_fully_extended_ik_and_serial_singularity(self):
        target = [0, math.sqrt(4 - 0.25)]
        solutions = ik_solutions(*target, 1, 1, 1)
        self.assertEqual(len(solutions), 1)
        q = solutions[0]
        assert_allclose(fk(*q, 1, 1, 1), target, atol=1e-12)
        assert_allclose(jacobian(*q, 1, 1, 1), np.zeros((2, 2)), atol=1e-12)

    def test_equal_height_fk_tie_break(self):
        upper = fk(0, -math.pi / 2, 1, 1, 1)
        lower = fk(0, -math.pi / 2, 1, 1, 1, branch="lower")
        assert_allclose(upper, [-0.5 + math.sqrt(0.75), 0.5])
        assert_allclose(lower, [-0.5 - math.sqrt(0.75), 0.5])

    def test_unreachable_and_underdetermined_configurations(self):
        for args in ((0, 0, 1, 1, 3), (math.pi / 2, -math.pi / 2, 1, 1, 2)):
            with self.assertRaises(ValueError):
                fk(*args)
        for args in ((0, 10, 1, 1, 1), (-0.5, 0, 1, 1, 1), (-0.5, 0, 1, 2, 1)):
            with self.assertRaises(ValueError):
                ik(*args)

    def test_invalid_inputs(self):
        for lengths in ((0, 1, 1), (1, -1, 1), (1, 1, 0), (math.inf, 1, 1)):
            for function in (fk, ik, jacobian):
                with self.assertRaises(ValueError):
                    function(0, 0, *lengths)
        for function in (fk, ik, jacobian):
            with self.assertRaises(ValueError):
                function(math.nan, 0, *self.geometry)
            with self.assertRaises(ValueError):
                function(0, 0, *self.geometry, branch="invalid")
        with self.assertRaises(ValueError):
            ik(0, 0.12, *self.geometry, elbows="invalid")

    def test_plot_render_and_existing_axes(self):
        q = ik(0, 0.12, *self.geometry)
        ax = plot_configuration(*q, *self.geometry)
        try:
            self.assertEqual(ax.get_aspect(), 1.0)
            result = plot_configuration(*q, *self.geometry, branch="lower", ax=ax)
            self.assertIs(result, ax)
            output = io.BytesIO()
            ax.figure.savefig(output, format="png")
            self.assertGreater(output.tell(), 1000)
        finally:
            plt.close(ax.figure)


if __name__ == "__main__":
    unittest.main()
