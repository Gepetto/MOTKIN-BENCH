"""Check force direction and torque/current units without opening USB."""

import sys
import runpy
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.testing import assert_allclose

from kinematics import ik, jacobian


class HapticBallTests(unittest.TestCase):
    def test_radial_repulsion_and_current_conversion(self):
        cases = [([.03, 0], [0, 0]), ([.02, 0], [0, 0]),
                 ([.01, 0], [.501, 0]), ([-.01, 0], [-.501, 0]),
                 ([0, .01], [0, .501]), ([0, -.01], [0, -.501]),
                 ([.006, .008], [.3006, .4008]),
                 ([.019999, 0], [.00105, 0]), ([.020001, 0], [0, 0]),
                 ([0, 0], [1.001, 0])]
        for offset, expected_force in cases:
            with self.subTest(offset=offset):
                q = ik(*(np.array([0.0, 0.12]) + offset), .06, .10, .10)
                controller = MagicMock()
                robot = controller.return_value.__enter__.return_value
                robot.m0.q, robot.m1.q = q
                robot.update.side_effect = KeyboardInterrupt
                module = SimpleNamespace(MotorUsbController=controller)
                with patch.dict(sys.modules, motkin_pcb=module), patch.object(sys, "path", sys.path.copy()):
                    with self.assertRaises(KeyboardInterrupt):
                        runpy.run_module("examples.haptic_ball", run_name="__main__")
                currents = []
                for motor in (robot.m0, robot.m1):
                    command = motor.set.call_args.kwargs
                    self.assertEqual((command['kp'], command['kd']), (0, 0))
                    currents.append(command['iff'])
                force = np.linalg.solve(jacobian(*q, .06, .10, .10).T, np.array(currents) * .08)
                assert_allclose(force, expected_force, atol=1e-12)
                controller.assert_called_once_with(timeout_ms=20, max_command_rate_hz=500)
                controller.return_value.__exit__.assert_called_once()
