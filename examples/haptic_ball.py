"""Run from the repository root: python3 -m examples.haptic_ball"""

import numpy as np
from kinematics import fk, jacobian

from motkin_pcb import MotorUsbController

CONTACT_FORCE = 0.5  # Constant outward force inside the ball [N]
KT = 0.08  # GM3506 estimate [Nm/A]; replace with measured value
GEOMETRY = (0.06, 0.10, 0.10)  # l1, l2, d [m]
CENTER, RADIUS, STIFFNESS = np.array([0.0, 0.12]), 0.02, 50.0  # m, m, N/m

with MotorUsbController(timeout_ms=20, max_command_rate_hz=500) as robot:
    robot.initialize()
    while True:
        q = (robot.m0.q, robot.m1.q)
        delta = fk(*q, *GEOMETRY) - CENTER
        distance = np.linalg.norm(delta)
        direction = delta / distance if distance > 1e-12 else np.array([1.0, 0.0])
        magnitude = CONTACT_FORCE + STIFFNESS * (RADIUS - distance) if distance < RADIUS - 1e-12 else 0.0
        force = magnitude * direction
        tau = jacobian(*q, *GEOMETRY).T @ force
        for motor, current in zip((robot.m0, robot.m1), tau / KT):
            motor.set(kp=0.0, kd=0.0, iff=current)
        robot.update(wait_state=True, timeout_s=0.02)
