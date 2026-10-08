# Virtual-ball haptics

Run the minimal script from the repository root:

```bash
python3 -m examples.haptic_ball
```

[`haptic_ball.py`](haptic_ball.py) uses the installed `motkin_pcb` library.
Follow the [repository setup](../README.md#python-examples) to install dependencies.
The script has no command-line arguments; edit its constants directly.

Edit the constants at the top to change the ball: center `(0, 120)` mm,
radius 20 mm, stiffness 50 N/m. Move the tip by hand: force is zero outside,
and pushes radially outward inside with magnitude
`CONTACT_FORCE + STIFFNESS * penetration`. `CONTACT_FORCE` defaults to 0.5 N;
increase it for a firmer initial contact. At the exact center, it pushes toward +X.

The script computes `tau = J.T @ force`, then commands `iff = tau / KT`
with `kp = kd = 0` and measured **KT = 0.062 Nm/A**.

Geometry uses meters; encoder angles are clockwise-positive from north.
Use the upper-tip assembly. Commands run at up to 500 Hz with a 20 ms
watchdog. Ctrl+C exits the context and disables the motors.
