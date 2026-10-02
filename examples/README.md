# Virtual-ball haptics

Run the minimal script from the repository root:

```bash
python3 -m examples.haptic_ball
```

[`haptic_ball.py`](haptic_ball.py) finds `motor_usb` in the neighboring
`pico_dual_PMSM_BUG79100G_DRV8316C/software` directory. Edit that path if
needed; `pyserial` must be installed.

Edit the constants at the top to change the ball: center `(0, 120)` mm,
radius 20 mm, stiffness 50 N/m. Move the tip by hand: force is zero outside,
and pushes radially outward inside with magnitude
`CONTACT_FORCE + STIFFNESS * penetration`. `CONTACT_FORCE` defaults to 0.001 N;
increase it for a firmer initial contact. At the exact center, it pushes toward +X.

The script computes `tau = J.T @ force`, then commands `iff = tau / KT`
with `kp = kd = 0`. `KT = 0.08` Nm/A is an approximate GM3506 value, not a
calibrated torque constant. The [manufacturer's specifications](https://shop.iflight.com/gimbal-motors-cat44/ipower-motor-gm3506-brushless-gimbal-motor-w-as5048a-encoder-pro1155)
give 600–1000 g·cm at 1 A, without specifying the current convention.
Replace `KT` with your measured torque/Iq ratio for accurate forces.

Geometry uses meters; encoder angles are clockwise-positive from north.
Use the upper-tip assembly. Commands run at up to 500 Hz with a 20 ms
watchdog. Ctrl+C exits the context and disables the motors.
