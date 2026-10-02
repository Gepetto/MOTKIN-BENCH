# MOTKINBENCH

MOTKINBENCH is a low-cost platform with two permanent-magnet synchronous motors (PMSMs)
for learning robotics and control on real hardware. The same motor base supports
two independent flywheels or a planar five-bar robot with two degrees of freedom.

The goal is to make it easy to experiment with motor control, robot kinematics,
and feedback control, and connect simulation with hands-on hardware experiments.

| Independent flywheels | Five-bar planar robot |
| --- | --- |
| ![MOTKINBENCH with two independent flywheels](TAPE_W.png) | ![MOTKINBENCH with a planar five-bar linkage](TAPE_5B.png) |

The motors use a custom [Raspberry Pi Pico dual-PMSM driver](https://github.com/thomasfla/pico_dual_PMSM_BU79100G_DRV8316C).

- [Mechanical design in Onshape](https://cad.onshape.com/documents/581c9cc37f21d16430f878db/)
- [URDF export and simulation guide](urdf/README.md)
- [Python five-bar kinematics and plotting](kinematics/README.md)
- [CNRS pen drawing and motor trajectory](drawing/README.md)
- [Virtual-ball haptic control with the Jacobian](examples/README.md)

## Materials

| Quantity | Item |
| --- | --- |
| 2 | GM3506 gimbal motor with encoder |
| 18 | M2.5 × 6 mm socket head cap screw |
| 4 | M3 × 5 mm socket head cap screw |
| 1 set | PLA printed parts for the chosen setup — [Onshape CAD](https://cad.onshape.com/documents/581c9cc37f21d16430f878db/) |
| 1 | Custom dual-PMSM motor driver — [GitHub](https://github.com/thomasfla/pico_dual_PMSM_BU79100G_DRV8316C) |

## Python examples

Run commands from this repository's root. `kinematics` is a normal Python
package; for example, `from kinematics import fk, ik, jacobian`.

```bash
python3 -m pip install -r kinematics/requirements.txt
python3 -m kinematics.plot_fivebar
python3 -m drawing.cnrs
python3 -m drawing.draw_robot --dry-run
```

Hardware examples use `MotorUsbController` from the
[Pico dual-PMSM control library](https://github.com/thomasfla/pico_dual_PMSM_BU79100G_DRV8316C),
located in its `software/motor_usb` directory. Install `pyserial` and add the
checkout's `software` directory to your Python path (adjust this local path):

```bash
python3 -m pip install pyserial
export PYTHONPATH="/home/tflayols/git/pico_dual_PMSM_BUG79100G_DRV8316C/software${PYTHONPATH:+:$PYTHONPATH}"
python3 -m drawing.draw_robot
```

See the [haptic example](examples/README.md) for pure torque control using
`tau = J.T @ force`. Both controllers use calibrated absolute encoders:
zero is north, positive angles turn clockwise, and `m0` is the left motor.

Run all hardware-free tests with:

```bash
python3 -m unittest discover -v
```
