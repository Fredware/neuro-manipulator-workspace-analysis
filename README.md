## Program commands
1. Load SEA chain into MuJoCo
```powershell
python -m mujoco.viewer --mjcf .\src\modular_arm\modules\test_arm.xml
```

```powershell
uv run python -m src.modular_arm.control.sweeper
uv run python -m src.modular_arm.analysis.visualize_envelope
```

1. Specify the physical properties of the ARM
```{bash}
src/modular_arm/core/robot_config.py
```
2. Sanity check visualization of the ARM
Runs a state-machine simulation in MuJoCo to step through discrete joint angles.
Use it to verify physical assembly, coordinate frames and clearance.
```{bash}
uv run pyton -m src.modular_arm.control.configuration_designer
```
3. Monte Carlo Volumetric Simulation
Bypass physics controller loop to rapidly sample thousands of random configurations.
Uses forward kinematics to generate a dense cloud required for statistical ADL zone intersection matrix.
```{bash}
uv run python -m src.modular_arm.analysis.monte_carlo_simulation
```
4. 3D Visualization of the Workspace
```{bash}
uv run python -m src.modular_arm.analysis.visualize_envelope
```