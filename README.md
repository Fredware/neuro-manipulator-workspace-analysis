## Program commands
1. Load SEA chain into MuJoCo
```powershell
python -m mujoco.viewer --mjcf .\src\modular_arm\modules\test_arm.xml
```

```powershell
uv run python -m src.modular_arm.control.sweeper
uv run python -m src.modular_arm.analysis.visualize_envelope
```