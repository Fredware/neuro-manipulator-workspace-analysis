## Program commands
1. Specify the physical properties of the ARM
    
   ```{bash}
    src/modular_arm/core/robot_config.py
    ```
   
2. Generate the MJCF XML scene and component files for the robot from robot_config.py
 
   ```{bash}
   uv run python -m src.modular_arm.modules.arm_assembler --mode both
   uv run python -m mujoco.viewer --mjcf=data/robot-models/mjcf/model.xml      
   uv run python -m mujoco.viewer --mjcf=data/robot-models/mjcf/neuro_arm.xml
   ```

3. Sanity check visualization of the ARM

    Runs a state-machine simulation in MuJoCo to step through discrete joint angles.
    Use it to verify physical assembly, coordinate frames and clearance.
    ```{bash}
    uv run pyton -m src.modular_arm.control.configuration_designer
    uv run python -m src.modular_arm.visualization.assemble_scene  
    ```
4. Monte Carlo Volumetric Simulation

   Bypass physics controller loop to rapidly sample thousands of random configurations.
   Uses forward kinematics to generate a dense cloud required for statistical ADL zone intersection matrix.
    ```{bash}
    uv run python -m src.modular_arm.analysis.monte_carlo_simulation
    ```
5. 3D Visualization of the Workspace

    ```{bash}
    uv run python -m src.modular_arm.analysis.visualize_envelope
    ```
6. Generate and visualize envelopes from motion tracking (kinematic) data
   
   ```{bash}
   uv run python -m src.modular_arm.analysis.adl_envelope_generator
   uv run python -m src.modular_arm.analysis.plot_adl_hulls_3d
   ```
## Installation instructions
To work with the ADA assets repo, run `git submodule update --init` to pull `ada_assets`, then `uv sync` to install the rest of the dependencies.

