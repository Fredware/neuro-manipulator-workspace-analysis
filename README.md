# neuro-manipulator-workspace-analysis

Workspace-coverage analysis and geometry optimization for the **neuro-manipulator** —
a 6-DOF wheelchair-mounted assistive arm built from Series Elastic Actuator (SEA)
modules — evaluated against ADL (Activities of Daily Living) workspace envelopes
derived from motion-capture data and simulated in MuJoCo.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management
- `git` (the `ada_assets` wheelchair + human models are a submodule)
- `dvc` for large-file versioning (installed as a dev dependency)
- Access to the project's UBox DVC remote (for `clean_adl_kinematics.csv`)

## Installation
### Fresh-clone setup

```bash
git clone <repo-url> && cd neuro-manipulator-workspace-analysis
git submodule update --init      # pull ada_assets (Permobil wheelchair + seated human)
uv sync                          # create the venv and install all dependencies

# Configure DVC remote for your machine (path is machine-specific, gitignored)
uv run dvc remote modify --local ubox "<this machine's Box sync path>/neuro-manipulator-dvc"
uv run dvc pull                                      # clean_adl_kinematics.csv from UBox

# Rebuild derived artifacts from the CSV (no MATLAB needed)
uv run python -m modular_arm.analysis.adl_envelope_generator --export-stl
uv run python -m modular_arm.robot.arm_assembler --mode both
```

### If DVC remote not available, regenerate artifacts from raw kinematic data:
```bash
# Fetch raw .mat files from Figshare
uv run python scripts/fetch_data.py data/external/kinematic-emg.yaml

# Export kinematic CSV from the .mat files
uv run python scripts/export_kinematics.py data/kinematic-emg-adl-dataset/

# Rebuild artifacts
uv run python -m modular_arm.analysis.adl_envelope_generator --export-stl
uv run python -m modular_arm.robot.arm_assembler --mode both
```

`uv run <cmd>` auto-syncs before running, so after the initial `uv sync` you can just
use `uv run ...` for everything below. All commands are run from the repository root.

## Package layout

Dependencies flow downward toward `core`; nothing in `core` imports upward.

| Package         | Responsibility                                                              |
|-----------------|-----------------------------------------------------------------------------|
| `core`          | Config loading (`config.py`) and coordinate-frame conventions (`frames.py`) |
| `robot`         | The physical arm model: SEA modules, MJCF assembler, robot config           |
| `scene`         | Integration: arm + wheelchair + seated human composed into one scene        |
| `analysis`      | Produces results/data: envelope generation, Monte Carlo, optimization       |
| `control`       | Controllers and interactive joint-sweep inspection                          |
| `visualization` | Matplotlib plotters for envelopes and workspace clouds                      |

## Data Provenance
Version Control Strategies

| Asset                              | Mechanism           | Reason                        |
|------------------------------------|---------------------|-------------------------------|
| `ada_assets`                       | git submodule       | independent repo              |
| `assets/etd2`                      | git LFS             | small meshes (~3 MB)          |
| `clean_adl_kinematics.csv`         | DVC pointer to UBox | derived from Figshare dataset |
| `.mat` Figshare files              | fetch-by-manifest   | Hosted externally, has a DOI  |
| ADL hulls, STLs, MJCFs, MC results | gitignored          | regenerated from the pipeline |
| Solidworks/STEP CAD files          | DVC UBox            |                               |



## Pipeline

### 1. Generate ADL workspace envelopes from the kinematic dataset

Loads the motion-capture CSV generated from the [Lucchetti et al. dataset](https://doi-org.ezproxy.lib.utah.edu/10.1038/s41597-025-06174-3), 
normalizes to sternum-relative coordinates, and builds convex-hull / Delaunay envelopes per zone. `--export-stl` also writes hulls for MuJoCo.

```bash
uv run python -m modular_arm.analysis.adl_envelope_generator --export-stl --stl-dir data/adl-envelopes/
```

### 2. Define the arm geometry

Edit the centralized robot definition (link lengths, masses, joint ranges):

```
src/modular_arm/robot/robot_config.py   # structural_definition
config.yaml                             # tunable lengths
```

### 3. Generate the robot MJCF (component + full scene)

```bash
uv run python -m modular_arm.robot.arm_assembler --mode both
uv run python -m mujoco.viewer --mjcf=data/robot-models/mjcf/neuro_arm.xml
```

### 4. Sanity-check the assembly in the viewer

Steps through discrete joint angles to verify assembly, coordinate frames, and
clearance; the scene assembler can also overlay the ADL hulls.

```bash
uv run python -m modular_arm.control.joint_sweep_verification
uv run python -m modular_arm.scene.assemble_scene
uv run python -m modular_arm.scene.assemble_scene --with-adl-hulls
```

### 5. Monte Carlo workspace sampling

Bypasses the controller loop and samples thousands of random configurations via
forward kinematics to build a dense reachability cloud.

```bash
uv run python -m modular_arm.analysis.monte_carlo_simulation
```

### 6. Optimize link lengths for ADL coverage

Searches link lengths (differential evolution) to maximize volumetric ADL
reachability. Use `--evaluate-nominal` to score the current design and
`--report-reach` to print the required-vs-available reach per zone.

```bash
uv run python -m modular_arm.analysis.optimize_link_lengths --report-reach
uv run python -m modular_arm.analysis.optimize_link_lengths --evaluate-nominal
uv run python -m modular_arm.analysis.optimize_link_lengths --maxiter 150 --workers -1
```

### 7. Visualize results

```bash
uv run python -m modular_arm.visualization.plot_arm_envelope    # workspace cloud vs ADL zones
uv run python -m modular_arm.visualization.plot_adl_hulls_3d      # HS vs ST envelope comparison
```
## External Data
The kinematic dataset is from Lucchetti et al. 2025.
>>> Lucchetti, F., Bailo, G., & Lencioni, T. A Detailed Kinematic and EMG Dataset for Upper Limb and Hand Movement Analysis in Post-Stroke and Healthy Subjects During Functional Daily Tasks. Scientific Data 12:1904. DOI: 10.6084/m9.figshare.c.7720187.v1

The manifest at `data/external/kinematic-emg.yaml` pins the collection version and `scripts/fetch_data.py` handles the download. `scripts/export_kinematics.py` replaces the original MATLAB export with a pure-Python pipeline.
