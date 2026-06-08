import yaml
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True) # froze=True emulates immutability by making class read-only
class PipelineSettings:
    adl_csv_path: str
    zone_mappings: dict[str, list[str]]

def load_settings(yaml_path: str = "config.yaml") -> PipelineSettings:
    path  = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError("Configuration file not found {path.absolute()}")
    with open(path, "r") as f:
        config_data = yaml.safe_load(f)

    pipeline_data = config_data.get("adl_envelope_pipeline", {})
    return PipelineSettings(
        adl_csv_path=pipeline_data.get("adl_csv_path"),
        zone_mappings=pipeline_data.get("zone_mappings", {}),
    )

settings = load_settings()