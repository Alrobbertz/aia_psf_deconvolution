from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_root: Path
    runs_root: Path
    scratch_root: Path
    psf_cache_root: Path
    deconvolved_cache_root: Path
    psf_source: str = "grigis_2012"
    deconvolution_method: str = "aiapy_rl"
    iterations: int = 25
    clip_negative: bool = True
    random_seed: int = 20260730

    @classmethod
    def from_env(cls) -> Settings:
        data_root_resolved = Path(
            os.getenv("AIA_DATA_ROOT", "data/aia_scene_experiment")
        ).resolve()
        data_root_parent = data_root_resolved.parent
        return cls(
            data_root=data_root_resolved,
            runs_root=Path(os.getenv("AIA_RUNS_ROOT", "runs")).resolve(),
            scratch_root=Path(os.getenv("AIA_SCRATCH_ROOT", "scratch")).resolve(),
            psf_cache_root=Path(
                os.getenv("AIA_PSF_CACHE_ROOT", str(data_root_parent / "psfs"))
            ).resolve(),
            deconvolved_cache_root=Path(
                os.getenv(
                    "AIA_DECONVOLVED_CACHE_ROOT",
                    str(data_root_parent / "deconvolved_scenes"),
                )
            ).resolve(),
            psf_source=os.getenv("AIA_PSF_SOURCE", "grigis_2012"),
            deconvolution_method=os.getenv("AIA_DECONV_METHOD", "aiapy_rl"),
            iterations=int(os.getenv("AIA_RL_ITERATIONS", "25")),
            clip_negative=_env_bool("AIA_CLIP_NEGATIVE", True),
            random_seed=int(os.getenv("AIA_RANDOM_SEED", "20260730")),
        )

    def as_serializable(self) -> dict[str, object]:
        values = asdict(self)
        for key in (
            "data_root",
            "runs_root",
            "scratch_root",
            "psf_cache_root",
            "deconvolved_cache_root",
        ):
            values[key] = str(values[key])
        return values

    @property
    def psf_cache_dir(self) -> Path:
        """PSF cache directory: psf_cache_root / psf_source."""
        return self.psf_cache_root / self.psf_source

    @property
    def reference_cache_dir(self) -> Path:
        """Reference cache directory: deconvolved_cache_root / deconvolution_method / psf_source."""
        return self.deconvolved_cache_root / self.deconvolution_method / self.psf_source

    def config_hash(self, treatment_ids: list[str]) -> str:
        payload = {
            "settings": self.as_serializable(),
            "treatments": treatment_ids,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def create_run_dir(self) -> tuple[str, Path]:
        self.runs_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}_{uuid4().hex[:8]}"
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=False, exist_ok=False)
        return run_id, run_dir
