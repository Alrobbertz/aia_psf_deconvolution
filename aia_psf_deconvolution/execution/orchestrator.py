from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd

from aia_psf_deconvolution.config.settings import Settings
from aia_psf_deconvolution.config.treatments import Treatment
from aia_psf_deconvolution.dataio.fitsio import read_image_header
from aia_psf_deconvolution.dataio.loaders import load_reference
from aia_psf_deconvolution.representation.encoder import encode
from aia_psf_deconvolution.representation.paths import RangeIntegrityError
from aia_psf_deconvolution.results.writer import ResultWriter
from aia_psf_deconvolution.verification.fidelity import fidelity_metrics
from aia_psf_deconvolution.verification.interop import interoperability_checks
from aia_psf_deconvolution.verification.range_integrity import range_integrity_metrics
from aia_psf_deconvolution.verification.sizes import size_metrics

from .scratch import ephemeral_fits


def _failure_reason(metrics: dict[str, object], treatment: Treatment) -> str | None:
    failures: list[str] = []
    if not metrics.get("source_lsb_pass", False):
        failures.append("source_lsb")
    if treatment.is_byte_exact_control and not metrics.get("byte_exact_pass", False):
        failures.append("byte_exact")
    if not metrics.get("range_integrity_pass", False):
        failures.append("range_integrity")
    if not metrics.get("interop_pass", False):
        failures.append("interoperability")
    return ",".join(failures) or None


def _treatment_fields(treatment: Treatment) -> dict[str, object]:
    return {
        "treatment_id": treatment.treatment_id,
        "path": treatment.path,
        "representation_dtype": treatment.dtype,
        "codec": treatment.codec,
        "quantize_level": treatment.quantize_level,
        "quantize_method": treatment.quantize_method,
        "is_byte_exact_control": treatment.is_byte_exact_control,
    }


def run_sweep(
    manifest: pd.DataFrame,
    references: pd.DataFrame,
    treatments: tuple[Treatment, ...],
    settings: Settings,
    writer: ResultWriter,
) -> None:
    reference_by_image = references.set_index("image_id").to_dict("index")
    total = len(manifest) * len(treatments)
    completed = 0

    for image in manifest.itertuples(index=False):
        reference_row = reference_by_image[image.image_id]
        reference = load_reference(Path(reference_row["reference_path"]))
        source_header = read_image_header(Path(image.source_path))
        for treatment in treatments:
            completed += 1
            print(
                f"[sweep {completed}/{total}] {image.image_id} {treatment.treatment_id}",
                flush=True,
            )
            base = {
                "image_id": image.image_id,
                "scene_id": image.scene_id,
                "scene_type": image.scene_type,
                "wavelength_angstrom": int(image.wavelength_angstrom),
                "source_saturated_pixels": int(
                    getattr(image, "source_saturated_pixels", 0)
                ),
                "source_saturated_fraction": float(
                    getattr(image, "source_saturated_fraction", 0.0)
                ),
                "source_is_saturated": bool(
                    getattr(image, "source_is_saturated", False)
                ),
                "source_adc_ceiling_dn": float(
                    getattr(image, "source_adc_ceiling_dn", float("nan"))
                ),
                **_treatment_fields(treatment),
            }
            stem = f"{image.image_id}_{treatment.treatment_id}"
            with ephemeral_fits(settings.scratch_root, stem) as encoded_path:
                try:
                    represented, write_seconds, seed = encode(
                        encoded_path,
                        reference,
                        source_header,
                        image.source_sha256,
                        reference_row["reference_sha256"],
                        treatment,
                        settings.iterations,
                        settings.clip_negative,
                    )
                    fidelity, decoded = fidelity_metrics(
                        encoded_path, reference, represented, treatment
                    )
                    integrity = range_integrity_metrics(
                        reference, represented, decoded, treatment
                    )
                    interop = interoperability_checks(
                        encoded_path, source_header, treatment, tuple(reference.shape)
                    )
                    sizes = size_metrics(
                        encoded_path,
                        int(image.source_file_bytes),
                        int(reference_row["reference_bytes"]),
                        int(image.source_integer_bytes),
                        int(reference.size),
                    )
                    metrics = {
                        **fidelity,
                        **integrity,
                        **interop,
                        **sizes,
                        "write_seconds": write_seconds,
                        "dither_seed": seed,
                    }
                    reason = _failure_reason(metrics, treatment)
                    writer.append(
                        {
                            **base,
                            "status": "completed",
                            "overall_pass": reason is None,
                            "failure_reason": reason,
                            **metrics,
                        }
                    )
                    del represented, decoded
                except RangeIntegrityError as exc:
                    writer.append(
                        {
                            **base,
                            "status": "failed",
                            "overall_pass": False,
                            "failure_stage": "representation",
                            "failure_reason": exc.reason,
                            "error": str(exc),
                            "observed_min": exc.observed_min,
                            "observed_max": exc.observed_max,
                        }
                    )
                except Exception as exc:
                    writer.append(
                        {
                            **base,
                            "status": "error",
                            "overall_pass": False,
                            "failure_stage": "encode_or_verify",
                            "failure_reason": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(limit=8),
                        }
                    )
        del reference
