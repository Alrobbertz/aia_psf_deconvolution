from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Treatment:
    treatment_id: str
    path: str
    dtype: str
    codec: str
    quantize_level: float | None = None
    quantize_method: int | None = None
    row_strip: bool = False
    is_byte_exact_control: bool = False

    @property
    def is_integer(self) -> bool:
        return self.path == "B"

    @property
    def is_quantized(self) -> bool:
        return self.path == "A" and self.codec == "RICE_1"

    def tile_shape(self, shape: tuple[int, int]) -> tuple[int, int] | None:
        if not self.row_strip:
            return None
        return (1, int(shape[1]))


def _build_registry() -> tuple[Treatment, ...]:
    treatments: list[Treatment] = []
    for dtype in ("float64", "float32"):
        short = "f64" if dtype == "float64" else "f32"
        treatments.extend(
            [
                Treatment(
                    f"A_{short}_none",
                    "A",
                    dtype,
                    "NONE",
                    is_byte_exact_control=True,
                ),
                Treatment(
                    f"A_{short}_gzip1",
                    "A",
                    dtype,
                    "GZIP_1",
                    quantize_level=0.0,
                    is_byte_exact_control=True,
                ),
                Treatment(
                    f"A_{short}_gzip2",
                    "A",
                    dtype,
                    "GZIP_2",
                    quantize_level=0.0,
                    is_byte_exact_control=True,
                ),
            ]
        )
        for level in (8, 16, 32):
            for mode, label in ((-1, "nodither"), (1, "dither1"), (2, "dither2")):
                treatments.append(
                    Treatment(
                        f"A_{short}_rice_q{level}_{label}",
                        "A",
                        dtype,
                        "RICE_1",
                        quantize_level=float(level),
                        quantize_method=mode,
                        row_strip=True,
                    )
                )
    for dtype in ("int16", "int32"):
        short = "i16" if dtype == "int16" else "i32"
        for codec, label in (
            ("NONE", "none"),
            ("GZIP_1", "gzip1"),
            ("GZIP_2", "gzip2"),
            ("RICE_1", "rice_bs32"),
        ):
            treatments.append(
                Treatment(
                    f"B_{short}_{label}",
                    "B",
                    dtype,
                    codec,
                    row_strip=codec == "RICE_1",
                )
            )
    return tuple(treatments)


TREATMENTS = _build_registry()


def select_treatments(requested: list[str] | None) -> tuple[Treatment, ...]:
    if not requested:
        return TREATMENTS
    by_id = {item.treatment_id: item for item in TREATMENTS}
    missing = sorted(set(requested) - by_id.keys())
    if missing:
        raise ValueError(f"Unknown treatment IDs: {', '.join(missing)}")
    return tuple(by_id[item] for item in requested)
