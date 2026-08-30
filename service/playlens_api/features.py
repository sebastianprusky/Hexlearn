from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError


FEATURE_NAMES = (
    "brightness",
    "contrast",
    "saturation",
    "non_dark_ratio",
    "edge_density",
    "center_brightness",
    "pixel_motion",
    "observed_motion",
    "board_occupancy",
    "near_center_occupancy",
    "outer_stack_occupancy",
    "radial_pressure",
    "sector_imbalance",
    "mean_stack_reach",
    "max_stack_reach",
    "sector_occupancy_0",
    "sector_occupancy_1",
    "sector_occupancy_2",
    "sector_occupancy_3",
    "sector_occupancy_4",
    "sector_occupancy_5",
    "sector_stack_reach_0",
    "sector_stack_reach_1",
    "sector_stack_reach_2",
    "sector_stack_reach_3",
    "sector_stack_reach_4",
    "sector_stack_reach_5",
)
FEATURE_SCHEMA_VERSION = "hextris-stack-v2"


class InvalidFrame(ValueError):
    pass


@dataclass(frozen=True)
class FrameFeatures:
    vector: np.ndarray
    gray: np.ndarray
    jpeg_bytes: bytes


def decode_data_url(image_data_url: str) -> bytes:
    try:
        header, encoded = image_data_url.split(",", 1)
    except ValueError as exc:
        raise InvalidFrame("Frame must be a data URL") from exc
    if not header.startswith("data:image/jpeg;base64"):
        raise InvalidFrame("Only base64 JPEG frames are accepted")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise InvalidFrame("Frame is not valid base64") from exc
    if len(payload) > 500_000:
        raise InvalidFrame("Frame exceeds the 500 KB limit")
    return payload


def extract_frame_features(
    image_data_url: str,
    previous_gray: np.ndarray | None,
    observed_motion: float,
) -> FrameFeatures:
    jpeg_bytes = decode_data_url(image_data_url)
    return extract_jpeg_features(jpeg_bytes, previous_gray, observed_motion)


def extract_jpeg_features(
    jpeg_bytes: bytes,
    previous_gray: np.ndarray | None,
    observed_motion: float,
) -> FrameFeatures:
    try:
        image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB").resize((80, 80))
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidFrame("Frame is not a readable JPEG") from exc

    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    brightness = float(gray.mean())
    contrast = float(gray.std())
    saturation = float((rgb.max(axis=2) - rgb.min(axis=2)).mean())
    non_dark_ratio = float((gray > 0.08).mean())
    horizontal_edges = np.abs(np.diff(gray, axis=1)).mean()
    vertical_edges = np.abs(np.diff(gray, axis=0)).mean()
    edge_density = float(horizontal_edges + vertical_edges)
    center_brightness = float(gray[20:60, 20:60].mean())
    pixel_motion = 0.0 if previous_gray is None else float(np.abs(gray - previous_gray).mean())

    coordinates = np.linspace(-1.0, 1.0, gray.shape[0], dtype=np.float32)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    radius = np.sqrt(x_grid**2 + y_grid**2)
    angle = (np.arctan2(y_grid, x_grid) + 2 * np.pi) % (2 * np.pi)
    colorful = (rgb.max(axis=2) - rgb.min(axis=2)) > 0.16
    stack_inner_radius = 0.12
    stack_outer_radius = 0.43
    board_mask = (radius >= stack_inner_radius) & (radius <= stack_outer_radius)
    near_center_mask = (radius >= stack_inner_radius) & (radius <= 0.27)
    outer_stack_mask = (radius >= 0.30) & (radius <= stack_outer_radius)
    board_occupancy = float(colorful[board_mask].mean())
    near_center_occupancy = float(colorful[near_center_mask].mean())
    outer_stack_occupancy = float(colorful[outer_stack_mask].mean())
    pressure_weights = np.clip(
        (radius - stack_inner_radius) / (stack_outer_radius - stack_inner_radius), 0.0, 1.0
    )
    radial_pressure = float((colorful * pressure_weights * board_mask).sum() / max(1, board_mask.sum()))
    sector_occupancies: list[float] = []
    sector_stack_reaches: list[float] = []
    sector_width = 2 * np.pi / 6
    for index in range(6):
        sector_mask = board_mask & (angle >= index * sector_width) & (angle < (index + 1) * sector_width)
        sector_occupancies.append(float(colorful[sector_mask].mean()))
        occupied_radii = radius[sector_mask & colorful]
        robust_reach = float(np.quantile(occupied_radii, 0.90)) if occupied_radii.size else stack_inner_radius
        sector_stack_reaches.append(
            float(
                np.clip(
                    (robust_reach - stack_inner_radius)
                    / (stack_outer_radius - stack_inner_radius),
                    0.0,
                    1.0,
                )
            )
        )
    sector_imbalance = float(np.std(sector_occupancies))
    mean_stack_reach = float(np.mean(sector_stack_reaches))
    max_stack_reach = float(np.max(sector_stack_reaches))

    vector = np.asarray(
        [
            brightness,
            contrast,
            saturation,
            non_dark_ratio,
            edge_density,
            center_brightness,
            pixel_motion,
            max(0.0, min(float(observed_motion) / 32.0, 1.0)),
            board_occupancy,
            near_center_occupancy,
            outer_stack_occupancy,
            radial_pressure,
            sector_imbalance,
            mean_stack_reach,
            max_stack_reach,
            *sector_occupancies,
            *sector_stack_reaches,
        ],
        dtype=np.float32,
    )
    return FrameFeatures(vector=vector, gray=gray, jpeg_bytes=jpeg_bytes)


def aggregate_window(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("At least one frame is required")
    matrix = np.vstack(vectors)
    start = matrix[: max(1, len(matrix) // 4)].mean(axis=0)
    end = matrix[-max(1, len(matrix) // 4) :].mean(axis=0)
    return np.concatenate(
        [matrix.mean(axis=0), matrix.std(axis=0), matrix.min(axis=0), matrix.max(axis=0), end - start]
    ).astype(np.float32)


def aggregate_feature_names() -> list[str]:
    return [f"{stat}_{name}" for stat in ("mean", "std", "min", "max", "delta") for name in FEATURE_NAMES]
