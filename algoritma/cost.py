import numpy as np


class CostParams:
    """Yol planlama için maliyet parametreleri."""

    def __init__(
        self,
        slope_weight: float = 8.0,
        uphill_extra: float = 1.5,
        shadow_weight: float = 10.0,
        crater_weight: float = 0.0,
        max_slope=None,
    ):
        self.slope_weight = float(slope_weight)
        self.uphill_extra = float(uphill_extra)
        self.shadow_weight = float(shadow_weight)
        self.crater_weight = float(crater_weight)
        self.max_slope = max_slope


def compute_local_roughness(z: np.ndarray, row: int, col: int) -> float:
    """3x3 komşulukta yerel arazi pürüzünü ölçer."""
    height, width = z.shape
    row_start = max(0, row - 1)
    row_end = min(height, row + 2)
    col_start = max(0, col - 1)
    col_end = min(width, col + 2)

    patch = z[row_start:row_end, col_start:col_end]
    values = patch[np.isfinite(patch)]
    if values.size == 0:
        return 0.0

    return float(np.std(values))


def compute_shadow_penalty(
    shadow_map: np.ndarray | None,
    row: int,
    col: int,
    shadow_weight: float,
) -> float:
    """Gölge haritası varsa gölgeli hücreye ek ceza verir."""
    if shadow_map is None:
        return 0.0

    return shadow_weight if float(shadow_map[row, col]) > 0.5 else 0.0


def compute_step_cost(
    z: np.ndarray,
    from_row: int,
    from_col: int,
    to_row: int,
    to_col: int,
    params: CostParams,
    shadow_map: np.ndarray | None = None,
) -> float | None:
    """Bir hücreden komşu hücreye geçiş maliyetini hesaplar."""
    base_distance = float(np.hypot(to_row - from_row, to_col - from_col))
    if base_distance <= 0:
        return None

    current_height = float(z[from_row, from_col])
    next_height = float(z[to_row, to_col])

    if not np.isfinite(current_height) or not np.isfinite(next_height):
        return None

    delta_height = next_height - current_height
    slope = abs(delta_height) / max(base_distance, 1e-6)

    if params.max_slope is not None and slope > params.max_slope:
        return None

    slope_penalty = params.slope_weight * slope
    uphill_penalty = max(0.0, delta_height) * params.uphill_extra
    shadow_penalty = compute_shadow_penalty(
        shadow_map=shadow_map,
        row=to_row,
        col=to_col,
        shadow_weight=params.shadow_weight,
    )
    roughness = compute_local_roughness(z, to_row, to_col)
    crater_penalty = params.crater_weight * roughness

    total_cost = (
        base_distance
        + slope_penalty
        + uphill_penalty
        + shadow_penalty
        + crater_penalty
    )
    return float(total_cost)