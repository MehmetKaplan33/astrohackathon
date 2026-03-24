import numpy as np

from algoritma.astar import astar_on_grid
from algoritma.cost import CostParams, compute_step_cost


def plan_path(
    z: np.ndarray,
    start_rc: tuple[int, int],
    goal_rc: tuple[int, int],
    diagonal: bool = True,
    slope_weight: float = 8.0,
    uphill_extra: float = 1.5,
    max_slope=None,
    shadow_map: np.ndarray | None = None,
    shadow_weight: float = 10.0,
    crater_weight: float = 0.0,
):
    """GUI tarafının çağıracağı üst seviye planlama fonksiyonu."""
    params = CostParams(
        slope_weight=slope_weight,
        uphill_extra=uphill_extra,
        shadow_weight=shadow_weight,
        crater_weight=crater_weight,
        max_slope=max_slope,
    )

    def step_cost_fn(from_row: int, from_col: int, to_row: int, to_col: int):
        return compute_step_cost(
            z=z,
            from_row=from_row,
            from_col=from_col,
            to_row=to_row,
            to_col=to_col,
            params=params,
            shadow_map=shadow_map,
        )

    return astar_on_grid(
        z=z,
        start_rc=start_rc,
        goal_rc=goal_rc,
        step_cost_fn=step_cost_fn,
        diagonal=diagonal,
    )