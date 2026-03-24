import heapq
import numpy as np


def astar_on_grid(
    z: np.ndarray,
    start_rc: tuple[int, int],
    goal_rc: tuple[int, int],
    step_cost_fn,
    diagonal: bool = True,
):
    """Genel amaçlı A* algoritması."""
    height, width = z.shape
    start_row, start_col = start_rc
    goal_row, goal_col = goal_rc

    if diagonal:
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def heuristic(row: int, col: int) -> float:
        return float(np.hypot(goal_row - row, goal_col - col))

    g_cost = np.full((height, width), np.inf, dtype=np.float64)
    parent_row = np.full((height, width), -1, dtype=np.int32)
    parent_col = np.full((height, width), -1, dtype=np.int32)
    closed = np.zeros((height, width), dtype=bool)

    g_cost[start_row, start_col] = 0.0
    priority_queue = [(heuristic(start_row, start_col), 0.0, start_row, start_col)]

    while priority_queue:
        _, current_g, row, col = heapq.heappop(priority_queue)

        if closed[row, col]:
            continue

        closed[row, col] = True

        if (row, col) == (goal_row, goal_col):
            break

        for delta_row, delta_col in neighbors:
            next_row = row + delta_row
            next_col = col + delta_col

            if next_row < 0 or next_row >= height or next_col < 0 or next_col >= width:
                continue
            if closed[next_row, next_col]:
                continue

            step_cost = step_cost_fn(row, col, next_row, next_col)
            if step_cost is None:
                continue

            new_g = current_g + step_cost

            if new_g < g_cost[next_row, next_col]:
                g_cost[next_row, next_col] = new_g
                parent_row[next_row, next_col] = row
                parent_col[next_row, next_col] = col
                heapq.heappush(
                    priority_queue,
                    (new_g + heuristic(next_row, next_col), new_g, next_row, next_col),
                )

    if not np.isfinite(g_cost[goal_row, goal_col]):
        return None, np.inf

    path = []
    row, col = goal_row, goal_col

    while True:
        path.append((row, col))
        if (row, col) == (start_row, start_col):
            break

        prev_row = parent_row[row, col]
        prev_col = parent_col[row, col]

        if prev_row < 0 or prev_col < 0:
            return None, np.inf

        row, col = int(prev_row), int(prev_col)

    path.reverse()
    return path, float(g_cost[goal_row, goal_col])