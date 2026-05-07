import numpy as np
from pogema import GridConfig

from heapq import heappop, heappush

INF = 1e7


class GridMemory:
    def __init__(self, start_r=64):
        self._memory = np.zeros(shape=(start_r * 2 + 1, start_r * 2 + 1), dtype=np.bool_)

    @staticmethod
    def _try_to_insert(x, y, source, target):
        r = source.shape[0] // 2
        try:
            target[x - r:x + r + 1, y - r:y + r + 1] = source
            return True
        except ValueError:
            return False

    def _increase_memory(self):
        m = self._memory
        r = self._memory.shape[0]
        self._memory = np.zeros(shape=(r * 2 + 1, r * 2 + 1))
        assert self._try_to_insert(r, r, m, self._memory)

    def update(self, x, y, obstacles):
        while True:
            r = self._memory.shape[0] // 2
            if self._try_to_insert(r + x, r + y, obstacles, self._memory):
                break
            self._increase_memory()

    def is_obstacle(self, x, y):
        r = self._memory.shape[0] // 2
        if -r <= x <= r and -r <= y <= r:
            return self._memory[r + x, r + y]
        return False


class Node:
    def __init__(self, coord: tuple[int, int] = (INF, INF), g: int = 0, h: int = 0):
        self.i, self.j = coord
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        if self.f != other.f:
            return self.f < other.f
        if self.g != other.g:
            return self.g < other.g
        return self.i < other.i or self.j < other.j


def h(node, target):
    nx, ny = node
    tx, ty = target
    return abs(nx - tx) + abs(ny - ty)


def a_star(start, target, grid: GridMemory, max_steps=10000):
    open_ = []
    closed = {start: None}
    heappush(open_, Node(start, 0, h(start, target)))

    for step in range(int(max_steps)):
        if not open_:
            break
        u = heappop(open_)
        if (u.i, u.j) == target:
            break

        for n in [(u.i - 1, u.j), (u.i, u.j + 1), (u.i + 1, u.j), (u.i, u.j - 1)]:
            if not grid.is_obstacle(*n) and n not in closed:
                heappush(open_, Node(n, u.g + 1, h(n, target)))
                closed[n] = (u.i, u.j)

    next_node = target if target in closed else None
    path = []
    while next_node is not None:
        path.append(next_node)
        next_node = closed[next_node]
    return list(reversed(path))


class AStarAgent:
    _DELTA_TO_HEADING = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}

    def __init__(self, seed=0):
        self._cfg = GridConfig()
        self._gm = None
        self._saved_xy = None
        self.clear_state()
        self._rnd = np.random.default_rng(seed)

    def _action_towards(self, heading: int, next_delta: tuple[int, int]) -> int:
        desired_heading = self._DELTA_TO_HEADING.get(next_delta)
        if desired_heading is None:
            return self._cfg.ACTION_WAIT
        turn = (desired_heading - int(heading)) % 4
        if turn == 0:
            return self._cfg.ACTION_FORWARD
        if turn == 3:
            return self._cfg.ACTION_TURN_LEFT
        if turn == 1:
            return self._cfg.ACTION_TURN_RIGHT
        return self._cfg.ACTION_TURN_RIGHT

    def act(self, obs):
        xy = tuple(obs['xy'])
        target_xy = tuple(obs['target_xy'])
        obstacles = obs['obstacles']
        heading = int(obs.get('heading', obs.get('global_heading', 0)))

        if self._saved_xy is not None and h(self._saved_xy, xy) > 1:
            raise IndexError("Agent moved more than 1 step. Please call clear_state before a new episode.")
        if self._saved_xy is not None and h(self._saved_xy, xy) == 0 and xy != target_xy:
            return int(self._rnd.integers(self._cfg.get_num_actions()))

        self._gm.update(*xy, obstacles)
        path = a_star(xy, target_xy, self._gm)
        if len(path) <= 1:
            action = self._cfg.ACTION_WAIT
        else:
            (x, y), (tx, ty), *_ = path
            action = self._action_towards(heading, (tx - x, ty - y))

        self._saved_xy = xy
        return action

    def clear_state(self):
        self._saved_xy = None
        self._gm = GridMemory()


class BatchAStarAgent:
    def __init__(self):
        self.astar_agents = {}

    def act(self, observations):
        actions = []
        for idx, obs in enumerate(observations):
            if idx not in self.astar_agents:
                self.astar_agents[idx] = AStarAgent()
            actions.append(self.astar_agents[idx].act(obs))
        return actions

    def reset_states(self):
        self.astar_agents = {}
