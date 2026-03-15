from __future__ import annotations

import glob
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class CVRPInstance:
    name: str
    dimension: int
    capacity: int
    coords: Dict[int, Tuple[float, float]]
    demand: Dict[int, int]
    depot: int
    edge_weight_type: str = "EUC_2D"
    explicit_dist: Optional[Dict[Tuple[int, int], int]] = None
    k_hint: Optional[int] = None

    @property
    def customers(self) -> List[int]:
        return [n for n in range(1, self.dimension + 1) if n != self.depot]


def euc_2d(a: Tuple[float, float], b: Tuple[float, float]) -> int:
    return int(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) + 0.5)


def parse_vrp(path: str) -> CVRPInstance:
    name = ""
    dimension = 0
    capacity = 0
    coords: Dict[int, Tuple[float, float]] = {}
    demand: Dict[int, int] = {}
    depot = 1
    edge_weight_type = "EUC_2D"
    edge_weight_values: List[int] = []

    mode = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("NAME"):
                name = line.split(":", 1)[1].strip()
                continue
            if line.startswith("CAPACITY"):
                capacity = int(line.split(":", 1)[1].strip())
                continue
            if line.startswith("DIMENSION"):
                dimension = int(line.split(":", 1)[1].strip())
                continue
            if line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split(":", 1)[1].strip()
                continue
            if line == "NODE_COORD_SECTION":
                mode = "coords"
                continue
            if line == "EDGE_WEIGHT_SECTION":
                mode = "weights"
                continue
            if line == "DEMAND_SECTION":
                mode = "demand"
                continue
            if line == "DEPOT_SECTION":
                mode = "depot"
                continue
            if line == "EOF":
                break

            if mode == "coords":
                node, x, y = line.split()
                coords[int(node)] = (float(x), float(y))
            elif mode == "weights":
                edge_weight_values.extend(int(x) for x in line.split())
            elif mode == "demand":
                node, d = line.split()
                demand[int(node)] = int(d)
            elif mode == "depot":
                v = int(line)
                if v == -1:
                    mode = None
                else:
                    depot = v

    k_match = re.search(r"-k(\d+)", name)
    k_hint = int(k_match.group(1)) if k_match else None
    explicit_dist: Optional[Dict[Tuple[int, int], int]] = None

    if edge_weight_type.upper() == "EXPLICIT":
        if dimension <= 0:
            raise ValueError(f"Invalid DIMENSION in {path}")
        needed = dimension * (dimension - 1) // 2
        if len(edge_weight_values) < needed:
            raise ValueError(f"Not enough EDGE_WEIGHT_SECTION values in {path}: got {len(edge_weight_values)}, need {needed}")
        explicit_dist = {}
        idx = 0
        for i in range(1, dimension + 1):
            explicit_dist[(i, i)] = 0
            for j in range(1, i):
                w = edge_weight_values[idx]
                idx += 1
                explicit_dist[(i, j)] = w
                explicit_dist[(j, i)] = w
        if not coords:
            coords = {i: (0.0, 0.0) for i in range(1, dimension + 1)}
    elif dimension <= 0:
        dimension = len(coords)

    return CVRPInstance(
        name=name,
        dimension=dimension,
        capacity=capacity,
        coords=coords,
        demand=demand,
        depot=depot,
        edge_weight_type=edge_weight_type,
        explicit_dist=explicit_dist,
        k_hint=k_hint,
    )


def parse_sol_cost(path: str) -> Optional[int]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"Cost\s+(\d+)", line)
            if m:
                return int(m.group(1))
    return None


def build_distance(inst: CVRPInstance) -> Dict[Tuple[int, int], int]:
    if inst.explicit_dist is not None:
        return inst.explicit_dist

    nodes = list(range(1, inst.dimension + 1))
    dist: Dict[Tuple[int, int], int] = {}
    for i in nodes:
        for j in nodes:
            if i == j:
                dist[(i, j)] = 0
            else:
                dist[(i, j)] = euc_2d(inst.coords[i], inst.coords[j])
    return dist


def route_cost(route: Sequence[int], depot: int, dist: Dict[Tuple[int, int], int]) -> int:
    if not route:
        return 0
    total = dist[(depot, route[0])]
    for i in range(len(route) - 1):
        total += dist[(route[i], route[i + 1])]
    total += dist[(route[-1], depot)]
    return total


def total_cost(routes: Sequence[Sequence[int]], depot: int, dist: Dict[Tuple[int, int], int]) -> int:
    return sum(route_cost(r, depot, dist) for r in routes)


def route_load(route: Sequence[int], demand: Dict[int, int]) -> int:
    return sum(demand[c] for c in route)


def is_feasible(routes: Sequence[Sequence[int]], inst: CVRPInstance, k_limit: Optional[int] = None) -> bool:
    seen = set()
    for r in routes:
        if route_load(r, inst.demand) > inst.capacity:
            return False
        for c in r:
            if c in seen:
                return False
            seen.add(c)
    if set(inst.customers) != seen:
        return False
    if k_limit is not None and len([r for r in routes if r]) > k_limit:
        return False
    return True


def collect_instances(base_dir: str) -> List[str]:
    patterns = [
        os.path.join(base_dir, "E", "*.vrp"),
        os.path.join(base_dir, "F", "*.vrp"),
        os.path.join(base_dir, "M", "*.vrp"),
        os.path.join(base_dir, "P", "*.vrp"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(files)
