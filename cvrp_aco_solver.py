from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from cvrp_utils import (
    CVRPInstance,
    build_distance,
    collect_instances,
    is_feasible,
    parse_sol_cost,
    parse_vrp,
    route_cost,
    route_load,
    total_cost,
)


@dataclass
class ACOConfig:
    ants: int = 40
    iterations: int = 180
    alpha: float = 1.0
    beta: float = 4.0
    rho: float = 0.15
    q0: float = 0.55
    xi: float = 0.08
    stall_iterations: int = 45
    time_limit_s: float = 3.5
    restarts: int = 2
    local_search_rounds: int = 45


@dataclass
class AntSolution:
    routes: List[List[int]]
    cost: int
    feasible: bool
    fitness: float


def route_edges(route: Sequence[int], depot: int) -> Iterable[Tuple[int, int]]:
    if not route:
        return
    yield depot, route[0]
    for i in range(len(route) - 1):
        yield route[i], route[i + 1]
    yield route[-1], depot


def greedy_baseline_cost(inst: CVRPInstance, dist: Dict[Tuple[int, int], int]) -> int:
    unvisited = set(inst.customers)
    routes: List[List[int]] = []

    while unvisited:
        route: List[int] = []
        load = 0
        cur = inst.depot

        while True:
            feasible = [c for c in unvisited if load + inst.demand[c] <= inst.capacity]
            if not feasible:
                break
            nxt = min(feasible, key=lambda c: dist[(cur, c)])
            route.append(nxt)
            unvisited.remove(nxt)
            load += inst.demand[nxt]
            cur = nxt

        routes.append(route)

    return total_cost(routes, inst.depot, dist)


def two_opt_route(route: List[int], depot: int, dist: Dict[Tuple[int, int], int]) -> Tuple[List[int], bool]:
    if len(route) < 4:
        return route, False

    changed = False
    best = route

    while True:
        improved = False
        n = len(best)

        for i in range(n - 2):
            a = depot if i == 0 else best[i - 1]
            b = best[i]
            for j in range(i + 1, n - 1):
                c = best[j]
                d = best[j + 1]
                delta = (dist[(a, c)] + dist[(b, d)]) - (dist[(a, b)] + dist[(c, d)])
                if delta < 0:
                    best = best[:i] + list(reversed(best[i : j + 1])) + best[j + 1 :]
                    changed = True
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return best, changed


def local_search_routes(
    routes: List[List[int]],
    inst: CVRPInstance,
    dist: Dict[Tuple[int, int], int],
    max_rounds: int,
) -> List[List[int]]:
    routes = [r[:] for r in routes if r]

    for _ in range(max_rounds):
        changed = False

        for ridx, r in enumerate(routes):
            nr, ok = two_opt_route(r, inst.depot, dist)
            if ok:
                routes[ridx] = nr
                changed = True

        loads = [route_load(r, inst.demand) for r in routes]
        moved = False

        for a_idx in range(len(routes)):
            if moved:
                break
            a = routes[a_idx]
            if not a:
                continue

            for i, c in enumerate(a):
                dem = inst.demand[c]
                prev_a = inst.depot if i == 0 else a[i - 1]
                next_a = inst.depot if i == len(a) - 1 else a[i + 1]
                remove_delta = dist[(prev_a, next_a)] - dist[(prev_a, c)] - dist[(c, next_a)]

                for b_idx in range(len(routes)):
                    if a_idx == b_idx:
                        continue
                    if loads[b_idx] + dem > inst.capacity:
                        continue
                    b = routes[b_idx]

                    for pos in range(len(b) + 1):
                        left = inst.depot if pos == 0 else b[pos - 1]
                        right = inst.depot if pos == len(b) else b[pos]
                        insert_delta = dist[(left, c)] + dist[(c, right)] - dist[(left, right)]

                        if remove_delta + insert_delta < 0:
                            routes[a_idx] = a[:i] + a[i + 1 :]
                            routes[b_idx] = b[:pos] + [c] + b[pos:]
                            routes = [r for r in routes if r]
                            moved = True
                            changed = True
                            break
                    if moved:
                        break
                if moved:
                    break

        if not changed:
            break

    return routes


def choose_next_customer(
    current: int,
    candidates: Sequence[int],
    tau: List[List[float]],
    dist: Dict[Tuple[int, int], int],
    cfg: ACOConfig,
    rng: random.Random,
) -> int:
    if len(candidates) == 1:
        return candidates[0]

    desirability: List[Tuple[int, float]] = []
    for c in candidates:
        inv_d = 1.0 / max(1.0, float(dist[(current, c)]))
        score = (tau[current][c] ** cfg.alpha) * (inv_d ** cfg.beta)
        desirability.append((c, score))

    if rng.random() < cfg.q0:
        return max(desirability, key=lambda x: x[1])[0]

    s = sum(v for _, v in desirability)
    if s <= 0:
        return rng.choice(list(candidates))

    r = rng.random() * s
    acc = 0.0
    for c, v in desirability:
        acc += v
        if acc >= r:
            return c
    return desirability[-1][0]


def try_empty_route(
    donor_idx: int,
    routes: List[List[int]],
    loads: List[int],
    inst: CVRPInstance,
    dist: Dict[Tuple[int, int], int],
) -> bool:
    donor = routes[donor_idx]
    if not donor:
        return True

    w_routes = [r[:] for r in routes]
    w_loads = loads[:]

    for c in donor:
        dem = inst.demand[c]
        best = None

        for ridx, r in enumerate(w_routes):
            if ridx == donor_idx:
                continue
            if w_loads[ridx] + dem > inst.capacity:
                continue

            for pos in range(len(r) + 1):
                left = inst.depot if pos == 0 else r[pos - 1]
                right = inst.depot if pos == len(r) else r[pos]
                delta = dist[(left, c)] + dist[(c, right)] - dist[(left, right)]
                if best is None or delta < best[0]:
                    best = (delta, ridx, pos)

        if best is None:
            return False

        _, ridx, pos = best
        w_routes[ridx].insert(pos, c)
        w_loads[ridx] += dem

    w_routes[donor_idx] = []
    w_loads[donor_idx] = 0

    routes[:] = [r for r in w_routes if r]
    loads[:] = [route_load(r, inst.demand) for r in routes]
    return True


def best_merge(
    r1: List[int],
    r2: List[int],
    inst: CVRPInstance,
    dist: Dict[Tuple[int, int], int],
) -> Tuple[List[int], int]:
    base = route_cost(r1, inst.depot, dist) + route_cost(r2, inst.depot, dist)
    variants = [
        r1 + r2,
        r1 + list(reversed(r2)),
        list(reversed(r1)) + r2,
        list(reversed(r1)) + list(reversed(r2)),
    ]

    best_r = variants[0]
    best_c = route_cost(best_r, inst.depot, dist)
    for cand in variants[1:]:
        c = route_cost(cand, inst.depot, dist)
        if c < best_c:
            best_c = c
            best_r = cand

    return best_r, best_c - base


def repair_to_k(
    routes: List[List[int]],
    inst: CVRPInstance,
    dist: Dict[Tuple[int, int], int],
    k_limit: Optional[int],
) -> List[List[int]]:
    if k_limit is None or len(routes) <= k_limit:
        return [r for r in routes if r]

    routes = [r[:] for r in routes if r]
    loads = [route_load(r, inst.demand) for r in routes]

    while len(routes) > k_limit:
        best = None
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                if loads[i] + loads[j] > inst.capacity:
                    continue
                merged, delta = best_merge(routes[i], routes[j], inst, dist)
                if best is None or delta < best[0]:
                    best = (delta, i, j, merged)

        if best is None:
            break

        _, i, j, merged = best
        routes[i] = merged
        loads[i] = route_load(merged, inst.demand)
        del routes[j]
        del loads[j]

    while len(routes) > k_limit:
        idx = min(range(len(routes)), key=lambda k: (len(routes[k]), loads[k]))
        if not try_empty_route(idx, routes, loads, inst, dist):
            break

    return [r for r in routes if r]


def construct_ant_solution(
    inst: CVRPInstance,
    dist: Dict[Tuple[int, int], int],
    tau: List[List[float]],
    tau0: float,
    cfg: ACOConfig,
    rng: random.Random,
) -> AntSolution:
    unvisited = set(inst.customers)
    total_unserved_demand = sum(inst.demand[c] for c in unvisited)
    routes: List[List[int]] = []

    k_limit = inst.k_hint

    while unvisited:
        route: List[int] = []
        load = 0
        current = inst.depot

        while True:
            feasible = [c for c in unvisited if load + inst.demand[c] <= inst.capacity]
            if not feasible:
                break

            if k_limit is not None:
                filtered: List[int] = []
                remaining_routes = max(0, k_limit - len(routes) - 1)
                for c in feasible:
                    projected_load = load + inst.demand[c]
                    rem_demand = total_unserved_demand - inst.demand[c]
                    current_slack = inst.capacity - projected_load
                    rem_after_current = max(0, rem_demand - current_slack)
                    lb_additional_routes = math.ceil(rem_after_current / inst.capacity) if rem_after_current > 0 else 0
                    if lb_additional_routes <= remaining_routes:
                        filtered.append(c)
                if filtered:
                    feasible = filtered

            if not feasible:
                break

            nxt = choose_next_customer(current, feasible, tau, dist, cfg, rng)
            route.append(nxt)
            unvisited.remove(nxt)
            total_unserved_demand -= inst.demand[nxt]
            load += inst.demand[nxt]

            tau[current][nxt] = (1.0 - cfg.xi) * tau[current][nxt] + cfg.xi * tau0
            tau[nxt][current] = tau[current][nxt]
            current = nxt

        if not route:
            c = max(unvisited, key=lambda x: inst.demand[x])
            route = [c]
            unvisited.remove(c)
            total_unserved_demand -= inst.demand[c]

        routes.append(route)

    routes = repair_to_k(routes, inst, dist, k_limit)
    feasible = is_feasible(routes, inst, k_limit=k_limit)
    cost = total_cost(routes, inst.depot, dist)

    penalty = 0.0
    if k_limit is not None and len(routes) > k_limit:
        penalty += 25000.0 * (len(routes) - k_limit) ** 2
    if not feasible:
        penalty += 1_000_000.0

    return AntSolution(routes=routes, cost=cost, feasible=feasible, fitness=cost + penalty)


def single_aco_run(
    inst: CVRPInstance,
    dist: Dict[Tuple[int, int], int],
    cfg: ACOConfig,
    seed: int,
) -> Tuple[List[List[int]], int, bool, float]:
    rng = random.Random(seed)
    customers = inst.customers
    n = inst.dimension

    if not customers:
        return [], 0, True, 0.0

    base_cost = max(1, greedy_baseline_cost(inst, dist))
    tau0 = 1.0 / (len(customers) * base_cost)
    tau = [[tau0 for _ in range(n + 1)] for _ in range(n + 1)]

    best_feasible: Optional[AntSolution] = None
    best_any: Optional[AntSolution] = None

    stall = 0
    iterations_done = 0
    started = time.perf_counter()

    for it in range(cfg.iterations):
        iterations_done = it + 1
        if cfg.time_limit_s > 0 and (time.perf_counter() - started) >= cfg.time_limit_s:
            break

        ants: List[AntSolution] = [construct_ant_solution(inst, dist, tau, tau0, cfg, rng) for _ in range(cfg.ants)]
        iter_best = min(ants, key=lambda a: a.fitness)

        if best_any is None or iter_best.fitness < best_any.fitness:
            best_any = iter_best

        iter_best_feasible = min((a for a in ants if a.feasible), key=lambda a: a.cost, default=None)
        improved = False
        if iter_best_feasible is not None and (best_feasible is None or iter_best_feasible.cost < best_feasible.cost):
            best_feasible = iter_best_feasible
            improved = True

        evap = 1.0 - cfg.rho
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i == j:
                    continue
                tau[i][j] *= evap
                if tau[i][j] < 1e-12:
                    tau[i][j] = 1e-12

        update_sol = iter_best_feasible if iter_best_feasible is not None else iter_best
        if update_sol.cost > 0:
            delta = 1.0 / update_sol.cost
            for r in update_sol.routes:
                for u, v in route_edges(r, inst.depot):
                    tau[u][v] += delta
                    tau[v][u] = tau[u][v]

        if best_feasible is not None and best_feasible.cost > 0:
            delta_g = 1.5 / best_feasible.cost
            for r in best_feasible.routes:
                for u, v in route_edges(r, inst.depot):
                    tau[u][v] += delta_g
                    tau[v][u] = tau[u][v]

        if improved:
            stall = 0
        else:
            stall += 1

        if stall >= cfg.stall_iterations:
            break

    picked = best_feasible if best_feasible is not None else best_any
    if picked is None:
        return [], 0, False, float(iterations_done)

    routes = [r[:] for r in picked.routes]
    cost = picked.cost
    feasible = picked.feasible and is_feasible(routes, inst, k_limit=inst.k_hint)
    return routes, cost, feasible, float(iterations_done)


def solve_instance_aco(
    path: str,
    cfg: ACOConfig,
    seed: int = 42,
) -> Tuple[List[List[int]], int, bool, Dict[str, float]]:
    inst = parse_vrp(path)
    dist = build_distance(inst)

    if not inst.customers:
        return [], 0, True, {"iterations": 0.0, "elapsed_s": 0.0, "restarts": 0.0}

    started = time.perf_counter()
    best: Optional[Tuple[List[List[int]], int, bool]] = None
    total_iterations = 0.0

    restarts = max(1, cfg.restarts)
    for r in range(restarts):
        local_seed = seed + r * 1_000_003
        routes, cost, feasible, iters = single_aco_run(inst, dist, cfg, local_seed)
        total_iterations += iters

        if cfg.local_search_rounds > 0 and routes:
            improved = local_search_routes(routes, inst, dist, max_rounds=cfg.local_search_rounds)
            if is_feasible(improved, inst, k_limit=inst.k_hint):
                improved_cost = total_cost(improved, inst.depot, dist)
                if improved_cost < cost:
                    routes = improved
                    cost = improved_cost
                    feasible = True

        candidate = (routes, cost, feasible)
        if best is None:
            best = candidate
        else:
            if candidate[2] and not best[2]:
                best = candidate
            elif candidate[2] == best[2] and candidate[1] < best[1]:
                best = candidate

    assert best is not None
    elapsed_s = time.perf_counter() - started
    routes, cost, feasible = best

    return routes, cost, feasible, {
        "iterations": total_iterations,
        "elapsed_s": elapsed_s,
        "restarts": float(restarts),
    }


def solve_instances_aco(
    instance_paths: Sequence[str],
    output_csv: str,
    cfg: ACOConfig,
    seed: int = 42,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for idx, path in enumerate(instance_paths):
        local_seed = seed + idx * 1009
        t0 = time.perf_counter()
        routes, cost, feasible, meta = solve_instance_aco(path, cfg, seed=local_seed)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        best_known = parse_sol_cost(str(Path(path).with_suffix(".sol")))
        gap = None
        if best_known is not None and feasible:
            gap = (cost - best_known) / best_known * 100.0

        rows.append(
            {
                "instance": Path(path).stem,
                "path": str(Path(path).resolve()),
                "family": Path(path).stem[0],
                "n": int(Path(path).stem.split("-")[1][1:]),
                "routes": len(routes),
                "cost": int(cost),
                "best_known": best_known if best_known is not None else "",
                "gap_percent": f"{gap:.2f}" if gap is not None else "",
                "feasible": int(feasible),
                "time_ms": f"{elapsed_ms:.1f}",
                "aco_iterations": int(meta["iterations"]),
                "aco_elapsed_s": f"{meta['elapsed_s']:.3f}",
                "aco_restarts": int(meta["restarts"]),
            }
        )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance",
                "path",
                "family",
                "n",
                "routes",
                "cost",
                "best_known",
                "gap_percent",
                "feasible",
                "time_ms",
                "aco_iterations",
                "aco_elapsed_s",
                "aco_restarts",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def solve_all_aco(
    base_dir: str,
    output_csv: str,
    cfg: ACOConfig,
    seed: int = 42,
    limit: Optional[int] = None,
) -> List[Dict[str, object]]:
    instances = collect_instances(base_dir)
    if limit is not None:
        instances = instances[:limit]

    rows = solve_instances_aco(instances, output_csv, cfg, seed=seed)

    feasible_rows = [r for r in rows if int(r["feasible"]) == 1]
    gaps = [float(r["gap_percent"]) for r in feasible_rows if r["gap_percent"] != ""]
    times = [float(r["time_ms"]) for r in rows]

    print(f"Solved: {len(rows)} instances")
    print(f"Feasible: {len(feasible_rows)}/{len(rows)}")
    if gaps:
        print(f"Average gap: {statistics.mean(gaps):.2f}%")
        print(f"Median gap: {statistics.median(gaps):.2f}%")
        print(f"Best gap: {min(gaps):.2f}%")
        print(f"Worst gap: {max(gaps):.2f}%")
    if times:
        print(f"Average runtime: {statistics.mean(times):.1f} ms")
    print(f"Results saved to: {output_csv}")

    return rows


def run_tuning(base_dir: str, output_csv: str, seed: int = 42) -> None:
    a_instances = sorted(glob.glob(os.path.join(base_dir, "A", "*.vrp")))
    b_instances = sorted(glob.glob(os.path.join(base_dir, "B", "*.vrp")))
    tuning_pool = a_instances + b_instances
    if not tuning_pool:
        raise FileNotFoundError("No A/B tuning instances found. Expected folders: base_dir/A and base_dir/B")

    if len(tuning_pool) <= 12:
        tuning_instances = tuning_pool
    else:
        idxs = [round(i * (len(tuning_pool) - 1) / 11) for i in range(12)]
        tuning_instances = [tuning_pool[i] for i in idxs]

    print("Tuning mode: A/B only")
    print(f"A instances found: {len(a_instances)}")
    print(f"B instances found: {len(b_instances)}")
    print(f"Tuning subset size: {len(tuning_instances)}")
    print("Tuning subset (first 6):")
    for p in tuning_instances[:6]:
        print(f"  - {p}")

    grid = [
        ACOConfig(
            ants=28,
            iterations=120,
            alpha=1.0,
            beta=3.8,
            rho=0.18,
            q0=0.50,
            xi=0.10,
            stall_iterations=30,
            time_limit_s=2.3,
            restarts=1,
            local_search_rounds=25,
        ),
        ACOConfig(
            ants=40,
            iterations=180,
            alpha=1.0,
            beta=4.2,
            rho=0.15,
            q0=0.55,
            xi=0.08,
            stall_iterations=45,
            time_limit_s=3.5,
            restarts=2,
            local_search_rounds=45,
        ),
        ACOConfig(
            ants=52,
            iterations=210,
            alpha=1.1,
            beta=4.5,
            rho=0.12,
            q0=0.60,
            xi=0.06,
            stall_iterations=55,
            time_limit_s=4.7,
            restarts=2,
            local_search_rounds=55,
        ),
    ]

    rows: List[Dict[str, object]] = []

    for idx, cfg in enumerate(grid, start=1):
        tmp_csv = Path(base_dir) / f"_tmp_aco_tune_{idx}.csv"
        results = solve_instances_aco(tuning_instances, str(tmp_csv), cfg, seed=seed + idx * 17)

        feas = [r for r in results if int(r["feasible"]) == 1]
        gaps = [float(r["gap_percent"]) for r in feas if r["gap_percent"] != ""]
        times = [float(r["time_ms"]) for r in results]

        rows.append(
            {
                "config_id": idx,
                "ants": cfg.ants,
                "iterations": cfg.iterations,
                "alpha": cfg.alpha,
                "beta": cfg.beta,
                "rho": cfg.rho,
                "q0": cfg.q0,
                "xi": cfg.xi,
                "stall_iterations": cfg.stall_iterations,
                "time_limit_s": cfg.time_limit_s,
                "restarts": cfg.restarts,
                "local_search_rounds": cfg.local_search_rounds,
                "instances": len(results),
                "feasible": len(feas),
                "avg_gap": f"{statistics.mean(gaps):.2f}" if gaps else "",
                "median_gap": f"{statistics.median(gaps):.2f}" if gaps else "",
                "avg_time_ms": f"{statistics.mean(times):.1f}" if times else "",
            }
        )

        try:
            tmp_csv.unlink()
        except FileNotFoundError:
            pass

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config_id",
                "ants",
                "iterations",
                "alpha",
                "beta",
                "rho",
                "q0",
                "xi",
                "stall_iterations",
                "time_limit_s",
                "restarts",
                "local_search_rounds",
                "instances",
                "feasible",
                "avg_gap",
                "median_gap",
                "avg_time_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Tuning results saved to: {output_csv}")
    for r in rows:
        print(
            "Config #{config_id}: feasible {feasible}/{instances}, avg_gap={avg_gap}%, avg_time={avg_time_ms} ms".format(
                **r
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ant Colony Optimization solver for CVRP benchmarks")

    parser.add_argument("--instance", type=str, default=None, help="Path to one .vrp instance")
    parser.add_argument("--all", action="store_true", help="Run all E/F/M/P instances")
    parser.add_argument("--tune", action="store_true", help="Run parameter tuning on representative instances")

    parser.add_argument("--base-dir", type=str, default=".", help="Base directory with E/F/M/P folders")
    parser.add_argument("--output-csv", type=str, default="cvrp_aco_results.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for --all")

    parser.add_argument("--ants", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--rho", type=float, default=0.15)
    parser.add_argument("--q0", type=float, default=0.55)
    parser.add_argument("--xi", type=float, default=0.08)
    parser.add_argument("--stall-iterations", type=int, default=45)
    parser.add_argument("--time-limit-s", type=float, default=3.5)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--local-search-rounds", type=int, default=45)

    args = parser.parse_args()

    cfg = ACOConfig(
        ants=args.ants,
        iterations=args.iterations,
        alpha=args.alpha,
        beta=args.beta,
        rho=args.rho,
        q0=args.q0,
        xi=args.xi,
        stall_iterations=args.stall_iterations,
        time_limit_s=args.time_limit_s,
        restarts=args.restarts,
        local_search_rounds=args.local_search_rounds,
    )

    if args.tune:
        run_tuning(args.base_dir, args.output_csv, seed=args.seed)
        return

    if args.instance:
        routes, cost, feasible, meta = solve_instance_aco(args.instance, cfg, seed=args.seed)
        print(f"Instance: {args.instance}")
        print(f"Feasible: {feasible}")
        print(f"Cost: {cost}")
        print(f"Routes: {len(routes)}")
        print(f"ACO iterations: {int(meta['iterations'])}")
        print(f"ACO restarts: {int(meta['restarts'])}")
        print(f"ACO elapsed: {meta['elapsed_s']:.3f} s")
        for idx, r in enumerate(routes, start=1):
            print(f"Route #{idx}: {' '.join(map(str, r))}")
        return

    if args.all:
        solve_all_aco(
            base_dir=args.base_dir,
            output_csv=args.output_csv,
            cfg=cfg,
            seed=args.seed,
            limit=args.limit,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
