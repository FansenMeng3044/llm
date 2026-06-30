"""OR-Tools route generator and evaluator for pkl-only RALTestSets.

This runner is intentionally separate from CTAS-D.py and TACO.py. It does not
reproduce the original Gurobi CTAS-D or TACO solvers; it builds OR-Tools VRP
routes directly from TaskEnv pickle files and evaluates them through the
repository's existing execute_by_route path.
"""
import argparse
import contextlib
import glob
import io
import math
import os
import pickle
import sys
from itertools import combinations, product

import numpy as np
import pandas as pd
from natsort import natsorted
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.task_env import TaskEnv  # noqa: E402

import __main__  # noqa: E402

__main__.TaskEnv = TaskEnv

DEFAULT_DATA_ROOT = os.path.join(ROOT_DIR, "RALTestSets")
METRIC_KEYS = ["success_rate", "makespan", "time_cost", "waiting_time", "travel_dist", "efficiency"]


def load_env(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def euclidean(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def choose_species_counts(requirements, abilities, species_limits):
    req = np.asarray(requirements, dtype=float)
    if np.all(req <= 0):
        return [0] * len(species_limits)

    abilities = np.asarray(abilities, dtype=float)
    max_agents = max(1, int(math.ceil(float(np.sum(req)))))
    ranges = []
    for species, limit in enumerate(species_limits):
        if np.sum(abilities[species]) <= 0:
            ranges.append(range(1))
        else:
            ranges.append(range(min(int(limit), max_agents) + 1))

    best = None
    best_score = None
    for counts in product(*ranges):
        total_agents = sum(counts)
        if total_agents == 0:
            continue
        coverage = np.matmul(np.asarray(counts, dtype=float), abilities)
        if np.all(coverage >= req):
            excess = float(np.sum(coverage - req))
            score = (total_agents, excess, counts)
            if best_score is None or score < best_score:
                best = list(counts)
                best_score = score

    if best is None:
        raise RuntimeError(
            "No feasible species coalition for task requirements "
            f"{requirements}; species abilities={abilities.tolist()}"
        )
    return best


def build_task_species_counts(env):
    species_limits = [int(x) for x in env.species_dict["number"]]
    abilities = np.asarray(env.species_dict["abilities"], dtype=float)
    counts_by_task = {}
    for task_id, task in env.task_dic.items():
        counts_by_task[task_id] = choose_species_counts(task["requirements"], abilities, species_limits)
    return counts_by_task


def make_transit_callback(manager, locations, service_times, scale):
    def transit(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel = euclidean(locations[from_node], locations[to_node]) / 0.2
        service = service_times[to_node]
        return int(round((travel + service) * scale))

    return transit


def fallback_species_routes(env, species, visit_task_ids):
    agent_ids = list(env.species_dict[species])
    routes = {agent_id: [0] for agent_id in agent_ids}
    route_tasks = {agent_id: set() for agent_id in agent_ids}
    route_locs = {agent_id: np.asarray(env.depot_dic[species]["location"]) for agent_id in agent_ids}

    for task_id in visit_task_ids:
        task_loc = np.asarray(env.task_dic[task_id]["location"])
        candidates = [a for a in agent_ids if task_id not in route_tasks[a]] or agent_ids
        agent_id = min(candidates, key=lambda a: (len(routes[a]), euclidean(route_locs[a], task_loc), a))
        routes[agent_id].append(task_id + 1)
        route_tasks[agent_id].add(task_id)
        route_locs[agent_id] = task_loc

    for agent_id in agent_ids:
        routes[agent_id].append(0)
    return routes


def solve_species_routes(env, species, visit_task_ids, time_limit, scale):
    agent_ids = list(env.species_dict[species])
    if not visit_task_ids:
        return {agent_id: [0, 0] for agent_id in agent_ids}

    locations = [np.asarray(env.depot_dic[species]["location"], dtype=float)]
    service_times = [0.0]
    for task_id in visit_task_ids:
        task = env.task_dic[task_id]
        locations.append(np.asarray(task["location"], dtype=float))
        service_times.append(float(np.asarray(task["time"]).reshape(-1)[0]))

    manager = pywrapcp.RoutingIndexManager(len(locations), len(agent_ids), 0)
    routing = pywrapcp.RoutingModel(manager)
    transit_index = routing.RegisterTransitCallback(make_transit_callback(manager, locations, service_times, scale))
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)
    routing.AddDimension(transit_index, 0, 10**9, True, "Time")
    routing.GetDimensionOrDie("Time").SetGlobalSpanCostCoefficient(100)

    duplicate_nodes = {}
    for node_id, task_id in enumerate(visit_task_ids, start=1):
        duplicate_nodes.setdefault(task_id, []).append(node_id)
    for nodes in duplicate_nodes.values():
        for left, right in combinations(nodes, 2):
            routing.solver().Add(
                routing.VehicleVar(manager.NodeToIndex(left)) != routing.VehicleVar(manager.NodeToIndex(right))
            )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = max(1, int(time_limit))

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        return fallback_species_routes(env, species, visit_task_ids)

    routes = {}
    for vehicle_id, agent_id in enumerate(agent_ids):
        route = [0]
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            node = manager.IndexToNode(index)
            if node != 0:
                route.append(visit_task_ids[node - 1] + 1)
        route.append(0)
        routes[agent_id] = route
    return routes


def build_ortools_routes(env, time_limit, scale):
    counts_by_task = build_task_species_counts(env)
    all_routes = [[0, 0] for _ in range(env.agents_num)]
    for species in range(env.species_num):
        visits = []
        for task_id in sorted(env.task_dic):
            visits.extend([task_id] * int(counts_by_task[task_id][species]))
        for agent_id, route in solve_species_routes(env, species, visits, time_limit, scale).items():
            all_routes[agent_id] = route
    return all_routes


def replay_routes(env, routes, instance_dir, quiet=True):
    env.init_state()
    for agent_id, route in enumerate(routes):
        env.pre_set_route(list(route[1:]), agent_id)
    env.force_wait = True
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            env.execute_by_route(instance_dir, "ortools", False)
    else:
        env.execute_by_route(instance_dir, "ortools", False)
    _, finished_tasks = env.get_episode_reward(100)
    return metrics_from_env(env, finished_tasks)


def metrics_from_env(env, finished_tasks):
    success_rate = float(np.sum(finished_tasks) / len(finished_tasks))
    if success_rate < 1:
        return {key: (success_rate if key == "success_rate" else np.nan) for key in METRIC_KEYS}
    return {
        "success_rate": success_rate,
        "makespan": float(env.current_time),
        "time_cost": float(np.nanmean(np.nan_to_num(env.get_matrix(env.task_dic, "time_start"), nan=100))),
        "waiting_time": float(np.mean(env.get_matrix(env.agent_dic, "sum_waiting_time"))),
        "travel_dist": float(np.sum(env.get_matrix(env.agent_dic, "travel_dist"))),
        "efficiency": float(np.mean(env.get_efficiency())),
    }


def run_instance(path, time_limit, scale, save_solution, quiet):
    env = load_env(path)
    routes = build_ortools_routes(env, time_limit=time_limit, scale=scale)
    instance_dir = os.path.splitext(path)[0]
    os.makedirs(instance_dir, exist_ok=True)
    if save_solution:
        with open(os.path.join(instance_dir, "ortools.solution"), "wb") as f:
            pickle.dump(routes, f)
    metrics = replay_routes(env, routes, instance_dir, quiet=quiet)
    metrics["env"] = os.path.splitext(os.path.basename(path))[0]
    return metrics


def list_dataset_dirs(data_root, dataset, run_all):
    if run_all:
        dirs = [p for p in glob.glob(os.path.join(data_root, "RALTestSet_M*_*")) if os.path.isdir(p)]
        return natsorted(dirs, key=lambda y: y.lower())
    if not dataset:
        raise SystemExit("Pass --dataset RALTestSet_M2_1 or --all.")
    path = os.path.join(data_root, dataset)
    if not os.path.isdir(path):
        raise SystemExit(f"Dataset not found: {path}")
    return [path]


def run_dataset(dataset_dir, limit, time_limit, scale, save_solution, quiet):
    files = natsorted(glob.glob(os.path.join(dataset_dir, "env_*.pkl")), key=lambda y: y.lower())
    if limit is not None:
        files = files[:limit]
    if not files:
        raise SystemExit(f"No env_*.pkl files found under {dataset_dir}")

    rows = []
    for index, path in enumerate(files, start=1):
        print(f"[{os.path.basename(dataset_dir)}] {index}/{len(files)} {os.path.basename(path)}")
        try:
            rows.append(run_instance(path, time_limit, scale, save_solution, quiet))
        except Exception as exc:
            row = {"env": os.path.splitext(os.path.basename(path))[0], **{key: np.nan for key in METRIC_KEYS}}
            row["error"] = str(exc)
            rows.append(row)
            print(f"  failed: {exc}")

    columns = ["env"] + METRIC_KEYS + (["error"] if any("error" in row for row in rows) else [])
    df = pd.DataFrame(rows).reindex(columns=columns)
    out_name = "ortools.csv" if limit is None else f"ortools_limit_{limit}.csv"
    out_path = os.path.join(dataset_dir, out_name)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Run OR-Tools routes on RALTestSets pkl files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Dataset name under RALTestSets, e.g. RALTestSet_M2_1")
    group.add_argument("--all", action="store_true", help="Run every RALTestSet_M*_* dataset")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Root containing RALTestSet_M*_* folders")
    parser.add_argument("--limit", type=int, default=None, help="Limit instances per dataset")
    parser.add_argument("--time-limit", type=int, default=1, help="OR-Tools time limit per species, in seconds")
    parser.add_argument("--scale", type=int, default=1000, help="Integer cost scale for OR-Tools")
    parser.add_argument("--no-save", action="store_true", help="Do not write env_i/ortools.solution")
    parser.add_argument("--verbose", action="store_true", help="Show execute_by_route output")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = os.path.abspath(args.data_root)
    for dataset_dir in list_dataset_dirs(data_root, args.dataset, args.all):
        run_dataset(
            dataset_dir,
            limit=args.limit,
            time_limit=args.time_limit,
            scale=args.scale,
            save_solution=not args.no_save,
            quiet=not args.verbose,
        )


if __name__ == "__main__":
    main()
