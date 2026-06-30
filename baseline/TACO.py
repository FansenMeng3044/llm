"""Replay precomputed solution routes through the TaskEnv evaluator.

This script does not implement the original TACO solver. It can replay true
`taco.solution` files when they exist, or replay OR-Tools-generated
`ortools.solution` files through the same evaluation path.
"""
import argparse
import contextlib
import glob
import io
import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd
from natsort import natsorted

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.task_env import TaskEnv  # noqa: E402

import __main__  # noqa: E402

__main__.TaskEnv = TaskEnv

DEFAULT_DATA_ROOT = os.path.join(ROOT_DIR, "RALTestSets")
METRIC_KEYS = ["success_rate", "makespan", "time_cost", "waiting_time", "travel_dist", "efficiency"]
VALID_METHODS = ("ortools", "taco", "sas")


def load_env(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_routes(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def preset_routes(env, routes):
    if routes is None:
        return False
    for agent_id, route in enumerate(routes):
        env.pre_set_route(list(route[1:]), agent_id)
    return True


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


def replay_solution(env_path, method, quiet=True):
    instance_dir = os.path.splitext(env_path)[0]
    solution_path = os.path.join(instance_dir, f"{method}.solution")
    routes = load_routes(solution_path)
    if routes is None:
        row = {key: np.nan for key in METRIC_KEYS}
        row["error"] = f"missing {method}.solution"
        return row

    env = load_env(env_path)
    env.init_state()
    if not preset_routes(env, routes):
        row = {key: np.nan for key in METRIC_KEYS}
        row["error"] = f"empty {method}.solution"
        return row

    env.force_wait = True
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            env.execute_by_route(instance_dir, method, False)
    else:
        env.execute_by_route(instance_dir, method, False)
    _, finished_tasks = env.get_episode_reward(100)
    return metrics_from_env(env, finished_tasks)


def output_name(method, limit):
    prefix = "ortools_taco_replay" if method == "ortools" else f"{method}_replay"
    if limit is not None:
        return f"{prefix}_limit_{limit}.csv"
    return f"{prefix}.csv"


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


def dataset_env_files(dataset_dir, limit):
    files = natsorted(glob.glob(os.path.join(dataset_dir, "env_*.pkl")), key=lambda y: y.lower())
    if limit is not None:
        files = files[:limit]
    if not files:
        raise SystemExit(f"No env_*.pkl files found under {dataset_dir}")
    return files


def copy_ortools_to_taco(dataset_dir, limit):
    files = dataset_env_files(dataset_dir, limit)
    copied = 0
    missing = 0
    for env_path in files:
        instance_dir = os.path.splitext(env_path)[0]
        src = os.path.join(instance_dir, "ortools.solution")
        dst = os.path.join(instance_dir, "taco.solution")
        if not os.path.exists(src):
            missing += 1
            print(f"missing {src}")
            continue
        shutil.copy2(src, dst)
        copied += 1
    print(f"[{os.path.basename(dataset_dir)}] copied {copied} ortools.solution files to taco.solution; missing={missing}")
    return copied, missing


def run_dataset(dataset_dir, method, limit, quiet):
    files = dataset_env_files(dataset_dir, limit)
    rows = []
    for index, env_path in enumerate(files, start=1):
        print(f"[{os.path.basename(dataset_dir)}] {index}/{len(files)} {os.path.basename(env_path)} via {method}.solution")
        try:
            row = replay_solution(env_path, method, quiet=quiet)
        except Exception as exc:
            row = {key: np.nan for key in METRIC_KEYS}
            row["error"] = str(exc)
            print(f"  failed: {exc}")
        row["env"] = os.path.splitext(os.path.basename(env_path))[0]
        rows.append(row)

    columns = ["env"] + METRIC_KEYS + (["error"] if any("error" in row for row in rows) else [])
    df = pd.DataFrame(rows).reindex(columns=columns)
    out_path = os.path.join(dataset_dir, output_name(method, limit))
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Replay TACO/SAS/OR-Tools solution files through TaskEnv.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Dataset name under RALTestSets, e.g. RALTestSet_M2_1")
    group.add_argument("--all", action="store_true", help="Run every RALTestSet_M*_* dataset")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Root containing RALTestSet_M*_* folders")
    parser.add_argument("--method", choices=VALID_METHODS, default=None, help="Solution prefix to replay")
    parser.add_argument("--limit", type=int, default=None, help="Limit instances per dataset")
    parser.add_argument("--copy-ortools-to-taco", action="store_true", help="Copy env_i/ortools.solution to env_i/taco.solution")
    parser.add_argument("--verbose", action="store_true", help="Show execute_by_route output")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = os.path.abspath(args.data_root)
    dataset_dirs = list_dataset_dirs(data_root, args.dataset, args.all)

    if args.copy_ortools_to_taco:
        for dataset_dir in dataset_dirs:
            copy_ortools_to_taco(dataset_dir, args.limit)

    if args.method is None:
        if not args.copy_ortools_to_taco:
            raise SystemExit("Pass --method ortools|taco|sas or --copy-ortools-to-taco.")
        return

    for dataset_dir in dataset_dirs:
        run_dataset(dataset_dir, args.method, args.limit, quiet=not args.verbose)


if __name__ == "__main__":
    main()
