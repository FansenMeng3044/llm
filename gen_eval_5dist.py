"""One-shot: build M2-M5 test sets over the five location distributions, run the
greedy heuristic on every instance, and export per-scale JSON + Markdown.

Layout produced under --out-root (default: RALtest_5dist/):
    M2RALtest/  env_0.pkl ... env_99.pkl   manifest.json  greedy.json  greedy.md
    M3RALtest/  ...
    M4RALtest/  ...
    M5RALtest/  ...
    summary.json   summary.md

Each M<k>RALtest holds 100 instances = 20 per distribution x 5 distributions
(uniform, gmm, ring, bipolar, edges), concatenated into one test set.

Scales (paper Table I; k_n=agents, k_s=species=5, k_m=tasks, k_b=skills=5):
    M2: 15 agents (3/species),  20 tasks
    M3: 25 agents (5/species),  20 tasks
    M4: 25 agents (5/species),  50 tasks
    M5: 50 agents (10/species), 50 tasks

Usage:
    python gen_eval_5dist.py                     # full run, all scales, 20/dist
    python gen_eval_5dist.py --per-dist 20 --out-root RALtest_5dist
    python gen_eval_5dist.py --no-save-pkl       # don't persist the instances
"""
import os
import io
import json
import pickle
import argparse
import contextlib

import numpy as np

from env.task_env import TaskEnv
from parameters import EnvParams

DISTS = ['uniform', 'gmm', 'ring', 'bipolar', 'edges']
METRIC_KEYS = ['success_rate', 'makespan', 'time_cost', 'waiting_time', 'travel_dist', 'efficiency']

# k_n = species * per_species ; k_s = species ; k_m = tasks ; k_b = traits_dim
SCALES = {
    'M2': {'per_species': 3,  'species': 5, 'tasks': 20},
    'M3': {'per_species': 5,  'species': 5, 'tasks': 20},
    'M4': {'per_species': 5,  'species': 5, 'tasks': 50},
    'M5': {'per_species': 10, 'species': 5, 'tasks': 50},
}


def build_env(scale_cfg, dist, seed):
    ps, sp, tk = scale_cfg['per_species'], scale_cfg['species'], scale_cfg['tasks']
    return TaskEnv((ps, ps), (sp, sp), (tk, tk), EnvParams.TRAIT_DIM, EnvParams.DECISION_DIM,
                   seed=seed, location_dist=dist)


def instance_plan(scale_name, per_dist):
    """Yield (global_i, dist, seed) for a scale, reproducibly. Shared by all evaluators
    so the same seeds map to the same instances everywhere."""
    scale_idx = list(SCALES).index(scale_name)
    gi = 0
    for dist_idx, dist in enumerate(DISTS):
        for i in range(per_dist):
            yield gi, dist, 900000 + scale_idx * 100000 + dist_idx * 10000 + i
            gi += 1


def metrics_from_env(env, finished):
    """Same metric extraction as baseline/ORTools.py so results are comparable."""
    sr = float(np.sum(finished) / len(finished))
    if sr < 1:  # infeasible/deadlock -> only success_rate is meaningful
        return {k: (sr if k == 'success_rate' else float('nan')) for k in METRIC_KEYS}
    return {
        'success_rate': sr,
        'makespan': float(env.current_time),
        'time_cost': float(np.nanmean(np.nan_to_num(env.get_matrix(env.task_dic, 'time_start'), nan=100))),
        'waiting_time': float(np.mean(env.get_matrix(env.agent_dic, 'sum_waiting_time'))),
        'travel_dist': float(np.sum(env.get_matrix(env.agent_dic, 'travel_dist'))),
        'efficiency': float(np.mean(env.get_efficiency())),
    }


def run_greedy(env):
    env.init_state()
    with contextlib.redirect_stdout(io.StringIO()):
        env.execute_greedy_action()
    _, finished = env.get_episode_reward(EnvParams.MAX_TIME)
    return metrics_from_env(env, finished)


def aggregate(rows):
    """Mean over instances; success_rate over all, other metrics over solved only."""
    out = {'instances': len(rows),
           'success_rate': round(float(np.mean([r['success_rate'] for r in rows])), 4)}
    for k in ['makespan', 'time_cost', 'waiting_time', 'travel_dist', 'efficiency']:
        vals = [r[k] for r in rows if not np.isnan(r[k])]
        out[k] = round(float(np.mean(vals)), 4) if vals else None
    out['solved'] = int(sum(1 for r in rows if r['success_rate'] >= 1))
    return out


def scale_markdown(name, cfg, per_dist_agg, overall):
    kn = cfg['per_species'] * cfg['species']
    lines = [f'# {name}RALtest  (greedy)',
             '',
             f"config: k_n={kn} agents ({cfg['per_species']}/species), k_s={cfg['species']} species, "
             f"k_m={cfg['tasks']} tasks, k_b={EnvParams.TRAIT_DIM} skills  |  gamma={cfg['tasks']/kn:.2f}",
             '',
             '| distribution | instances | solved | success | makespan | waiting | travel_dist | efficiency |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for d in DISTS + ['OVERALL']:
        a = overall if d == 'OVERALL' else per_dist_agg[d]
        lines.append(f"| {d} | {a['instances']} | {a['solved']} | {a['success_rate']} | "
                     f"{a['makespan']} | {a['waiting_time']} | {a['travel_dist']} | {a['efficiency']} |")
    return '\n'.join(lines) + '\n'


def run_scale(name, cfg, per_dist, out_root, save_pkl):
    scale_idx = list(SCALES).index(name)
    scale_dir = os.path.join(out_root, f'{name}RALtest')
    os.makedirs(scale_dir, exist_ok=True)

    per_instance, manifest = [], []
    per_dist_rows = {d: [] for d in DISTS}
    global_i = 0
    for dist_idx, dist in enumerate(DISTS):
        for i in range(per_dist):
            seed = 900000 + scale_idx * 100000 + dist_idx * 10000 + i  # reproducible, disjoint from training
            env = build_env(cfg, dist, seed)
            if save_pkl:
                with open(os.path.join(scale_dir, f'env_{global_i}.pkl'), 'wb') as f:
                    pickle.dump(env, f)
            m = run_greedy(env)
            row = {'env': f'env_{global_i}', 'dist': dist, 'seed': seed,
                   **{k: (round(m[k], 4) if not np.isnan(m[k]) else None) for k in METRIC_KEYS}}
            per_instance.append(row)
            per_dist_rows[dist].append(m)
            manifest.append({'env': f'env_{global_i}', 'dist': dist, 'seed': seed})
            global_i += 1
        print(f'  {name} / {dist:8s}: done {per_dist} instances')

    per_dist_agg = {d: aggregate(per_dist_rows[d]) for d in DISTS}
    overall = aggregate([m for d in DISTS for m in per_dist_rows[d]])
    kn = cfg['per_species'] * cfg['species']

    report = {'scale': name,
              'config': {'agents': kn, 'per_species': cfg['per_species'], 'species': cfg['species'],
                         'tasks': cfg['tasks'], 'skills': EnvParams.TRAIT_DIM,
                         'gamma': round(cfg['tasks'] / kn, 3)},
              'baseline': 'greedy',
              'per_distribution': per_dist_agg,
              'overall': overall,
              'per_instance': per_instance}

    with open(os.path.join(scale_dir, 'greedy.json'), 'w') as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(scale_dir, 'greedy.md'), 'w', encoding='utf-8') as f:
        f.write(scale_markdown(name, cfg, per_dist_agg, overall))
    if save_pkl:
        with open(os.path.join(scale_dir, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser(description='Generate M2-M5 x 5-distribution test sets and run greedy.')
    ap.add_argument('--per-dist', type=int, default=20, help='instances per distribution per scale')
    ap.add_argument('--out-root', default='RALtest_5dist', help='output root directory')
    ap.add_argument('--scales', nargs='+', default=list(SCALES), choices=list(SCALES))
    ap.add_argument('--no-save-pkl', action='store_true', help='do not persist instance pkl files')
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    reports = {}
    for name in args.scales:
        print(f'=== {name} (per_species={SCALES[name]["per_species"]}, tasks={SCALES[name]["tasks"]}) ===')
        reports[name] = run_scale(name, SCALES[name], args.per_dist, args.out_root, not args.no_save_pkl)

    # combined summary across scales
    summary = {name: {'config': r['config'], 'overall': r['overall'],
                      'per_distribution': {d: {'success_rate': r['per_distribution'][d]['success_rate'],
                                               'makespan': r['per_distribution'][d]['makespan']}
                                           for d in DISTS}}
               for name, r in reports.items()}
    with open(os.path.join(args.out_root, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    lines = ['# M2-M5 x 5-distribution greedy summary', '']
    for name, r in reports.items():
        c, o = r['config'], r['overall']
        lines.append(f"## {name}RALtest  (agents={c['agents']}, tasks={c['tasks']}, gamma={c['gamma']})")
        lines.append(f"overall: success={o['success_rate']}  makespan={o['makespan']}  solved={o['solved']}/{o['instances']}")
        lines.append('')
        lines.append('| dist | success | makespan |')
        lines.append('| --- | --- | --- |')
        for d in DISTS:
            a = r['per_distribution'][d]
            lines.append(f"| {d} | {a['success_rate']} | {a['makespan']} |")
        lines.append('')
    with open(os.path.join(args.out_root, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'\nwrote results under {args.out_root}/ (per-scale greedy.json/greedy.md + summary.json/summary.md)')


if __name__ == '__main__':
    main()
