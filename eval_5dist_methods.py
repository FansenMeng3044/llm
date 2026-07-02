"""Run greedy AND the trained RL model on the M2-M5 x 5-distribution test sets,
and export a side-by-side comparison (JSON + Markdown) per scale + a summary.

RL uses the same eval path as test.py: load the checkpoint, wrap it in a Worker,
and call worker.run_episode(training=False). Both methods run on the SAME instance
and share the metric convention in gen_eval_5dist.metrics_from_env (makespan/etc.
are reported only when success_rate == 1).

Instances come from <test-root>/M<k>RALtest/env_*.pkl if present (created by
gen_eval_5dist.py); otherwise they are regenerated deterministically from the same
(scale, distribution, seed) plan, so this script also works standalone.

Usage:
    python eval_5dist_methods.py --model model/save_1                 # greedy + RL, all scales
    python eval_5dist_methods.py --model model/save --rl-samples 10   # RL best-of-10 (RL(s.10))
    python eval_5dist_methods.py --methods greedy                     # greedy only, no model needed
    python eval_5dist_methods.py --scales M4 M5 --test-root RALtest_5dist
"""
import os
import io
import json
import pickle
import argparse
import contextlib

import numpy as np
import torch

from attention import AttentionNet
from worker import Worker
from parameters import TrainParams, EnvParams, SaverParams
from gen_eval_5dist import (SCALES, DISTS, METRIC_KEYS, build_env, metrics_from_env,
                            run_greedy, aggregate, instance_plan)


def load_rl(model_path, weights_key, device):
    net = AttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM).to(device)
    ckpt = torch.load(os.path.join(model_path, 'checkpoint.pth'), map_location='cpu')
    key = weights_key if weights_key in ckpt else ('best_model' if 'best_model' in ckpt else 'model')
    net.load_state_dict(ckpt[key])
    net.eval()
    return net, ckpt.get('episode'), key


def _rl_better(m, best):
    a = (m['success_rate'], -(m['makespan'] if not np.isnan(m['makespan']) else 1e9))
    b = (best['success_rate'], -(best['makespan'] if not np.isnan(best['makespan']) else 1e9))
    return a > b


def run_rl(worker, env, rl_samples):
    """test.py-style: run the policy; with rl_samples>1 sample and keep the best run.
    Metrics recomputed via metrics_from_env so they match the greedy convention."""
    best = None
    sample = rl_samples > 1
    for _ in range(max(1, rl_samples)):
        env.init_state()
        worker.env = env
        with contextlib.redirect_stdout(io.StringIO()):
            worker.run_episode(False, sample, False)
        _, finished = worker.env.get_episode_reward(EnvParams.MAX_TIME)
        m = metrics_from_env(worker.env, finished)
        if best is None or _rl_better(m, best):
            best = m
    return best


def scale_instances(scale_dir, scale_name, per_dist):
    """Yield (env_name, dist, env). Prefer the saved manifest+pkl; else regenerate."""
    manifest_path = os.path.join(scale_dir, 'manifest.json')
    if os.path.exists(manifest_path):
        for e in json.load(open(manifest_path)):
            pkl = os.path.join(scale_dir, e['env'] + '.pkl')
            env = pickle.load(open(pkl, 'rb')) if os.path.exists(pkl) else build_env(SCALES[scale_name], e['dist'], e['seed'])
            yield e['env'], e['dist'], env
    else:
        for gi, dist, seed in instance_plan(scale_name, per_dist):
            yield f'env_{gi}', dist, build_env(SCALES[scale_name], dist, seed)


def _fmt(v):
    return v if v is None else round(v, 3)


def scale_markdown(name, cfg, methods, per_dist_agg, overall):
    kn = cfg['agents']
    head = f"# {name}RALtest  (greedy vs RL)\n\nconfig: k_n={kn} agents, k_s={cfg['species']} species, " \
           f"k_m={cfg['tasks']} tasks, k_b={cfg['skills']} skills  |  gamma={cfg['gamma']}\n"
    cols = '| distribution |' + ''.join(f' {m} success | {m} makespan |' for m in methods)
    if set(methods) == {'greedy', 'rl'}:
        cols += ' rl-greedy makespan |'
    sep = '| --- |' + ' --- | --- |' * len(methods) + (' --- |' if set(methods) == {'greedy', 'rl'} else '')
    lines = [head, cols, sep]
    for d in DISTS + ['OVERALL']:
        agg = overall if d == 'OVERALL' else per_dist_agg[d]
        row = f'| {d} |'
        for m in methods:
            a = agg[m]
            row += f" {a['success_rate']} | {_fmt(a['makespan'])} |"
        if set(methods) == {'greedy', 'rl'}:
            g, r = agg['greedy']['makespan'], agg['rl']['makespan']
            gap = round(r - g, 3) if (g is not None and r is not None) else None
            row += f' {gap} |'
        lines.append(row)
    return '\n'.join(lines) + '\n'


def run_scale(name, scale_dir, methods, per_dist, worker, rl_samples):
    per_dist_rows = {d: {m: [] for m in methods} for d in DISTS}
    per_instance = []
    for env_name, dist, env in scale_instances(scale_dir, name, per_dist):
        rec = {'env': env_name, 'dist': dist}
        if 'rl' in methods:
            m = run_rl(worker, env, rl_samples)
            per_dist_rows[dist]['rl'].append(m)
            rec['rl'] = {k: _fmt(m[k]) if not (isinstance(m[k], float) and np.isnan(m[k])) else None for k in METRIC_KEYS}
        if 'greedy' in methods:
            m = run_greedy(env)
            per_dist_rows[dist]['greedy'].append(m)
            rec['greedy'] = {k: _fmt(m[k]) if not (isinstance(m[k], float) and np.isnan(m[k])) else None for k in METRIC_KEYS}
        per_instance.append(rec)
    print(f'  {name}: evaluated {len(per_instance)} instances with {methods}')

    per_dist_agg = {d: {m: aggregate(per_dist_rows[d][m]) for m in methods} for d in DISTS}
    overall = {m: aggregate([row for d in DISTS for row in per_dist_rows[d][m]]) for m in methods}
    return per_dist_agg, overall, per_instance


def main():
    ap = argparse.ArgumentParser(description='Greedy + RL comparison on M2-M5 x 5-distribution test sets.')
    ap.add_argument('--test-root', default='RALtest_5dist', help='root with M<k>RALtest folders')
    ap.add_argument('--model', default=SaverParams.MODEL_PATH, help='dir containing checkpoint.pth (for RL)')
    ap.add_argument('--methods', nargs='+', default=['greedy', 'rl'], choices=['greedy', 'rl'])
    ap.add_argument('--weights-key', default='best_model', choices=['best_model', 'model'])
    ap.add_argument('--rl-samples', type=int, default=1, help='RL rollouts per instance, keep best (1=argmax)')
    ap.add_argument('--per-dist', type=int, default=20, help='used only if a scale has no saved manifest')
    ap.add_argument('--scales', nargs='+', default=list(SCALES), choices=list(SCALES))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out-suffix', default='methods', help='output file stem: <scale>/<suffix>.json/.md')
    args = ap.parse_args()

    worker = None
    rl_episode = rl_key = None
    if 'rl' in args.methods:
        net, rl_episode, rl_key = load_rl(args.model, args.weights_key, torch.device(args.device))
        worker = Worker(0, net, net, 0, torch.device(args.device))
        print(f'loaded RL model from {args.model} (key={rl_key}, episode={rl_episode})')

    os.makedirs(args.test_root, exist_ok=True)
    summary = {}
    for name in args.scales:
        scale_dir = os.path.join(args.test_root, f'{name}RALtest')
        os.makedirs(scale_dir, exist_ok=True)
        print(f'=== {name} ===')
        per_dist_agg, overall, per_instance = run_scale(name, scale_dir, args.methods, args.per_dist, worker, args.rl_samples)

        kn = SCALES[name]['per_species'] * SCALES[name]['species']
        cfg = {'agents': kn, 'per_species': SCALES[name]['per_species'], 'species': SCALES[name]['species'],
               'tasks': SCALES[name]['tasks'], 'skills': EnvParams.TRAIT_DIM, 'gamma': round(SCALES[name]['tasks'] / kn, 3)}
        report = {'scale': name, 'config': cfg, 'methods': args.methods,
                  'rl_episode': rl_episode, 'rl_weights_key': rl_key,
                  'per_distribution': per_dist_agg, 'overall': overall, 'per_instance': per_instance}
        with open(os.path.join(scale_dir, f'{args.out_suffix}.json'), 'w') as f:
            json.dump(report, f, indent=2)
        with open(os.path.join(scale_dir, f'{args.out_suffix}.md'), 'w', encoding='utf-8') as f:
            f.write(scale_markdown(name, cfg, args.methods, per_dist_agg, overall))
        summary[name] = {'config': cfg, 'overall': overall}

    # combined summary
    with open(os.path.join(args.test_root, f'{args.out_suffix}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    lines = ['# M2-M5 x 5-distribution: greedy vs RL summary', '']
    for name, s in summary.items():
        c = s['config']
        lines.append(f"## {name}RALtest  (agents={c['agents']}, tasks={c['tasks']}, gamma={c['gamma']})")
        lines.append('| method | success | makespan | solved |')
        lines.append('| --- | --- | --- | --- |')
        for m in args.methods:
            o = s['overall'][m]
            lines.append(f"| {m} | {o['success_rate']} | {_fmt(o['makespan'])} | {o['solved']}/{o['instances']} |")
        lines.append('')
    with open(os.path.join(args.test_root, f'{args.out_suffix}_summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\nwrote {args.out_suffix}.json/.md per scale + {args.out_suffix}_summary.* under {args.test_root}/')


if __name__ == '__main__':
    main()
