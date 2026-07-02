"""In-training closed loop, called by driver.py every REWEIGHT_EVERY episodes:

  1. evaluate the current best model on the M2-M5 x 5-distribution test sets,
  2. run greedy on the same instances (cached; instances are fixed),
  3. write a Markdown report (per-scale + per-distribution results, RL-vs-greedy gap),
  4. send that report + current mix + history to the DeepSeek API,
  5. return validated new sampling weights for hot-reload (or None to keep the mix).

Reweight axis = the five location distributions; the M2-M5 results are aggregated
per distribution across scales to form the signal.
"""
import os
import io
import json
import pickle
import contextlib
import datetime

import numpy as np

from worker import Worker
from parameters import EnvParams
from deepseek_reweight import call_deepseek, parse_weights, clamp_change, LOCATION_DISTS

DISTS = LOCATION_DISTS
SCALE_NAMES = ['M2', 'M3', 'M4', 'M5']
METRIC_KEYS = ['success_rate', 'makespan', 'time_cost', 'waiting_time', 'travel_dist', 'efficiency']

_greedy_cache = {}  # {test_root: {scale: {env_name: metrics}}} -- greedy is instance-fixed, compute once

SYSTEM_PROMPT = (
    "You tune the training-data mix (five task-location distributions) for a reinforcement-"
    "learning policy solving heterogeneous multi-robot task allocation, to improve robust "
    "generalization. Objective: minimize the WORST distribution's makespan gap to the greedy "
    "reference, without letting any distribution regress. Reply with ONLY a JSON object mapping "
    "each of the five distributions to a non-negative weight. No prose, no code fences."
)


def _metrics(env, finished):
    sr = float(np.sum(finished) / len(finished))
    if sr < 1:
        return {k: (sr if k == 'success_rate' else float('nan')) for k in METRIC_KEYS}
    return {'success_rate': sr,
            'makespan': float(env.current_time),
            'time_cost': float(np.nanmean(np.nan_to_num(env.get_matrix(env.task_dic, 'time_start'), nan=100))),
            'waiting_time': float(np.mean(env.get_matrix(env.agent_dic, 'sum_waiting_time'))),
            'travel_dist': float(np.sum(env.get_matrix(env.agent_dic, 'travel_dist'))),
            'efficiency': float(np.mean(env.get_efficiency()))}


def _greedy(env):
    env.init_state()
    with contextlib.redirect_stdout(io.StringIO()):
        env.execute_greedy_action()
    _, fin = env.get_episode_reward(EnvParams.MAX_TIME)
    return _metrics(env, fin)


def _rl(worker, env):
    env.init_state()
    worker.env = env
    with contextlib.redirect_stdout(io.StringIO()):
        worker.run_episode(False, False, False)  # eval, argmax policy, no max-waiting
    _, fin = worker.env.get_episode_reward(EnvParams.MAX_TIME)
    return _metrics(worker.env, fin)


def _load_instances(test_root, scale):
    d = os.path.join(test_root, f'{scale}RALtest')
    man = os.path.join(d, 'manifest.json')
    if not os.path.exists(man):
        return []
    out = []
    for e in json.load(open(man)):
        pkl = os.path.join(d, e['env'] + '.pkl')
        if os.path.exists(pkl):
            out.append((e['env'], e['dist'], pickle.load(open(pkl, 'rb'))))
    return out


def _agg(rows):
    if not rows:
        return {'instances': 0, 'success_rate': None, 'makespan': None, 'solved': 0}
    ms = [r['makespan'] for r in rows if not np.isnan(r['makespan'])]
    return {'instances': len(rows),
            'success_rate': round(float(np.mean([r['success_rate'] for r in rows])), 4),
            'makespan': round(float(np.mean(ms)), 4) if ms else None,
            'solved': int(sum(1 for r in rows if r['success_rate'] >= 1))}


def evaluate(net, device, test_root):
    """RL (best model) + greedy on every M2-M5 instance; aggregate per scale x dist and per dist."""
    worker = Worker(0, net, net, 0, device)
    gcache = _greedy_cache.setdefault(test_root, {})
    per_scale, rl_by_dist, gr_by_dist = {}, {d: [] for d in DISTS}, {d: [] for d in DISTS}
    for scale in SCALE_NAMES:
        insts = _load_instances(test_root, scale)
        if not insts:
            continue
        rl_rows, gr_rows = {d: [] for d in DISTS}, {d: [] for d in DISTS}
        scache = gcache.setdefault(scale, {})
        for env_name, dist, env in insts:
            rlm = _rl(worker, env)
            rl_rows[dist].append(rlm)
            rl_by_dist[dist].append(rlm)
            grm = scache.get(env_name) or _greedy(env)
            scache[env_name] = grm
            gr_rows[dist].append(grm)
            gr_by_dist[dist].append(grm)
        per_scale[scale] = {d: {'rl': _agg(rl_rows[d]), 'greedy': _agg(gr_rows[d])} for d in DISTS}
    per_dist = {d: {'rl': _agg(rl_by_dist[d]), 'greedy': _agg(gr_by_dist[d])} for d in DISTS}
    return {'per_scale': per_scale, 'per_dist': per_dist}


def _gap(rl, gr):
    if rl['makespan'] is None or gr['makespan'] is None:
        return None
    return round(rl['makespan'] - gr['makespan'], 3)


def to_markdown(report, episode, current_mix):
    L = [f'# Reweight report @ episode {episode}', '',
         f'current mix: `{json.dumps(current_mix)}`', '',
         '## Per-distribution, aggregated over M2-M5  (reweight signal)', '',
         '| dist | RL makespan | greedy makespan | gap (RL-greedy) | RL success |',
         '| --- | --- | --- | --- | --- |']
    for d in DISTS:
        a = report['per_dist'][d]
        L.append(f"| {d} | {a['rl']['makespan']} | {a['greedy']['makespan']} | "
                 f"{_gap(a['rl'], a['greedy'])} | {a['rl']['success_rate']} |")
    for scale in SCALE_NAMES:
        if scale not in report['per_scale']:
            continue
        L += ['', f'## {scale}', '', '| dist | RL makespan | greedy makespan | gap |', '| --- | --- | --- | --- |']
        for d in DISTS:
            a = report['per_scale'][scale][d]
            L.append(f"| {d} | {a['rl']['makespan']} | {a['greedy']['makespan']} | {_gap(a['rl'], a['greedy'])} |")
    return '\n'.join(L) + '\n'


def _history(path, k):
    if not os.path.exists(path):
        return []
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    return [json.loads(l) for l in lines[-k:]]


def run(net, device, test_root, episode, current_mix, out_dir, model_name,
        floor=0.5, max_change=3.0, history_k=5):
    """Full step: eval -> md -> DeepSeek -> new weights. Returns (weights|None, md_path)."""
    report = evaluate(net, device, test_root)
    md = to_markdown(report, episode, current_mix)
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, f'reweight_ep{episode}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)

    if all(report['per_dist'][d]['rl']['makespan'] is None for d in DISTS):
        print('[reweight] no test instances found under', test_root, '-> keeping current mix')
        return None, md_path

    hist_path = os.path.join(out_dir, 'reweight_history.jsonl')
    obs = {d: {'rl': report['per_dist'][d]['rl']['makespan'],
               'greedy': report['per_dist'][d]['greedy']['makespan']} for d in DISTS}
    with open(hist_path, 'a') as f:
        f.write(json.dumps({'time': datetime.datetime.now().isoformat(), 'episode': episode,
                            'mix': current_mix, 'per_dist': obs}) + '\n')
    history = _history(hist_path, history_k)

    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        print('[reweight] DEEPSEEK_API_KEY not set -> keeping current mix (report at %s)' % md_path)
        return None, md_path

    user = (md
            + '\n\n## Decide the next mix\n'
            + f'current_mix: {json.dumps(current_mix)}\n'
            + f'history (mix -> resulting per-dist makespan): {json.dumps(history)}\n'
            + f'constraints: each new weight in [current/{max_change}, current*{max_change}] and >= {floor}; '
              'incremental moves; do not drop any distribution; up-weight the worst gap.\n'
            + f'Return ONLY a JSON object with keys: {", ".join(DISTS)}.')
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}]
    try:
        reply = call_deepseek(messages, model_name, api_key)
        weights = clamp_change(parse_weights(reply, floor), current_mix, floor, max_change)
    except Exception as exc:
        print('[reweight] DeepSeek call/parse failed -> keeping current mix:', exc)
        return None, md_path

    with open(os.path.join(out_dir, 'reweight_log.jsonl'), 'a') as f:
        f.write(json.dumps({'time': datetime.datetime.now().isoformat(), 'episode': episode,
                            'from': current_mix, 'to': weights, 'reply': reply}) + '\n')
    return weights, md_path
