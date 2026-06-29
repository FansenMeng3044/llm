"""Feed the per-distribution RL-vs-baseline gap (eval_gap.json) PLUS the history of
past (mix -> resulting performance) decisions to the DeepSeek API, and write the
new training proportions to dist_weights.json, which the running trainer
hot-reloads at the next checkpoint boundary.

What the model is given (closed-loop, not a memoryless snapshot):
  * the latest per-distribution eval (RL vs baseline gap + success rates),
  * the trajectory of recent (mix -> observed performance) pairs, so it can do
    credit assignment ("last time I up-weighted edges, did edges improve?"),
  * the current mix and a hard change-cap, so adjustments stay incremental,
  * an explicit objective: shrink the WORST distribution's gap (robust generalization).

Auth: set the env var DEEPSEEK_API_KEY. Endpoint is OpenAI-compatible.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    python deepseek_reweight.py                 # reads model/save_1/eval_gap.json
    python deepseek_reweight.py --dry-run        # print proposed weights, don't write
"""
import os
import re
import json
import shutil
import argparse
import datetime
import urllib.request

from parameters import SaverParams

LOCATION_DISTS = ['uniform', 'gmm', 'ring', 'bipolar', 'edges']
DEEPSEEK_URL = 'https://api.deepseek.com/chat/completions'

SYSTEM_PROMPT = (
    "You are a controller tuning the training-data mix for a reinforcement-learning "
    "policy that solves heterogeneous multi-robot task allocation. Training instances "
    "are drawn from five task-location distributions; you set the sampling proportions "
    "for the next training phase to improve GENERALIZATION across all five. "
    "Objective: minimize the WORST distribution's gap to the baseline (robust, minimax), "
    "while not letting any distribution regress. Reply with ONLY a JSON object mapping "
    "each of the five distribution names to a non-negative weight. No prose, no code fences."
)


def summarize(report):
    """Pull the few numbers per distribution the controller actually needs."""
    out = {}
    for d in LOCATION_DISTS:
        pd = report.get('per_dist', {}).get(d)
        if not pd:
            continue
        out[d] = {
            'rl_makespan': pd['rl'].get('makespan'),
            'base_makespan': pd['baseline'].get('makespan'),
            'makespan_gap_rl_minus_base': pd['gap'].get('makespan_rl_minus_base'),
            'rl_success': pd['rl'].get('success_rate'),
            'success_gap_rl_minus_base': pd['gap'].get('success_rate_rl_minus_base'),
        }
    return out


def load_history(path, k):
    if not os.path.exists(path):
        return []
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    return [json.loads(l) for l in lines[-k:]]


def build_messages(report, current_mix, history, floor, max_change):
    latest = summarize(report)
    traj = [{'episode': h.get('episode'), 'mix': h.get('mix'), 'observed': h.get('observed')}
            for h in history]
    user = {
        'objective': 'choose next-phase sampling weights to shrink the worst distribution gap; '
                     'reward = makespan (lower better), success_rate (higher better). '
                     'makespan_gap_rl_minus_base > 0 means RL is WORSE than baseline there.',
        'constraints': {
            'min_weight': floor,
            'max_change_factor': max_change,
            'note': f'each new weight must stay within [current/{max_change}, current*{max_change}] '
                    f'and >= {floor}; make incremental moves, do not drop any distribution.'
        },
        'current_mix': current_mix,
        'history_mix_to_outcome': traj,
        'latest_eval': {'episode': report.get('episode'), 'baseline': report.get('baseline'),
                        'per_dist': latest},
        'reply_format': 'ONLY a JSON object with keys: ' + ', '.join(LOCATION_DISTS),
    }
    return [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': json.dumps(user, indent=2)}]


def call_deepseek(messages, model, api_key, timeout=120):
    payload = json.dumps({'model': model, 'messages': messages,
                          'temperature': 0.2, 'stream': False}).encode('utf-8')
    req = urllib.request.Request(DEEPSEEK_URL, data=payload, method='POST', headers={
        'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode('utf-8'))
    return body['choices'][0]['message']['content']


def parse_weights(text, floor):
    """Extract the first JSON object and coerce it into validated weights."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f'no JSON object in model reply:\n{text}')
    raw = json.loads(match.group(0))
    weights = {}
    for d in LOCATION_DISTS:
        try:
            weights[d] = max(float(floor), float(raw[d]))
        except (KeyError, TypeError, ValueError):
            raise ValueError(f'reply missing/invalid weight for "{d}":\n{text}')
    if sum(weights.values()) <= 0:
        raise ValueError('all weights non-positive')
    return weights


def clamp_change(new, current, floor, max_change):
    """Guardrail: keep each weight within max_change x of the current mix (and >= floor)."""
    out = {}
    for d in LOCATION_DISTS:
        cur = float(current.get(d, 1.0)) or floor
        lo, hi = cur / max_change, cur * max_change
        out[d] = round(max(floor, min(max(new[d], lo), hi)), 4)
    return out


def run_reweight(report, model_path, llm_model='deepseek-chat', floor=0.5,
                 history_k=5, max_change=3.0, dry_run=False):
    """Shared entry point used by this CLI and by adapt_weights.py."""
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        raise SystemExit('set DEEPSEEK_API_KEY in the environment.')

    weights_path = os.path.join(model_path, 'dist_weights.json')
    history_path = os.path.join(model_path, 'adapt_history.jsonl')
    current_mix = (json.load(open(weights_path)) if os.path.exists(weights_path)
                   else {d: 1 for d in LOCATION_DISTS})

    # record what THIS mix produced (so future calls can do credit assignment), then
    # read back the recent trajectory (including this observation).
    with open(history_path, 'a') as f:
        f.write(json.dumps({'time': datetime.datetime.now().isoformat(),
                            'episode': report.get('episode'), 'mix': current_mix,
                            'observed': summarize(report)}) + '\n')
    history = load_history(history_path, history_k)

    messages = build_messages(report, current_mix, history, floor, max_change)
    reply = call_deepseek(messages, llm_model, api_key)
    weights = clamp_change(parse_weights(reply, floor), current_mix, floor, max_change)
    print('current mix :', {d: round(float(current_mix.get(d, 0)), 3) for d in LOCATION_DISTS})
    print('DeepSeek -> :', weights)

    if dry_run:
        print('(dry-run: dist_weights.json not modified)')
        return weights, reply

    if os.path.exists(weights_path):
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        shutil.copy(weights_path, weights_path.replace('.json', f'.{ts}.bak.json'))
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    with open(weights_path, 'w') as f:
        json.dump(weights, f, indent=2)
    with open(os.path.join(model_path, 'reweight_log.jsonl'), 'a') as f:
        f.write(json.dumps({'time': datetime.datetime.now().isoformat(),
                            'episode': report.get('episode'), 'from': current_mix,
                            'to': weights, 'reply': reply}) + '\n')
    print('wrote', weights_path, '(trainer hot-reloads it at the next 512-episode boundary)')
    return weights, reply


def main():
    ap = argparse.ArgumentParser(description='DeepSeek-driven distribution reweighting (closed-loop).')
    ap.add_argument('--model-path', default=SaverParams.MODEL_PATH)
    ap.add_argument('--eval-json', default=None, help='default <model-path>/eval_gap.json')
    ap.add_argument('--llm-model', default='deepseek-chat', help='deepseek-chat or deepseek-reasoner')
    ap.add_argument('--floor', type=float, default=0.5, help='minimum weight per distribution')
    ap.add_argument('--history', type=int, default=5, help='past (mix->outcome) records shown to the model')
    ap.add_argument('--max-change', type=float, default=3.0, help='max per-step weight change factor')
    ap.add_argument('--dry-run', action='store_true', help='print proposed weights, do not write')
    args = ap.parse_args()

    eval_json = args.eval_json or os.path.join(args.model_path, 'eval_gap.json')
    report = json.load(open(eval_json))
    run_reweight(report, args.model_path, args.llm_model, args.floor,
                 args.history, args.max_change, args.dry_run)


if __name__ == '__main__':
    main()
