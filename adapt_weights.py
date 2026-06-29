"""One-shot adaptive reweighting: evaluate the current saved model across the
five distributions against a reference baseline, then ask DeepSeek for the next
training mix and write dist_weights.json (which the trainer hot-reloads).

Run this once the model has roughly converged. Training can keep running -- it
picks up the new mix at the next checkpoint boundary -- or you can stop, run
this, and resume with LOAD_MODEL=True.

    export DEEPSEEK_API_KEY=sk-...
    python adapt_weights.py                                  # eval (RL vs greedy) -> DeepSeek -> weights
    python adapt_weights.py --instances 50 --rl-samples 10
    python adapt_weights.py --baseline ctasd --testset-root RALTestSet_5dist
    python adapt_weights.py --dry-run                        # do everything except overwrite weights
"""
import argparse

import torch

from parameters import EnvParams, SaverParams
import eval_distributions as ev
import deepseek_reweight as ds


def main():
    ap = argparse.ArgumentParser(description='Eval -> DeepSeek -> new training mix, in one step.')
    ap.add_argument('--model', default=SaverParams.MODEL_PATH)
    ap.add_argument('--instances', type=int, default=30)
    ap.add_argument('--baseline', choices=['greedy', 'ctasd'], default='greedy')
    ap.add_argument('--rl-samples', type=int, default=1)
    ap.add_argument('--weights-key', default='best_model', choices=['best_model', 'model'])
    ap.add_argument('--testset-root', default=None)
    ap.add_argument('--tasks', type=int, default=EnvParams.TASKS_RANGE[1])
    ap.add_argument('--species', type=int, default=EnvParams.SPECIES_RANGE[1])
    ap.add_argument('--per-species', type=int, default=EnvParams.SPECIES_AGENTS_RANGE[0])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--llm-model', default='deepseek-chat')
    ap.add_argument('--floor', type=float, default=0.5)
    ap.add_argument('--history', type=int, default=5, help='past (mix->outcome) records shown to the model')
    ap.add_argument('--max-change', type=float, default=3.0, help='max per-step weight change factor')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    print('=== step 1/2: evaluating current model per distribution ===')
    scale = (args.per_species, args.species, args.tasks)
    report = ev.evaluate(args.model, args.instances, scale, args.baseline, args.rl_samples,
                         args.weights_key, args.testset_root, torch.device(args.device))

    print('\n=== step 2/2: asking DeepSeek for the next training mix (with feedback history) ===')
    ds.run_reweight(report, args.model, args.llm_model, args.floor,
                    args.history, args.max_change, args.dry_run)
    if not args.dry_run:
        print('training will hot-reload this at the next 512-episode checkpoint, or resume with LOAD_MODEL=True.')


if __name__ == '__main__':
    main()
