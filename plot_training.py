"""Headless training visualizer for servers without TensorBoard UI access.

Reads the data already produced by training -- the TensorBoard event files
(all scalar curves) and dist_stats.json (latest per-distribution snapshot) --
and writes static PNG figures + a single self-contained HTML report + a text
summary. Nothing here needs a browser tunnel: scp the PNGs or the one HTML file
and open them locally.

Usage:
    python plot_training.py                      # one-shot, defaults from parameters.py
    python plot_training.py --logdir train/save_1 --outdir train/save_1/plots
    python plot_training.py --smooth 0.9         # heavier EMA smoothing (0..1)
    python plot_training.py --watch 300          # regenerate every 5 min while training runs
"""
import os
import io
import sys
import json
import time
import base64
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend -> render straight to files, no display needed
import matplotlib.pyplot as plt

try:
    from parameters import SaverParams
    DEFAULT_LOGDIR = SaverParams.TRAIN_PATH
    DEFAULT_MODELDIR = SaverParams.MODEL_PATH
except Exception:
    DEFAULT_LOGDIR = 'train/save_1'
    DEFAULT_MODELDIR = 'model/save_1'


def load_scalars(logdir):
    """Return {tag: (steps, values)} from the TensorBoard event files in `logdir`."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        sys.exit("Need the 'tensorboard' package to read event files: pip install tensorboard")
    acc = EventAccumulator(logdir, size_guidance={'scalars': 0})  # 0 = load every point
    acc.Reload()
    out = {}
    for tag in acc.Tags().get('scalars', []):
        events = acc.Scalars(tag)
        out[tag] = (np.array([e.step for e in events]), np.array([e.value for e in events]))
    return out


def ema(values, alpha):
    """Exponential moving average; alpha in [0,1), higher = smoother."""
    if alpha <= 0 or len(values) == 0:
        return values
    out = np.empty_like(values, dtype=float)
    acc = values[0]
    for i, v in enumerate(values):
        acc = alpha * acc + (1 - alpha) * v
        out[i] = acc
    return out


def plot_grid(scalars, tags, title, ncols=3, smooth=0.6):
    """Render the given tags as a grid of line charts. Returns a matplotlib Figure."""
    tags = [t for t in tags if t in scalars]
    if not tags:
        return None
    nrows = int(np.ceil(len(tags) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for i, tag in enumerate(tags):
        ax = axes[i // ncols][i % ncols]
        ax.axis('on')
        steps, vals = scalars[tag]
        ax.plot(steps, vals, color='#B5D4F4', linewidth=1, label='raw')
        ax.plot(steps, ema(vals, smooth), color='#185FA5', linewidth=1.8, label='smoothed')
        ax.set_title(tag, fontsize=11)
        ax.set_xlabel('episode', fontsize=9)
        ax.grid(True, alpha=0.25)
        if len(vals):
            ax.annotate(f'{vals[-1]:.3g}', xy=(steps[-1], vals[-1]), fontsize=9,
                        color='#185FA5', ha='right', va='bottom')
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_per_distribution(scalars, dists, metrics, smooth=0.6):
    """One subplot per metric; one colored line per distribution. Returns a Figure or None."""
    colors = {'uniform': '#185FA5', 'gmm': '#1D9E75', 'ring': '#BA7517',
              'bipolar': '#D4537E', 'edges': '#D85A30'}
    present = [m for m in metrics if any(f'Dist/{d}/{m}' in scalars for d in dists)]
    if not present:
        return None
    ncols = 2
    nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for i, metric in enumerate(present):
        ax = axes[i // ncols][i % ncols]
        ax.axis('on')
        for d in dists:
            tag = f'Dist/{d}/{metric}'
            if tag in scalars:
                steps, vals = scalars[tag]
                ax.plot(steps, ema(vals, smooth), color=colors.get(d), linewidth=1.8, label=d)
        ax.set_title(metric, fontsize=11)
        ax.set_xlabel('episode', fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc='best')
    fig.suptitle('per-distribution training performance', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def text_summary(scalars, stats):
    """A compact console/markdown summary: latest aggregate values + per-dist table."""
    lines = []
    latest = {t: v[-1] for t, (s, v) in scalars.items() if len(v)}
    key_order = ['Loss/Reward', 'Perf/Makespan', 'Perf/Success rate',
                 'Perf/Waiting time', 'Perf/Traveling distance', 'Perf/Waiting Efficiency',
                 'Loss/Policy Loss', 'Loss/Entropy', 'Loss/Grad Norm', 'Loss/Learning Rate']
    lines.append('=== latest aggregate metrics ===')
    for k in key_order:
        if k in latest:
            lines.append(f'  {k:28s} {latest[k]:.4g}')
    if stats:
        lines.append('')
        lines.append(f"=== per-distribution snapshot (episode {stats.get('episode', '?')}) ===")
        lines.append(f"  weights: {stats.get('weights', {})}")
        per = stats.get('per_dist', {})
        cols = ['episodes', 'reward', 'makespan', 'success_rate', 'waiting_time', 'efficiency']
        header = '  ' + 'dist'.ljust(9) + ''.join(c.rjust(13) for c in cols)
        lines.append(header)
        for d, m in per.items():
            row = '  ' + d.ljust(9)
            for c in cols:
                val = m.get(c)
                row += (f'{val:.3g}'.rjust(13) if isinstance(val, (int, float)) else '-'.rjust(13))
            lines.append(row)
    return '\n'.join(lines)


def build_report(logdir, outdir, modeldir, smooth):
    os.makedirs(outdir, exist_ok=True)
    scalars = load_scalars(logdir)
    if not scalars:
        print(f'No scalar data found in {logdir} yet (has training logged anything?).')
        return

    stats_path = os.path.join(modeldir, 'dist_stats.json')
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else None
    dists = ['uniform', 'gmm', 'ring', 'bipolar', 'edges']

    figs = {}
    figs['overview'] = plot_grid(scalars,
                                 ['Loss/Reward', 'Perf/Makespan', 'Perf/Success rate',
                                  'Perf/Waiting time', 'Perf/Traveling distance', 'Perf/Waiting Efficiency'],
                                 'training overview', smooth=smooth)
    figs['losses'] = plot_grid(scalars,
                               ['Loss/Policy Loss', 'Loss/Entropy', 'Loss/Grad Norm', 'Loss/Learning Rate'],
                               'optimization', ncols=2, smooth=smooth)
    figs['per_dist'] = plot_per_distribution(scalars, dists,
                                             ['reward', 'makespan', 'success_rate', 'efficiency'], smooth=smooth)

    saved = []
    for name, fig in figs.items():
        if fig is not None:
            path = os.path.join(outdir, f'{name}.png')
            fig.savefig(path, dpi=110, bbox_inches='tight')
            saved.append(path)

    # single self-contained HTML (images embedded) -> scp one file, open locally
    html_imgs = ''
    for name, fig in figs.items():
        if fig is not None:
            html_imgs += f'<h2>{name}</h2><img src="data:image/png;base64,{fig_to_base64(fig)}" style="max-width:100%"/>'
    summary = text_summary(scalars, stats)
    html = (f'<html><head><meta charset="utf-8"><title>training report</title></head>'
            f'<body style="font-family:sans-serif;max-width:1100px;margin:auto">'
            f'<h1>training report</h1><pre style="background:#f5f5f5;padding:12px;'
            f'border-radius:6px;overflow:auto">{summary}</pre>{html_imgs}</body></html>')
    html_path = os.path.join(outdir, 'training_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    for fig in figs.values():
        if fig is not None:
            plt.close(fig)

    print(summary)
    print()
    print('wrote:')
    for p in saved + [html_path]:
        print('  ', p)


def main():
    ap = argparse.ArgumentParser(description='Headless training visualizer (no TensorBoard UI needed).')
    ap.add_argument('--logdir', default=DEFAULT_LOGDIR, help='TensorBoard event dir (default from parameters.py)')
    ap.add_argument('--modeldir', default=DEFAULT_MODELDIR, help='dir holding dist_stats.json')
    ap.add_argument('--outdir', default=None, help='where to write PNGs/HTML (default <logdir>/plots)')
    ap.add_argument('--smooth', type=float, default=0.6, help='EMA smoothing in [0,1), higher = smoother')
    ap.add_argument('--watch', type=int, default=0, help='regenerate every N seconds (0 = run once)')
    args = ap.parse_args()
    outdir = args.outdir or os.path.join(args.logdir, 'plots')

    while True:
        build_report(args.logdir, outdir, args.modeldir, args.smooth)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == '__main__':
    main()
