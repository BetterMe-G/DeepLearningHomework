"""
compare.py — aggregate the 9 ablation runs into report-ready figures.

Outputs to <ablation>/results/:
  - loss_overlay_singles.png   G/D loss: none + *_only + full
  - loss_overlay_sn_combos.png G/D loss: SN combinations (hinge / LS / noise / EMA)
  - loss_<run>.png             G/D loss per run (one figure each)
  - sample_grid.png            3x3 grid of NxN samples from each run (same z)
  - fid.csv                    table to be filled in from `evaluate.py` output
  - fid_bar.png                horizontal bar chart of FID + IS per run

Usage (run from the repo root):
    python3 ablation/scripts/compare.py
    python3 ablation/scripts/compare.py --n 6
    python3 ablation/scripts/compare.py --runs none full hinge_sn
    python3 ablation/scripts/compare.py --smooth_window 15001   # stronger smoothing
    python3 ablation/scripts/compare.py --show_raw              # faint raw line on grouped plots too
"""
import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.signal import savgol_filter

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Repo layout: this script is at ablation/scripts/compare.py
THIS_FILE  = Path(__file__).resolve()
SCRIPTS    = THIS_FILE.parent
ABLATION   = SCRIPTS.parent
REPO_ROOT  = ABLATION.parent
sys.path.insert(0, str(REPO_ROOT))

from config import Config               # noqa: E402
from models import Generator           # noqa: E402
from utils import set_seed, denorm     # noqa: E402

# -------- Run registry (order = display order) --------
RUN_NAMES = [
    "none", "sn_only", "ema_only", "hinge_only", "hinge_sn",
    "sn_ls", "sn_dnoise", "sn_ema", "full",
]
RUN_LABELS = {
    "none":       "1. none        (no defenses)",
    "sn_only":    "2. sn_only     (SN only)",
    "ema_only":   "3. ema_only    (EMA only)",
    "hinge_only": "4. hinge_only  (hinge only)",
    "hinge_sn":   "5. hinge_sn    (hinge + SN)",
    "sn_ls":      "6. sn_ls       (SN + label smooth)",
    "sn_dnoise":  "7. sn_dnoise   (SN + instance noise)",
    "sn_ema":     "8. sn_ema      (SN + EMA)",
    "full":       "9. full        (all on)",
}
COLORS = {
    "none":       "#d62728",  # red
    "sn_only":    "#1f77b4",  # blue
    "ema_only":   "#2ca02c",  # green
    "hinge_only": "#ff7f0e",  # orange
    "hinge_sn":   "#9467bd",  # purple
    "sn_ls":      "#17becf",  # cyan
    "sn_dnoise":  "#bcbd22",  # olive
    "sn_ema":     "#e377c2",  # pink
    "full":       "#8c564b",  # brown
}


# ======================================================================
# 1. Loss overlay
# ======================================================================
# Two grouped figures + one per-run figure.
SINGLE_RUNS = ["none", "sn_only", "ema_only", "hinge_only", "full"]
COMBO_RUNS  = ["hinge_sn", "sn_ls", "sn_dnoise", "sn_ema"]


def load_scalars(log_dir, tag):
    """Return (steps, values) numpy arrays, or (None, None) if missing."""
    if not log_dir.exists():
        return None, None
    try:
        ea = EventAccumulator(str(log_dir))
        ea.Reload()
    except Exception as e:
        print(f"[compare] skip {log_dir} (read error: {e})")
        return None, None
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    vals  = np.array([e.value for e in events])
    return steps, vals


def smooth_curve(y, window=None, polyorder=3):
    """Savitzky-Golay smoothing — polynomial fit, no phase shift, preserves peaks.

    `window` is the number of samples in the sliding window. Larger = smoother.
    If `window` is None we auto-pick ~10% of the curve length, floored to an
    odd integer >= 501. Falls back to a flat mean for very short arrays.

    polyorder=3 (cubic) gives the polynomial enough flexibility to track sharp
    transitions in the underlying trend without snapping to noise, which
    eliminates the "loop" artifact that polyorder=2 produces at sudden drops.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y

    if window is None:
        # ~10% of length, min 501, must be odd
        window = max(501, (n // 10) | 1)
    window = min(window, n - 1)
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        return np.full_like(y, float(np.mean(y)))
    return savgol_filter(y, window_length=window, polyorder=polyorder)


def _draw_loss(ax, steps, vals, color, label, smooth_window, show_raw=False):
    """Plot Savitzky-Golay smoothed loss curve (bold) on `ax`.

    If `show_raw` is True, also draw the raw curve as a very faint background
    line (no legend entry). For grouped plots with many runs, leave `show_raw`
    False — the noise dominates the visual and hurts readability.
    """
    if steps is None or vals is None or len(vals) == 0:
        return None
    smooth = smooth_curve(vals, window=smooth_window)
    if show_raw:
        ax.plot(steps, vals, color=color, alpha=0.10, linewidth=0.5, zorder=1)
    ax.plot(steps, smooth, color=color, alpha=1.0, linewidth=1.6,
            label=label, zorder=2)
    return smooth


def _safe_ylim(all_smoothed, pad=0.10):
    """Compute a robust ylim from the concatenated smoothed values.

    Uses the 1st/99th percentiles with `pad` margin on each side. Returns None
    if there's nothing to clip to.
    """
    pieces = [np.asarray(v) for v in all_smoothed if v is not None and len(v) > 0]
    if not pieces:
        return None
    flat = np.concatenate(pieces)
    if flat.size == 0:
        return None
    lo, hi = np.percentile(flat, 1), np.percentile(flat, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        return None
    margin = (hi - lo) * pad
    return (lo - margin, hi + margin)


def plot_loss_overlay(runs_subset, results_dir, filename, suptitle,
                      smooth_window=None, show_raw=False):
    """Two side-by-side subplots: D loss (left), G loss (right) for a subset of runs.

    Grouped plots: raw line is OFF by default (noise drowns the trends with 5+
    runs sharing one axis). Set `show_raw=True` to overlay it.
    """
    fig, (ax_d, ax_g) = plt.subplots(1, 2, figsize=(14, 5))

    plotted = 0
    d_smoothed, g_smoothed = [], []
    for name in runs_subset:
        log_dir = ABLATION / "logs" / name
        s_d, v_d = load_scalars(log_dir, "loss/D")
        s_g, v_g = load_scalars(log_dir, "loss/G")
        color = COLORS.get(name, None)
        label = RUN_LABELS.get(name, name)

        sm_d = _draw_loss(ax_d, s_d, v_d, color, label, smooth_window, show_raw=show_raw)
        if sm_d is not None:
            d_smoothed.append(sm_d)
            plotted += 1
        sm_g = _draw_loss(ax_g, s_g, v_g, color, label, smooth_window, show_raw=show_raw)
        if sm_g is not None:
            g_smoothed.append(sm_g)

    for ax, title, ylabel, pool in [
        (ax_d, "Discriminator loss", "loss_D", d_smoothed),
        (ax_g, "Generator loss",     "loss_G", g_smoothed),
    ]:
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.15)
        ylim = _safe_ylim(pool)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if plotted > 0:
            ax.legend(fontsize=8, loc="best", framealpha=0.9)

    w = smooth_window if smooth_window is not None else "auto"
    fig.suptitle(
        f"{suptitle}  (SavGol-smoothed, window={w})",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    out = results_dir / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out}  (plotted {plotted}/{len(runs_subset)} runs)")


def plot_loss_individual(runs, results_dir, smooth_window=None, show_raw=True):
    """One figure per run: D loss (left), G loss (right), with smoothed curve.

    Individual plots have only one line per axis so a very faint raw trace
    is OK as "this is real data" — on by default.
    """
    written = 0
    for name in runs:
        log_dir = ABLATION / "logs" / name
        s_d, v_d = load_scalars(log_dir, "loss/D")
        s_g, v_g = load_scalars(log_dir, "loss/G")
        if s_d is None and s_g is None:
            print(f"[compare] skip individual {name} (no log data)")
            continue

        fig, (ax_d, ax_g) = plt.subplots(1, 2, figsize=(12, 4.5))
        color = COLORS.get(name, "#888888")

        sm_d = _draw_loss(ax_d, s_d, v_d, color, "loss_D", smooth_window, show_raw=show_raw)
        sm_g = _draw_loss(ax_g, s_g, v_g, color, "loss_G", smooth_window, show_raw=show_raw)

        for ax, title, ylabel, sm in [
            (ax_d, "Discriminator loss", "loss_D", sm_d),
            (ax_g, "Generator loss",     "loss_G", sm_g),
        ]:
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("iteration")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.15)
            ylim = _safe_ylim([sm])
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.legend(fontsize=8, loc="best", framealpha=0.9)

        w = smooth_window if smooth_window is not None else "auto"
        fig.suptitle(
            f"{RUN_LABELS.get(name, name)}  (SavGol-smoothed, window={w})",
            fontsize=12, y=1.02,
        )
        fig.tight_layout()
        out = results_dir / f"loss_{name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[compare] wrote {out}")
        written += 1
    return written


# ======================================================================
# 2. Sample grid (3x3 panel of NxN samples per run, same z)
# ======================================================================
def load_generator(ckpt_path, device, z_dim):
    """Load Generator from a checkpoint dict; prefer G_ema when present."""
    G = Generator(z_dim=z_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if ckpt.get("G_ema") is not None:
            G.load_state_dict(ckpt["G_ema"])
            print(f"[compare]   {ckpt_path.parent.name}: using G_ema")
        elif "G" in ckpt:
            G.load_state_dict(ckpt["G"])
        else:
            raise ValueError(f"No G/G_ema in {ckpt_path}")
    else:
        G.load_state_dict(ckpt)
    G.eval()
    return G


def tensor_to_panel(img_tensor):
    """(N*N, 3, 64, 64) in [-1,1] -> (H, W, 3) uint8 panel for matplotlib."""
    imgs = denorm(img_tensor).clamp(0, 1).mul(255).byte()
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()    # (N*N, 64, 64, 3)
    n = int(np.sqrt(imgs.shape[0]))
    assert n * n == imgs.shape[0], "panel count must be a perfect square"
    rows = [np.concatenate(list(imgs[i*n:(i+1)*n]), axis=1) for i in range(n)]
    return np.concatenate(rows, axis=0)              # (n*64, n*64, 3)


def build_sample_grid(runs, n, z_dim, device, results_dir):
    """Assemble a 3x3 panel of NxN samples from each run (same z across runs)."""
    set_seed(0)
    z = torch.randn(n * n, z_dim, 1, 1, device=device)

    nrows, ncols = 3, 3
    fig = plt.figure(figsize=(ncols * 4.0, nrows * 4.0))
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.05, hspace=0.15)

    plotted = 0
    for idx, name in enumerate(runs):
        ckpt = ABLATION / "ckpts" / name / "latest.pt"
        if not ckpt.exists():
            print(f"[compare] skip {name} (no {ckpt})")
            continue
        try:
            G = load_generator(ckpt, device, z_dim)
            with torch.no_grad():
                fake = G(z)
            panel = tensor_to_panel(fake.cpu())
        except Exception as e:
            print(f"[compare] skip {name} (load/gen error: {e})")
            continue

        row, col = idx // ncols, idx % ncols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(panel)
        ax.set_title(RUN_LABELS.get(name, name), fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        plotted += 1

    fig.suptitle(
        f"DCGAN mode-collapse ablation: {n}x{n} samples per run (same z)",
        fontsize=14, y=0.99,
    )
    out = results_dir / "sample_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out}  (plotted {plotted}/{len(runs)} runs)")


# ======================================================================
# 3. FID + IS table
# ======================================================================
def parse_fidelity_log(path):
    """Extract FID and IS from compute_fid.py output (or torch-fidelity).

    Supported formats:
      FID: 12.345                  (compute_fid.py)
      Frechet Inception Distance: 12.3456  (torch-fidelity)
      IS: 2.345 ± 0.123            (compute_fid.py)
      Inception Score mean: 1.2345        (torch-fidelity)
    """
    if not path.exists():
        return None, None
    text = path.read_text()
    fid = isc = None
    m = re.search(r"(?:Frechet Inception Distance|FID):\s*([0-9.]+)", text)
    if m:
        fid = float(m.group(1))
    m = re.search(r"(?:Inception Score mean|IS):\s*([0-9.]+)", text)
    if m:
        isc = float(m.group(1))
    return fid, isc


def write_fid_table(runs, results_dir):
    """Write a CSV; if ablation/eval/<run>/fid.log exists, parse FID/IS from it."""
    out = results_dir / "fid.csv"
    rows = []
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "FID", "InceptionScore", "source"])
        for name in runs:
            log = ABLATION / "eval" / name / "fid.log"
            fid, isc = parse_fidelity_log(log)
            if fid is not None:
                w.writerow([name, f"{fid:.3f}", f"{isc:.3f}" if isc else "", "auto"])
                rows.append((name, fid, isc if isc is not None else float("nan")))
            else:
                w.writerow([name, "", "", ""])
    print(f"[compare] wrote {out}")
    return rows


def plot_fid_bar(rows, results_dir):
    """Horizontal bar chart of FID + IS per run, sorted by FID ascending."""
    if not rows:
        print("[compare] no FID rows, skipping fid_bar.png")
        return
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        print("[compare] no valid FID values, skipping fid_bar.png")
        return
    rows.sort(key=lambda r: r[1])

    # Short, uniform 2-line labels (no hand-aligned spaces)
    SHORT_LABELS = {
        "none":       "none\n(no defense)",
        "sn_only":    "sn_only\n(SN)",
        "ema_only":   "ema_only\n(EMA)",
        "hinge_only": "hinge_only\n(hinge)",
        "hinge_sn":   "hinge_sn\n(hinge + SN)",
        "sn_ls":      "sn_ls\n(SN + LS)",
        "sn_dnoise":  "sn_dnoise\n(SN + noise)",
        "sn_ema":     "sn_ema\n(SN + EMA)",
        "full":       "full\n(all defenses)",
    }

    names  = [r[0] for r in rows]
    fids   = [r[1] for r in rows]
    iscs   = [r[2] for r in rows]
    colors = [COLORS.get(n, "#888888") for n in names]
    labels = [SHORT_LABELS.get(n, n) for n in names]

    fig, (ax_fid, ax_is) = plt.subplots(
        1, 2, figsize=(13, max(4, 0.55 * len(names) + 1.5)),
        gridspec_kw={"wspace": 0.45},
    )
    y = np.arange(len(names))

    # ----- FID (left, log scale) -----
    bars = ax_fid.barh(y, fids, color=colors, edgecolor="black",
                       linewidth=0.5, height=0.7)
    ax_fid.set_yticks(y)
    ax_fid.set_yticklabels(labels, fontsize=10)
    ax_fid.invert_yaxis()  # best at top
    ax_fid.set_xscale("log")
    ax_fid.set_xlabel("FID  (log scale, lower = better)", fontsize=10)
    ax_fid.set_title("Frechet Inception Distance", fontsize=12, fontweight="bold")
    ax_fid.grid(True, axis="x", alpha=0.3, which="both")
    ax_fid.set_axisbelow(True)
    for bar, v in zip(bars, fids):
        ax_fid.text(bar.get_width() * 1.08, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", fontsize=10, fontweight="bold")

    # ----- IS (right, linear scale) -----
    has_is = any(not np.isnan(v) for v in iscs)
    if has_is:
        isc_plot = [v if not np.isnan(v) else 0 for v in iscs]
        bars_is = ax_is.barh(y, isc_plot, color=colors, edgecolor="black",
                             linewidth=0.5, height=0.7)
        ax_is.set_yticks(y)
        ax_is.set_yticklabels(labels, fontsize=10)
        ax_is.invert_yaxis()
        ax_is.set_xlabel("Inception Score  (higher = better)", fontsize=10)
        ax_is.set_title("Inception Score", fontsize=12, fontweight="bold")
        ax_is.grid(True, axis="x", alpha=0.3)
        ax_is.set_axisbelow(True)
        for bar, v in zip(bars_is, iscs):
            if not np.isnan(v):
                ax_is.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                           f"{v:.2f}", va="center", fontsize=10, fontweight="bold")
    else:
        ax_is.set_axis_off()
        ax_is.text(0.5, 0.5, "IS not available\n(run compute_fid.py for all runs)",
                   ha="center", va="center", transform=ax_is.transAxes, fontsize=11)

    fig.suptitle("DCGAN mode-collapse ablation  ·  CelebA 64x64  ·  80 epochs",
                 fontsize=13, y=0.98, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = results_dir / "fid_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out}")


# ======================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8,
                   help="NxN samples per run (must be a perfect square)")
    p.add_argument("--z_dim", type=int, default=Config.z_dim)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--runs", nargs="+", default=None,
                   help="subset of runs to include (default: all 9)")
    p.add_argument("--smooth_window", type=int, default=None,
                   help="Savitzky-Golay window for smoothed loss curves "
                        "(in samples, must be odd). Default: auto ≈ 10% of "
                        "curve length, odd, >= 501. Try 15001-20001 for "
                        "extremely noisy G loss curves.")
    p.add_argument("--show_raw", action="store_true",
                   help="Overlay a faint raw curve on top of the smoothed one. "
                        "Off by default for grouped plots, on for individual.")
    args = p.parse_args()

    runs = args.runs if args.runs else RUN_NAMES
    results_dir = ABLATION / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compare] ablation root: {ABLATION}")
    print(f"[compare] runs: {runs}")
    print(f"[compare] device: {args.device}")

    # --- Loss overlays: 2 grouped + 1 per run ---
    sw = args.smooth_window
    # Grouped plots: raw is OFF by default. --show_raw turns it on.
    # Individual plots: raw is ON by default (single line, noise is OK).
    plot_loss_overlay(
        SINGLE_RUNS, results_dir,
        filename="loss_overlay_singles.png",
        suptitle="DCGAN ablation · singles: none + *_only + full",
        smooth_window=sw,
        show_raw=args.show_raw,
    )
    plot_loss_overlay(
        COMBO_RUNS, results_dir,
        filename="loss_overlay_sn_combos.png",
        suptitle="DCGAN ablation · SN combinations: SN + (hinge / LS / noise / EMA)",
        smooth_window=sw,
        show_raw=args.show_raw,
    )
    plot_loss_individual(
        runs, results_dir,
        smooth_window=sw,
        show_raw=True,
    )

    # --- Sample grid + FID ---
    build_sample_grid(runs, args.n, args.z_dim, args.device, results_dir)
    fid_rows = write_fid_table(runs, results_dir)
    plot_fid_bar(fid_rows, results_dir)

    print("[compare] done.")


if __name__ == "__main__":
    main()
