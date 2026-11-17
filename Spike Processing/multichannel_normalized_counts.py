#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multichannel spike counts normalised to the baseline mean (baseline = 100%).

One figure per recording plus one combined timeline. Probe channels are
overlaid; the needle channel gets its own panel. Spike detection is delegated
to the canonical counter in this folder, including its duplicate-event
deduplication and optional waveform-shape noise rejection.

Baseline reference, per channel:

    rate_hz = spike_count / window_duration_s
    normalized_rate_pct = 100 * rate_hz / mean(baseline rate_hz)

Example:

  python "Spike Processing/multichannel_normalized_counts.py" ^
    --baseline "Final Store/.../raw/basline.csv" ^
    --post "Final Store/.../raw/post0.csv" "Final Store/.../raw/post1.csv" ^
    --out-dir "Output/multichannel_normalized_counts"

Reuse a standard SPIE count table (no detection, no raw recordings):

  python "Spike Processing/multichannel_normalized_counts.py" ^
    --counts-csv "spike_counts_per_min/ALL_RECORDINGS__spike_counts_per_min.csv"
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
COUNTER_PATH = ROOT / "Spike Processing" / "Spike Count Multiple CSVs ordered.py"
COUNTS_NAME = "ALL_RECORDINGS__spike_counts_per_min.csv"


def counts_cache_name(window_s: float) -> str:
    if math.isclose(float(window_s), 60.0):
        return COUNTS_NAME
    label = f"{float(window_s):g}_sec" if window_s < 60 else f"{float(window_s) / 60.0:g}_min"
    return f"ALL_RECORDINGS__spike_counts_per_{label}.csv"


# =========================
# LOAD CANONICAL COUNTER
# =========================

def load_counter():
    """Import the space-containing counter script by path."""
    spec = importlib.util.spec_from_file_location("spike_count_multiple_csvs", COUNTER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spike counter: {COUNTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # Counter inserts its own parent into sys.path so spike_shape_qc resolves.
    spec.loader.exec_module(module)
    return module


def label_for_path(path: str | Path) -> str:
    """Baseline file -> 'Baseline'; post0.csv -> 'Post'; post1.csv -> 'Post 1'."""
    stem = Path(path).stem.strip().lower()
    if stem.startswith("bas"):
        return "Baseline"
    if stem.startswith("post"):
        suffix = stem[4:].strip(" _-")
        if suffix.lstrip("0") == "":
            return "Post"
        return f"Post {suffix}"
    return Path(path).stem


def shape_settings(args):
    return {
        "mode": args.shape_mode,
        "separation_ms": args.shape_separation_ms,
        "depth_fraction": args.shape_depth_fraction,
        "prominence_fraction": args.shape_prominence_fraction,
        "noise_multiplier": args.shape_noise_multiplier,
        "padding_ms": args.shape_padding_ms,
    }


def counts_subdir(polarity: str, shape_mode: str, window_s: float) -> Path:
    """Settings-keyed cache folder; different detector settings never share counts."""
    return Path("counts") / f"{polarity}_shape-{shape_mode}_win{float(window_s):g}s"


def run_counting(args, out_dir: Path) -> Path:
    counter = load_counter()
    counts_dir = out_dir / counts_subdir(args.polarity, args.shape_mode, args.window_s)
    counts_dir.mkdir(parents=True, exist_ok=True)

    paths = [str(args.baseline)] + [str(p) for p in args.post]
    labels = {str(Path(p)): label_for_path(p) for p in paths}
    print(f"[count] shape mode: {args.shape_mode}")
    for path, label in labels.items():
        print(f"[count] {label}: {path}")

    outputs = counter.process_csvs(
        paths,
        out_dir=str(counts_dir),
        epoch_labels_by_path=labels,
        window_sec=args.window_s,
        polarity=args.polarity,
        shape_settings=shape_settings(args),
        return_outputs=True,
    )
    return Path(outputs["spike_count_csv"])


# =========================
# NORMALISE
# =========================

def _display_labels(counts: pd.DataFrame) -> pd.Series:
    """Prefer epoch label, then phase, then recording name."""
    labels = counts["epoch_label"].fillna("").astype(str).str.strip()
    if "phase" in counts:
        phase = counts["phase"].fillna("").astype(str).str.strip()
        labels = labels.mask(labels == "", phase)
    return labels.mask(labels == "", counts["recording_name"].astype(str))


def _baseline_mask(counts, baseline_label, baseline_recording=None):
    labels = counts["epoch_label"].fillna("").astype(str).str.strip().str.lower()
    mask = labels == str(baseline_label).strip().lower()
    if mask.any():
        return mask, f"epoch label {baseline_label!r}"

    if baseline_recording:
        names = counts["recording_name"].astype(str).str.strip().str.lower()
        mask = names == str(baseline_recording).strip().lower()
        if not mask.any():
            raise ValueError(f"Baseline recording {baseline_recording!r} not found in the spike-count table.")
        return mask, f"recording {baseline_recording!r}"

    if "phase" in counts:
        phase = counts["phase"].fillna("").astype(str).str.strip().str.lower()
        mask = phase == "baseline"
        if mask.any():
            return mask, "phase 'baseline'"

    first = counts["recording_name"].dropna().astype(str).iloc[0]
    mask = counts["recording_name"].astype(str) == first
    return mask, f"first recording {first!r} (no baseline label found)"


def normalise_counts(counts: pd.DataFrame, baseline_label: str = "Baseline",
                     baseline_recording: str | None = None) -> pd.DataFrame:
    """Add rate_hz, baseline_mean_rate_hz and normalized_rate_pct per channel."""
    required = {"channel", "epoch_label", "spike_count", "window_duration_s", "recording_name"}
    missing = required - set(counts.columns)
    if missing:
        raise ValueError(f"Spike-count table is missing columns: {sorted(missing)}")

    out = counts.copy()
    duration = out["window_duration_s"].to_numpy(dtype=float)
    if np.any(duration <= 0):
        raise ValueError("Spike-count windows must have a positive duration.")
    out["rate_hz"] = out["spike_count"].to_numpy(dtype=float) / duration
    out["spikes_per_min"] = out["rate_hz"] * 60.0
    out["display_label"] = _display_labels(out)

    is_baseline, source = _baseline_mask(out, baseline_label, baseline_recording)
    print(f"[baseline] {source}: {int(is_baseline.sum())} windows")

    baseline_mean = out.loc[is_baseline].groupby("channel")["rate_hz"].mean()
    out["baseline_mean_rate_hz"] = out["channel"].map(baseline_mean)
    out["normalized_rate_pct"] = (100.0 * out["rate_hz"] / out["baseline_mean_rate_hz"]).replace(
        [np.inf, -np.inf], np.nan
    )

    zero_baseline = baseline_mean[~np.isfinite(baseline_mean) | (baseline_mean <= 0)]
    if len(zero_baseline):
        print(f"[warn] non-positive baseline rate, normalisation is NaN: {', '.join(zero_baseline.index)}")
    return out


# =========================
# PLOT
# =========================

TITLE_SIZE = 20
LABEL_SIZE = 18
TICK_SIZE = 14
LEGEND_SIZE = 14
PHASE_LABEL_SIZE = 17
LINE_WIDTH = 2.0
MARKER_SIZE = 5.0
NEEDLE_COLOR = "#1f77b4"
PHASE_COLORS = ("#d9d9d9", "#fde9e0", "#cfe8f8", "#d9f7d9")


def channel_colors(channels, needle: str | None = None) -> dict:
    cmap = plt.get_cmap("tab10")
    colors = {}
    index = 1 if needle in channels else 0
    for channel in channels:
        if channel == needle:
            colors[channel] = NEEDLE_COLOR
        else:
            colors[channel] = cmap(index % 10)
            index += 1
    return colors


def _plot_panel(ax, frame, colors, needle=None, only_needle=False, ylim=None, yscale="linear",
                value_col="normalized_rate_pct", ylabel=None,
                baseline_line=100.0):
    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=10.0)
    elif yscale in ("linear", "log"):
        ax.set_yscale(yscale)
    for channel, group in frame.groupby("channel", sort=False):
        if (channel == needle) != only_needle:
            continue
        midpoint_min = 0.5 * (group["window_start_s"] + group["window_end_s"]) / 60.0
        ax.plot(
            midpoint_min,
            group[value_col],
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            color=colors[channel],
            label=f"Channel: {channel}",
            zorder=3,
        )

    if baseline_line is not None:
        ax.axhline(baseline_line, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8, zorder=2)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def auto_ylim(frame: pd.DataFrame, value_col: str = "normalized_rate_pct",
              floor: float = 120.0) -> tuple[float, float]:
    values = frame[value_col].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    upper = float(values.max()) if values.size else 100.0
    return 0.0, max(floor, upper * 1.15)


def _make_axes(has_probe: bool, has_needle: bool):
    if has_probe and has_needle:
        fig, axes = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(14, 8),
            gridspec_kw={"height_ratios": (3.2, 1.4)},
        )
        return fig, list(axes)
    fig, ax = plt.subplots(figsize=(14, 6.5))
    return fig, [ax]


def _channel_handles(channels, colors):
    return [
        Line2D([0], [0], color=colors[channel], marker="o", markersize=MARKER_SIZE,
               linewidth=LINE_WIDTH, label=f"Channel: {channel}")
        for channel in channels
    ]


def plot_recording(
    frame: pd.DataFrame,
    needle: str,
    out_path: Path,
    colors: dict,
    ylim: tuple[float, float],
    title: str,
    dpi: int = 200,
    yscale: str = "linear",
) -> plt.Figure:
    channels = list(frame["channel"].drop_duplicates())
    has_probe = any(channel != needle for channel in channels)
    has_needle = needle in channels

    frame = frame.copy()
    start_s = float(frame["window_start_s"].min())
    frame["window_start_s"] = frame["window_start_s"] - start_s
    frame["window_end_s"] = frame["window_end_s"] - start_s

    fig, axes = _make_axes(has_probe, has_needle)
    probe_ax = axes[0] if has_probe else None
    needle_ax = axes[-1] if has_needle else None

    if probe_ax is not None:
        _plot_panel(probe_ax, frame, colors, needle=needle, only_needle=False, ylim=ylim, yscale=yscale)
    if needle_ax is not None:
        _plot_panel(needle_ax, frame, colors, needle=needle, only_needle=True, ylim=ylim, yscale=yscale)

    axes[-1].set_xlabel("Time (min)", fontsize=LABEL_SIZE)
    axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=10))
    axes[0].set_title(title, fontsize=TITLE_SIZE, pad=14)
    fig.supylabel("Spike rate (% of baseline mean)", fontsize=LABEL_SIZE, x=0.02)
    fig.legend(
        handles=_channel_handles(channels, colors),
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=LEGEND_SIZE,
        frameon=True,
        borderpad=0.8,
        labelspacing=0.6,
    )
    fig.tight_layout(rect=(0.045, 0, 1, 1))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


def _block_label(block: pd.DataFrame) -> str:
    for column in ("display_label", "epoch_label"):
        if column in block:
            values = block[column].dropna().astype(str).str.strip()
            values = values[values != ""]
            if len(values):
                return str(values.iloc[0])
    return str(block["recording_name"].iloc[0])


def plot_timeline(
    frame: pd.DataFrame,
    needle: str,
    out_path: Path,
    colors: dict,
    ylim: tuple[float, float],
    dpi: int = 200,
    yscale: str = "linear",
    value_col: str = "normalized_rate_pct",
    ylabel: str = "Spike rate (% of baseline mean)",
    title: str = "Spike rate normalised to baseline",
    event_labels: list[str] | None = None,
) -> plt.Figure:
    recordings = list(frame["recording_name"].drop_duplicates())
    channels = list(frame["channel"].drop_duplicates())
    has_probe = any(channel != needle for channel in channels)
    has_needle = needle in channels

    fig, axes = _make_axes(has_probe, has_needle)
    probe_ax = axes[0] if has_probe else None
    needle_ax = axes[-1] if has_needle else None

    offset_min = 0.0
    boundaries = []
    phase_centres = []
    phase_spans = []
    tick_labels = {}
    for i, recording in enumerate(recordings):
        block = frame[frame["recording_name"] == recording]
        start_s = float(block["window_start_s"].min())
        end_s = float(block["window_end_s"].max())
        duration_min = (end_s - start_s) / 60.0
        local = block.copy()
        local["window_start_s"] = local["window_start_s"] - start_s + offset_min * 60.0
        local["window_end_s"] = local["window_end_s"] - start_s + offset_min * 60.0

        label = _block_label(block)
        color = PHASE_COLORS[i % len(PHASE_COLORS)]
        phase_centres.append((offset_min + duration_min / 2.0, label))
        phase_spans.append((offset_min, offset_min + duration_min, label, color))

        step = max(1, math.ceil(duration_min / 10.0))
        for minute in range(0, int(math.floor(duration_min)) + 1, step):
            tick_labels[offset_min + minute] = str(minute)

        if probe_ax is not None:
            _plot_panel(probe_ax, local, colors, needle=needle, only_needle=False, ylim=ylim,
                        yscale=yscale, value_col=value_col,
                        baseline_line=100.0 if value_col == "normalized_rate_pct" else None)
        if needle_ax is not None:
            _plot_panel(needle_ax, local, colors, needle=needle, only_needle=True, ylim=ylim,
                        yscale=yscale, value_col=value_col,
                        baseline_line=100.0 if value_col == "normalized_rate_pct" else None)

        offset_min += duration_min
        boundaries.append(offset_min)

    for ax in axes:
        for start, end, _, color in phase_spans:
            ax.axvspan(start, end, color=color, alpha=1.0, linewidth=0, zorder=0)
        for boundary in boundaries[:-1]:
            ax.axvline(boundary, color="black", linestyle="--", linewidth=1.5, zorder=2)
        positions = sorted(tick_labels)
        ax.set_xticks(positions)
        ax.set_xticklabels([tick_labels[position] for position in positions])
        ax.set_xlim(0, offset_min)
    for centre, label in phase_centres:
        axes[0].text(centre, 0.96, label, transform=axes[0].get_xaxis_transform(),
                     ha="center", va="top", fontsize=PHASE_LABEL_SIZE, weight="bold")

    event_handles = []
    for index, boundary in enumerate(boundaries[:-1]):
        if event_labels is None or index >= len(event_labels):
            continue
        text = str(event_labels[index]).strip()
        if not text:
            continue
        axes[0].text(
            boundary + 0.15,
            0.5,
            text,
            transform=axes[0].get_xaxis_transform(),
            rotation=90,
            ha="left",
            va="center",
            fontsize=12,
            color="#222222",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        )
        event_handles.append(Line2D([], [], color="black", linestyle="--", linewidth=1.5, label=text))

    legend_handles = _channel_handles(channels, colors)
    seen_labels = set()
    for _, _, label, color in phase_spans:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        legend_handles.append(Patch(facecolor=color, edgecolor="#999999", linewidth=0.8, label=label))
    legend_handles.extend(event_handles)
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=LEGEND_SIZE,
        frameon=True,
        borderpad=0.8,
        labelspacing=0.6,
    )

    axes[-1].set_xlabel("Time (min)", fontsize=LABEL_SIZE)
    axes[0].set_title(title, fontsize=TITLE_SIZE, pad=14)
    fig.supylabel(ylabel, fontsize=LABEL_SIZE, x=0.02)
    fig.tight_layout(rect=(0.045, 0, 1, 1))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


# =========================
# CLI
# =========================

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Plot multichannel spike counts normalised to the baseline mean."
    )
    parser.add_argument("--baseline", type=Path, default=None, help="Baseline raw recording CSV (detection mode).")
    parser.add_argument("--post", type=Path, nargs="+", default=None, help="Treated raw recording CSV(s) (detection mode).")
    parser.add_argument("--counts-csv", type=Path, default=None, help="Existing combined spike-count CSV; skips detection.")
    parser.add_argument("--recount", action="store_true", help="Re-run detection even if cached counts exist.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--needle", default="A-000", help="Channel plotted in its own panel.")
    parser.add_argument("--channels", nargs="+", default=None, help="Optional subset of channels to plot.")
    parser.add_argument("--window-s", type=float, default=60.0, help="Spike-count window length in seconds.")
    parser.add_argument("--polarity", default="both", help="Spike polarity: both, neg, or pos.")
    parser.add_argument("--baseline-label", default="Baseline", help="Epoch label used as the 100%% reference.")
    parser.add_argument("--baseline-recording", default=None, help="Recording name used as the baseline when labels are missing.")
    parser.add_argument("--event-label", nargs="+", default=None,
                        help="Annotation shown at each phase boundary, in recording order.")
    parser.add_argument("--title", default=None, help="Figure title prefix; absolute figure gets a suffix.")
    parser.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument("--yscale", choices=("auto", "linear", "log", "symlog"), default="auto",
                        help="Y scale; auto uses symlog when normalised rates exceed 1000%%.")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--shape-mode", choices=("off", "flag_only", "reject"), default="reject")
    parser.add_argument("--shape-separation-ms", type=float, default=1.0)
    parser.add_argument("--shape-depth-fraction", type=float, default=0.2)
    parser.add_argument("--shape-prominence-fraction", type=float, default=0.2)
    parser.add_argument("--shape-noise-multiplier", type=float, default=3.0)
    parser.add_argument("--shape-padding-ms", type=float, default=0.5)
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.counts_csv is None and (args.baseline is None or not args.post):
        parser.error("Provide --counts-csv, or --baseline plus at least one --post recording.")
    if args.out_dir is None:
        args.out_dir = (
            args.counts_csv.parent / "multichannel_normalized"
            if args.counts_csv is not None
            else ROOT.parent / "Output" / "multichannel_normalized_counts"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.counts_csv is not None:
        counts_csv = args.counts_csv
        if not counts_csv.exists():
            raise FileNotFoundError(f"Spike-count CSV not found: {counts_csv}")
    else:
        counts_csv = (
            args.out_dir
            / counts_subdir(args.polarity, args.shape_mode, args.window_s)
            / counts_cache_name(args.window_s)
        )
        if args.recount or not counts_csv.exists():
            counts_csv = run_counting(args, args.out_dir)
        else:
            print(f"[reuse] {counts_csv}")

    counts = pd.read_csv(counts_csv)
    if args.channels is not None:
        wanted = list(dict.fromkeys([*args.channels, args.needle]))
        counts = counts[counts["channel"].isin(wanted)]

    normalised = normalise_counts(
        counts, baseline_label=args.baseline_label, baseline_recording=args.baseline_recording
    )
    if args.counts_csv is None:
        normalised["shape_mode"] = args.shape_mode
    normalised_csv = args.out_dir / "normalized_counts_all.csv"
    normalised.to_csv(normalised_csv, index=False)
    print(f"[wrote] {normalised_csv}")

    ylim = tuple(args.ylim) if args.ylim is not None else auto_ylim(normalised)
    absolute_ylim = auto_ylim(normalised, value_col="spikes_per_min", floor=1.0)
    if args.yscale == "auto":
        finite = normalised["normalized_rate_pct"].replace([np.inf, -np.inf], np.nan).dropna()
        yscale = "symlog" if not finite.empty and float(finite.max()) > 1000.0 else "linear"
    else:
        yscale = args.yscale
    print(f"[plot] y scale: {yscale}")
    channels = list(normalised["channel"].drop_duplicates())
    colors = channel_colors(channels, needle=args.needle)
    norm_title = args.title or "Spike rate normalised to baseline"
    absolute_title = f"{args.title} - absolute spike rate" if args.title else "Absolute spike rate"

    for recording, frame in normalised.groupby("recording_name", sort=False):
        epoch = _block_label(frame)
        out_path = args.out_dir / f"{_safe_name(epoch)}_normalized.png"
        plot_recording(
            frame,
            needle=args.needle,
            out_path=out_path,
            colors=colors,
            ylim=ylim,
            title=f"{epoch} - spike rate normalised to baseline",
            dpi=args.dpi,
            yscale=yscale,
        )
        print(f"[wrote] {out_path}")

    timeline_path = args.out_dir / "all_phases_normalized.png"
    plot_timeline(
        normalised,
        needle=args.needle,
        out_path=timeline_path,
        colors=colors,
        ylim=ylim,
        dpi=args.dpi,
        yscale=yscale,
        title=norm_title,
        event_labels=args.event_label,
    )
    print(f"[wrote] {timeline_path}")

    absolute_path = args.out_dir / "all_phases_absolute.png"
    plot_timeline(
        normalised,
        needle=args.needle,
        out_path=absolute_path,
        colors=colors,
        ylim=absolute_ylim,
        dpi=args.dpi,
        yscale="linear",
        value_col="spikes_per_min",
        ylabel="Spike count per 1 min",
        title=absolute_title,
        event_labels=args.event_label,
    )
    print(f"[wrote] {absolute_path}")


def _safe_name(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value).strip()) or "recording"


if __name__ == "__main__":
    main()
