#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Detect spikes directly from MCS HDF5 electrode recordings.

The CSV-based spike counter needs a wide CSV first, which is impractical for
MEA files with 60-120 channels and 10-30 minutes of 25 kHz data. This module
reads channels from HDF5, asks the existing spike detector for threshold
crossings, and writes only the spike tables.

When a file's chunk spans many channels, reading one channel decompresses every
channel. Those files are first staged once into an int32 memmap under the output
folder; detection then reads single rows from that memmap. Contiguous files and
files chunked along time are read directly.

Each direct worker opens its own read-only HDF5 handle. HDF5 reads are
serialized with a lock, while filtering and event detection run in parallel.
"""

import argparse
import importlib.util
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MCS_MODULE_PATH = ROOT / "Raw Conversion" / "MCS_to_CSV.py"
SPIKE_MODULE_PATH = ROOT / "Spike Processing" / "Spike Count Multiple CSVs ordered.py"

HDF5_LOCK = threading.Lock()
DEFAULT_OUT_DIR_NAME = "spike_counts_per_min"
# Keep 20% of free RAM for the OS; two CPUs are kept free by the worker cap.
RAM_HEADROOM = 0.8
# Measured peak per worker: raw int32 (4) + float64 signal (8) + time array (8)
# + searchsorted temporaries (~24) + filtered trace and z-score (~16) + slack.
# 48 was too low; 7 workers swapped on a 45M-sample channel.
BYTES_PER_SAMPLE_ESTIMATE = 128
WORKER_FIXED_BYTES = 256 * 1024 * 1024
STAGE_DIR_NAME = "stage"
STAGE_BYTES_PER_SAMPLE = 4


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_dependencies():
    mcs = _load_module(MCS_MODULE_PATH, "mcs_to_csv_for_spikes")
    spike = _load_module(SPIKE_MODULE_PATH, "spike_counter_for_mcs")
    return mcs, spike


def _available_ram_bytes():
    """Best-effort free physical memory, without adding psutil."""
    if os.name == "nt":
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return None

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _plan_workers(total_tasks, max_samples, available_bytes=None, cpu_count=None):
    """Keep two CPUs and 20% of free RAM for the system, then cap the worker count."""
    cpu = int(cpu_count or os.cpu_count() or 2)
    cpu_limit = max(1, cpu - 2)
    workers = min(int(total_tasks), cpu_limit)
    detail = {
        "cpu_count": cpu,
        "cpu_limit": cpu_limit,
        "initial_workers": workers,
        "available_ram_bytes": available_bytes,
        "estimated_bytes_per_worker": None,
        "ram_limit": None,
        "final_workers": workers,
        "ram_limited": False,
    }
    if available_bytes and max_samples > 0:
        per_worker = max(
            1,
            int(max_samples) * BYTES_PER_SAMPLE_ESTIMATE + WORKER_FIXED_BYTES,
        )
        ram_limit = max(1, int(available_bytes * RAM_HEADROOM) // per_worker)
        detail["estimated_bytes_per_worker"] = per_worker
        detail["ram_limit"] = ram_limit
        if ram_limit < workers:
            workers = ram_limit
            detail["ram_limited"] = True
    detail["final_workers"] = max(1, workers) if total_tasks else 0
    return detail["final_workers"], detail


def _empty_bundle():
    return {
        "results_rows": [],
        "spike_event_rows": [],
        "diagnostic_output_rows": [],
        "waveform_summary_rows": [],
        "shape_audit_dirs": [],
        "amplitude_recording_rows": [],
        "amplitude_event_rows": [],
        "amplitude_waveform_rows": [],
    }


def _inspect_file(path, mcs, spike):
    """Read metadata only; the parent never keeps a shared HDF5 handle."""
    with h5py.File(path, "r") as handle:
        protocol = mcs._attr_text(handle, "McsHdf5ProtocolType")
        if protocol.lower() != "rawdata":
            raise ValueError(
                f"{Path(path).name} has protocol {protocol or 'unknown'!r}. "
                "SPIE import needs an MCS 'RawData' HDF5 export."
            )
        version = int(handle.attrs.get("McsHdf5ProtocolVersion", 1))
        if version not in (1, 2, 3):
            raise ValueError(f"Unsupported MCS RawData protocol version: {version}")

        recording_group = mcs._select_recording(handle["Data"], recording=None)
        if "AnalogStream" not in recording_group:
            raise ValueError(
                f"{Path(path).name} contains no analog streams (only events/segments)."
            )
        stream_name, stream, meta, indices, chosen, tick_us = mcs._plan_channels(
            recording_group["AnalogStream"]
        )
        data = stream["ChannelData"]
        shape = tuple(int(value) for value in data.shape)
        chunks = tuple(int(value) for value in data.chunks) if data.chunks else None
        compression = data.compression
        shuffle = bool(data.shuffle)
        amplification = int(chunks[0]) if chunks else 1
        strategy = "staged" if chunks is not None and chunks[0] > 2 else "direct"
        n_samples = int(data.shape[1])
        segments = mcs._segments(stream["ChannelDataTimeStamps"][()], tick_us)
        fs = mcs._sample_rate_hz(tick_us)
        if fs / 2.0 <= spike.HP_SPIKE_BAND[0]:
            raise ValueError(
                f"{Path(path).name}: sampling rate {fs:g} Hz cannot support spike band "
                f"{spike.HP_SPIKE_BAND[0]:g}-{spike.HP_SPIKE_BAND[1]:g} Hz "
                f"(Nyquist {fs / 2.0:g} Hz). Needs fs > "
                f"{2 * spike.HP_SPIKE_BAND[0]:g} Hz. Use LFP analysis for low-rate recordings."
            )

        scales = np.array(
            [
                mcs._unit_scale_to_uv(
                    channel["unit"], channel["exponent"], channel["conversion_factor"]
                )
                for channel in meta
            ],
            dtype=np.float64,
        )
        zeros = np.array([channel["ad_zero"] for channel in meta], dtype=np.float64)
        recording_name = recording_group.name.split("/")[-1]
        stream_path = stream.name

    return {
        "path": str(path),
        "name": Path(path).name,
        "recording": recording_name,
        "stream_name": stream_name,
        "stream_path": stream_path,
        "channels": list(chosen),
        "channel_indices": [int(index) for index in indices],
        "channel_scales": scales[indices],
        "channel_zeros": zeros[indices],
        "tick_us": int(tick_us),
        "fs": float(fs),
        "n_samples": n_samples,
        "segments": segments,
        "protocol_version": version,
        "shape": shape,
        "chunks": chunks,
        "compression": compression,
        "shuffle": shuffle,
        "amplification": amplification,
        "strategy": strategy,
        "file_size": int(Path(path).stat().st_size),
    }


def _select_channels(recording, channels):
    """Keep the requested channel labels in file order; blank means all."""
    requested = [str(label).strip() for label in (channels or []) if str(label).strip()]
    if not requested:
        return
    requested_set = set(requested)
    available = list(recording["channels"])
    missing = [label for label in requested if label not in set(available)]
    keep = [index for index, label in enumerate(available) if label in requested_set]
    if missing:
        print(
            f"[mcs-spikes] {recording['name']}: channel(s) not found: "
            f"{', '.join(missing)}",
            flush=True,
        )
    if not keep:
        raise ValueError(
            f"{recording['name']}: none of the requested channels were found "
            f"({', '.join(requested)}). Available: {', '.join(available)}"
        )
    recording["channels"] = [available[index] for index in keep]
    recording["channel_indices"] = [recording["channel_indices"][index] for index in keep]
    recording["channel_scales"] = np.asarray(recording["channel_scales"])[keep]
    recording["channel_zeros"] = np.asarray(recording["channel_zeros"])[keep]
    print(
        f"[mcs-spikes] {recording['name']}: "
        f"{len(keep)}/{len(available)} channel(s) selected",
        flush=True,
    )


def _stage_recording(recording, out_dir):
    """Decompress a channel-amplified file once into an int32 memmap."""
    stage_root = Path(out_dir) / STAGE_DIR_NAME
    stage_dir = stage_root / Path(recording["path"]).stem
    stage_dir.mkdir(parents=True, exist_ok=True)
    memmap_path = stage_dir / "channels.dat"
    selected = [int(index) for index in recording["channel_indices"]]
    n_samples = int(recording["n_samples"])
    required = len(selected) * n_samples * STAGE_BYTES_PER_SAMPLE
    free = int(shutil.disk_usage(str(out_dir)).free)
    if free < required:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise OSError(
            f"Not enough free space for staging in {out_dir}: need "
            f"{required / 1024**3:.1f} GB, have {free / 1024**3:.1f} GB."
        )
    print(
        f"[mcs-spikes] {recording['rec_name']}: staging "
        f"{len(selected)}/{recording['shape'][0]} channels -> "
        f"{required / 1024**3:.1f} GB at {memmap_path}",
        flush=True,
    )
    total_gb = n_samples * recording["shape"][0] * STAGE_BYTES_PER_SAMPLE / 1024**3
    print(
        f"[mcs-spikes] {recording['rec_name']}: staging started, "
        f"~{total_gb:.1f} GB decompressed, ~{total_gb / 0.1476 / 60:.1f} min",
        flush=True,
    )
    started = time.perf_counter()
    next_report = 0.0
    try:
        mm = np.memmap(
            memmap_path,
            dtype=np.int32,
            mode="w+",
            shape=(len(selected), n_samples),
        )
        with h5py.File(recording["path"], "r") as handle:
            data = handle[recording["stream_path"]]["ChannelData"]
            chunk_cols = int(data.chunks[1]) if data.chunks else 1_000_000
            block = chunk_cols * max(1, 250_000 // chunk_cols)
            for start in range(0, n_samples, block):
                stop = min(start + block, n_samples)
                block_data = data[:, start:stop]
                mm[:, start:stop] = block_data[selected, :]
                del block_data
                elapsed = time.perf_counter() - started
                if elapsed >= next_report:
                    done_gb = stop * recording["shape"][0] * STAGE_BYTES_PER_SAMPLE / 1024**3
                    rate = done_gb / max(elapsed, 1e-9)
                    eta = (total_gb - done_gb) / rate if rate > 0 else 0.0
                    print(
                        f"[mcs-spikes] {recording['rec_name']}: staging "
                        f"{100.0 * stop / n_samples:.0f}% "
                        f"({done_gb:.1f}/{total_gb:.1f} GB decompressed) "
                        f"elapsed {elapsed:.0f}s ETA {eta:.0f}s",
                        flush=True,
                    )
                    next_report = elapsed + 2.0
        mm.flush()
        del mm
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    recording["stage_path"] = str(memmap_path)
    recording["stage_dir"] = str(stage_dir)
    recording["stage_seconds"] = time.perf_counter() - started
    print(
        f"[mcs-spikes] {recording['rec_name']}: staged in "
        f"{recording['stage_seconds']:.1f}s",
        flush=True,
    )


def _recording_context(recording, rec_idx, spike, label_lookup, metadata_lookup):
    rec_meta = spike._metadata_for_path(recording["path"], metadata_lookup)
    epoch_label = spike._epoch_label_for_path(recording["path"], label_lookup)
    if not epoch_label:
        epoch_label = str(rec_meta.get("epoch_label") or rec_meta.get("label") or "")
    phase = str(rec_meta.get("phase") or "").strip() or spike._phase_from_epoch_label(epoch_label)
    phase_group, phase_warning_reason = spike.normalize_amplitude_phase(
        phase=phase, epoch_label=epoch_label
    )
    rec_name = Path(recording["path"]).stem
    return {
        "rec_idx": rec_idx,
        "rec_name": rec_name,
        "source_file": recording["path"],
        "epoch_label": epoch_label,
        "phase": phase,
        "phase_group": phase_group,
        "phase_warning_reason": phase_warning_reason,
        "segment_id": str(rec_meta.get("segment_id") or rec_meta.get("recording_id") or rec_name),
        "preparation_id": str(rec_meta.get("preparation_id") or ""),
        "metadata_present": bool(rec_meta),
    }


def _run_channel(task, mcs, spike, shape_config, selected_polarity,
                 classification_window_ms, selected_window_sec, out_dir):
    # Time the first channel of each file so a slow run is diagnosable.
    timings = {} if task["channel_index"] == 1 else None
    stamp = time.perf_counter()
    if task.get("stage_path"):
        staged = np.memmap(
            task["stage_path"],
            dtype=np.int32,
            mode="r",
            shape=(task["stage_rows"], task["n_samples"]),
        )
        raw = np.array(staged[task["stage_row"]], dtype=np.int32, copy=True)
        del staged
    else:
        with HDF5_LOCK:
            with h5py.File(task["path"], "r") as handle:
                raw = handle[task["stream_path"]]["ChannelData"][task["row_index"], :]
    if timings is not None:
        timings["read"] = time.perf_counter() - stamp
        stamp = time.perf_counter()

    x = np.asarray(raw, dtype=np.float64)
    x = np.round((x - task["zero"]) * task["scale"], 6)
    del raw
    if timings is not None:
        timings["scale"] = time.perf_counter() - stamp
        stamp = time.perf_counter()

    t = mcs._time_for_block(
        task["segments"], task["tick_us"], 0, task["n_samples"], False, task["fs"]
    )
    t = np.round(t, 6)
    if timings is not None:
        timings["time"] = time.perf_counter() - stamp
        stamp = time.perf_counter()

    if task["window"] is not None:
        mask = (t >= task["window"][0]) & (t < task["window"][1])
        t = t[mask]
        x = x[mask]
    if x.size == 0:
        return _empty_bundle()

    bundle = spike._process_channel_signal(
        x,
        t,
        task["channel"],
        task["rec_name"],
        task["path"],
        task["rec_idx"],
        task["epoch_label"],
        task["phase"],
        task["phase_group"],
        task["phase_warning_reason"],
        task["preparation_id"],
        task["segment_id"],
        task["metadata_present"],
        selected_polarity,
        shape_config,
        classification_window_ms,
        selected_window_sec,
        out_dir,
        include_diagnostics=False,
        include_amplitude=False,
    )
    if timings is not None:
        timings["detect"] = time.perf_counter() - stamp
        print(
            f"[mcs-spikes] phase {task['rec_name']}: "
            f"read {timings['read']:.1f}s scale {timings['scale']:.1f}s "
            f"time {timings['time']:.1f}s detect {timings['detect']:.1f}s "
            f"({task['n_samples']:,} samples)",
            flush=True,
        )
    return bundle


class _Progress:
    def __init__(self, total_channels, total_samples, file_count, mcs):
        self.total_channels = total_channels
        self.total_samples = total_samples
        self.file_count = file_count
        self.mcs = mcs
        self.done_channels = 0
        self.done_samples = 0
        self.started = time.time()
        self.lock = threading.Lock()

    def update(self, task, samples):
        with self.lock:
            self.done_channels += 1
            self.done_samples += int(samples)
            elapsed = max(time.time() - self.started, 1e-9)
            rate = self.done_samples / elapsed
            percent = 100.0 * self.done_samples / self.total_samples if self.total_samples else 0.0
            remaining = max(self.total_samples - self.done_samples, 0)
            eta = remaining / rate if rate > 0 else None
            print(
                f"[mcs-spikes] file {task['file_index']}/{self.file_count} "
                f"ch {task['channel_index']}/{task['file_channels']} "
                f"({self.done_samples:,}/{self.total_samples:,} samples, {percent:.1f}%) "
                f"elapsed {self.mcs._format_duration(elapsed)} "
                f"ETA {self.mcs._format_duration(eta) if eta is not None else '--:--'} "
                f"{self.mcs._format_rate(rate)}",
                flush=True,
            )


def _normalise_lookups(spike, epoch_labels_by_path, recording_metadata_by_path):
    return (
        spike._normalise_label_lookup(epoch_labels_by_path),
        spike._normalise_metadata_lookup(recording_metadata_by_path),
    )


def _prepare(paths, mcs, spike, epoch_labels_by_path, recording_metadata_by_path,
             channels=None):
    label_lookup, metadata_lookup = _normalise_lookups(
        spike, epoch_labels_by_path, recording_metadata_by_path
    )
    recordings = []
    for rec_idx, path in enumerate(paths, start=1):
        recorded = _inspect_file(path, mcs, spike)
        print(
            f"[mcs-spikes] {recorded['name']}: chunks={recorded['chunks']} "
            f"{recorded['compression'] or 'contiguous'}, "
            f"amplification {recorded['amplification']}x, "
            f"strategy={recorded['strategy']}",
            flush=True,
        )
        _select_channels(recorded, channels)
        recorded.update(
            _recording_context(recorded, rec_idx, spike, label_lookup, metadata_lookup)
        )
        recordings.append(recorded)
    return recordings


def _build_tasks(recordings, window):
    tasks = []
    for recording in recordings:
        for channel_index, (channel, row_index, scale, zero) in enumerate(
            zip(
                recording["channels"],
                recording["channel_indices"],
                recording["channel_scales"],
                recording["channel_zeros"],
            ),
            start=1,
        ):
            tasks.append(
                {
                    "path": recording["path"],
                    "stream_path": recording["stream_path"],
                    "row_index": int(row_index),
                    "stage_path": recording.get("stage_path"),
                    "stage_row": channel_index - 1,
                    "stage_rows": len(recording["channels"]),
                    "channel": channel,
                    "channel_index": channel_index,
                    "file_index": recording["rec_idx"],
                    "file_channels": len(recording["channels"]),
                    "scale": float(scale),
                    "zero": float(zero),
                    "tick_us": recording["tick_us"],
                    "fs": recording["fs"],
                    "n_samples": recording["n_samples"],
                    "segments": recording["segments"],
                    "window": window,
                    "rec_idx": recording["rec_idx"],
                    "rec_name": recording["rec_name"],
                    "epoch_label": recording["epoch_label"],
                    "phase": recording["phase"],
                    "phase_group": recording["phase_group"],
                    "phase_warning_reason": recording["phase_warning_reason"],
                    "preparation_id": recording["preparation_id"],
                    "segment_id": recording["segment_id"],
                    "metadata_present": recording["metadata_present"],
                }
            )
    return tasks


def _write_outputs(out_dir, recordings, result_rows, event_rows, spike, spike_events_path,
                   selected_window_sec, provenance):
    out_table = pd.DataFrame(result_rows)
    out_name = (
        "ALL_RECORDINGS__spike_counts_per_min.csv"
        if abs(selected_window_sec - 60.0) < 1e-9
        else f"ALL_RECORDINGS__spike_counts_per_{spike.format_window_label(selected_window_sec)}.csv"
    )
    out_csv = os.path.join(out_dir, out_name)
    out_table.to_csv(out_csv, index=False)

    events_csv = spike_events_path or os.path.join(out_dir, "spike_events.csv")
    pd.DataFrame(event_rows, columns=spike.SPIKE_EVENT_COLUMNS).to_csv(events_csv, index=False)

    provenance = dict(provenance)
    provenance.update(
        {
            "ordered_recording_list": [recording["path"] for recording in recordings],
            "labels_by_path": {recording["path"]: recording["epoch_label"] for recording in recordings},
            "spike_count_csv": out_csv,
            "spike_events_csv": events_csv,
            "spike_count_rows": len(result_rows),
            "spike_event_rows": len(event_rows),
        }
    )
    provenance_json = os.path.join(out_dir, "mcs_hdf5_spike_provenance.json")
    Path(provenance_json).write_text(
        json.dumps(provenance, indent=2, default=str), encoding="utf-8"
    )
    print(f"[mcs-spikes] wrote {out_csv}")
    print(f"[mcs-spikes] wrote {events_csv}")
    print(f"[mcs-spikes] wrote {provenance_json}")
    return out_csv, events_csv, provenance_json, provenance


def process_hdf5s(
    h5_paths,
    out_dir=None,
    epoch_labels_by_path=None,
    window_sec=None,
    polarity=None,
    classification_window_ms=None,
    spike_events_path=None,
    recording_metadata_by_path=None,
    return_outputs=False,
    shape_settings=None,
    max_workers=None,
    time_window=None,
    channels=None,
):
    """
    Detect spikes in one or more MCS RawData HDF5 files without writing raw CSVs.

    Parameters
    ----------
    h5_paths : list of .h5 files. Order is preserved.
    out_dir : output folder; defaults to the first file's parent / spike_counts_per_min.
    epoch_labels_by_path : optional {path: label} mapping, one label per recording.
    window_sec : spike-count window size in seconds; default is the spike script's 1 min.
    polarity : \"both\", \"neg\", or \"pos\".
    classification_window_ms : peak-alignment window passed to the detector.
    spike_events_path : optional override for the event CSV.
    recording_metadata_by_path : optional metadata dicts keyed by path.
    return_outputs : return a dict instead of only the counts CSV path.
    shape_settings : forwarded to spike_shape_qc; fused mode writes no shape audit.
    max_workers : optional cap, still limited by CPU count - 2 and the RAM guard.
    time_window : optional (start_s, end_s) for quick checks.
    channels : optional list of channel labels; blank or None means all channels.
    """
    mcs, spike = _load_dependencies()
    paths = [str(path) for path in (h5_paths if isinstance(h5_paths, (list, tuple)) else [h5_paths])]
    if not paths:
        raise FileNotFoundError("No HDF5 files were provided for spike counting.")

    selected_polarity = spike.normalize_polarity(polarity)
    shape_config = spike.shape_qc.settings(shape_settings)
    selected_window_sec = float(window_sec if window_sec is not None else spike.WINDOW_SEC)
    if selected_window_sec <= 0:
        raise ValueError("Spike-count window size must be greater than zero.")

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(paths[0]) or ".", DEFAULT_OUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    window = None
    if time_window is not None:
        start_s, end_s = (float(value) for value in time_window)
        if end_s <= start_s:
            raise ValueError("time_window end must be greater than start.")
        window = (start_s, end_s)

    print(f"[settings] spike polarity: {spike.polarity_label(selected_polarity)} ({selected_polarity})")
    requested_channels = [
        str(label).strip() for label in (channels or []) if str(label).strip()
    ]
    recordings = _prepare(
        paths,
        mcs,
        spike,
        epoch_labels_by_path,
        recording_metadata_by_path,
        channels=requested_channels,
    )
    total_channels = sum(len(recording["channels"]) for recording in recordings)
    total_samples = sum(
        recording["n_samples"] * len(recording["channels"]) for recording in recordings
    )

    available = _available_ram_bytes()
    worker_count, worker_detail = _plan_workers(
        total_channels, max(recording["n_samples"] for recording in recordings), available
    )
    if max_workers is not None:
        worker_count = max(1, min(worker_count, int(max_workers)))
        worker_detail["requested_max_workers"] = int(max_workers)
    print(
        f"[mcs-spikes] {len(recordings)} file(s), {total_channels} channel(s), "
        f"{total_samples:,} samples"
    )
    print(
        f"[mcs-spikes] {worker_count} worker(s), {worker_detail['cpu_count']} logical CPUs - 2"
    )
    if worker_detail["ram_limited"]:
        print(
            f"[mcs-spikes] RAM guard: {available / 1024**3:.1f} GB free, "
            f"~{worker_detail['estimated_bytes_per_worker'] / 1024**3:.1f} GB per channel, "
            f"limit {worker_detail['ram_limit']} worker(s)"
        )

    stage_dirs = []
    try:
        for recording in recordings:
            if recording["strategy"] == "staged":
                _stage_recording(recording, out_dir)
                stage_dirs.append(recording["stage_dir"])

        tasks = _build_tasks(recordings, window)
        progress = _Progress(total_channels, total_samples, len(recordings), mcs)
        bundles = [None] * len(tasks)
        if worker_count <= 1:
            for index, task in enumerate(tasks):
                bundles[index] = _run_channel(
                    task, mcs, spike, shape_config, selected_polarity,
                    classification_window_ms, selected_window_sec, out_dir,
                )
                progress.update(task, task["n_samples"])
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = {
                    pool.submit(
                        _run_channel, task, mcs, spike, shape_config, selected_polarity,
                        classification_window_ms, selected_window_sec, out_dir,
                    ): index
                    for index, task in enumerate(tasks)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    bundles[index] = future.result()
                    progress.update(tasks[index], tasks[index]["n_samples"])

        result_rows = []
        event_rows = []
        for bundle in bundles:
            if bundle:
                result_rows.extend(bundle["results_rows"])
                event_rows.extend(bundle["spike_event_rows"])
        if not result_rows:
            raise RuntimeError("No spike-count results were generated.")

        provenance = {
            "pipeline": "MCS_to_spikes.py::process_hdf5s",
            "mcs_script_path": str(MCS_MODULE_PATH),
            "spike_script_path": str(SPIKE_MODULE_PATH),
            "files": [
                {
                    "path": recording["path"],
                    "recording": recording["recording"],
                    "stream": recording["stream_name"],
                    "fs_hz": recording["fs"],
                    "samples": recording["n_samples"],
                    "channels": recording["channels"],
                    "protocol_version": recording["protocol_version"],
                    "shape": recording["shape"],
                    "chunks": recording["chunks"],
                    "compression": recording["compression"],
                    "shuffle": recording["shuffle"],
                    "amplification": recording["amplification"],
                    "strategy": recording["strategy"],
                    "stage_seconds": recording.get("stage_seconds"),
                }
                for recording in recordings
            ],
            "stage_root": str(Path(out_dir) / STAGE_DIR_NAME),
            "requested_channels": requested_channels,
            "window_sec": selected_window_sec,
            "time_window": list(window) if window else None,
            "polarity": selected_polarity,
            "shape_settings": shape_config,
            "spike_band_hz": list(spike.HP_SPIKE_BAND),
            "spike_z_threshold": float(spike.SPIKE_Z_THR),
            "refractory_ms": float(spike.REFRACTORY_MS),
            "worker_plan": worker_detail,
            "worker_count": worker_count,
        }
        out_csv, events_csv, provenance_json, provenance = _write_outputs(
            out_dir, recordings, result_rows, event_rows, spike, spike_events_path,
            selected_window_sec, provenance,
        )
        outputs = {
            "spike_count_csv": out_csv,
            "spike_events_csv": events_csv,
            "spike_processing_provenance_json": provenance_json,
            "spike_processing_provenance": provenance,
            "worker_count": worker_count,
            "worker_plan": worker_detail,
        }
        return outputs if return_outputs else out_csv
    finally:
        for stage_dir in stage_dirs:
            shutil.rmtree(stage_dir, ignore_errors=True)
        try:
            (Path(out_dir) / STAGE_DIR_NAME).rmdir()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="MCS .h5 file or folder containing .h5 files")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--polarity", default=None, choices=("both", "neg", "pos"))
    parser.add_argument("--window-sec", type=float, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--channels", default=None, help="comma-separated channel labels")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        paths = sorted(str(path) for path in input_path.glob("*.h5"))
    else:
        paths = [str(input_path)]
    channels = (
        [label.strip() for label in args.channels.split(",") if label.strip()]
        if args.channels
        else None
    )
    process_hdf5s(
        paths,
        out_dir=args.out_dir,
        polarity=args.polarity,
        window_sec=args.window_sec,
        max_workers=args.max_workers,
        channels=channels,
    )


if __name__ == "__main__":
    main()
