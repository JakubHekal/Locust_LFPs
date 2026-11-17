#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert MCS (Multi Channel Systems) HDF5 RawData files to the SPIE CSV format.

Supported input
---------------
- .h5 files exported by Multi Channel DataManager / Multi Channel Experimenter
  with root attribute McsHdf5ProtocolType == "RawData" (protocol versions 1-3).

Not supported
-------------
- Old .mcd recordings (convert them to .h5 in the MCS software first).
- CMOS .cmcr / .cmtr files (no continuous electrode traces).

Output
------
One CSV with columns ``time_s, <label 1>, ..., <label N>`` in microvolts, or one
CSV per channel when ``split_by_channel=True``. Time is reconstructed from the
per-stream ``ChannelDataTimeStamps`` so gaps between acquisition segments are
preserved. ``continuous_time=True`` ignores the stored timestamps and uses
``sample_index / fs`` instead.
"""

import re
from pathlib import Path

import h5py
import numpy as np

VOLTAGE_UNITS = {"v", "mv", "uv", "µv", "μv"}
MCS_TICK_US = 1.0  # one MCS tick is 1 microsecond
BLOCK_SAMPLES = 1_000_000
TIME_FORMAT = "%.6f"


def _text(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "ignore").strip()
    return str(value).strip()


def _attr_text(group, name: str) -> str:
    if name not in group.attrs:
        return ""
    return _text(group.attrs[name])


def _list_h5_files(input_path: str) -> list[str]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() != ".h5":
            raise ValueError(
                f"Unsupported file type: {path.name}. MCS import needs a .h5 RawData export, "
                "not .mcd/.cmcr/.cmtr."
            )
        return [str(path)]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    files = sorted(str(p) for p in path.iterdir() if p.suffix.lower() == ".h5")
    if not files:
        raise FileNotFoundError(f"No .h5 files found in: {input_path}")
    if len(files) > 1:
        names = ", ".join(Path(f).name for f in files)
        raise ValueError(f"Folder contains several .h5 files ({names}). Select one file.")
    return files


def _select_recording(data_group: h5py.Group, recording=None) -> h5py.Group:
    names = sorted(name for name in data_group.keys() if name.startswith("Recording_"))
    if not names:
        raise ValueError("No Data/Recording_* group found in the HDF5 file.")
    if recording is None:
        if len(names) > 1:
            raise ValueError(
                f"File contains several recordings ({', '.join(names)}). Select one with recording=."
            )
        return data_group[names[0]]
    for name in names:
        if name == str(recording) or name == f"Recording_{recording}":
            return data_group[name]
    raise ValueError(f"Recording {recording!r} not found. Available: {', '.join(names)}")


def _sample_rate_hz(tick_us: int) -> float:
    if tick_us <= 0:
        raise ValueError(f"Invalid Tick value in channel metadata: {tick_us}")
    return 1e6 / (tick_us * MCS_TICK_US)


def _unit_scale_to_uv(unit: str, exponent: int, conversion_factor: int) -> float:
    """Volts per ADC count, converted to microvolts."""
    unit_key = unit.strip().lower()
    if unit_key not in VOLTAGE_UNITS:
        raise ValueError(f"Channel unit {unit!r} is not a voltage.")
    volts = float(conversion_factor) * (10.0 ** int(exponent))
    if unit_key == "v":
        return volts * 1e6
    if unit_key == "mv":
        return volts * 1e3
    return volts  # already microvolts


def _segments(time_stamps: np.ndarray, tick_us: int) -> list[dict]:
    """Return one entry per acquisition segment: first index, start time (us), count."""
    segments = []
    for row in np.atleast_2d(time_stamps):
        first_ts, first_index, last_index = (int(v) for v in row[:3])
        count = last_index - first_index + 1
        if count <= 0:
            continue
        segments.append(
            {
                "first_index": first_index,
                "start_us": float(first_ts) * MCS_TICK_US,
                "count": count,
            }
        )
    if not segments:
        raise ValueError("No valid timestamps in ChannelDataTimeStamps.")
    return segments


def _time_for_block(segments: list[dict], tick_us: int, start: int, stop: int,
                    continuous: bool, fs: float) -> np.ndarray:
    idx = np.arange(start, stop, dtype=np.int64)
    if continuous:
        return idx / fs
    starts = np.array([seg["first_index"] for seg in segments], dtype=np.int64)
    seg_ids = np.searchsorted(starts, idx, side="right") - 1
    seg_ids = np.clip(seg_ids, 0, len(segments) - 1)
    seg_first = starts[seg_ids]
    seg_start_us = np.array([seg["start_us"] for seg in segments], dtype=np.float64)[seg_ids]
    if np.any(idx < seg_first) or np.any(idx >= seg_first + np.array([seg["count"] for seg in segments])[seg_ids]):
        raise ValueError("ChannelDataTimeStamps do not cover every sample in the stream.")
    return (seg_start_us + (idx - seg_first) * tick_us * MCS_TICK_US) / 1e6


def _channel_meta(info_channel: np.ndarray) -> list[dict]:
    meta = []
    for channel in info_channel:
        meta.append(
            {
                "label": _text(channel["Label"]),
                "unit": _text(channel["Unit"]),
                "exponent": int(channel["Exponent"]),
                "ad_zero": int(channel["ADZero"]),
                "tick_us": int(channel["Tick"]),
                "conversion_factor": int(channel["ConversionFactor"]),
            }
        )
    return meta


def _is_electrode_stream(group: h5py.Group, meta: list[dict]) -> bool:
    subtype = _attr_text(group, "DataSubType").lower()
    if subtype and subtype != "electrode":
        return False
    return bool(meta) and all(ch["unit"].lower() in VOLTAGE_UNITS for ch in meta)


def _describe_stream(group: h5py.Group, name: str) -> str:
    label = _attr_text(group, "Label") or name
    subtype = _attr_text(group, "DataSubType") or "unknown"
    return f"{name} ({subtype}: {label})"


def _unique_labels(labels: list[str]) -> list[str]:
    seen = {}
    out = []
    for label in labels:
        name = label or "channel"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        out.append(name)
    return out


def _safe_filename(label: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", label).strip(" .") or "channel"


def _plan_channels(analog_stream: h5py.Group, target_channels=None):
    streams = []
    for name in sorted(analog_stream.keys()):
        group = analog_stream[name]
        if not isinstance(group, h5py.Group) or "ChannelData" not in group:
            continue
        meta = _channel_meta(group["InfoChannel"][()])
        streams.append((name, group, meta))

    electrode = [(name, group, meta) for name, group, meta in streams if _is_electrode_stream(group, meta)]
    if not electrode:
        available = ", ".join(_describe_stream(group, name) for name, group, _ in streams) or "none"
        raise ValueError(
            "No electrode (voltage) analog stream found. Available streams: " + available
        )

    primary_name, primary_group, primary_meta = electrode[0]
    for name, group, meta in electrode[1:]:
        print(f"[skip] non-primary electrode stream: {_describe_stream(group, name)}")

    for name, group, meta in streams:
        if name not in {item[0] for item in electrode}:
            print(f"[skip] non-electrode stream: {_describe_stream(group, name)}")

    ticks = {ch["tick_us"] for ch in primary_meta}
    if len(ticks) > 1:
        raise ValueError(f"Channels in {primary_name} use different Tick values: {sorted(ticks)}")
    tick_us = ticks.pop()

    labels = _unique_labels([ch["label"] for ch in primary_meta])
    index_by_label = {label: i for i, label in enumerate(labels)}

    if target_channels:
        missing = [name for name in target_channels if name not in index_by_label]
        if missing:
            raise ValueError(
                f"Channels not found in {primary_name}: {', '.join(missing)}. "
                f"Available: {', '.join(labels)}"
            )
        indices = [index_by_label[name] for name in target_channels]
        chosen = [labels[i] for i in indices]
    else:
        indices = list(range(len(labels)))
        chosen = labels

    return primary_name, primary_group, primary_meta, indices, chosen, tick_us


def combine_mcs_to_one_csv(
    input_path: str,
    out_csv: str,
    target_channels=None,
    diff_pairs=(),
    continuous_time: bool = False,
    time_col_name: str = "time_s",
    split_by_channel: bool = False,
    time_window=None,
    recording=None,
):
    """
    Convert one MCS HDF5 RawData file into SPIE-ready CSV file(s).

    Parameters
    ----------
    input_path : .h5 file, or a folder containing exactly one .h5 file.
    out_csv : output CSV path. With ``split_by_channel`` it is treated as a base
        name and each channel is written next to it as ``<stem>_<channel>.csv``.
    target_channels : optional list of channel labels; unknown labels raise.
    diff_pairs : optional list of (a, b) label pairs; adds an ``a-b`` column.
    continuous_time : ignore stored timestamps, use ``sample_index / fs``.
    time_col_name : name of the time column, SPIE expects ``time_s``.
    split_by_channel : write one CSV per channel instead of one wide CSV.
    time_window : optional (start_s, end_s) in seconds, end exclusive.
    recording : optional recording index or "Recording_N" name.
    """
    path = _list_h5_files(input_path)[0]
    out_path = Path(out_csv)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "r") as handle:
        protocol = _attr_text(handle, "McsHdf5ProtocolType")
        if protocol.lower() != "rawdata":
            raise ValueError(
                f"{Path(path).name} has protocol {protocol or 'unknown'!r}. "
                "SPIE import needs an MCS 'RawData' HDF5 export."
            )
        version = int(handle.attrs.get("McsHdf5ProtocolVersion", 1))
        if version not in (1, 2, 3):
            raise ValueError(f"Unsupported MCS RawData protocol version: {version}")

        recording_group = _select_recording(handle["Data"], recording=recording)
        if "AnalogStream" not in recording_group:
            raise ValueError(
                f"{Path(path).name} contains no analog streams (only events/segments). "
                "SPIE needs continuous electrode traces."
            )
        analog = recording_group["AnalogStream"]
        stream_name, stream, meta, channel_indices, chosen, tick_us = _plan_channels(
            analog, target_channels=target_channels
        )

        fs = _sample_rate_hz(tick_us)
        data = stream["ChannelData"]
        n_samples = int(data.shape[1])
        segments = _segments(stream["ChannelDataTimeStamps"][()], tick_us)

        pairs = []
        for a, b in diff_pairs or []:
            for name in (a, b):
                if name not in chosen:
                    raise ValueError(f"Differential pair channel {name!r} is not in the selected channels.")
            pairs.append((a, b))

        print(f"[mcs] {Path(path).name}")
        print(f"[mcs] recording={recording_group.name.split('/')[-1]} stream={stream_name}")
        print(f"[mcs] {len(chosen)} channels, {n_samples:,} samples, fs={fs:g} Hz, "
              f"segments={len(segments)}")
        print(f"[mcs] protocol v{version}")

        scales = np.array([_unit_scale_to_uv(ch["unit"], ch["exponent"], ch["conversion_factor"])
                           for ch in meta], dtype=np.float64)
        ad_zeros = np.array([ch["ad_zero"] for ch in meta], dtype=np.float64)

        selected_scale = scales[channel_indices]
        selected_zero = ad_zeros[channel_indices]

        window = None
        if time_window is not None:
            start_s, end_s = (float(v) for v in time_window)
            if end_s <= start_s:
                raise ValueError("time_window end must be greater than start.")
            window = (start_s, end_s)

        if split_by_channel:
            writers = {}
            try:
                for label in chosen + [f"{a}-{b}" for a, b in pairs]:
                    target = out_dir / f"{out_path.stem}_{_safe_filename(label)}.csv"
                    handle_out = open(target, "w", newline="", encoding="utf-8")
                    handle_out.write(f"{time_col_name},{label}\n")
                    writers[label] = handle_out
                    print(f"[out] {target}")
                _write_blocks(
                    data, selected_scale, selected_zero, channel_indices,
                    chosen, pairs, tick_us, fs, segments, continuous_time, window,
                    split_writers=writers,
                )
            finally:
                for handle_out in writers.values():
                    handle_out.close()
            print(f"[mcs] wrote {len(writers)} per-channel CSVs next to: {out_path}")
            return str(out_path)

        columns = chosen + [f"{a}-{b}" for a, b in pairs]
        with open(out_path, "w", newline="", encoding="utf-8") as handle_out:
            handle_out.write(",".join([time_col_name] + columns) + "\n")
            _write_blocks(
                data, selected_scale, selected_zero, channel_indices,
                chosen, pairs, tick_us, fs, segments, continuous_time, window,
                wide_writer=handle_out,
            )

    print(f"[mcs] saved: {out_path}")
    return str(out_path)


def _write_blocks(data, scales, zeros, channel_indices, chosen, pairs,
                  tick_us, fs, segments, continuous_time, window, split_writers=None,
                  wide_writer=None):
    n_samples = int(data.shape[1])
    sorted_indices = np.sort(np.asarray(channel_indices))
    order = np.searchsorted(sorted_indices, channel_indices)
    for start in range(0, n_samples, BLOCK_SAMPLES):
        stop = min(start + BLOCK_SAMPLES, n_samples)
        t = _time_for_block(segments, tick_us, start, stop, continuous_time, fs)

        lo, hi = 0, stop - start
        if window is not None:
            inside = np.flatnonzero((t >= window[0]) & (t < window[1]))
            if inside.size == 0:
                continue
            lo, hi = int(inside[0]), int(inside[-1]) + 1
            t = t[lo:hi]

        block = data[sorted_indices, start + lo:start + hi].astype(np.float64)[order]
        chosen_uv = (block - zeros[:, None]) * scales[:, None]

        columns = {label: values for label, values in zip(chosen, chosen_uv)}
        for a, b in pairs:
            columns[f"{a}-{b}"] = chosen_uv[chosen.index(a)] - chosen_uv[chosen.index(b)]

        if split_writers is not None:
            for label, handle_out in split_writers.items():
                _write_column_block(handle_out, t, [columns[label]])
        else:
            _write_column_block(wide_writer, t, [columns[label] for label in columns])
        print(f"[mcs] {stop:,}/{n_samples:,} samples", flush=True)


WRITE_CHUNK_ROWS = 250_000


def _write_column_block(handle, t: np.ndarray, columns: list[np.ndarray]) -> None:
    """Write time + signal columns with fixed 6-decimal formatting, chunked."""
    fmt = ",".join([TIME_FORMAT] * (len(columns) + 1))
    for start in range(0, t.size, WRITE_CHUNK_ROWS):
        stop = min(start + WRITE_CHUNK_ROWS, t.size)
        rows = zip(t[start:stop], *(column[start:stop] for column in columns))
        handle.write("\n".join(fmt % tuple(row) for row in rows))
        handle.write("\n")


# -------- entry point (run this file) --------
INPUT_PATH = r"C:\path\to\recording.h5"
OUT_CSV = r"C:\path\to\recording_raw.csv"
TARGET_CHANNELS = None  # None = all electrode channels; or ["21", "31"]
SPLIT_BY_CHANNEL = False

if __name__ == "__main__":
    combine_mcs_to_one_csv(
        input_path=INPUT_PATH,
        out_csv=OUT_CSV,
        target_channels=TARGET_CHANNELS,
        split_by_channel=SPLIT_BY_CHANNEL,
    )
