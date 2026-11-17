"""Fused MCS HDF5 spike extraction must match the CSV handoff."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mcs = load(ROOT / "Raw Conversion" / "MCS_to_CSV.py", "mcs_for_fused_test")
spike = load(ROOT / "Spike Processing" / "Spike Count Multiple CSVs ordered.py", "spike_for_fused_test")
fused = load(ROOT / "Spike Processing" / "MCS_to_spikes.py", "fused_for_test")

CHANNEL_DTYPE = np.dtype([
    ("Label", h5py.string_dtype()),
    ("Unit", h5py.string_dtype()),
    ("Exponent", "<i4"),
    ("ADZero", "<i4"),
    ("Tick", "<i8"),
    ("ConversionFactor", "<i8"),
])


def channel(label, tick):
    return (label, "V", -6, 0, tick, 1)


def write_h5(path, signals, tick, fs, chunks=None, compression=None):
    n = signals.shape[1]
    with h5py.File(path, "w") as handle:
        handle.attrs["McsHdf5ProtocolType"] = np.bytes_(b"RawData")
        handle.attrs["McsHdf5ProtocolVersion"] = np.int32(3)
        stream = handle.create_group("Data/Recording_0/AnalogStream/Stream_0")
        stream.attrs["DataSubType"] = np.bytes_(b"Electrode")
        stream.attrs["Label"] = np.bytes_(b"Electrode")
        stream.create_dataset(
            "ChannelData",
            data=np.asarray(signals, dtype=np.int32),
            chunks=chunks,
            compression=compression,
            shuffle=bool(compression),
        )
        stream.create_dataset("ChannelDataTimeStamps", data=np.array([[0, 0, n - 1]], dtype=np.int64))
        stream.create_dataset(
            "InfoChannel",
            data=np.array([channel(f"E{index + 1}", tick) for index in range(signals.shape[0])], dtype=CHANNEL_DTYPE),
        )


def synthetic_signals(fs=10_000.0, duration_s=3.0, channels=2, seed=0):
    rng = np.random.default_rng(seed)
    n = int(fs * duration_s)
    signals = rng.normal(0, 2.0, (channels, n))
    for index in range(channels):
        for spike_time in np.arange(0.4, duration_s, 0.45):
            at = int(spike_time * fs)
            signals[index, at - 1:at + 2] += [-100.0, -300.0, -100.0]
    return signals


def same_frame(left, right, left_dir, right_dir):
    for frame, directory in ((left, left_dir), (right, right_dir)):
        for column in frame.columns:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                frame[column] = (
                    frame[column].astype(str)
                    .str.replace(str(directory), "", regex=False)
                    .str.replace(".csv", "", regex=False)
                    .str.replace(".h5", "", regex=False)
                )
    pd.testing.assert_frame_equal(left, right, check_exact=True)


class FusedSpikeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_fused_output_matches_csv_handoff_with_labels_and_order(self):
        h5_paths = []
        labels = {}
        for index, label in enumerate(("Baseline", "Post 1")):
            h5_path = self.dir / f"rec_{index}.h5"
            write_h5(h5_path, synthetic_signals(seed=index), tick=100, fs=10_000.0)
            h5_paths.append(str(h5_path))
            labels[str(h5_path)] = label

        csv_paths = []
        for index, h5_path in enumerate(h5_paths):
            csv_path = self.dir / f"rec_{index}.csv"
            mcs.combine_mcs_to_one_csv(h5_path, str(csv_path))
            csv_paths.append(str(csv_path))
        csv_labels = {csv_path: labels[h5_path] for csv_path, h5_path in zip(csv_paths, h5_paths)}

        csv_out = self.dir / "csv_out"
        csv_result = spike.process_csvs(
            csv_paths,
            out_dir=str(csv_out),
            epoch_labels_by_path=csv_labels,
            polarity="both",
            return_outputs=True,
            shape_settings={"mode": "off"},
        )

        fused_out = self.dir / "fused_out"
        fused_result = fused.process_hdf5s(
            h5_paths,
            out_dir=str(fused_out),
            epoch_labels_by_path=labels,
            polarity="both",
            return_outputs=True,
            shape_settings={"mode": "off"},
            max_workers=2,
        )

        counts_left = pd.read_csv(csv_result["spike_count_csv"])
        counts_right = pd.read_csv(fused_result["spike_count_csv"])
        self.assertEqual(list(counts_right["epoch_label"].unique()), ["Baseline", "Post 1"])
        self.assertEqual(list(counts_right["recording_index"].unique()), [1, 2])
        same_frame(counts_left, counts_right, csv_out, fused_out)

        events_left = pd.read_csv(csv_result["spike_events_csv"])
        events_right = pd.read_csv(fused_result["spike_events_csv"])
        self.assertGreater(len(events_right), 0)
        same_frame(events_left, events_right, csv_out, fused_out)
        self.assertEqual(
            fused_result["spike_processing_provenance"]["files"][0]["strategy"],
            "direct",
        )

    def test_staged_output_matches_csv_handoff_and_cleans_up(self):
        h5_path = self.dir / "staged.h5"
        write_h5(
            h5_path,
            synthetic_signals(channels=4, seed=3),
            tick=100,
            fs=10_000.0,
            chunks=(4, 500),
            compression="gzip",
        )
        csv_path = self.dir / "staged.csv"
        mcs.combine_mcs_to_one_csv(str(h5_path), str(csv_path))

        csv_out = self.dir / "staged_csv_out"
        csv_result = spike.process_csvs(
            [str(csv_path)],
            out_dir=str(csv_out),
            polarity="both",
            return_outputs=True,
            shape_settings={"mode": "off"},
        )

        fused_out = self.dir / "staged_fused_out"
        fused_result = fused.process_hdf5s(
            [str(h5_path)],
            out_dir=str(fused_out),
            polarity="both",
            return_outputs=True,
            shape_settings={"mode": "off"},
            max_workers=2,
        )

        counts_left = pd.read_csv(csv_result["spike_count_csv"])
        counts_right = pd.read_csv(fused_result["spike_count_csv"])
        same_frame(counts_left, counts_right, csv_out, fused_out)

        events_left = pd.read_csv(csv_result["spike_events_csv"])
        events_right = pd.read_csv(fused_result["spike_events_csv"])
        same_frame(events_left, events_right, csv_out, fused_out)

        files = fused_result["spike_processing_provenance"]["files"]
        self.assertEqual(files[0]["strategy"], "staged")
        self.assertEqual(files[0]["amplification"], 4)
        self.assertFalse((fused_out / "stage").exists())

    def test_channel_selection_limits_output_and_cleans_stage(self):
        h5_path = self.dir / "selected.h5"
        write_h5(
            h5_path,
            synthetic_signals(channels=3, seed=4),
            tick=100,
            fs=10_000.0,
            chunks=(3, 500),
            compression="gzip",
        )
        out_dir = self.dir / "selected_out"
        result = fused.process_hdf5s(
            [str(h5_path)],
            out_dir=str(out_dir),
            channels=["E2"],
            polarity="both",
            return_outputs=True,
            shape_settings={"mode": "off"},
            max_workers=1,
        )

        counts = pd.read_csv(result["spike_count_csv"])
        events = pd.read_csv(result["spike_events_csv"])
        self.assertEqual(set(counts["channel"]), {"E2"})
        self.assertEqual(set(events["channel"]), {"E2"})
        self.assertFalse((out_dir / "stage").exists())

    def test_missing_channel_raises(self):
        h5_path = self.dir / "missing.h5"
        write_h5(h5_path, synthetic_signals(channels=2), tick=100, fs=10_000.0)
        with self.assertRaisesRegex(ValueError, "none of the requested channels"):
            fused.process_hdf5s(
                [str(h5_path)],
                out_dir=str(self.dir / "missing_out"),
                channels=["E9"],
                max_workers=1,
            )

    def test_low_sample_rate_fails_fast(self):
        path = self.dir / "slow.h5"
        write_h5(path, synthetic_signals(fs=500.0, duration_s=1.0), tick=2000, fs=500.0)
        with self.assertRaisesRegex(
            ValueError, r"sampling rate 500 Hz.*Needs fs > 600 Hz"
        ):
            fused.process_hdf5s([str(path)], out_dir=str(self.dir / "out"), max_workers=1)

    def test_worker_plan_cpu_and_ram_guards(self):
        workers, detail = fused._plan_workers(100, 45_000_000, 19 * 1024**3, cpu_count=12)
        self.assertEqual(detail["cpu_limit"], 10)
        self.assertLess(workers, 10)
        self.assertTrue(detail["ram_limited"])

        workers, detail = fused._plan_workers(3, 1000, None, cpu_count=12)
        self.assertEqual(workers, 3)
        self.assertFalse(detail["ram_limited"])

        workers, detail = fused._plan_workers(100, 1000, 1, cpu_count=12)
        self.assertEqual(workers, 1)
        self.assertTrue(detail["ram_limited"])


if __name__ == "__main__":
    unittest.main()
