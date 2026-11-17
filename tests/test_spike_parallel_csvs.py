"""Worker threads must not change spike-count output, only its speed."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("spike_counter", ROOT / "Spike Processing" / "Spike Count Multiple CSVs ordered.py")
counter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(counter)


def write_raw_csv(path, fs=10_000.0, duration_s=6.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    frame = {"time_s": t}
    for channel in ("E1", "E2"):
        x = rng.normal(0, 2.0, n)
        for spike_time in np.arange(0.5, duration_s, 0.5):
            index = int(spike_time * fs)
            x[index - 1:index + 2] += [-100.0, -300.0, -100.0]
        frame[channel] = x
    pd.DataFrame(frame).to_csv(path, index=False)


def run(csv_paths, out_dir, cpu_count):
    with patch.object(counter.os, "cpu_count", return_value=cpu_count):
        return counter.process_csvs(
            csv_paths, out_dir=str(out_dir), polarity="neg", return_outputs=True
        )


class ParallelSpikeTests(unittest.TestCase):
    def test_low_sample_rate_fails_with_clear_message(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            path = temp / "slow_raw.csv"
            write_raw_csv(path, fs=500.0)
            with self.assertRaisesRegex(
                ValueError, r"sampling rate 500 Hz.*Needs fs > 600 Hz"
            ):
                run([str(path)], temp / "out", cpu_count=1)

    def test_threaded_output_matches_serial(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            files = []
            for index in range(3):
                path = temp / f"rec_{index}.csv"
                write_raw_csv(path, seed=index)
                files.append(str(path))

            serial = run(files, temp / "serial", cpu_count=1)
            pooled = run(files, temp / "pooled", cpu_count=5)

            serial_dir = temp / "serial"
            pooled_dir = temp / "pooled"

            def normalize(frame, out_dir):
                frame = frame.copy()
                for column in frame.columns:
                    if not pd.api.types.is_numeric_dtype(frame[column]):
                        frame[column] = (
                            frame[column].astype(str)
                            .str.replace(str(out_dir), "", regex=False)
                            .str.replace(str(temp), "", regex=False)
                        )
                return frame

            self.assertGreater(len(pd.read_csv(serial["spike_events_csv"])), 0)
            for key in ("spike_count_csv", "spike_events_csv", "spike_waveform_summary_csv"):
                pd.testing.assert_frame_equal(
                    normalize(pd.read_csv(serial[key]), serial_dir),
                    normalize(pd.read_csv(pooled[key]), pooled_dir),
                    check_exact=True,
                )


if __name__ == "__main__":
    unittest.main()
