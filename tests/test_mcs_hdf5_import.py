"""MCS HDF5 (Multi Channel Systems) imports must feed the SPIE CSV pipeline."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mcs_to_csv", ROOT / "Raw Conversion" / "MCS_to_CSV.py")
mcs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcs)

CHANNEL_DTYPE = np.dtype([
    ("Label", h5py.string_dtype()), ("Unit", h5py.string_dtype()),
    ("Exponent", "<i4"), ("ADZero", "<i4"), ("Tick", "<i8"), ("ConversionFactor", "<i8"),
])


def channel(label, unit="V", exponent=-6, ad_zero=0, tick=1000, conversion=1):
    return (label, unit, exponent, ad_zero, tick, conversion)


def write_mcs(path, streams, version=3, protocol="RawData", with_analog=True):
    """streams: list of dicts with data/channels/timestamps/subtype/label."""
    with h5py.File(path, "w") as handle:
        handle.attrs["McsHdf5ProtocolType"] = np.bytes_(protocol.encode())
        handle.attrs["McsHdf5ProtocolVersion"] = np.int32(version)
        recording = handle.create_group("Data/Recording_0")
        if with_analog:
            analog = recording.create_group("AnalogStream")
            for index, item in enumerate(streams):
                node = analog.create_group(f"Stream_{index}")
                node.attrs["DataSubType"] = np.bytes_(item.get("subtype", "Electrode"))
                node.attrs["Label"] = np.bytes_(item.get("label", "Stream"))
                node.create_dataset("ChannelData", data=np.asarray(item["data"], dtype=np.int32))
                node.create_dataset("ChannelDataTimeStamps",
                                    data=np.asarray(item["timestamps"], dtype=np.int64))
                node.create_dataset("InfoChannel",
                                    data=np.array(item["channels"], dtype=CHANNEL_DTYPE))


class McsImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_electrode_stream_is_scaled_to_microvolts(self):
        path = self.dir / "rec.h5"
        write_mcs(path, [{
            "data": [[0, 1, 2], [10, 11, 12]],
            "channels": [channel("E1"), channel("E2")],
            "timestamps": [[0, 0, 2]],
        }])
        out = self.dir / "out.csv"
        mcs.combine_mcs_to_one_csv(str(path), str(out))

        frame = pd.read_csv(out)
        self.assertEqual(list(frame.columns), ["time_s", "E1", "E2"])
        np.testing.assert_allclose(frame["time_s"], [0.0, 0.001, 0.002])
        np.testing.assert_allclose(frame["E1"], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(frame["E2"], [10.0, 11.0, 12.0])

    def test_auxiliary_and_digital_streams_are_skipped(self):
        path = self.dir / "rec.h5"
        write_mcs(path, [
            {"data": [[5, 6]], "channels": [channel("E1")], "timestamps": [[0, 0, 1]]},
            {"data": [[1, 2]], "channels": [channel("Aux1")],
             "timestamps": [[0, 0, 1]], "subtype": "Auxiliary"},
            {"data": [[1, 0]], "channels": [channel("Dig", unit="NoUnit")],
             "timestamps": [[0, 0, 1]], "subtype": "Digital"},
        ])
        out = self.dir / "out.csv"
        mcs.combine_mcs_to_one_csv(str(path), str(out))
        self.assertEqual(list(pd.read_csv(out).columns), ["time_s", "E1"])

    def test_gap_between_segments_is_preserved(self):
        path = self.dir / "rec.h5"
        write_mcs(path, [{
            "data": [[0, 1, 2, 3, 4, 5]],
            "channels": [channel("E1")],
            "timestamps": [[0, 0, 2], [500000, 3, 5]],
        }])
        out = self.dir / "out.csv"
        mcs.combine_mcs_to_one_csv(str(path), str(out))
        np.testing.assert_allclose(
            pd.read_csv(out)["time_s"], [0.0, 0.001, 0.002, 0.5, 0.501, 0.502]
        )

    def test_continuous_time_ignores_timestamps(self):
        path = self.dir / "rec.h5"
        write_mcs(path, [{
            "data": [[0, 1, 2, 3]],
            "channels": [channel("E1")],
            "timestamps": [[700000, 0, 3]],
        }])
        out = self.dir / "out.csv"
        mcs.combine_mcs_to_one_csv(str(path), str(out), continuous_time=True)
        np.testing.assert_allclose(pd.read_csv(out)["time_s"], [0.0, 0.001, 0.002, 0.003])

    def test_target_channels_and_diff_pair(self):
        path = self.dir / "rec.h5"
        write_mcs(path, [{
            "data": [[1, 2], [3, 4]],
            "channels": [channel("E1"), channel("E2")],
            "timestamps": [[0, 0, 1]],
        }])
        out = self.dir / "out.csv"
        mcs.combine_mcs_to_one_csv(
            str(path), str(out), target_channels=["E2", "E1"], diff_pairs=[("E2", "E1")]
        )
        frame = pd.read_csv(out)
        self.assertEqual(list(frame.columns), ["time_s", "E2", "E1", "E2-E1"])
        np.testing.assert_allclose(frame["E2-E1"], [2.0, 2.0])

    def test_split_by_channel_writes_one_file_per_channel(self):
        path = self.dir / "rec.h5"
        write_mcs(path, [{
            "data": [[1, 2, 3], [4, 5, 6]],
            "channels": [channel("E1"), channel("E2")],
            "timestamps": [[0, 0, 2]],
        }])
        out = self.dir / "split.csv"
        mcs.combine_mcs_to_one_csv(str(path), str(out), split_by_channel=True)
        files = sorted(p.name for p in self.dir.glob("*.csv"))
        self.assertEqual(files, ["split_E1.csv", "split_E2.csv"])
        np.testing.assert_allclose(pd.read_csv(self.dir / "split_E2.csv")["E2"], [4.0, 5.0, 6.0])

    def test_time_window_selects_samples(self):
        path = self.dir / "rec.h5"
        write_mcs(path, [{
            "data": [[0, 1, 2, 3]],
            "channels": [channel("E1")],
            "timestamps": [[0, 0, 3]],
        }])
        out = self.dir / "out.csv"
        mcs.combine_mcs_to_one_csv(str(path), str(out), time_window=(0.001, 0.003))
        np.testing.assert_allclose(pd.read_csv(out)["E1"], [1.0, 2.0])

    def test_unknown_channel_raises(self):
        path = self.dir / "rec.h5"
        write_mcs(path, [{
            "data": [[0, 1]],
            "channels": [channel("E1")],
            "timestamps": [[0, 0, 1]],
        }])
        with self.assertRaisesRegex(ValueError, "not found"):
            mcs.combine_mcs_to_one_csv(str(path), str(self.dir / "out.csv"), target_channels=["E9"])

    def test_file_without_electrode_streams_raises(self):
        path = self.dir / "rec.h5"
        write_mcs(path, [{
            "data": [[0, 1]],
            "channels": [channel("Dig", unit="NoUnit")],
            "timestamps": [[0, 0, 1]],
            "subtype": "Digital",
        }])
        with self.assertRaisesRegex(ValueError, "No electrode"):
            mcs.combine_mcs_to_one_csv(str(path), str(self.dir / "out.csv"))

    def test_file_without_analog_section_raises(self):
        path = self.dir / "rec.h5"
        write_mcs(path, [], with_analog=False)
        with self.assertRaisesRegex(ValueError, "no analog streams"):
            mcs.combine_mcs_to_one_csv(str(path), str(self.dir / "out.csv"))

    def test_non_rawdata_protocol_raises(self):
        path = self.dir / "rec.h5"
        write_mcs(path, [{
            "data": [[0, 1]],
            "channels": [channel("E1")],
            "timestamps": [[0, 0, 1]],
        }], protocol="CMOS_MEA")
        with self.assertRaisesRegex(ValueError, "RawData"):
            mcs.combine_mcs_to_one_csv(str(path), str(self.dir / "out.csv"))


class AdaptiveSizingTests(unittest.TestCase):
    def test_grows_then_halves_after_rate_drop(self):
        tuner = mcs.AdaptiveChunker("write_chunk", initial=100, minimum=10, maximum=1000)
        self.assertIn("growth", tuner.observe(1.0, 100))
        self.assertEqual(tuner.next_size(), 200)
        self.assertIn("growth", tuner.observe(0.5, 200))
        self.assertEqual(tuner.next_size(), 400)

        event = tuner.observe(4.0, 400)
        self.assertIn("rate drop", event)
        self.assertEqual(tuner.next_size(), 200)

    def test_tune_log_names_change_rate_and_unit(self):
        tuner = mcs.AdaptiveChunker("write_chunk", initial=100, minimum=10, maximum=1000)
        growth = tuner.observe(1.0, 100)
        self.assertIn("write_chunk 100 -> 200 rows", growth)
        self.assertIn("growth", growth)
        self.assertIn("100 rows/s", growth)

        tuner.observe(0.5, 200)
        drop = tuner.observe(4.0, 400)
        self.assertIn("write_chunk 400 -> 200 rows", drop)
        self.assertIn("rate drop", drop)
        self.assertIn("% below peak", drop)

    def test_respects_maximum(self):
        tuner = mcs.AdaptiveChunker("read_block", initial=100, minimum=10, maximum=150)
        tuner.observe(1.0, 100)
        tuner.observe(0.1, 150)
        self.assertEqual(tuner.next_size(), 150)

    def test_progress_line_has_percent_rate_and_eta(self):
        line = mcs._progress_line(20_600_000, 45_000_000, 192.0, 107_000)
        self.assertIn("(45.8%)", line)
        self.assertIn("elapsed 3:12", line)
        self.assertIn("ETA 3:48", line)
        self.assertIn("107k samples/s", line)

    def test_progress_line_without_rate_shows_unknown_eta(self):
        line = mcs._progress_line(0, 100, 0.0, 0.0)
        self.assertIn("ETA --:--", line)


if __name__ == "__main__":
    unittest.main()
