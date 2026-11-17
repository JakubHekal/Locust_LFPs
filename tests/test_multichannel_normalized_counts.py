#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for multichannel normalized spike-count plotting."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "Spike Processing" / "multichannel_normalized_counts.py"


def load_module():
    spec = importlib.util.spec_from_file_location("multichannel_normalized_counts_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plot = load_module()


def count_row(recording, epoch, channel, start_s, duration_s, count):
    return {
        "recording_index": 1 if epoch == "Baseline" else 2,
        "recording_name": recording,
        "epoch_label": epoch,
        "channel": channel,
        "window_start_s": start_s,
        "window_end_s": start_s + duration_s,
        "window_duration_s": duration_s,
        "spike_count": count,
    }


class NormaliseCountsTests(unittest.TestCase):
    def test_baseline_mean_is_100_percent_and_post_scales(self):
        counts = pd.DataFrame(
            [
                count_row("basline", "Baseline", "A-002", 0, 60, 10),
                count_row("basline", "Baseline", "A-002", 60, 60, 10),
                count_row("post0", "Post 0", "A-002", 0, 60, 20),
            ]
        )
        result = plot.normalise_counts(counts)
        baseline = result[result["epoch_label"] == "Baseline"]
        post = result[result["epoch_label"] == "Post 0"]
        self.assertTrue(np.allclose(baseline["normalized_rate_pct"], 100.0))
        self.assertTrue(np.allclose(post["normalized_rate_pct"], 200.0))

    def test_partial_window_uses_rate_not_raw_count(self):
        counts = pd.DataFrame(
            [
                count_row("basline", "Baseline", "A-002", 0, 60, 10),
                count_row("post1", "Post 1", "A-002", 0, 30, 10),
            ]
        )
        result = plot.normalise_counts(counts)
        post = result[result["epoch_label"] == "Post 1"]
        self.assertAlmostEqual(float(post["normalized_rate_pct"].iloc[0]), 200.0, places=6)

    def test_zero_baseline_gives_nan(self):
        counts = pd.DataFrame(
            [
                count_row("basline", "Baseline", "A-002", 0, 60, 0),
                count_row("post0", "Post 0", "A-002", 0, 60, 12),
            ]
        )
        result = plot.normalise_counts(counts)
        self.assertTrue(result["normalized_rate_pct"].isna().all())

    def test_missing_baseline_falls_back_to_first_recording(self):
        counts = pd.DataFrame(
            [
                count_row("pre", "", "A-002", 0, 60, 10),
                count_row("post0", "", "A-002", 0, 60, 20),
            ]
        )
        result = plot.normalise_counts(counts)
        self.assertTrue(np.allclose(result[result["recording_name"] == "pre"]["normalized_rate_pct"], 100.0))
        self.assertTrue(np.allclose(result[result["recording_name"] == "post0"]["normalized_rate_pct"], 200.0))

    def test_baseline_inferred_from_phase(self):
        counts = pd.DataFrame(
            [
                {**count_row("pre", "", "A-002", 0, 60, 10), "phase": "baseline"},
                {**count_row("post0", "", "A-002", 0, 60, 20), "phase": "post"},
            ]
        )
        result = plot.normalise_counts(counts)
        self.assertTrue(np.allclose(result[result["recording_name"] == "pre"]["normalized_rate_pct"], 100.0))
        self.assertTrue(np.allclose(result[result["recording_name"] == "post0"]["normalized_rate_pct"], 200.0))

    def test_baseline_recording_override(self):
        counts = pd.DataFrame(
            [
                count_row("pre", "", "A-002", 0, 60, 10),
                count_row("post0", "", "A-002", 0, 60, 20),
            ]
        )
        result = plot.normalise_counts(counts, baseline_recording="pre")
        self.assertTrue(np.allclose(result["normalized_rate_pct"].iloc[[0, 1]], [100.0, 200.0]))


class LabelTests(unittest.TestCase):
    def test_labels_from_filenames(self):
        self.assertEqual(plot.label_for_path("basline.csv"), "Baseline")
        self.assertEqual(plot.label_for_path("post0.csv"), "Post 0")
        self.assertEqual(plot.label_for_path("post1.csv"), "Post 1")


class ShapeSettingsTests(unittest.TestCase):
    def test_counts_only_cli_does_not_require_raw_recordings(self):
        args = plot.build_arg_parser().parse_args(["--counts-csv", "counts.csv"])
        self.assertIsNone(args.baseline)
        self.assertIsNone(args.post)
    def test_default_shape_mode_is_reject(self):
        args = plot.build_arg_parser().parse_args(["--baseline", "b.csv", "--post", "p.csv"])
        settings = plot.shape_settings(args)
        self.assertEqual(settings["mode"], "reject")
        self.assertEqual(settings["separation_ms"], 1.0)
        self.assertEqual(settings["noise_multiplier"], 3.0)

    def test_counter_module_exposes_process_csvs(self):
        counter = plot.load_counter()
        self.assertTrue(callable(counter.process_csvs))
        self.assertTrue(callable(counter.detect_spikes_with_details))


class PlotTests(unittest.TestCase):
    def frame(self):
        rows = []
        for channel, factor in (("A-000", 0.5), ("A-002", 2.0), ("A-003", 1.0)):
            for index, minutes in enumerate((0.5, 1.5)):
                rows.append(
                    {
                        "recording_index": 1,
                        "recording_name": "post0",
                        "epoch_label": "Post 0",
                        "channel": channel,
                        "window_start_s": minutes * 60 - 30,
                        "window_end_s": minutes * 60 + 30,
                        "window_duration_s": 60,
                        "spike_count": int(10 * factor),
                        "normalized_rate_pct": 100.0 * factor,
                    }
                )
        return pd.DataFrame(rows)

    def test_needle_gets_own_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "recording.png"
            fig = plot.plot_recording(
                self.frame(),
                needle="A-000",
                out_path=out,
                colors=plot.channel_colors(["A-000", "A-002", "A-003"]),
                ylim=(0, 200),
                title="test",
            )
            self.assertEqual(len(fig.axes), 2)
            labels = [text.get_text() for text in fig.legends[0].get_texts()]
            self.assertEqual(labels, ["Channel: A-000", "Channel: A-002", "Channel: A-003"])
            self.assertEqual(fig.axes[1].get_xlabel(), "Time (min)")
            self.assertTrue(out.exists())

    def test_probe_only_recording_uses_one_panel(self):
        frame = self.frame()
        frame = frame[frame["channel"] != "A-000"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "recording.png"
            fig = plot.plot_recording(
                frame,
                needle="A-000",
                out_path=out,
                colors=plot.channel_colors(["A-002", "A-003"]),
                ylim=(0, 200),
                title="test",
            )
            self.assertEqual(len(fig.axes), 1)
            self.assertEqual(fig.axes[0].get_xlabel(), "Time (min)")
            self.assertTrue(out.exists())

    def test_timeline_concatenates_recordings(self):
        frame = self.frame()
        baseline = frame.copy()
        baseline["recording_name"] = "basline"
        baseline["epoch_label"] = "Baseline"
        baseline["recording_index"] = 1
        frame["recording_index"] = 2
        combined = pd.concat([baseline, frame], ignore_index=True)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "timeline.png"
            fig = plot.plot_timeline(
                combined,
                needle="A-000",
                out_path=out,
                colors=plot.channel_colors(["A-000", "A-002", "A-003"]),
                ylim=(0, 200),
            )
            self.assertEqual(len(fig.axes), 2)
            self.assertAlmostEqual(fig.axes[0].get_xlim()[1], 4.0, places=6)
            self.assertTrue(out.exists())

    def timeline_frame(self):
        frame = self.frame()
        baseline = frame.copy()
        baseline["recording_name"] = "basline"
        baseline["epoch_label"] = "Baseline"
        baseline["recording_index"] = 1
        frame["recording_index"] = 2
        combined = pd.concat([baseline, frame], ignore_index=True)
        combined["spikes_per_min"] = combined["spike_count"].astype(float)
        return combined

    def test_timeline_draws_phase_bands_and_legend(self):
        combined = self.timeline_frame()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "timeline.png"
            fig = plot.plot_timeline(
                combined,
                needle="A-000",
                out_path=out,
                colors=plot.channel_colors(["A-000", "A-002", "A-003"]),
                ylim=(0, 200),
            )
            bands = [patch for patch in fig.axes[0].patches if patch.get_width() > 0]
            self.assertEqual(len(bands), 2)
            self.assertAlmostEqual(bands[0].get_width(), 2.0, places=6)
            self.assertTrue(fig.legends)
            labels = [text.get_text() for text in fig.legends[0].get_texts()]
            self.assertEqual(labels[-2:], ["Baseline", "Post 0"])
            self.assertEqual(list(fig.axes[-1].get_xticklabels())[0].get_text(), "0")
            self.assertEqual([text.get_text() for text in fig.axes[-1].get_xticklabels()].count("0"), 2)

    def test_boundary_event_label_is_annotated_and_in_legend(self):
        combined = self.timeline_frame()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "timeline.png"
            fig = plot.plot_timeline(
                combined,
                needle="A-000",
                out_path=out,
                colors=plot.channel_colors(["A-000", "A-002", "A-003"]),
                ylim=(0, 200),
                event_labels=["20min @ -200nA"],
            )
            texts = [text.get_text() for text in fig.axes[0].texts]
            self.assertIn("20min @ -200nA", texts)
            labels = [text.get_text() for text in fig.legends[0].get_texts()]
            self.assertIn("20min @ -200nA", labels)

    def test_absolute_timeline_plots_spikes_per_min(self):
        combined = self.timeline_frame()
        combined.loc[combined["channel"] == "A-002", "spikes_per_min"] = 42.0
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "absolute.png"
            fig = plot.plot_timeline(
                combined,
                needle="A-000",
                out_path=out,
                colors=plot.channel_colors(["A-000", "A-002", "A-003"]),
                ylim=(0, 100),
                value_col="spikes_per_min",
                ylabel="Spike rate\n(spikes/min)",
                title="absolute",
            )
            lines = {line.get_label(): line for line in fig.axes[0].get_lines()}
            self.assertTrue(np.allclose(lines["Channel: A-002"].get_ydata(), 42.0))
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
