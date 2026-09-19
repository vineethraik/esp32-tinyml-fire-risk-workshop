"""Check the small labeling exercise without touching student data."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_labeling_windows import prepare_windows  # noqa: E402
from train_heatgun_neural_net import load_student_windows  # noqa: E402


class StudentWindowsTest(unittest.TestCase):
    def make_readings(self, count):
        return pd.DataFrame({
            "sequence": range(1, count + 1),
            "temp_c": np.arange(count, dtype=float) + 20,
            "humidity_pct": np.arange(count, dtype=float) / 10 + 40,
        })

    def test_100_readings_give_90_windows(self):
        windows, skipped, used = prepare_windows(self.make_readings(120))
        self.assertEqual((len(windows), skipped, used), (90, 0, 100))
        self.assertEqual((windows.iloc[0].start_sequence, windows.iloc[0].end_sequence), (2, 11))
        self.assertEqual((windows.iloc[-1].start_sequence, windows.iloc[-1].end_sequence), (91, 100))
        self.assertEqual(windows.iloc[0].temp_c_0, 21)
        self.assertEqual(windows.iloc[0].humidity_pct_9, 41)
        self.assertTrue((windows["label"] == "").all())

    def test_minimum_and_later_start(self):
        windows, _, used = prepare_windows(self.make_readings(11))
        self.assertEqual((len(windows), used), (1, 11))
        later, _, _ = prepare_windows(self.make_readings(120), start_row=20)
        self.assertEqual((later.iloc[0].start_sequence, later.iloc[-1].end_sequence), (22, 120))
        with self.assertRaisesRegex(ValueError, "at least 11"):
            prepare_windows(self.make_readings(10))

    def test_completed_labels_import_and_blanks_ignored(self):
        windows, _, _ = prepare_windows(self.make_readings(100))
        windows.loc[0, "label"] = "normal"
        windows.loc[1, "label"] = "HIGH_THERMAL_RISK"
        with TemporaryDirectory() as folder:
            path = Path(folder) / "labels.csv"
            windows.to_csv(path, index=False)
            features, labels = load_student_windows(path)
            self.assertEqual(features.shape, (2, 20))
            self.assertEqual(labels.tolist(), [0, 2])
            self.assertEqual(features[0, 0], 21)
            self.assertAlmostEqual(features[0, 1], 40.1, places=4)

            windows.loc[2, "label"] = "FIRE"
            windows.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Unknown student label"):
                load_student_windows(path)


if __name__ == "__main__":
    unittest.main()
