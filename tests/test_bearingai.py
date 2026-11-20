"""Focused inference regression checks. Run unittest discovery, not root training scripts."""
import unittest

import numpy as np
import pandas as pd

from utils.prediction import (CLASSES, describe_class, evaluate_model, load_dataset,
                              load_model, predict_windows)
from utils.preprocessing import fft_features, numeric_signal, read_csv_signal, segment_signal


class PreprocessingTests(unittest.TestCase):
    def test_fft_matches_original_formula(self):
        samples = np.random.default_rng(42).normal(size=(3, 32, 32))
        x = samples.reshape(len(samples), -1)
        expected = np.log1p(np.abs(np.fft.rfft(
            (x - x.mean(axis=1, keepdims=True)) * np.hanning(x.shape[1]), axis=1)))
        np.testing.assert_array_equal(fft_features(samples), expected)

    def test_segmentation_never_silently_pads_or_drops(self):
        for x in (np.arange(100), np.arange(2050)):
            with self.assertRaises(ValueError):
                segment_signal(x)
        windows, tail = segment_signal(np.arange(2050), discard_tail=True)
        self.assertEqual(windows.shape, (2, 1024))
        self.assertEqual(tail, 2)
        self.assertEqual(windows[-1, -1], 2047)
        for x in (np.zeros((1, 100)), np.full((1, 1024), np.nan)):
            with self.assertRaises(ValueError):
                fft_features(x)

    def test_csv_header_and_missing_values(self):
        frame = read_csv_signal(b"1\n2\n3\n", False, ",")
        np.testing.assert_array_equal(numeric_signal(frame, "0"), [1, 2, 3])
        frame = read_csv_signal(b"time;amplitude\n0;1\n1;2\n", True, ";")
        np.testing.assert_array_equal(numeric_signal(frame, "amplitude"), [1, 2])
        for invalid in ([1, None], [1, "abc"], [1, np.inf]):
            with self.assertRaises(ValueError):
                numeric_signal(pd.DataFrame({"a": invalid}), "a")

    def test_class_metadata(self):
        self.assertEqual(describe_class("IR_014")["Diamètre nominal (in)"], .014)
        self.assertIsNone(describe_class("Normal")["Diamètre nominal (in)"])


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def test_real_prediction(self):
        x, y = load_dataset()
        results = predict_windows(self.model, x[:3].reshape(3, 1024))
        self.assertEqual(len(results), 3)
        self.assertTrue(set(results["Classe"]).issubset(CLASSES))

    def test_documented_holdout(self):
        results = evaluate_model(self.model)
        self.assertEqual(results["n_test"], 920)
        self.assertEqual(int(results["matrix"].sum()), 920)
        self.assertEqual(results["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
