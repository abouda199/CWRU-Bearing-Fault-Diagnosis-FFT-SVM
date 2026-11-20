"""Streamlit interaction checks with real artifacts, without a browser."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


class ApplicationTests(unittest.TestCase):
    def test_dashboard_performance_about(self):
        app = AppTest.from_file(APP, default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 4)
        app.sidebar.radio[0].set_value("Model Performance").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[1].value, "920")
        app.sidebar.radio[0].set_value("About Project").run()
        self.assertEqual(len(app.exception), 0)

    def test_analysis_and_real_prediction(self):
        app = AppTest.from_file(APP, default_timeout=30).run()
        app.sidebar.radio[0].set_value("Signal Analysis").run()
        app.radio(key="source").set_value("Exemple NPZ local").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 6)
        app.sidebar.radio[0].set_value("AI Prediction").run()
        app.radio(key="source").set_value("Exemple NPZ local").run()
        self.assertTrue(app.button[0].disabled)
        consent = next(c for c in app.checkbox if "hypothèse" in c.label)
        consent.check().run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 3)
        app.run()
        self.assertEqual(len(app.metric), 3, "Results must survive download reruns")


if __name__ == "__main__":
    unittest.main()
