import json
import os

import numpy as np

from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.callbacks.performance_profiler import PerformanceProfiler


class PerformanceProfilerTest(testing.TestCase):
    def test_report_and_save(self):
        model = models.Sequential([layers.Dense(1)])
        model.compile(optimizer="sgd", loss="mse")
        path = os.path.join(self.get_temp_dir(), "report.json")
        profiler = PerformanceProfiler(batch_size=4, report_path=path, verbose=0)
        x = np.random.random((16, 3))
        y = np.random.random((16, 1))
        model.fit(x, y, batch_size=4, epochs=2, callbacks=[profiler], verbose=0)
        report = profiler.get_report()
        self.assertEqual(report["num_epochs"], 2)
        self.assertEqual(report["num_batches"], 8)
        self.assertEqual(len(report["loss_history"]), 2)
        self.assertGreater(report["avg_throughput_samples_per_sec"], 0)
        self.assertIn("Performance Report", profiler.summary())
        with open(path) as f:
            self.assertEqual(json.load(f)["num_epochs"], 2)
