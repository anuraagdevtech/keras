import json
import time

import numpy as np

from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.utils import file_utils


def _peak_memory_mb():
    """Peak resident memory of the process in MB, or `None` if unknown."""
    try:
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is in bytes on macOS and kilobytes elsewhere.
        return peak / (1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0)
    except Exception:
        pass
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return None


@keras_export("keras.callbacks.PerformanceProfiler")
class PerformanceProfiler(Callback):
    """Callback that profiles training time, memory and throughput.

    It tracks total training time, time per epoch and per batch, peak
    memory usage, throughput (samples/sec) and loss / accuracy trends.
    A summary is available through `get_report()` / `summary()` and is
    optionally printed and saved to a JSON file when training ends.

    Args:
        batch_size: Optional integer. Number of samples per batch, used for
            throughput. If `None`, the `size` entry of the batch logs or the
            `batch_size` passed to `fit()` is not available, so throughput
            is only reported when this is set.
        report_path: Optional string. If set, the report is saved as JSON
            to this path at the end of training.
        verbose: Integer. If `1`, prints per-epoch metrics and the final
            report.

    Example:

    ```python
    profiler = keras.callbacks.PerformanceProfiler(batch_size=32)
    model.fit(x, y, batch_size=32, callbacks=[profiler])
    report = profiler.get_report()
    ```
    """

    def __init__(self, batch_size=None, report_path=None, verbose=1):
        super().__init__()
        self.batch_size = batch_size
        self.report_path = report_path
        self.verbose = verbose
        self._reset()

    def _reset(self):
        self.train_start = None
        self.total_time = 0.0
        self.epoch_times = []
        self.batch_times = []
        self.epoch_throughputs = []
        self.loss_history = []
        self.accuracy_history = []
        self.peak_memory_mb = None
        self._epoch_start = None
        self._batch_start = None
        self._epoch_batches = 0

    def _update_memory(self):
        mem = _peak_memory_mb()
        if mem is not None:
            self.peak_memory_mb = max(self.peak_memory_mb or 0.0, mem)

    def on_train_begin(self, logs=None):
        self._reset()
        self.train_start = time.perf_counter()
        self._update_memory()

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.perf_counter()
        self._epoch_batches = 0

    def on_train_batch_begin(self, batch, logs=None):
        self._batch_start = time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
        if self._batch_start is not None:
            self.batch_times.append(time.perf_counter() - self._batch_start)
            self._batch_start = None
        self._epoch_batches += 1
        self._update_memory()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.perf_counter() - self._epoch_start
        self.epoch_times.append(elapsed)
        throughput = None
        if self.batch_size and elapsed > 0:
            throughput = self._epoch_batches * self.batch_size / elapsed
        self.epoch_throughputs.append(throughput)
        if "loss" in logs:
            self.loss_history.append(float(logs["loss"]))
        for key in ("accuracy", "acc"):
            if key in logs:
                self.accuracy_history.append(float(logs[key]))
                break
        self._update_memory()
        if self.verbose:
            msg = f"[PerformanceProfiler] epoch {epoch + 1}: {elapsed:.3f}s"
            if throughput is not None:
                msg += f", {throughput:.1f} samples/sec"
            print(msg)

    def on_train_end(self, logs=None):
        if self.train_start is not None:
            self.total_time = time.perf_counter() - self.train_start
        self._update_memory()
        if self.report_path:
            self.save_report(self.report_path)
        if self.verbose:
            print(self.summary())

    @staticmethod
    def _trend(values):
        if len(values) < 2:
            return None
        return values[-1] - values[0]

    def get_report(self):
        """Returns the performance report as a dict."""
        throughputs = [t for t in self.epoch_throughputs if t is not None]
        return {
            "total_training_time_sec": self.total_time,
            "num_epochs": len(self.epoch_times),
            "avg_epoch_time_sec": (
                float(np.mean(self.epoch_times)) if self.epoch_times else None
            ),
            "epoch_times_sec": list(self.epoch_times),
            "avg_batch_time_sec": (
                float(np.mean(self.batch_times)) if self.batch_times else None
            ),
            "num_batches": len(self.batch_times),
            "peak_memory_mb": self.peak_memory_mb,
            "avg_throughput_samples_per_sec": (
                float(np.mean(throughputs)) if throughputs else None
            ),
            "loss_history": list(self.loss_history),
            "loss_change": self._trend(self.loss_history),
            "accuracy_history": list(self.accuracy_history),
            "accuracy_change": self._trend(self.accuracy_history),
        }

    def summary(self):
        """Returns a human-readable performance report."""

        def fmt(value, spec):
            return "n/a" if value is None else format(value, spec)

        r = self.get_report()
        return "\n".join(
            [
                "=== Performance Report ===",
                f"Total training time: {r['total_training_time_sec']:.3f}s",
                f"Epochs: {r['num_epochs']}",
                f"Avg epoch time: {fmt(r['avg_epoch_time_sec'], '.3f')}s",
                f"Avg batch time: {fmt(r['avg_batch_time_sec'], '.5f')}s",
                f"Peak memory: {fmt(r['peak_memory_mb'], '.1f')} MB",
                "Avg throughput: "
                f"{fmt(r['avg_throughput_samples_per_sec'], '.1f')} samples/sec",
                f"Loss change: {fmt(r['loss_change'], '+.4f')}",
                f"Accuracy change: {fmt(r['accuracy_change'], '+.4f')}",
            ]
        )

    def save_report(self, path):
        """Saves the report as JSON to `path`."""
        with file_utils.File(path, "w") as f:
            json.dump(self.get_report(), f, indent=2)
