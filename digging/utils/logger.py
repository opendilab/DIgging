import json
import os
from datetime import datetime
import numpy as np

from .event import DiggingEvent
from ding.torch_utils import to_list


def get_logger(name='screen'):
    assert name in ['screen', 'json'], "Not supported logger name: " + name
    return {
        'screen': ScreenLogger,
        'json': JSONLogger,
    }[name]


class StepTracker():

    def __init__(self) -> None:
        self._iterations = 0

        self._previous_best_score = None
        self._previous_best_sample = None

        self._start_time = None
        self._previous_time = None

    def _update_tracker(self, event, digger):
        if event in {DiggingEvent.STEP, DiggingEvent.SKIP}:
            self._iterations += 1

            current_best = digger.provide_best()
            if current_best and (self._previous_best_score is None
                                 or current_best['score'] > self._previous_best_score):
                self._previous_best_sample = current_best['sample']
                self._previous_best_score = current_best['score']
        elif event == DiggingEvent.END:
            self._iterations = 0
            self._previous_best_score = None
            self._previous_best_sample = None
            self._start_time = None
            self._previous_time = None

    def _time_metrics(self):
        now = datetime.now()
        if self._start_time is None:
            self._start_time = now
        if self._previous_time is None:
            self._previous_time = now

        time_elapsed = now - self._start_time
        time_step = now - self._previous_time

        self._previous_time = now
        return (now.strftime("%Y-%m-%d %H:%M:%S"), time_elapsed.total_seconds(), time_step.total_seconds())


class ScreenLogger(StepTracker):
    _default_cell_size = 9
    _default_precision = 3
    _default_max_length = 80
    _default_data_max_col = 3

    def __init__(self, verbose=2):
        self._verbose = verbose
        self._header_length = None
        super(ScreenLogger, self).__init__()

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, v):
        self._verbose = v

    def _format_number(self, x):
        if isinstance(x, int) or isinstance(x, np.int64):
            s = "{x:< {s}}".format(
                x=x,
                s=self._default_cell_size,
            )
        else:
            s = "{x:< {s}.{p}}".format(
                x=x,
                s=self._default_cell_size,
                p=self._default_precision,
            )

        if len(s) > self._default_cell_size:
            if "." in s:
                return s[:self._default_cell_size]
            else:
                return s[:self._default_cell_size - 3] + "..."
        return s

    def _format_array(self, x, max_len):
        with np.printoptions(precision=self._default_precision):
            s = str(x).replace('\n', ' ')
        if len(s) > max_len:
            return s[:max_len - 3] + "..."
        elif len(s) < max_len:
            s += " " * (max_len - len(s))
        return s

    def _format_key(self, key):
        s = "{key:^{s}}".format(key=key, s=self._default_cell_size)
        if len(s) > self._default_cell_size:
            return s[:self._default_cell_size - 3] + "..."
        return s

    def _row(self, target, samples):
        cells = [
            self._format_number(self._iterations + 1),
            self._format_number(target) if isinstance(target, (int, float)) else self._format_key(target),
        ]
        cur_len = (self._default_cell_size + 4) * 2
        if isinstance(samples, np.ndarray):
            cells.append(self._format_array(samples, self._default_max_length - cur_len - 2))
        else:
            res_max_len = (self._default_max_length - cur_len) // len(samples) - 2
            for s in samples:
                cells.append(self._format_array(s, res_max_len))
        return "| " + " | ".join(cells) + " |"

    def _step(self, digger):
        res = digger.provide_best()
        return self._row(res['score'], digger.space.get_log_data(res['sample'], self._default_data_max_col))

    def _skip(self):
        return self._row('<skipped>', [])

    def _header(self, digger):
        cells = [
            self._format_key("iter"),
            self._format_key("target"),
        ]
        cur_len = (self._default_cell_size + 4) * 2
        keys = digger.space.get_log_title(self._default_data_max_col)
        res_max_len = (self._default_max_length - cur_len) // len(keys) - 2
        for key in digger.space.get_log_title(self._default_data_max_col):
            cells.append(self._format_array(key, res_max_len))

        line = "| " + " | ".join(cells) + " |"
        self._header_length = len(line)
        return ("=" * self._header_length) + '\n' + line + "\n" + ("-" * self._header_length)

    def _is_new_best(self, digger):
        best = digger.provide_best()
        if best:
            if self._previous_best_score is None:
                return True
            return best["score"] > self._previous_best_score
        else:
            return False

    def update(self, event, digger):
        if event == DiggingEvent.START:
            line = self._header(digger) + "\n"
        elif event == DiggingEvent.STEP:
            is_new_best = self._is_new_best(digger)
            if self._verbose == 1 and not is_new_best:
                line = ""
            else:
                line = self._step(digger) + "\n"
        elif event == DiggingEvent.SKIP:
            line = self._skip(digger) + "\n"
        elif event == DiggingEvent.END:
            line = "=" * self._header_length + "\n"
        else:
            raise ValueError(f'Unknown event - {event}.')  # pragma: no cover

        if self._verbose:
            print(line, end="")
        self._update_tracker(event, digger)


class JSONLogger(StepTracker):

    def __init__(self, path, reset=True):
        self._path = path if path[-5:] == ".json" else path + ".json"
        if reset:
            try:
                os.remove(self._path)
            except OSError:
                pass
        super(JSONLogger, self).__init__()

    def update(self, event, digger):
        if event == DiggingEvent.STEP:
            data = digger.latest
            data['sample'] = digger.space.get_log_data(data['sample'], len(data['sample']))
            data['sample'] = to_list(data['sample'])

            now, time_elapsed, time_delta = self._time_metrics()
            data["datetime"] = {
                "datetime": now,
                "elapsed": time_elapsed,
                "delta": time_delta,
            }
            data['iterations'] = self._iterations

            with open(self._path, "a") as f:
                f.write(json.dumps(data) + "\n")

        self._update_tracker(event, digger)
