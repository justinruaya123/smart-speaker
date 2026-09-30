"""Exclusive, persistent compute accounting across CV and all candidate runs."""
import json
from pathlib import Path
import time
from .contract import write_json


class ComputeBudget:
    def __init__(self, path, run, epoch_limit=200, seconds_limit=24 * 3600):
        self.path, self.run = Path(path), str(run)
        self.limits = {'epochs': epoch_limit, 'seconds': seconds_limit}
        if not 0 < epoch_limit <= 200 or not 0 < seconds_limit <= 86400:
            raise ValueError('Compute limits exceed fixed protocol')
        self.lock = self.path.with_suffix('.lock')

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A stale lock is evidence of an interrupted run, not permission to erase it.
        self.handle = self.lock.open('x')
        try:
            self.handle.write(self.run); self.handle.flush()
            self.state = (json.loads(self.path.read_text()) if self.path.exists() else
                          {'limits': self.limits, 'seconds': 0., 'epochs': []})
            if self.state['limits'] != self.limits:
                raise ValueError('Cannot change existing compute limits')
            self.started = time.monotonic()
            self.base_seconds = self.state['seconds']
            self.check()
            return self
        except BaseException:
            self.handle.close(); self.lock.unlink()
            raise

    def _save(self):
        self.state['seconds'] = self.base_seconds + time.monotonic() - self.started
        temporary = self.path.with_suffix('.pending')
        write_json(temporary, self.state)
        temporary.replace(self.path)

    def check(self):
        self._save()
        if self.state['seconds'] >= self.limits['seconds']:
            raise RuntimeError('Global compute budget exhausted (time)')

    def start_epoch(self, label):
        self.check()
        if len(self.state['epochs']) >= self.limits['epochs']:
            raise RuntimeError('Global compute budget exhausted (epochs)')
        self.state['epochs'].append({'run': self.run, 'label': label})
        self._save()  # Interrupted epochs are conservatively charged in full.

    def __exit__(self, *exc):
        self._save()
        self.handle.close()
        self.lock.unlink()

