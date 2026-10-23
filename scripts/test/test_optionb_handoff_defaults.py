"""The ordinary CLIs must select the latest additive verification locks."""
from pathlib import Path
import argparse
import runpy

import pytest


@pytest.mark.parametrize('script,expected', [
    ('optionb_release_check.py', 'experiments/speaker-optionb-release-v1/protocol-v3.json'),
    ('optionb_pc_benchmark.py', 'experiments/speaker-optionb-release-v1/pc-benchmark-protocol-v2.json'),
])
def test_cli_defaults_to_current_corrected_lock(monkeypatch, script, expected):
    observed = {}

    class ParsingObserved(Exception):
        pass

    def inspect_defaults(parser, *args, **kwargs):
        observed['protocol'] = parser.get_default('protocol')
        raise ParsingObserved

    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', inspect_defaults)
    with pytest.raises(ParsingObserved):
        runpy.run_path(str(Path(__file__).with_name(script)), run_name='__main__')
    assert observed['protocol'] == Path(expected)
