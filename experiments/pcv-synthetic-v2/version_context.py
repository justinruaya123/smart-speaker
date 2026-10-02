"""Explicit single-study context for the shared v1 implementation, restored on exit.

This avoids copying a trainer or mutating v1 files while v1 synthesis is active.
Each CLI process runs one version; this adapter is not a concurrent server API.
"""
from contextlib import contextmanager
import revision as rev
import corpus


@contextmanager
def bind():
    import study,synthbench,fitting,finishing,reporting
    modules=(study,synthbench,fitting,finishing,reporting)
    changes=[]
    def set_(module,key,value):
        changes.append((module,key,getattr(module,key)));setattr(module,key,value)
    for module in modules:
        for key,value in [('HERE',rev.HERE),('CACHE',rev.CACHE),('Contract',rev.RevisionContract)]:
            if hasattr(module,key):set_(module,key,value)
    for module in (study,synthbench):set_(module,'validate_frozen_study',rev.validate)
    for module in (synthbench,fitting):
        set_(module,'compile_synthetic_catalog',rev.compile_catalog)
        set_(module,'load_split',corpus.load_split)
        set_(module,'audit_all_splits',corpus.audit_all_splits)
    synthbench.intent_use_cases.cache_clear()
    try:yield
    finally:
        for module,key,value in reversed(changes):setattr(module,key,value)
        synthbench.intent_use_cases.cache_clear()
