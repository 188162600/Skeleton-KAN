"""Exact completed-pair cache with explicit per-operation timeout records."""
from __future__ import annotations
import multiprocessing as mp
import sys
import time
import traceback
from . import cache


def _calculate(sender, test, spec, left, right, bootstrap=None):
    try:
        from .acuos2 import generalize, matches, restore_pair_modules
        if bootstrap is not None:
            test, spec = restore_pair_modules(bootstrap)
        a, b = spec.parseTerm(left), spec.parseTerm(right)
        assert a is not None and b is not None
        if matches(b, a):
            choices, info = [str(a)], {'shortcut': 'left_subsumes_right', 'complete': True}
        elif matches(a, b):
            choices, info = [str(b)], {'shortcut': 'right_subsumes_left', 'complete': True}
        else:
            choices, info = generalize(test, spec, left, right)
        assert all(matches(a, spec.parseTerm(x)) and matches(b, spec.parseTerm(x)) for x in choices)
        sender.send({'status': 'succeeded', 'choices': choices, 'info': info})
    except BaseException as exc:
        sender.send({'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
    finally:
        sender.close()


def pair_operation(test, spec, left, right, context, timeout_seconds):
    identity = {'context': context, 'left': left, 'right': right, 'entry': 'auto'}
    completed = cache.get('acuos2_completed_pair_v1', identity)
    if completed is not None:
        return {**completed, 'cache_hit': True}
    limited = {'pair': identity, 'budget_seconds': timeout_seconds}
    timeout = cache.get('acuos2_pair_limit_v1', limited)
    if timeout is not None:
        return {**timeout, 'cache_hit': True}
    # Preserve the original Linux path. Other platforms reconstruct the exact
    # native signature in a spawned child instead of pickling Maude modules.
    # No native enumerator is truncated into a purported partial answer.
    method = 'fork' if sys.platform == 'linux' and 'fork' in mp.get_all_start_methods() else 'spawn'
    ctx = mp.get_context(method)
    receiver, sender = ctx.Pipe(duplex=False)
    if method == 'spawn':
        from .acuos2 import pair_bootstrap
        args = (sender, None, None, left, right, pair_bootstrap())
    else:
        args = (sender, test, spec, left, right)
    child = ctx.Process(target=_calculate, args=args)
    started = time.monotonic()
    child.start()
    sender.close()
    try:
        if receiver.poll(timeout_seconds):
            try:
                result = receiver.recv()
            except EOFError:
                result = {'status': 'failed', 'error': 'Native pair subprocess exited without a result'}
        else:
            result = {'status': 'timeout', 'budget_seconds': timeout_seconds,
                      'partial_generalizer_available': False}
    finally:
        if child.is_alive():
            child.terminate()
        child.join(2)
        if child.is_alive():
            child.kill()
            child.join()
        receiver.close()
    result.update(wall_seconds=time.monotonic()-started, cache_hit=False)
    if result['status'] == 'succeeded':
        cache.put('acuos2_completed_pair_v1', identity, result)
    elif result['status'] == 'timeout':
        cache.put('acuos2_pair_limit_v1', limited, result)
    return result
